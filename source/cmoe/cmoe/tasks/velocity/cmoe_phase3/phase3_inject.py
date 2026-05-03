"""Phase 3 inject — VAE estimator + AE estimator + actor/critic 接 v_pred + z_H + z_E.

============================================================================
设计 (Phase 3, paper §III-C Eq. 5 + Fig. 3):
============================================================================

obs.policy (per-term flatten, history_length=5):
  layout = [base_lin_vel(15), base_ang_vel(15), gravity(15), cmd(15),
            jpos(145), jvel(145), last_act(145), elevation(480)]
  total = 60 + 435 + 480 = 975 dim

obs.critic (no history):
  layout = [base_lin_vel(3), base_ang_vel(3), gravity(3), cmd(3),
            jpos(29), jvel(29), last_act(29), elevation(96)]
  total = 99 + 96 = 195 dim

我们 inject 替换 alg.policy.actor = VAEAEAugmentedActor(...):
  forward(x_975):
    o_t_full   = extract_frame(x_975, 4)         # 最新 99 dim (proprioception)
    e_t_latest = extract_elevation_latest(x_975) # 最新 96 dim (elevation)
    
    vae_history = [extract_frame(x, i)[:, 3:] for i in 0..4]  # 5×96 = 480
    
    with no_grad: z_H, v_pred = vae(vae_history)   # paper §III-C
    with no_grad: z_E         = ae.encode(e_t_latest)  # paper Eq.5
    
    augmented = cat([o_t_full(99), v_pred(3), z_H(32), z_E(32)])  # 166 dim
    return actor_mlp(augmented)

inject 替换 alg.policy.critic = AEAugmentedCritic(...):
  forward(x_195):
    o_critic = x_195[:, :99]    # proprioception
    e_critic = x_195[:, 99:195] # elevation (96 dim)
    
    with no_grad: z_E = ae.encode(e_critic)
    augmented = cat([o_critic(99), z_E(32)])  # 131 dim
    return critic_mlp(augmented)

VAE supervision (与 Phase 2b 完全相同):
  vel_gt     = obs.critic[:, :3]
  vae_target = vae_history 最新一帧 (96 dim)

AE supervision (NEW):
  e_gt = obs.critic[:, 99:195] = elevation ground truth (96 dim, no noise)
============================================================================
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# 常量 (Phase 3 obs structure)
# ============================================================
POLICY_FRAME_DIM = 99    # 每个 proprioception 帧 99 dim
BASE_LIN_VEL_DIM = 3     # 单帧 obs 前 3 dim
OBS96_DIM = POLICY_FRAME_DIM - BASE_LIN_VEL_DIM   # 96 dim (paper Eq 2)
HISTORY_LEN = 5
ELEVATION_DIM = 96       # 12 × 8 grid (paper §IV-A)
VAE_INPUT_DIM = OBS96_DIM * HISTORY_LEN           # 480
ACTOR_MLP_INPUT_DIM = POLICY_FRAME_DIM + 3 + 32 + 32   # 99 + v_pred(3) + z_H(32) + z_E(32) = 166
CRITIC_MLP_INPUT_DIM = POLICY_FRAME_DIM + 32           # 99 + z_E(32) = 131
EXPECTED_POLICY_OBS_DIM = (POLICY_FRAME_DIM + ELEVATION_DIM) * HISTORY_LEN  # 195 × 5 = 975
EXPECTED_CRITIC_OBS_DIM = POLICY_FRAME_DIM + ELEVATION_DIM                  # 99 + 96 = 195

# IsaacLab obs layout (per-term flatten, with history_length=5):
# Each obs term in policy is (term_dim,) → flattened to (term_dim × 5,)
# Term order: base_lin_vel, base_ang_vel, gravity, cmd, jpos, jvel, last_action, elevation
OBS_TERM_DIMS = (3, 3, 3, 3, 29, 29, 29, 96)
def _compute_term_offsets():
    """Compute starting index of each term in flattened (975,) obs."""
    offsets = [0]
    for term_dim in OBS_TERM_DIMS[:-1]:
        offsets.append(offsets[-1] + term_dim * HISTORY_LEN)
    return offsets

OBS_TERM_OFFSETS = _compute_term_offsets()
# = [0, 15, 30, 45, 60, 205, 350, 495]
# 7 个 proprio terms 占 [0:495], elevation 占 [495:975]
assert sum(d * HISTORY_LEN for d in OBS_TERM_DIMS) == EXPECTED_POLICY_OBS_DIM, \
    f"obs layout mismatch: {sum(d * HISTORY_LEN for d in OBS_TERM_DIMS)} != {EXPECTED_POLICY_OBS_DIM}"


def _extract_frame_from_obs(x: torch.Tensor, frame_idx: int) -> torch.Tensor:
    """Extract a single proprioception frame (99 dim) — exclude elevation.
    
    For VAE (uses only first 7 terms = 99 dim per frame).
    """
    parts = []
    for term_id in range(7):  # exclude elevation (term_id=7)
        term_dim = OBS_TERM_DIMS[term_id]
        start = OBS_TERM_OFFSETS[term_id] + frame_idx * term_dim
        parts.append(x[:, start : start + term_dim])
    return torch.cat(parts, dim=-1)


def _extract_elevation_latest(x: torch.Tensor) -> torch.Tensor:
    """Extract LATEST elevation frame (96 dim) from per-term-flattened obs."""
    elev_offset = OBS_TERM_OFFSETS[7]  # 495
    elev_dim = OBS_TERM_DIMS[7]        # 96
    # Latest frame = index HISTORY_LEN-1
    start = elev_offset + (HISTORY_LEN - 1) * elev_dim
    return x[:, start : start + elev_dim]


def _extract_vae_history_target(x: torch.Tensor):
    """Extract VAE inputs from per-term-flattened obs.
    
    vae_history: 5 帧 × 96 dim (drop base_lin_vel, drop elevation) = 480
    vae_target:  最新一帧的 96 dim
    """
    history_frames = []
    for i in range(HISTORY_LEN):
        f = _extract_frame_from_obs(x, i)              # (N, 99) 不含 elevation
        history_frames.append(f[:, BASE_LIN_VEL_DIM:]) # (N, 96) drop base_lin_vel
    vae_history = torch.cat(history_frames, dim=-1)    # (N, 480)
    vae_target  = history_frames[-1]                   # (N, 96)
    return vae_history, vae_target


# ============================================================
# VAEAEAugmentedActor: actor(o_t + v_pred + z_H + z_E)
# ============================================================
class VAEAEAugmentedActor(nn.Module):
    """Actor that augments proprioception with VAE (v_pred, z_H) AND AE (z_E).
    
    Replaces rsl_rl ActorCritic.actor.
    """
    def __init__(
        self,
        vae_estimator,
        ae_estimator,
        actor_mlp: nn.Sequential,
        policy_frame_dim: int = POLICY_FRAME_DIM,
        history_len: int = HISTORY_LEN,
        base_lin_vel_dim: int = BASE_LIN_VEL_DIM,
        elevation_dim: int = ELEVATION_DIM,
    ):
        super().__init__()
        self.vae_estimator = vae_estimator
        self.ae_estimator = ae_estimator
        self.actor_mlp = actor_mlp
        self.policy_frame_dim = policy_frame_dim
        self.history_len = history_len
        self.base_lin_vel_dim = base_lin_vel_dim
        self.elevation_dim = elevation_dim

    def __getitem__(self, idx):
        return self.actor_mlp[idx]

    def __len__(self):
        return len(self.actor_mlp)

    def _split_x(self, x: torch.Tensor):
        """Extract (o_t, vae_history, vae_target, e_t_latest) from x_975."""
        D = x.shape[-1]
        if D != EXPECTED_POLICY_OBS_DIM:
            raise RuntimeError(
                f"[VAEAEAugmentedActor] Expected x.shape[-1] = {EXPECTED_POLICY_OBS_DIM}, got {D}"
            )
        o_t_full = _extract_frame_from_obs(x, HISTORY_LEN - 1)
        vae_history, vae_target = _extract_vae_history_target(x)
        e_t_latest = _extract_elevation_latest(x)
        return o_t_full, vae_history, vae_target, e_t_latest

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o_t_full, vae_history, vae_target, e_t_latest = self._split_x(x)

        # VAE encode (no_grad: VAE 单独训练)
        with torch.no_grad():
            z_H, v_pred = self.vae_estimator.get_latent_and_velocity(vae_history)
            z_H = z_H.detach()
            v_pred = v_pred.detach()
            # AE encode (no_grad: AE 单独训练)
            z_E = self.ae_estimator.get_latent(e_t_latest).detach()

        augmented = torch.cat([o_t_full, v_pred, z_H, z_E], dim=-1)  # 166 dim
        return self.actor_mlp(augmented)


# ============================================================
# AEAugmentedCritic: critic(o_critic + z_E)
# ============================================================
class AEAugmentedCritic(nn.Module):
    """Critic that augments proprioception with AE (z_E).
    
    Replaces rsl_rl ActorCritic.critic.
    Note: critic does NOT use VAE (paper Fig.3 critic only uses elevation latent).
    """
    def __init__(
        self,
        ae_estimator,
        critic_mlp: nn.Sequential,
        critic_proprio_dim: int = POLICY_FRAME_DIM,
        elevation_dim: int = ELEVATION_DIM,
    ):
        super().__init__()
        self.ae_estimator = ae_estimator
        self.critic_mlp = critic_mlp
        self.critic_proprio_dim = critic_proprio_dim
        self.elevation_dim = elevation_dim

    def __getitem__(self, idx):
        return self.critic_mlp[idx]

    def __len__(self):
        return len(self.critic_mlp)

    def _split_x(self, x: torch.Tensor):
        """Extract (o_critic, e_critic) from x_195 (no history, simple split)."""
        D = x.shape[-1]
        if D != EXPECTED_CRITIC_OBS_DIM:
            raise RuntimeError(
                f"[AEAugmentedCritic] Expected x.shape[-1] = {EXPECTED_CRITIC_OBS_DIM}, got {D}"
            )
        o_critic = x[:, :self.critic_proprio_dim]                # (N, 99)
        e_critic = x[:, self.critic_proprio_dim : self.critic_proprio_dim + self.elevation_dim]  # (N, 96)
        return o_critic, e_critic

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o_critic, e_critic = self._split_x(x)

        # AE encode (no_grad)
        with torch.no_grad():
            z_E = self.ae_estimator.get_latent(e_critic).detach()

        augmented = torch.cat([o_critic, z_E], dim=-1)  # 131 dim
        return self.critic_mlp(augmented)


# ============================================================
# Helper: build & warm-start MLP
# ============================================================
def _build_mlp(in_dim: int, out_dim: int, hidden_dims: list,
               activation_class: type) -> nn.Sequential:
    layers = []
    cur = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(cur, h))
        layers.append(activation_class())
        cur = h
    layers.append(nn.Linear(cur, out_dim))
    return nn.Sequential(*layers)


def _copy_lora_warm_start(new_mlp: nn.Sequential, old_seq: nn.Sequential, old_in_dim: int, label: str):
    """LoRA-style warm-start: 旧列复制, 新列 = 0.
    
    new_mlp[0] = Linear(new_in, hidden)
    old_seq[0] = Linear(old_in_dim, hidden)
      new_mlp[0].weight[:, :old_in_dim] ← old_seq[0].weight
      new_mlp[0].weight[:, old_in_dim:] ← 0
      new_mlp[0].bias ← old_seq[0].bias
      其他层直接复制 (相同 shape)
    """
    new_first = new_mlp[0]
    old_first = old_seq[0]

    if not (isinstance(new_first, nn.Linear) and isinstance(old_first, nn.Linear)):
        print(f"[Phase3] {label}: first layer not Linear, skipping warm-start")
        return False

    if old_first.in_features != old_in_dim:
        print(f"[Phase3] {label}: old first layer in={old_first.in_features}, expected {old_in_dim}, skipping")
        return False

    new_in = new_first.in_features
    extra = new_in - old_in_dim

    new_first.weight.data.zero_()
    new_first.weight.data[:, :old_in_dim] = old_first.weight.data.clone()
    if new_first.bias is not None and old_first.bias is not None:
        new_first.bias.data = old_first.bias.data.clone()
    print(f"[Phase3] {label} first layer LoRA-style warm-start: "
          f"copy [:, :{old_in_dim}] from old, new [:, {old_in_dim}:{new_in}] = 0 "
          f"(zero init for {extra} new cols → preserve Phase 2b behavior at iter 0)")

    # 其余层直接复制
    for i in range(1, min(len(new_mlp), len(old_seq))):
        nl = new_mlp[i]; ol = old_seq[i]
        if isinstance(nl, nn.Linear) and isinstance(ol, nn.Linear) \
                and nl.weight.shape == ol.weight.shape:
            nl.weight.data = ol.weight.data.clone()
            if nl.bias is not None and ol.bias is not None:
                nl.bias.data = ol.bias.data.clone()
    return True


def _stabilize_vae(vae_estimator):
    """稳定 VAE 输出层: log_var=0, fc_mu/fc_velocity/decoder[-1] × 0.01."""
    with torch.no_grad():
        if hasattr(vae_estimator, "fc_log_var"):
            vae_estimator.fc_log_var.weight.data.zero_()
            vae_estimator.fc_log_var.bias.data.zero_()
        if hasattr(vae_estimator, "fc_mu"):
            vae_estimator.fc_mu.weight.data *= 0.01
            vae_estimator.fc_mu.bias.data.zero_()
        if hasattr(vae_estimator, "fc_velocity"):
            vae_estimator.fc_velocity.weight.data *= 0.01
            vae_estimator.fc_velocity.bias.data.zero_()
        if hasattr(vae_estimator, "decoder") and len(vae_estimator.decoder) > 0:
            last_dec = vae_estimator.decoder[-1]
            if isinstance(last_dec, nn.Linear):
                last_dec.weight.data *= 0.01
                if last_dec.bias is not None:
                    last_dec.bias.data.zero_()


def _stabilize_ae(ae_estimator):
    """稳定 AE 输出层: encoder 最后一层 × 0.01, decoder 最后一层 × 0.01.
    
    AE 在平地下 input ≈ 常数 (-0.78m), recon loss 不会爆炸但仍稳定 init.
    """
    with torch.no_grad():
        # encoder 最后一层 (输出 z_E)
        if hasattr(ae_estimator, "encoder") and len(ae_estimator.encoder) > 0:
            last_enc = ae_estimator.encoder[-1]
            if isinstance(last_enc, nn.Linear):
                last_enc.weight.data *= 0.01
                if last_enc.bias is not None:
                    last_enc.bias.data.zero_()
        # decoder 最后一层 (输出 e_recon)
        if hasattr(ae_estimator, "decoder") and len(ae_estimator.decoder) > 0:
            last_dec = ae_estimator.decoder[-1]
            if isinstance(last_dec, nn.Linear):
                last_dec.weight.data *= 0.01
                if last_dec.bias is not None:
                    last_dec.bias.data.zero_()


# ============================================================
# Main inject function
# ============================================================
def inject_vae_ae_actor(alg, cmoe_cfg: dict, device: str = "cpu"):
    """Phase 3 inject — 替换 actor 为 VAEAEAugmentedActor, critic 为 AEAugmentedCritic.
    
    前置条件: PolicyCfg.history_length=5, 加 elevation term → obs.policy=975 dim
              CriticCfg 加 elevation term → obs.critic=195 dim
    """
    est_cfg = cmoe_cfg.get("estimator", None)
    if est_cfg is None or "vae" not in est_cfg or "ae" not in est_cfg:
        raise RuntimeError("[Phase3] cmoe_cfg.estimator.vae and .ae required")

    vae_cfg = est_cfg["vae"]
    ae_cfg = est_cfg["ae"]

    # ---- 1. 创建 VAE estimator ----
    from cmoe.custom_classes.modules import VAEEstimator
    vae_estimator = VAEEstimator(**vae_cfg).to(device)
    print(f"[Phase3] VAE Estimator: input={vae_estimator.encoder_input_dim}, "
          f"latent={vae_cfg['latent_dim']}, vel={vae_cfg['velocity_dim']}")
    _stabilize_vae(vae_estimator)
    print("[Phase3] VAE output layers stabilized")

    # ---- 2. 创建 AE estimator ----
    from cmoe.custom_classes.modules.ae_estimator import AEEstimator
    ae_estimator = AEEstimator(**ae_cfg).to(device)
    print(f"[Phase3] AE Estimator: input={ae_cfg['input_dim']}, "
          f"latent={ae_cfg['latent_dim']}")
    _stabilize_ae(ae_estimator)
    print("[Phase3] AE output layers stabilized")

    # ---- 3. 关联到 alg ----
    alg.vae_estimator = vae_estimator
    alg.ae_estimator = ae_estimator

    # 单独 optimizers for VAE and AE
    alg.vae_optimizer = optim.Adam(
        vae_estimator.parameters(), lr=est_cfg["learning_rate"]
    )
    alg.ae_optimizer = optim.Adam(
        ae_estimator.parameters(), lr=est_cfg["learning_rate"]
    )
    print(f"[Phase3] VAE/AE optimizers: Adam lr={est_cfg['learning_rate']}")

    # ---- 4. 推断旧 actor / critic 架构 ----
    # rsl_rl 创建的 actor/critic 是某种 nn.Module (在 3.x 中显示为 "MLP" 类),
    # 通常是 nn.Sequential 或其子类. 我们用 .modules() 提取所有 Linear 层,
    # 这是最鲁棒的方式 (不依赖 __iter__ 顺序或具体类型).
    old_actor_mlp = alg.policy.actor
    old_critic_mlp = alg.policy.critic

    if not isinstance(old_actor_mlp, nn.Sequential):
        print(f"[Phase3] WARNING: old actor type is {type(old_actor_mlp).__name__}, "
              f"not nn.Sequential — will use .modules() fallback")
    if not isinstance(old_critic_mlp, nn.Sequential):
        print(f"[Phase3] WARNING: old critic type is {type(old_critic_mlp).__name__}, "
              f"not nn.Sequential — will use .modules() fallback")

    def _extract_mlp_arch(module):
        """从 nn.Module 提取所有 Linear 层 (in_order). 兼容 nn.Sequential 和子类."""
        # 优先尝试 Sequential-style 直接迭代 (与 phase2b 一致)
        if isinstance(module, nn.Sequential):
            linears = [l for l in module if isinstance(l, nn.Linear)]
        else:
            # Fallback: 用 .modules() 递归提取所有 Linear, 按声明顺序
            linears = [m for m in module.modules() if isinstance(m, nn.Linear)]
        return linears

    actor_linears = _extract_mlp_arch(old_actor_mlp)
    critic_linears = _extract_mlp_arch(old_critic_mlp)

    if len(actor_linears) < 2:
        raise RuntimeError(
            f"[Phase3] actor has {len(actor_linears)} Linear layers, expected ≥ 2 "
            f"(at least 1 hidden + 1 output). Module structure: {old_actor_mlp}"
        )
    if len(critic_linears) < 2:
        raise RuntimeError(
            f"[Phase3] critic has {len(critic_linears)} Linear layers, expected ≥ 2. "
            f"Module structure: {old_critic_mlp}"
        )

    actor_hidden_dims = [l.out_features for l in actor_linears[:-1]]
    num_actions = actor_linears[-1].out_features
    critic_hidden_dims = [l.out_features for l in critic_linears[:-1]]

    # 找 activation class — 优先从 actor 中找非 Linear 模块
    activation_class = nn.ELU
    for layer in old_actor_mlp.modules():
        if not isinstance(layer, nn.Linear) and not isinstance(layer, nn.Sequential) \
                and layer is not old_actor_mlp:
            activation_class = type(layer)
            break

    print(f"[Phase3] inferred actor: hidden={actor_hidden_dims}, num_actions={num_actions}")
    print(f"[Phase3] inferred critic: hidden={critic_hidden_dims}")
    print(f"[Phase3] inferred activation: {activation_class.__name__}")

    # ---- 5. 创建新的 actor_mlp (input 166) ----
    new_actor_mlp = _build_mlp(
        ACTOR_MLP_INPUT_DIM, num_actions, actor_hidden_dims, activation_class
    ).to(device)
    print(f"[Phase3] new actor_mlp: Linear({ACTOR_MLP_INPUT_DIM}, {actor_hidden_dims[0]}) "
          f"→ ... → Linear({actor_hidden_dims[-1]}, {num_actions})")

    # ---- 6. 创建新的 critic_mlp (input 131) ----
    new_critic_mlp = _build_mlp(
        CRITIC_MLP_INPUT_DIM, 1, critic_hidden_dims, activation_class
    ).to(device)
    print(f"[Phase3] new critic_mlp: Linear({CRITIC_MLP_INPUT_DIM}, {critic_hidden_dims[0]}) "
          f"→ ... → Linear({critic_hidden_dims[-1]}, 1)")

    # ---- 7. 替换 alg.policy.actor / .critic ----
    augmented_actor = VAEAEAugmentedActor(
        vae_estimator=vae_estimator,
        ae_estimator=ae_estimator,
        actor_mlp=new_actor_mlp,
    ).to(device)
    augmented_critic = AEAugmentedCritic(
        ae_estimator=ae_estimator,
        critic_mlp=new_critic_mlp,
    ).to(device)
    alg.policy.actor = augmented_actor
    alg.policy.critic = augmented_critic
    print("[Phase3] alg.policy.actor replaced with VAEAEAugmentedActor")
    print("[Phase3] alg.policy.critic replaced with AEAugmentedCritic")

    # ---- 8. 重建 PPO optimizer (排除 VAE/AE) ----
    ppo_params = [p for p in alg.policy.parameters()
                  if all(p is not q for q in vae_estimator.parameters())
                  and all(p is not q for q in ae_estimator.parameters())]
    total_params = sum(p.numel() for p in alg.policy.parameters())
    ppo_param_count = sum(p.numel() for p in ppo_params)
    alg.optimizer = optim.Adam(ppo_params, lr=alg.learning_rate)
    print(f"[Phase3] PPO optimizer rebuilt (lr={alg.learning_rate}), "
          f"params {ppo_param_count}/{total_params} (excluded VAE+AE)")

    # ---- 9. Wrap alg.act for caching VAE/AE training data ----
    _orig_act = alg.act
    diag = True

    def _phase3_wrapped_act(*args, **kwargs):
        obs = args[0] if len(args) > 0 else kwargs.get("obs", None)
        try:
            if obs is not None and hasattr(obs, "get"):
                po = obs.get("policy", None)
                co = obs.get("critic", None)

                if not hasattr(alg, "_phase3_first_call"):
                    alg._phase3_first_call = True
                    if hasattr(obs, "keys"):
                        print(f"[Phase3-DEBUG] first act() obs keys: {list(obs.keys())}")
                    if po is not None:
                        print(f"[Phase3-DEBUG]   obs['policy'].shape = {tuple(po.shape)} "
                              f"(expect (N, {EXPECTED_POLICY_OBS_DIM}))")
                    if co is not None:
                        print(f"[Phase3-DEBUG]   obs['critic'].shape = {tuple(co.shape)} "
                              f"(expect (N, {EXPECTED_CRITIC_OBS_DIM}))")

                if (po is not None and co is not None
                        and po.shape[-1] == EXPECTED_POLICY_OBS_DIM
                        and co.shape[-1] == EXPECTED_CRITIC_OBS_DIM):
                    # VAE training data
                    vae_history, vae_target = _extract_vae_history_target(po)
                    vel_gt = co[:, :3]   # base_lin_vel ground truth (critic 单帧 layout)
                    # AE training data — elevation_gt from critic (no noise = privileged)
                    elevation_gt = co[:, 99:99 + ELEVATION_DIM]   # (N, 96)

                    alg._vae_buf["vae_history"].append(vae_history.data.clone())
                    alg._vae_buf["vae_target"].append(vae_target.data.clone())
                    alg._vae_buf["vel_gt"].append(vel_gt.data.clone())
                    alg._ae_buf["elevation_gt"].append(elevation_gt.data.clone())
        except Exception as ex:
            if alg._vae_iter_count <= 3:
                print(f"[Phase3] act hook exception: {ex}")
                import traceback; traceback.print_exc()

        return _orig_act(*args, **kwargs)

    alg._vae_buf = {"vae_history": [], "vae_target": [], "vel_gt": []}
    alg._ae_buf = {"elevation_gt": []}
    alg._vae_iter_count = 0
    alg.act = _phase3_wrapped_act
    print("[Phase3] alg.act wrapped: cache vae_history+target+vel_gt+elevation_gt")

    # ---- 10. Wrap alg.update — 追加 VAE & AE 训练 ----
    _orig_update = alg.update

    def _phase3_wrapped_update(*args, **kwargs):
        result = _orig_update(*args, **kwargs)

        # ===== 训练 VAE (与 Phase 2b 完全相同) =====
        try:
            if (len(alg._vae_buf["vae_history"]) > 0
                    and len(alg._vae_buf["vae_target"]) > 0
                    and len(alg._vae_buf["vel_gt"]) > 0):
                vh_all  = torch.cat(alg._vae_buf["vae_history"],  dim=0)
                vt_all  = torch.cat(alg._vae_buf["vae_target"],   dim=0)
                vel_all = torch.cat(alg._vae_buf["vel_gt"],       dim=0)
                total = vh_all.shape[0]
                vae_loss_avg = 0.0
                n_batches = 0

                if total > 0:
                    n_epochs = getattr(alg, "num_learning_epochs", 5)
                    n_mb = getattr(alg, "num_mini_batches", 4)
                    mbs = max(total // n_mb, 256)

                    alg.vae_estimator.train()
                    for ep in range(n_epochs):
                        perm = torch.randperm(total, device=vh_all.device)
                        for s in range(0, total, mbs):
                            e = min(s + mbs, total)
                            idx = perm[s:e]
                            if len(idx) == 0:
                                continue
                            mb_vh  = vh_all[idx]
                            mb_vt  = vt_all[idx]
                            mb_vel = vel_all[idx]

                            alg.vae_optimizer.zero_grad()
                            loss, _ = alg.vae_estimator.compute_loss(mb_vh, mb_vt, mb_vel)
                            if torch.isnan(loss) or torch.isinf(loss):
                                continue
                            loss.backward()
                            nn.utils.clip_grad_norm_(
                                alg.vae_estimator.parameters(),
                                getattr(alg, "max_grad_norm", 1.0),
                            )
                            alg.vae_optimizer.step()
                            vae_loss_avg += loss.item()
                            n_batches += 1
                    alg.vae_estimator.eval()
        except Exception as ex:
            if diag:
                print(f"[Phase3] VAE train exception: {ex}")
                import traceback; traceback.print_exc()

        # ===== 训练 AE (NEW Phase 3) =====
        try:
            if len(alg._ae_buf["elevation_gt"]) > 0:
                e_all = torch.cat(alg._ae_buf["elevation_gt"], dim=0)
                total = e_all.shape[0]
                ae_loss_avg = 0.0
                n_ae_batches = 0

                if total > 0:
                    n_epochs = getattr(alg, "num_learning_epochs", 5)
                    n_mb = getattr(alg, "num_mini_batches", 4)
                    mbs = max(total // n_mb, 256)

                    alg.ae_estimator.train()
                    for ep in range(n_epochs):
                        perm = torch.randperm(total, device=e_all.device)
                        for s in range(0, total, mbs):
                            e = min(s + mbs, total)
                            idx = perm[s:e]
                            if len(idx) == 0:
                                continue
                            mb_e = e_all[idx]

                            alg.ae_optimizer.zero_grad()
                            loss, _ = alg.ae_estimator.compute_loss(mb_e)
                            if torch.isnan(loss) or torch.isinf(loss):
                                continue
                            loss.backward()
                            nn.utils.clip_grad_norm_(
                                alg.ae_estimator.parameters(),
                                getattr(alg, "max_grad_norm", 1.0),
                            )
                            alg.ae_optimizer.step()
                            ae_loss_avg += loss.item()
                            n_ae_batches += 1
                    alg.ae_estimator.eval()

                    if n_ae_batches > 0 and alg._vae_iter_count % 50 == 0:
                        print(f"[Phase3b] iter={alg._vae_iter_count}  "
                              f"ae_buf={len(alg._ae_buf['elevation_gt'])}  "
                              f"ae_batches={n_ae_batches}  "
                              f"ae_loss={ae_loss_avg / max(n_ae_batches, 1):.4f}")
        except Exception as ex:
            if diag:
                print(f"[Phase3] AE train exception: {ex}")
                import traceback; traceback.print_exc()

        # 清空缓冲区
        alg._vae_buf = {"vae_history": [], "vae_target": [], "vel_gt": []}
        alg._ae_buf = {"elevation_gt": []}
        alg._vae_iter_count += 1

        return result

    alg.update = _phase3_wrapped_update
    print("[Phase3] alg.update wrapped: PPO + VAE training + AE training")
    print("[Phase3] VAE+AE+actor+critic injection complete")


# ============================================================
# Runner save / load patches
# ============================================================
def inject_phase3_runner_patches(runner):
    """Patch runner.save / runner.load 以保存/加载 VAE+AE state."""
    _orig_save = runner.save
    _orig_load = runner.load

    def patched_save(path, infos=None):
        # 先调原始 save (保存 model_state_dict / optimizer_state_dict)
        _orig_save(path, infos)

        # 追加 VAE / AE state
        try:
            ckpt = torch.load(path, weights_only=False, map_location="cpu")
            alg = runner.alg
            if hasattr(alg, "vae_estimator"):
                ckpt["vae_estimator_state_dict"] = alg.vae_estimator.state_dict()
            if hasattr(alg, "vae_optimizer"):
                ckpt["vae_optimizer_state_dict"] = alg.vae_optimizer.state_dict()
            if hasattr(alg, "ae_estimator"):
                ckpt["ae_estimator_state_dict"] = alg.ae_estimator.state_dict()
            if hasattr(alg, "ae_optimizer"):
                ckpt["ae_optimizer_state_dict"] = alg.ae_optimizer.state_dict()
            torch.save(ckpt, path)
        except Exception as ex:
            print(f"[Phase3-SAVE] failed to append VAE/AE state: {ex}")

    def patched_load(path, load_optimizer=True):
        ckpt = torch.load(path, weights_only=False, map_location="cpu")
        alg = runner.alg
        device = next(alg.policy.parameters()).device

        # 判断 checkpoint 类型
        is_phase3 = "ae_estimator_state_dict" in ckpt
        is_phase2b = "vae_estimator_state_dict" in ckpt and not is_phase3

        if is_phase3:
            print("[Phase3-LOAD] Phase 3 checkpoint (resume)")
            try:
                ret = alg.policy.load_state_dict(ckpt["model_state_dict"], strict=False)
                if hasattr(ret, "missing_keys") and hasattr(ret, "unexpected_keys"):
                    m = list(ret.missing_keys); u = list(ret.unexpected_keys)
                    if m: print(f"[Phase3-LOAD]   policy missing (first 3): {m[:3]}")
                    if u: print(f"[Phase3-LOAD]   policy unexpected (first 3): {u[:3]}")
                # 验证 actor 真的被加载
                with torch.no_grad():
                    aw = alg.policy.actor.actor_mlp[0].weight
                    new_cols_norm = aw[:, 134:].norm().item()  # z_E columns
                    print(f"[Phase3-LOAD] policy ✓ (actor z_E cols [:, 134:] norm={new_cols_norm:.4f})")
                    cw = alg.policy.critic.critic_mlp[0].weight
                    crit_z_e_norm = cw[:, 99:].norm().item()
                    print(f"[Phase3-LOAD] critic ✓ (critic z_E cols [:, 99:] norm={crit_z_e_norm:.4f})")
            except Exception as ex:
                print(f"[Phase3-LOAD] strict=False load returned weird value ({ex})")
                # fallback: manual key copy
                policy_sd = alg.policy.state_dict()
                ckpt_sd = ckpt["model_state_dict"]
                copied = 0
                with torch.no_grad():
                    for k, v in ckpt_sd.items():
                        if k in policy_sd and policy_sd[k].shape == v.shape:
                            policy_sd[k].copy_(v.to(policy_sd[k].device))
                            copied += 1
                print(f"[Phase3-LOAD] manual: copied {copied} keys")

            if "optimizer_state_dict" in ckpt and load_optimizer:
                try:
                    alg.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    print("[Phase3-LOAD] PPO optimizer ✓")
                except Exception as ex:
                    print(f"[Phase3-LOAD] PPO optimizer skipped: {ex}")

        elif is_phase2b:
            # Warm-start from Phase 2b: 134-dim actor, 99-dim critic
            print("[Phase3-LOAD] Phase 2b checkpoint (warm-start path)")
            try:
                old_state = ckpt["model_state_dict"]
                # 提取 Phase 2b 的 actor / critic Linear weights
                old_actor_keys = {k: v for k, v in old_state.items() if k.startswith("actor.")}
                old_critic_keys = {k: v for k, v in old_state.items() if k.startswith("critic.")}

                # 重构 Phase 2b actor_mlp (input 134)
                from cmoe.tasks.velocity.cmoe_phase2b.phase2b_inject import (
                    _build_actor_mlp as _phase2b_build_mlp,
                    ACTOR_MLP_INPUT_DIM as PHASE2B_ACTOR_IN,
                )
                # phase2b actor_mlp keys: actor.actor_mlp.0.weight, actor.actor_mlp.0.bias, ...
                # 但 phase2b 的 saved actor 是 VAEAugmentedActor wrapper, 内部 actor_mlp
                # 实际 keys: actor.actor_mlp.X.weight (X = 0, 2, 4, 6 for 4 Linear layers)
                # 我们要构建临时 phase2b actor_mlp 来做 warm-start

                # 找 phase2b actor_mlp 的 hidden dims
                phase2b_actor_layers = sorted(
                    [(int(k.split(".")[2]), v) for k, v in old_actor_keys.items()
                     if k.startswith("actor.actor_mlp.") and ".weight" in k],
                    key=lambda x: x[0]
                )
                if len(phase2b_actor_layers) == 0:
                    raise RuntimeError("Phase 2b actor.actor_mlp keys not found in checkpoint")

                # 构建一个临时 nn.Sequential 来承载 phase2b actor weights
                act_class = type(alg.policy.actor.actor_mlp[1])
                phase2b_hidden_dims = []
                phase2b_out_dim = None
                for idx, w in phase2b_actor_layers:
                    if phase2b_out_dim is None:
                        phase2b_hidden_dims.append(w.shape[0])
                    phase2b_out_dim = w.shape[0]
                phase2b_hidden_dims = phase2b_hidden_dims[:-1]
                num_actions = phase2b_out_dim

                tmp_phase2b_actor = _build_mlp(
                    PHASE2B_ACTOR_IN, num_actions, phase2b_hidden_dims, act_class
                )
                # 加载 phase2b 权重到 tmp
                tmp_state = {}
                for k, v in old_actor_keys.items():
                    if k.startswith("actor.actor_mlp."):
                        new_k = k[len("actor.actor_mlp."):]
                        tmp_state[new_k] = v
                tmp_phase2b_actor.load_state_dict(tmp_state, strict=False)

                # LoRA-style warm-start to new (166-dim) actor_mlp
                _copy_lora_warm_start(
                    alg.policy.actor.actor_mlp, tmp_phase2b_actor,
                    PHASE2B_ACTOR_IN, "actor"
                )

                # ----- Critic: phase2b critic 是简单 MLP (input 99), keys: critic.0.weight, critic.2.weight, ...
                # phase3 critic_mlp input 131
                phase2b_critic_layers = sorted(
                    [(int(k.split(".")[1]), v) for k, v in old_critic_keys.items()
                     if k.startswith("critic.") and ".weight" in k and len(k.split(".")) == 3],
                    key=lambda x: x[0]
                )
                if len(phase2b_critic_layers) > 0:
                    phase2b_critic_hidden_dims = []
                    phase2b_critic_out_dim = None
                    for idx, w in phase2b_critic_layers:
                        if phase2b_critic_out_dim is None:
                            phase2b_critic_hidden_dims.append(w.shape[0])
                        phase2b_critic_out_dim = w.shape[0]
                    phase2b_critic_hidden_dims = phase2b_critic_hidden_dims[:-1]

                    tmp_phase2b_critic = _build_mlp(
                        99, phase2b_critic_out_dim, phase2b_critic_hidden_dims, act_class
                    )
                    tmp_critic_state = {}
                    for k, v in old_critic_keys.items():
                        if k.startswith("critic.") and len(k.split(".")) == 3:
                            new_k = k[len("critic."):]
                            tmp_critic_state[new_k] = v
                    tmp_phase2b_critic.load_state_dict(tmp_critic_state, strict=False)

                    _copy_lora_warm_start(
                        alg.policy.critic.critic_mlp, tmp_phase2b_critic,
                        99, "critic"
                    )
            except Exception as ex:
                print(f"[Phase3-LOAD] Phase 2b warm-start failed: {ex}")
                import traceback; traceback.print_exc()
            # 不加载 PPO optimizer (架构不同)
            print("[Phase3-LOAD]   skipping PPO optimizer (architecture changed)")

            # 加载 Phase 2b 的 VAE
            if "vae_estimator_state_dict" in ckpt:
                try:
                    state = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                             for k, v in ckpt["vae_estimator_state_dict"].items()}
                    ret = alg.vae_estimator.load_state_dict(state, strict=False)
                    if hasattr(ret, "missing_keys"):
                        m = list(ret.missing_keys); u = list(ret.unexpected_keys)
                        if m or u:
                            print(f"[Phase3-LOAD] vae_estimator: missing={m[:3]} unexpected={u[:3]}")
                        else:
                            print("[Phase3-LOAD] vae_estimator ✓ (warm-start from Phase 2b)")
                    else:
                        print("[Phase3-LOAD] vae_estimator ✓ (warm-start from Phase 2b)")
                except Exception as ex:
                    print(f"[Phase3-LOAD] vae_estimator load failed: {ex}")

            # AE: 没有 phase 2b 版本, 保持 fresh init
            print("[Phase3-LOAD] ae_estimator: fresh init (Phase 2b had no AE)")

        else:
            print(f"[Phase3-LOAD] unknown checkpoint type, attempting policy load only")
            try:
                ret = alg.policy.load_state_dict(ckpt.get("model_state_dict", {}), strict=False)
                print(f"[Phase3-LOAD] policy partial load: type={type(ret).__name__}")
            except Exception as ex:
                print(f"[Phase3-LOAD] policy load failed: {ex}")

        # 加载 Phase 3 VAE / VAE optim / AE / AE optim (resume only)
        if is_phase3:
            if "vae_estimator_state_dict" in ckpt:
                try:
                    state = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                             for k, v in ckpt["vae_estimator_state_dict"].items()}
                    ret = alg.vae_estimator.load_state_dict(state, strict=False)
                    print("[Phase3-LOAD] vae_estimator ✓")
                except Exception as ex:
                    print(f"[Phase3-LOAD] vae_estimator load failed: {ex}")
            if "vae_optimizer_state_dict" in ckpt:
                try:
                    alg.vae_optimizer.load_state_dict(ckpt["vae_optimizer_state_dict"])
                    print("[Phase3-LOAD] vae_optimizer ✓")
                except Exception as ex:
                    print(f"[Phase3-LOAD] vae_optimizer skipped: {ex}")
            if "ae_estimator_state_dict" in ckpt:
                try:
                    state = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                             for k, v in ckpt["ae_estimator_state_dict"].items()}
                    ret = alg.ae_estimator.load_state_dict(state, strict=False)
                    print("[Phase3-LOAD] ae_estimator ✓")
                except Exception as ex:
                    print(f"[Phase3-LOAD] ae_estimator load failed: {ex}")
            if "ae_optimizer_state_dict" in ckpt:
                try:
                    alg.ae_optimizer.load_state_dict(ckpt["ae_optimizer_state_dict"])
                    print("[Phase3-LOAD] ae_optimizer ✓")
                except Exception as ex:
                    print(f"[Phase3-LOAD] ae_optimizer skipped: {ex}")

        # 标记 last load iteration
        if "iter" in ckpt:
            runner.current_learning_iteration = ckpt["iter"]

    runner.save = patched_save
    runner.load = patched_load
    print("[Phase3] runner.save / runner.load patched ✓")
