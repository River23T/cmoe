"""Phase 2b inject — VAE estimator + actor 接收 v_pred + z_H.

============================================================================
设计 (修订, 经实测验证):
============================================================================

PolicyCfg.history_length = 5 → obs.policy 是 495 dim concat

============================================================================
⚠️ CRITICAL — IsaacLab obs layout (fixed v3, 2026-05-03):
============================================================================
IsaacLab 的 ObservationGroupCfg.history_length=N 使用 **per-term flatten**
layout, NOT per-frame layout!

正确 layout (IsaacLab confirmed):
  obs.policy = [
    base_lin_vel_history(15),      # 5×3 = [t-4, t-3, t-2, t-1, t]
    base_ang_vel_history(15),
    projected_gravity_history(15),
    velocity_commands_history(15),
    joint_pos_rel_history(145),    # 5×29 = [jpos_t-4(29), jpos_t-3(29), ..., jpos_t(29)]
    joint_vel_rel_history(145),
    last_action_history(145),
  ]
  total = 4×15 + 3×145 = 60 + 435 = 495 ✓

每个 term 内部: [t-4, t-3, t-2, t-1, t] (oldest to newest, IsaacLab convention)

错误 layout (旧版 _split_x 假设, garbage):
  obs.policy = [frame_t-4(99), frame_t-3(99), ..., frame_t(99)]
  其中 frame_i = [base_lin_vel(3), base_ang_vel(3), gravity(3), cmd(3),
                  jpos(29), jvel(29), last_act(29)]

旧版 _split_x 把 495 错误地切成了 5 个 "frame", 但每个 "frame" 实际上是
garbage (混合不同 term 的不同时间步). 训练时 actor 因此 overfit 到
具体的 garbage 分布, 评估时 obs 分布略变 (64 envs vs 4096) 即失效.

============================================================================
我们 inject 替换 alg.policy.actor = VAEAugmentedActor(...):
  forward(x_495):
    o_t_full   = extract_frame(x_495, 0)        # 最新一帧 99 dim, REAL frame!
    
    vae_history = []
    for i in range(5):                          # 旧→新: i=0 是最旧
      f = extract_frame(x_495, 4 - i)           
      vae_history.append(f[:, 3:])              # drop base_lin_vel, 96 dim
    vae_history = torch.cat(vae_history, -1)    # 96×5 = 480 dim
    
    with no_grad: z_H, v_pred = vae(vae_history)
    augmented = cat([o_t_full 99, v_pred 3, z_H 32])  # 134 dim
    return actor_mlp(augmented)

actor_mlp 真正第一层 Linear(134, 512), warm-start 兼容 Phase 2a.

VAE supervision:
  vel_gt = obs.critic[:, :3]   (base_lin_vel ground truth)
  vae_target = extract_frame(x_495, 0)[:, 3:]  (最新一帧 96 dim)
============================================================================
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim


# 常量 (Phase 2b 平地 obs 结构)
POLICY_FRAME_DIM = 99    # 单帧 policy obs (含 base_lin_vel)
BASE_LIN_VEL_DIM = 3     # 单帧 obs 前 3 dim 是 base_lin_vel
OBS96_DIM = POLICY_FRAME_DIM - BASE_LIN_VEL_DIM   # 96 (paper Eq 2)
HISTORY_LEN = 5          # 5 frames
VAE_INPUT_DIM = OBS96_DIM * HISTORY_LEN   # 480
ACTOR_MLP_INPUT_DIM = POLICY_FRAME_DIM + 3 + 32   # 99 + v_pred(3) + z_H(32) = 134
EXPECTED_POLICY_OBS_DIM = POLICY_FRAME_DIM * HISTORY_LEN   # 495

# IsaacLab obs layout: 每个 term 的 dim (单帧, 不带 history)
# 顺序必须与 PolicyCfg 中 ObsTerm 定义顺序一致!
OBS_TERM_DIMS = (3, 3, 3, 3, 29, 29, 29)
# 累计 offset (每个 term 在 495 中的起始位置, history 已展平):
# base_lin_vel: 0:15  (3*5=15)
# base_ang_vel: 15:30
# gravity:      30:45
# cmd:          45:60
# jpos:         60:205 (29*5=145)
# jvel:         205:350
# last_action:  350:495
def _compute_term_offsets():
    """Compute starting index of each term in flattened (495,) obs."""
    offsets = [0]
    for term_dim in OBS_TERM_DIMS[:-1]:
        offsets.append(offsets[-1] + term_dim * HISTORY_LEN)
    return offsets

OBS_TERM_OFFSETS = _compute_term_offsets()  # [0, 15, 30, 45, 60, 205, 350]
# 验证总维度:
assert sum(d * HISTORY_LEN for d in OBS_TERM_DIMS) == EXPECTED_POLICY_OBS_DIM, \
    f"obs layout mismatch: {sum(d * HISTORY_LEN for d in OBS_TERM_DIMS)} != {EXPECTED_POLICY_OBS_DIM}"


def _extract_frame_from_obs(x: torch.Tensor, frame_idx: int) -> torch.Tensor:
    """Extract a single frame (99 dim) from per-term-flattened obs (495 dim).
    
    Args:
        x: (N, 495) flattened obs with per-term-history layout.
        frame_idx: 0=oldest (t-4), 4=newest (t).
    
    Returns:
        (N, 99) single frame [base_lin_vel(3), base_ang_vel(3), gravity(3),
                              cmd(3), jpos(29), jvel(29), last_action(29)]
    """
    parts = []
    for term_id, term_dim in enumerate(OBS_TERM_DIMS):
        # In per-term layout, term_id occupies [offset : offset + term_dim*HISTORY_LEN]
        # within that, frame_idx selects [frame_idx*term_dim : (frame_idx+1)*term_dim]
        start = OBS_TERM_OFFSETS[term_id] + frame_idx * term_dim
        parts.append(x[:, start : start + term_dim])
    return torch.cat(parts, dim=-1)


def _extract_vae_history_target(x: torch.Tensor):
    """Extract VAE inputs from per-term-flattened obs.
    
    Args:
        x: (N, 495) flattened obs.
    
    Returns:
        vae_history: (N, 480) = 5 frames × 96 dim (each = frame minus base_lin_vel)
                     ordered oldest→newest [t-4, t-3, t-2, t-1, t]
        vae_target:  (N, 96)  = newest frame minus base_lin_vel (t)
    """
    history_frames = []
    for i in range(HISTORY_LEN):  # 0=oldest, HISTORY_LEN-1=newest
        f = _extract_frame_from_obs(x, i)              # (N, 99)
        history_frames.append(f[:, BASE_LIN_VEL_DIM:]) # (N, 96), drop base_lin_vel
    vae_history = torch.cat(history_frames, dim=-1)    # (N, 480)
    vae_target  = history_frames[-1]                   # (N, 96), newest frame
    return vae_history, vae_target


class VAEAugmentedActor(nn.Module):
    """Actor wrapper: 接收 495 dim policy obs, 内部 VAE encode + concat 后送给 actor_mlp.
    
    forward(x_495):
      o_t = x_495[:, -99:]                                    # 最近一帧
      vae_history = 拼接 5 帧的 [3:] 部分 = 480 dim
      with no_grad: z_H, v_pred = vae(vae_history)
      augmented = cat([o_t, v_pred, z_H], -1)                # 134 dim
      return actor_mlp(augmented)
    
    __getitem__: 转发到 actor_mlp[idx], 兼容 rsl_rl 内部 self.actor[i] 访问.
    """

    def __init__(
        self,
        vae_estimator: nn.Module,
        actor_mlp: nn.Sequential,
        policy_frame_dim: int = POLICY_FRAME_DIM,
        history_len: int = HISTORY_LEN,
        base_lin_vel_dim: int = BASE_LIN_VEL_DIM,
    ):
        super().__init__()
        self.vae_estimator = vae_estimator
        self.actor_mlp = actor_mlp
        self.policy_frame_dim = policy_frame_dim
        self.history_len = history_len
        self.base_lin_vel_dim = base_lin_vel_dim
        # 缓存最近一次 VAE 输入和输出, 给 update wrap 训练 VAE 用
        # (避免重复从 storage 提取 + 重复 forward)
        self._cached_vae_input = None  # (N, 480)
        self._cached_vae_target = None  # (N, 96), 最新一帧
        self._cache_buffer_x = None    # 缓存所有 act 输入用于 update 训 VAE

    def __getitem__(self, idx):
        return self.actor_mlp[idx]

    def __len__(self):
        return len(self.actor_mlp)

    def _split_x(self, x: torch.Tensor):
        """从 495 dim per-term-flattened obs 提取 (o_t_full 99, vae_history 480, vae_target 96).
        
        ⚠️ IsaacLab obs layout 是 per-term flatten, NOT per-frame!
        见模块 docstring 的 layout 说明.
        """
        D = x.shape[-1]
        expected = self.policy_frame_dim * self.history_len
        if D != expected:
            raise RuntimeError(
                f"[VAEAugmentedActor] Expected x.shape[-1] = {expected} "
                f"(={self.policy_frame_dim}×{self.history_len}), got {D}"
            )

        # 最新一帧 99 dim (frame_idx = HISTORY_LEN-1 = 4 = newest = t)
        o_t_full = _extract_frame_from_obs(x, HISTORY_LEN - 1)

        # vae_history: 5 帧 (oldest→newest), 每帧去掉 base_lin_vel = 96 dim
        # vae_target: 最新一帧的 96 dim
        vae_history, vae_target = _extract_vae_history_target(x)

        return o_t_full, vae_history, vae_target

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o_t_full, vae_history, vae_target = self._split_x(x)

        # VAE encode (no_grad: 不让 grad 流到 VAE; VAE 单独训练)
        with torch.no_grad():
            z_H, v_pred = self.vae_estimator.get_latent_and_velocity(vae_history)
            z_H = z_H.detach()
            v_pred = v_pred.detach()

        augmented = torch.cat([o_t_full, v_pred, z_H], dim=-1)   # 134 dim
        return self.actor_mlp(augmented)


def _build_actor_mlp(in_dim: int, out_dim: int, hidden_dims: list,
                     activation_class: type) -> nn.Sequential:
    layers = []
    cur = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(cur, h))
        layers.append(activation_class())
        cur = h
    layers.append(nn.Linear(cur, out_dim))
    return nn.Sequential(*layers)


def _copy_actor_warm_start(new_mlp: nn.Sequential, old_seq: nn.Sequential, old_in_dim: int):
    """Copy old MLP weights to new MLP with first-layer LoRA-style extension.
    
    new_mlp[0] = Linear(134, 512)
    old_seq[0] = Linear(99, 512)  (Phase 2a checkpoint)
      new_mlp[0].weight[:, :99] ← old_seq[0].weight (复制旧权重)
      new_mlp[0].weight[:, 99:] ← 0 (LoRA-style 全 0 init: 新增输入对 actor 输出无影响)
      new_mlp[0].bias ← old_seq[0].bias (复制旧 bias)
    
    为何 [:, 99:] = 0?
      Phase 2a 的 VAE checkpoint 实际未训练 (Phase 2a history_length 不匹配, VAE 只是
      初始化状态). VAE 的 random init forward 会输出 v_pred 和 z_H, 数值可能数倍随机.
      若 [:, 99:] 用 Xavier × 0.1 (有正负), 与 random VAE 输出相乘会扰动 actor 行为,
      破坏 Phase 2a 学到的步态 (实测 49% 摔倒).
      
      [:, 99:] = 0 时, 在 forward 计算 W·x 中, 随机 VAE 输出 (3+32=35 dim) 不会贡献任何
      信号给 actor 输出 → Phase 2a 步态 100% 保留, 然后 gradient 训练逐步学到 v_pred/z_H.
    
    其余 Linear 层 (1, 2, ..., 5) 形状相同, 直接复制 weights & bias.
    """
    new_first = new_mlp[0]
    old_first = old_seq[0]

    if not (isinstance(new_first, nn.Linear) and isinstance(old_first, nn.Linear)):
        print("[Phase2b] warm-start: first layers not Linear, skipping copy")
        return

    if old_first.in_features != old_in_dim:
        print(f"[Phase2b] warm-start: old first layer in={old_first.in_features}, expected {old_in_dim}, skipping")
        return

    new_in = new_first.in_features
    extra = new_in - old_in_dim   # 35 = 3 (v_pred) + 32 (z_H)

    # LoRA-style: 旧列 = 旧权重, 新列 = 0
    new_first.weight.data.zero_()
    new_first.weight.data[:, :old_in_dim] = old_first.weight.data.clone()
    if new_first.bias is not None and old_first.bias is not None:
        new_first.bias.data = old_first.bias.data.clone()
    print(f"[Phase2b] actor first layer LoRA-style warm-start: "
          f"copy [:, :{old_in_dim}] from old, new [:, {old_in_dim}:{new_in}] = 0 "
          f"(zero init for {extra} new cols → preserve Phase 2a behavior at iter 0)")

    # 其余 Linear 层直接复制 (相同 shape)
    for i in range(1, min(len(new_mlp), len(old_seq))):
        nl = new_mlp[i]; ol = old_seq[i]
        if isinstance(nl, nn.Linear) and isinstance(ol, nn.Linear) \
                and nl.weight.shape == ol.weight.shape:
            nl.weight.data = ol.weight.data.clone()
            if nl.bias is not None and ol.bias is not None:
                nl.bias.data = ol.bias.data.clone()


def inject_vae_actor(alg, cmoe_cfg: dict, device: str = "cpu"):
    """Phase 2b inject — 替换 alg.policy.actor 为 VAEAugmentedActor.
    
    前置条件: PolicyCfg.history_length = 5 (obs.policy 是 495 dim).
    """
    est_cfg = cmoe_cfg.get("estimator", None)
    if est_cfg is None or "vae" not in est_cfg:
        raise RuntimeError("[Phase2b] cmoe_cfg.estimator.vae missing")

    vae_cfg = est_cfg["vae"]

    # ---- 1. 创建 VAE estimator ----
    from cmoe.custom_classes.modules import VAEEstimator
    vae_estimator = VAEEstimator(**vae_cfg).to(device)
    print(f"[Phase2b] VAE Estimator: input={vae_estimator.encoder_input_dim}, "
          f"latent={vae_cfg['latent_dim']}, vel={vae_cfg['velocity_dim']}")

    # ---- 1b. 稳定 VAE 初始化: 关键输出层 ~0 init 避免 first-iter loss 爆炸 ----
    # VAE random init 时, 经过 encoder 2 层 + decoder 3 层, 输出可能数值很大,
    # MSE recon loss 可达 1e9+ (上次 log 见 6.9e9). 避免方法:
    #   - log_var 输出层初始为 0  → log_var=0 → exp(log_var)=1 → KL 项小
    #   - decoder 最后一层 weight 缩小 → recon 输出 ~0 → recon_loss = vae_target^2 ~ O(1)
    #   - fc_velocity 输出层缩小 → v_pred ~ 0 → vel_loss = vel_gt^2 ~ O(1)
    with torch.no_grad():
        # log_var 输出 = 0 (KL 起点合理)
        if hasattr(vae_estimator, "fc_log_var"):
            vae_estimator.fc_log_var.weight.data.zero_()
            vae_estimator.fc_log_var.bias.data.zero_()
        # mu 输出小 (latent z ~ 0, 与 prior N(0,1) 接近)
        if hasattr(vae_estimator, "fc_mu"):
            vae_estimator.fc_mu.weight.data *= 0.01
            vae_estimator.fc_mu.bias.data.zero_()
        # velocity 输出小 (v_pred ~ 0, vel_loss = ||vel_gt||^2 ~ O(1))
        if hasattr(vae_estimator, "fc_velocity"):
            vae_estimator.fc_velocity.weight.data *= 0.01
            vae_estimator.fc_velocity.bias.data.zero_()
        # decoder 最后一层小 (recon ~ 0, recon_loss = ||vae_target||^2 ~ O(1))
        if hasattr(vae_estimator, "decoder") and len(vae_estimator.decoder) > 0:
            last_dec = vae_estimator.decoder[-1]
            if isinstance(last_dec, nn.Linear):
                last_dec.weight.data *= 0.01
                last_dec.bias.data.zero_()
    print("[Phase2b] VAE output layers stabilized: fc_log_var=0, "
          "fc_mu/fc_velocity/decoder[-1] × 0.01 (avoids first-iter loss explosion)")

    # ---- 2. VAE optimizer (单独, 不在 PPO optimizer 中) ----
    lr = est_cfg.get("learning_rate", 1e-3)
    alg.vae_optimizer = optim.Adam(vae_estimator.parameters(), lr=lr)
    print(f"[Phase2b] VAE optimizer: Adam lr={lr}")

    # ---- 3. 找到 alg.policy.actor 旧 MLP, 推断维度 ----
    policy = alg.policy
    if not hasattr(policy, "actor"):
        raise RuntimeError("[Phase2b] alg.policy has no .actor attribute")
    old_actor = policy.actor

    if not isinstance(old_actor, nn.Sequential):
        raise RuntimeError(f"[Phase2b] expected old actor Sequential, got {type(old_actor)}")
    if len(old_actor) == 0:
        raise RuntimeError("[Phase2b] old actor empty")
    first = old_actor[0]
    if not isinstance(first, nn.Linear):
        raise RuntimeError(f"[Phase2b] expected first layer Linear, got {type(first)}")

    if first.in_features != EXPECTED_POLICY_OBS_DIM:
        print(f"[Phase2b] WARNING: old actor first in_features={first.in_features}, "
              f"expected {EXPECTED_POLICY_OBS_DIM} (= 99 × 5 history). "
              f"Did PolicyCfg.history_length = 5?")

    # 推断 hidden dims 和 num_actions
    linears = [l for l in old_actor if isinstance(l, nn.Linear)]
    hidden_dims = [l.out_features for l in linears[:-1]]
    num_actions = linears[-1].out_features
    print(f"[Phase2b] inferred from old actor: hidden_dims={hidden_dims}, num_actions={num_actions}")

    # 找 activation class
    act_class = nn.ELU
    for layer in old_actor:
        if not isinstance(layer, nn.Linear):
            act_class = type(layer)
            break

    # ---- 4. 构建新 actor_mlp (input 134) ----
    actor_mlp = _build_actor_mlp(ACTOR_MLP_INPUT_DIM, num_actions, hidden_dims, act_class).to(device)
    print(f"[Phase2b] new actor_mlp: Linear({ACTOR_MLP_INPUT_DIM}, {hidden_dims[0]}) → ... → Linear({hidden_dims[-1]}, {num_actions})")

    # ---- 5. 替换 alg.policy.actor ----
    aug_actor = VAEAugmentedActor(
        vae_estimator=vae_estimator,
        actor_mlp=actor_mlp,
    ).to(device)
    policy.actor = aug_actor
    alg.vae_estimator = vae_estimator
    print("[Phase2b] alg.policy.actor replaced with VAEAugmentedActor")

    # ---- 6. 重建 PPO optimizer (排除 VAE 参数, VAE 有单独 optimizer) ----
    if hasattr(alg, "optimizer") and alg.optimizer is not None:
        try:
            ppo_lr = alg.optimizer.param_groups[0]["lr"]
        except Exception:
            ppo_lr = 1e-3
        non_vae_params = [
            p for n, p in alg.policy.named_parameters()
            if "vae_estimator" not in n
        ]
        alg.optimizer = optim.Adam(non_vae_params, lr=ppo_lr)
        n_total = sum(p.numel() for p in alg.policy.parameters())
        n_non_vae = sum(p.numel() for p in non_vae_params)
        print(f"[Phase2b] PPO optimizer rebuilt (lr={ppo_lr}), "
              f"params {n_non_vae}/{n_total} (excluded VAE)")

    # ---- 7. VAE rollout buffer ----
    alg._vae_buf = {
        "vae_history": [],
        "vae_target":  [],
        "vel_gt":      [],
    }
    alg._vae_iter_count = 0
    alg._vae_diag_every = 50

    # ---- 8. Wrap alg.act 仅用于缓冲 vae_history+vae_target+vel_gt ----
    _orig_act = alg.act

    def _phase2b_wrapped_act(*args, **kwargs):
        obs = args[0] if len(args) > 0 else kwargs.get("obs", None)
        try:
            if obs is not None and hasattr(obs, "get"):
                po = obs.get("policy", None)
                co = obs.get("critic", None)

                if not hasattr(alg, "_phase2b_first_call"):
                    alg._phase2b_first_call = True
                    if hasattr(obs, "keys"):
                        print(f"[Phase2b-DEBUG] first act() obs keys: {list(obs.keys())}")
                    if po is not None:
                        print(f"[Phase2b-DEBUG]   obs['policy'].shape = {tuple(po.shape)} (expect (N, {EXPECTED_POLICY_OBS_DIM}))")
                    if co is not None:
                        print(f"[Phase2b-DEBUG]   obs['critic'].shape = {tuple(co.shape)} (expect (N, 99))")

                if po is not None and co is not None and po.shape[-1] == EXPECTED_POLICY_OBS_DIM:
                    # 从 obs.policy 提取 vae_history (480 dim) 和 vae_target (96 dim)
                    # 用正确的 per-term-flattened layout (与 VAEAugmentedActor 一致)
                    vae_history, vae_target = _extract_vae_history_target(po)
                    vel_gt = co[:, :3]   # base_lin_vel ground truth (critic obs 单帧 layout, 前 3 dim)

                    alg._vae_buf["vae_history"].append(vae_history.data.clone())
                    alg._vae_buf["vae_target"].append(vae_target.data.clone())
                    alg._vae_buf["vel_gt"].append(vel_gt.data.clone())
        except Exception as ex:
            if alg._vae_iter_count <= 3:
                print(f"[Phase2b] act hook exception: {ex}")
                import traceback; traceback.print_exc()

        return _orig_act(*args, **kwargs)

    alg.act = _phase2b_wrapped_act
    print("[Phase2b] alg.act wrapped: cache vae_history+target+vel_gt")

    # ---- 9. Wrap alg.update — 追加 VAE 训练 ----
    _orig_update = alg.update

    def _phase2b_wrapped_update(*args, **kwargs):
        info = _orig_update(*args, **kwargs)
        alg._vae_iter_count += 1
        diag = (alg._vae_iter_count <= 5) or (alg._vae_iter_count % alg._vae_diag_every == 0)

        vae_loss_avg = 0.0
        n_batches = 0
        if len(alg._vae_buf["vae_history"]) > 0:
            try:
                with torch.enable_grad():
                    vh_all  = torch.cat(alg._vae_buf["vae_history"], dim=0)
                    vt_all  = torch.cat(alg._vae_buf["vae_target"],  dim=0)
                    vel_all = torch.cat(alg._vae_buf["vel_gt"],      dim=0)

                    total = vh_all.shape[0]
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
                    print(f"[Phase2b] VAE train exception: {ex}")
                    import traceback; traceback.print_exc()

        if n_batches > 0:
            vae_loss_avg /= n_batches

        if diag:
            buf_len = len(alg._vae_buf["vae_history"])
            print(f"[Phase2b] iter={alg._vae_iter_count}  "
                  f"vae_buf={buf_len}  vae_batches={n_batches}  "
                  f"vae_loss={vae_loss_avg:.4f}")
            if alg._vae_iter_count <= 3 and n_batches > 0:
                with torch.no_grad():
                    z_eval, v_eval = alg.vae_estimator.get_latent_and_velocity(mb_vh)
                    err = (v_eval - mb_vel).abs().mean().item()
                    print(f"[Phase2b] v_pred[0]={v_eval[0].cpu().numpy()}  "
                          f"v_gt[0]={mb_vel[0].cpu().numpy()}  abs_err_mean={err:.4f}")

        for k in alg._vae_buf:
            alg._vae_buf[k] = []

        try:
            if isinstance(info, dict):
                info["estimator"] = vae_loss_avg
        except Exception:
            pass
        return info

    alg.update = _phase2b_wrapped_update
    print("[Phase2b] alg.update wrapped: PPO + separate VAE training")
    print("[Phase2b] VAE-actor injection complete")


# ==============================================================================
# Runner save/load patches
# ==============================================================================
def inject_phase2b_runner_patches(runner):
    alg = runner.alg
    if not hasattr(alg, "vae_estimator") or alg.vae_estimator is None:
        print("[Phase2b] No VAE on alg, skipping runner patch")
        return

    orig_save = runner.save
    orig_load = runner.load

    def patched_save(path, infos=None):
        orig_save(path, infos=infos)
        try:
            saved = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            saved = torch.load(path, map_location="cpu")
        saved["vae_estimator_state_dict"] = alg.vae_estimator.state_dict()
        if hasattr(alg, "vae_optimizer") and alg.vae_optimizer is not None:
            saved["vae_optimizer_state_dict"] = alg.vae_optimizer.state_dict()
        torch.save(saved, path)
        cur_iter = getattr(runner, "current_learning_iteration", 0)
        if cur_iter <= 5 or cur_iter % 1000 == 0:
            keys = [k for k in saved.keys() if "state_dict" in k]
            print(f"[Phase2b-SAVE] iter={cur_iter} → wrote keys: {sorted(keys)}")

    def patched_load(path, load_optimizer=True):
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location="cpu")

        device = next(alg.policy.parameters()).device

        # 检测 checkpoint 类型
        is_phase2a = False
        is_phase2b = False
        if "model_state_dict" in ckpt:
            for k in ckpt["model_state_dict"].keys():
                if "actor.actor_mlp" in k:
                    is_phase2b = True; break
                elif k == "actor.0.weight":
                    is_phase2a = True

        if is_phase2b:
            print("[Phase2b-LOAD] Phase 2b checkpoint (resume)")
            # 兼容 rsl_rl 不同版本: load_state_dict() 在 strict=False 时
            # 返回 _IncompatibleKeys(missing, unexpected) 或 bool, 必须分开处理.
            try:
                ret = alg.policy.load_state_dict(ckpt["model_state_dict"], strict=False)
                if hasattr(ret, "missing_keys") and hasattr(ret, "unexpected_keys"):
                    m = list(ret.missing_keys); u = list(ret.unexpected_keys)
                    if m: print(f"[Phase2b-LOAD]   missing (first 3): {m[:3]}")
                    if u: print(f"[Phase2b-LOAD]   unexpected (first 3): {u[:3]}")
                # 验证 actor 权重确实被加载 (LoRA-style 0 init 的 [:, 99:] 应该非 0)
                with torch.no_grad():
                    aw = alg.policy.actor.actor_mlp[0].weight
                    new_cols_norm = aw[:, 99:].norm().item()
                    if new_cols_norm < 1e-6:
                        raise RuntimeError(
                            f"[Phase2b-LOAD] actor LoRA cols [:, 99:] norm={new_cols_norm:.2e} ≈ 0 "
                            f"→ checkpoint NOT loaded (likely random init). FATAL."
                        )
                    print(f"[Phase2b-LOAD] policy ✓ (actor LoRA cols [:, 99:] norm={new_cols_norm:.4f})")
            except Exception as ex:
                # 后备方案: 手动按 key 加载, 必须成功否则 raise
                print(f"[Phase2b-LOAD] strict=False load returned weird value ({ex})")
                print("[Phase2b-LOAD] falling back to manual key-by-key load...")
                policy_sd = alg.policy.state_dict()
                ckpt_sd = ckpt["model_state_dict"]
                copied = 0; skipped = 0
                with torch.no_grad():
                    for k, v in ckpt_sd.items():
                        if k in policy_sd and policy_sd[k].shape == v.shape:
                            policy_sd[k].copy_(v.to(policy_sd[k].device))
                            copied += 1
                        else:
                            skipped += 1
                # 验证 actor 被加载了
                aw = alg.policy.actor.actor_mlp[0].weight
                new_cols_norm = aw[:, 99:].norm().item()
                print(f"[Phase2b-LOAD] manual: copied {copied} keys, skipped {skipped}")
                print(f"[Phase2b-LOAD] policy ✓ (actor LoRA cols [:, 99:] norm={new_cols_norm:.4f})")
                if new_cols_norm < 1e-6:
                    raise RuntimeError(
                        "[Phase2b-LOAD] manual load also failed: actor LoRA cols are all 0. "
                        "Check that checkpoint contains 'actor.actor_mlp.0.weight' keys."
                    )
            if "optimizer_state_dict" in ckpt and load_optimizer:
                try:
                    alg.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    print("[Phase2b-LOAD] PPO optimizer ✓")
                except Exception as ex:
                    print(f"[Phase2b-LOAD] PPO optimizer skipped: {ex}")
        elif is_phase2a:
            print("[Phase2b-LOAD] Phase 2a checkpoint (warm-start path)")
            mksd = ckpt["model_state_dict"]
            phase2a_actor_keys  = {k: v for k, v in mksd.items() if k.startswith("actor.")}
            phase2a_critic_keys = {k: v for k, v in mksd.items() if k.startswith("critic.")}

            # 重建旧 actor Sequential 用于 warm-start copy
            old_actor_dict = {}
            for k, v in phase2a_actor_keys.items():
                old_actor_dict[k[len("actor."):]] = v
            indices = sorted({int(k.split(".")[0]) for k in old_actor_dict if k.split(".")[0].isdigit()})
            old_layers = []
            for idx in indices:
                w_key = f"{idx}.weight"; b_key = f"{idx}.bias"
                if w_key in old_actor_dict:
                    w = old_actor_dict[w_key]
                    out_d, in_d = w.shape
                    layer = nn.Linear(in_d, out_d).to(device)
                    layer.weight.data = w.to(device)
                    if b_key in old_actor_dict:
                        layer.bias.data = old_actor_dict[b_key].to(device)
                    old_layers.append(layer)
            old_seq_list = []
            for i, layer in enumerate(old_layers):
                old_seq_list.append(layer)
                if i < len(old_layers) - 1:
                    old_seq_list.append(nn.ELU())
            old_actor_seq = nn.Sequential(*old_seq_list).to(device)
            print(f"[Phase2b-LOAD]   reconstructed Phase 2a actor: {len(old_actor_seq)} layers, "
                  f"first Linear in={old_actor_seq[0].in_features}")

            _copy_actor_warm_start(
                alg.policy.actor.actor_mlp, old_actor_seq, old_in_dim=old_actor_seq[0].in_features,
            )

            # critic: 直接 load
            critic_state = {k[len("critic."):]: v.to(device) for k, v in phase2a_critic_keys.items()}
            try:
                ret = alg.policy.critic.load_state_dict(critic_state, strict=False)
                if hasattr(ret, "missing_keys"):
                    print(f"[Phase2b-LOAD]   critic loaded: missing={len(ret.missing_keys)} "
                          f"unexpected={len(ret.unexpected_keys)}")
                else:
                    print(f"[Phase2b-LOAD]   critic loaded (return type: {type(ret).__name__})")
            except Exception as ex:
                print(f"[Phase2b-LOAD]   critic load failed: {ex}")
            print("[Phase2b-LOAD]   skipping PPO optimizer (architecture changed)")
        else:
            print("[Phase2b-LOAD] unknown checkpoint type, trying standard load (strict=False)")
            try:
                if "model_state_dict" in ckpt:
                    alg.policy.load_state_dict(ckpt["model_state_dict"], strict=False)
            except Exception as ex:
                print(f"[Phase2b-LOAD] failed: {ex}")

        # VAE: 仅 Phase 2b checkpoint 加载 (Phase 2a 的 VAE 实际未训练成功因 history dim 不匹配,
        # 加载它会覆盖我们 stabilized 的随机初始化, 反而导致 first-iter loss 爆炸).
        if is_phase2a:
            print("[Phase2b-LOAD] vae_estimator: skipping (Phase 2a VAE was not trained, "
                  "use stabilized fresh init)")
            print("[Phase2b-LOAD] vae_optimizer: skipping (matches fresh VAE)")
        elif "vae_estimator_state_dict" in ckpt:
            try:
                state = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                         for k, v in ckpt["vae_estimator_state_dict"].items()}
                ret = alg.vae_estimator.load_state_dict(state, strict=False)
                if hasattr(ret, "missing_keys"):
                    m = list(ret.missing_keys); u = list(ret.unexpected_keys)
                    if m or u:
                        print(f"[Phase2b-LOAD] vae_estimator: missing={m[:3]} unexpected={u[:3]}")
                    else:
                        print("[Phase2b-LOAD] vae_estimator ✓")
                else:
                    print("[Phase2b-LOAD] vae_estimator ✓ (no key info)")
            except Exception as ex:
                print(f"[Phase2b-LOAD] vae_estimator load failed (using random init): {ex}")
        else:
            print("[Phase2b-LOAD] No vae_estimator in checkpoint (using random init)")

        if not is_phase2a and "vae_optimizer_state_dict" in ckpt and hasattr(alg, "vae_optimizer") \
                and alg.vae_optimizer is not None and load_optimizer:
            try:
                alg.vae_optimizer.load_state_dict(ckpt["vae_optimizer_state_dict"])
                print("[Phase2b-LOAD] vae_optimizer ✓")
            except Exception as ex:
                print(f"[Phase2b-LOAD] vae_optimizer skipped: {ex}")

    runner.save = patched_save
    runner.load = patched_load
    print("[Phase2b] runner.save / runner.load patched ✓")
