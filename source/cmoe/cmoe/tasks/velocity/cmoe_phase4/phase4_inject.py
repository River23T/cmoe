"""Phase 4 inject — VAE + AE + MoE actor/critic + SwAV contrastive learning.

============================================================================
Phase 4 = Phase 3 + MoE + SwAV (paper §III-D §III-E)
============================================================================

Architecture (paper Fig.3, Eq. 6, 7, 8):

  obs.policy (975 dim, per-term flatten, 8 terms × 5 history frames):
    [base_lin_vel(15), base_ang_vel(15), gravity(15), cmd(15),
     jpos(145), jvel(145), last_act(145), elevation(480)]

  obs.critic (195 dim, no history):
    [base_lin_vel(3), base_ang_vel(3), gravity(3), cmd(3),
     jpos(29), jvel(29), last_act(29), elevation(96)]

  Actor (MoE):
    input: o_t(99) + v_pred(3) + z_H(32) + z_E(32) = 166 dim
    5 expert MLPs: Linear(166,512) → ELU → Linear(512,256) → ELU → Linear(256,128) → ELU → Linear(128,29)
    Gating(z_E):  z_E(32) → Linear(32,64) → ELU → Linear(64,32) → ELU → Linear(32,5)
    output = Σ softmax(g_i) × μ_i   (Eq. 6)

  Critic (MoE):
    input: o_critic(99) + z_E(32) = 131 dim
    5 expert MLPs: Linear(131,512) → ... → Linear(128,1)
    Gating: SHARED with actor (paper requirement)
    output = Σ softmax(g_i) × v_i

  SwAV contrastive (paper §III-E):
    Project gate_logits & z_E → 64 dim → L2-normalize
    Compute soft prototype assignments (32 prototypes)
    Cross-predict via Sinkhorn-Knopp
    Loss: J_SwAV = -1/2 * (q_g log p_e + q_e log p_g)   (Eq. 8)

Warm-start from Phase 3:
  - 5 actor experts each = Phase 3 single actor (clone weights, no LoRA needed since
    Phase 3 actor input is exactly 166 dim already)
  - 5 critic experts each = Phase 3 single critic (clone weights, also 131 dim)
  - VAE: copied from Phase 3 (already trained)
  - AE: copied from Phase 3 (already trained)
  - Gating: random init (5 experts will start identical, slowly diverge)
  - SwAV: random init prototypes
============================================================================
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ============================================================
# 常量 (与 Phase 3 完全相同, 因为 obs structure 不变)
# ============================================================
POLICY_FRAME_DIM = 99
BASE_LIN_VEL_DIM = 3
OBS96_DIM = POLICY_FRAME_DIM - BASE_LIN_VEL_DIM   # 96
HISTORY_LEN = 5
ELEVATION_DIM = 96
VAE_INPUT_DIM = OBS96_DIM * HISTORY_LEN           # 480
ACTOR_OBS_DIM = POLICY_FRAME_DIM + 3 + 32 + 32    # 166
CRITIC_OBS_DIM = POLICY_FRAME_DIM + 32            # 131
EXPECTED_POLICY_OBS_DIM = (POLICY_FRAME_DIM + ELEVATION_DIM) * HISTORY_LEN  # 975
EXPECTED_CRITIC_OBS_DIM = POLICY_FRAME_DIM + ELEVATION_DIM                  # 195
NUM_ACTIONS = 29

OBS_TERM_DIMS = (3, 3, 3, 3, 29, 29, 29, 96)
def _compute_term_offsets():
    offsets = [0]
    for term_dim in OBS_TERM_DIMS[:-1]:
        offsets.append(offsets[-1] + term_dim * HISTORY_LEN)
    return offsets

OBS_TERM_OFFSETS = _compute_term_offsets()  # [0, 15, 30, 45, 60, 205, 350, 495]
assert sum(d * HISTORY_LEN for d in OBS_TERM_DIMS) == EXPECTED_POLICY_OBS_DIM


def _extract_frame_from_obs(x: torch.Tensor, frame_idx: int) -> torch.Tensor:
    """Extract single proprio frame (99 dim) — exclude elevation."""
    parts = []
    for term_id in range(7):
        term_dim = OBS_TERM_DIMS[term_id]
        start = OBS_TERM_OFFSETS[term_id] + frame_idx * term_dim
        parts.append(x[:, start : start + term_dim])
    return torch.cat(parts, dim=-1)


def _extract_elevation_latest(x: torch.Tensor) -> torch.Tensor:
    """Extract latest elevation frame (96 dim)."""
    elev_offset = OBS_TERM_OFFSETS[7]
    elev_dim = OBS_TERM_DIMS[7]
    start = elev_offset + (HISTORY_LEN - 1) * elev_dim
    return x[:, start : start + elev_dim]


def _extract_vae_history_target(x: torch.Tensor):
    """Extract VAE inputs (history 480, target 96)."""
    history_frames = []
    for i in range(HISTORY_LEN):
        f = _extract_frame_from_obs(x, i)
        history_frames.append(f[:, BASE_LIN_VEL_DIM:])
    vae_history = torch.cat(history_frames, dim=-1)
    vae_target = history_frames[-1]
    return vae_history, vae_target


# ============================================================
# Helpers
# ============================================================
def _build_mlp(in_dim: int, out_dim: int, hidden_dims: list, activation_class: type) -> nn.Sequential:
    layers = []
    cur = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(cur, h))
        layers.append(activation_class())
        cur = h
    layers.append(nn.Linear(cur, out_dim))
    return nn.Sequential(*layers)


# ============================================================
# Gating Network (paper §III-D, Fig.3)
# ============================================================
class GatingNetwork(nn.Module):
    """Gating: z_E → expert weights logits.
    
    Input: z_E (gate_input_dim, default 32)
    Output: raw logits (num_experts,) — softmax applied externally
    """
    def __init__(self, input_dim: int, num_experts: int, hidden_dims: list, activation_class: type):
        super().__init__()
        self.net = _build_mlp(input_dim, num_experts, hidden_dims, activation_class)
        self.num_experts = num_experts

    def forward(self, z_e: torch.Tensor) -> torch.Tensor:
        return self.net(z_e)


# ============================================================
# MoE Actor (paper §III-D Eq. 6)
# ============================================================
class MoEAugmentedActor(nn.Module):
    """MoE Actor with VAE + AE + 5 experts + shared gating.
    
    forward(x_975):
      o_t          = extract_frame(x, 4)        # 99 (proprio newest)
      vae_history  = extract_vae_history(x)     # 480
      e_t          = extract_elevation(x)       # 96
      
      with no_grad: z_H, v_pred = vae(vae_history)
      with no_grad: z_E         = ae.encode(e_t)
      
      input = cat([o_t, v_pred, z_H, z_E])      # 166
      
      Each expert_i forward(input) = action_i (29)
      Gating(z_E) = logits (5,) → softmax → weights
      action = Σ weights_i × action_i
    
    Note: log_std is shared across experts (single learnable param vector of 29 dim).
    """
    def __init__(
        self,
        vae_estimator,
        ae_estimator,
        experts: nn.ModuleList,        # 5 expert MLPs (each Sequential)
        gating: GatingNetwork,
        action_dim: int = NUM_ACTIONS,
        init_noise_std: float = 1.0,
    ):
        super().__init__()
        self.vae_estimator = vae_estimator
        self.ae_estimator = ae_estimator
        self.experts = experts
        self.gating = gating
        self.num_experts = len(experts)
        self.action_dim = action_dim

        # Cache for SwAV (last gate logits, shared with critic for shared gating)
        self._last_gate_logits = None
        self._last_gate_weights = None
        self._last_z_E = None

    @property
    def actor_mlp(self):
        """Compatibility shim: rsl_rl might query .actor_mlp[0] for first layer.
        We return a fake Sequential containing the first expert, ONLY for warm-start
        verification log compatibility. Don't use this for real forward."""
        return self.experts[0]

    def __getitem__(self, idx):
        """rsl_rl may probe actor[0] to inspect structure."""
        return self.experts[0][idx]

    def __len__(self):
        return len(self.experts[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Validate input shape
        if x.shape[-1] != EXPECTED_POLICY_OBS_DIM:
            raise RuntimeError(
                f"[MoEAugmentedActor] Expected x.shape[-1] = {EXPECTED_POLICY_OBS_DIM}, got {x.shape[-1]}"
            )
        N = x.shape[0]

        # Split obs
        o_t = _extract_frame_from_obs(x, HISTORY_LEN - 1)          # (N, 99)
        vae_history, _ = _extract_vae_history_target(x)             # (N, 480), (N, 96)
        e_t = _extract_elevation_latest(x)                          # (N, 96)

        # VAE encode (no_grad: VAE 单独训练)
        with torch.no_grad():
            z_H, v_pred = self.vae_estimator.get_latent_and_velocity(vae_history)
            z_H = z_H.detach()
            v_pred = v_pred.detach()
            z_E = self.ae_estimator.get_latent(e_t).detach()

        # Build expert input (166 dim)
        expert_input = torch.cat([o_t, v_pred, z_H, z_E], dim=-1)  # (N, 166)

        # Compute expert outputs (N, K, 29)
        expert_outputs = torch.stack(
            [expert(expert_input) for expert in self.experts], dim=1
        )

        # Gating
        gate_logits = self.gating(z_E)                              # (N, K)
        gate_weights = F.softmax(gate_logits, dim=-1)               # (N, K)

        # Cache for SwAV / critic (shared gating)
        self._last_gate_logits = gate_logits
        self._last_gate_weights = gate_weights
        self._last_z_E = z_E

        # Weighted sum (Eq. 6)
        action_mean = (gate_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)
        return action_mean


# ============================================================
# MoE Critic (paper §III-D)
# ============================================================
class MoEAugmentedCritic(nn.Module):
    """MoE Critic with shared gating from actor.
    
    forward(x_195):
      o_critic = x[:, :99]
      e_critic = x[:, 99:195]
      
      with no_grad: z_E = ae.encode(e_critic)
      input = cat([o_critic, z_E])              # 131
      
      Each expert_i forward(input) = value_i (1)
      gate_weights = SAME from actor (shared gating)
      value = Σ weights_i × value_i
    
    Important: when called during PPO update, actor's last_gate_weights might
    not be aligned. We just recompute via the SHARED gating module on the
    same z_E here.
    """
    def __init__(
        self,
        ae_estimator,
        experts: nn.ModuleList,
        shared_gating: GatingNetwork,    # 与 actor 共享同一个 GatingNetwork module
    ):
        super().__init__()
        self.ae_estimator = ae_estimator
        self.experts = experts
        self.shared_gating = shared_gating
        self.num_experts = len(experts)

    @property
    def critic_mlp(self):
        """Compatibility shim — returns first expert."""
        return self.experts[0]

    def __getitem__(self, idx):
        return self.experts[0][idx]

    def __len__(self):
        return len(self.experts[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != EXPECTED_CRITIC_OBS_DIM:
            raise RuntimeError(
                f"[MoEAugmentedCritic] Expected x.shape[-1] = {EXPECTED_CRITIC_OBS_DIM}, got {x.shape[-1]}"
            )
        N = x.shape[0]

        o_critic = x[:, :POLICY_FRAME_DIM]
        e_critic = x[:, POLICY_FRAME_DIM:POLICY_FRAME_DIM + ELEVATION_DIM]

        with torch.no_grad():
            z_E = self.ae_estimator.get_latent(e_critic).detach()

        # Build expert input (131 dim)
        expert_input = torch.cat([o_critic, z_E], dim=-1)

        # Expert values (N, K, 1)
        expert_values = torch.stack(
            [expert(expert_input) for expert in self.experts], dim=1
        )

        # Gating: use SHARED gating module on the same z_E (paper requirement)
        gate_logits = self.shared_gating(z_E)
        gate_weights = F.softmax(gate_logits, dim=-1)

        # Weighted sum
        value = (gate_weights.unsqueeze(-1) * expert_values).sum(dim=1)
        return value


# ============================================================
# SwAV Contrastive (paper §III-E Eq. 7-8)
# ============================================================
class SwAVContrastiveLoss(nn.Module):
    """SwAV-based contrastive loss between gate output and elevation latent.
    
    paper §III-E:
      Project gate_logits & z_E → projection_dim
      L2 normalize → dot product with prototypes
      Sinkhorn-Knopp → balanced cluster assignments q_g, q_e
      Cross-predict probabilities p_g, p_e (with temperature τ)
      L = -1/(2H) Σ (q_g log p_e + q_e log p_g)   (Eq. 8)
    """
    def __init__(
        self,
        gate_dim: int = 5,
        elevation_dim: int = 32,
        projection_dim: int = 64,
        num_prototypes: int = 32,
        temperature: float = 0.2,
        sinkhorn_iters: int = 3,
    ):
        super().__init__()
        self.temperature = temperature
        self.num_prototypes = num_prototypes
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_eps = 0.05

        self.gate_projector = nn.Sequential(
            nn.Linear(gate_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.elev_projector = nn.Sequential(
            nn.Linear(elevation_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )
        # Prototypes
        self.prototypes = nn.Linear(projection_dim, num_prototypes, bias=False)

    @torch.no_grad()
    def _sinkhorn_knopp(self, scores: torch.Tensor) -> torch.Tensor:
        """Compute balanced cluster assignment via Sinkhorn-Knopp."""
        scaled = scores / self.sinkhorn_eps
        scaled = scaled - scaled.max(dim=-1, keepdim=True).values
        Q = torch.exp(scaled).T  # (K, B)

        sum_Q = Q.sum()
        if sum_Q == 0 or torch.isnan(sum_Q) or torch.isinf(sum_Q):
            B = Q.shape[1]
            return torch.ones(B, Q.shape[0], device=Q.device) / Q.shape[0]
        Q /= sum_Q

        K, B = Q.shape
        for _ in range(self.sinkhorn_iters):
            row_sum = Q.sum(dim=1, keepdim=True).clamp(min=1e-8)
            Q /= row_sum
            Q /= K
            col_sum = Q.sum(dim=0, keepdim=True).clamp(min=1e-8)
            Q /= col_sum
            Q /= B

        Q = Q.T  # (B, K)
        Q *= B
        Q = torch.nan_to_num(Q, nan=1.0 / K, posinf=1.0, neginf=0.0)
        return Q

    def forward(self, gate_logits: torch.Tensor, z_E: torch.Tensor) -> torch.Tensor:
        # Project to shared dim
        g_z = self.gate_projector(gate_logits)
        e_z = self.elev_projector(z_E)

        # L2 normalize
        g_z = F.normalize(g_z, dim=-1, eps=1e-8)
        e_z = F.normalize(e_z, dim=-1, eps=1e-8)

        # L2-normalize prototypes (paper requirement)
        with torch.no_grad():
            w = self.prototypes.weight.data.clone()
            w = F.normalize(w, dim=1, eps=1e-8)
            self.prototypes.weight.copy_(w)

        # Dot products with prototypes
        raw_scores_g = self.prototypes(g_z)
        raw_scores_e = self.prototypes(e_z)

        # Predicted probabilities (Eq. 7)
        scores_g = raw_scores_g / self.temperature
        scores_e = raw_scores_e / self.temperature
        p_g = F.softmax(scores_g, dim=-1)
        p_e = F.softmax(scores_e, dim=-1)

        # Cluster assignments (Sinkhorn balanced)
        q_g = self._sinkhorn_knopp(raw_scores_g.detach())
        q_e = self._sinkhorn_knopp(raw_scores_e.detach())

        # Cross-entropy (Eq. 8)
        loss = -0.5 * (
            (q_g * torch.log(p_e + 1e-8)).sum(dim=-1).mean()
            + (q_e * torch.log(p_g + 1e-8)).sum(dim=-1).mean()
        )

        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=gate_logits.device, requires_grad=True)

        return loss


# ============================================================
# Stabilization helpers
# ============================================================
def _stabilize_vae(vae_estimator):
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
            last = vae_estimator.decoder[-1]
            if isinstance(last, nn.Linear):
                last.weight.data *= 0.01
                if last.bias is not None: last.bias.data.zero_()


def _stabilize_ae(ae_estimator):
    with torch.no_grad():
        if hasattr(ae_estimator, "encoder") and len(ae_estimator.encoder) > 0:
            last = ae_estimator.encoder[-1]
            if isinstance(last, nn.Linear):
                last.weight.data *= 0.01
                if last.bias is not None: last.bias.data.zero_()
        if hasattr(ae_estimator, "decoder") and len(ae_estimator.decoder) > 0:
            last = ae_estimator.decoder[-1]
            if isinstance(last, nn.Linear):
                last.weight.data *= 0.01
                if last.bias is not None: last.bias.data.zero_()


# ============================================================
# Main inject function
# ============================================================
def inject_moe_swav(alg, cmoe_cfg: dict, device: str = "cpu"):
    """Phase 4 inject — replace actor/critic with MoE versions, add SwAV."""
    est_cfg = cmoe_cfg.get("estimator", None)
    moe_cfg = cmoe_cfg.get("moe", None)
    contrastive_cfg = cmoe_cfg.get("contrastive", None)
    if est_cfg is None or "vae" not in est_cfg or "ae" not in est_cfg:
        raise RuntimeError("[Phase4] cmoe_cfg.estimator.vae and .ae required")
    if moe_cfg is None:
        raise RuntimeError("[Phase4] cmoe_cfg.moe required")
    if contrastive_cfg is None:
        raise RuntimeError("[Phase4] cmoe_cfg.contrastive required")

    vae_cfg = est_cfg["vae"]
    ae_cfg = est_cfg["ae"]

    # ---- 1. VAE estimator ----
    from cmoe.custom_classes.modules import VAEEstimator
    vae_estimator = VAEEstimator(**vae_cfg).to(device)
    print(f"[Phase4] VAE Estimator: latent={vae_cfg['latent_dim']}")
    _stabilize_vae(vae_estimator)

    # ---- 2. AE estimator ----
    from cmoe.custom_classes.modules.ae_estimator import AEEstimator
    ae_estimator = AEEstimator(**ae_cfg).to(device)
    print(f"[Phase4] AE Estimator: latent={ae_cfg['latent_dim']}")
    _stabilize_ae(ae_estimator)

    alg.vae_estimator = vae_estimator
    alg.ae_estimator = ae_estimator
    alg.vae_optimizer = optim.Adam(vae_estimator.parameters(), lr=est_cfg["learning_rate"])
    alg.ae_optimizer = optim.Adam(ae_estimator.parameters(), lr=est_cfg["learning_rate"])
    print(f"[Phase4] VAE/AE optimizers: Adam lr={est_cfg['learning_rate']}")

    # ---- 3. Build MoE actor ----
    num_experts = moe_cfg["num_experts"]
    expert_hidden_dims = moe_cfg["expert_hidden_dims"]
    actor_obs_dim = moe_cfg["actor_obs_dim"]
    critic_obs_dim = moe_cfg["critic_obs_dim"]
    action_dim = moe_cfg["action_dim"]
    gate_input_dim = moe_cfg["gate_input_dim"]
    gate_hidden_dims = moe_cfg["gate_hidden_dims"]

    # Activation: read from existing alg.policy.actor (rsl_rl created)
    activation_class = nn.ELU
    for layer in alg.policy.actor.modules():
        if not isinstance(layer, nn.Linear) and not isinstance(layer, nn.Sequential) \
                and layer is not alg.policy.actor:
            activation_class = type(layer)
            break
    print(f"[Phase4] activation: {activation_class.__name__}")

    # Build 5 actor experts
    actor_experts = nn.ModuleList([
        _build_mlp(actor_obs_dim, action_dim, expert_hidden_dims, activation_class)
        for _ in range(num_experts)
    ]).to(device)

    # Build gating network
    actor_gating = GatingNetwork(
        gate_input_dim, num_experts, gate_hidden_dims, activation_class
    ).to(device)

    # Build 5 critic experts
    critic_experts = nn.ModuleList([
        _build_mlp(critic_obs_dim, 1, expert_hidden_dims, activation_class)
        for _ in range(num_experts)
    ]).to(device)

    print(f"[Phase4] MoE: {num_experts} experts × Linear({actor_obs_dim},{expert_hidden_dims[0]})")
    print(f"[Phase4] Gating: Linear({gate_input_dim}, {gate_hidden_dims[0]}) → ... → Linear({gate_hidden_dims[-1]}, {num_experts})")

    # ---- 4. Wrap into MoEAugmentedActor / Critic ----
    moe_actor = MoEAugmentedActor(
        vae_estimator=vae_estimator,
        ae_estimator=ae_estimator,
        experts=actor_experts,
        gating=actor_gating,
        action_dim=action_dim,
    ).to(device)
    moe_critic = MoEAugmentedCritic(
        ae_estimator=ae_estimator,
        experts=critic_experts,
        shared_gating=actor_gating,    # share gating!
    ).to(device)

    alg.policy.actor = moe_actor
    alg.policy.critic = moe_critic
    print("[Phase4] alg.policy.actor replaced with MoEAugmentedActor")
    print("[Phase4] alg.policy.critic replaced with MoEAugmentedCritic (shared gating)")

    # ---- 5. SwAV contrastive ----
    swav = SwAVContrastiveLoss(
        gate_dim=num_experts,
        elevation_dim=ae_cfg["latent_dim"],
        projection_dim=contrastive_cfg["projection_dim"],
        num_prototypes=contrastive_cfg["num_prototypes"],
        temperature=contrastive_cfg["temperature"],
        sinkhorn_iters=contrastive_cfg["sinkhorn_iters"],
    ).to(device)
    alg.swav_loss = swav
    alg.swav_loss_weight = contrastive_cfg.get("loss_weight", 0.1)
    alg.swav_optimizer = optim.Adam(
        swav.parameters(),
        lr=contrastive_cfg.get("learning_rate", 1e-3),
    )
    print(f"[Phase4] SwAV: K={contrastive_cfg['num_prototypes']}, "
          f"τ={contrastive_cfg['temperature']}, "
          f"projection_dim={contrastive_cfg['projection_dim']}")

    # ---- 6. Rebuild PPO optimizer (exclude VAE/AE/SwAV) ----
    # Get all non-VAE non-AE non-SwAV params from policy
    excluded_params = set()
    for p in vae_estimator.parameters():
        excluded_params.add(id(p))
    for p in ae_estimator.parameters():
        excluded_params.add(id(p))
    for p in swav.parameters():
        excluded_params.add(id(p))
    ppo_params = [p for p in alg.policy.parameters() if id(p) not in excluded_params]
    total_params = sum(p.numel() for p in alg.policy.parameters())
    ppo_param_count = sum(p.numel() for p in ppo_params)
    alg.optimizer = optim.Adam(ppo_params, lr=alg.learning_rate)
    print(f"[Phase4] PPO optimizer rebuilt (lr={alg.learning_rate}), "
          f"params {ppo_param_count}/{total_params} (excluded VAE+AE+SwAV)")

    # ---- 7. Wrap alg.act for caching VAE/AE/SwAV training data ----
    _orig_act = alg.act

    def _phase4_wrapped_act(*args, **kwargs):
        obs = args[0] if len(args) > 0 else kwargs.get("obs", None)
        try:
            if obs is not None and hasattr(obs, "get"):
                po = obs.get("policy", None)
                co = obs.get("critic", None)

                if not hasattr(alg, "_phase4_first_call"):
                    alg._phase4_first_call = True
                    if po is not None:
                        print(f"[Phase4-DEBUG] obs['policy'].shape = {tuple(po.shape)} "
                              f"(expect (N, {EXPECTED_POLICY_OBS_DIM}))")
                    if co is not None:
                        print(f"[Phase4-DEBUG] obs['critic'].shape = {tuple(co.shape)} "
                              f"(expect (N, {EXPECTED_CRITIC_OBS_DIM}))")

                if (po is not None and co is not None
                        and po.shape[-1] == EXPECTED_POLICY_OBS_DIM
                        and co.shape[-1] == EXPECTED_CRITIC_OBS_DIM):
                    vae_history, vae_target = _extract_vae_history_target(po)
                    vel_gt = co[:, :3]
                    elevation_gt = co[:, 99:99 + ELEVATION_DIM]

                    alg._vae_buf["vae_history"].append(vae_history.data.clone())
                    alg._vae_buf["vae_target"].append(vae_target.data.clone())
                    alg._vae_buf["vel_gt"].append(vel_gt.data.clone())
                    alg._ae_buf["elevation_gt"].append(elevation_gt.data.clone())
        except Exception as ex:
            if alg._vae_iter_count <= 3:
                print(f"[Phase4] act hook exception: {ex}")
                import traceback; traceback.print_exc()

        return _orig_act(*args, **kwargs)

    alg._vae_buf = {"vae_history": [], "vae_target": [], "vel_gt": []}
    alg._ae_buf = {"elevation_gt": []}
    alg._vae_iter_count = 0
    alg.act = _phase4_wrapped_act
    print("[Phase4] alg.act wrapped: cache vae_history+target+vel_gt+elevation_gt")

    # ---- 8. Wrap alg.update for VAE + AE + SwAV training ----
    _orig_update = alg.update

    def _phase4_wrapped_update(*args, **kwargs):
        result = _orig_update(*args, **kwargs)

        # Train VAE
        try:
            if (len(alg._vae_buf["vae_history"]) > 0
                    and len(alg._vae_buf["vae_target"]) > 0
                    and len(alg._vae_buf["vel_gt"]) > 0):
                vh_all = torch.cat(alg._vae_buf["vae_history"], dim=0)
                vt_all = torch.cat(alg._vae_buf["vae_target"], dim=0)
                vel_all = torch.cat(alg._vae_buf["vel_gt"], dim=0)
                total = vh_all.shape[0]
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
                            if len(idx) == 0: continue
                            alg.vae_optimizer.zero_grad()
                            loss, _ = alg.vae_estimator.compute_loss(
                                vh_all[idx], vt_all[idx], vel_all[idx]
                            )
                            if torch.isnan(loss) or torch.isinf(loss): continue
                            loss.backward()
                            nn.utils.clip_grad_norm_(
                                alg.vae_estimator.parameters(),
                                getattr(alg, "max_grad_norm", 1.0),
                            )
                            alg.vae_optimizer.step()
                    alg.vae_estimator.eval()
        except Exception as ex:
            print(f"[Phase4] VAE train exception: {ex}")
            import traceback; traceback.print_exc()

        # Train AE
        try:
            if len(alg._ae_buf["elevation_gt"]) > 0:
                e_all = torch.cat(alg._ae_buf["elevation_gt"], dim=0)
                total = e_all.shape[0]
                ae_loss_avg = 0.0
                n_batches = 0
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
                            if len(idx) == 0: continue
                            alg.ae_optimizer.zero_grad()
                            loss, _ = alg.ae_estimator.compute_loss(e_all[idx])
                            if torch.isnan(loss) or torch.isinf(loss): continue
                            loss.backward()
                            nn.utils.clip_grad_norm_(
                                alg.ae_estimator.parameters(),
                                getattr(alg, "max_grad_norm", 1.0),
                            )
                            alg.ae_optimizer.step()
                            ae_loss_avg += loss.item()
                            n_batches += 1
                    alg.ae_estimator.eval()
        except Exception as ex:
            print(f"[Phase4] AE train exception: {ex}")
            import traceback; traceback.print_exc()

        # Train SwAV (using cached elevation as proxy for z_E and gate logits)
        # Note: gate_logits are recomputed via current gating module on z_E
        try:
            if len(alg._ae_buf["elevation_gt"]) > 0:
                e_all = torch.cat(alg._ae_buf["elevation_gt"], dim=0) \
                    if len(alg._ae_buf["elevation_gt"]) > 0 else None
                # We re-use freshly-computed z_E and gate_logits.
                # IMPORTANT: gate logits depend on z_E, so we recompute everything fresh.
                if e_all is not None and e_all.shape[0] > 0:
                    total = e_all.shape[0]
                    n_epochs = getattr(alg, "num_learning_epochs", 5)
                    n_mb = getattr(alg, "num_mini_batches", 4)
                    mbs = max(total // n_mb, 256)
                    swav_loss_avg = 0.0
                    n_swav_batches = 0
                    alg.swav_loss.train()
                    for ep in range(n_epochs):
                        perm = torch.randperm(total, device=e_all.device)
                        for s in range(0, total, mbs):
                            e = min(s + mbs, total)
                            idx = perm[s:e]
                            if len(idx) == 0: continue
                            mb_e = e_all[idx]
                            # Recompute z_E with current AE
                            with torch.no_grad():
                                mb_zE = alg.ae_estimator.get_latent(mb_e).detach()
                            # Compute current gate logits
                            with torch.no_grad():
                                mb_gate = alg.policy.actor.gating(mb_zE).detach()

                            alg.swav_optimizer.zero_grad()
                            loss = alg.swav_loss(mb_gate, mb_zE)
                            if torch.isnan(loss) or torch.isinf(loss): continue
                            loss.backward()
                            nn.utils.clip_grad_norm_(
                                alg.swav_loss.parameters(),
                                getattr(alg, "max_grad_norm", 1.0),
                            )
                            alg.swav_optimizer.step()
                            swav_loss_avg += loss.item()
                            n_swav_batches += 1
                    alg.swav_loss.eval()

                    if n_swav_batches > 0 and alg._vae_iter_count % 50 == 0:
                        print(f"[Phase4] iter={alg._vae_iter_count}  "
                              f"swav_loss={swav_loss_avg / max(n_swav_batches, 1):.4f}")
        except Exception as ex:
            print(f"[Phase4] SwAV train exception: {ex}")
            import traceback; traceback.print_exc()

        # Reset buffers
        alg._vae_buf = {"vae_history": [], "vae_target": [], "vel_gt": []}
        alg._ae_buf = {"elevation_gt": []}
        alg._vae_iter_count += 1

        return result

    alg.update = _phase4_wrapped_update
    print("[Phase4] alg.update wrapped: PPO + VAE + AE + SwAV training")
    print("[Phase4] MoE+SwAV+VAE+AE injection complete")


# ============================================================
# Runner save/load patches
# ============================================================
def inject_phase4_runner_patches(runner):
    """Patch save/load for VAE+AE+MoE experts+SwAV."""
    _orig_save = runner.save
    _orig_load = runner.load

    def patched_save(path, infos=None):
        _orig_save(path, infos)
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
            if hasattr(alg, "swav_loss"):
                ckpt["swav_state_dict"] = alg.swav_loss.state_dict()
            if hasattr(alg, "swav_optimizer"):
                ckpt["swav_optimizer_state_dict"] = alg.swav_optimizer.state_dict()
            torch.save(ckpt, path)
        except Exception as ex:
            print(f"[Phase4-SAVE] failed to append state: {ex}")

    def _copy_phase3_to_moe_experts(phase3_actor_state, moe_experts: nn.ModuleList,
                                    phase3_actor_in_dim: int):
        """Copy Phase 3 single actor weights to ALL 5 MoE experts.
        
        Phase 3 actor MLP was a Sequential[Linear(166,512), ELU, Linear(512,256), ...].
        Its keys in checkpoint look like: 'actor.actor_mlp.0.weight', 'actor.actor_mlp.0.bias', etc.
        
        We copy these to each expert in moe_experts (which have same structure).
        """
        # Extract phase3 keys (filter out actor.vae_estimator.* and actor.ae_estimator.*)
        phase3_keys = {}
        for k, v in phase3_actor_state.items():
            # Phase 3 actor structure: actor.actor_mlp.X.weight/bias
            if k.startswith("actor.actor_mlp."):
                # Strip "actor.actor_mlp." prefix
                new_k = k[len("actor.actor_mlp."):]
                phase3_keys[new_k] = v
        if len(phase3_keys) == 0:
            print("[Phase4-LOAD] WARNING: no actor.actor_mlp.* keys in Phase 3 checkpoint")
            return False

        copied_count = 0
        for expert_idx, expert in enumerate(moe_experts):
            expert_state = expert.state_dict()
            tmp = {}
            for k, v in phase3_keys.items():
                if k in expert_state and expert_state[k].shape == v.shape:
                    tmp[k] = v
            ret = expert.load_state_dict(tmp, strict=False)
            copied_count += len(tmp)
        print(f"[Phase4-LOAD] copied Phase 3 actor weights to all {len(moe_experts)} experts "
              f"(total {copied_count} tensors)")
        return True

    def _copy_phase3_critic_to_moe_critics(phase3_critic_state, moe_experts: nn.ModuleList):
        """Copy Phase 3 single critic weights to ALL 5 MoE critic experts."""
        phase3_keys = {}
        for k, v in phase3_critic_state.items():
            if k.startswith("critic.critic_mlp."):
                new_k = k[len("critic.critic_mlp."):]
                phase3_keys[new_k] = v
        if len(phase3_keys) == 0:
            print("[Phase4-LOAD] WARNING: no critic.critic_mlp.* keys in Phase 3 checkpoint")
            return False

        copied_count = 0
        for expert_idx, expert in enumerate(moe_experts):
            expert_state = expert.state_dict()
            tmp = {}
            for k, v in phase3_keys.items():
                if k in expert_state and expert_state[k].shape == v.shape:
                    tmp[k] = v
            ret = expert.load_state_dict(tmp, strict=False)
            copied_count += len(tmp)
        print(f"[Phase4-LOAD] copied Phase 3 critic weights to all {len(moe_experts)} experts "
              f"(total {copied_count} tensors)")
        return True

    def patched_load(path, load_optimizer=True):
        ckpt = torch.load(path, weights_only=False, map_location="cpu")
        alg = runner.alg
        device = next(alg.policy.parameters()).device

        is_phase4 = "swav_state_dict" in ckpt
        is_phase3 = "ae_estimator_state_dict" in ckpt and not is_phase4

        if is_phase4:
            # Resume Phase 4: load policy directly (architecture matches)
            print("[Phase4-LOAD] Phase 4 checkpoint (resume)")
            try:
                ret = alg.policy.load_state_dict(ckpt["model_state_dict"], strict=False)
                if hasattr(ret, "missing_keys"):
                    m = list(ret.missing_keys); u = list(ret.unexpected_keys)
                    if m: print(f"[Phase4-LOAD]   policy missing (first 3): {m[:3]}")
                    if u: print(f"[Phase4-LOAD]   policy unexpected (first 3): {u[:3]}")
                # Verify experts loaded
                with torch.no_grad():
                    expert0_first = alg.policy.actor.experts[0][0].weight
                    print(f"[Phase4-LOAD] policy ✓ (actor expert[0] first layer norm={expert0_first.norm().item():.4f})")
                    crit0_first = alg.policy.critic.experts[0][0].weight
                    print(f"[Phase4-LOAD] critic ✓ (critic expert[0] first layer norm={crit0_first.norm().item():.4f})")
                    gate_w = alg.policy.actor.gating.net[0].weight
                    print(f"[Phase4-LOAD] gating ✓ (first layer norm={gate_w.norm().item():.4f})")
            except Exception as ex:
                print(f"[Phase4-LOAD] strict=False load failed: {ex}")

            if "optimizer_state_dict" in ckpt and load_optimizer:
                try:
                    alg.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    print("[Phase4-LOAD] PPO optimizer ✓")
                except Exception as ex:
                    print(f"[Phase4-LOAD] PPO optimizer skipped: {ex}")

            # Load Phase 4 components
            for key, attr, label in [
                ("vae_estimator_state_dict", "vae_estimator", "vae_estimator"),
                ("vae_optimizer_state_dict", "vae_optimizer", "vae_optimizer"),
                ("ae_estimator_state_dict",  "ae_estimator",  "ae_estimator"),
                ("ae_optimizer_state_dict",  "ae_optimizer",  "ae_optimizer"),
                ("swav_state_dict",          "swav_loss",     "swav"),
                ("swav_optimizer_state_dict","swav_optimizer","swav_optimizer"),
            ]:
                if key in ckpt and hasattr(alg, attr):
                    try:
                        state = ckpt[key]
                        if isinstance(state, dict):
                            state = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                                     for k, v in state.items()}
                        getattr(alg, attr).load_state_dict(state, strict=False)
                        print(f"[Phase4-LOAD] {label} ✓")
                    except Exception as ex:
                        print(f"[Phase4-LOAD] {label} skipped: {ex}")

        elif is_phase3:
            # Warm-start from Phase 3
            print("[Phase4-LOAD] Phase 3 checkpoint (warm-start path)")
            try:
                # Copy Phase 3 actor (single MLP) → all 5 actor experts
                _copy_phase3_to_moe_experts(
                    ckpt["model_state_dict"],
                    alg.policy.actor.experts,
                    phase3_actor_in_dim=ACTOR_OBS_DIM,   # 166
                )
                # Copy Phase 3 critic → all 5 critic experts
                _copy_phase3_critic_to_moe_critics(
                    ckpt["model_state_dict"],
                    alg.policy.critic.experts,
                )
                print(f"[Phase4-LOAD] gating: random init (Phase 3 had no gating)")
            except Exception as ex:
                print(f"[Phase4-LOAD] Phase 3 warm-start failed: {ex}")
                import traceback; traceback.print_exc()
            print("[Phase4-LOAD] skipping PPO optimizer (architecture changed)")

            # Load VAE / AE from Phase 3
            for key, attr, label in [
                ("vae_estimator_state_dict", "vae_estimator", "vae_estimator"),
                ("ae_estimator_state_dict",  "ae_estimator",  "ae_estimator"),
            ]:
                if key in ckpt and hasattr(alg, attr):
                    try:
                        state = ckpt[key]
                        state = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                                 for k, v in state.items()}
                        getattr(alg, attr).load_state_dict(state, strict=False)
                        print(f"[Phase4-LOAD] {label} ✓ (warm-start from Phase 3)")
                    except Exception as ex:
                        print(f"[Phase4-LOAD] {label} skipped: {ex}")
            print("[Phase4-LOAD] swav_loss: random init (Phase 3 had no SwAV)")

        else:
            print(f"[Phase4-LOAD] unknown checkpoint type")

        if "iter" in ckpt:
            runner.current_learning_iteration = ckpt["iter"]

    runner.save = patched_save
    runner.load = patched_load
    print("[Phase4] runner.save / runner.load patched ✓")
