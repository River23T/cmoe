"""CMoE Post-construction Injection Module.

Architecture (Paper Fig. 3):
  1. VAE encoder: observation history -> z_H_t + velocity estimate
  2. AE encoder: elevation map -> z_E_t
  3. MoE Actor: N expert MLPs + gating(z_E) -> weighted action (Eq. 6)
  4. MoE Critic: N expert MLPs + shared gating -> weighted value
  5. SwAV contrastive loss: gate activations <-> elevation encoding (Eq. 7-8)

KEY: Uses a custom rollout buffer for PPO update because rsl_rl 3.3.0's
mini_batch_generator returns tuples (not named batch objects).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Any, Optional


def inject_cmoe(alg, cmoe_cfg: dict, device: str = "cpu"):
    est_cfg = cmoe_cfg.get("estimator", None)
    moe_cfg = cmoe_cfg.get("moe", None)
    contrastive_cfg = cmoe_cfg.get("contrastive", None)
    if est_cfg is not None:
        _inject_estimators(alg, est_cfg, device)
    if moe_cfg is not None:
        _inject_moe(alg, moe_cfg, device)
    if contrastive_cfg is not None and moe_cfg is not None:
        _inject_contrastive(alg, contrastive_cfg, device)
    _patch_mode_and_save(alg)
    if moe_cfg is not None:
        _patch_forward_pass(alg)
    if hasattr(alg, "moe_actor") and alg.moe_actor is not None and hasattr(alg, "policy"):
        policy = alg.policy
        moe_actor = alg.moe_actor
        policy.__class__.action_std = property(lambda self, _ma=moe_actor: _ma.std.detach())
        print("[CMoE] Patched alg.policy.action_std -> moe_actor.std")
    print("[CMoE] All components injected successfully")


def _inject_estimators(alg, est_cfg, device):
    from cmoe.custom_classes.modules import VAEEstimator, AEEstimator
    vae_cfg = est_cfg.get("vae", None)
    ae_cfg = est_cfg.get("ae", None)
    params = []
    if vae_cfg:
        alg.vae_estimator = VAEEstimator(**vae_cfg).to(device)
        params.extend(alg.vae_estimator.parameters())
        print(f"[CMoE] VAE Estimator: latent={vae_cfg['latent_dim']}")
    else:
        alg.vae_estimator = None
    if ae_cfg:
        alg.ae_estimator = AEEstimator(**ae_cfg).to(device)
        params.extend(alg.ae_estimator.parameters())
        print(f"[CMoE] AE Estimator: latent={ae_cfg['latent_dim']}")
    else:
        alg.ae_estimator = None
    if params:
        lr = est_cfg.get("learning_rate", 1e-3)
        alg.estimator_optimizer = optim.Adam(params, lr=lr)
        print(f"[CMoE] Estimator optimizer: lr={lr}")
    else:
        alg.estimator_optimizer = None


def _find_actor_critic(alg):
    if hasattr(alg, "actor") and hasattr(alg, "critic"):
        a, c = getattr(alg, "actor"), getattr(alg, "critic")
        if isinstance(a, nn.Module) and isinstance(c, nn.Module):
            print("[CMoE] Found separate alg.actor and alg.critic")
            return a, c, False
    if hasattr(alg, "policy"):
        policy = getattr(alg, "policy")
        if isinstance(policy, nn.Module):
            print(f"[CMoE] Found alg.policy = {type(policy).__name__}")
            asub = csub = None
            for n in ["actor", "_actor"]:
                if hasattr(policy, n):
                    x = getattr(policy, n)
                    if isinstance(x, nn.Module): asub = x; print(f"[CMoE] policy.{n} found"); break
            for n in ["critic", "_critic"]:
                if hasattr(policy, n):
                    x = getattr(policy, n)
                    if isinstance(x, nn.Module): csub = x; print(f"[CMoE] policy.{n} found"); break
            if asub and csub: return asub, csub, False
            return policy, policy, True
    if hasattr(alg, "actor_critic"):
        ac = getattr(alg, "actor_critic")
        if isinstance(ac, nn.Module):
            if hasattr(ac, "actor") and hasattr(ac, "critic"): return ac.actor, ac.critic, False
            return ac, ac, True
    raise AttributeError("Cannot find actor/critic on PPO object.")


def _extract_mlp_dims(model, role="actor"):
    input_dim = output_dim = None; hidden_dims = []
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if input_dim is None: input_dim = module.in_features
            output_dim = module.out_features; hidden_dims.append(module.out_features)
    if hidden_dims: hidden_dims = hidden_dims[:-1]
    return input_dim, output_dim, hidden_dims


def _extract_dims_from_combined_model(model):
    actor_info = critic_info = None; linear_groups = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parent = name.rsplit('.', 1)[0] if '.' in name else ''
            linear_groups.setdefault(parent, []).append(module)
    for pn, linears in linear_groups.items():
        if len(linears) < 2: continue
        fi, lo, hd = linears[0].in_features, linears[-1].out_features, [l.out_features for l in linears[:-1]]
        if lo > 1 and not actor_info:
            actor_info = (fi, lo, hd); print(f"[CMoE] Detected actor MLP: {fi} -> {hd} -> {lo}")
        elif lo == 1 and not critic_info:
            critic_info = (fi, lo, hd); print(f"[CMoE] Detected critic MLP: {fi} -> {hd} -> {lo}")
    return actor_info, critic_info


def _inject_moe(alg, moe_cfg, device):
    """
    R15 PAPER-FAITHFUL FIX
    ======================
    Previously the actor received only the raw policy history (480-dim) and the
    VAE outputs (v_pred, z_H) were COMPLETELY DISCARDED. This left the actor
    velocity-blind: it received only the velocity *command* but no estimate of
    its own velocity, breaking velocity tracking.

    Per paper §III-D, every expert MUST receive
        [v^p_t (3), z^H_t (vae_latent), z^E_t (ae_latent),
         o^c_t (current obs only, no history), e_t (elevation map)]
    so the actor obs_dim is recomputed here from cfg, NOT from the rsl_rl-built
    actor MLP (which has the wrong shape because rsl_rl warns and ignores the
    obs_groups dict — see eval log).

    Concretely:
        single_obs_dim = 96  (paper Eq. 2: ω+g+c+θ+θ̇+a_{t-1})
        vae_latent_dim = 32
        vae_velocity_dim = 3
        ae_latent_dim = 32
        elevation_dim = 96
        actor obs_dim = 3 + 32 + 32 + 96 + 96 = 259
    """
    from cmoe.custom_classes.models.cmoe_moe_model import MoEActorModel, MoECriticModel
    ne = moe_cfg.get("num_experts", 5)
    gid = moe_cfg.get("gate_input_dim", 32)
    ghd = moe_cfg.get("gate_hidden_dims", [64, 32])
    ins = moe_cfg.get("init_noise_std", 1.0)

    # ---- R15: paper-faithful actor obs_dim ----
    single_obs_dim = moe_cfg.get("single_obs_dim", 96)
    vae_latent_dim = 32
    vae_velocity_dim = 3
    ae_latent_dim = moe_cfg.get("ae_latent_dim", 32)
    elevation_dim = moe_cfg.get("elevation_dim", 96)
    aid = vae_velocity_dim + vae_latent_dim + ae_latent_dim + single_obs_dim + elevation_dim
    # = 3 + 32 + 32 + 96 + 96 = 259
    alg._cmoe_systematic_dims = {
        "v_pred": vae_velocity_dim,
        "z_H":    vae_latent_dim,
        "z_E":    ae_latent_dim,
        "o_c":    single_obs_dim,
        "e_t":    elevation_dim,
        "total":  aid,
    }

    actor_mlp, critic_mlp, is_combined = _find_actor_critic(alg)
    if is_combined:
        ai, ci = _extract_dims_from_combined_model(actor_mlp)
        if ai: _, aod, ahd = ai
        else: _, aod, ahd = _extract_mlp_dims(actor_mlp)
        if ci: cid, _, chd = ci
        else: cid, _, chd = _extract_mlp_dims(critic_mlp)
    else:
        _, aod, ahd = _extract_mlp_dims(actor_mlp)
        cid, _, chd = _extract_mlp_dims(critic_mlp)

    # R15: critic also paper-faithful — uses [privileged_obs (99) + e_t (96)]
    # The original rsl_rl-built critic has cid=195 already (99+96), so reuse cid.
    print(f"[CMoE-R15] Actor dims: input={aid} (paper-faithful: v(3)+z_H(32)+z_E(32)+o_c(96)+e_t(96)), output={aod}, hidden={ahd}")
    print(f"[CMoE-R15] Critic dims: input={cid}, hidden={chd}")
    moe_actor = MoEActorModel(obs_dim=aid, action_dim=aod, num_experts=ne,
        expert_hidden_dims=ahd or [512,256,128], gate_input_dim=gid,
        gate_hidden_dims=ghd, init_noise_std=ins).to(device)
    moe_critic = MoECriticModel(obs_dim=cid, num_experts=ne,
        expert_hidden_dims=chd or [512,256,128], shared_gating=moe_actor.gating).to(device)
    alg.moe_actor = moe_actor; alg.moe_critic = moe_critic; alg.moe_enabled = True
    alg._original_actor = actor_mlp; alg._original_critic = critic_mlp
    mp = list(moe_actor.parameters()) + list(moe_critic.parameters())
    ep = list(actor_mlp.parameters()) + (list(critic_mlp.parameters()) if not is_combined else [])
    seen = set(); up = []
    for p in ep + mp:
        if id(p) not in seen: seen.add(id(p)); up.append(p)
    alg.optimizer = optim.Adam(up, lr=alg.learning_rate)
    print(f"[CMoE-R15] MoE: {ne} experts, {sum(p.numel() for p in mp):,} params")


def _inject_contrastive(alg, cfg, device):
    from cmoe.custom_classes.models.cmoe_contrastive import SwAVContrastiveLoss
    ne = alg.moe_actor.num_experts if hasattr(alg, "moe_actor") else 5
    alg.contrastive_loss_fn = SwAVContrastiveLoss(
        gate_dim=ne, elevation_dim=cfg.get("elevation_dim", 32),
        projection_dim=cfg.get("projection_dim", 64),
        num_prototypes=cfg.get("num_prototypes", 32),
        temperature=cfg.get("temperature", 0.2),
        sinkhorn_iters=cfg.get("sinkhorn_iters", 3)).to(device)
    alg.contrastive_weight = cfg.get("weight", 0.1)
    alg.contrastive_enabled = True
    for p in alg.contrastive_loss_fn.parameters():
        alg.optimizer.add_param_group({"params": p, "lr": alg.learning_rate})
    print(f"[CMoE] Contrastive: K={cfg.get('num_prototypes',32)}, tau={cfg.get('temperature',0.2)}")


def _patch_mode_and_save(alg):
    """Patch train_mode/eval_mode and alg.save to include CMoE modules.

    NOTE: alg.save patch is kept for backward compat, but the REAL save
    integration happens in inject_cmoe_runner_patches() which patches
    runner.save / runner.load (rsl_rl's OnPolicyRunner.save does NOT
    call alg.save — it directly reads alg.policy.state_dict()).
    """
    ot, oe, os_ = getattr(alg,"train_mode",None), getattr(alg,"eval_mode",None), getattr(alg,"save",None)
    cmoe_modules = ["vae_estimator","ae_estimator","moe_actor","moe_critic","contrastive_loss_fn"]
    def tm():
        if ot: ot()
        for a in cmoe_modules:
            m = getattr(alg, a, None)
            if m and isinstance(m, nn.Module): m.train()
    def em():
        if oe: oe()
        for a in cmoe_modules:
            m = getattr(alg, a, None)
            if m and isinstance(m, nn.Module): m.eval()
    def sv():
        d = os_() if os_ else {}
        for a in cmoe_modules:
            m = getattr(alg, a, None)
            if m and isinstance(m, nn.Module): d[f"{a}_state_dict"] = m.state_dict()
        if hasattr(alg,"estimator_optimizer") and alg.estimator_optimizer:
            d["estimator_optimizer_state_dict"] = alg.estimator_optimizer.state_dict()
        return d
    alg.train_mode = tm; alg.eval_mode = em; alg.save = sv


def _escape_inference(t):
    """Create a fresh tensor from an inference-mode tensor so it can be used in backward."""
    o = torch.empty_like(t); o.copy_(t); return o


def _patch_forward_pass(alg):
    from tensordict import TensorDict
    _orig_act = alg.act; _orig_update = alg.update
    # Rollout buffer: stores per-step data for PPO update
    # This is needed because rsl_rl 3.3.0's mini_batch_generator returns tuples,
    # not named batch objects, so we maintain our own buffer.
    alg._cmoe_rb = {
        "policy_obs": [],    # [num_envs, policy_dim] per step
        "elevation_obs": [], # [num_envs, elev_dim] per step (or None)
        "critic_obs": [],    # [num_envs, critic_dim] per step
        "actions": [],       # [num_envs, action_dim] per step
        "old_log_probs": [], # [num_envs, 1] per step
        "old_mu": [],        # [num_envs, action_dim] per step
        "old_sigma": [],     # [num_envs, action_dim] per step
    }
    alg._cmoe_iter_count = 0

    def _build_systematic_obs(po, eo, ze):
        """R15 PAPER-FAITHFUL: build the actor input per paper §III-D.
        Returns [N, 259] = [v_pred(3), z_H(32), z_E(32), o_c(96), e_t(96)].

        Args:
            po: [N, 480] policy history (5 timesteps × 96)
            eo: [N, 96]  raw elevation map
            ze: [N, 32]  AE-encoded elevation latent

        Returns: actor_input [N, 259]
        """
        # VAE: history -> z_H_t (32) and v_pred (3)
        if alg.vae_estimator is None or po is None:
            raise RuntimeError("R15: VAE estimator and policy obs required")
        z_H, v_pred = alg.vae_estimator.get_latent_and_velocity(po)
        # Current obs is the LAST single_obs_dim of the history (paper Eq. 2)
        single_obs_dim = alg._cmoe_systematic_dims["o_c"]
        o_c = po[:, -single_obs_dim:]            # [N, 96]
        # AE latent (already computed by caller as ze)
        if ze is None:
            raise RuntimeError("R15: AE-encoded elevation (ze) required")
        # Raw elevation (e_t)
        if eo is None:
            raise RuntimeError("R15: elevation obs (eo) required")
        # Concatenate: [v_pred, z_H, z_E, o_c, e_t]
        return torch.cat([v_pred, z_H, ze, o_c, eo], dim=-1)

    def _cmoe_act(obs: TensorDict) -> torch.Tensor:
        po = obs.get("policy", None)
        eo = obs.get("elevation", None)
        co = obs.get("critic", None)
        ze = None
        if alg.ae_estimator and eo is not None:
            ze = alg.ae_estimator.encode(eo)
        # R15: VAE no longer called separately — _build_systematic_obs uses it
        if po is None:
            return _orig_act(obs)
        if not hasattr(alg, '_dchk'):
            alg._dchk = True
            print(f"[CMoE-R15] Actor obs: expected={alg.moe_actor.obs_dim}, "
                  f"input from systematic obs builder")
        # R15: build paper-faithful systematic observation
        sys_obs = _build_systematic_obs(po, eo, ze)
        # MoE actor forward (now with systematic obs)
        am = alg.moe_actor(sys_obs, ze)
        astd = torch.clamp(alg.moe_actor.std, min=1e-6)
        am = torch.nan_to_num(am, nan=0.0, posinf=10.0, neginf=-10.0)
        am = torch.clamp(am, -100.0, 100.0)
        dist = torch.distributions.Normal(am, astd)
        act = dist.sample().clamp(-100.0, 100.0)
        # MoE critic forward
        cp = []
        if co is not None: cp.append(co)
        if eo is not None: cp.append(eo)
        ci = torch.cat(cp, dim=-1) if cp else po
        gw = alg.moe_actor.get_gate_weights()
        vals = alg.moe_critic(ci, gw)
        vals = torch.nan_to_num(vals, nan=0.0, posinf=100.0, neginf=-100.0)
        # Store transition for rsl_rl's storage (used by compute_returns)
        alg.transition.actions = act.detach()
        alg.transition.values = vals.detach()
        log_prob = dist.log_prob(act).sum(-1, keepdim=True).detach()
        alg.transition.actions_log_prob = log_prob
        amd = am.detach(); asd = astd.expand_as(am).detach()
        alg.transition.distribution_params = (amd, asd)
        alg.transition.action_mean = amd
        alg.transition.action_sigma = asd
        if hasattr(alg,"policy") and hasattr(alg.policy,"get_hidden_state"):
            alg.transition.hidden_states = (alg.policy.get_hidden_state(), alg.policy.get_hidden_state())
        else:
            alg.transition.hidden_states = (torch.tensor([]), torch.tensor([]))
        alg.transition.observations = obs
        # Accumulate in our rollout buffer (using .data.clone() to escape inference mode)
        rb = alg._cmoe_rb
        rb["policy_obs"].append(po.data.clone())
        rb["elevation_obs"].append(eo.data.clone() if eo is not None else None)
        rb["critic_obs"].append(ci.data.clone())
        rb["actions"].append(act.data.clone())
        rb["old_log_probs"].append(log_prob.data.clone())
        rb["old_mu"].append(amd.data.clone())
        rb["old_sigma"].append(asd.data.clone())
        return act.detach()

    def _cmoe_update() -> dict:
        st = alg.storage
        rb = alg._cmoe_rb
        alg._cmoe_iter_count += 1
        diag = (alg._cmoe_iter_count <= 10) or (alg._cmoe_iter_count % 50 == 0)

        mv = ms = me = mc = mest = 0.0
        nb = 0; ns = 0

        # --- 1. Assemble rollout buffer into flat tensors ---
        n_steps = len(rb["policy_obs"])
        if n_steps == 0:
            if diag: print("[CMoE] WARNING: empty rollout buffer, falling back")
            st.clear()
            for k in rb: rb[k] = []
            return {"value": 0.0, "surrogate": 0.0, "entropy": 0.0, "contrastive": 0.0, "estimator": 0.0}

        all_policy = torch.cat(rb["policy_obs"], dim=0)       # [T*N, policy_dim]
        el_list = [t for t in rb["elevation_obs"] if t is not None]
        all_elev = torch.cat(el_list, dim=0) if el_list else None  # [T*N, elev_dim] or None
        all_critic = torch.cat(rb["critic_obs"], dim=0)       # [T*N, critic_dim]
        all_actions = torch.cat(rb["actions"], dim=0)          # [T*N, action_dim]
        all_old_lp = torch.cat(rb["old_log_probs"], dim=0)    # [T*N, 1]
        all_old_mu = torch.cat(rb["old_mu"], dim=0)            # [T*N, action_dim]
        all_old_sigma = torch.cat(rb["old_sigma"], dim=0)      # [T*N, action_dim]

        # Get advantages and returns from rsl_rl's storage (computed by compute_returns)
        all_advantages = st.advantages.reshape(-1, 1)          # [T*N, 1]
        all_returns = st.returns.reshape(-1, 1)                # [T*N, 1]
        all_values = st.values.reshape(-1, 1)                  # [T*N, 1]

        total = all_actions.shape[0]

        # Sanity check alignment
        if total != all_advantages.shape[0]:
            if diag:
                print(f"[CMoE] WARNING: buffer size {total} != storage size {all_advantages.shape[0]}")
            # Truncate to the smaller size
            min_size = min(total, all_advantages.shape[0])
            all_policy = all_policy[:min_size]
            if all_elev is not None: all_elev = all_elev[:min_size]
            all_critic = all_critic[:min_size]
            all_actions = all_actions[:min_size]
            all_old_lp = all_old_lp[:min_size]
            all_old_mu = all_old_mu[:min_size]
            all_old_sigma = all_old_sigma[:min_size]
            all_advantages = all_advantages[:min_size]
            all_returns = all_returns[:min_size]
            all_values = all_values[:min_size]
            total = min_size

        # --- 2. PPO update with mini-batches ---
        mbs = total // alg.num_mini_batches
        if mbs < 64:
            mbs = total  # use full batch if too small

        for epoch in range(alg.num_learning_epochs):
            # Shuffle indices
            perm = torch.randperm(total, device=alg.device)
            for start in range(0, total, mbs):
                end = min(start + mbs, total)
                idx = perm[start:end]
                if len(idx) == 0:
                    continue

                mb_policy = all_policy[idx]
                mb_elev = all_elev[idx] if all_elev is not None else None
                mb_critic = all_critic[idx]
                mb_actions = all_actions[idx]
                mb_old_lp = all_old_lp[idx]
                mb_old_mu = all_old_mu[idx]
                mb_old_sigma = all_old_sigma[idx]
                mb_adv = all_advantages[idx]
                mb_ret = all_returns[idx]
                mb_val = all_values[idx]

                # Normalize advantages per mini-batch
                if alg.normalize_advantage_per_mini_batch:
                    with torch.no_grad():
                        mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # --- MoE Actor forward (R15: paper-faithful systematic obs) ---
                # R15: skip if elevation missing (paper-faithful obs requires it)
                if mb_elev is None:
                    continue
                ze = None
                if alg.ae_estimator and mb_elev is not None:
                    ze = alg.ae_estimator.encode(mb_elev)
                # R15: build systematic obs for actor
                # Note: in update, mb_policy still gets its grad through actor
                # but VAE/AE are detached (their training is in section 4 below)
                with torch.no_grad():
                    z_H_mb, v_pred_mb = alg.vae_estimator.get_latent_and_velocity(mb_policy)
                    ze_detached = ze.detach() if ze is not None else None
                single_obs_dim = alg._cmoe_systematic_dims["o_c"]
                o_c_mb = mb_policy[:, -single_obs_dim:]
                sys_obs_mb = torch.cat([v_pred_mb, z_H_mb, ze_detached, o_c_mb, mb_elev], dim=-1)
                am = alg.moe_actor(sys_obs_mb, ze)
                astd = torch.clamp(alg.moe_actor.std, min=1e-6)
                am = torch.clamp(am, -100.0, 100.0)
                dist = torch.distributions.Normal(am, astd)
                new_lp = dist.log_prob(mb_actions).sum(-1, keepdim=True)
                entropy = dist.entropy().sum(-1)

                # --- MoE Critic forward ---
                gw = alg.moe_actor.get_gate_weights()
                new_val = alg.moe_critic(mb_critic, gw)

                # --- Adaptive LR ---
                if alg.desired_kl and alg.schedule == "adaptive":
                    with torch.inference_mode():
                        kl = torch.distributions.kl_divergence(
                            torch.distributions.Normal(mb_old_mu, mb_old_sigma),
                            torch.distributions.Normal(am.detach(), astd.expand_as(am).detach())
                        ).sum(-1).mean()
                        if kl > alg.desired_kl * 2.0:
                            alg.learning_rate = max(1e-5, alg.learning_rate / 1.5)
                        elif kl < alg.desired_kl / 2.0 and kl > 0:
                            alg.learning_rate = min(1e-2, alg.learning_rate * 1.5)
                        for pg in alg.optimizer.param_groups:
                            pg["lr"] = alg.learning_rate

                # --- PPO Surrogate Loss ---
                log_ratio = torch.clamp(new_lp - mb_old_lp, -20.0, 20.0)
                ratio = torch.clamp(torch.exp(log_ratio), 0.0, 10.0)
                adv_sq = mb_adv.squeeze()
                surr1 = -adv_sq * ratio.squeeze()
                surr2 = -adv_sq * torch.clamp(ratio.squeeze(), 1 - alg.clip_param, 1 + alg.clip_param)
                surrogate_loss = torch.max(surr1, surr2).mean()

                # --- Value Loss ---
                if alg.use_clipped_value_loss:
                    val_clipped = mb_val + (new_val - mb_val).clamp(-alg.clip_param, alg.clip_param)
                    vl1 = (new_val - mb_ret).pow(2)
                    vl2 = (val_clipped - mb_ret).pow(2)
                    value_loss = torch.max(vl1, vl2).mean()
                else:
                    value_loss = (mb_ret - new_val).pow(2).mean()

                # --- Contrastive Loss (Paper §III-E) ---
                c_term = torch.tensor(0.0, device=alg.device)
                c_val = 0.0
                if hasattr(alg, "contrastive_enabled") and alg.contrastive_enabled and ze is not None:
                    gl = alg.moe_actor.get_gate_logits()
                    if gl is not None:
                        c_term = alg.contrastive_loss_fn(gl, ze)
                        if torch.isnan(c_term) or torch.isinf(c_term):
                            c_term = torch.tensor(0.0, device=alg.device)
                        else:
                            c_val = c_term.item()

                # --- Total Loss ---
                loss = (surrogate_loss
                        + alg.value_loss_coef * value_loss
                        - alg.entropy_coef * entropy.mean()
                        + alg.contrastive_weight * c_term)

                if torch.isnan(loss) or torch.isinf(loss):
                    ns += 1
                    continue

                # --- Backward + Step ---
                alg.optimizer.zero_grad()
                loss.backward()
                all_params = list(alg.moe_actor.parameters()) + list(alg.moe_critic.parameters())
                if hasattr(alg, "contrastive_loss_fn") and alg.contrastive_loss_fn:
                    all_params += list(alg.contrastive_loss_fn.parameters())
                nn.utils.clip_grad_norm_(all_params, alg.max_grad_norm)
                # Replace NaN gradients with zero
                for p in all_params:
                    if p.grad is not None:
                        p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                alg.optimizer.step()

                mv += value_loss.item()
                ms += surrogate_loss.item()
                me += entropy.mean().item()
                mc += c_val
                nb += 1

        # --- 3. Diagnostics ---
        if diag:
            tb = alg.num_learning_epochs * alg.num_mini_batches
            print(f"[CMoE] iter={alg._cmoe_iter_count}: {nb}/{tb} batches ok, {ns} nan_skipped, "
                  f"lr={alg.learning_rate:.2e}")
            if alg._cmoe_iter_count <= 3:
                for nm, p in alg.moe_actor.named_parameters():
                    if p.grad is not None and p.grad.norm().item() > 0:
                        print(f"[CMoE] Grad OK: {nm} norm={p.grad.norm().item():.6f}")
                        break
                else:
                    print("[CMoE] WARNING: No grads on moe_actor!")

        # --- 4. Estimator training (VAE + AE) using ALL buffer data ---
        if alg.estimator_optimizer and n_steps > 0:
            try:
                with torch.enable_grad():
                    # Escape inference mode for all tensors
                    est_policy = torch.cat([_escape_inference(t) for t in rb["policy_obs"]], 0)
                    est_critic = torch.cat([_escape_inference(t) for t in rb["critic_obs"]], 0)
                    el_raw = [t for t in rb["elevation_obs"] if t is not None]
                    est_elev = torch.cat([_escape_inference(t) for t in el_raw], 0) if el_raw else None

                    et = est_policy.shape[0]
                    emb = max(et // 4, 256)
                    ne_ = 0; es_ = 0.0

                    for s in range(0, et, emb):
                        e = min(s + emb, et)
                        alg.estimator_optimizer.zero_grad()
                        el_ = torch.zeros(1, device=alg.device)

                        if alg.vae_estimator:
                            # Critic obs first 3 dims = base_lin_vel (ground truth velocity)
                            co_slice = est_critic[s:e, :99] if est_critic.shape[-1] >= 99 else est_critic[s:e]
                            vel_gt = co_slice[:, :3]
                            cur_obs = est_policy[s:e, -alg.vae_estimator.obs_dim:]
                            vl_, _ = alg.vae_estimator.compute_loss(est_policy[s:e], cur_obs, vel_gt)
                            el_ = el_ + vl_

                        if alg.ae_estimator and est_elev is not None:
                            al_, _ = alg.ae_estimator.compute_loss(est_elev[s:e])
                            el_ = el_ + al_

                        if el_.requires_grad:
                            el_.backward()
                            if alg.vae_estimator:
                                nn.utils.clip_grad_norm_(alg.vae_estimator.parameters(), alg.max_grad_norm)
                            if alg.ae_estimator:
                                nn.utils.clip_grad_norm_(alg.ae_estimator.parameters(), alg.max_grad_norm)
                            alg.estimator_optimizer.step()
                            es_ += el_.item(); ne_ += 1

                    mest = es_ / max(ne_, 1)
            except Exception as ex:
                if diag:
                    print(f"[CMoE] Est err: {ex}")

        # --- 5. Cleanup ---
        for k in rb: rb[k] = []
        st.clear()
        n = max(nb, 1)
        return {
            "value": mv / n,
            "surrogate": ms / n,
            "entropy": me / n,
            "contrastive": mc / n,
            "estimator": mest,
        }

    alg.act = _cmoe_act
    alg.update = _cmoe_update

    # --- Patch compute_returns to use MoE critic ---
    _orig_cr = alg.compute_returns

    def _cmoe_cr(obs: TensorDict):
        eo = obs.get("elevation", None)
        co = obs.get("critic", None)
        ze = None
        if alg.ae_estimator and eo is not None:
            ze = alg.ae_estimator.encode(eo)
        cp = []
        if co is not None: cp.append(co)
        if eo is not None: cp.append(eo)
        if not cp:
            return _orig_cr(obs)
        ci = torch.cat(cp, dim=-1)
        gw = None
        if ze is not None:
            gl = alg.moe_actor.gating(ze)
            gw = F.softmax(gl, dim=-1)
        lv = alg.moe_critic(ci, gw).detach()
        st = alg.storage
        adv = 0
        for step in reversed(range(st.num_transitions_per_env)):
            nv = lv if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            nnt = 1.0 - st.dones[step].float()
            delta = st.rewards[step] + nnt * alg.gamma * nv - st.values[step]
            adv = delta + nnt * alg.gamma * alg.lam * adv
            st.returns[step] = adv + st.values[step]
        st.advantages = st.returns - st.values
        if not alg.normalize_advantage_per_mini_batch:
            st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

    alg.compute_returns = _cmoe_cr
    print("[CMoE] Patched: act(), update(), compute_returns() use MoE")


# ==============================================================================
# CRITICAL FIX (April 2026):
#
# rsl_rl's OnPolicyRunner.save() does NOT call alg.save() — it directly
# constructs the saved_dict from alg.policy.state_dict() and
# alg.optimizer.state_dict(). This means the original _patch_mode_and_save()
# above silently failed to save MoE weights for the entire 17-hour training
# run on the first attempt.
#
# This function patches OnPolicyRunner.save() and OnPolicyRunner.load() to
# add/load the CMoE-specific state dicts. It MUST be called AFTER inject_cmoe()
# in train.py and evaluate scripts.
# ==============================================================================
def inject_cmoe_runner_patches(runner):
    """Patch runner.save and runner.load to handle CMoE state dicts.

    Without this patch, the trained MoE/AE/VAE weights are NEVER saved
    to disk. rsl_rl's OnPolicyRunner.save reads only alg.policy.state_dict
    and alg.optimizer.state_dict, completely bypassing the patched alg.save.
    """
    alg = runner.alg
    cmoe_modules = ["vae_estimator", "ae_estimator", "moe_actor", "moe_critic", "contrastive_loss_fn"]

    # check that alg has the cmoe modules - if not, this is a no-op
    has_cmoe = any(getattr(alg, name, None) is not None for name in cmoe_modules)
    if not has_cmoe:
        print("[CMoE] inject_cmoe_runner_patches: no CMoE modules found, skipping patch")
        return

    orig_save = runner.save
    orig_load = runner.load

    def patched_save(path, infos=None):
        """Save with CMoE state dicts appended to the checkpoint dict."""
        # call original save which writes alg.policy / alg.optimizer / iter / infos
        orig_save(path, infos=infos)
        # now reload, append CMoE state dicts, and resave
        try:
            saved = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            # older torch without weights_only arg
            saved = torch.load(path, map_location="cpu")

        for attr_name in cmoe_modules:
            module = getattr(alg, attr_name, None)
            if module is not None and isinstance(module, nn.Module):
                saved[f"{attr_name}_state_dict"] = module.state_dict()

        if hasattr(alg, "estimator_optimizer") and alg.estimator_optimizer is not None:
            saved["estimator_optimizer_state_dict"] = alg.estimator_optimizer.state_dict()

        torch.save(saved, path)
        # report only on first save and every 1000 iters
        cur_iter = getattr(runner, "current_learning_iteration", 0)
        if cur_iter <= 5 or cur_iter % 1000 == 0:
            keys = [k for k in saved.keys() if "state_dict" in k]
            print(f"[CMoE-SAVE] iter={cur_iter} → wrote keys: {sorted(keys)}")

    def patched_load(path, load_optimizer=True):
        """Load: invoke original load, then load CMoE state dicts."""
        # call original load (handles model_state_dict + optimizer_state_dict)
        result = orig_load(path, load_optimizer=load_optimizer) if _accepts_kwarg(orig_load, "load_optimizer") else orig_load(path)

        try:
            loaded = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            loaded = torch.load(path, map_location="cpu")

        n_loaded = 0
        n_missing = 0
        device = next(alg.policy.parameters()).device
        for attr_name in cmoe_modules:
            module = getattr(alg, attr_name, None)
            if module is None:
                continue
            key = f"{attr_name}_state_dict"
            if key in loaded:
                state = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                         for k, v in loaded[key].items()}
                missing, unexpected = module.load_state_dict(state, strict=False)
                if missing or unexpected:
                    print(f"[CMoE-LOAD] {attr_name}: ◐ missing={missing[:2]} unexpected={unexpected[:2]}")
                else:
                    print(f"[CMoE-LOAD] {attr_name}: ✓ loaded cleanly")
                n_loaded += 1
            else:
                print(f"[CMoE-LOAD] {attr_name}: ✗ key '{key}' NOT in checkpoint")
                n_missing += 1

        if hasattr(alg, "estimator_optimizer") and alg.estimator_optimizer is not None:
            if "estimator_optimizer_state_dict" in loaded and load_optimizer:
                try:
                    alg.estimator_optimizer.load_state_dict(loaded["estimator_optimizer_state_dict"])
                    print("[CMoE-LOAD] estimator_optimizer: ✓ loaded")
                except Exception as e:
                    print(f"[CMoE-LOAD] estimator_optimizer: ✗ {e}")

        print(f"[CMoE-LOAD] Summary: {n_loaded} CMoE modules loaded, {n_missing} missing")
        return result

    runner.save = patched_save
    runner.load = patched_load
    print("[CMoE] runner.save / runner.load patched ✓ — MoE weights will be saved/loaded.")


def _accepts_kwarg(func, kwarg_name):
    """Check if a callable accepts a given kwarg."""
    import inspect
    try:
        sig = inspect.signature(func)
        return kwarg_name in sig.parameters
    except (ValueError, TypeError):
        return False
