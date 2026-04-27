"""Custom PPO with CMoE integration (Paper III-D, III-E).

Based on humanoid_locomotion reference project's custom PPO pattern.
Uses separate self.actor and self.critic (both MLPModel), then adds
MoE, estimators, and contrastive learning.

Key: MoE actor/critic are trained through PPO loss in update(),
not just used during act(). This ensures gradient flow.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from itertools import chain
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups, resolve_optimizer

# MLPModel only exists in newer rsl_rl (>=4.0). In rsl_rl 3.3.0 (bundled with
# IsaacLab 2.x / IsaacSim 5.x) the actor/critic are plain nn.Module subclasses
# created by the OnPolicyRunner. We only need the name for type hints.
try:
    from rsl_rl.models import MLPModel
except ImportError:
    MLPModel = nn.Module  # type: ignore[misc,assignment]

try:
    from rsl_rl.extensions import RandomNetworkDistillation, resolve_rnd_config, resolve_symmetry_config
except ImportError:
    RandomNetworkDistillation = None
    resolve_rnd_config = None
    resolve_symmetry_config = None

from cmoe.custom_classes.modules import VAEEstimator, AEEstimator
from cmoe.custom_classes.models.cmoe_moe_model import MoEActorModel, MoECriticModel
from cmoe.custom_classes.models.cmoe_contrastive import SwAVContrastiveLoss


class PPO:
    """PPO with Contrastive Mixture of Experts (CMoE).

    Architecture (Paper Fig. 3):
      act():  obs -> AE(elevation)->z_E -> gating(z_E)->g_i
              obs -> expert_i(obs)->mu_i -> action = sum(softmax(g_i)*mu_i)
      update(): recompute MoE forward -> PPO loss + contrastive loss + estimator loss
    """

    actor: MLPModel
    critic: MLPModel

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
        cmoe_cfg: dict | None = None,
    ) -> None:
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        self.gpu_global_rank = multi_gpu_cfg["global_rank"] if multi_gpu_cfg else 0
        self.gpu_world_size = multi_gpu_cfg["world_size"] if multi_gpu_cfg else 1

        # RND
        if rnd_cfg and RandomNetworkDistillation is not None:
            rnd_lr = rnd_cfg.pop("learning_rate", 1e-3)
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            self.rnd_optimizer = optim.Adam(self.rnd.predictor.parameters(), lr=rnd_lr)
        else:
            self.rnd = None
            self.rnd_optimizer = None

        # Symmetry
        if symmetry_cfg is not None:
            if "data_augmentation_func" in symmetry_cfg:
                symmetry_cfg["data_augmentation_func"] = resolve_callable(
                    symmetry_cfg["data_augmentation_func"]
                )
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # Base models
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.storage = storage
        self.transition = RolloutStorage.Transition()

        # PPO params
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        # CMoE components
        self.moe_actor = None
        self.moe_critic = None
        self.vae_estimator = None
        self.ae_estimator = None
        self.contrastive_loss_fn = None
        self.contrastive_weight = 0.1
        self.estimator_optimizer = None

        if cmoe_cfg is not None:
            self._init_cmoe(cmoe_cfg)

        # Main optimizer: base actor/critic + MoE + contrastive
        all_params = list(self.actor.parameters()) + list(self.critic.parameters())
        if self.moe_actor is not None:
            all_params += list(self.moe_actor.parameters())
        if self.moe_critic is not None:
            all_params += list(self.moe_critic.parameters())
        if self.contrastive_loss_fn is not None:
            all_params += list(self.contrastive_loss_fn.parameters())
        seen = set()
        unique = []
        for p in all_params:
            if id(p) not in seen:
                seen.add(id(p))
                unique.append(p)
        self.optimizer = resolve_optimizer(optimizer)(unique, lr=learning_rate)

    def _init_cmoe(self, cfg: dict):
        est_cfg = cfg.get("estimator", {})
        moe_cfg = cfg.get("moe", {})
        cont_cfg = cfg.get("contrastive", {})

        # Estimators
        est_params = []
        vae_c = est_cfg.get("vae")
        if vae_c:
            self.vae_estimator = VAEEstimator(**vae_c).to(self.device)
            est_params += list(self.vae_estimator.parameters())
            print(f"[CMoE] VAE: latent={vae_c['latent_dim']}")

        ae_c = est_cfg.get("ae")
        if ae_c:
            self.ae_estimator = AEEstimator(**ae_c).to(self.device)
            est_params += list(self.ae_estimator.parameters())
            print(f"[CMoE] AE: latent={ae_c['latent_dim']}")

        if est_params:
            self.estimator_optimizer = optim.Adam(est_params, lr=est_cfg.get("learning_rate", 1e-3))

        # MoE
        if moe_cfg:
            n_exp = moe_cfg.get("num_experts", 5)
            gate_in = moe_cfg.get("gate_input_dim", 32)
            gate_h = moe_cfg.get("gate_hidden_dims", [64, 32])
            std0 = moe_cfg.get("init_noise_std", 1.0)

            a_in, a_out, a_h = self._dims(self.actor)
            c_in, _, c_h = self._dims(self.critic)
            print(f"[CMoE] Actor: {a_in}->{a_h}->{a_out}, Critic: {c_in}->{c_h}->1")

            self.moe_actor = MoEActorModel(
                obs_dim=a_in, action_dim=a_out, num_experts=n_exp,
                expert_hidden_dims=a_h or [512, 256, 128],
                gate_input_dim=gate_in, gate_hidden_dims=gate_h,
                init_noise_std=std0,
            ).to(self.device)

            self.moe_critic = MoECriticModel(
                obs_dim=c_in, num_experts=n_exp,
                expert_hidden_dims=c_h or [512, 256, 128],
                shared_gating=self.moe_actor.gating,
            ).to(self.device)
            print(f"[CMoE] MoE: {n_exp} experts, gate_in={gate_in}")

        # Contrastive
        if cont_cfg and self.moe_actor:
            self.contrastive_loss_fn = SwAVContrastiveLoss(
                gate_dim=self.moe_actor.num_experts,
                elevation_dim=cont_cfg.get("elevation_dim", 32),
                projection_dim=cont_cfg.get("projection_dim", 64),
                num_prototypes=cont_cfg.get("num_prototypes", 32),
                temperature=cont_cfg.get("temperature", 0.2),
                sinkhorn_iters=cont_cfg.get("sinkhorn_iters", 3),
            ).to(self.device)
            self.contrastive_weight = cont_cfg.get("weight", 0.1)
            print(f"[CMoE] Contrastive: K={cont_cfg.get('num_prototypes',32)}, tau={cont_cfg.get('temperature',0.2)}")

        print("[CMoE] Init complete")

    def _dims(self, m):
        i = o = None
        h = []
        for mod in m.modules():
            if isinstance(mod, nn.Linear):
                if i is None: i = mod.in_features
                o = mod.out_features
                h.append(mod.out_features)
        if h: h = h[:-1]
        return i, o, h

    def _ae_encode(self, obs):
        """Encode elevation obs through AE, return z_E or None."""
        e = obs.get("elevation", None)
        if self.ae_estimator is not None and e is not None:
            return self.ae_estimator.encode(e), e
        return None, e

    # ==================================================================
    def act(self, obs: TensorDict) -> torch.Tensor:
        self.transition.hidden_states = (self.actor.get_hidden_state(), self.critic.get_hidden_state())

        if self.moe_actor is not None:
            actor_obs = self.actor.get_obs(obs)
            z_e, elev_raw = self._ae_encode(obs)

            action_mean = self.moe_actor(actor_obs, z_e)
            action_std = self.moe_actor.std
            dist = torch.distributions.Normal(action_mean, action_std)
            actions = dist.sample()
            log_prob = dist.log_prob(actions).sum(-1, keepdim=True)

            critic_obs = self.critic.get_obs(obs)
            gw = self.moe_actor.get_gate_weights()
            values = self.moe_critic(critic_obs, gw)

            self.transition.actions = actions.detach()
            self.transition.values = values.detach()
            self.transition.actions_log_prob = log_prob.detach()
            self.transition.distribution_params = (
                action_mean.detach(), action_std.expand_as(action_mean).detach()
            )
            self.transition.observations = obs
            return actions.detach()
        else:
            self.transition.actions = self.actor(obs, stochastic_output=True).detach()
            self.transition.values = self.critic(obs).detach()
            self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()
            self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
            self.transition.observations = obs
            return self.transition.actions

    def process_env_step(self, obs, rewards, dones, extras):
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)
        if self.rnd: self.rnd.update_normalization(obs)
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if self.rnd:
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            self.transition.rewards += self.intrinsic_rewards
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
            )
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)

    def compute_returns(self, obs: TensorDict):
        st = self.storage
        if self.moe_critic is not None:
            critic_obs = self.critic.get_obs(obs)
            z_e, _ = self._ae_encode(obs)
            gw = None
            if z_e is not None:
                gl = self.moe_actor.gating(z_e)
                gw = F.softmax(gl, dim=-1)
            last_values = self.moe_critic(critic_obs, gw).detach()
        else:
            last_values = self.critic(obs).detach()

        adv = 0
        for step in reversed(range(st.num_transitions_per_env)):
            nv = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            nnt = 1.0 - st.dones[step].float()
            delta = st.rewards[step] + nnt * self.gamma * nv - st.values[step]
            adv = delta + nnt * self.gamma * self.lam * adv
            st.returns[step] = adv + st.values[step]
        st.advantages = st.returns - st.values
        if not self.normalize_advantage_per_mini_batch:
            st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

    def update(self):
        mv = ms = me = mest = mc = 0.0
        gen = (self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
               if (self.actor.is_recurrent or self.critic.is_recurrent)
               else self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs))

        for batch in gen:
            obs_bs = batch.observations.batch_size[0]
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

            # Recompute with current params
            if self.moe_actor is not None:
                actor_obs = self.actor.get_obs(batch.observations)
                critic_obs = self.critic.get_obs(batch.observations)
                z_e, _ = self._ae_encode(batch.observations)

                action_mean = self.moe_actor(actor_obs, z_e)
                action_std = self.moe_actor.std
                dist = torch.distributions.Normal(action_mean, action_std)
                alp = dist.log_prob(batch.actions).sum(-1, keepdim=True)
                ent = dist.entropy().sum(-1)[:obs_bs]
                gw = self.moe_actor.get_gate_weights()
                vals = self.moe_critic(critic_obs, gw)
                dp = (action_mean[:obs_bs], action_std.expand_as(action_mean)[:obs_bs])
            else:
                self.actor(batch.observations, masks=batch.masks,
                           hidden_state=batch.hidden_states[0], stochastic_output=True)
                alp = self.actor.get_output_log_prob(batch.actions)
                vals = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
                dp = tuple(p[:obs_bs] for p in self.actor.output_distribution_params)
                ent = self.actor.output_entropy[:obs_bs]

            # Adaptive LR
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    if self.moe_actor is not None:
                        om, os_ = batch.old_distribution_params[0], batch.old_distribution_params[1]
                        nm, ns_ = dp[0], dp[1]
                        kl = torch.distributions.kl_divergence(
                            torch.distributions.Normal(om, os_),
                            torch.distributions.Normal(nm, ns_)
                        ).sum(-1).mean()
                    else:
                        kl = torch.mean(self.actor.get_kl_divergence(batch.old_distribution_params, dp))
                    if self.gpu_global_rank == 0:
                        if kl > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl < self.desired_kl / 2.0 and kl > 0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = self.learning_rate

            # PPO losses
            ratio = torch.exp(alp - torch.squeeze(batch.old_actions_log_prob))
            surr = -torch.squeeze(batch.advantages) * ratio
            surr_c = -torch.squeeze(batch.advantages) * torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param)
            surr_loss = torch.max(surr, surr_c).mean()

            if self.use_clipped_value_loss:
                vc = batch.values + (vals - batch.values).clamp(-self.clip_param, self.clip_param)
                vl = (vals - batch.returns).pow(2)
                vlc = (vc - batch.returns).pow(2)
                val_loss = torch.max(vl, vlc).mean()
            else:
                val_loss = (batch.returns - vals).pow(2).mean()

            loss = surr_loss + self.value_loss_coef * val_loss - self.entropy_coef * ent.mean()

            # Contrastive (Eq. 8)
            cl = 0.0
            if self.contrastive_loss_fn is not None and self.moe_actor is not None and z_e is not None:
                gl = self.moe_actor.get_gate_logits()
                if gl is not None:
                    c_loss = self.contrastive_loss_fn(gl, z_e)
                    loss = loss + self.contrastive_weight * c_loss
                    cl = c_loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            if self.moe_actor: nn.utils.clip_grad_norm_(self.moe_actor.parameters(), self.max_grad_norm)
            if self.moe_critic: nn.utils.clip_grad_norm_(self.moe_critic.parameters(), self.max_grad_norm)
            if self.contrastive_loss_fn: nn.utils.clip_grad_norm_(self.contrastive_loss_fn.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mv += val_loss.item()
            ms += surr_loss.item()
            me += ent.mean().item()
            mc += cl

        # Estimator training (separate optimizer)
        if self.estimator_optimizer is not None:
            self.estimator_optimizer.zero_grad()
            el = torch.tensor(0.0, device=self.device)
            try:
                so = self.storage.observations
                if so is not None:
                    if self.vae_estimator is not None:
                        po = so.get("policy", None)
                        co = so.get("critic", None)
                        if po is not None and co is not None:
                            si = torch.randint(0, po.shape[0], (1,)).item()
                            vl_gt = co[si][:, :3]
                            cur = po[si][:, -self.vae_estimator.obs_dim:]
                            vl, _ = self.vae_estimator.compute_loss(po[si], cur, vl_gt)
                            el = el + vl
                    if self.ae_estimator is not None:
                        eo = so.get("elevation", None)
                        if eo is not None:
                            si = torch.randint(0, eo.shape[0], (1,)).item()
                            al, _ = self.ae_estimator.compute_loss(eo[si])
                            el = el + al
                    if el.requires_grad:
                        el.backward()
                        if self.vae_estimator: nn.utils.clip_grad_norm_(self.vae_estimator.parameters(), self.max_grad_norm)
                        if self.ae_estimator: nn.utils.clip_grad_norm_(self.ae_estimator.parameters(), self.max_grad_norm)
                        self.estimator_optimizer.step()
                        mest = el.item()
            except Exception as e:
                print(f"[CMoE] Estimator warning: {e}")

        n = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        return {"value": mv/n, "surrogate": ms/n, "entropy": me/n, "estimator": mest, "contrastive": mc/n}

    def train_mode(self):
        self.actor.train(); self.critic.train()
        for m in [self.moe_actor, self.moe_critic, self.vae_estimator, self.ae_estimator, self.contrastive_loss_fn]:
            if m and isinstance(m, nn.Module): m.train()

    def eval_mode(self):
        self.actor.eval(); self.critic.eval()
        for m in [self.moe_actor, self.moe_critic, self.vae_estimator, self.ae_estimator, self.contrastive_loss_fn]:
            if m and isinstance(m, nn.Module): m.eval()

    def save(self):
        d = {"actor_state_dict": self.actor.state_dict(), "critic_state_dict": self.critic.state_dict(),
             "optimizer_state_dict": self.optimizer.state_dict()}
        for n in ["moe_actor","moe_critic","vae_estimator","ae_estimator","contrastive_loss_fn"]:
            m = getattr(self, n, None)
            if m and isinstance(m, nn.Module): d[f"{n}_state_dict"] = m.state_dict()
        if self.estimator_optimizer: d["estimator_optimizer_state_dict"] = self.estimator_optimizer.state_dict()
        return d

    def load(self, ld, lc, strict):
        if lc is None: lc = {"actor":True,"critic":True,"optimizer":True,"iteration":True}
        if lc.get("actor"): self.actor.load_state_dict(ld["actor_state_dict"], strict=strict)
        if lc.get("critic"): self.critic.load_state_dict(ld["critic_state_dict"], strict=strict)
        if lc.get("optimizer") and "optimizer_state_dict" in ld:
            try: self.optimizer.load_state_dict(ld["optimizer_state_dict"])
            except: pass
        for n in ["moe_actor","moe_critic","vae_estimator","ae_estimator","contrastive_loss_fn"]:
            k = f"{n}_state_dict"
            m = getattr(self, n, None)
            if m and k in ld: m.load_state_dict(ld[k], strict=strict)
        return lc.get("iteration", False)

    def get_policy(self): return self.actor

    @staticmethod
    def construct_algorithm(obs, env, cfg, device):
        alg_cls = resolve_callable(cfg["algorithm"].pop("class_name"))
        act_cls = resolve_callable(cfg["actor"].pop("class_name"))
        crt_cls = resolve_callable(cfg["critic"].pop("class_name"))
        ds = ["actor","critic"]
        if cfg["algorithm"].get("rnd_cfg"): ds.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ds)
        if resolve_rnd_config: cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        if resolve_symmetry_config: cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)
        actor = act_cls(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"Actor Model: {actor}")
        if cfg["algorithm"].pop("share_cnn_encoders", None):
            cfg["critic"]["cnns"] = actor.cnns
        critic = crt_cls(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")
        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
        return alg_cls(actor, critic, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"])

    def broadcast_parameters(self):
        mp = [self.actor.state_dict(), self.critic.state_dict()]
        torch.distributed.broadcast_object_list(mp, src=0)
        self.actor.load_state_dict(mp[0]); self.critic.load_state_dict(mp[1])

    def reduce_parameters(self):
        ap = list(chain(self.actor.parameters(), self.critic.parameters()))
        if self.moe_actor: ap += list(self.moe_actor.parameters())
        if self.moe_critic: ap += list(self.moe_critic.parameters())
        if self.contrastive_loss_fn: ap += list(self.contrastive_loss_fn.parameters())
        gs = [p.grad.view(-1) for p in ap if p.grad is not None]
        if gs:
            ag = torch.cat(gs)
            torch.distributed.all_reduce(ag, op=torch.distributed.ReduceOp.SUM)
            ag /= self.gpu_world_size
            off = 0
            for p in ap:
                if p.grad is not None:
                    n = p.numel()
                    p.grad.data.copy_(ag[off:off+n].view_as(p.grad.data))
                    off += n
