"""VAE-based proprioceptive state estimator (paper III-C, Eq. 3-4).

This module implements the beta-VAE that processes observation history o_t^H
and outputs:
  - Estimated body velocity v_t (3D linear velocity)
  - Latent representation z_H_t (for downstream actor input)

The module is trained with a combined loss:
  L_CS = MSE(v_pred, v_gt) + L_VAE
  L_VAE = MSE(o_pred_{t+1}, o_{t+1}) + beta * KL(q(z|o_H) || p(z))
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VAEEstimator(nn.Module):
    """beta-VAE for proprioceptive state estimation.

    Args:
        obs_dim: Dimension of a single-timestep proprioceptive observation.
        history_length: Number of past timesteps stacked with current obs.
            Encoder input dim = obs_dim * (history_length + 1).
        latent_dim: Dimension of the latent space z_H_t.
        velocity_dim: Dimension of the velocity estimate (default 3 for xyz).
        encoder_hidden_dims: Hidden layer sizes for the encoder MLP.
        decoder_hidden_dims: Hidden layer sizes for the decoder MLP.
        beta: Weight for the KL divergence term.
        activation: Activation function name.
    """

    def __init__(
        self,
        obs_dim: int,
        history_length: int = 5,
        latent_dim: int = 32,
        velocity_dim: int = 3,
        encoder_hidden_dims: list[int] | tuple[int, ...] = (256, 128),
        decoder_hidden_dims: list[int] | tuple[int, ...] = (128, 256),
        beta: float = 0.001,
        activation: str = "elu",
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.history_length = history_length
        self.latent_dim = latent_dim
        self.velocity_dim = velocity_dim
        self.beta = beta

        # rsl_rl stacks history as [o_{t-H}, ..., o_{t-1}, o_t]
        self.encoder_input_dim = obs_dim * (history_length + 1)

        act_fn = _get_activation(activation)

        # --- Encoder ---
        encoder_layers = []
        in_dim = self.encoder_input_dim
        for h_dim in encoder_hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            encoder_layers.append(act_fn)
            in_dim = h_dim
        self.encoder_backbone = nn.Sequential(*encoder_layers)

        # Separate heads for mu, log_var, and velocity
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_log_var = nn.Linear(in_dim, latent_dim)
        self.fc_velocity = nn.Linear(in_dim, velocity_dim)

        # --- Decoder ---
        decoder_layers = []
        in_dim = latent_dim
        for h_dim in decoder_hidden_dims:
            decoder_layers.append(nn.Linear(in_dim, h_dim))
            decoder_layers.append(act_fn)
            in_dim = h_dim
        decoder_layers.append(nn.Linear(in_dim, obs_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, obs_history: torch.Tensor):
        """Encode observation history into latent distribution and velocity.

        Args:
            obs_history: [N, (H+1)*obs_dim] flattened observation history.

        Returns:
            mu, log_var, velocity, z (sampled via reparameterization).
        """
        h = self.encoder_backbone(obs_history)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        velocity = self.fc_velocity(h)
        z = self._reparameterize(mu, log_var)
        return mu, log_var, velocity, z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent z to reconstruct next observation [N, obs_dim]."""
        return self.decoder(z)

    def forward(self, obs_history: torch.Tensor):
        """Full forward: encode -> sample -> decode.

        Returns: velocity, z, recon, mu, log_var
        """
        mu, log_var, velocity, z = self.encode(obs_history)
        recon = self.decode(z)
        return velocity, z, recon, mu, log_var

    def compute_loss(
        self,
        obs_history: torch.Tensor,
        next_obs: torch.Tensor,
        velocity_gt: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute combined VAE loss (paper Eq. 3-4).

        L_CS = MSE(v_pred, v_gt) + MSE(o_pred, o_{t+1}) + beta * KL

        Args:
            obs_history: [N, (H+1)*obs_dim].
            next_obs: [N, obs_dim] ground-truth next observation.
            velocity_gt: [N, velocity_dim] ground-truth body velocity.

        Returns:
            total_loss, loss_dict
        """
        velocity_pred, z, recon, mu, log_var = self.forward(obs_history)

        vel_loss = nn.functional.mse_loss(velocity_pred, velocity_gt)
        recon_loss = nn.functional.mse_loss(recon, next_obs)
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
        )

        total_loss = vel_loss + recon_loss + self.beta * kl_loss

        loss_dict = {
            "vae/velocity_loss": vel_loss.item(),
            "vae/reconstruction_loss": recon_loss.item(),
            "vae/kl_loss": kl_loss.item(),
            "vae/total_loss": total_loss.item(),
        }
        return total_loss, loss_dict

    def get_latent_and_velocity(self, obs_history: torch.Tensor):
        """Inference-only: get z_H and velocity without decoder.

        Used during rollout collection to provide z_H_t and v_t to actor.

        Returns: z [N, latent_dim], velocity [N, velocity_dim]
        """
        with torch.no_grad():
            mu, log_var, velocity, z = self.encode(obs_history)
        return z, velocity

    def _reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + sigma * epsilon."""
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            return mu


def _get_activation(name: str) -> nn.Module:
    activations = {
        "relu": nn.ReLU(),
        "elu": nn.ELU(),
        "tanh": nn.Tanh(),
        "leaky_relu": nn.LeakyReLU(),
        "selu": nn.SELU(),
    }
    if name.lower() not in activations:
        raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
    return activations[name.lower()]
