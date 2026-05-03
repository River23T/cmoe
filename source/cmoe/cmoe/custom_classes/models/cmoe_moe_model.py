"""CMoE Actor Model with Mixture of Experts (Paper §III-D).

Architecture per Fig. 3 and Eq. 6:
- N expert MLPs (default N=5), each produces action mean μ_i
- Gating network receives AE-encoded elevation z_E_t, outputs expert activations g_i
- Final action = Σ softmax(g_i) * μ_i  (Eq. 6)
- The gating network is shared with the critic for consistency

Compatible with rsl_rl 3.3.0 (no MLPModel dependency).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


def _build_mlp(input_dim: int, hidden_dims: List[int], output_dim: int, activation: str = "elu") -> nn.Sequential:
    """Build a simple MLP with given hidden dims and activation."""
    act_fn = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "selu": nn.SELU}[activation]
    layers = []
    prev_dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev_dim, h))
        layers.append(act_fn())
        prev_dim = h
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class GatingNetwork(nn.Module):
    """Gating network that maps encoded elevation to expert activation weights.

    Paper §III-D: "a perceptual gating network dynamically adjusts the activation
    of experts based on environmental perception"

    Input: z_E_t (AE-encoded elevation latent, dim=latent_dim)
    Output: g = [g_1, ..., g_N] raw logits (softmax applied externally)
    """

    def __init__(self, input_dim: int, num_experts: int, hidden_dims: List[int] = [64, 32]):
        super().__init__()
        self.net = _build_mlp(input_dim, hidden_dims, num_experts, activation="elu")
        self.num_experts = num_experts

    def forward(self, z_elevation: torch.Tensor) -> torch.Tensor:
        """Returns raw gate logits [batch, num_experts]."""
        return self.net(z_elevation)


class MoEActorModel(nn.Module):
    """Mixture-of-Experts Actor following CMoE paper §III-D.

    Each expert is an MLP: obs → action_mean
    Gating network: z_E → expert weights
    Output: weighted sum of expert means (Eq. 6)

    The std (action noise) is shared across all experts as a learnable parameter.

    Args:
        obs_dim: Total observation dimension for actor (policy obs, concatenated)
        action_dim: Action dimension (29 for G1)
        num_experts: Number of expert networks (paper uses 5)
        expert_hidden_dims: Hidden layer sizes for each expert MLP
        gate_input_dim: Dimension of gating input (AE latent dim, default 32)
        gate_hidden_dims: Hidden layers for gating network
        init_noise_std: Initial action noise std
        activation: Activation function name
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_experts: int = 5,
        expert_hidden_dims: List[int] = [512, 256, 128],
        gate_input_dim: int = 32,
        gate_hidden_dims: List[int] = [64, 32],
        init_noise_std: float = 1.0,
        activation: str = "elu",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_experts = num_experts
        self.gate_input_dim = gate_input_dim

        # Expert MLPs: each maps obs → action_mean
        self.experts = nn.ModuleList([
            _build_mlp(obs_dim, expert_hidden_dims, action_dim, activation)
            for _ in range(num_experts)
        ])

        # Shared gating network (shared with critic, see MoECriticModel)
        self.gating = GatingNetwork(gate_input_dim, num_experts, gate_hidden_dims)

        # Learnable action noise std (shared across experts)
        self.log_std = nn.Parameter(torch.full((action_dim,), torch.log(torch.tensor(init_noise_std))))

        # Store last gate weights for contrastive learning (§III-E)
        self._last_gate_logits = None
        self._last_gate_weights = None

    def forward(self, obs: torch.Tensor, z_elevation: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass producing action mean via MoE weighted sum.

        Args:
            obs: [batch, obs_dim] policy observation (concatenated)
            z_elevation: [batch, gate_input_dim] AE-encoded elevation latent.
                         If None, uses uniform gating (1/N per expert).

        Returns:
            action_mean: [batch, action_dim]
        """
        batch_size = obs.shape[0]

        # Compute expert outputs: list of [batch, action_dim]
        expert_outputs = torch.stack([expert(obs) for expert in self.experts], dim=1)
        # expert_outputs: [batch, num_experts, action_dim]

        # Compute gate weights
        if z_elevation is not None:
            gate_logits = self.gating(z_elevation)  # [batch, num_experts]
            self._last_gate_logits = gate_logits
            gate_weights = F.softmax(gate_logits, dim=-1)  # [batch, num_experts]
        else:
            gate_weights = torch.ones(batch_size, self.num_experts, device=obs.device) / self.num_experts
            self._last_gate_logits = None

        self._last_gate_weights = gate_weights

        # Weighted sum: Eq. 6: μ_weighted = Σ softmax(g_i) * μ_i
        # gate_weights: [batch, num_experts, 1] for broadcasting
        action_mean = (gate_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)
        # action_mean: [batch, action_dim]

        return action_mean

    @property
    def std(self) -> torch.Tensor:
        # Clamp log_std to prevent exp() overflow/underflow -> NaN
        clamped = torch.clamp(self.log_std, min=-20.0, max=2.0)
        return torch.exp(clamped)

    def get_gate_weights(self) -> torch.Tensor | None:
        """Return last computed gate weights for contrastive learning."""
        return self._last_gate_weights

    def get_gate_logits(self) -> torch.Tensor | None:
        """Return last computed gate logits for contrastive learning."""
        return self._last_gate_logits


class MoECriticModel(nn.Module):
    """Mixture-of-Experts Critic following CMoE paper §III-D.

    Paper: "each expert module comprises a dedicated actor-critic pair,
    where each critic evaluates only its corresponding actor using privileged observation"
    "the same gating network is shared across both the actor and critic MoE components"

    Args:
        obs_dim: Total observation dimension for critic (privileged obs, concatenated)
        num_experts: Number of expert networks (must match actor)
        expert_hidden_dims: Hidden layer sizes for each expert MLP
        shared_gating: Shared GatingNetwork from MoEActorModel
        activation: Activation function name
    """

    def __init__(
        self,
        obs_dim: int,
        num_experts: int = 5,
        expert_hidden_dims: List[int] = [512, 256, 128],
        shared_gating: GatingNetwork | None = None,
        activation: str = "elu",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_experts = num_experts

        # Expert MLPs: each maps privileged_obs → value (scalar)
        self.experts = nn.ModuleList([
            _build_mlp(obs_dim, expert_hidden_dims, 1, activation)
            for _ in range(num_experts)
        ])

        # Use shared gating from actor (paper requirement)
        self.shared_gating = shared_gating

    def forward(self, obs: torch.Tensor, gate_weights: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass producing value via MoE weighted sum.

        Args:
            obs: [batch, obs_dim] privileged observation
            gate_weights: [batch, num_experts] from actor's gating network.
                          If None, uses uniform.

        Returns:
            value: [batch, 1]
        """
        batch_size = obs.shape[0]

        # Compute expert values
        expert_values = torch.stack([expert(obs) for expert in self.experts], dim=1)
        # expert_values: [batch, num_experts, 1]

        # Use gate weights from actor (shared gating)
        if gate_weights is not None:
            weights = gate_weights  # [batch, num_experts]
        else:
            weights = torch.ones(batch_size, self.num_experts, device=obs.device) / self.num_experts

        # Weighted sum
        value = (weights.unsqueeze(-1) * expert_values).sum(dim=1)
        # value: [batch, 1]

        return value
