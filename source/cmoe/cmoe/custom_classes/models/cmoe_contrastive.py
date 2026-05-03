"""CMoE Terrain Contrastive Learning (Paper §III-E, Eq. 7-8).

SwAV-inspired contrastive learning that:
1. Projects gate outputs g and elevation encodings e into shared dimension
2. Computes cluster assignments via prototypes + Sinkhorn-Knopp
3. Cross-predicts: gate predicts elevation's cluster, elevation predicts gate's cluster
4. Loss = -1/(2H) * sum(q_g * log(p_e) + q_e * log(p_g))  (Eq. 8)

Paper parameters (§IV-A): num_prototypes=32, temperature=0.2
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwAVContrastiveLoss(nn.Module):
    """SwAV-based contrastive learning between gate activations and elevation encodings.

    Paper §III-E: "we first adopt two MLP to transform the gate output and elevation map
    into the same dimension g_z_t and e_z_t"

    Args:
        gate_dim: Dimension of gate output (= num_experts, e.g. 5)
        elevation_dim: Dimension of AE-encoded elevation (= latent_dim, e.g. 32)
        projection_dim: Shared projection dimension for both modalities
        num_prototypes: Number of cluster prototypes K (paper: 32)
        temperature: Softmax temperature tau (paper: 0.2)
        sinkhorn_iters: Number of Sinkhorn-Knopp iterations
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
        self.sinkhorn_eps = 0.05  # Sinkhorn temperature for assignments

        # Projection MLPs: map gate and elevation into same dimension
        self.gate_projector = nn.Sequential(
            nn.Linear(gate_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.elevation_projector = nn.Sequential(
            nn.Linear(elevation_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )

        # Learnable prototypes: K prototypes of dimension projection_dim
        # Paper: "L2-normalization on the prototype to obtain normalized matrix E"
        self.prototypes = nn.Linear(projection_dim, num_prototypes, bias=False)

    @torch.no_grad()
    def _sinkhorn_knopp(self, scores: torch.Tensor) -> torch.Tensor:
        """Sinkhorn-Knopp algorithm for balanced cluster assignments.

        Paper §III-E: "the Sinkhorn-Knopp algorithm is applied to both encoders"
        Ensures non-trivial solutions by enforcing equipartition constraint.

        IMPORTANT: scores here are the RAW dot products (NOT already divided by tau).
        We divide by sinkhorn_eps internally.

        Args:
            scores: [batch, num_prototypes] raw dot products with prototypes

        Returns:
            q: [batch, num_prototypes] balanced assignment probabilities
        """
        # Numerical stability: subtract max before exp to prevent overflow
        # scores / eps can be very large, causing exp() -> Inf -> NaN
        scaled = scores / self.sinkhorn_eps
        scaled = scaled - scaled.max(dim=-1, keepdim=True).values  # log-sum-exp trick
        Q = torch.exp(scaled)
        Q = Q.T  # [K, batch]

        # Normalize
        sum_Q = Q.sum()
        if sum_Q == 0 or torch.isnan(sum_Q) or torch.isinf(sum_Q):
            # Fallback: uniform assignments
            B = Q.shape[1]
            return torch.ones(B, Q.shape[0], device=Q.device) / Q.shape[0]

        Q /= sum_Q

        K, B = Q.shape

        for _ in range(self.sinkhorn_iters):
            # Row normalization
            row_sum = Q.sum(dim=1, keepdim=True)
            row_sum = torch.clamp(row_sum, min=1e-8)
            Q /= row_sum
            Q /= K
            # Column normalization
            col_sum = Q.sum(dim=0, keepdim=True)
            col_sum = torch.clamp(col_sum, min=1e-8)
            Q /= col_sum
            Q /= B

        Q = Q.T  # [batch, K]
        Q *= B  # Scale back

        # Final safety check
        Q = torch.nan_to_num(Q, nan=1.0 / K, posinf=1.0, neginf=0.0)

        return Q

    def forward(
        self,
        gate_output: torch.Tensor,
        elevation_encoding: torch.Tensor,
    ) -> torch.Tensor:
        """Compute SwAV contrastive loss (Eq. 8).

        Args:
            gate_output: [batch, gate_dim] raw gate logits from MoE gating network
            elevation_encoding: [batch, elevation_dim] AE-encoded elevation latent z_E_t

        Returns:
            loss: scalar contrastive loss
        """
        # Project to shared dimension
        g_z = self.gate_projector(gate_output)  # [batch, projection_dim]
        e_z = self.elevation_projector(elevation_encoding)  # [batch, projection_dim]

        # L2-normalize projections
        g_z = F.normalize(g_z, dim=-1, eps=1e-8)
        e_z = F.normalize(e_z, dim=-1, eps=1e-8)

        # L2-normalize prototypes (paper: "L2-normalization on the prototype")
        with torch.no_grad():
            w = self.prototypes.weight.data.clone()
            w = F.normalize(w, dim=1, eps=1e-8)
            self.prototypes.weight.copy_(w)

        # Compute raw dot products with prototypes
        # After L2-normalization, dot products are in [-1, 1]
        raw_scores_g = self.prototypes(g_z)  # [batch, K], values in [-1, 1]
        raw_scores_e = self.prototypes(e_z)  # [batch, K], values in [-1, 1]

        # Predicted probabilities (Eq. 7): divide by temperature tau
        scores_g = raw_scores_g / self.temperature  # [batch, K]
        scores_e = raw_scores_e / self.temperature  # [batch, K]
        p_g = F.softmax(scores_g, dim=-1)  # [batch, K]
        p_e = F.softmax(scores_e, dim=-1)  # [batch, K]

        # Cluster assignments via Sinkhorn-Knopp (balanced, non-trivial)
        # Pass RAW scores (not divided by tau) to Sinkhorn — it has its own eps
        q_g = self._sinkhorn_knopp(raw_scores_g.detach())  # [batch, K]
        q_e = self._sinkhorn_knopp(raw_scores_e.detach())  # [batch, K]

        # Cross-entropy loss (Eq. 8):
        # J_SwAV = -1/(2H) * sum(q_g * log(p_e) + q_e * log(p_g))
        loss = -0.5 * (
            (q_g * torch.log(p_e + 1e-8)).sum(dim=-1).mean()
            + (q_e * torch.log(p_g + 1e-8)).sum(dim=-1).mean()
        )

        # Final NaN guard
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=gate_output.device, requires_grad=True)

        return loss
