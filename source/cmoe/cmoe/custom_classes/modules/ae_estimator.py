"""AutoEncoder for elevation map feature extraction (paper III-C, Eq. 5).

This module encodes the elevation map e_t into a latent z_E_t,
then decodes to reconstruct e_t for self-supervised training.

Loss: L_AE = MSE(e_pred, e_gt)  (paper Eq. 5)

The latent z_E_t is used as input to the gating network and
is concatenated into the system observation for the actor.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AEEstimator(nn.Module):
    """AutoEncoder for elevation map compression.

    Args:
        input_dim: Dimension of the flattened elevation map.
        latent_dim: Dimension of the latent space z_E_t.
        encoder_hidden_dims: Hidden layer sizes for the encoder.
        decoder_hidden_dims: Hidden layer sizes for the decoder.
        activation: Activation function name.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        encoder_hidden_dims: list[int] | tuple[int, ...] = (128, 64),
        decoder_hidden_dims: list[int] | tuple[int, ...] = (64, 128),
        activation: str = "elu",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        act_fn = _get_activation(activation)

        # --- Encoder ---
        encoder_layers = []
        in_dim = input_dim
        for h_dim in encoder_hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            encoder_layers.append(act_fn)
            in_dim = h_dim
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # --- Decoder ---
        decoder_layers = []
        in_dim = latent_dim
        for h_dim in decoder_hidden_dims:
            decoder_layers.append(nn.Linear(in_dim, h_dim))
            decoder_layers.append(act_fn)
            in_dim = h_dim
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, elevation_map: torch.Tensor) -> torch.Tensor:
        """Encode elevation map to latent z_E_t.

        Args:
            elevation_map: [N, input_dim] flattened elevation map.

        Returns:
            z_E: [N, latent_dim] latent representation.
        """
        return self.encoder(elevation_map)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to reconstruct elevation map.

        Args:
            z: [N, latent_dim].

        Returns:
            Reconstructed elevation map [N, input_dim].
        """
        return self.decoder(z)

    def forward(self, elevation_map: torch.Tensor):
        """Full forward: encode -> decode.

        Returns: z_E, recon
        """
        z = self.encode(elevation_map)
        recon = self.decode(z)
        return z, recon

    def compute_loss(
        self,
        elevation_map: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute AE reconstruction loss (paper Eq. 5).

        Args:
            elevation_map: [N, input_dim] ground-truth elevation map.

        Returns:
            loss, loss_dict
        """
        z, recon = self.forward(elevation_map)
        loss = nn.functional.mse_loss(recon, elevation_map)
        loss_dict = {
            "ae/reconstruction_loss": loss.item(),
        }
        return loss, loss_dict

    def get_latent(self, elevation_map: torch.Tensor) -> torch.Tensor:
        """Inference-only: get z_E without decoder.

        Args:
            elevation_map: [N, input_dim].

        Returns:
            z_E: [N, latent_dim].
        """
        with torch.no_grad():
            return self.encode(elevation_map)


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
