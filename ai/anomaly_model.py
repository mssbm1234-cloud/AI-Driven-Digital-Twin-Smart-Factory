"""PyTorch Deep Autoencoder architecture for Industrial Telemetry Anomaly Detection."""

import torch
import torch.nn as nn


class TelemetryAutoencoder(nn.Module):
    """Deep Autoencoder for detecting anomalies in industrial multi-sensor telemetry."""

    def __init__(self, input_dim: int = 6, latent_dim: int = 4):
        """Initialize autoencoder encoder and decoder layers.

        Args:
            input_dim: Number of input telemetry features (default 6: temp, vib, curr, rpm, press, power).
            latent_dim: Dimension of compressed bottleneck representation.
        """
        super(TelemetryAutoencoder, self).__init__()

        # Encoder network: Compresses high-dimensional physical dynamics
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 8),
            nn.LeakyReLU(0.1),
            nn.Linear(8, latent_dim),
        )

        # Decoder network: Reconstructs expected nominal telemetry
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.LeakyReLU(0.1),
            nn.Linear(8, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder and decoder."""
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

    def compute_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate per-sample Mean Squared Error (MSE) reconstruction loss.

        Args:
            x: Input tensor of shape (batch_size, input_dim) or (input_dim,).

        Returns:
            1D Tensor containing reconstruction error for each sample.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        with torch.no_grad():
            reconstructed = self.forward(x)
            # Per-feature squared error averaged across features
            error = torch.mean((x - reconstructed) ** 2, dim=-1)

        return error
