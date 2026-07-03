from __future__ import annotations

import torch
from torch import nn


class ConvBranch1D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_dim: int = 64,
        dropout: float = 0.25,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),

            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class ScalarBranch(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        out_dim: int = 16,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.1),

            nn.Linear(16, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class MultimodalDeepClassifier(nn.Module):
    """
    Multimodal CNN for bacterial particle classification.

    Inputs
    ------
    spectrometer : (N,1,L)
    lifetime     : (N,1,L)
    scattering   : (N,1,L)
    scalar        : (N,2)
    """

    def __init__(
        self,
        n_classes: int,
        branch_dim: int = 64,
        scalar_dim: int = 16,
        dropout: float = 0.25,
    ):
        super().__init__()

        self.spectrometer_branch = ConvBranch1D(
            out_dim=branch_dim,
            dropout=dropout,
        )

        self.lifetime_branch = ConvBranch1D(
            out_dim=branch_dim,
            dropout=dropout,
        )

        self.scattering_branch = ConvBranch1D(
            out_dim=branch_dim,
            dropout=dropout,
        )

        self.scalar_branch = ScalarBranch(
            input_dim=2,
            out_dim=scalar_dim,
        )

        fusion_dim = branch_dim * 3 + scalar_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.35),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(64, n_classes),
        )

    def forward(self, batch):

        z_spec = self.spectrometer_branch(batch["spectrometer"])
        z_life = self.lifetime_branch(batch["lifetime"])
        z_scat = self.scattering_branch(batch["scattering"])
        z_scalar = self.scalar_branch(batch["scalar"])

        z = torch.cat(
            [
                z_spec,
                z_life,
                z_scat,
                z_scalar,
            ],
            dim=1,
        )

        return self.classifier(z)