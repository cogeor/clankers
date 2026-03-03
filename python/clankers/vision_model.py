"""Vision + proprioception policy network for imitation learning.

Architecture::

    image (B, C, H, W) ──→ CNN backbone ──→ img_features (B, F_img)
    positions (B, J)    ──→ MLP encoder  ──→ pos_features (B, F_pos)
                                                │
                          concat([img, pos]) ────┘
                                 │
                          Fusion MLP ──→ velocity (B, J)

The CNN backbone uses a lightweight stack of convolutions with
``AdaptiveAvgPool2d(1)`` so it accepts any spatial resolution.

Requires ``torch>=2.0.0``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VisionPolicyNet(nn.Module):
    """Multi-input policy: camera image + joint positions → velocity.

    Parameters
    ----------
    image_channels : int
        Number of colour channels (3 for RGB, 4 for RGBA).
    joint_dim : int
        Number of joint positions in the input.
    velocity_dim : int
        Number of velocity outputs (typically same as ``joint_dim``).
    cnn_features : int
        Dimensionality of the CNN feature vector after pooling.
    pos_features : int
        Dimensionality of the position encoder output.
    hidden : int
        Hidden size of the fusion MLP.
    """

    def __init__(
        self,
        image_channels: int = 3,
        joint_dim: int = 6,
        velocity_dim: int = 6,
        cnn_features: int = 128,
        pos_features: int = 64,
        hidden: int = 128,
    ) -> None:
        super().__init__()

        self.joint_dim = joint_dim
        self.velocity_dim = velocity_dim

        # CNN backbone: resolution-agnostic via AdaptiveAvgPool2d
        self.cnn = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, cnn_features, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # Position encoder
        self.pos_encoder = nn.Sequential(
            nn.Linear(joint_dim, pos_features),
            nn.ReLU(),
        )

        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(cnn_features + pos_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, velocity_dim),
        )

    def forward(self, image: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        image : Tensor
            ``(B, C, H, W)`` float32 in ``[0, 1]``.
        positions : Tensor
            ``(B, J)`` float32 joint positions in radians.

        Returns
        -------
        Tensor
            ``(B, V)`` float32 predicted joint velocities in rad/s.
        """
        img_features = self.cnn(image)
        pos_features = self.pos_encoder(positions)
        fused = torch.cat([img_features, pos_features], dim=1)
        return self.fusion(fused)
