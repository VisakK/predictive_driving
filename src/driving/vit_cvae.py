from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ViTEncoder(nn.Module):
    """Vision Transformer encoder for small occupancy grids."""

    def __init__(
        self,
        n_channels: int = 5,
        grid_h: int = 11,
        grid_w: int = 11,
        embed_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        mlp_ratio: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        n_patches = grid_h * grid_w
        self.patch_embed = nn.Linear(n_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid: (batch, C, H, W)
        Returns:
            (batch, embed_dim) CLS token embedding
        """
        B = grid.shape[0]
        x = grid.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x = self.patch_embed(x)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed

        x = self.encoder(x)
        return self.norm(x[:, 0])


class CVAE(nn.Module):
    """Conditional VAE that reconstructs agent kinematics from scene context."""

    def __init__(
        self,
        scene_dim: int,
        n_agents: int,
        agent_feat_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.agent_feat_dim = agent_feat_dim
        self.latent_dim = latent_dim
        kin_dim = n_agents * agent_feat_dim

        self.encoder = nn.Sequential(
            nn.Linear(scene_dim + kin_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + scene_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, kin_dim),
        )

    def encode(
        self, scene_embed: torch.Tensor, agent_kin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([scene_embed, agent_kin.flatten(1)], dim=1)
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(
        self, z: torch.Tensor, scene_embed: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([z, scene_embed], dim=1)
        return self.decoder(x).reshape(-1, self.n_agents, self.agent_feat_dim)

    def forward(
        self, scene_embed: torch.Tensor, agent_kin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(scene_embed, agent_kin)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, scene_embed)
        return recon, mu, logvar

    def sample(self, scene_embed: torch.Tensor) -> torch.Tensor:
        z = torch.randn(
            scene_embed.shape[0], self.latent_dim, device=scene_embed.device
        )
        return self.decode(z, scene_embed)

    def loss(
        self, scene_embed: torch.Tensor, agent_kin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon, mu, logvar = self.forward(scene_embed, agent_kin)
        presence = agent_kin[:, :, 0:1].clamp(min=0)
        recon_loss = (F.mse_loss(recon, agent_kin, reduction="none") * presence).sum() / presence.sum().clamp(min=1)
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean()
        return recon_loss + kl_loss, recon_loss, kl_loss


class Discriminator(nn.Module):
    """Classifies trajectories as real (observed) vs counterfactual (CVAE-sampled)."""

    def __init__(
        self,
        scene_dim: int,
        agent_feat_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(scene_dim + agent_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, scene_embed: torch.Tensor, agent_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            scene_embed: (B, scene_dim)
            agent_features: (B, N, feat_dim)
        Returns:
            (B, N) logits — positive = real
        """
        B, N, F = agent_features.shape
        scene_exp = scene_embed.unsqueeze(1).expand(-1, N, -1)
        x = torch.cat([scene_exp, agent_features], dim=-1)
        logits = self.net(x.reshape(B * N, -1))
        return logits.reshape(B, N)

    def loss(
        self,
        scene_embed: torch.Tensor,
        real_kin: torch.Tensor,
        fake_kin: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        presence = real_kin[:, :, 0]
        mask = presence > 0.5

        real_logits = self.forward(scene_embed, real_kin)
        fake_logits = self.forward(scene_embed, fake_kin.detach())

        real_loss = F.binary_cross_entropy_with_logits(
            real_logits, torch.ones_like(real_logits), reduction="none"
        )
        fake_loss = F.binary_cross_entropy_with_logits(
            fake_logits, torch.zeros_like(fake_logits), reduction="none"
        )

        total = mask.float()
        n_valid = total.sum().clamp(min=1)
        loss = ((real_loss + fake_loss) * total).sum() / n_valid

        with torch.no_grad():
            real_acc = ((real_logits > 0) & mask).float().sum()
            fake_acc = ((fake_logits <= 0) & mask).float().sum()
            acc = (real_acc + fake_acc) / (2 * n_valid)

        return loss, acc.item()


class ViTCVAEExtractor(BaseFeaturesExtractor):
    """SB3 feature extractor: ViT + CVAE + Discriminator for adversarial PPO."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 64,
        n_agents: int = 15,
        agent_feat_dim: int = 7,
        vit_embed_dim: int = 64,
        vit_n_heads: int = 4,
        vit_n_layers: int = 3,
        cvae_latent_dim: int = 32,
        cvae_hidden_dim: int = 128,
        disc_hidden_dim: int = 128,
    ):
        super().__init__(observation_space, features_dim)

        grid_space = observation_space["occupancy_grid"]
        n_channels, grid_h, grid_w = grid_space.shape

        self.vit = ViTEncoder(
            n_channels=n_channels,
            grid_h=grid_h,
            grid_w=grid_w,
            embed_dim=vit_embed_dim,
            n_heads=vit_n_heads,
            n_layers=vit_n_layers,
        )

        self.cvae = CVAE(
            scene_dim=vit_embed_dim,
            n_agents=n_agents,
            agent_feat_dim=agent_feat_dim,
            latent_dim=cvae_latent_dim,
            hidden_dim=cvae_hidden_dim,
        )

        self.discriminator = Discriminator(
            scene_dim=vit_embed_dim,
            agent_feat_dim=agent_feat_dim,
            hidden_dim=disc_hidden_dim,
        )

        self._cached_scene_embed: torch.Tensor | None = None

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        grid = observations["occupancy_grid"].float()
        scene_embed = self.vit(grid)
        self._cached_scene_embed = scene_embed
        return scene_embed

    def compute_auxiliary_losses(
        self, observations: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor | float]:
        scene_embed = self._cached_scene_embed
        if scene_embed is None:
            grid = observations["occupancy_grid"].float()
            scene_embed = self.vit(grid)

        agent_kin = observations["agent_kinematics"].float()

        cvae_loss, recon_loss, kl_loss = self.cvae.loss(scene_embed, agent_kin)

        with torch.no_grad():
            fake_kin = self.cvae.sample(scene_embed)
        disc_loss, disc_acc = self.discriminator.loss(
            scene_embed, agent_kin, fake_kin
        )

        return {
            "cvae_loss": cvae_loss,
            "cvae_recon_loss": recon_loss.item(),
            "cvae_kl_loss": kl_loss.item(),
            "disc_loss": disc_loss,
            "disc_accuracy": disc_acc,
        }

    def compute_anomaly_scores(
        self, observations: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Per-env anomaly score for reward shaping (higher = more anomalous)."""
        grid = observations["occupancy_grid"].float()
        agent_kin = observations["agent_kinematics"].float()

        scene_embed = self.vit(grid)
        disc_logits = self.discriminator(scene_embed, agent_kin)
        disc_probs = torch.sigmoid(disc_logits)
        anomaly = 1.0 - disc_probs

        presence = agent_kin[:, :, 0]
        distances = torch.sqrt(
            agent_kin[:, :, 1] ** 2 + agent_kin[:, :, 2] ** 2 + 1e-8
        )
        proximity = torch.clamp(1.0 - distances, min=0.0)
        weighted = (anomaly * presence * proximity).sum(1) / presence.sum(1).clamp(min=1)
        return weighted
