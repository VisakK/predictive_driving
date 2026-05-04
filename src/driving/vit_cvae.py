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

    def forward(self, grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            grid: (batch, C, H, W)
        Returns:
            cls: (batch, embed_dim) CLS token embedding
            patches: (batch, H*W, embed_dim) patch token embeddings
        """
        B = grid.shape[0]
        x = grid.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x = self.patch_embed(x)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed

        x = self.encoder(x)
        x = self.norm(x)
        return x[:, 0], x[:, 1:]


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


class KinematicsHistoryEncoder(nn.Module):
    """MLP encoder for (N_agents, N_frames, feat_dim) per-agent temporal kinematics."""

    def __init__(
        self,
        n_agents: int,
        n_frames: int,
        feat_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 64,
    ):
        super().__init__()
        in_dim = n_agents * n_frames * feat_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, kin_history: torch.Tensor) -> torch.Tensor:
        return self.net(kin_history.flatten(1))


class AgentAnomalyEncoder(nn.Module):
    """MLP encoder for per-agent causal anomaly features."""

    def __init__(
        self,
        n_agents: int,
        anomaly_feat_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 32,
    ):
        super().__init__()
        in_dim = n_agents * anomaly_feat_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, anomaly_features: torch.Tensor) -> torch.Tensor:
        return self.net(anomaly_features.flatten(1))


class OnlineKinematicsPredictor(nn.Module):
    """Predicts current agent kinematics from prior history frames."""

    def __init__(
        self,
        n_agents: int,
        n_history_frames: int,
        agent_feat_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.agent_feat_dim = agent_feat_dim
        self.net = nn.Sequential(
            nn.Linear(n_agents * (n_history_frames - 1) * agent_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_agents * agent_feat_dim),
        )

    def forward(self, kin_history: torch.Tensor) -> torch.Tensor:
        prior = kin_history[:, :, :-1, :]
        pred = self.net(prior.flatten(1))
        return pred.reshape(-1, self.n_agents, self.agent_feat_dim)

    def loss(self, kin_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pred = self.forward(kin_history)
        target = kin_history[:, :, -1, :]
        presence = target[:, :, 0:1].clamp(min=0)
        err = F.mse_loss(pred, target, reduction="none")
        loss = (err * presence).sum() / presence.sum().clamp(min=1)
        return loss, pred

    @staticmethod
    def anomaly_from_prediction(
        pred: torch.Tensor,
        current: torch.Tensor,
    ) -> torch.Tensor:
        presence = current[:, :, 0]
        pos_err = torch.linalg.norm(current[:, :, 1:3] - pred[:, :, 1:3], dim=-1)
        vel_err = torch.linalg.norm(current[:, :, 3:5] - pred[:, :, 3:5], dim=-1)
        heading_err = torch.linalg.norm(current[:, :, 5:7] - pred[:, :, 5:7], dim=-1)
        raw = pos_err + 0.5 * vel_err + 0.25 * heading_err
        anomaly = torch.clamp(raw / 0.35, 0.0, 1.0) * presence
        distance = torch.sqrt(current[:, :, 1] ** 2 + current[:, :, 2] ** 2 + 1e-8)
        proximity = torch.clamp(1.0 - distance, min=0.0, max=1.0)
        closing = torch.clamp(-current[:, :, 3], min=0.0, max=1.0)
        risk = anomaly * (0.5 + 0.5 * proximity) * (0.5 + 0.5 * closing)
        return torch.stack([presence, anomaly, risk, torch.clamp(raw, 0.0, 1.0)], dim=-1)


class _RiskTemperatureCrossAttn(nn.Module):
    """Multi-head cross-attention with per-query multiplicative scaling on
    attention logits. Used to inject per-agent risk/anomaly directly into
    the attention dynamics so the gradient cannot route around the anomaly
    channels (053 design)."""

    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(
        self,
        q: torch.Tensor,           # (B, N, D)
        kv: torch.Tensor,          # (B, S, D)
        attn_bias: torch.Tensor | None = None,   # (B, N, S) additive
        q_scale: torch.Tensor | None = None,     # (B, N) multiplicative on logits
    ) -> torch.Tensor:
        B, N, D = q.shape
        S = kv.shape[1]
        H = self.n_heads
        Dh = self.head_dim

        Q = self.q_proj(q).reshape(B, N, H, Dh).transpose(1, 2)
        K = self.k_proj(kv).reshape(B, S, H, Dh).transpose(1, 2)
        V = self.v_proj(kv).reshape(B, S, H, Dh).transpose(1, 2)

        logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Dh)  # (B, H, N, S)
        if attn_bias is not None:
            logits = logits + attn_bias.unsqueeze(1)
        if q_scale is not None:
            logits = logits * q_scale.unsqueeze(1).unsqueeze(-1)

        attn = F.softmax(logits, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, V).transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)


class AnomalyAttentionEncoder(nn.Module):
    """Per-agent self-attention + cross-attention to ViT scene tokens.

    Replaces the flat-MLP `AgentAnomalyEncoder` for the H10 anomaly path.
    See AnomalyInputDesign.md for the full design.

    When ``use_risk_attention_bias`` is set, the cross-attention is replaced
    with a custom implementation that scales each agent-query's attention
    logits by ``temp_i = 1 + softplus(s_a) * anomaly_i + softplus(s_r) * risk_i``,
    forcing a gradient pathway through the anomaly channels (053 design).
    """

    def __init__(
        self,
        n_agents: int = 15,
        kin_feat_dim: int = 7,
        anomaly_feat_dim: int = 4,
        scene_dim: int = 64,
        embed_dim: int = 64,
        n_heads: int = 4,
        ffn_ratio: int = 2,
        dropout: float = 0.1,
        spatial_sigma: float = 1.5,
        grid_h: int = 11,
        grid_w: int = 11,
        use_risk_attention_bias: bool = False,
        use_per_slot_gru: bool = False,
        gru_hidden: int = 32,
        kin_history_feat_dim: int | None = None,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.embed_dim = embed_dim
        self.spatial_sigma = spatial_sigma
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.use_risk_attention_bias = use_risk_attention_bias
        self.use_per_slot_gru = use_per_slot_gru
        self.gru_hidden = gru_hidden

        token_in_dim = kin_feat_dim + anomaly_feat_dim
        if self.use_per_slot_gru:
            hist_feat_dim = (
                kin_history_feat_dim if kin_history_feat_dim is not None else kin_feat_dim
            )
            self.per_slot_gru = nn.GRU(
                input_size=hist_feat_dim,
                hidden_size=gru_hidden,
                num_layers=1,
                batch_first=True,
            )
            token_in_dim += gru_hidden
        self.token_proj = nn.Linear(token_in_dim, embed_dim)
        self.slot_pos_emb = nn.Parameter(torch.zeros(1, n_agents, embed_dim))

        self_attn_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * ffn_ratio,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.agent_self_attn = nn.TransformerEncoder(self_attn_layer, num_layers=1)

        self.scene_proj = (
            nn.Linear(scene_dim, embed_dim) if scene_dim != embed_dim else nn.Identity()
        )
        self.cross_pre_norm_q = nn.LayerNorm(embed_dim)
        self.cross_pre_norm_kv = nn.LayerNorm(embed_dim)
        if self.use_risk_attention_bias:
            self.agent_to_scene = _RiskTemperatureCrossAttn(
                embed_dim=embed_dim, n_heads=n_heads, dropout=dropout,
            )
            # softplus(0) = ln(2) ≈ 0.69 → max scale at init ≈ 1 + 0.69 + 0.69 = 2.38
            self.s_anomaly = nn.Parameter(torch.zeros(1))
            self.s_risk = nn.Parameter(torch.zeros(1))
        else:
            self.agent_to_scene = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True,
            )
        self.cross_ffn_norm = nn.LayerNorm(embed_dim)
        self.cross_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ffn_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * ffn_ratio, embed_dim),
        )

        # patch centers in normalized [-1, 1] for spatial bias.
        ys = torch.linspace(-1.0, 1.0, grid_h)
        xs = torch.linspace(-1.0, 1.0, grid_w)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        patch_xy = torch.stack([gx, gy], dim=-1).reshape(-1, 2)  # (H*W, 2)
        self.register_buffer("patch_xy", patch_xy, persistent=False)

        nn.init.trunc_normal_(self.slot_pos_emb, std=0.02)

    def _spatial_bias(self, agent_xy: torch.Tensor) -> torch.Tensor:
        """Log-Gaussian additive bias on attention logits. -inf for absent slots
        is applied later via key_padding_mask, not here."""
        # agent_xy: (B, N, 2) in normalized coords
        diff = agent_xy.unsqueeze(2) - self.patch_xy.unsqueeze(0).unsqueeze(0)
        d2 = diff.pow(2).sum(-1)  # (B, N, H*W)
        return -d2 / (2.0 * self.spatial_sigma ** 2)

    def forward(
        self,
        agent_kin: torch.Tensor,        # (B, N, kin_feat_dim)
        agent_anom: torch.Tensor,       # (B, N, anomaly_feat_dim)
        scene_patches: torch.Tensor,    # (B, H*W, scene_dim)
        agent_kin_history: torch.Tensor | None = None,  # (B, N, T, hist_feat_dim)
    ) -> torch.Tensor:
        B, N, _ = agent_kin.shape
        presence = agent_kin[:, :, 0]                  # (B, N)
        pad_mask = presence < 0.5                      # True = ignore

        token_parts = [agent_kin, agent_anom]
        if self.use_per_slot_gru:
            assert agent_kin_history is not None, (
                "AnomalyAttentionEncoder.use_per_slot_gru requires agent_kin_history"
            )
            T = agent_kin_history.shape[2]
            F_h = agent_kin_history.shape[3]
            flat = agent_kin_history.reshape(B * N, T, F_h)
            _, h = self.per_slot_gru(flat)             # (1, B*N, gru_hidden)
            temporal = h[-1].reshape(B, N, self.gru_hidden)
            token_parts.append(temporal)
        token_in = torch.cat(token_parts, dim=-1)
        agent_tokens = self.token_proj(token_in) + self.slot_pos_emb

        # Stage B — agent self-attention.
        agent_tokens = self.agent_self_attn(
            agent_tokens, src_key_padding_mask=pad_mask
        )

        # Stage C — agent-as-query cross-attention to ViT patches.
        agent_xy = agent_kin[:, :, 1:3]                # (B, N, 2)
        spatial_bias = self._spatial_bias(agent_xy)    # (B, N, H*W)

        q = self.cross_pre_norm_q(agent_tokens)
        kv = self.cross_pre_norm_kv(self.scene_proj(scene_patches))

        if self.use_risk_attention_bias:
            anomaly_scalar = agent_anom[:, :, 1]       # (B, N)
            risk_scalar = agent_anom[:, :, 2]          # (B, N)
            q_scale = (
                1.0
                + F.softplus(self.s_anomaly) * anomaly_scalar
                + F.softplus(self.s_risk) * risk_scalar
            )
            attn_out = self.agent_to_scene(
                q, kv, attn_bias=spatial_bias, q_scale=q_scale,
            )
        else:
            n_heads = self.agent_to_scene.num_heads
            attn_mask = (
                spatial_bias.unsqueeze(1)
                .expand(-1, n_heads, -1, -1)
                .reshape(B * n_heads, N, -1)
            )
            attn_out, _ = self.agent_to_scene(q, kv, kv, attn_mask=attn_mask)
        agent_tokens = agent_tokens + attn_out
        agent_tokens = agent_tokens + self.cross_ffn(self.cross_ffn_norm(agent_tokens))

        # Stage D — presence-weighted pool. Zero out padded slots first.
        weights = presence.unsqueeze(-1)               # (B, N, 1)
        pooled = (agent_tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        return pooled


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
        use_kinematics_policy: bool = False,
        n_history_frames: int = 10,
        kin_encoder_hidden: int = 128,
        kin_encoder_out: int = 64,
        use_anomaly_policy: bool = False,
        anomaly_encoder_hidden: int = 64,
        anomaly_encoder_out: int = 32,
        use_online_predictor: bool = False,
        use_learned_anomaly_policy: bool = False,
        predictor_hidden_dim: int = 128,
        use_anomaly_attention_policy: bool = False,
        anomaly_attn_embed_dim: int = 64,
        anomaly_attn_n_heads: int = 4,
        anomaly_attn_spatial_sigma: float = 1.5,
        anomaly_attn_use_risk_bias: bool = False,
        anomaly_attn_use_per_slot_gru: bool = False,
        anomaly_attn_gru_hidden: int = 32,
    ):
        self.use_kinematics_policy = use_kinematics_policy and (
            "agent_kin_history" in observation_space.spaces
        )
        self.use_anomaly_attention_policy = use_anomaly_attention_policy and (
            "agent_anomaly" in observation_space.spaces
            and "agent_kinematics" in observation_space.spaces
        )
        self.anomaly_attn_use_per_slot_gru = (
            anomaly_attn_use_per_slot_gru
            and self.use_anomaly_attention_policy
            and ("agent_kin_history" in observation_space.spaces)
        )
        if anomaly_attn_use_per_slot_gru and not self.anomaly_attn_use_per_slot_gru:
            raise ValueError(
                "anomaly_attn_use_per_slot_gru=True requires "
                "use_anomaly_attention_policy=True and 'agent_kin_history' in obs"
            )
        # Attention policy supersedes the flat-MLP anomaly path.
        self.use_anomaly_policy = (
            use_anomaly_policy
            and not self.use_anomaly_attention_policy
            and ("agent_anomaly" in observation_space.spaces)
        )
        self.use_online_predictor = use_online_predictor and (
            "agent_kin_history" in observation_space.spaces
        )
        self.use_learned_anomaly_policy = (
            use_learned_anomaly_policy
            and self.use_online_predictor
            and not self.use_anomaly_attention_policy
        )
        actual_features_dim = vit_embed_dim
        if self.use_kinematics_policy:
            actual_features_dim += kin_encoder_out
        if self.use_anomaly_policy or self.use_learned_anomaly_policy:
            actual_features_dim += anomaly_encoder_out
        if self.use_anomaly_attention_policy:
            actual_features_dim += anomaly_attn_embed_dim
        super().__init__(observation_space, actual_features_dim)

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

        if self.use_kinematics_policy:
            self.kin_encoder = KinematicsHistoryEncoder(
                n_agents=n_agents,
                n_frames=n_history_frames,
                feat_dim=agent_feat_dim,
                hidden_dim=kin_encoder_hidden,
                out_dim=kin_encoder_out,
            )

        if self.use_anomaly_policy or self.use_learned_anomaly_policy:
            if self.use_anomaly_policy:
                anomaly_feat_dim = observation_space["agent_anomaly"].shape[-1]
            else:
                anomaly_feat_dim = 4
            self.anomaly_encoder = AgentAnomalyEncoder(
                n_agents=n_agents,
                anomaly_feat_dim=anomaly_feat_dim,
                hidden_dim=anomaly_encoder_hidden,
                out_dim=anomaly_encoder_out,
            )

        if self.use_online_predictor:
            self.online_predictor = OnlineKinematicsPredictor(
                n_agents=n_agents,
                n_history_frames=n_history_frames,
                agent_feat_dim=agent_feat_dim,
                hidden_dim=predictor_hidden_dim,
            )

        if self.use_anomaly_attention_policy:
            anom_dim = observation_space["agent_anomaly"].shape[-1]
            kin_dim = observation_space["agent_kinematics"].shape[-1]
            hist_feat_dim = (
                observation_space["agent_kin_history"].shape[-1]
                if self.anomaly_attn_use_per_slot_gru
                else None
            )
            self.anomaly_attn_encoder = AnomalyAttentionEncoder(
                n_agents=n_agents,
                kin_feat_dim=kin_dim,
                anomaly_feat_dim=anom_dim,
                scene_dim=vit_embed_dim,
                embed_dim=anomaly_attn_embed_dim,
                n_heads=anomaly_attn_n_heads,
                spatial_sigma=anomaly_attn_spatial_sigma,
                grid_h=grid_h,
                grid_w=grid_w,
                use_risk_attention_bias=anomaly_attn_use_risk_bias,
                use_per_slot_gru=self.anomaly_attn_use_per_slot_gru,
                gru_hidden=anomaly_attn_gru_hidden,
                kin_history_feat_dim=hist_feat_dim,
            )

        self._cached_scene_embed: torch.Tensor | None = None

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        grid = observations["occupancy_grid"].float()
        scene_embed, scene_patches = self.vit(grid)
        self._cached_scene_embed = scene_embed
        features = [scene_embed]
        if self.use_kinematics_policy:
            kin_hist = observations["agent_kin_history"].float()
            kin_embed = self.kin_encoder(kin_hist)
            features.append(kin_embed)
        if self.use_anomaly_attention_policy:
            agent_kin = observations["agent_kinematics"].float()
            agent_anom = observations["agent_anomaly"].float()
            agent_kin_history = (
                observations["agent_kin_history"].float()
                if self.anomaly_attn_use_per_slot_gru
                else None
            )
            features.append(
                self.anomaly_attn_encoder(
                    agent_kin, agent_anom, scene_patches, agent_kin_history
                )
            )
        elif self.use_anomaly_policy:
            anomaly = observations["agent_anomaly"].float()
            features.append(self.anomaly_encoder(anomaly))
        elif self.use_learned_anomaly_policy:
            kin_hist = observations["agent_kin_history"].float()
            current = observations["agent_kinematics"].float()
            with torch.no_grad():
                pred = self.online_predictor(kin_hist)
                anomaly = self.online_predictor.anomaly_from_prediction(pred, current)
            features.append(self.anomaly_encoder(anomaly))
        return torch.cat(features, dim=1)

    def compute_auxiliary_losses(
        self, observations: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor | float]:
        scene_embed = self._cached_scene_embed
        if scene_embed is None:
            grid = observations["occupancy_grid"].float()
            scene_embed, _ = self.vit(grid)

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

    def compute_predictor_loss(
        self, observations: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor | float]:
        if not self.use_online_predictor:
            zero = torch.zeros((), device=observations["occupancy_grid"].device)
            return {"predictor_loss": zero, "predictor_mean_anomaly": 0.0}
        kin_hist = observations["agent_kin_history"].float()
        current = observations["agent_kinematics"].float()
        loss, pred = self.online_predictor.loss(kin_hist)
        with torch.no_grad():
            anomaly = self.online_predictor.anomaly_from_prediction(pred, current)
            presence = anomaly[:, :, 0]
            mean_anomaly = (
                anomaly[:, :, 1] * presence
            ).sum() / presence.sum().clamp(min=1)
        return {
            "predictor_loss": loss,
            "predictor_mean_anomaly": float(mean_anomaly.item()),
        }

    def compute_anomaly_scores(
        self, observations: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Per-env anomaly score for reward shaping (higher = more anomalous)."""
        if self.use_learned_anomaly_policy and "agent_kin_history" in observations:
            kin_hist = observations["agent_kin_history"].float()
            current = observations["agent_kinematics"].float()
            pred = self.online_predictor(kin_hist)
            anomaly_features = self.online_predictor.anomaly_from_prediction(
                pred, current
            )
            presence = anomaly_features[:, :, 0]
            risk = anomaly_features[:, :, 2]
            return (risk * presence).sum(1) / presence.sum(1).clamp(min=1)

        if "agent_anomaly" in observations:
            anomaly_features = observations["agent_anomaly"].float()
            presence = anomaly_features[:, :, 0]
            risk = anomaly_features[:, :, 2]
            return (risk * presence).sum(1) / presence.sum(1).clamp(min=1)

        grid = observations["occupancy_grid"].float()
        agent_kin = observations["agent_kinematics"].float()

        scene_embed, _ = self.vit(grid)
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
