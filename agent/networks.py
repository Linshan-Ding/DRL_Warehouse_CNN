"""Policy and value networks (manuscript Section 4.2 and 4.5).

The encoder is the four-block convolutional network of Fig. 5: the state tensor
is treated as a small image over the picking-point grid, convolved, and pooled
into a single feature vector.  Because the pooling is adaptive, the *encoder* is
independent of the grid size -- but the actor head is not: its output dimension
is ``|A| = K*N_w*N_l + R*(N_w*N_l + 1)``.  A trained policy is therefore specific
to one (N_w, N_l, K, R) configuration and cannot be transferred to another one.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from configs.config import AlgoCfg, EnvCfg


def _group_norm(num_channels: int) -> nn.Module:
    for groups in (32, 16, 8, 4, 2, 1):
        if num_channels % groups == 0:
            return nn.GroupNorm(groups, num_channels)
    return nn.GroupNorm(1, num_channels)


class CNNFeatureExtractor(nn.Module):
    """(B, C, N_w, N_l) -> (B, feature_dim).

    GroupNorm rather than BatchNorm because the event-driven simulator produces
    single-state batches during rollout.
    """

    def __init__(self, input_channels: int, feature_dim: int):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1), nn.ReLU(), _group_norm(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), _group_norm(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(), _group_norm(256),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(), _group_norm(512),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, feature_dim), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.flatten(1)
        return self.fc_layers(x)


class PolicyNetwork(nn.Module):
    """Actor: one logit per action of the composite action space."""

    def __init__(self, env_cfg: EnvCfg, algo_cfg: AlgoCfg):
        super().__init__()
        self.cnn = CNNFeatureExtractor(env_cfg.n_state_channels, algo_cfg.cnn_output_dim)
        hidden = algo_cfg.policy_hidden
        self.mlp = nn.Sequential(
            nn.Linear(algo_cfg.cnn_output_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, env_cfg.n_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.cnn(state))


class ValueNetwork(nn.Module):
    """Critic: state value.  Separate encoder, weights are not shared."""

    def __init__(self, env_cfg: EnvCfg, algo_cfg: AlgoCfg):
        super().__init__()
        self.cnn = CNNFeatureExtractor(env_cfg.n_state_channels, algo_cfg.cnn_output_dim)
        self.value_head = nn.Sequential(
            nn.Linear(algo_cfg.cnn_output_dim, algo_cfg.value_hidden), nn.ReLU(),
            nn.Linear(algo_cfg.value_hidden, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.cnn(state)).squeeze(-1)


def count_parameters(*modules: nn.Module) -> int:
    return int(sum(p.numel() for module in modules for p in module.parameters()))
