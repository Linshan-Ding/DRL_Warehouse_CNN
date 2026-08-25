"""Networks shared by every learning algorithm in this project.

The encoder is the four-block convolutional network of Fig. 5, applied to the
state tensor (4 spatial channels + 5 configuration planes).  The action head is
sized for the resource *envelope* (k_max, r_max), so one set of parameters
serves every (K, R, C, tau, layout) scenario -- resources that do not exist in
the current scenario are simply never feasible.  Only the grid size (N_w, N_l)
changes the input geometry and therefore requires training from scratch.

``QNetwork`` reuses the same trunk with |A| outputs interpreted as Q-values, so
the value-based baselines see exactly the same observation and action space as
the policy-gradient methods.
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


class QNetwork(nn.Module):
    """Q-value head over the envelope action space (DQN-family baselines)."""

    def __init__(self, env_cfg: EnvCfg, algo_cfg):
        super().__init__()
        self.cnn = CNNFeatureExtractor(env_cfg.n_state_channels, algo_cfg.cnn_output_dim)
        hidden = algo_cfg.policy_hidden
        self.head = nn.Sequential(
            nn.Linear(algo_cfg.cnn_output_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, env_cfg.n_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.head(self.cnn(state))


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
