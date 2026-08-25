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

import math

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


# --------------------------------------------------------------------------- #
# SAPPO-specific variants.  The baselines keep the vanilla classes above; the
# additions below only change SAPPO's optimisation behaviour, not the
# architecture (same trunk, same heads, same parameter count).

def _orthogonal_init(root: nn.Module, output_layer: nn.Linear, output_gain: float) -> None:
    """Orthogonal init: hidden layers gain sqrt(2), the output layer a custom
    gain (0.01 for the policy head keeps the initial policy near-uniform over
    the 5420-action envelope instead of committing to arbitrary logits)."""
    for module in root.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    nn.init.orthogonal_(output_layer.weight, gain=output_gain)
    nn.init.zeros_(output_layer.bias)


class SAPPOPolicyNetwork(PolicyNetwork):
    def __init__(self, env_cfg: EnvCfg, algo_cfg: AlgoCfg):
        super().__init__(env_cfg, algo_cfg)
        _orthogonal_init(self, self.mlp[-1], output_gain=0.01)


class SAPPOValueNetwork(ValueNetwork):
    """Critic that regresses *normalised* returns.

    Raw returns are simulation-seconds sums whose magnitude varies by an order
    of magnitude across sampled scenarios; regressing them directly keeps the
    critic gradient permanently saturated against the grad-norm clip.  The
    network therefore outputs values in a normalised space and carries the
    running return statistics as buffers -- they travel inside ``state_dict``,
    so the shared-memory broadcast to the collection workers needs no change.
    """

    STD_FLOOR = 1e-2

    def __init__(self, env_cfg: EnvCfg, algo_cfg: AlgoCfg):
        super().__init__(env_cfg, algo_cfg)
        _orthogonal_init(self, self.value_head[-1], output_gain=1.0)
        self.register_buffer("ret_mean", torch.zeros(1))
        self.register_buffer("ret_std", torch.ones(1))
        self.register_buffer("ret_count", torch.zeros(1))

    @torch.no_grad()
    def update_stats(self, returns: torch.Tensor, momentum: float) -> None:
        """Debiased EMA over per-round return statistics: the effective
        momentum ramps from 0 (first rounds average directly, no cold-start
        crawl away from the (0, 1) prior) up to ``momentum``."""
        mean = returns.mean().reshape(1).to(self.ret_mean.device)
        std = returns.std().clamp(min=self.STD_FLOOR).reshape(1).to(self.ret_mean.device)
        count = self.ret_count.item()
        m_eff = min(momentum, count / (count + 1.0))
        self.ret_mean.mul_(m_eff).add_(mean * (1.0 - m_eff))
        self.ret_std.mul_(m_eff).add_(std * (1.0 - m_eff))
        self.ret_count.add_(1.0)

    def normalize(self, raw: torch.Tensor) -> torch.Tensor:
        return (raw - self.ret_mean) / self.ret_std

    def denormalized(self, state: torch.Tensor) -> torch.Tensor:
        """Value prediction back in raw simulation-seconds (collection / GAE)."""
        return self.forward(state) * self.ret_std + self.ret_mean


def count_parameters(*modules: nn.Module) -> int:
    return int(sum(p.numel() for module in modules for p in module.parameters()))
