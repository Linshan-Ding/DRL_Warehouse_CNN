"""Common contract of every learning algorithm in this project.

The training loop and the parallel collector only speak this interface, so all
five algorithms (SAPPO and the four base-algorithm baselines) are trained,
checkpointed and evaluated identically:

* worker side  -- ``load_actor_state`` + ``act_collect`` (CPU, exploration on);
* learner side -- ``learn(episodes)`` (GPU, batched);
* evaluation   -- ``act_greedy`` and ``act_stochastic`` (the 3-sample protocol);
* persistence  -- ``save``/``load`` of ONE parameter file per algorithm.
"""
from __future__ import annotations

import abc
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

MASK_FILL = -1e9


class Episode:
    """One collected episode, as returned by a worker."""

    __slots__ = ("states", "actions", "rewards", "dones", "legals",
                 "log_probs", "values", "final_state", "final_legal",
                 "scenario", "summary")

    def __init__(self, states, actions, rewards, dones, legals,
                 log_probs=None, values=None, final_state=None, final_legal=(),
                 scenario=None, summary=None):
        self.states = states                  # np.float32 [T, C, H, W]
        self.actions = actions                # list[int]
        self.rewards = rewards                # list[float]
        self.dones = dones                    # list[bool]
        self.legals = legals                  # list[list[int]]
        self.log_probs = log_probs            # list[float] | None (PG only)
        self.values = values                  # list[float] | None (PG only)
        self.final_state = final_state        # np.float32 [C, H, W]
        self.final_legal = list(final_legal)
        self.scenario = dict(scenario or {})
        self.summary = dict(summary or {})

    def __len__(self) -> int:
        return len(self.rewards)


class LearningAgent(abc.ABC):
    name: str = "agent"
    needs_log_probs: bool = False             # workers compute logprob/value for PG

    def __init__(self):
        self.episodes_trained = 0

    # -- worker side ------------------------------------------------------- #
    @abc.abstractmethod
    def actor_state(self) -> Dict[str, dict]:
        """CPU state_dicts the workers need for acting."""

    @abc.abstractmethod
    def load_actor_state(self, state: Dict[str, dict]) -> None:
        """Load broadcast parameters (worker side)."""

    @abc.abstractmethod
    def act_collect(self, env, state: np.ndarray, exploration: float):
        """Exploring action during collection -> (action, log_prob|None, value|None)."""

    def exploration(self) -> float:
        """Exploration parameter for the current amount of training (0 for PG)."""
        return 0.0

    # -- learner side ------------------------------------------------------ #
    @abc.abstractmethod
    def learn(self, episodes: List[Episode]) -> Dict[str, float]:
        """One update from freshly collected episodes; returns logging stats."""

    # -- evaluation -------------------------------------------------------- #
    @abc.abstractmethod
    def act_greedy(self, env, state: np.ndarray) -> int:
        """Deterministic action (checkpoint selection on validation streams)."""

    @abc.abstractmethod
    def act_stochastic(self, env, state: np.ndarray) -> int:
        """Stochastic action for the 3-sample evaluation protocol."""

    # -- persistence ------------------------------------------------------- #
    @abc.abstractmethod
    def save(self, path: str) -> None: ...

    @abc.abstractmethod
    def load(self, path: str) -> None: ...

    # -- shared helpers ---------------------------------------------------- #
    @staticmethod
    def _tensor(state: np.ndarray, device: str) -> torch.Tensor:
        tensor = torch.as_tensor(np.asarray(state, dtype=np.float32), device=device)
        return tensor.unsqueeze(0) if tensor.dim() == 3 else tensor

    @staticmethod
    def _masked(logits: torch.Tensor, legal: Sequence[int]) -> torch.Tensor:
        invalid = torch.ones_like(logits, dtype=torch.bool)
        invalid[:, list(legal)] = False
        return logits.masked_fill(invalid, MASK_FILL)


def check_compatibility(payload: dict, env_cfg) -> None:
    meta = payload.get("meta", {})
    if meta.get("n_actions") != env_cfg.n_actions or \
       meta.get("n_state_channels") != env_cfg.n_state_channels:
        raise ValueError(
            f"checkpoint was trained for |A|={meta.get('n_actions')} / "
            f"{meta.get('n_state_channels')} channels but the current configuration "
            f"needs |A|={env_cfg.n_actions} / {env_cfg.n_state_channels}. The head is "
            "sized for the resource envelope and the grid; a different grid trains "
            "from scratch.")
