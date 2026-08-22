"""Abstract agent interface.

Everything that solves the picking problem -- SAPPO, a dispatching rule, a
future RL baseline -- speaks this interface, so ``eval.py`` can treat them
uniformly.  Only the flat action index crosses the boundary.
"""
from __future__ import annotations

import abc

import numpy as np


class Agent(abc.ABC):
    name: str = "agent"

    @abc.abstractmethod
    def act(self, env, state: np.ndarray) -> int:
        """Action index used while collecting experience (stochastic)."""

    def act_greedy(self, env, state: np.ndarray) -> int:
        """Action index used for evaluation (deterministic)."""
        return self.act(env, state)

    def save(self, path: str) -> None:  # pragma: no cover - optional for rule agents
        raise NotImplementedError

    def load(self, path: str) -> None:  # pragma: no cover - optional for rule agents
        raise NotImplementedError
