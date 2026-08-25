"""On-policy rollout storage with GAE-lambda advantages."""
from __future__ import annotations

from typing import Dict, Iterator, List, Sequence

import numpy as np
import torch


class RolloutBuffer:
    """Stores one update's worth of transitions.

    The feasibility mask must be stored alongside the transition: the policy is
    defined on ``o_t = (s_t, M_t)``, so the update has to rebuild exactly the
    mask that was in force when the action was sampled.
    """

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.legal_actions: List[Sequence[int]] = []

    def __len__(self) -> int:
        return len(self.rewards)

    def add_decision(self, state: np.ndarray, action: int, log_prob: float,
                     value: float, legal_actions: Sequence[int]) -> None:
        self.states.append(state)
        self.actions.append(int(action))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.legal_actions.append(list(legal_actions))

    def add_outcome(self, reward: float, done: bool) -> None:
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    # ------------------------------------------------------------------ #
    def compute_gae(self, gamma: float, gae_lambda: float, advantage_clip: float,
                    last_value: float = 0.0):
        """Backward GAE-lambda recursion; advantages are standardised and clipped."""
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        gae = 0.0
        next_value = last_value
        for t in reversed(range(n)):
            not_done = 1.0 - float(self.dones[t])
            delta = self.rewards[t] + gamma * next_value * not_done - self.values[t]
            gae = delta + gamma * gae_lambda * not_done * gae
            advantages[t] = gae
            returns[t] = gae + self.values[t]
            next_value = self.values[t]

        if advantages.std() > 0:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantages = np.clip(advantages, -advantage_clip, advantage_clip)
        return returns, advantages

    # ------------------------------------------------------------------ #
    def tensors(self, device: str, gamma: float, gae_lambda: float,
                advantage_clip: float) -> Dict[str, torch.Tensor]:
        returns, advantages = self.compute_gae(gamma, gae_lambda, advantage_clip)
        return {
            "states": torch.as_tensor(np.asarray(self.states, dtype=np.float32), device=device),
            "actions": torch.as_tensor(self.actions, dtype=torch.long, device=device),
            "log_probs": torch.as_tensor(self.log_probs, dtype=torch.float32, device=device),
            "values": torch.as_tensor(self.values, dtype=torch.float32, device=device),
            "returns": torch.as_tensor(returns, dtype=torch.float32, device=device),
            "advantages": torch.as_tensor(advantages, dtype=torch.float32, device=device),
        }

    def mask_tensor(self, indices: Sequence[int], n_actions: int,
                    device: str) -> torch.Tensor:
        """Dense ``(len(indices), n_actions)`` boolean tensor: True marks infeasible."""
        invalid = torch.ones((len(indices), n_actions), dtype=torch.bool, device=device)
        rows, cols = [], []
        for row, t in enumerate(indices):
            legal = self.legal_actions[t]
            rows.extend([row] * len(legal))
            cols.extend(legal)
        if rows:
            invalid[torch.as_tensor(rows, dtype=torch.long, device=device),
                    torch.as_tensor(cols, dtype=torch.long, device=device)] = False
        return invalid

    def minibatches(self, batch_size: int) -> Iterator[np.ndarray]:
        order = np.random.permutation(len(self.rewards))
        for start in range(0, len(order), batch_size):
            yield order[start:start + batch_size]


class ReplayBuffer:
    """Uniform experience replay for the DQN-family baselines.

    States are stored as float16 to keep 100k transitions of a (9, 9, 20)
    tensor around 700 MB.  The feasibility mask of the *next* state is stored
    as an index list so that the bootstrap maximum only ranges over feasible
    actions.
    """

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.next_states: List[np.ndarray] = []
        self.dones: List[bool] = []
        self.next_legals: List[Sequence[int]] = []
        self._cursor = 0

    def __len__(self) -> int:
        return len(self.rewards)

    def push(self, state, action, reward, next_state, done, next_legal) -> None:
        entry = (np.asarray(state, dtype=np.float16), int(action), float(reward),
                 np.asarray(next_state, dtype=np.float16), bool(done), list(next_legal))
        if len(self.rewards) < self.capacity:
            self.states.append(entry[0]); self.actions.append(entry[1])
            self.rewards.append(entry[2]); self.next_states.append(entry[3])
            self.dones.append(entry[4]); self.next_legals.append(entry[5])
        else:
            i = self._cursor
            (self.states[i], self.actions[i], self.rewards[i],
             self.next_states[i], self.dones[i], self.next_legals[i]) = entry
            self._cursor = (i + 1) % self.capacity

    def sample(self, batch_size: int, n_actions: int, device: str):
        idx = np.random.randint(0, len(self.rewards), size=min(batch_size, len(self.rewards)))
        states = torch.as_tensor(
            np.stack([self.states[i] for i in idx]).astype(np.float32), device=device)
        next_states = torch.as_tensor(
            np.stack([self.next_states[i] for i in idx]).astype(np.float32), device=device)
        actions = torch.as_tensor([self.actions[i] for i in idx], dtype=torch.long, device=device)
        rewards = torch.as_tensor([self.rewards[i] for i in idx], dtype=torch.float32, device=device)
        dones = torch.as_tensor([float(self.dones[i]) for i in idx], device=device)
        invalid_next = torch.ones((len(idx), n_actions), dtype=torch.bool, device=device)
        rows, cols = [], []
        for row, i in enumerate(idx):
            legal = self.next_legals[i]
            rows.extend([row] * len(legal)); cols.extend(legal)
        if rows:
            invalid_next[torch.as_tensor(rows, device=device),
                         torch.as_tensor(cols, device=device)] = False
        return states, actions, rewards, next_states, dones, invalid_next
