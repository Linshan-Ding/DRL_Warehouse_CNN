"""SAPPO: spatially-aware PPO for human-robot collaborative order picking.

The algorithm is the clipped-surrogate PPO of the manuscript (Algorithm 1):
GAE-lambda advantages, a clipped policy objective with an additional cap on the
log-ratio, a clipped value loss and an entropy bonus.  Hyperparameters come from
``configs/algo.yaml`` and reproduce Table 4.

One implementation detail differs from the version that produced the submitted
results: the update evaluates a whole minibatch in a single forward pass instead
of looping over transitions one by one.  The objective is unchanged -- this is
purely how the same computation is executed.
"""
from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np
import torch
import torch.optim as optim

from agent.base import Agent
from agent.buffer import RolloutBuffer
from agent.networks import PolicyNetwork, ValueNetwork, count_parameters
from configs.config import AlgoCfg, EnvCfg

MASK_FILL = -1e9


class SAPPOAgent(Agent):
    name = "SAPPO"

    def __init__(self, env_cfg: EnvCfg, algo_cfg: AlgoCfg, device: str = "cpu"):
        self.env_cfg = env_cfg
        self.cfg = algo_cfg
        self.device = device
        self.n_actions = env_cfg.n_actions

        self.policy_net = PolicyNetwork(env_cfg, algo_cfg).to(device)
        self.value_net = ValueNetwork(env_cfg, algo_cfg).to(device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=algo_cfg.actor_lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=algo_cfg.critic_lr)

        self.buffer = RolloutBuffer()

    # ------------------------------------------------------------------ #
    @property
    def n_parameters(self) -> int:
        return count_parameters(self.policy_net, self.value_net)

    def _state_tensor(self, state: np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device)
        return tensor.unsqueeze(0) if tensor.dim() == 3 else tensor

    def _masked_logits(self, logits: torch.Tensor, legal: Sequence[int]) -> torch.Tensor:
        invalid = torch.ones_like(logits, dtype=torch.bool)
        invalid[:, list(legal)] = False
        return logits.masked_fill(invalid, MASK_FILL)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def act(self, env, state: np.ndarray) -> int:
        """Sample a feasible action and record the transition in the buffer."""
        legal = env.legal_actions()
        if not legal:
            raise RuntimeError("no feasible action at this decision epoch")

        tensor = self._state_tensor(state)
        value = self.value_net(tensor)
        logits = self._masked_logits(self.policy_net(tensor), legal)
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample()

        self.buffer.add_decision(np.asarray(state, dtype=np.float32),
                                 int(action.item()),
                                 float(distribution.log_prob(action).item()),
                                 float(value.item()), legal)
        return int(action.item())

    @torch.no_grad()
    def act_greedy(self, env, state: np.ndarray) -> int:
        legal = env.legal_actions()
        if not legal:
            raise RuntimeError("no feasible action at this decision epoch")
        logits = self._masked_logits(self.policy_net(self._state_tensor(state)), legal)
        return int(torch.argmax(logits, dim=1).item())

    def observe(self, reward: float, done: bool) -> None:
        self.buffer.add_outcome(reward, done)

    # ------------------------------------------------------------------ #
    def update(self) -> Dict[str, float]:
        """One PPO update over the collected rollout."""
        if len(self.buffer) == 0:
            return {}

        batch = self.buffer.tensors(self.device, self.cfg.gamma, self.cfg.gae_lambda,
                                    self.cfg.advantage_clip)
        log_ratio_cap = math.log(self.cfg.ratio_cap)

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                 "approx_kl": 0.0, "clip_fraction": 0.0}
        n_batches = 0

        for _ in range(self.cfg.ppo_epochs):
            for index in self.buffer.minibatches(self.cfg.minibatch_size):
                idx = torch.as_tensor(index, dtype=torch.long, device=self.device)
                states = batch["states"][idx]
                actions = batch["actions"][idx]
                old_log_probs = batch["log_probs"][idx]
                old_values = batch["values"][idx]
                returns = batch["returns"][idx]
                advantages = batch["advantages"][idx]
                invalid = self.buffer.mask_tensor(index, self.n_actions, self.device)

                logits = self.policy_net(states)
                if not torch.isfinite(logits).all():
                    raise RuntimeError("policy network produced non-finite logits")
                distribution = torch.distributions.Categorical(
                    logits=logits.masked_fill(invalid, MASK_FILL))

                new_log_probs = distribution.log_prob(actions)
                entropy = distribution.entropy().mean()

                log_ratio = torch.clamp(new_log_probs - old_log_probs,
                                        -log_ratio_cap, log_ratio_cap)
                ratio = torch.exp(log_ratio)
                surrogate = torch.min(
                    ratio * advantages,
                    torch.clamp(ratio, 1.0 - self.cfg.clip_eps,
                                1.0 + self.cfg.clip_eps) * advantages)
                policy_loss = -surrogate.mean()

                value_pred = self.value_net(states)
                value_clipped = old_values + torch.clamp(value_pred - old_values,
                                                         -self.cfg.clip_eps,
                                                         self.cfg.clip_eps)
                value_loss = torch.max((value_pred - returns).pow(2),
                                       (value_clipped - returns).pow(2)).mean()
                value_loss = torch.clamp(value_loss, max=1e6)

                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy

                self.policy_optimizer.zero_grad(set_to_none=True)
                self.value_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(),
                                               self.cfg.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(),
                                               self.cfg.max_grad_norm)
                self.policy_optimizer.step()
                self.value_optimizer.step()

                with torch.no_grad():
                    clipped = ((ratio < 1.0 - self.cfg.clip_eps)
                               | (ratio > 1.0 + self.cfg.clip_eps)).float().mean()
                    stats["policy_loss"] += float(policy_loss.item())
                    stats["value_loss"] += float(value_loss.item())
                    stats["entropy"] += float(entropy.item())
                    stats["approx_kl"] += float((old_log_probs - new_log_probs).mean().item())
                    stats["clip_fraction"] += float(clipped.item())
                n_batches += 1

        self.buffer.clear()
        if n_batches:
            stats = {key: value / n_batches for key, value in stats.items()}
        return stats

    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        torch.save({"policy": self.policy_net.state_dict(),
                    "value": self.value_net.state_dict(),
                    "env_cfg": self.env_cfg.__dict__,
                    "n_actions": self.n_actions}, path)

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if payload.get("n_actions") != self.n_actions:
            raise ValueError(
                f"checkpoint was trained for |A| = {payload.get('n_actions')} but the "
                f"current configuration needs |A| = {self.n_actions}. The actor head "
                "depends on (N_w, N_l, K, R), so a policy cannot be reused across "
                "configurations -- train this configuration from scratch.")
        self.policy_net.load_state_dict(payload["policy"])
        self.value_net.load_state_dict(payload["value"])

    def eval_mode(self) -> None:
        self.policy_net.eval()
        self.value_net.eval()

    def train_mode(self) -> None:
        self.policy_net.train()
        self.value_net.train()
