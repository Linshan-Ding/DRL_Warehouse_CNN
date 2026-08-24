"""SAPPO: spatially-aware PPO (the paper's method), on the shared contract.

Algorithm 1 of the manuscript: clipped-surrogate PPO with GAE-lambda
advantages, an additional cap on the log-ratio, a clipped value loss and an
entropy bonus.  Hyperparameters come from ``configs/train.yaml`` (section
``algo``) and reproduce Table 4.

Collection is on-policy: the workers act with the broadcast parameters and the
update happens strictly between collection rounds, so the synchronous parallel
collection preserves PPO's assumptions.
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import torch
import torch.optim as optim

from agent.base import MASK_FILL, Episode, LearningAgent, check_compatibility
from agent.buffer import RolloutBuffer
from agent.networks import PolicyNetwork, ValueNetwork, count_parameters
from configs.config import AlgoCfg, EnvCfg


class SAPPOAgent(LearningAgent):
    name = "SAPPO"
    needs_log_probs = True

    def __init__(self, env_cfg: EnvCfg, algo_cfg: AlgoCfg, device: str = "cpu"):
        super().__init__()
        self.env_cfg = env_cfg
        self.cfg = algo_cfg
        self.device = device
        self.n_actions = env_cfg.n_actions

        self.policy_net = PolicyNetwork(env_cfg, algo_cfg).to(device)
        self.value_net = ValueNetwork(env_cfg, algo_cfg).to(device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=algo_cfg.actor_lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=algo_cfg.critic_lr)
        self.buffer = RolloutBuffer()

    @property
    def n_parameters(self) -> int:
        return count_parameters(self.policy_net, self.value_net)

    # -- worker side ------------------------------------------------------- #
    def actor_state(self) -> Dict[str, dict]:
        return {"policy": {k: v.cpu() for k, v in self.policy_net.state_dict().items()},
                "value": {k: v.cpu() for k, v in self.value_net.state_dict().items()}}

    def load_actor_state(self, state: Dict[str, dict]) -> None:
        self.policy_net.load_state_dict(state["policy"])
        self.value_net.load_state_dict(state["value"])

    @torch.no_grad()
    def act_collect(self, env, state: np.ndarray, exploration: float):
        legal = env.legal_actions()
        tensor = self._tensor(state, self.device)
        logits = self._masked(self.policy_net(tensor), legal)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return (int(action.item()), float(dist.log_prob(action).item()),
                float(self.value_net(tensor).item()))

    # -- learner side ------------------------------------------------------ #
    def learn(self, episodes: List[Episode]) -> Dict[str, float]:
        self.buffer.clear()
        for episode in episodes:
            for t in range(len(episode)):
                self.buffer.add_decision(episode.states[t], episode.actions[t],
                                         episode.log_probs[t], episode.values[t],
                                         episode.legals[t])
                self.buffer.add_outcome(episode.rewards[t], episode.dones[t])
        stats = self._update()
        self.episodes_trained += len(episodes)
        return stats

    def _update(self) -> Dict[str, float]:
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
                dist = torch.distributions.Categorical(
                    logits=logits.masked_fill(invalid, MASK_FILL))
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                log_ratio = torch.clamp(new_log_probs - old_log_probs,
                                        -log_ratio_cap, log_ratio_cap)
                ratio = torch.exp(log_ratio)
                surrogate = torch.min(
                    ratio * advantages,
                    torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * advantages)
                policy_loss = -surrogate.mean()

                value_pred = self.value_net(states)
                value_clipped = old_values + torch.clamp(value_pred - old_values,
                                                         -self.cfg.clip_eps, self.cfg.clip_eps)
                value_loss = torch.max((value_pred - returns).pow(2),
                                       (value_clipped - returns).pow(2)).mean()
                value_loss = torch.clamp(value_loss, max=1e6)

                loss = policy_loss + self.cfg.value_coef * value_loss \
                    - self.cfg.entropy_coef * entropy
                self.policy_optimizer.zero_grad(set_to_none=True)
                self.value_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.cfg.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.cfg.max_grad_norm)
                self.policy_optimizer.step()
                self.value_optimizer.step()

                with torch.no_grad():
                    clipped = ((ratio < 1 - self.cfg.clip_eps)
                               | (ratio > 1 + self.cfg.clip_eps)).float().mean()
                    stats["policy_loss"] += float(policy_loss)
                    stats["value_loss"] += float(value_loss)
                    stats["entropy"] += float(entropy)
                    stats["approx_kl"] += float((old_log_probs - new_log_probs).mean())
                    stats["clip_fraction"] += float(clipped)
                n_batches += 1
        self.buffer.clear()
        return {k: v / max(1, n_batches) for k, v in stats.items()}

    # -- evaluation -------------------------------------------------------- #
    @torch.no_grad()
    def act_greedy(self, env, state: np.ndarray) -> int:
        logits = self._masked(self.policy_net(self._tensor(state, self.device)),
                              env.legal_actions())
        return int(torch.argmax(logits, dim=1).item())

    @torch.no_grad()
    def act_stochastic(self, env, state: np.ndarray) -> int:
        logits = self._masked(self.policy_net(self._tensor(state, self.device)),
                              env.legal_actions())
        return int(torch.distributions.Categorical(logits=logits).sample().item())

    # -- persistence ------------------------------------------------------- #
    def save(self, path: str) -> None:
        torch.save({"algo": self.name,
                    "nets": self.actor_state(),
                    "meta": {"n_actions": self.n_actions,
                             "n_state_channels": self.env_cfg.n_state_channels,
                             "grid": [self.env_cfg.n_aisles, self.env_cfg.n_positions],
                             "envelope": [self.env_cfg.k_max, self.env_cfg.r_max],
                             "gamma": self.cfg.gamma,
                             "episodes_trained": self.episodes_trained}}, path)

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        check_compatibility(payload, self.env_cfg)
        self.policy_net.load_state_dict(payload["nets"]["policy"])
        self.value_net.load_state_dict(payload["nets"]["value"])
        self.episodes_trained = payload["meta"].get("episodes_trained", 0)
        self.policy_net.eval(); self.value_net.eval()
