"""Actor-critic baselines: SOA+A2C (A2C, n = 200) and DRLG (n = 20).

Base advantage actor-critic on the shared observation / envelope-head / mask
contract: n-step bootstrapped returns, a single gradient pass per collection
round (no PPO ratio, no clipping), entropy regularisation.  The two methods of
the manuscript differ here exactly as Table 6 describes them -- the rollout
horizon: 200 steps for SOA+A2C, 20 steps for DRLG (the multi-worker collection
is provided by the shared parallel collector).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.optim as optim

from agent.base import MASK_FILL, Episode, LearningAgent, check_compatibility
from agent.networks import PolicyNetwork, ValueNetwork, count_parameters
from configs.config import AlgoCfg, BaselineCfg, EnvCfg


def n_step_returns(rewards, values, dones, gamma: float, n: int) -> np.ndarray:
    """Bootstrapped n-step returns computed backward over one episode."""
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float32)
    for start in range(0, T, n):
        end = min(start + n, T)
        bootstrap = 0.0 if (end == T or dones[end - 1]) else values[end]
        running = bootstrap
        for t in range(end - 1, start - 1, -1):
            running = rewards[t] + gamma * running * (0.0 if dones[t] else 1.0)
            returns[t] = running
    return returns


class ActorCriticAgent(LearningAgent):
    needs_log_probs = True

    def __init__(self, env_cfg: EnvCfg, algo_cfg: AlgoCfg, baseline_cfg: BaselineCfg,
                 device: str = "cpu", n_step: int = 200, name: str = "SOA+A2C"):
        super().__init__()
        self.name = name
        self.env_cfg = env_cfg
        self.net_cfg = algo_cfg
        self.cfg = baseline_cfg
        self.device = device
        self.n_step = n_step
        self.n_actions = env_cfg.n_actions

        self.policy_net = PolicyNetwork(env_cfg, algo_cfg).to(device)
        self.value_net = ValueNetwork(env_cfg, algo_cfg).to(device)
        self.optimizer = optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            lr=baseline_cfg.lr)

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
        states, actions, returns, advantages = [], [], [], []
        legal_lists = []
        for episode in episodes:
            ep_returns = n_step_returns(episode.rewards, episode.values, episode.dones,
                                        self.cfg.gamma, self.n_step)
            ep_adv = ep_returns - np.asarray(episode.values, dtype=np.float32)
            states.append(np.asarray(episode.states, dtype=np.float32))
            actions.extend(episode.actions)
            returns.append(ep_returns)
            advantages.append(ep_adv)
            legal_lists.extend(episode.legals)
        states = torch.as_tensor(np.concatenate(states), device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        returns = torch.as_tensor(np.concatenate(returns), device=self.device)
        advantages = torch.as_tensor(np.concatenate(advantages), device=self.device)
        if advantages.std() > 0:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        order = np.random.permutation(len(actions))
        n_batches = 0
        for start in range(0, len(order), self.cfg.batch_size):
            idx_np = order[start:start + self.cfg.batch_size]
            idx = torch.as_tensor(idx_np, dtype=torch.long, device=self.device)
            invalid = torch.ones((len(idx_np), self.n_actions), dtype=torch.bool,
                                 device=self.device)
            rows, cols = [], []
            for row, i in enumerate(idx_np):
                legal = legal_lists[i]
                rows.extend([row] * len(legal)); cols.extend(legal)
            if rows:
                invalid[torch.as_tensor(rows, device=self.device),
                        torch.as_tensor(cols, device=self.device)] = False

            logits = self.policy_net(states[idx]).masked_fill(invalid, MASK_FILL)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions[idx])
            entropy = dist.entropy().mean()
            policy_loss = -(log_probs * advantages[idx]).mean()
            value_loss = (self.value_net(states[idx]) - returns[idx]).pow(2).mean()
            loss = policy_loss + self.cfg.value_coef * value_loss \
                - self.cfg.entropy_coef * entropy

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.policy_net.parameters()) + list(self.value_net.parameters()),
                self.cfg.max_grad_norm)
            self.optimizer.step()
            stats["policy_loss"] += float(policy_loss)
            stats["value_loss"] += float(value_loss)
            stats["entropy"] += float(entropy)
            n_batches += 1
        self.episodes_trained += len(episodes)
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
        torch.save({"algo": self.name, "nets": self.actor_state(),
                    "meta": {"n_actions": self.n_actions,
                             "n_state_channels": self.env_cfg.n_state_channels,
                             "grid": [self.env_cfg.n_aisles, self.env_cfg.n_positions],
                             "envelope": [self.env_cfg.k_max, self.env_cfg.r_max],
                             "n_step": self.n_step,
                             "episodes_trained": self.episodes_trained}}, path)

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        check_compatibility(payload, self.env_cfg)
        self.policy_net.load_state_dict(payload["nets"]["policy"])
        self.value_net.load_state_dict(payload["nets"]["value"])
        self.episodes_trained = payload["meta"].get("episodes_trained", 0)
        self.policy_net.eval(); self.value_net.eval()
