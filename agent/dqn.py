"""Value-based baselines: AG-DQN (DQN) and HSDDQN (Double DQN).

The cited methods' full designs (attention-guided heads, hybrid scheduling
architectures) are tied to their own problem structures; the comparison here
uses their base algorithms -- DQN and Double DQN -- adapted to this problem
through the shared observation, envelope action head and feasibility mask, as
stated in the manuscript.  Common settings follow Table 6: lr 1e-4, gamma 0.99,
batch 64, replay buffer, target sync every 2000 transitions.

Exploration: epsilon-greedy over the feasible set, annealed linearly per
episode; stochastic evaluation uses a small fixed epsilon (Table note).
"""
from __future__ import annotations

import random
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from agent.base import MASK_FILL, Episode, LearningAgent, check_compatibility
from agent.buffer import ReplayBuffer
from agent.networks import QNetwork, count_parameters
from configs.config import AlgoCfg, BaselineCfg, EnvCfg


class DQNAgent(LearningAgent):
    needs_log_probs = False

    def __init__(self, env_cfg: EnvCfg, algo_cfg: AlgoCfg, baseline_cfg: BaselineCfg,
                 device: str = "cpu", double: bool = False, name: str = "AG-DQN"):
        super().__init__()
        self.name = name
        self.env_cfg = env_cfg
        self.net_cfg = algo_cfg
        self.cfg = baseline_cfg
        self.device = device
        self.double = double
        self.n_actions = env_cfg.n_actions

        self.q_net = QNetwork(env_cfg, algo_cfg).to(device)
        self.target_net = QNetwork(env_cfg, algo_cfg).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=baseline_cfg.lr)
        self.replay = ReplayBuffer(baseline_cfg.replay_size)
        self.transitions_seen = 0
        self.gradient_steps = 0

    @property
    def n_parameters(self) -> int:
        return count_parameters(self.q_net)

    # -- exploration ------------------------------------------------------- #
    def exploration(self) -> float:
        frac = min(1.0, self.episodes_trained / max(1, self.cfg.epsilon_decay_episodes))
        return self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    # -- worker side ------------------------------------------------------- #
    def actor_state(self) -> Dict[str, dict]:
        return {"q": {k: v.cpu() for k, v in self.q_net.state_dict().items()}}

    def load_actor_state(self, state: Dict[str, dict]) -> None:
        self.q_net.load_state_dict(state["q"])

    @torch.no_grad()
    def act_collect(self, env, state: np.ndarray, exploration: float):
        legal = env.legal_actions()
        if random.random() < exploration:
            return random.choice(legal), None, None
        q = self._legal_subvector(self.q_net(self._tensor(state, self.device)),
                                  legal, f"{self.name}.act_collect")
        return int(legal[int(torch.argmax(q).item())]), None, None

    # -- learner side ------------------------------------------------------ #
    def learn(self, episodes: List[Episode]) -> Dict[str, float]:
        new_transitions = 0
        for episode in episodes:
            T = len(episode)
            for t in range(T):
                next_state = episode.states[t + 1] if t + 1 < T else episode.final_state
                next_legal = episode.legals[t + 1] if t + 1 < T else episode.final_legal
                self.replay.push(episode.states[t], episode.actions[t], episode.rewards[t],
                                 next_state, episode.dones[t], next_legal)
                new_transitions += 1
        self.transitions_seen += new_transitions
        self.episodes_trained += len(episodes)

        losses = []
        n_steps = max(1, new_transitions // self.cfg.dqn_train_freq)
        for _ in range(n_steps):
            if len(self.replay) < self.cfg.batch_size:
                break
            states, actions, rewards, next_states, dones, invalid_next = \
                self.replay.sample(self.cfg.batch_size, self.n_actions, self.device)
            q_pred = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                target_q = self.target_net(next_states).masked_fill(invalid_next, MASK_FILL)
                if self.double:
                    online_q = self.q_net(next_states).masked_fill(invalid_next, MASK_FILL)
                    best = online_q.argmax(dim=1, keepdim=True)
                    next_value = target_q.gather(1, best).squeeze(1)
                else:
                    next_value = target_q.max(dim=1).values
                next_value = torch.where(invalid_next.all(dim=1),
                                         torch.zeros_like(next_value), next_value)
                target = rewards + self.cfg.gamma * (1.0 - dones) * next_value
            loss = F.smooth_l1_loss(q_pred, target)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.cfg.max_grad_norm)
            self.optimizer.step()
            self.gradient_steps += 1
            losses.append(float(loss))
            if self.gradient_steps % max(1, self.cfg.target_update // self.cfg.dqn_train_freq) == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())
        return {"q_loss": float(np.mean(losses)) if losses else 0.0,
                "epsilon": self.exploration(),
                "replay_size": float(len(self.replay))}

    # -- evaluation -------------------------------------------------------- #
    @torch.no_grad()
    def act_greedy(self, env, state: np.ndarray) -> int:
        legal = env.legal_actions()
        q = self._legal_subvector(self.q_net(self._tensor(state, self.device)),
                                  legal, f"{self.name}.act_greedy")
        return int(legal[int(torch.argmax(q).item())])

    @torch.no_grad()
    def act_stochastic(self, env, state: np.ndarray) -> int:
        legal = env.legal_actions()
        if random.random() < self.cfg.eval_epsilon:
            return random.choice(legal)
        q = self._legal_subvector(self.q_net(self._tensor(state, self.device)),
                                  legal, f"{self.name}.act_stochastic")
        return int(legal[int(torch.argmax(q).item())])

    # -- persistence ------------------------------------------------------- #
    def save(self, path: str) -> None:
        torch.save({"algo": self.name, "nets": self.actor_state(),
                    "meta": {"n_actions": self.n_actions,
                             "n_state_channels": self.env_cfg.n_state_channels,
                             "grid": [self.env_cfg.n_aisles, self.env_cfg.n_positions],
                             "envelope": [self.env_cfg.k_max, self.env_cfg.r_max],
                             "double": self.double,
                             "episodes_trained": self.episodes_trained}}, path)

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        check_compatibility(payload, self.env_cfg)
        self.q_net.load_state_dict(payload["nets"]["q"])
        self.target_net.load_state_dict(payload["nets"]["q"])
        self.episodes_trained = payload["meta"].get("episodes_trained", 0)
        self.q_net.eval()
