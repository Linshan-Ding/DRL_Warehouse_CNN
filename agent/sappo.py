"""SAPPO: spatially-aware PPO (the paper's method), on the shared contract.

Algorithm 1 of the manuscript: clipped-surrogate PPO with GAE-lambda
advantages, an additional cap on the log-ratio, a clipped value loss and an
entropy bonus.  Hyperparameters come from ``configs/train.yaml`` (section
``algo``) and reproduce Table 4 of the revised manuscript.

Global-policy training mixes scenarios whose return magnitudes differ by an
order of magnitude, so on top of the vanilla recipe this implementation adds
(and the revised Table 4 documents):

* the critic regresses *normalised* returns (running statistics live as
  buffers inside ``SAPPOValueNetwork`` and ride the shared-memory broadcast);
* advantages are standardised per episode, giving every sampled scenario an
  equal voice in the gradient (``algo.advantage_norm``);
* PPO epochs stop early once the mean approximate KL of an epoch exceeds
  ``algo.kl_target``;
* actor/critic learning rates and the entropy coefficient decay linearly to
  ``lr_end_factor`` x initial and ``entropy_coef_end`` over the episode budget.

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
from agent.networks import SAPPOPolicyNetwork, SAPPOValueNetwork, count_parameters
from configs.config import AlgoCfg, EnvCfg


class SAPPOAgent(LearningAgent):
    name = "SAPPO"
    needs_log_probs = True

    RET_MOMENTUM = 0.995   # EMA over per-round return statistics (~200-round window)

    def __init__(self, env_cfg: EnvCfg, algo_cfg: AlgoCfg, device: str = "cpu",
                 total_episodes: int = 0):
        super().__init__()
        self.env_cfg = env_cfg
        self.cfg = algo_cfg
        self.device = device
        self.n_actions = env_cfg.n_actions
        self.total_episodes = int(total_episodes)   # 0 disables the decay schedules
        self._entropy_coef = algo_cfg.entropy_coef

        self.policy_net = SAPPOPolicyNetwork(env_cfg, algo_cfg).to(device)
        self.value_net = SAPPOValueNetwork(env_cfg, algo_cfg).to(device)
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
        sub = self._legal_subvector(self.policy_net(tensor), legal, "SAPPO.act_collect")
        dist = torch.distributions.Categorical(logits=sub)
        pick = dist.sample()
        return (int(legal[int(pick.item())]), float(dist.log_prob(pick).item()),
                float(self.value_net.denormalized(tensor).item()))

    # -- learner side ------------------------------------------------------ #
    def _anneal(self) -> None:
        """Linear decay of learning rates and entropy coefficient over the
        episode budget (Table 4 of the revised manuscript)."""
        if not self.total_episodes:
            return
        frac = min(1.0, self.episodes_trained / self.total_episodes)
        lr_factor = 1.0 - (1.0 - self.cfg.lr_end_factor) * frac
        for group in self.policy_optimizer.param_groups:
            group["lr"] = self.cfg.actor_lr * lr_factor
        for group in self.value_optimizer.param_groups:
            group["lr"] = self.cfg.critic_lr * lr_factor
        self._entropy_coef = self.cfg.entropy_coef \
            + (self.cfg.entropy_coef_end - self.cfg.entropy_coef) * frac

    def learn(self, episodes: List[Episode]) -> Dict[str, float]:
        self._anneal()
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
                                    self.cfg.advantage_clip, self.cfg.advantage_norm)
        # The critic trains in normalised-return space; GAE upstream already ran
        # on the raw values the workers collected via ``denormalized``.
        self.value_net.update_stats(batch["returns"], self.RET_MOMENTUM)
        returns_norm = self.value_net.normalize(batch["returns"])
        values_norm = self.value_net.normalize(batch["values"])

        log_ratio_cap = math.log(self.cfg.ratio_cap)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                 "approx_kl": 0.0, "clip_fraction": 0.0}
        n_batches = 0
        epochs_used = 0
        for _ in range(self.cfg.ppo_epochs):
            epoch_kl_sum, epoch_batches = 0.0, 0
            for index in self.buffer.minibatches(self.cfg.minibatch_size):
                idx = torch.as_tensor(index, dtype=torch.long, device=self.device)
                states = batch["states"][idx]
                actions = batch["actions"][idx]
                old_log_probs = batch["log_probs"][idx]
                old_values = values_norm[idx]
                returns = returns_norm[idx]
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

                value_pred = self.value_net(states)   # normalised space, O(1) loss
                value_clipped = old_values + torch.clamp(value_pred - old_values,
                                                         -self.cfg.clip_eps, self.cfg.clip_eps)
                value_loss = torch.max((value_pred - returns).pow(2),
                                       (value_clipped - returns).pow(2)).mean()

                loss = policy_loss + self.cfg.value_coef * value_loss \
                    - self._entropy_coef * entropy
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
                    approx_kl = float((old_log_probs - new_log_probs).mean())
                    stats["policy_loss"] += float(policy_loss)
                    stats["value_loss"] += float(value_loss)
                    stats["entropy"] += float(entropy)
                    stats["approx_kl"] += approx_kl
                    stats["clip_fraction"] += float(clipped)
                epoch_kl_sum += approx_kl
                epoch_batches += 1
                n_batches += 1
            epochs_used += 1
            if epoch_batches and epoch_kl_sum / epoch_batches > self.cfg.kl_target:
                break   # the policy has moved far enough for this round's data
        self.buffer.clear()
        out = {k: v / max(1, n_batches) for k, v in stats.items()}
        out["ppo_epochs_used"] = float(epochs_used)
        out["ret_mean"] = float(self.value_net.ret_mean.item())
        out["ret_std"] = float(self.value_net.ret_std.item())
        out["actor_lr"] = float(self.policy_optimizer.param_groups[0]["lr"])
        out["entropy_coef"] = float(self._entropy_coef)
        return out

    # -- evaluation -------------------------------------------------------- #
    @torch.no_grad()
    def act_greedy(self, env, state: np.ndarray) -> int:
        legal = env.legal_actions()
        sub = self._legal_subvector(self.policy_net(self._tensor(state, self.device)),
                                    legal, "SAPPO.act_greedy")
        return int(legal[int(torch.argmax(sub).item())])

    @torch.no_grad()
    def act_stochastic(self, env, state: np.ndarray) -> int:
        legal = env.legal_actions()
        sub = self._legal_subvector(self.policy_net(self._tensor(state, self.device)),
                                    legal, "SAPPO.act_stochastic")
        return int(legal[int(torch.distributions.Categorical(logits=sub).sample().item())])

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
