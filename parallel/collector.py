"""Parallel episode collection with persistent worker processes.

Why processes: the environment is a pure-Python discrete-event simulation that
cannot be tensorised, so a single process is CPU-bound at a few hundred
decisions per second while the GPU idles.  Episode-level parallelism is the
simple, on-policy-safe cure: N workers each run whole episodes with a CPU copy
of the current parameters, the learner updates on the GPU strictly between
collection rounds, and fresh parameters reach the workers before the next
round.  There is no policy lag and no importance correction to worry about.

Mechanics:

* ``torch.multiprocessing`` with the ``spawn`` start method (works on Windows
  and Linux, and is safe with CUDA in the parent);
* parameters live in **shared-memory CPU tensors** created once at start-up;
  after every update the learner copies the new values in place
  (``collector.sync``) and the workers reload from shared memory at the start
  of each task -- rounds are synchronous, so reads never race with the write;
* a task is just (exploration, scenario); the result is the full episode
  (states, actions, rewards, dones, feasibility lists, and log-probs / values
  for the policy-gradient algorithms);
* scenarios are sampled in the parent so the sampling weights live in one
  place; workers only execute them; each worker pins itself to one torch
  thread to avoid oversubscription.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from agent.base import Episode

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_workers() -> int:
    return max(1, (os.cpu_count() or 4) - 2)


# --------------------------------------------------------------------------- #
def _worker_main(worker_id: int, algo_name: str, cfg_dict: dict, shared_params,
                 task_queue, result_queue) -> None:
    import sys
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    os.chdir(_REPO_ROOT)

    import torch
    torch.set_num_threads(1)

    from agent.registry import build
    from configs.config import config_from_dict
    from data.generator import sample_orders
    from environment.env import WarehouseEnv

    cfg = config_from_dict(cfg_dict)
    agent = build(algo_name, cfg, device="cpu")

    import random
    rng = random.Random()

    while True:
        task = task_queue.get()
        if task is None:
            return
        exploration, scenario = task
        agent.load_actor_state(shared_params)   # reads the shared tensors

        env_cfg = cfg.scenario(**scenario["env"])
        env = WarehouseEnv(env_cfg)
        orders = sample_orders(env.warehouse, cfg.instance, cfg.instance.n_orders,
                               scenario["interarrival"], rng)
        state = env.reset(orders)

        states, actions, rewards, dones, legals = [], [], [], [], []
        log_probs, values = [], []
        for _ in range(env_cfg.max_steps):
            legal = env.legal_actions()
            action, log_prob, value = agent.act_collect(env, state, exploration)
            states.append(np.asarray(state, dtype=np.float32))
            actions.append(action); legals.append(legal)
            if agent.needs_log_probs:
                log_probs.append(log_prob); values.append(value)
            state, reward, done, _ = env.step(action)
            rewards.append(reward); dones.append(done)
            if done:
                break

        result_queue.put({
            "worker_id": worker_id,
            "states": np.asarray(states, dtype=np.float32),
            "actions": actions, "rewards": rewards, "dones": dones, "legals": legals,
            "log_probs": log_probs if agent.needs_log_probs else None,
            "values": values if agent.needs_log_probs else None,
            "final_state": np.asarray(state, dtype=np.float32),
            "final_legal": env.legal_actions() if not env.done else [],
            "scenario": scenario,
            "summary": env.episode_summary(),
        })


# --------------------------------------------------------------------------- #
class EpisodeCollector:
    """Persistent pool of episode workers for one algorithm."""

    def __init__(self, cfg, algo_name: str, initial_params: Dict[str, dict],
                 n_workers: int = 0):
        import torch.multiprocessing as mp

        self.n_workers = int(n_workers) or default_workers()
        os.environ.setdefault("PYTHONPATH", _REPO_ROOT)

        # Shared-memory replicas of the actor parameters.
        self._shared = {group: {key: tensor.detach().clone().share_memory_()
                                for key, tensor in state.items()}
                        for group, state in initial_params.items()}

        ctx = mp.get_context("spawn")
        self._task_queue = ctx.Queue()
        self._result_queue = ctx.Queue()
        self._workers = []
        cfg_dict = cfg.to_dict()
        for wid in range(self.n_workers):
            proc = ctx.Process(target=_worker_main,
                               args=(wid, algo_name, cfg_dict, self._shared,
                                     self._task_queue, self._result_queue),
                               daemon=True)
            proc.start()
            self._workers.append(proc)

    def sync(self, params: Dict[str, dict]) -> None:
        """Copy fresh parameters into shared memory (call between rounds only)."""
        for group, state in params.items():
            shared = self._shared[group]
            for key, tensor in state.items():
                shared[key].copy_(tensor)

    def collect(self, exploration: float, scenarios: Sequence[dict]) -> List[Episode]:
        """Run one episode per scenario across the pool; blocks until all return."""
        for scenario in scenarios:
            self._task_queue.put((exploration, scenario))
        episodes: List[Episode] = []
        for _ in scenarios:
            record = self._result_queue.get()
            episodes.append(Episode(
                states=record["states"], actions=record["actions"],
                rewards=record["rewards"], dones=record["dones"],
                legals=record["legals"], log_probs=record["log_probs"],
                values=record["values"], final_state=record["final_state"],
                final_legal=record["final_legal"], scenario=record["scenario"],
                summary=record["summary"]))
        return episodes

    def close(self) -> None:
        for _ in self._workers:
            self._task_queue.put(None)
        for proc in self._workers:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
