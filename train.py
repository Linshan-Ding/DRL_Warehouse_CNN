"""Train SAPPO on one warehouse / resource configuration.

    python train.py --config configs/exp/e0_baseline.yaml --run-name e0_run1

Everything that defines the experiment lives in the stacked YAML files; the only
command-line arguments are the config files and the run name.

Note on transferability: the actor head has ``|A| = K*N_w*N_l + R*(N_w*N_l + 1)``
outputs, so a policy is specific to the configuration it was trained on.  Every
configuration is trained from scratch and no weights are ever carried over --
the checkpoint stores ``|A|`` and refuses to load into a different setting.
"""
from __future__ import annotations

import argparse
import os
import random
import time
from typing import List, Optional

import numpy as np

from agent.ppo import SAPPOAgent
from configs.config import Config, add_config_arguments, config_from_args
from data.dataset import instance_path, make_eval_instances
from data.generator import load_orders, sample_orders
from environment.env import WarehouseEnv
from result.logger import RunLogger, gpu_memory_gb
from result.metrics import episode_metrics


def _training_orders(env: WarehouseEnv, cfg: Config, rng: random.Random) -> List:
    """Order stream for one training episode."""
    if cfg.instance.train_mode == "sampled":
        return sample_orders(env.warehouse, cfg.instance, cfg.instance.n_orders,
                             cfg.instance.mean_interarrival, rng)
    path = instance_path(cfg, "main", cfg.instance.mean_interarrival)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run `python -m data.dataset` first, or set "
            "instance.train_mode to 'sampled'")
    return load_orders(env.warehouse, path)


def _validation_streams(cfg: Config) -> List[str]:
    directory = os.path.join(cfg.instance.instances_dir, "val")
    if not os.path.isdir(directory):
        return []
    return sorted(os.path.join(directory, name) for name in os.listdir(directory)
                  if name.endswith(".csv"))


def run_episode(env: WarehouseEnv, agent: SAPPOAgent, cfg: Config, orders) -> dict:
    """Collect one on-policy episode; returns its metrics."""
    state = env.reset(orders)
    reward_sum = 0.0
    decision_seconds = 0.0
    started = time.time()

    for _ in range(cfg.env.max_steps):
        tick = time.perf_counter()
        action = agent.act(env, state)
        decision_seconds += time.perf_counter() - tick

        state, reward, done, _ = env.step(action)
        agent.observe(reward, done)
        reward_sum += reward
        if done:
            break

    return episode_metrics(env, decision_seconds, time.time() - started,
                           reward_sum).as_row()


def evaluate_greedy(cfg: Config, agent: SAPPOAgent, streams: List[str]) -> tuple:
    """Greedy roll-outs on the fixed validation streams."""
    if not streams:
        return float("nan"), float("nan")
    agent.eval_mode()
    env = WarehouseEnv(cfg.env)
    flows = []
    for path in streams:
        state = env.reset(load_orders(env.warehouse, path))
        for _ in range(cfg.env.max_steps):
            state, _, done, _ = env.step(agent.act_greedy(env, state))
            if done:
                break
        flows.append(env.episode_summary()["mean_flow_time"])
    agent.train_mode()
    return float(np.mean(flows)), float(np.std(flows))


def train(cfg: Config) -> str:
    device = cfg.torch_device
    env = WarehouseEnv(cfg.env)
    agent = SAPPOAgent(cfg.env, cfg.algo, device)
    logger = RunLogger(cfg)
    rng = random.Random()

    validation = _validation_streams(cfg)
    header = (f"SAPPO | device={device} | N_w={cfg.env.n_aisles} N_l={cfg.env.n_positions} "
              f"K={cfg.env.n_pickers} R={cfg.env.n_robots} C={cfg.env.robot_capacity} "
              f"channels={cfg.env.state_channels} gamma={cfg.algo.gamma} "
              f"|A|={cfg.env.n_actions} params={agent.n_parameters:,}")
    print(header)
    logger.text(header)

    best_flow = float("inf")
    best_path = logger.path("checkpoint_best.pt")
    last_path = logger.path("checkpoint_last.pt")
    total_decisions = 0

    for episode in range(1, cfg.algo.n_episodes + 1):
        metrics = run_episode(env, agent, cfg, _training_orders(env, cfg, rng))
        total_decisions += int(metrics["n_decisions"])

        if episode % cfg.algo.update_every_episodes == 0:
            metrics.update(agent.update())

        metrics["episode"] = episode
        metrics["sps"] = total_decisions / max(logger.elapsed_s, 1e-9)
        metrics["gpu_mem_gb"] = gpu_memory_gb()

        if cfg.algo.eval_interval and episode % cfg.algo.eval_interval == 0:
            eval_mean, eval_std = evaluate_greedy(cfg, agent, validation)
            metrics["eval_flow_mean"] = eval_mean
            metrics["eval_flow_std"] = eval_std
            if eval_mean == eval_mean and eval_mean < best_flow:   # skips NaN
                best_flow = eval_mean
                agent.save(best_path)

        logger.log(episode, metrics)
        print(f"[ep {episode:>5}] F_bar={metrics['mean_flow_time']:10.3f} "
              f"reward={metrics['reward_sum']:10.3f} "
              f"decisions={int(metrics['n_decisions']):>5} "
              f"sps={metrics['sps']:7.1f}"
              + (f" eval={metrics['eval_flow_mean']:.3f}" if "eval_flow_mean" in metrics else ""))

    agent.save(last_path)
    if best_flow == float("inf"):
        agent.save(best_path)

    _write_training_cost(cfg, logger, agent, total_decisions)
    logger.close()
    print(f"\nrun directory: {logger.run_dir}")
    return logger.run_dir


def _write_training_cost(cfg: Config, logger: RunLogger, agent: SAPPOAgent,
                         total_decisions: int) -> None:
    """One row summarising what this configuration cost to train."""
    import csv

    path = logger.path("training_cost.csv")
    row = {
        "run_name": cfg.run.run_name,
        "n_aisles": cfg.env.n_aisles,
        "n_positions": cfg.env.n_positions,
        "n_pickers": cfg.env.n_pickers,
        "n_robots": cfg.env.n_robots,
        "robot_capacity": cfg.env.robot_capacity,
        "state_channels": cfg.env.state_channels,
        "gamma": cfg.algo.gamma,
        "n_actions": cfg.env.n_actions,
        "n_parameters": agent.n_parameters,
        "n_episodes": cfg.algo.n_episodes,
        "total_decisions": total_decisions,
        "wall_clock_s": round(logger.elapsed_s, 3),
        "decisions_per_second": round(total_decisions / max(logger.elapsed_s, 1e-9), 3),
        "device": cfg.torch_device,
    }
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(f"training cost -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.add_argument("--skip-dataset", action="store_true",
                        help="assume the fixed instances already exist")
    args, extra = parser.parse_known_args()
    cfg = config_from_args(args, extra)
    if not args.skip_dataset:
        make_eval_instances(cfg)
    train(cfg)


if __name__ == "__main__":
    main()
