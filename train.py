"""Train ONE global policy for one algorithm.

    python train.py --algo SAPPO
    python train.py --algo AG-DQN --config configs/exp/smoke.yaml

Every episode instantiates a fresh environment from a scenario sampled off the
parameter table (arrival rate, fleet, capacity, picking time, layout -- weights
in ``configs/train.yaml``) with a freshly sampled order stream, so the single
saved parameter file serves every evaluation scenario.  Collection runs on
parallel worker processes; the update runs batched on the GPU strictly between
collection rounds (on-policy safe).

The single deliverable is ``models/<algo>.pt`` -- the parameters with the best
mean greedy flow time on the validation mini-grid (three fixed lambda streams,
each evaluated on the fleets in ``train.val_fleets``).  Training curves on the
representative cases (C06/C13/C15/C24, for the Fig. 8 style figure) are logged
but never used for selection.
"""
from __future__ import annotations

import argparse
import os
import random
import time
from typing import Dict, List

import numpy as np

from agent.registry import ALGORITHMS, build, model_path
from configs.config import Config, add_config_arguments, config_from_args
from data.dataset import case_path, make_instances, read_index
from data.generator import load_orders
from environment.env import WarehouseEnv
from parallel.collector import EpisodeCollector, default_workers
from result.logger import RunLogger, gpu_memory_gb


# --------------------------------------------------------------------------- #
def sample_scenario(cfg: Config, rng: random.Random) -> dict:
    """One training scenario drawn from the parameter table (train.yaml)."""
    tc = cfg.train
    total = tc.w_main + tc.w_scale + tc.w_perturb
    roll = rng.random() * total

    lam = rng.choice(cfg.instance.case_interarrivals)
    env_overrides: Dict[str, object] = {}

    # Scale configs beyond the action-head envelope (shrunk by the grid
    # overlays) cannot be represented -- filter them out; a scale roll with no
    # representable config falls back to the main-grid branch.
    scale_configs = [(k, r) for k, r in tc.scale_configs
                     if k <= cfg.env.k_max and r <= cfg.env.r_max]

    if tc.w_main <= roll < tc.w_main + tc.w_scale and scale_configs:
        k, r = rng.choice(scale_configs)
        env_overrides["n_pickers"] = int(k)
        env_overrides["n_robots"] = int(r)
    elif roll < tc.w_main + tc.w_scale:
        env_overrides["n_pickers"] = rng.choice(cfg.instance.case_pickers)
        env_overrides["n_robots"] = rng.choice(cfg.instance.case_robots)
    else:
        env_overrides["n_pickers"] = rng.choice(cfg.instance.case_pickers)
        env_overrides["n_robots"] = rng.choice(cfg.instance.case_robots)
        axis = rng.random()
        if axis < 0.45:
            env_overrides["robot_capacity"] = int(rng.choice(tc.perturb_capacities))
        elif axis < 0.9:
            env_overrides["pick_time"] = float(rng.choice(tc.perturb_pick_times))
        if rng.random() < tc.perturb_layout_prob:
            env_overrides["layout"] = "three_cross_aisles"

    return {"env": env_overrides, "interarrival": float(lam)}


# --------------------------------------------------------------------------- #
def _greedy_flow(cfg: Config, agent, stream_path: str, env_overrides: dict) -> float:
    env = WarehouseEnv(cfg.scenario(**env_overrides))
    state = env.reset(load_orders(env.warehouse, stream_path))
    for _ in range(cfg.env.max_steps):
        state, _, done, _ = env.step(agent.act_greedy(env, state))
        if done:
            break
    return env.episode_summary()["mean_flow_time"]


def validate(cfg: Config, agent) -> float:
    """Greedy mean flow time over the validation mini-grid.

    Each fixed lambda stream is evaluated on every fleet in
    ``cfg.train.val_fleets``, so the saved checkpoint is the best across load
    regimes -- a single-scenario validation would bias selection toward the
    centre of the case grid.
    """
    flows = [_greedy_flow(cfg, agent, row["path"],
                          {"n_pickers": int(k), "n_robots": int(r)})
             for row in read_index(cfg, tier="val")
             for k, r in cfg.train.val_fleets]
    return float(np.mean(flows)) if flows else float("nan")


def curve_eval(cfg: Config, agent) -> Dict[str, float]:
    """Greedy flow time on the representative cases (training-curve logging only)."""
    metrics = {}
    index = {row["case"]: row for row in read_index(cfg, tier="case")}
    for case in cfg.train.curve_cases:
        row = index.get(case)
        if row is None:
            continue
        metrics[f"curve_{case}"] = _greedy_flow(
            cfg, agent, row["path"],
            {"n_pickers": int(row["n_pickers"]), "n_robots": int(row["n_robots"])})
    return metrics


# --------------------------------------------------------------------------- #
def train(cfg: Config, algo_name: str) -> str:
    device = cfg.torch_device
    agent = build(algo_name, cfg, device)
    n_workers = cfg.train.n_workers or default_workers()
    per_round = cfg.train.episodes_per_round or n_workers
    rng = random.Random()

    cfg.run.run_name = cfg.run.run_name if cfg.run.run_name != "run" \
        else f"train_{algo_name.lower().replace('+', '_').replace('-', '_')}"
    logger = RunLogger(cfg)
    os.makedirs(cfg.run.models_dir, exist_ok=True)
    best_path = model_path(cfg, algo_name)

    header = (f"{algo_name} | device={device} | workers={n_workers} | "
              f"episodes={cfg.train.n_episodes} | |A|={cfg.env.n_actions} "
              f"(envelope K<={cfg.env.k_max}, R<={cfg.env.r_max}) | "
              f"channels={cfg.env.n_state_channels} | params={agent.n_parameters:,}")
    print(header)
    logger.text(header)

    best_val = float("inf")
    total_decisions = 0
    episode_count = 0
    round_index = 0

    with EpisodeCollector(cfg, algo_name, agent.actor_state(), n_workers) as collector:
        while episode_count < cfg.train.n_episodes:
            round_index += 1
            scenarios = [sample_scenario(cfg, rng) for _ in range(per_round)]
            tick = time.perf_counter()
            episodes = collector.collect(agent.exploration(), scenarios)
            collect_s = time.perf_counter() - tick

            tick = time.perf_counter()
            stats = agent.learn(episodes)
            learn_s = time.perf_counter() - tick
            collector.sync(agent.actor_state())

            episode_count += len(episodes)
            round_decisions = sum(len(e) for e in episodes)
            total_decisions += round_decisions
            flow = float(np.mean([e.summary["mean_flow_time"] for e in episodes]))

            metrics = {"episode": episode_count, "mean_flow_time": flow,
                       "reward_sum": float(np.mean([sum(e.rewards) for e in episodes])),
                       "n_decisions": round_decisions,
                       "collect_s": round(collect_s, 3), "learn_s": round(learn_s, 3),
                       "sps": round(total_decisions / max(logger.elapsed_s, 1e-9), 1),
                       "gpu_mem_gb": gpu_memory_gb(), **stats}

            if round_index % cfg.train.eval_interval == 0:
                val = validate(cfg, agent)
                metrics["eval_flow_mean"] = val
                metrics.update(curve_eval(cfg, agent))
                if val == val and val < best_val:
                    best_val = val
                    agent.save(best_path)
                    metrics["checkpoint_saved"] = 1.0

            logger.log(episode_count, metrics)
            print(f"[{algo_name} ep {episode_count:>6}] F_bar={flow:9.1f} "
                  f"sps={metrics['sps']:7.1f} collect={collect_s:5.1f}s "
                  f"learn={learn_s:5.1f}s"
                  + (f" val={metrics['eval_flow_mean']:.1f}" if "eval_flow_mean" in metrics else ""))

    if best_val == float("inf"):
        agent.save(best_path)      # tiny smoke budgets may never reach an eval round

    _write_training_cost(cfg, logger, agent, algo_name, total_decisions, episode_count)
    logger.close()
    print(f"\nmodel  -> {best_path}\nlogs   -> {logger.run_dir}")
    return best_path


def _write_training_cost(cfg, logger, agent, algo_name, total_decisions, episodes) -> None:
    import csv
    path = logger.path("training_cost.csv")
    row = {"algorithm": algo_name, "n_actions": cfg.env.n_actions,
           "n_state_channels": cfg.env.n_state_channels,
           "k_max": cfg.env.k_max, "r_max": cfg.env.r_max,
           "grid": f"{cfg.env.n_aisles}x{cfg.env.n_positions}",
           "n_parameters": agent.n_parameters, "n_episodes": episodes,
           "total_decisions": total_decisions,
           "wall_clock_s": round(logger.elapsed_s, 3),
           "decisions_per_second": round(total_decisions / max(logger.elapsed_s, 1e-9), 3),
           "n_workers": cfg.train.n_workers or default_workers(),
           "device": cfg.torch_device}
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        writer.writeheader(); writer.writerow(row)
    print(f"training cost -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.add_argument("--algo", required=True, choices=list(ALGORITHMS))
    parser.add_argument("--skip-dataset", action="store_true")
    args, extra = parser.parse_known_args()
    cfg = config_from_args(args, extra)
    if not args.skip_dataset:
        make_instances(cfg)
    train(cfg, args.algo)


if __name__ == "__main__":
    main()
