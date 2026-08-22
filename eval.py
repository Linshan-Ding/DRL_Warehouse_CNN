"""Evaluate SAPPO and the priority dispatching rules on the fixed instances.

    python eval.py --ckpt result/e0_run1/checkpoint_best.pt --tiers main \
                   --methods SAPPO MQ-ND MQ-MinRQ MQ-MI MI-MinRQ MI-MI \
                   --run-name e0_run1

Writes ``result/<run-name>/eval_results.csv`` with one row per
(instance, method, run): the configuration that produced it, the mean order flow
time, and *two* decision-time columns that must not be confused --

``decision_time_ms``       true wall-clock milliseconds per decision (this is
                           what "average decision time" is defined as in the
                           manuscript);
``sim_time_per_decision``  simulated seconds per decision epoch, i.e.
                           makespan / #epochs -- a measure of decision
                           granularity, not of computational effort.

Only SAPPO and the five dispatching rules are available until the four RL
baselines are archived under ``baselines/rl/`` (see its README).
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Dict, List, Optional, Sequence

from agent.ppo import SAPPOAgent
from baselines.rules import PAPER_RULES, RulePolicy
from configs.config import Config, add_config_arguments, case_id, config_from_args
from data.dataset import read_index
from data.generator import load_orders
from environment.env import WarehouseEnv
from result.metrics import episode_metrics

RESULT_FIELDS = (
    "instance_id", "tier", "mean_interarrival", "case_id", "method", "run_id",
    "n_aisles", "n_positions", "n_pickers", "n_robots", "robot_capacity",
    "state_channels", "layout", "pick_time", "gamma",
    "mean_flow_time", "makespan", "n_completed", "n_orders", "n_decisions",
    "decision_time_ms", "sim_time_per_decision", "solve_wall_clock_s",
)


class RuleAgent:
    """Adapter so a dispatching rule looks like an agent to this script."""

    def __init__(self, name: str):
        self.name = name
        self.policy = RulePolicy(name)

    def act_greedy(self, env, state) -> int:
        return self.policy.act(env)


def solve_instance(cfg: Config, agent, stream_path: str) -> Dict[str, float]:
    env = WarehouseEnv(cfg.env)
    state = env.reset(load_orders(env.warehouse, stream_path))

    decision_seconds = 0.0
    started = time.time()
    for _ in range(cfg.env.max_steps):
        tick = time.perf_counter()
        action = agent.act_greedy(env, state)
        decision_seconds += time.perf_counter() - tick
        state, _, done, _ = env.step(action)
        if done:
            break

    return episode_metrics(env, decision_seconds, time.time() - started).as_row()


def build_agents(cfg: Config, methods: Sequence[str], checkpoint: Optional[str]) -> List:
    agents = []
    for name in methods:
        if name.upper() == "SAPPO":
            if not checkpoint:
                raise ValueError("--ckpt is required to evaluate SAPPO")
            agent = SAPPOAgent(cfg.env, cfg.algo, cfg.torch_device)
            agent.load(checkpoint)
            agent.eval_mode()
            agent.name = "SAPPO"
            agents.append(agent)
        elif name in PAPER_RULES or "-" in name:
            agents.append(RuleAgent(name))
        else:
            raise ValueError(
                f"unknown method {name!r}; available: SAPPO and the dispatching "
                f"rules {', '.join(PAPER_RULES)}. RL baselines have to be added "
                "under baselines/rl/ first.")
    return agents


def evaluate(cfg: Config, methods: Sequence[str], tiers: Sequence[str],
             checkpoint: Optional[str], run_id: int, out_path: str) -> str:
    index = [row for row in read_index(cfg) if row["tier"] in tiers]
    if not index:
        raise ValueError(f"no instances for tier(s) {list(tiers)} in the index")

    agents = build_agents(cfg, methods, checkpoint)
    rows: List[Dict[str, object]] = []

    for entry in index:
        mean_interarrival = float(entry["mean_interarrival"])
        for agent in agents:
            metrics = solve_instance(cfg, agent, entry["path"])
            rows.append({
                "instance_id": entry["instance_id"],
                "tier": entry["tier"],
                "mean_interarrival": mean_interarrival,
                "case_id": case_id(mean_interarrival, cfg.env.n_pickers,
                                   cfg.env.n_robots, cfg.instance),
                "method": agent.name,
                "run_id": run_id,
                "n_aisles": cfg.env.n_aisles,
                "n_positions": cfg.env.n_positions,
                "n_pickers": cfg.env.n_pickers,
                "n_robots": cfg.env.n_robots,
                "robot_capacity": cfg.env.robot_capacity,
                "state_channels": cfg.env.state_channels,
                "layout": cfg.env.layout,
                "pick_time": cfg.env.pick_time,
                "gamma": cfg.algo.gamma,
                **{key: metrics[key] for key in (
                    "mean_flow_time", "makespan", "n_completed", "n_orders",
                    "n_decisions", "decision_time_ms", "sim_time_per_decision",
                    "solve_wall_clock_s")},
            })
            print(f"  {entry['instance_id']:>8} {agent.name:<10} "
                  f"F_bar={metrics['mean_flow_time']:10.3f}  "
                  f"D_bar={metrics['decision_time_ms']:7.3f} ms  "
                  f"sim/dec={metrics['sim_time_per_decision']:7.3f} s")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(RESULT_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)} rows -> {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.add_argument("--ckpt", default=None, help="SAPPO checkpoint to evaluate")
    parser.add_argument("--methods", nargs="*", default=list(PAPER_RULES),
                        help="SAPPO and/or dispatching rule names")
    parser.add_argument("--tiers", nargs="*", default=["main"],
                        help="instance tiers: main, val, large")
    parser.add_argument("--run-id", type=int, default=1,
                        help="index of the independent run these results belong to")
    parser.add_argument("--out", default=None, help="output CSV path")
    args, extra = parser.parse_known_args()

    cfg = config_from_args(args, extra)
    out = args.out or os.path.join(cfg.run.result_dir, cfg.run.run_name, "eval_results.csv")
    evaluate(cfg, args.methods, args.tiers, args.ckpt, args.run_id, out)


if __name__ == "__main__":
    main()
