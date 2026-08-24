"""Evaluate trained global policies and dispatching rules on the fixed cases.

    python eval.py --methods SAPPO AG-DQN HSDDQN SOA+A2C DRLG \
                   MQ-ND MQ-MinRQ MQ-MI MI-MinRQ MI-MI --out result/main

Protocol (manuscript conventions):

* every RL method loads its ONE parameter file from ``models/`` and evaluates
  each case with **3 stochastic-policy samples** (policy-gradient methods
  sample from the policy distribution; value-based methods use epsilon-greedy
  with the small evaluation epsilon of Table 6) -> mean +/- std of F-bar;
* the dispatching rules are deterministic and evaluated once per case;
* **D-bar is the true average decision time**: wall-clock milliseconds per
  decision, measured around the action computation only.  The simulated
  seconds between decision epochs are logged separately as
  ``sim_time_per_decision`` and are a property of the system, not of the
  algorithm.  Evaluation is single-process on purpose so the timing is clean.

Scenario overrides turn the same command into every sensitivity study, e.g.
``--set-capacity 2`` or ``--set-fleet 8 16`` -- the global policy is evaluated
zero-shot, without retraining.
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Dict, List, Optional, Sequence

from agent.registry import ALGORITHMS, build, model_path
from baselines.rules import PAPER_RULES, RulePolicy
from configs.config import Config, add_config_arguments, case_id, config_from_args
from data.dataset import read_index
from data.generator import load_orders
from environment.env import WarehouseEnv
from result.metrics import episode_metrics

RESULT_FIELDS = (
    "case", "mean_interarrival", "case_id", "method", "sample_id",
    "n_aisles", "n_positions", "n_pickers", "n_robots", "robot_capacity",
    "state_channels", "layout", "pick_time", "gamma",
    "mean_flow_time", "makespan", "n_completed", "n_orders", "n_decisions",
    "decision_time_ms", "sim_time_per_decision", "solve_wall_clock_s",
)


class RuleMethod:
    deterministic = True

    def __init__(self, name: str):
        self.name = name
        self.policy = RulePolicy(name)

    def act(self, env, state) -> int:
        return self.policy.act(env)


class ModelMethod:
    deterministic = False

    def __init__(self, cfg: Config, name: str, checkpoint: Optional[str] = None):
        self.name = name
        self.agent = build(name, cfg, cfg.torch_device)
        path = checkpoint or model_path(cfg, name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found -- train {name} first (experiments/run_1x_train_*.py)")
        self.agent.load(path)

    def act(self, env, state) -> int:
        return self.agent.act_stochastic(env, state)


def solve(cfg: Config, method, stream_path: str, env_overrides: dict) -> Dict[str, float]:
    env = WarehouseEnv(cfg.scenario(**env_overrides))
    state = env.reset(load_orders(env.warehouse, stream_path))
    decision_seconds = 0.0
    started = time.time()
    for _ in range(env.cfg.max_steps):
        tick = time.perf_counter()
        action = method.act(env, state)
        decision_seconds += time.perf_counter() - tick
        state, _, done, _ = env.step(action)
        if done:
            break
    return episode_metrics(env, decision_seconds, time.time() - started).as_row()


def evaluate(cfg: Config, methods: Sequence[str], out_dir: str,
             cases: Optional[Sequence[str]] = None, samples: int = 3,
             env_overrides: Optional[dict] = None,
             fleet_from_index: bool = True) -> str:
    """Evaluate ``methods`` on the fixed cases; returns the CSV path.

    ``fleet_from_index=True`` (main protocol) takes (K, R) from each case's
    index entry; overrides like capacity / pick time / layout stack on top.
    ``False`` keeps the fleet from ``env_overrides`` for every stream (used by
    the resource-scale study, which sweeps fleets over the lambda=40 streams).
    """
    env_overrides = dict(env_overrides or {})
    index = read_index(cfg, tier="case")
    if cases:
        wanted = set(cases)
        index = [row for row in index if row["case"] in wanted]
    if not index:
        raise ValueError("no cases selected")

    built: List = []
    for name in methods:
        built.append(ModelMethod(cfg, name) if name in ALGORITHMS else RuleMethod(name))

    rows: List[Dict] = []
    for entry in index:
        overrides = dict(env_overrides)
        if fleet_from_index:
            overrides.setdefault("n_pickers", int(entry["n_pickers"]))
            overrides.setdefault("n_robots", int(entry["n_robots"]))
        lam = float(entry["mean_interarrival"])
        for method in built:
            n_samples = 1 if method.deterministic else samples
            for sample in range(1, n_samples + 1):
                metrics = solve(cfg, method, entry["path"], overrides)
                scenario = cfg.scenario(**overrides)
                rows.append({
                    "case": entry["case"], "mean_interarrival": lam,
                    "case_id": case_id(lam, scenario.n_pickers, scenario.n_robots,
                                       cfg.instance),
                    "method": method.name, "sample_id": sample,
                    "n_aisles": scenario.n_aisles, "n_positions": scenario.n_positions,
                    "n_pickers": scenario.n_pickers, "n_robots": scenario.n_robots,
                    "robot_capacity": scenario.robot_capacity,
                    "state_channels": scenario.state_channels,
                    "layout": scenario.layout, "pick_time": scenario.pick_time,
                    "gamma": cfg.algo.gamma,
                    **{key: metrics[key] for key in (
                        "mean_flow_time", "makespan", "n_completed", "n_orders",
                        "n_decisions", "decision_time_ms", "sim_time_per_decision",
                        "solve_wall_clock_s")},
                })
            done_rows = [r for r in rows if r["case"] == entry["case"]
                         and r["method"] == method.name]
            flows = [r["mean_flow_time"] for r in done_rows]
            times = [r["decision_time_ms"] for r in done_rows]
            print(f"  {entry['case']} {method.name:<9} "
                  f"F_bar={sum(flows)/len(flows):9.1f}  "
                  f"D_bar={sum(times)/len(times):7.3f} ms  ({len(flows)} sample(s))")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "eval_results.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(RESULT_FIELDS))
        writer.writeheader(); writer.writerows(rows)
    print(f"\n{len(rows)} rows -> {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.add_argument("--methods", nargs="+", required=True,
                        help=f"any of {', '.join(ALGORITHMS)} and rules "
                             f"{', '.join(PAPER_RULES)}")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--cases", nargs="*", default=None, help="subset, e.g. C18")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--set-fleet", nargs=2, type=int, metavar=("K", "R"),
                        default=None, help="override the fleet for every stream")
    parser.add_argument("--set-capacity", type=int, default=None)
    parser.add_argument("--set-pick-time", type=float, default=None)
    parser.add_argument("--set-layout", default=None,
                        choices=["two_cross_aisles", "three_cross_aisles"])
    args, extra = parser.parse_known_args()
    cfg = config_from_args(args, extra)

    overrides: Dict[str, object] = {}
    fleet_from_index = True
    if args.set_fleet:
        overrides["n_pickers"], overrides["n_robots"] = args.set_fleet
        fleet_from_index = False
    if args.set_capacity is not None:
        overrides["robot_capacity"] = args.set_capacity
    if args.set_pick_time is not None:
        overrides["pick_time"] = args.set_pick_time
    if args.set_layout is not None:
        overrides["layout"] = args.set_layout

    evaluate(cfg, args.methods, args.out, args.cases, args.samples,
             overrides, fleet_from_index)


if __name__ == "__main__":
    main()
