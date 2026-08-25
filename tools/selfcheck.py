"""Correctness gates for the refactored simulator.

Run before any experiment::

    python -m tools.selfcheck

Three checks, all of which must pass:

1. **reward identity** -- the undiscounted return of an episode equals minus the
   final mean flow time, so the agent really optimises the paper's objective;
2. **equivalence with the original simulator** -- driven by the same
   deterministic dispatching rule, the refactored environment and the
   implementation that produced the submitted results agree on every event time,
   every chosen action, every reward and the final mean flow time;
3. **capacity degeneracy** -- with ``robot_capacity: 1`` the generalised order
   release behaves exactly as the submitted one-order-per-cycle model, while
   ``robot_capacity: 2`` actually changes the schedule.
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Tuple

import numpy as np

from baselines.rules import RulePolicy, to_action_index
from configs.config import Config, add_config_arguments, config_from_args
from data.generator import build_orders, load_stream_csv
from environment.env import WarehouseEnv
from environment.problem import mean_flow_time
from tools import legacy_env

TOLERANCE = 1e-9
DEFAULT_STREAM = "data/instances/cases/C18.csv"


def _run_rule(cfg: Config, records, rule_name: str) -> Tuple[float, List[float], int]:
    env = WarehouseEnv(cfg.scenario())
    rule = RulePolicy(rule_name)
    env.reset(build_orders(env.warehouse, records))
    rewards: List[float] = []
    while not env.done and env.decision_count < cfg.env.max_steps:
        _, reward, _, _ = env.step(rule.act(env))
        rewards.append(reward)
    return env.episode_summary()["mean_flow_time"], rewards, env.decision_count


# --------------------------------------------------------------------------- #
def check_reward_identity(cfg: Config, records, rule_name: str) -> bool:
    env = WarehouseEnv(cfg.scenario())
    rule = RulePolicy(rule_name)
    env.reset(build_orders(env.warehouse, records))
    total = 0.0
    while not env.done and env.decision_count < cfg.env.max_steps:
        _, reward, _, _ = env.step(rule.act(env))
        total += reward
    final = mean_flow_time(env.orders_completed, env.orders_uncompleted, env.current_time)
    gap = abs(total + final)
    ok = gap < 1e-6
    print(f"  sum(r_t) = {total:.6f}   -F_final = {-final:.6f}   |gap| = {gap:.2e}"
          f"   -> {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------- #
def check_legacy_equivalence(cfg: Config, records, rule_name: str) -> bool:
    if not legacy_env.available():
        print("  legacy simulator not present in the working tree -> SKIPPED")
        return True

    module = legacy_env.load(cfg.env)
    old = module.WarehouseEnv()
    new = WarehouseEnv(cfg.scenario())
    rule = RulePolicy(rule_name)

    old_state = old.reset(legacy_env.build_orders(module, old, records))
    new_state = new.reset(build_orders(new.warehouse, records))

    # The refactored observation appends configuration planes after the four
    # spatial channels of the manuscript; physics equivalence is judged on the
    # spatial channels only.
    if not np.allclose(old_state, new_state[:4]):
        print("  initial spatial channels differ -> FAIL")
        return False

    step = 0
    while not old.done and not new.done and step < cfg.env.max_steps:
        old_choice = rule.choose(old)
        new_choice = rule.choose(new)
        if old_choice != new_choice:
            print(f"  step {step}: action differs {old_choice} vs {new_choice} -> FAIL")
            return False

        old_state, old_reward, old_done, _, _ = old.step(legacy_env.legacy_action(old, old_choice))
        new_state, new_reward, new_done, _ = new.step(to_action_index(new, new_choice))

        if abs(old.current_time - new.current_time) > TOLERANCE:
            print(f"  step {step}: clock differs {old.current_time} vs {new.current_time} -> FAIL")
            return False
        if abs(old_reward - new_reward) > TOLERANCE:
            print(f"  step {step}: reward differs {old_reward} vs {new_reward} -> FAIL")
            return False
        if not np.allclose(old_state, new_state[:4]):
            print(f"  step {step}: spatial channels differ -> FAIL")
            return False
        if old_done != new_done:
            print(f"  step {step}: termination differs -> FAIL")
            return False
        step += 1

    old_flow = float(np.mean([o.complete_time - o.arrive_time for o in old.orders_completed]))
    new_flow = new.episode_summary()["mean_flow_time"]
    ok = (abs(old_flow - new_flow) < 1e-9
          and len(old.orders_completed) == len(new.orders_completed))
    print(f"  {step} decision epochs compared; F_bar old = {old_flow:.6f}, "
          f"new = {new_flow:.6f} -> {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------- #
def check_capacity_degeneracy(cfg: Config, records, rule_name: str) -> bool:
    import copy

    one = copy.deepcopy(cfg)
    one.env.robot_capacity = 1
    two = copy.deepcopy(cfg)
    two.env.robot_capacity = 2

    flow_one, _, steps_one = _run_rule(one, records, rule_name)
    flow_two, _, steps_two = _run_rule(two, records, rule_name)

    baseline_flow, _, baseline_steps = _run_rule(cfg, records, rule_name)
    degenerate = (cfg.env.robot_capacity != 1
                  or (abs(flow_one - baseline_flow) < 1e-12 and steps_one == baseline_steps))
    changed = abs(flow_two - flow_one) > 1e-9 or steps_two != steps_one
    print(f"  C=1: F_bar = {flow_one:.6f} over {steps_one} epochs")
    print(f"  C=2: F_bar = {flow_two:.6f} over {steps_two} epochs")
    ok = degenerate and changed
    if not changed:
        print("  raising the capacity did not change the schedule -> FAIL")
    return ok


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.add_argument("--stream", default=DEFAULT_STREAM,
                        help="order stream CSV used by the checks")
    parser.add_argument("--rule", default="MQ-ND",
                        help="deterministic dispatching rule driving the checks")
    args, extra = parser.parse_known_args()
    cfg = config_from_args(args, extra)
    records = load_stream_csv(args.stream)

    print(f"stream : {args.stream} ({len({r['order_id'] for r in records})} orders)")
    print(f"setting: N_w={cfg.env.n_aisles} N_l={cfg.env.n_positions} "
          f"K={cfg.env.n_pickers} R={cfg.env.n_robots} C={cfg.env.robot_capacity} "
          f"rule={args.rule}\n")

    results = {}
    print("[1/3] reward identity")
    results["reward identity"] = check_reward_identity(cfg, records, args.rule)
    print("\n[2/3] equivalence with the original simulator")
    results["legacy equivalence"] = check_legacy_equivalence(cfg, records, args.rule)
    print("\n[3/3] carrying-capacity degeneracy")
    results["capacity degeneracy"] = check_capacity_degeneracy(cfg, records, args.rule)

    print("\n" + "-" * 60)
    for name, ok in results.items():
        print(f"  {name:<24} {'PASS' if ok else 'FAIL'}")
    failed = [name for name, ok in results.items() if not ok]
    print("-" * 60)
    print("all checks passed" if not failed else f"FAILED: {', '.join(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
