"""Materialise the 27 fixed evaluation cases and the validation streams.

A *case* is one cell of the 3 x 3 x 3 grid of the manuscript:
``1/lambda in {20, 40, 60} x K in {1, 2, 3} x R in {2, 4, 6}``, numbered C01..C27
with the arrival rate as the outer loop, then pickers, then robots (the same
ordering as Table 5).  Every case owns an independent order stream, materialised
once as ``data/instances/cases/Cxx.csv`` -- these files, not a random seed, are
the reproduction baseline and are never regenerated once present.

Validation streams (checkpoint selection during training) live under
``data/instances/val/`` and are never reported.

Run: right-click ``experiments/run_01_make_instances.py``.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
from typing import Dict, List

from configs.config import Config, add_config_arguments, config_from_args
from data.generator import load_stream_csv, sample_order_records, save_stream_csv
from environment.problem import Warehouse

INDEX_FIELDS = ("case", "tier", "mean_interarrival", "n_pickers", "n_robots",
                "n_orders", "n_rows", "path")


def case_specs(cfg: Config) -> List[Dict]:
    specs = []
    for i_lam, lam in enumerate(cfg.instance.case_interarrivals):
        for i_k, k in enumerate(cfg.instance.case_pickers):
            for i_r, r in enumerate(cfg.instance.case_robots):
                cid = 9 * i_lam + 3 * i_k + i_r + 1
                specs.append({"case": f"C{cid:02d}", "mean_interarrival": float(lam),
                              "n_pickers": int(k), "n_robots": int(r)})
    return specs


def make_instances(cfg: Config) -> str:
    warehouse = Warehouse(cfg.env)
    rng = random.Random()
    rows: List[Dict] = []

    cases_dir = os.path.join(cfg.instance.instances_dir, "cases")
    os.makedirs(cases_dir, exist_ok=True)
    for spec in case_specs(cfg):
        path = os.path.join(cases_dir, f"{spec['case']}.csv")
        if not os.path.exists(path):
            save_stream_csv(sample_order_records(warehouse, cfg.instance,
                                                 cfg.instance.n_orders,
                                                 spec["mean_interarrival"], rng), path)
            print(f"[case] generated {path}  (1/lambda={spec['mean_interarrival']:g} "
                  f"K={spec['n_pickers']} R={spec['n_robots']})")
        records = load_stream_csv(path)
        rows.append({**spec, "tier": "case",
                     "n_orders": len({row["order_id"] for row in records}),
                     "n_rows": len(records), "path": path})

    val_dir = os.path.join(cfg.instance.instances_dir, "val")
    os.makedirs(val_dir, exist_ok=True)
    for i in range(cfg.instance.n_val):
        path = os.path.join(val_dir, f"val{i:02d}.csv")
        if not os.path.exists(path):
            save_stream_csv(sample_order_records(warehouse, cfg.instance,
                                                 cfg.instance.n_orders,
                                                 cfg.instance.val_interarrival, rng), path)
            print(f"[val] generated {path}")
        records = load_stream_csv(path)
        rows.append({"case": f"val{i:02d}", "tier": "val",
                     "mean_interarrival": cfg.instance.val_interarrival,
                     "n_pickers": cfg.env.n_pickers, "n_robots": cfg.env.n_robots,
                     "n_orders": len({row["order_id"] for row in records}),
                     "n_rows": len(records), "path": path})

    index_path = os.path.join(cfg.instance.instances_dir, "index.csv")
    with open(index_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=INDEX_FIELDS)
        writer.writeheader(); writer.writerows(rows)
    print(f"[index] {len(rows)} instances -> {index_path}")
    return index_path


def read_index(cfg: Config, tier: str = "case") -> List[Dict[str, str]]:
    index_path = os.path.join(cfg.instance.instances_dir, "index.csv")
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"{index_path} not found -- run experiments/run_01_make_instances.py first")
    with open(index_path, "r", newline="", encoding="utf-8") as fh:
        return [row for row in csv.DictReader(fh) if row["tier"] == tier]


def case_path(cfg: Config, case: str) -> str:
    return os.path.join(cfg.instance.instances_dir, "cases", f"{case}.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    args, extra = parser.parse_known_args()
    make_instances(config_from_args(args, extra))


if __name__ == "__main__":
    main()
