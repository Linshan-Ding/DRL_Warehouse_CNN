"""Materialise the fixed evaluation and validation instances.

Run once::

    python -m data.dataset --config configs/instance.yaml

Products: ``data/instances/{main,val,large}/*.csv`` plus ``data/instances/index.csv``.

Existing files are never overwritten -- these instances, not a random seed, are
the reproduction baseline.  The three ``main`` streams are the ones that
produced the submitted results: if the legacy files from the original repository
layout are still present they are migrated verbatim rather than regenerated.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
from typing import Dict, List

from configs.config import Config, add_config_arguments, config_from_args
from data.generator import (FIELDNAMES, load_stream_csv, sample_order_records,
                            save_stream_csv)
from environment.problem import Warehouse

INDEX_FIELDS = ("instance_id", "tier", "mean_interarrival", "n_orders", "n_rows", "path")

# Streams shipped with the original repository, kept as the "main" tier.
LEGACY_MAIN = {20.0: "data/data/instances/orders_20.csv",
               40.0: "data/data/instances/orders_40.csv",
               60.0: "data/data/instances/orders_60.csv"}


def _instance_path(cfg: Config, tier: str, name: str) -> str:
    return os.path.join(cfg.instance.instances_dir, tier, f"{name}.csv")


def _register(rows: List[Dict[str, object]], instance_id: str, tier: str,
              mean_interarrival: float, path: str) -> None:
    records = load_stream_csv(path)
    n_orders = len({row["order_id"] for row in records})
    rows.append({"instance_id": instance_id, "tier": tier,
                 "mean_interarrival": mean_interarrival, "n_orders": n_orders,
                 "n_rows": len(records), "path": path})


def make_eval_instances(cfg: Config) -> str:
    warehouse = Warehouse(cfg.env)
    inst = cfg.instance
    rng = random.Random()
    rows: List[Dict[str, object]] = []

    # --- main tier: one stream per arrival rate ------------------------- #
    for mean_interarrival in inst.main_interarrivals:
        name = f"lam{int(mean_interarrival)}"
        path = _instance_path(cfg, "main", name)
        if not os.path.exists(path):
            legacy = LEGACY_MAIN.get(float(mean_interarrival))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if legacy and os.path.exists(legacy):
                shutil.copyfile(legacy, path)
                print(f"[main] migrated published stream {legacy} -> {path}")
            else:
                save_stream_csv(sample_order_records(warehouse, inst, inst.n_orders,
                                                     mean_interarrival, rng), path)
                print(f"[main] generated {path}")
        _register(rows, name, "main", mean_interarrival, path)

    # --- validation tier: checkpoint selection only --------------------- #
    for i in range(inst.n_val):
        name = f"val{i:02d}"
        path = _instance_path(cfg, "val", name)
        if not os.path.exists(path):
            save_stream_csv(sample_order_records(warehouse, inst, inst.n_orders,
                                                 inst.val_interarrival, rng), path)
            print(f"[val] generated {path}")
        _register(rows, name, "val", inst.val_interarrival, path)

    # --- large tier: arrival rates outside the parameter table ---------- #
    for mean_interarrival in inst.large_interarrivals:
        name = f"lam{int(mean_interarrival)}"
        path = _instance_path(cfg, "large", name)
        if not os.path.exists(path):
            save_stream_csv(sample_order_records(warehouse, inst, inst.large_n_orders,
                                                 mean_interarrival, rng), path)
            print(f"[large] generated {path}")
        _register(rows, name, "large", mean_interarrival, path)

    index_path = os.path.join(inst.instances_dir, "index.csv")
    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    with open(index_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[index] {len(rows)} instances -> {index_path}")
    return index_path


def instance_path(cfg: Config, tier: str, mean_interarrival: float) -> str:
    return _instance_path(cfg, tier, f"lam{int(mean_interarrival)}")


def read_index(cfg: Config) -> List[Dict[str, str]]:
    index_path = os.path.join(cfg.instance.instances_dir, "index.csv")
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"{index_path} not found -- run `python -m data.dataset` first")
    with open(index_path, "r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    args, extra = parser.parse_known_args()
    make_eval_instances(config_from_args(args, extra))


if __name__ == "__main__":
    main()
