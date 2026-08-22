"""Order-stream generation and single-instance CSV I/O.

An *instance* here is one stream of customer orders: arrival times drawn from a
Poisson process with mean inter-arrival time ``1/lambda`` and, for each order, a
set of items sampled uniformly over the SKUs of the warehouse.  A stream is
stored as a long table with one row per (order, item)::

    order_id,arrival_time,item_id,pick_point_id

Evaluation and validation streams are materialised once and never regenerated;
training streams may be re-sampled every episode (``instance.train_mode``).
"""
from __future__ import annotations

import csv
import os
import random
from typing import Dict, List, Sequence

from configs.config import InstanceCfg
from environment.entities import Order
from environment.problem import Warehouse

FIELDNAMES = ("order_id", "arrival_time", "item_id", "pick_point_id")


def sample_order_records(warehouse: Warehouse,
                         cfg: InstanceCfg,
                         n_orders: int,
                         mean_interarrival: float,
                         rng: random.Random | None = None) -> List[Dict[str, object]]:
    """Draw one order stream from the instance parameter table."""
    rng = rng or random.Random()
    all_items = warehouse.item_list()
    max_items = min(cfg.max_items_per_order, len(all_items))
    min_items = min(cfg.min_items_per_order, max_items)

    records: List[Dict[str, object]] = []
    arrival_time = 0
    for order_id in range(1, n_orders + 1):
        arrival_time += int(rng.expovariate(1.0 / mean_interarrival))
        n_items = rng.randint(min_items, max_items)
        for item in rng.sample(all_items, n_items):
            records.append({"order_id": order_id,
                            "arrival_time": arrival_time,
                            "item_id": item.item_id,
                            "pick_point_id": item.pick_point_id})
    return records


def save_stream_csv(records: Sequence[Dict[str, object]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row[key] for key in FIELDNAMES})


def load_stream_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def build_orders(warehouse: Warehouse, records: Sequence[Dict[str, object]]) -> List[Order]:
    """Turn a long table into ``Order`` objects holding the warehouse's items."""
    grouped: Dict[int, Dict[str, object]] = {}
    for row in records:
        order_id = int(row["order_id"])
        entry = grouped.setdefault(order_id, {"arrival_time": float(row["arrival_time"]),
                                              "items": []})
        item_id = str(row["item_id"])
        item = warehouse.items.get(item_id)
        if item is None:
            raise KeyError(
                f"item {item_id!r} is not part of the current warehouse layout "
                f"({warehouse.N_w} aisles x {warehouse.N_l} positions); "
                "the instance file and configs/env.yaml disagree")
        entry["items"].append(item)

    return [Order(order_id, entry["items"], entry["arrival_time"])
            for order_id, entry in sorted(grouped.items())]


def load_orders(warehouse: Warehouse, path: str) -> List[Order]:
    return build_orders(warehouse, load_stream_csv(path))


def sample_orders(warehouse: Warehouse, cfg: InstanceCfg, n_orders: int,
                  mean_interarrival: float,
                  rng: random.Random | None = None) -> List[Order]:
    return build_orders(warehouse,
                        sample_order_records(warehouse, cfg, n_orders,
                                             mean_interarrival, rng))
