"""Loader for the original (pre-refactor) simulator, used only by the self-checks.

``tools/reference/env_I_submitted.py`` is the implementation that produced the
submitted results, kept verbatim.  It is deliberately outside the package
layout -- it hard-codes its parameters and imports ``gymnasium`` -- but it is the
reference against which the refactored environment is validated, so this module
loads it in a controlled way:

* ``gymnasium`` is stubbed, because only ``gym.Env`` as a base class is needed;
* ``Config.parameter`` is patched so that the legacy simulator runs with the
  parameters of the current ``configs/env.yaml``.

If the legacy file has been removed from the working tree the equivalence check
reports itself as skipped rather than failing.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
from typing import List, Optional

from configs.config import EnvCfg

LEGACY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "reference", "env_I_submitted.py")


def _install_gym_stub() -> None:
    if "gymnasium" in sys.modules:
        return
    stub = types.ModuleType("gymnasium")

    class _Env:  # minimal stand-in for gym.Env
        pass

    stub.Env = _Env
    sys.modules["gymnasium"] = stub


def available() -> bool:
    return os.path.exists(LEGACY_PATH)


def load(cfg: EnvCfg):
    """Import the legacy module with its parameters bound to ``cfg``."""
    if not available():
        raise FileNotFoundError(LEGACY_PATH)
    _install_gym_stub()

    spec = importlib.util.spec_from_file_location("legacy_env_I", LEGACY_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    parameters = {
        "warehouse": {"shelf_capacity": cfg.n_positions,
                      "shelf_levels": 1,
                      "aisle_num": cfg.n_aisles,
                      "shelf_length": cfg.shelf_length,
                      "shelf_width": cfg.shelf_width,
                      "aisle_width": cfg.aisle_width,
                      "entrance_width": cfg.entrance_width,
                      "depot_position": tuple(cfg.depot_position)},
        "robot": {"robot_speed": cfg.robot_speed, "robot_num": cfg.n_robots},
        "picker": {"picker_speed": cfg.picker_speed, "picker_num": cfg.n_pickers},
        "order": {"pack_time": cfg.pack_time, "unit_delay_cost": 0.1},
        "item": {"pick_time": cfg.pick_time},
    }
    module.Config.parameter = lambda self: parameters
    return module


def build_orders(module, warehouse, records) -> List:
    """Build legacy ``Order`` objects that reference the legacy item instances."""
    grouped = {}
    for row in records:
        order_id = int(row["order_id"])
        entry = grouped.setdefault(order_id, {"arrival_time": float(row["arrival_time"]),
                                              "items": []})
        entry["items"].append(warehouse.items[str(row["item_id"])])
    return [module.Order(order_id, entry["items"], entry["arrival_time"])
            for order_id, entry in sorted(grouped.items())]


def legacy_action(env, choice) -> tuple:
    """Translate a :class:`baselines.rules.Choice` into the legacy action tuple."""
    if choice.kind == "picker":
        return (env.pickers[choice.actor], env.pick_points_list[choice.target]), None
    robot = env.robots[choice.actor]
    if choice.target < 0:
        return None, (robot, env.depot_object)
    return None, (robot, env.pick_points_list[choice.target])
