"""Observation, envelope action space and feasibility mask.

The observation given to every learning algorithm is ``o_t = (s_t, M_t)``:

* ``s_t`` -- 4 spatial channels over the picking-point grid (Section 4.2 of the
  manuscript) plus 5 constant *configuration planes* that let one network serve
  every scenario: K/k_max, R/r_max, C/c_max, tau_pick/tau_ref and the layout
  flag.  The optional ``plus_agent`` variant appends 2 per-robot channels
  carrying exactly the information the mask uses (state-sufficiency ablation).
* ``M_t`` -- the feasibility mask, built from per-resource information.  It is
  part of the observation, not a post-processing step.

The flat action index is laid out for the resource *envelope* (k_max, r_max),
so a single policy head serves every (K, R) scenario; pickers ``k >= K`` and
robots ``r >= R`` simply never appear in the feasible set.
"""
from __future__ import annotations

from typing import List

import numpy as np

DEPOT_TARGET = -1


# --------------------------------------------------------------------------- #
# envelope action layout:
#   [0, k_max*P)                      picker k -> point j
#   [k_max*P, (k_max+r_max)*P)        robot r  -> point j
#   [(k_max+r_max)*P, ... + r_max)    robot r  -> depot
# --------------------------------------------------------------------------- #
def picker_action_index(picker_idx: int, point_idx: int, n_points: int) -> int:
    return picker_idx * n_points + point_idx

def robot_action_index(robot_idx: int, point_idx: int, n_points: int, k_max: int) -> int:
    return k_max * n_points + robot_idx * n_points + point_idx

def robot_depot_index(robot_idx: int, n_points: int, k_max: int, r_max: int) -> int:
    return (k_max + r_max) * n_points + robot_idx

def n_actions(n_points: int, k_max: int, r_max: int) -> int:
    return (k_max + r_max) * n_points + r_max

def decode_action(index: int, n_points: int, k_max: int, r_max: int):
    """-> ("picker"|"robot", actor_idx, point_idx | DEPOT_TARGET)."""
    picker_block = k_max * n_points
    robot_block = picker_block + r_max * n_points
    if index < picker_block:
        return "picker", index // n_points, index % n_points
    if index < robot_block:
        offset = index - picker_block
        return "robot", offset // n_points, offset % n_points
    return "robot", index - robot_block, DEPOT_TARGET


# --------------------------------------------------------------------------- #
def build_state(env) -> np.ndarray:
    """State tensor, shape (n_channels, N_w, N_l).

    Channels 0-3 (manuscript Section 4.2):
      M_r robots queueing | M_k picker present | M_u unpicked items |
      M_q items of arrived-but-unassigned orders
    Channels 4-8 -- configuration planes (constant over the grid):
      K/k_max | R/r_max | C/c_max | tau_pick/tau_ref | layout flag
    Channels 9-10 (``plus_agent`` only):
      residual demand of robots awaiting routing | idle-resource positions
    """
    warehouse = env.warehouse
    height, width = warehouse.N_w, warehouse.N_l
    cfg = env.cfg

    m_queue = np.zeros((height, width), dtype=np.float32)
    m_picker = np.zeros((height, width), dtype=np.float32)
    m_unpicked = np.zeros((height, width), dtype=np.float32)
    m_unassigned = np.zeros((height, width), dtype=np.float32)

    for point in warehouse.pick_points_list:
        row, col = warehouse.grid_position(point.point_id)
        m_queue[row, col] = len(point.robot_queue)
        m_picker[row, col] = 0.0 if point.picker is None else 1.0
    for order in env.orders_uncompleted:
        for item in order.unpicked_items:
            row, col = warehouse.grid_position(item.pick_point_id)
            m_unpicked[row, col] += 1.0
    for order in env.orders_unassigned:
        for item in order.unpicked_items:
            row, col = warehouse.grid_position(item.pick_point_id)
            m_unassigned[row, col] += 1.0

    def plane(value: float) -> np.ndarray:
        return np.full((height, width), value, dtype=np.float32)

    channels = [
        m_queue, m_picker, m_unpicked, m_unassigned,
        plane(cfg.n_pickers / cfg.k_max),
        plane(cfg.n_robots / cfg.r_max),
        plane(cfg.robot_capacity / cfg.capacity_max),
        plane(cfg.pick_time / cfg.pick_time_ref),
        plane(1.0 if cfg.layout == "three_cross_aisles" else 0.0),
    ]

    if cfg.state_channels != "base":
        m_residual = np.zeros((height, width), dtype=np.float32)
        m_resource = np.zeros((height, width), dtype=np.float32)
        for robot in env.robots:
            if robot.state != "idle" or not robot.orders:
                continue
            for item in robot.item_pick_order:
                row, col = warehouse.grid_position(item.pick_point_id)
                m_residual[row, col] += 1.0
            point = warehouse.pick_point_by_position.get(tuple(robot.position))
            if point is not None:
                row, col = warehouse.grid_position(point.point_id)
                m_resource[row, col] += 1.0
        for picker in env.pickers:
            if picker.state != "idle":
                continue
            point = warehouse.pick_point_by_position.get(tuple(picker.position))
            if point is not None:
                row, col = warehouse.grid_position(point.point_id)
                m_resource[row, col] += 1.0
        channels.extend([m_residual, m_resource])

    return np.stack(channels, axis=0)


# --------------------------------------------------------------------------- #
def legal_action_indices(env) -> List[int]:
    """Feasible envelope-action indices at the current decision epoch.

    Feasibility rules are those of Section 4.3; resources beyond the scenario's
    (K, R) never appear, which is how the envelope head stays valid for every
    scenario.
    """
    warehouse = env.warehouse
    cfg = env.cfg
    n_points = warehouse.n_pick_points
    index_of = warehouse.pick_point_index

    legal: set[int] = set()
    idle_points = [index_of[p.point_id] for p in warehouse.pick_points_list if p.is_idle]
    if idle_points:
        for picker_idx, picker in enumerate(env.pickers):
            if picker.state != "idle":
                continue
            for point_idx in idle_points:
                legal.add(picker_action_index(picker_idx, point_idx, n_points))

    for robot_idx, robot in enumerate(env.robots):
        if robot.state != "idle" or not robot.orders:
            continue
        if robot.item_pick_order:
            for item in robot.item_pick_order:
                point_idx = index_of[item.pick_point_id]
                legal.add(robot_action_index(robot_idx, point_idx, n_points, cfg.k_max))
        elif robot.pick_point is not None:
            legal.add(robot_depot_index(robot_idx, n_points, cfg.k_max, cfg.r_max))
    return sorted(legal)
