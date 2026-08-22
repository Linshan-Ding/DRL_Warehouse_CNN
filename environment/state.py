"""State representation, action space and feasibility mask (manuscript Section 4.2-4.3).

Two things are worth stating explicitly, because Reviewer #2 asked about them:

* the state tensor ``s_t`` aggregates everything **per picking point** -- it
  carries no robot or picker identity;
* the feasibility mask ``M_t`` is built from per-resource information (which
  robot still needs which locations, which picker is idle).

The policy therefore acts on the pair ``o_t = (s_t, M_t)``; the mask is part of
the observation, not a post-processing step.  The optional ``plus_agent``
channel set makes the per-resource information visible to the network as well
and exists so that this design choice can be tested (experiment E5).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# flat action index layout (identical to the submitted implementation)
#   [0, K*P)                 picker k -> picking point j
#   [K*P, (K+R)*P)           robot r  -> picking point j
#   [(K+R)*P, (K+R)*P + R)   robot r  -> depot
# --------------------------------------------------------------------------- #
DEPOT_TARGET = -1


def picker_action_index(picker_idx: int, point_idx: int, n_points: int) -> int:
    return picker_idx * n_points + point_idx


def robot_action_index(robot_idx: int, point_idx: int, n_points: int, n_pickers: int) -> int:
    return n_pickers * n_points + robot_idx * n_points + point_idx


def robot_depot_index(robot_idx: int, n_points: int, n_pickers: int, n_robots: int) -> int:
    return (n_pickers + n_robots) * n_points + robot_idx


def n_actions(n_points: int, n_pickers: int, n_robots: int) -> int:
    return (n_pickers + n_robots) * n_points + n_robots


def decode_action(index: int, n_points: int, n_pickers: int, n_robots: int):
    """Return ``("picker", picker_idx, point_idx)`` or ``("robot", robot_idx, target)``.

    ``target`` is a picking-point index, or ``DEPOT_TARGET`` for a return trip.
    """
    picker_block = n_pickers * n_points
    robot_block = picker_block + n_robots * n_points
    if index < picker_block:
        return "picker", index // n_points, index % n_points
    if index < robot_block:
        offset = index - picker_block
        return "robot", offset // n_points, offset % n_points
    return "robot", index - robot_block, DEPOT_TARGET


# --------------------------------------------------------------------------- #
# state tensor
# --------------------------------------------------------------------------- #
def build_state(env) -> np.ndarray:
    """Multi-channel state tensor, shape ``(n_channels, N_w, N_l)``.

    Base channels (Section 4.2):
      0  M_r  robots queueing at the picking point
      1  M_k  picker present (0/1)
      2  M_u  unpicked items of arrived, uncompleted orders
      3  M_q  items of orders that arrived but have no robot yet

    Additional channels for ``state_channels: plus_agent`` (experiment E5):
      4  residual demand of the robots that are currently awaiting a routing
         decision -- the per-robot information the mask uses
      5  positions of the idle resources awaiting a decision
    """
    warehouse = env.warehouse
    height, width = warehouse.N_w, warehouse.N_l

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

    channels = [m_queue, m_picker, m_unpicked, m_unassigned]

    if env.cfg.state_channels != "base":
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
# feasibility mask
# --------------------------------------------------------------------------- #
def legal_action_indices(env) -> List[int]:
    """Indices of the feasible actions at the current decision epoch.

    Feasibility rules (Section 4.3):

    * picker action ``(k, j)``: picker ``k`` is idle, point ``j`` has at least
      one waiting robot and no picker assigned;
    * robot action ``(r, j)``: robot ``r`` is idle, carries at least one order
      and still requires service at ``j``;
    * robot action ``(r, depot)``: robot ``r`` has collected every required item
      and stands at a picking point.
    """
    warehouse = env.warehouse
    n_points = warehouse.n_pick_points
    n_pickers = len(env.pickers)
    n_robots = len(env.robots)
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
                legal.add(robot_action_index(robot_idx, point_idx, n_points, n_pickers))
        elif robot.pick_point is not None:
            legal.add(robot_depot_index(robot_idx, n_points, n_pickers, n_robots))

    return sorted(legal)


def action_description(env, index: int) -> str:
    kind, actor, target = decode_action(index, env.warehouse.n_pick_points,
                                        len(env.pickers), len(env.robots))
    if target == DEPOT_TARGET:
        return f"robot {actor} -> depot"
    point = env.warehouse.pick_points_list[target]
    return f"{kind} {actor} -> point {point.point_id}"
