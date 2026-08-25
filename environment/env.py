"""Discrete-event simulation of the human-robot collaborative picking system.

The environment exposes the standard contract used by every method in this
project::

    state = env.reset(orders)
    state, reward, done, info = env.step(action_index)
    mask  = env.legal_actions()

Decisions are event driven (Section 4.1): the simulator advances until either an
idle picker can be dispatched to a point where robots are waiting, or an idle
robot that carries orders needs a routing decision.  Exactly one resource moves
per decision epoch.

The reward is the improvement of the mean order flow time between consecutive
decision epochs, Eq. (14)::

    r_t = F_bar_{t-1} - F_bar_t

Because ``F_bar_0 = 0``, the *undiscounted* return of an episode equals
``-F_bar_final``; ``tools/selfcheck.py`` asserts this identity.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from configs.config import EnvCfg
from environment import state as state_mod
from environment.entities import INF, Order, Picker, Robot
from environment.problem import Warehouse, completed_mean_flow_time, mean_flow_time

# Pickers start spread along the grid, one every PICKER_SPACING picking points.
PICKER_SPACING = 5


class WarehouseEnv:
    """Event-driven simulator of one picking shift."""

    def __init__(self, cfg: EnvCfg):
        self.cfg = cfg
        self.warehouse = Warehouse(cfg)

        self.robots: List[Robot] = []
        self.pickers: List[Picker] = []

        self.orders: List[Order] = []
        self.orders_not_arrived: List[Order] = []
        self.orders_unassigned: List[Order] = []
        self.orders_uncompleted: List[Order] = []
        self.orders_completed: List[Order] = []

        self.current_time = 0.0
        self.last_decision_time = 0.0
        self.order_handle_time = 0.0
        self.decision_count = 0
        self.done = False

    # ------------------------------------------------------------------ #
    # properties
    # ------------------------------------------------------------------ #
    @property
    def depot(self):
        return self.warehouse.depot

    @property
    def n_actions(self) -> int:
        return state_mod.n_actions(self.warehouse.n_pick_points,
                                   self.cfg.k_max, self.cfg.r_max)

    @property
    def state_shape(self):
        return (self.cfg.n_state_channels, self.warehouse.N_w, self.warehouse.N_l)

    @property
    def idle_pickers(self) -> List[Picker]:
        return [p for p in self.pickers if p.state == "idle"]

    @property
    def idle_pick_points(self):
        return [p for p in self.warehouse.pick_points_list if p.is_idle]

    @property
    def robots_needing_routing(self) -> List[Robot]:
        return [r for r in self.robots if r.state == "idle" and r.orders]

    # ------------------------------------------------------------------ #
    # setup
    # ------------------------------------------------------------------ #
    def _create_resources(self) -> None:
        depot_position = self.warehouse.depot.position
        self.robots = [Robot(i, depot_position, self.cfg.robot_speed,
                             self.cfg.robot_capacity)
                       for i in range(self.cfg.n_robots)]

        points = self.warehouse.pick_points_list
        self.pickers = []
        for i in range(self.cfg.n_pickers):
            picker = Picker(i, self.cfg.picker_speed)
            # Spread the pickers over the grid; clamped so that configurations
            # with many pickers remain valid.
            slot = min(i * PICKER_SPACING, len(points) - 1)
            picker.position = points[slot].position
            picker.pick_point = points[slot]
            self.pickers.append(picker)

    def reset(self, orders: Sequence[Order]) -> np.ndarray:
        self.current_time = 0.0
        self.last_decision_time = 0.0
        self.order_handle_time = 0.0
        self.decision_count = 0
        self.done = False
        self._create_resources()

        self.orders = list(orders)
        self.orders_not_arrived = sorted(self.orders, key=lambda o: o.arrive_time)
        self.orders_unassigned = []
        self.orders_uncompleted = []
        self.orders_completed = []

        for point in self.warehouse.pick_points_list:
            point.robot_queue = []
            point.picker = None

        self._advance_to_decision_epoch()
        return self.observe()

    # ------------------------------------------------------------------ #
    # observation
    # ------------------------------------------------------------------ #
    def observe(self) -> np.ndarray:
        return state_mod.build_state(self)

    def legal_actions(self) -> List[int]:
        return state_mod.legal_action_indices(self)

    # ------------------------------------------------------------------ #
    # simulation
    # ------------------------------------------------------------------ #
    def _assign_orders_at_depot(self) -> None:
        """FIFO order release at the depot.

        Order-to-robot assignment is a system rule, not a policy decision: the
        earliest waiting order goes to a robot standing idle at the depot, which
        takes up to ``C`` orders per service cycle.
        """
        depot_position = self.warehouse.depot.position
        waiting = [r for r in self.robots
                   if r.state == "idle" and r.has_free_capacity
                   and r.position == depot_position]
        if not waiting or not self.orders_unassigned:
            return
        for robot in waiting:
            while robot.has_free_capacity and self.orders_unassigned:
                robot.assign_order(self.orders_unassigned.pop(0))

    def _advance_to_decision_epoch(self) -> None:
        """Advance the simulation clock until a decision is required."""
        while not self.done:
            self._assign_orders_at_depot()

            # (1) picker assignment decision
            if self.idle_pickers and self.idle_pick_points:
                return
            # (2) robot routing decision
            if self.robots_needing_routing:
                return

            next_time = self._next_event_time()
            if next_time is None:
                return
            self.current_time = next_time
            self._handle_events()

    def _next_event_time(self) -> Optional[float]:
        events: List[float] = []
        if self.orders_not_arrived:
            events.append(self.orders_not_arrived[0].arrive_time)
        for robot in self.robots:
            for stamp in (robot.move_to_pick_point_time,
                          robot.pick_point_complete_time,
                          robot.move_to_depot_time):
                if stamp > self.current_time:
                    events.append(stamp)
        for picker in self.pickers:
            for stamp in (picker.pick_start_time, picker.pick_end_time):
                if stamp > self.current_time:
                    events.append(stamp)

        events = [t for t in events if t != INF]
        if events:
            return min(events)

        self.done = True
        if self.orders_not_arrived or self.orders_unassigned or self.orders_uncompleted:
            raise RuntimeError(
                "simulation stalled with work remaining: "
                f"not_arrived={len(self.orders_not_arrived)} "
                f"unassigned={len(self.orders_unassigned)} "
                f"uncompleted={len(self.orders_uncompleted)} at t={self.current_time}")
        return None

    def _handle_events(self) -> None:
        now = self.current_time

        # A. order arrivals
        while self.orders_not_arrived and now >= self.orders_not_arrived[0].arrive_time:
            order = self.orders_not_arrived.pop(0)
            self.orders_unassigned.append(order)
            self.orders_uncompleted.append(order)

        # B. robots reaching a picking point join its queue
        for robot in self.robots:
            if now == robot.move_to_pick_point_time:
                point = robot.pick_point
                point.robot_queue.append(robot)
                robot.position = point.position
                robot.move_to_pick_point_time = INF

        # C. pickers reaching their picking point start serving
        for picker in self.pickers:
            if now == picker.pick_start_time:
                picker.pick_start_time = INF

        # D1. a robot has been served at its picking point
        for robot in self.robots:
            if now == robot.pick_point_complete_time:
                robot.pick_point_complete_time = INF
                point = robot.pick_point
                if robot in point.robot_queue:
                    point.robot_queue.remove(robot)
                for item in robot.items:
                    for order in robot.orders:
                        if item in order.unpicked_items:
                            order.unpicked_items.remove(item)
                            order.picked_items.append(item)
                    if item in robot.item_pick_order:
                        robot.item_pick_order.remove(item)
                robot.state = "idle"

        # D2. a picker finished the whole queue
        for picker in self.pickers:
            if now == picker.pick_end_time:
                picker.pick_end_time = INF
                picker.state = "idle"
                if picker.pick_point is not None:
                    picker.pick_point.picker = None
                    picker.pick_point = None

        # E. robots returning to the depot complete their orders
        for robot in self.robots:
            if now == robot.move_to_depot_time:
                robot.move_to_depot_time = INF
                robot.state = "idle"
                robot.position = self.warehouse.depot.position
                for order in robot.release_orders():
                    order.complete_time = now
                    if order in self.orders_uncompleted:
                        self.orders_uncompleted.remove(order)
                        self.orders_completed.append(order)

    # ------------------------------------------------------------------ #
    def step(self, action_index: int):
        """Apply one action, advance to the next decision epoch, return the transition."""
        kind, actor, target = state_mod.decode_action(
            int(action_index), self.warehouse.n_pick_points,
            self.cfg.k_max, self.cfg.r_max)

        if kind == "picker":
            self._apply_picker_action(self.pickers[actor],
                                      self.warehouse.pick_points_list[target])
        else:
            robot = self.robots[actor]
            destination = (self.warehouse.depot if target == state_mod.DEPOT_TARGET
                           else self.warehouse.pick_points_list[target])
            self._apply_robot_action(robot, destination)

        self.decision_count += 1
        self._advance_to_decision_epoch()

        dt = self.current_time - self.last_decision_time
        reward = self.compute_reward()
        info = {"dt": float(dt), "makespan": float(self.current_time)}
        return self.observe(), reward, self.done, info

    def _apply_picker_action(self, picker: Picker, point) -> None:
        picker.state = "busy"
        picker.pick_point = point
        point.picker = picker

        travel_time = self.warehouse.distance(picker.position, point.position) / picker.speed
        picker.pick_start_time = self.current_time + travel_time
        picker.position = point.position

        # Items of every robot queueing here are picked sequentially, so the
        # service times accumulate along the queue.
        cumulative = 0.0
        for robot in point.robot_queue:
            cumulative += sum(item.pick_time for item in robot.items)
            robot.pick_point_complete_time = picker.pick_start_time + cumulative
        picker.pick_end_time = picker.pick_start_time + cumulative

    def _apply_robot_action(self, robot: Robot, destination) -> None:
        if not robot.orders:
            raise RuntimeError(f"robot {robot.robot_id} has no order to serve")
        robot.state = "busy"
        travel_time = self.warehouse.distance(robot.position, destination.position) / robot.speed

        if destination is self.warehouse.depot:
            # One packing time per order carried on this trip.
            packing = self.cfg.pack_time * len(robot.orders)
            robot.move_to_depot_time = self.current_time + travel_time + packing
            robot.pick_point = None
        else:
            robot.pick_point = destination
            robot.move_to_pick_point_time = self.current_time + travel_time

    # ------------------------------------------------------------------ #
    def compute_reward(self) -> float:
        """r_t = F_bar_{t-1} - F_bar_t, Eq. (14)."""
        average = mean_flow_time(self.orders_completed, self.orders_uncompleted,
                                 self.current_time)
        reward = self.order_handle_time - average
        self.last_decision_time = self.current_time
        self.order_handle_time = average
        return reward

    # ------------------------------------------------------------------ #
    def episode_summary(self) -> Dict[str, float]:
        """Metrics of a finished episode."""
        return {
            "mean_flow_time": completed_mean_flow_time(self.orders_completed),
            "makespan": float(self.current_time),
            "n_completed": float(len(self.orders_completed)),
            "n_orders": float(len(self.orders)),
            "n_decisions": float(self.decision_count),
            "sim_time_per_decision": (float(self.current_time) / self.decision_count
                                      if self.decision_count else 0.0),
        }
