"""Entities of the human-robot collaborative order picking system.

The classes mirror Section 3.1 of the manuscript: AMRs transport whole orders,
human pickers retrieve items from the storage bins, and the two only interact at
picking points.  The only generalisation with respect to the submitted version
is ``Robot.capacity`` (C): an AMR may take up to C orders per service cycle.
Setting C = 1 recovers assumption (A1) exactly.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

INF = float("inf")


class Depot:
    """Robot dispatch point and order fulfilment station."""

    def __init__(self, position: Tuple[float, float]):
        self.position = position


class StorageBin:
    def __init__(self, bin_id: str, position, item_id: str, pick_point_id: str):
        self.bin_id = bin_id
        self.position = position
        self.item_id = item_id
        self.pick_point_id = pick_point_id


class Item:
    def __init__(self, item_id: str, bin_id: str, position, pick_point_id: str,
                 pick_time: float):
        self.item_id = item_id            # "<nw>-<nl>-<side>-item"
        self.bin_id = bin_id              # "<nw>-<nl>-<side>"
        self.position = position          # (x, y) of the picking point
        self.pick_point_id = pick_point_id
        self.pick_time = pick_time        # tau_pick


class Order:
    def __init__(self, order_id: int, items: Sequence[Item], arrive_time: float):
        self.order_id = order_id
        self.items = list(items)
        self.arrive_time = arrive_time
        self.complete_time: Optional[float] = None
        self.unpicked_items: List[Item] = list(items)
        self.picked_items: List[Item] = []

    @property
    def flow_time(self) -> Optional[float]:
        if self.complete_time is None:
            return None
        return self.complete_time - self.arrive_time


class PickPoint:
    def __init__(self, point_id: str, position, item_ids, storage_bin_ids):
        self.point_id = point_id          # "<nw>-<nl>"
        self.position = position
        self.item_ids = item_ids
        self.storage_bin_ids = storage_bin_ids
        self.robot_queue: List["Robot"] = []
        self.picker: Optional["Picker"] = None

    @property
    def is_idle(self) -> bool:
        """A picker may be dispatched here: robots are waiting and nobody serves them."""
        return len(self.robot_queue) > 0 and self.picker is None


class Robot:
    """Autonomous mobile robot: carries orders, queues at picking points."""

    def __init__(self, robot_id: int, position, speed: float, capacity: int = 1):
        self.robot_id = robot_id
        self.position = position
        self.speed = speed
        self.capacity = capacity          # C, orders per service cycle
        self.state = "idle"               # idle | busy

        self.orders: List[Order] = []
        self.item_pick_order: List[Item] = []   # items still to be collected
        self.pick_point: Optional[PickPoint] = None

        self.move_to_pick_point_time = INF
        self.pick_point_complete_time = INF
        self.move_to_depot_time = INF

    # -- order handling ---------------------------------------------------- #
    @property
    def has_free_capacity(self) -> bool:
        return len(self.orders) < self.capacity

    def assign_order(self, order: Order) -> None:
        self.orders.append(order)
        self.item_pick_order.extend(order.items)

    def release_orders(self) -> List[Order]:
        released, self.orders = self.orders, []
        self.item_pick_order = []
        self.pick_point = None
        return released

    @property
    def items(self) -> List[Item]:
        """Items of the carried orders located at the robot's current picking point."""
        if not self.orders or self.pick_point is None:
            return []
        point_id = self.pick_point.point_id
        return [item for order in self.orders for item in order.items
                if item.pick_point_id == point_id]


class Picker:
    """Human picker: walks to a picking point and serves the whole robot queue."""

    def __init__(self, picker_id: int, speed: float, position=(0.0, 0.0)):
        self.picker_id = picker_id
        self.speed = speed
        self.state = "idle"               # idle | busy
        self.position = position

        self.pick_point: Optional[PickPoint] = None
        self.pick_start_time = INF
        self.pick_end_time = INF
