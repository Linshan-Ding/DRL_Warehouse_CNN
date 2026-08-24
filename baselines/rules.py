"""Combined priority dispatching rules (manuscript Section 5.3).

Each benchmark strategy pairs a picker-assignment rule with a robot-routing
rule and is written ``"<picker rule>-<robot rule>"``:

* ``MQ``    (picker)  the picking point with the largest number of waiting robots
* ``MI``    (picker)  the picking point with the largest number of pending items
* ``ND``    (robot)   the required picking point closest to the robot
* ``MinRQ`` (robot)   the required picking point with the fewest waiting robots
* ``MI``    (robot)   the required picking point with the most pending items

The five strategies reported in the manuscript are MQ-ND, MQ-MinRQ, MQ-MI,
MI-MinRQ and MI-MI.

At every decision epoch the simulator offers a picker assignment whenever one is
feasible, so the policy applies the picker rule first and falls back to the
robot rule otherwise; ties are broken by the lowest picking-point index, which
makes every rule fully deterministic.

The selection logic is written against attribute names shared by the current and
the original simulator, so ``tools/selfcheck.py`` can drive both with the very
same rule and compare them event by event.
"""
from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Sequence

from environment.state import (DEPOT_TARGET, picker_action_index,
                               robot_action_index, robot_depot_index)

PICKER_RULES = ("MQ", "MI")
ROBOT_RULES = ("ND", "MinRQ", "MI")
PAPER_RULES = ("MQ-ND", "MQ-MinRQ", "MQ-MI", "MI-MinRQ", "MI-MI")


class Choice(NamedTuple):
    kind: str           # "picker" | "robot"
    actor: int          # index into env.pickers / env.robots
    target: int         # picking-point index, or DEPOT_TARGET


# --------------------------------------------------------------------------- #
# helpers that work on both the current and the original simulator
# --------------------------------------------------------------------------- #
def _points(env) -> List:
    warehouse = getattr(env, "warehouse", None)
    return warehouse.pick_points_list if warehouse is not None else env.pick_points_list


def _distance(env, position1, position2) -> float:
    warehouse = getattr(env, "warehouse", None)
    if warehouse is not None:
        return warehouse.distance(position1, position2)
    x1, y1 = position1
    x2, y2 = position2
    if x1 == x2:
        return abs(y1 - y2)
    horizontal = abs(x1 - x2)
    top = env.S_b * 1.5 + env.N_l * env.S_l
    return min(abs(y1 - env.S_b / 2) + abs(y2 - env.S_b / 2) + horizontal,
               abs(y1 - top) + abs(y2 - top) + horizontal)


def _robot_has_orders(robot) -> bool:
    orders = getattr(robot, "orders", None)
    if orders is not None:
        return bool(orders)
    return getattr(robot, "order", None) is not None


def _pending_items(env) -> Dict[str, int]:
    """Number of not-yet-picked items per picking point over arrived orders."""
    counts: Dict[str, int] = {}
    for order in env.orders_uncompleted:
        for item in order.unpicked_items:
            counts[item.pick_point_id] = counts.get(item.pick_point_id, 0) + 1
    return counts


# --------------------------------------------------------------------------- #
class RulePolicy:
    """A combined priority dispatching rule."""

    def __init__(self, name: str):
        picker_rule, _, robot_rule = name.partition("-")
        if picker_rule not in PICKER_RULES or robot_rule not in ROBOT_RULES:
            raise ValueError(
                f"unknown rule {name!r}; expected <{'|'.join(PICKER_RULES)}>-"
                f"<{'|'.join(ROBOT_RULES)}>")
        self.name = name
        self.picker_rule = picker_rule
        self.robot_rule = robot_rule

    # -- picker assignment ------------------------------------------------ #
    def _choose_picker_action(self, env) -> Optional[Choice]:
        idle_pickers = [i for i, p in enumerate(env.pickers) if p.state == "idle"]
        if not idle_pickers:
            return None
        points = _points(env)
        candidates = [i for i, point in enumerate(points) if point.is_idle]
        if not candidates:
            return None

        if self.picker_rule == "MQ":
            key = lambda i: len(points[i].robot_queue)
        else:                                    # MI
            pending = _pending_items(env)
            key = lambda i: pending.get(points[i].point_id, 0)

        best = max(candidates, key=lambda i: (key(i), -i))
        return Choice("picker", idle_pickers[0], best)

    # -- robot routing ---------------------------------------------------- #
    def _choose_robot_action(self, env) -> Optional[Choice]:
        points = _points(env)
        index_of = {point.point_id: i for i, point in enumerate(points)}

        for robot_idx, robot in enumerate(env.robots):
            if robot.state != "idle" or not _robot_has_orders(robot):
                continue
            if not robot.item_pick_order:
                if robot.pick_point is not None:
                    return Choice("robot", robot_idx, DEPOT_TARGET)
                continue

            candidates = sorted({index_of[item.pick_point_id]
                                 for item in robot.item_pick_order})
            if self.robot_rule == "ND":
                key = lambda i: -_distance(env, robot.position, points[i].position)
            elif self.robot_rule == "MinRQ":
                key = lambda i: -len(points[i].robot_queue)
            else:                                # MI
                pending = _pending_items(env)
                key = lambda i: pending.get(points[i].point_id, 0)

            best = max(candidates, key=lambda i: (key(i), -i))
            return Choice("robot", robot_idx, best)
        return None

    # -- public API ------------------------------------------------------- #
    def choose(self, env) -> Choice:
        choice = self._choose_picker_action(env) or self._choose_robot_action(env)
        if choice is None:
            raise RuntimeError("no feasible action at this decision epoch")
        return choice

    def act(self, env) -> int:
        """Flat action index for the current simulator."""
        return to_action_index(env, self.choose(env))


def to_action_index(env, choice: Choice) -> int:
    """Flat envelope-action index for the current simulator."""
    n_points = len(_points(env))
    k_max = env.cfg.k_max
    r_max = env.cfg.r_max
    if choice.kind == "picker":
        return picker_action_index(choice.actor, choice.target, n_points)
    if choice.target == DEPOT_TARGET:
        return robot_depot_index(choice.actor, n_points, k_max, r_max)
    return robot_action_index(choice.actor, choice.target, n_points, k_max)


def build(names: Sequence[str] | None = None) -> List[RulePolicy]:
    return [RulePolicy(name) for name in (names or PAPER_RULES)]
