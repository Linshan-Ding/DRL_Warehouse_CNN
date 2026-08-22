"""Problem definition: warehouse geometry, travel distances and the objective.

This module holds the "physical rules" of the system and nothing else -- it
knows nothing about MDPs, policies or learning.  Equations refer to the
manuscript:

* Eq. (1)  picking-point coordinates
* Eq. (2)  rectilinear travel distance through the bottom/top cross-aisles
* Eq. (13) mean order flow time, the objective being minimised
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

from configs.config import EnvCfg
from environment.entities import Depot, Item, PickPoint, StorageBin

SIDES = ("left", "right")


class Warehouse:
    """Static layout: picking points, storage bins, items and the depot."""

    def __init__(self, cfg: EnvCfg):
        self.cfg = cfg
        self.N_w = cfg.n_aisles
        self.N_l = cfg.n_positions
        self.S_l = cfg.shelf_length
        self.S_w = cfg.shelf_width
        self.S_a = cfg.aisle_width
        self.S_b = cfg.bottom_aisle_width
        self.S_d = cfg.entrance_width
        self.layout = cfg.layout

        self.depot = Depot(tuple(cfg.depot_position))
        self.pick_points: Dict[str, PickPoint] = {}
        self.pick_points_list: List[PickPoint] = []
        self.pick_point_by_position: Dict[Tuple[float, float], PickPoint] = {}
        self.pick_point_index: Dict[str, int] = {}
        self.storage_bins: Dict[str, StorageBin] = {}
        self.items: Dict[str, Item] = {}

        self._build_grid()

        # Cross-aisle centre lines used by the travel-distance model.
        self._y_bottom = self.S_b / 2.0
        self._y_top = self.S_b * 1.5 + self.N_l * self.S_l
        self._y_mid = self.S_b + self.N_l * self.S_l / 2.0

    # ------------------------------------------------------------------ #
    def _build_grid(self) -> None:
        for nw in range(1, self.N_w + 1):
            for nl in range(1, self.N_l + 1):
                # Eq. (1)
                x = self.S_d + (2 * nw - 1) * self.S_w + (2 * nw - 1) / 2 * self.S_a
                y = self.S_b + (2 * nl - 1) / 2 * self.S_l
                position = (x, y)
                point_id = f"{nw}-{nl}"

                item_ids, bin_ids = [], []
                for side in SIDES:
                    bin_id = f"{point_id}-{side}"
                    item_id = f"{bin_id}-item"
                    self.storage_bins[bin_id] = StorageBin(bin_id, position, item_id, point_id)
                    self.items[item_id] = Item(item_id, bin_id, position, point_id,
                                               self.cfg.pick_time)
                    item_ids.append(item_id)
                    bin_ids.append(bin_id)

                point = PickPoint(point_id, position, item_ids, bin_ids)
                self.pick_points[point_id] = point
                self.pick_point_index[point_id] = len(self.pick_points_list)
                self.pick_points_list.append(point)
                self.pick_point_by_position[position] = point

    # ------------------------------------------------------------------ #
    def distance(self, position1, position2) -> float:
        """Shortest rectilinear travel distance between two positions, Eq. (2).

        Within an aisle the distance reduces to the vertical gap; between aisles
        the path detours through the bottom or the top cross-aisle, whichever is
        shorter.  With ``layout: three_cross_aisles`` an additional middle
        cross-aisle is offered as a third candidate (experiment E8); it is
        modelled as a zero-width corridor, so no other coordinate changes.
        """
        x1, y1 = position1
        x2, y2 = position2
        if x1 == x2:
            return abs(y1 - y2)

        horizontal = abs(x1 - x2)
        candidates = [
            abs(y1 - self._y_bottom) + abs(y2 - self._y_bottom) + horizontal,
            abs(y1 - self._y_top) + abs(y2 - self._y_top) + horizontal,
        ]
        if self.layout == "three_cross_aisles":
            candidates.append(abs(y1 - self._y_mid) + abs(y2 - self._y_mid) + horizontal)
        return min(candidates)

    # ------------------------------------------------------------------ #
    def grid_position(self, point_id: str) -> Tuple[int, int]:
        """"<nw>-<nl>" -> zero-based (row, column) of the state tensor."""
        nw, nl = point_id.split("-")
        return int(nw) - 1, int(nl) - 1

    @property
    def n_pick_points(self) -> int:
        return len(self.pick_points_list)

    def item_list(self) -> List[Item]:
        return list(self.items.values())


# --------------------------------------------------------------------------- #
# objective
# --------------------------------------------------------------------------- #
def mean_flow_time(completed: Sequence, uncompleted: Sequence, now: float) -> float:
    """Mean order flow time at time ``now`` -- Eq. (13).

    Completed orders contribute (completion - arrival); orders that have arrived
    but are still in the system contribute (now - arrival).  Orders that have
    not arrived yet are excluded from both the numerator and the denominator.
    """
    total = sum(order.complete_time - order.arrive_time for order in completed)
    total += sum(now - order.arrive_time for order in uncompleted)
    n = len(completed) + len(uncompleted)
    return total / max(1, n)


def completed_mean_flow_time(completed: Iterable) -> float:
    """Mean flow time over completed orders only -- the metric reported as F-bar."""
    flows = [order.complete_time - order.arrive_time for order in completed
             if order.complete_time is not None]
    if not flows:
        return 0.0
    return float(sum(flows) / len(flows))
