"""Metrics of one episode.

Two quantities in this project are easy to confuse, so they carry different
names and are logged separately:

``mean_flow_time`` (F-bar)
    Average of (completion time - arrival time) over the completed orders.
    This is the objective of the manuscript, Eq. (13).

``decision_time_ms`` (D-bar, computational)
    Wall-clock time a method needs to produce one decision, in milliseconds.
    This is what "average decision time" is *defined* as in the manuscript.

``sim_time_per_decision``
    Simulated seconds elapsed per decision epoch, i.e. makespan / #epochs.
    It measures how coarse a method's decisions are, not how fast the method is,
    and must never be reported as a computation time.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Optional


@dataclass
class EpisodeMetrics:
    mean_flow_time: float = 0.0        # F-bar over completed orders
    makespan: float = 0.0              # simulated horizon
    n_completed: int = 0
    n_orders: int = 0
    n_decisions: int = 0
    decision_time_ms: float = 0.0      # true wall-clock per decision
    sim_time_per_decision: float = 0.0 # makespan / #decision epochs
    solve_wall_clock_s: float = 0.0    # wall clock of the whole episode
    reward_sum: float = 0.0
    extra: Dict[str, float] = field(default_factory=dict)

    def as_row(self) -> Dict[str, float]:
        row = asdict(self)
        row.pop("extra")
        row.update(self.extra)
        return row


def episode_metrics(env, decision_wall_clock_s: float, episode_wall_clock_s: float,
                    reward_sum: float = 0.0,
                    extra: Optional[Dict[str, float]] = None) -> EpisodeMetrics:
    summary = env.episode_summary()
    n_decisions = int(summary["n_decisions"])
    return EpisodeMetrics(
        mean_flow_time=float(summary["mean_flow_time"]),
        makespan=float(summary["makespan"]),
        n_completed=int(summary["n_completed"]),
        n_orders=int(summary["n_orders"]),
        n_decisions=n_decisions,
        decision_time_ms=(1000.0 * decision_wall_clock_s / n_decisions) if n_decisions else 0.0,
        sim_time_per_decision=float(summary["sim_time_per_decision"]),
        solve_wall_clock_s=float(episode_wall_clock_s),
        reward_sum=float(reward_sum),
        extra=dict(extra or {}),
    )
