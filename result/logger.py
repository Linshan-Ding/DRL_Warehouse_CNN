"""Training logger: live visdom curves plus a CSV that outlives the run.

visdom answers "is this run diverging right now?"; ``log.csv`` is what the paper
and ``result/plot.py`` read afterwards.  Starting the visdom server is optional
(``python -m visdom.server``); when it is not reachable training continues
without live plots.

Efficiency counters (``sps``, ``wall_clock_s``, ``gpu_mem_gb``) are logged next
to the losses: they are the source of the training-cost figures, which cannot be
recovered after the fact.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from typing import Dict, List, Optional

from configs.config import Config


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def gpu_memory_gb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated()) / 1024 ** 3
    except Exception:
        pass
    return 0.0


class RunLogger:
    """One directory per run: config snapshot, log.csv, checkpoints, figures."""

    def __init__(self, cfg: Config, run_dir: Optional[str] = None):
        self.cfg = cfg
        self.run_dir = run_dir or os.path.join(cfg.run.result_dir, cfg.run.run_name)
        os.makedirs(self.run_dir, exist_ok=True)

        self.csv_path = os.path.join(self.run_dir, "log.csv")
        self._fieldnames: Optional[List[str]] = None
        self._csv_file = None
        self._writer: Optional[csv.DictWriter] = None
        self._start = time.time()

        cfg.dump(os.path.join(self.run_dir, "config_snapshot.yaml"))
        with open(os.path.join(self.run_dir, "run_info.json"), "w", encoding="utf-8") as fh:
            json.dump({"run_name": cfg.run.run_name,
                       "git_commit": _git_commit(),
                       "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "device": cfg.torch_device}, fh, indent=2)

        self.vis = None
        self._windows: Dict[str, object] = {}
        if cfg.run.visdom_enabled:
            self._connect_visdom()

    # ------------------------------------------------------------------ #
    def _connect_visdom(self) -> None:
        try:
            from visdom import Visdom

            self.vis = Visdom(server=self.cfg.run.visdom_server,
                              port=self.cfg.run.visdom_port,
                              env=f"{self.cfg.run.visdom_env}_{self.cfg.run.run_name}",
                              raise_exceptions=False)
            if not self.vis.check_connection():
                print("[logger] visdom server not reachable; continuing without live plots")
                self.vis = None
        except Exception:
            print("[logger] visdom not installed; continuing without live plots")
            self.vis = None

    # ------------------------------------------------------------------ #
    @property
    def elapsed_s(self) -> float:
        return time.time() - self._start

    def log(self, step: int, metrics: Dict[str, float]) -> None:
        row = {"step": step, "wall_clock_s": round(self.elapsed_s, 3)}
        row.update({key: value for key, value in metrics.items() if value is not None})

        # Columns may appear later than the first write (e.g. the periodic
        # evaluation metrics).  When that happens the whole CSV is rewritten
        # with the widened header so no column is ever silently dropped.
        new_keys = [key for key in row if self._fieldnames is None
                    or key not in self._fieldnames]
        if self._writer is None:
            self._fieldnames = list(row)
            self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._csv_file, fieldnames=self._fieldnames)
            self._writer.writeheader()
        elif new_keys:
            self._csv_file.close()
            with open(self.csv_path, "r", newline="", encoding="utf-8") as fh:
                existing = list(csv.DictReader(fh))
            self._fieldnames = self._fieldnames + new_keys
            self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._csv_file, fieldnames=self._fieldnames)
            self._writer.writeheader()
            for old_row in existing:
                self._writer.writerow({key: old_row.get(key, "") for key in self._fieldnames})
        self._writer.writerow({key: row.get(key, "") for key in self._fieldnames})
        self._csv_file.flush()

        if self.vis is not None:
            self._push(step, row)

    def _push(self, step: int, row: Dict[str, float]) -> None:
        import numpy as np

        for key, value in row.items():
            if key in ("step", "wall_clock_s") or not isinstance(value, (int, float)):
                continue
            if key.startswith("curve_"):
                # The representative-case curves share one multi-line window so
                # convergence across cases can be compared at a glance.
                window_key = "representative_cases"
                update = "append" if window_key in self._windows else None
                window = self.vis.line(
                    X=np.array([step]), Y=np.array([float(value)]),
                    win=self._windows.get(window_key), update=update,
                    name=key.replace("curve_", ""),
                    opts=dict(title="Representative cases (greedy F-bar)",
                              xlabel="episode", ylabel="mean flow time",
                              showlegend=True))
                self._windows[window_key] = window
                continue
            update = "append" if key in self._windows else None
            window = self.vis.line(X=np.array([step]), Y=np.array([float(value)]),
                                   win=self._windows.get(key), update=update,
                                   opts=dict(title=key, xlabel="episode", ylabel=key))
            self._windows[key] = window

    def text(self, message: str, win: str = "info") -> None:
        if self.vis is not None:
            self.vis.text(message, win=win)

    # ------------------------------------------------------------------ #
    def path(self, *parts: str) -> str:
        target = os.path.join(self.run_dir, *parts)
        os.makedirs(os.path.dirname(target) or self.run_dir, exist_ok=True)
        return target

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._writer = None
