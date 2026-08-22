"""Structured configuration for the SAPPO human-robot collaborative picking project.

All tunable quantities live in the YAML files next to this module; the code only
reads them from here.  Several YAML files can be stacked (later files override
earlier ones) and individual fields can still be overridden on the command line::

    python train.py --config configs/env.yaml configs/exp/gamma_1.0.yaml \
                    --algo.gamma 1.0 --run-name my_run

Default values reproduce the setting reported in the manuscript
(Table 3 for the system/instance parameters, Table 4 for the SAPPO
hyperparameters).  Nothing in this file may be hard-coded elsewhere.
"""
from __future__ import annotations

import argparse
import copy
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Sequence, Tuple

import yaml

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CONFIG_DIR)

DEFAULT_CONFIG_FILES = ("env.yaml", "instance.yaml", "algo.yaml")


@dataclass
class EnvCfg:
    """Warehouse geometry, resources and service times (manuscript Table 3)."""

    # --- layout: picking-point grid ---
    n_aisles: int = 9                 # N_w
    n_positions: int = 20             # N_l, storage positions per aisle
    shelf_length: float = 1.0         # S_l
    shelf_width: float = 1.0          # S_w
    aisle_width: float = 2.0          # S_a and S_b (bottom cross-aisle)
    bottom_aisle_width: float = 2.0   # S_b
    entrance_width: float = 2.0       # S_d
    depot_position: Tuple[float, float] = (18.0, 0.0)
    # "two_cross_aisles" reproduces Eq. (2); "three_cross_aisles" adds a middle
    # cross-aisle and is only used by the layout sensitivity experiment (E8).
    layout: str = "two_cross_aisles"

    # --- mobile resources ---
    n_robots: int = 6                 # R
    n_pickers: int = 3                # K
    robot_speed: float = 3.0          # v_r  (m/s)
    picker_speed: float = 0.75        # v_k  (m/s)
    # Maximum number of orders an AMR carries in one service cycle.
    # C = 1 is assumption (A1) of the submitted manuscript; C > 1 is the
    # generalisation requested by Reviewer #1 (experiment E7).
    robot_capacity: int = 1           # C

    # --- service times ---
    pick_time: float = 10.0           # tau_pick, per item
    pack_time: float = 20.0           # tau_pack, per order

    # --- observation ---
    # "base"       -> the four channels of the manuscript (Section 4.2)
    # "plus_agent" -> base + 2 channels exposing the per-resource information
    #                 that currently only the action mask sees (experiment E5)
    state_channels: str = "base"

    # Safety bound on the number of decision epochs in one episode.
    max_steps: int = 20000

    @property
    def n_pick_points(self) -> int:
        return self.n_aisles * self.n_positions

    @property
    def n_state_channels(self) -> int:
        return 4 if self.state_channels == "base" else 6

    @property
    def n_actions(self) -> int:
        """|A| = K*N_w*N_l + R*(N_w*N_l + 1), i.e. Eq. in Section 4.8."""
        return (self.n_pickers + self.n_robots) * self.n_pick_points + self.n_robots


@dataclass
class InstanceCfg:
    """Order-arrival parameter table and the fixed evaluation instance tiers."""

    n_orders: int = 100
    min_items_per_order: int = 5
    max_items_per_order: int = 5
    # Mean inter-arrival time 1/lambda used when a single value is needed
    # (training and the smoke tests).
    mean_interarrival: float = 40.0

    # --- fixed evaluation / validation instances (materialised as CSV) ---
    # The 27 cases of the manuscript are the full cross product below;
    # the order stream only depends on the arrival rate, so one CSV per lambda.
    main_interarrivals: List[float] = field(default_factory=lambda: [20.0, 40.0, 60.0])
    main_pickers: List[int] = field(default_factory=lambda: [1, 2, 3])
    main_robots: List[int] = field(default_factory=lambda: [2, 4, 6])
    n_main_streams: int = 3        # one order stream per arrival rate
    n_val: int = 3                 # validation streams, never reported
    val_interarrival: float = 40.0
    # "large" tier: arrival rates outside the parameter table above, used to
    # check behaviour beyond the training distribution.
    large_interarrivals: List[float] = field(default_factory=lambda: [10.0, 100.0])
    large_n_orders: int = 200

    # How training episodes obtain their order stream.
    #   "fixed"   : always the materialised main-tier stream whose arrival rate
    #               matches ``mean_interarrival`` -- this is what produced the
    #               submitted results and is required to reproduce them (E0);
    #   "sampled" : a fresh stream drawn from the parameter table every episode.
    train_mode: str = "fixed"

    instances_dir: str = "data/instances"


@dataclass
class AlgoCfg:
    """SAPPO hyperparameters (manuscript Table 4, verbatim)."""

    actor_lr: float = 1e-4
    critic_lr: float = 3e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.2
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 2
    minibatch_size: int = 64
    ratio_cap: float = 5.0
    advantage_clip: float = 5.0

    # network sizes
    cnn_output_dim: int = 256
    policy_hidden: int = 2048
    value_hidden: int = 1024

    # training schedule
    n_episodes: int = 2000
    update_every_episodes: int = 1
    eval_interval: int = 10
    n_eval_episodes: int = 3

    device: str = "auto"             # auto | cpu | cuda


@dataclass
class RunCfg:
    """Bookkeeping for one training or evaluation run."""

    run_name: str = "run"
    result_dir: str = "result"
    visdom_enabled: bool = True
    visdom_server: str = "http://localhost"
    visdom_port: int = 8097
    visdom_env: str = "sappo"


@dataclass
class Config:
    env: EnvCfg = field(default_factory=EnvCfg)
    instance: InstanceCfg = field(default_factory=InstanceCfg)
    algo: AlgoCfg = field(default_factory=AlgoCfg)
    run: RunCfg = field(default_factory=RunCfg)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def dump(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, allow_unicode=True, sort_keys=False)

    @property
    def torch_device(self) -> str:
        if self.algo.device != "auto":
            return self.algo.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:  # pragma: no cover - torch is a hard dependency at run time
            return "cpu"


# --------------------------------------------------------------------------- #
# loading / merging
# --------------------------------------------------------------------------- #
def _deep_update(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _coerce(value: Any, template: Any) -> Any:
    """Cast a YAML/CLI value to the type of the dataclass default."""
    if template is None:
        return value
    if isinstance(template, tuple):
        return tuple(value)
    if isinstance(template, bool):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if isinstance(template, list):
        if isinstance(value, str):
            value = [v for v in value.replace(",", " ").split() if v]
        inner = template[0] if template else None
        return [_coerce(v, inner) for v in value]
    if isinstance(template, int) and not isinstance(template, bool):
        return int(float(value))
    if isinstance(template, float):
        return float(value)
    if isinstance(template, str):
        return str(value)
    return value


def _build(cls, values: Dict[str, Any]):
    defaults = cls()
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise KeyError(f"unknown option(s) for {cls.__name__}: {sorted(unknown)}")
    kwargs = {k: _coerce(v, getattr(defaults, k)) for k, v in values.items()}
    return cls(**kwargs)


def config_from_dict(raw: Dict[str, Any]) -> Config:
    raw = copy.deepcopy(raw or {})
    unknown = set(raw) - {f.name for f in fields(Config)}
    if unknown:
        raise KeyError(f"unknown config section(s): {sorted(unknown)}")
    return Config(
        env=_build(EnvCfg, raw.get("env", {})),
        instance=_build(InstanceCfg, raw.get("instance", {})),
        algo=_build(AlgoCfg, raw.get("algo", {})),
        run=_build(RunCfg, raw.get("run", {})),
    )


def load_config(paths: Sequence[str] | None = None,
                overrides: Dict[str, Any] | None = None) -> Config:
    """Stack YAML files (later wins), then apply dotted-key overrides."""
    merged: Dict[str, Any] = {}
    for name in DEFAULT_CONFIG_FILES:
        default_path = os.path.join(CONFIG_DIR, name)
        if os.path.exists(default_path):
            with open(default_path, "r", encoding="utf-8") as fh:
                _deep_update(merged, yaml.safe_load(fh) or {})

    for path in paths or []:
        if not os.path.exists(path):
            raise FileNotFoundError(f"config file not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            _deep_update(merged, yaml.safe_load(fh) or {})

    for dotted, value in (overrides or {}).items():
        section, _, key = dotted.partition(".")
        if not key:
            raise KeyError(f"override must be of the form section.key, got {dotted!r}")
        merged.setdefault(section, {})[key] = value

    return config_from_dict(merged)


def parse_overrides(extra: Sequence[str]) -> Dict[str, Any]:
    """Turn leftover CLI tokens into dotted overrides.

    Accepts both ``--algo.gamma 1.0`` and ``--algo.gamma=1.0``.
    """
    overrides: Dict[str, Any] = {}
    i = 0
    while i < len(extra):
        token = extra[i]
        if not token.startswith("--"):
            raise ValueError(f"unexpected argument: {token}")
        token = token[2:]
        if "=" in token:
            key, value = token.split("=", 1)
            i += 1
        else:
            key = token
            if i + 1 >= len(extra):
                raise ValueError(f"missing value for --{key}")
            value = extra[i + 1]
            i += 2
        overrides[key] = value
    return overrides


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", nargs="*", default=[],
                        help="extra YAML files stacked on top of the defaults")
    parser.add_argument("--run-name", default=None, help="name of the run directory")


def config_from_args(args: argparse.Namespace, extra: Sequence[str]) -> Config:
    cfg = load_config(args.config, parse_overrides(extra))
    if getattr(args, "run_name", None):
        cfg.run.run_name = args.run_name
    return cfg


def case_id(interarrival: float, n_pickers: int, n_robots: int,
            cfg: InstanceCfg | None = None) -> int | None:
    """Map a (1/lambda, K, R) triple to the manuscript case label C1..C27.

    The ordering is the one used in Table 5: arrival rate is the outer loop,
    then the number of pickers, then the number of robots.  It was verified
    against the manuscript, where C18 is stated to be (1/lambda = 40, K = 3,
    R = 6) and reproduces the reported C1/C27 extremes.
    """
    cfg = cfg or InstanceCfg()
    try:
        i_lam = [float(v) for v in cfg.main_interarrivals].index(float(interarrival))
        i_k = [int(v) for v in cfg.main_pickers].index(int(n_pickers))
        i_r = [int(v) for v in cfg.main_robots].index(int(n_robots))
    except ValueError:
        return None
    return 9 * i_lam + 3 * i_k + i_r + 1
