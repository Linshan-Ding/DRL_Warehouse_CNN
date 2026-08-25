"""Structured configuration for the SAPPO human-robot collaborative picking project.

Everything tunable lives in the YAML files next to this module; the code only
reads it from here.  YAML files stack (later wins) and any single field can be
overridden with a dotted key, e.g. ``{"algo.gamma": 1.0}``.

The defaults reproduce the manuscript: Table 3 for the system parameters,
Table 4 for the SAPPO hyperparameters, Table 6 for the baseline settings.
"""
from __future__ import annotations

import argparse
import copy
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Sequence, Tuple

import yaml

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CONFIG_DIR)
DEFAULT_CONFIG_FILES = ("env.yaml", "instance.yaml", "train.yaml")


@dataclass
class EnvCfg:
    """Warehouse geometry, resources, service times and the observation."""

    # --- picking-point grid (Eq. 1) ---
    n_aisles: int = 9                 # N_w
    n_positions: int = 20             # N_l
    shelf_length: float = 1.0         # S_l
    shelf_width: float = 1.0          # S_w
    aisle_width: float = 2.0          # S_a
    bottom_aisle_width: float = 2.0   # S_b
    entrance_width: float = 2.0       # S_d
    depot_position: Tuple[float, float] = (18.0, 0.0)
    layout: str = "two_cross_aisles"  # or "three_cross_aisles" (adds a middle cross-aisle)

    # --- mobile resources of THIS scenario ---
    n_robots: int = 6                 # R
    n_pickers: int = 3                # K
    robot_speed: float = 3.0          # v_r
    picker_speed: float = 0.75        # v_k
    robot_capacity: int = 1           # C, orders per service cycle

    # --- resource envelope the networks are sized for ---
    # One global policy serves every (K, R) scenario: the actor head is sized
    # for (k_max, r_max) and resources that do not exist in the current
    # scenario are permanently masked.  This is what makes a single parameter
    # file per algorithm possible.
    k_max: int = 10
    r_max: int = 20
    capacity_max: int = 3             # normaliser for the C context plane
    pick_time_ref: float = 20.0       # normaliser for the tau context plane

    # --- service times ---
    pick_time: float = 10.0           # tau_pick
    pack_time: float = 20.0           # tau_pack

    # --- observation ---
    # "base"       -> 4 spatial channels + 5 configuration planes = 9
    # "plus_agent" -> base + 2 per-robot channels                = 11
    state_channels: str = "base"

    max_steps: int = 20000

    @property
    def n_pick_points(self) -> int:
        return self.n_aisles * self.n_positions

    @property
    def n_state_channels(self) -> int:
        return 9 if self.state_channels == "base" else 11

    @property
    def n_actions(self) -> int:
        """Envelope action space: (k_max + r_max) * points + r_max."""
        return (self.k_max + self.r_max) * self.n_pick_points + self.r_max


@dataclass
class InstanceCfg:
    """The 3 x 3 x 3 fixed evaluation cases and order-stream parameters."""

    n_orders: int = 100
    min_items_per_order: int = 5
    max_items_per_order: int = 5

    case_interarrivals: List[float] = field(default_factory=lambda: [20.0, 40.0, 60.0])
    case_pickers: List[int] = field(default_factory=lambda: [1, 2, 3])
    case_robots: List[int] = field(default_factory=lambda: [2, 4, 6])

    n_val: int = 3
    val_interarrival: float = 40.0

    instances_dir: str = "data/instances"


@dataclass
class TrainCfg:
    """Global-policy training: scenario randomisation and parallel collection."""

    n_episodes: int = 15000
    n_workers: int = 0                # 0 = auto (cpu_count - 2)
    episodes_per_round: int = 0       # 0 = same as n_workers

    # scenario sampling weights (renormalised)
    w_main: float = 0.70              # the 27-case grid
    w_scale: float = 0.15             # larger fleets + extreme ratios
    w_perturb: float = 0.15           # capacity / pick-time / layout perturbations
    scale_configs: List[List[int]] = field(default_factory=lambda: [
        [4, 8], [5, 10], [6, 12], [8, 16], [10, 20], [1, 1], [3, 1], [4, 2]])
    perturb_capacities: List[int] = field(default_factory=lambda: [2, 3])
    perturb_pick_times: List[float] = field(default_factory=lambda: [15.0, 20.0])
    perturb_layout_prob: float = 0.34

    # checkpoint selection and curve logging
    eval_interval: int = 10           # rounds between periodic evaluations
    curve_cases: List[str] = field(default_factory=lambda: ["C06", "C13", "C15", "C24"])


@dataclass
class AlgoCfg:
    """SAPPO (PPO) hyperparameters -- manuscript Table 4."""

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

    cnn_output_dim: int = 256
    policy_hidden: int = 2048
    value_hidden: int = 1024

    device: str = "auto"


@dataclass
class BaselineCfg:
    """Base-algorithm baselines -- common settings follow manuscript Table 6."""

    lr: float = 1e-4
    gamma: float = 0.99
    batch_size: int = 64

    # DQN family (AG-DQN, HSDDQN)
    replay_size: int = 100000         # transitions, stored float16
    target_update: int = 2000         # environment transitions between syncs
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 5000
    dqn_train_freq: int = 4           # one gradient step per this many transitions
    eval_epsilon: float = 0.05        # stochastic-evaluation epsilon

    # actor-critic family (SOA+A2C: n=200; DRLG: n=20)
    a2c_n_step: int = 200
    drlg_n_step: int = 20
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5


@dataclass
class RunCfg:
    run_name: str = "run"
    result_dir: str = "result"
    models_dir: str = "models"
    visdom_enabled: bool = True
    visdom_server: str = "http://localhost"
    visdom_port: int = 8097
    visdom_env: str = "sappo"


@dataclass
class Config:
    env: EnvCfg = field(default_factory=EnvCfg)
    instance: InstanceCfg = field(default_factory=InstanceCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    algo: AlgoCfg = field(default_factory=AlgoCfg)
    baselines: BaselineCfg = field(default_factory=BaselineCfg)
    run: RunCfg = field(default_factory=RunCfg)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def dump(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, allow_unicode=True, sort_keys=False)

    def scenario(self, **overrides) -> EnvCfg:
        """A copy of the environment config with per-episode overrides applied."""
        variant = copy.deepcopy(self.env)
        for key, value in overrides.items():
            if not hasattr(variant, key):
                raise KeyError(f"unknown EnvCfg field {key!r}")
            setattr(variant, key, value)
        return variant

    @property
    def torch_device(self) -> str:
        if self.algo.device != "auto":
            return self.algo.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:  # pragma: no cover
            return "cpu"


# --------------------------------------------------------------------------- #
def _deep_update(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _coerce(value: Any, template: Any) -> Any:
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
    return cls(**{k: _coerce(v, getattr(defaults, k)) for k, v in values.items()})


def config_from_dict(raw: Dict[str, Any]) -> Config:
    raw = copy.deepcopy(raw or {})
    sections = {f.name for f in fields(Config)}
    unknown = set(raw) - sections
    if unknown:
        raise KeyError(f"unknown config section(s): {sorted(unknown)}")
    return Config(**{name: _build(type(getattr(Config(), name)), raw.get(name, {}))
                     for name in sections})


def load_config(paths: Sequence[str] | None = None,
                overrides: Dict[str, Any] | None = None) -> Config:
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
            raise KeyError(f"override must be section.key, got {dotted!r}")
        merged.setdefault(section, {})[key] = value
    return config_from_dict(merged)


def parse_overrides(extra: Sequence[str]) -> Dict[str, Any]:
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
    parser.add_argument("--run-name", default=None)


def config_from_args(args: argparse.Namespace, extra: Sequence[str]) -> Config:
    cfg = load_config(args.config, parse_overrides(extra))
    if getattr(args, "run_name", None):
        cfg.run.run_name = args.run_name
    return cfg


def case_id(interarrival: float, n_pickers: int, n_robots: int,
            cfg: InstanceCfg | None = None) -> int | None:
    """(1/lambda, K, R) -> manuscript case number 1..27 (lambda outer, K, R inner)."""
    cfg = cfg or InstanceCfg()
    try:
        i_lam = [float(v) for v in cfg.case_interarrivals].index(float(interarrival))
        i_k = [int(v) for v in cfg.case_pickers].index(int(n_pickers))
        i_r = [int(v) for v in cfg.case_robots].index(int(n_robots))
    except ValueError:
        return None
    return 9 * i_lam + 3 * i_k + i_r + 1
