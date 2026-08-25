"""Name -> algorithm mapping used by train.py, eval.py and the collector.

Column names of the manuscript are kept; each maps to the base algorithm of
the cited method, adapted to this problem through the shared observation,
envelope action head and feasibility mask (see the manuscript's Table 6 note).
"""
from __future__ import annotations

from configs.config import Config

ALGORITHMS = ("SAPPO", "AG-DQN", "HSDDQN", "SOA+A2C", "DRLG")

MODEL_FILES = {"SAPPO": "sappo.pt", "AG-DQN": "ag_dqn.pt", "HSDDQN": "hsddqn.pt",
               "SOA+A2C": "soa_a2c.pt", "DRLG": "drlg.pt"}


def build(name: str, cfg: Config, device: str):
    from agent.actor_critic import ActorCriticAgent
    from agent.dqn import DQNAgent
    from agent.sappo import SAPPOAgent

    if name == "SAPPO":
        return SAPPOAgent(cfg.env, cfg.algo, device)
    if name == "AG-DQN":
        return DQNAgent(cfg.env, cfg.algo, cfg.baselines, device, double=False, name=name)
    if name == "HSDDQN":
        return DQNAgent(cfg.env, cfg.algo, cfg.baselines, device, double=True, name=name)
    if name == "SOA+A2C":
        return ActorCriticAgent(cfg.env, cfg.algo, cfg.baselines, device,
                                n_step=cfg.baselines.a2c_n_step, name=name)
    if name == "DRLG":
        return ActorCriticAgent(cfg.env, cfg.algo, cfg.baselines, device,
                                n_step=cfg.baselines.drlg_n_step, name=name)
    raise KeyError(f"unknown algorithm {name!r}; available: {', '.join(ALGORITHMS)}")


def model_path(cfg: Config, name: str, suffix: str = "") -> str:
    import os
    stem, ext = MODEL_FILES[name].rsplit(".", 1)
    filename = f"{stem}{('_' + suffix) if suffix else ''}.{ext}"
    return os.path.join(cfg.run.models_dir, filename)
