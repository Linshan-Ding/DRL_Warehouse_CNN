# Reinforcement-learning baselines

The comparison in Section 5.4 of the manuscript uses four learning-based
baselines: **AG-DQN**, **HSDDQN**, **SOA+A2C** and **DRLG**. Their
implementations are *not* part of this repository yet and must be archived here
before resubmission, because the manuscript's data-availability statement points
readers to this repository.

## How to plug a baseline in

A baseline only has to speak the environment contract, exactly like
`agent/ppo.py` does:

```python
state = env.reset(orders)             # numpy array, shape (C, N_w, N_l)
legal = env.legal_actions()           # list of feasible flat action indices
state, reward, done, info = env.step(action_index)
```

Put one module per method in this directory (`ag_dqn.py`, `hsddqn.py`,
`soa_a2c.py`, `drlg.py`), expose a class with `act(env, state)` /
`act_greedy(env, state)` / `update(...)`, and register it in `eval.py` so that
`--methods` accepts its name. Its hyperparameters belong in
`configs/algo.yaml`-style overlays under `configs/exp/`, never in the code.

Until then `eval.py` only reports SAPPO and the five priority dispatching rules,
and any new table produced by the supplementary experiments will be missing the
four RL columns.
