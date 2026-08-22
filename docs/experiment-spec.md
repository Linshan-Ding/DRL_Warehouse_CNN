# Experiment specification

Contract for every experiment in this repository. Code, results and manuscript
all refer back to this file: if an experiment is not described here it should
not be run, and if a number is not produced by one of the files listed in
Section 3 it should not appear in the paper.

| | |
|---|---|
| Version | 1.0 |
| Status | active |
| Paper | *Spatially-Aware Deep Reinforcement Learning for Human-Robot Collaborative Order Picking Optimization in Smart Warehouse Systems* (CAIE, second revision) |

### Change log

| Version | Change | Reason | Affects |
|---|---|---|---|
| 1.0 | Initial specification, reverse-engineered from the submitted `manuscript.tex` and the second-round reviewer comments. | The repository had no specification; the supplementary experiments requested by the reviewers need one. | all downstream stages |

---

## 1. Problem and MDP

A parts-to-picker warehouse: `R` AMRs transport orders, `K` human pickers
retrieve items at picking points. Orders arrive online as a Poisson process.
The objective is the mean order flow time `F̄` (Eq. 13).

| Element | Definition | Code |
|---|---|---|
| Decision epoch | an idle picker can be dispatched to a point where robots wait, **or** an idle robot carrying orders needs a routing decision | `environment/env.py::_advance_to_decision_epoch` |
| State `s_t` | 4 channels over the `N_w x N_l` picking-point grid: robot queue, picker presence, unpicked items, unassigned-order items | `environment/state.py::build_state` |
| Mask `M_t` | feasible actions, derived from per-robot residual demand and per-picker idleness | `environment/state.py::legal_action_indices` |
| Observation | `o_t = (s_t, M_t)` — the mask is part of the observation, not post-processing | `agent/ppo.py` |
| Action | one flat index over `K·N_w·N_l` picker assignments, `R·N_w·N_l` robot routings and `R` return-to-depot moves | `environment/state.py` |
| Reward | `r_t = F̄_{t-1} − F̄_t` (Eq. 14) | `environment/env.py::compute_reward` |
| Termination | every order has arrived, been served and returned to the depot | `environment/env.py::_next_event_time` |

**Reward identity.** Because `F̄_0 = 0`, the *undiscounted* return of an episode
equals `−F̄_final` exactly. `tools/selfcheck.py` asserts this. Under discounting
(`γ = 0.99`) the return instead equals
`F̄_0 − γ^{T−1} F̄_T − (1−γ) Σ_t γ^{t−1} F̄_t`, so the learning objective stays
aligned with the operational one but is not identical to it. Experiment E4
measures how much this matters.

**Order release is a system rule, not a policy decision.** Waiting orders are
handed FIFO to robots standing idle at the depot, up to the carrying capacity
`C`. Only picker assignment and robot routing are learned.

**Carrying capacity.** `C = 1` is assumption (A1) of the submitted manuscript.
`C > 1` lets an AMR take several orders per cycle, visit the union of their
required picking points and be packed once per order on return. `C = 1`
reproduces the submitted model event for event.

---

## 2. Evaluation setup

### 2.1 Instance parameter table

| Parameter | Symbol | Value |
|---|---|---|
| Aisles | `N_w` | 9 |
| Picking points per aisle | `N_l` | 20 |
| Rack width / point spacing / aisle widths / entrance | `S_w, S_l, S_a, S_b, S_d` | 1.0, 1.0, 2.0, 2.0, 2.0 m |
| Robots | `R` | {2, 4, 6} (extended by E1/E2) |
| Pickers | `K` | {1, 2, 3} (extended by E1/E2) |
| Robot / picker speed | `v_r, v_k` | 3.0 / 0.75 m/s |
| Mean inter-arrival time | `1/λ` | {20, 40, 60} s |
| Picking / packing time | `τ_pick, τ_pack` | 10.0 / 20.0 s |
| Orders per instance | | 100, 5 items each |

### 2.2 Instance tiers

Materialised once by `python -m data.dataset`, one CSV per stream, never
regenerated. **These files, not a random seed, are the reproduction baseline** —
no seed is fixed anywhere in the project, and independent repetitions come from
repeated training runs.

| Tier | Streams | Purpose |
|---|---|---|
| `main` | 3 (`1/λ` ∈ {20, 40, 60}) | the 27 published cases C1–C27, crossed with the resource settings from `configs/env.yaml` |
| `val` | 3 (`1/λ` = 40) | checkpoint selection during training; never reported |
| `large` | 2 (`1/λ` ∈ {10, 100}, 200 orders) | arrival rates outside the parameter table |

The three `main` streams are the ones that produced the submitted results and
are carried over verbatim from the original repository layout.

**Case numbering.** `case_id = 9·i_λ + 3·i_K + i_R + 1` with `λ` the outer loop,
then pickers, then robots. Verified against the manuscript: C18 = (40, 3, 6) as
stated in Section 5.6, and the C1/C27 extremes match Table 5.

### 2.3 Methods compared

| Method | Status |
|---|---|
| SAPPO | `agent/ppo.py` |
| MQ-ND, MQ-MinRQ, MQ-MI, MI-MinRQ, MI-MI | `baselines/rules.py` |
| AG-DQN, HSDDQN, SOA+A2C, DRLG | **not in this repository yet** — see `baselines/rl/README.md` |

Until the four RL baselines are archived, any new table can only carry SAPPO and
the five dispatching rules.

### 2.4 Metrics

| Metric | Meaning |
|---|---|
| `mean_flow_time` (`F̄`) | mean of (completion − arrival) over completed orders — the objective |
| `decision_time_ms` | **wall-clock milliseconds per decision**, i.e. computational effort |
| `sim_time_per_decision` | simulated seconds per decision epoch = makespan / #epochs |

`decision_time_ms` and `sim_time_per_decision` measure entirely different
things and are logged as separate columns on purpose. Measured on the reference
machine: a dispatching rule needs ≈0.03 ms per decision, SAPPO ≈3.4 ms on CPU,
while consecutive decision epochs are ≈7 simulated seconds apart.

### 2.5 Repetitions and statistics

Independent runs come from repeated training (no seed is fixed). Report at least
3 runs per ablation configuration and give mean ± standard deviation with the
run count stated. Method comparisons are **paired over the shared fixed
instances**: paired t-test, Wilcoxon signed-rank test and Cohen's d, as in
Section 5.5 of the manuscript (`result/stats.py`).

---

## 3. Data landing list

| File | Produced by | Key columns |
|---|---|---|
| `data/instances/<tier>/*.csv` | `data.dataset` | `order_id, arrival_time, item_id, pick_point_id` |
| `data/instances/index.csv` | `data.dataset` | `instance_id, tier, mean_interarrival, n_orders, n_rows, path` |
| `result/<run>/log.csv` | `train.py` | `step, episode, mean_flow_time, reward_sum, n_decisions, policy_loss, value_loss, entropy, approx_kl, clip_fraction, sps, wall_clock_s, gpu_mem_gb, eval_flow_mean, eval_flow_std` |
| `result/<run>/training_cost.csv` | `train.py` | `n_actions, n_parameters, n_episodes, total_decisions, wall_clock_s, decisions_per_second, device` + configuration |
| `result/<run>/eval_results.csv` | `eval.py` | `instance_id, tier, mean_interarrival, case_id, method, run_id, n_pickers, n_robots, robot_capacity, state_channels, layout, pick_time, gamma, mean_flow_time, makespan, n_decisions, decision_time_ms, sim_time_per_decision, solve_wall_clock_s` |
| `result/<run>/config_snapshot.yaml`, `run_info.json` | `train.py` | effective configuration, git commit, device |
| `result/stats_summary.csv` | `result.stats` | per-method summary and paired tests |
| `result/figures/*.pdf/.png` | `result.plot` | draft figures |

---

## 4. Reviewer comment → experiment → data mapping

Every second-round comment either maps to an experiment producing specific
columns, or is explicitly marked as a text-only revision.

| Comment | Requirement | Experiment | Data | Serves |
|---|---|---|---|---|
| R1.1 | roles of humans and robots | — (text) | — | new roles table in Section 3.1 |
| R1.2 | 1:1, 1:n, n:1 configurations | **E1** `(K,R)` ∈ {(1,1), (3,1), (4,2)} | `eval_results.csv` → `n_pickers, n_robots, mean_flow_time` | new ratio table |
| R1.3 | movement between racks | **E8** middle cross-aisle | `eval_results.csv` → `layout` | Eq. (2) discussion, Fig. 2 |
| R1.4 | rack levels above the first | **E6** `τ_pick` ∈ {10, 15, 20} | `eval_results.csv` → `pick_time` | new assumption (A8) + sensitivity |
| R1.5 | robot carrying capacity | **E7** `C` ∈ {1, 2, 3} | `eval_results.csv` → `robot_capacity` | generalised (A1), Eq. (5) capacity constraint |
| R1.6 | larger fleets | **E2** `(K,R)` ∈ {(8,16), (10,20)} | `eval_results.csv`, `training_cost.csv` | extension of Table 10 |
| R2.1 | retraining vs transfer, training cost | **E3** (instrumentation, rides along with E0–E2) | `training_cost.csv` → `n_actions, n_parameters, wall_clock_s` | explicit "trained from scratch" statement + cost table |
| R2.2 | hyperparameter selection | — (text) | `configs/algo.yaml` provenance | moderated claim in Section 5.4 |
| R2.3 | discount factor vs telescoping | **E4** `γ` ∈ {0.95, 0.99, 1.0}, ≥3 runs each | `eval_results.csv` → `gamma` | corrected Eq. (14) discussion |
| R2.4 | state sufficiency | **E5** `base` vs `plus_agent` channels, ≥3 runs each | `eval_results.csv` → `state_channels` | `o_t = (s_t, M_t)` formalisation + ablation |
| — | reproduction gate | **E0** case C18 | `eval_results.csv` | credibility of everything above |

---

## 5. Registered deviations

Deviations from the code-generation guideline, each with its reason.

| Deviation | Guideline default | Reason |
|---|---|---|
| `algo.gamma = 0.99` | `1.0` for combinatorial problems | Table 4 of the submitted manuscript reports 0.99; reproducing the submitted results takes precedence. `γ = 1.0` is one arm of E4. |
| `instance.train_mode = fixed` | sample a fresh instance every episode | The submitted results were produced by training on the fixed `main` stream matching the case's arrival rate. E0 could not reproduce them otherwise. `sampled` is available and is the better setting for future work. |
| No multiprocess environment workers | introduce them when the environment is hard to tensorise | This *is* such an environment, but the guideline requires a measured justification (profiler showing environment stepping dominates), and no training run has been executed yet. The rollout collector is kept replaceable; revisit once `sps` has been measured. |
| PPO update evaluates minibatches in one forward pass | — | The submitted implementation looped over transitions individually. The objective is unchanged; only the execution differs. |
| Four RL baselines absent | baselines run at equal budget | Their implementations are not in this repository. They must be archived under `baselines/rl/` before resubmission. |

---

## 6. Open items before resubmission

1. **Archive the four RL baselines.** The manuscript's data-availability
   statement points at this repository; the comparison in Tables 5–12 cannot be
   reproduced from it today.
2. **Reconcile the dispatching-rule numbers.** With this simulator MQ-ND on case
   C18 gives `F̄ = 1892.17` against the 1922.517 of Table 5 (−1.6 %), and
   `sim_time_per_decision = 6.808` against the reported `D̄ = 6.827` (−0.3 %).
   The residual difference comes from tie-breaking details of the original
   dispatching script, which is not in the repository. Regenerating Table 5 with
   this implementation would put every method on one simulator — which is
   exactly what the fairness claim in Section 5.4 needs.
3. **Fix the decision-time column.** The values reported as `D̄` equal
   makespan / #decision epochs, not computation time; see Section 2.4.
