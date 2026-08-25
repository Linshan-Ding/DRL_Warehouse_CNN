# Experiment specification

Contract for every experiment in this repository. If an experiment is not here
it should not run; if a number is not produced by a file in Section 4 it should
not appear in the paper.

| | |
|---|---|
| Version | 2.0 |
| Status | active (full rebuild — replaces every result of the submitted rounds) |
| Paper | *Spatially-Aware DRL for Human-Robot Collaborative Order Picking* (CAIE, R2) |
| How to run | right-click the scripts under `experiments/`; see `README.md` / `README_EN.md` |

### Change log

| Version | Change | Reason |
|---|---|---|
| 2.0 | Full rebuild: single global policy per algorithm (envelope head K≤10/R≤20 + configuration planes), per-episode scenario randomisation, four RL baselines implemented as the cited methods' base algorithms, D̄ redefined as true wall-clock decision time, 27 independent case streams, 3-stochastic-sample evaluation, multi-process collection. All previous results are superseded. | The submitted D̄ column mixed simulated-time quantities; baselines were absent from the repository; per-configuration retraining was the reviewers' central objection (R2.1). |
| 1.x | (history) per-configuration training, E0–E9 supplementary experiments. | superseded |

---

## 1. Problem and MDP (unchanged physics)

Discrete-event simulation verified event-equivalent to the implementation that
produced the submitted results (`tools/selfcheck.py`, gate 2). Reward
`r_t = F̄_{t-1} − F̄_t`; undiscounted return ≡ −F̄_final (gate 1). Carrying
capacity C generalises assumption (A1); C = 1 degenerates exactly (gate 3).

**Observation** `o_t = (s_t, M_t)`: 4 spatial channels + 5 configuration planes
(K/k_max, R/r_max, C/c_max, τ/τ_ref, layout); optional +2 per-robot channels
(ablation). **Action space**: envelope (k_max=10, r_max=20) ⇒ |A| = 5420 at the
9×20 grid; infeasible-by-scenario resources are permanently masked.

## 2. Methods

| Column | Base algorithm | Notes |
|---|---|---|
| SAPPO | PPO | Table 4 hyperparameters |
| AG-DQN / HSDDQN | DQN / Double DQN | replay 100k (float16), target 2000, ε 1→0.05 |
| SOA+A2C / DRLG | A2C (n=200) / short-rollout AC (n=20) | Table 6 common settings |
| MQ-ND, MQ-MinRQ, MQ-MI, MI-MinRQ, MI-MI | deterministic dispatching rules | Section 5.3 |

One parameter file per RL algorithm (`models/<algo>.pt`), selected by greedy
validation on the fixed val streams; representative-case curves (C06/C13/C15/
C24) are logged for Fig. 8 only.

## 3. Training and evaluation protocol

* **Training**: every episode samples a scenario from the parameter table —
  70 % the 27-case grid, 15 % scale/extreme fleets {(4,8),(5,10),(6,12),(8,16),
  (10,20),(1,1),(3,1),(4,2)}, 15 % perturbations (C∈{2,3}, τ∈{15,20}, middle
  cross-aisle) — plus a fresh order stream. Parallel episode collection
  (spawn workers, shared-memory parameter sync), GPU minibatch updates between
  rounds. No random seeds anywhere; repetition = independent stochastic runs.
* **Evaluation**: 27 fixed case streams C01–C27 (3×3×3 grid, λ outer / K / R
  inner). RL: 3 stochastic samples per case (PG: policy sampling; value-based:
  ε=0.05 greedy) → mean ± std. Rules: deterministic, once. Sensitivity studies
  evaluate the SAME policy files zero-shot on the nine λ=40 streams with the
  scenario overridden; only the warehouse grid retrains (12×20, 9×30, own
  streams). Evaluation is single-process so that D̄ is clean.
* **Metrics**: F̄ (mean order flow time); D̄ = wall-clock ms per decision
  (action computation only); `sim_time_per_decision` = makespan/#decisions is a
  diagnostic and never reported as D̄.
* **Statistics**: paired t-test, Wilcoxon signed-rank, Cohen's d over the 27
  per-case means, each method vs SAPPO (Tables 8–9).

## 4. Data landing list

| File | Produced by | Serves |
|---|---|---|
| `data/instances/cases/C01..C27.csv`, `val/*.csv`, `index.csv` | `run_01` | all evaluations |
| `models/<algo>.pt` (+ `_g*`, `_plus`, `_<grid>` variants) | `run_10..14`, `run_30..32` | all evaluations |
| `result/train_*/log.csv` (`curve_C*` columns), `training_cost.csv` | trainings | Fig. 8, R2.1 cost table |
| `result/main/eval_results.csv` | `run_20` | Tables 5, 7, 8, 9; Figs 9–11 |
| `result/scale/K*_R*/eval_results.csv` | `run_21` | Table 10 |
| `result/capacity/c*/`, `picktime/tau*/`, `layout/three/` | `run_22..24` | R1.5 / R1.4 / R1.3 tables |
| `result/gamma/g*/`, `result/state_plus/` | `run_30..31` | R2.3 / R2.4 tables |
| `result/grid/<AxB>/eval_results.csv` | `run_32` | Tables 11, 12 |
| `result/stats_summary.csv`, `paper_assets/tables/*.tex`, `result/figures/` | `run_40` | manuscript + response letter |

## 5. Reviewer-comment map

| Comment | Answered by |
|---|---|
| R1.1 roles | text + Fig. 2 redraw (no data) |
| R1.2 ratios | `tab_ratio` (main + scale reorganised by K:R) |
| R1.3 racks/layout | Eq. (2) text + `tab_layout` |
| R1.4 rack levels | assumption (A8) + `tab_picktime` |
| R1.5 capacity | generalised (A1)/Eq. (5) + `tab_capacity` |
| R1.6 larger fleets | `tab_scale` rows (8,16), (10,20) + `tab_training_cost` |
| R2.1 transfer/retraining | ONE policy zero-shot across fleets (`tab_scale`); grids retrain (`tab_grid`); cost table |
| R2.2 tuning | text (matched common settings; no exhaustive search, incl. SAPPO) |
| R2.3 discounting | Abel identity in text + `tab_gamma` |
| R2.4 state sufficiency | o_t=(s_t,M_t) formalisation + `tab_state` |

## 6. Registered deviations

| Deviation | Reason |
|---|---|
| Baselines are base algorithms under the cited names | the cited designs are structurally tied to other problems; footnote in the manuscript |
| γ = 0.99 default (not 1.0) | Table 4 of the paper; γ = 1.0 is an ablation arm |
| No random seeds | fixed case CSVs + independent stochastic runs are the reproduction baseline |
| Evaluation single-process | keeps the D̄ timing clean |
