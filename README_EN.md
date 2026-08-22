# SAPPO — Human-Robot Collaborative Order Picking

> 中文文档见 [README.md](README.md)

Code for *Spatially-Aware Deep Reinforcement Learning for Human-Robot Collaborative
Order Picking Optimization in Smart Warehouse Systems* (CAIE, second revision).

This file is a reproduction manual, not an overview: executed top to bottom it
produces every experiment data file the paper relies on. Every experiment is
started by **right-clicking a script in PyCharm and choosing Run** — no
arguments, no command lines to assemble.

```
configs/       parameter files — nothing is hard-coded in the code
data/          order-stream generation and the fixed evaluation instances
environment/   warehouse model, state, action mask, discrete-event simulation
agent/         CNN encoder, actor / critic, rollout buffer, PPO update
baselines/     priority dispatching rules (and the slot for the RL baselines)
experiments/   ready-to-run scripts — the entry point for everything
result/        logging, metrics, statistics, figures — and the run outputs
tools/         correctness gates and the reference implementation
docs/          the experiment specification this repository is contracted to
```

---

## 1. Problem assumptions

A parts-to-picker warehouse of `N_w` aisles with `N_l` picking points each.
`R` AMRs carry orders and queue at picking points; `K` human pickers walk to a
point and serve the whole queue there. Orders arrive online (Poisson).
The objective is the **mean order flow time** `F̄`.

* **Decision epochs are event driven.** The simulator runs until an idle picker
  can be dispatched to a point where robots wait, or an idle robot carrying
  orders needs its next destination. Exactly one resource moves per epoch.
* **Order release is a system rule, not a learned decision.** Waiting orders go
  FIFO to robots standing idle at the depot, up to the carrying capacity `C`.
  Only picker assignment and robot routing are learned.
* **The policy observes `o_t = (s_t, M_t)`**: a four-channel tensor over the
  picking-point grid (robot queues, picker presence, unpicked items,
  unassigned-order items) **and** the feasibility mask. The mask is built from
  per-resource information and is part of the observation, not post-processing.
* **Reward** `r_t = F̄_{t-1} − F̄_t`. Undiscounted, an episode's return equals
  `−F̄_final` exactly; `run_00_selfcheck.py` asserts it.
* **Carrying capacity `C`.** `C = 1` is assumption (A1) of the submitted paper.
  `C > 1` lets an AMR take several orders per cycle and visit the union of their
  picking points. `C = 1` reproduces the submitted model event for event.
* **Travel distance** is rectilinear through the bottom or top cross-aisle
  (Eq. 2). `layout: three_cross_aisles` adds a middle cross-aisle.
* **Rack levels are not modelled.** Vertical retrieval is absorbed into
  `τ_pick`; experiment E6 varies it as a proxy for higher rack levels.

A trained policy is **specific to one `(N_w, N_l, K, R)` configuration**: the
actor head has `|A| = K·N_w·N_l + R·(N_w·N_l + 1)` outputs. Every configuration
is trained from scratch; checkpoints store `|A|` and refuse to load into a
different setting rather than silently producing wrong results.

---

## 2. Setting up in PyCharm

1. **File → Open** the repository root (not its parent).
2. **Settings → Project → Python Interpreter**: select a Python 3.11 interpreter.
3. Open `requirements.txt` from the project tree; PyCharm offers
   **Install requirements** at the top of the editor — one click installs
   everything. (Equivalently, `pip install -r requirements.txt` in the PyCharm
   terminal.)
4. The working directory needs no attention: every script under `experiments/`
   starts with `import _bootstrap`, which puts the repository root on `sys.path`
   and switches the working directory, so the scripts run correctly no matter
   where they are launched from. (If you write your own run configuration,
   setting Working directory to the repository root is the simplest choice.)

**Optional: live training curves.** For loss and flow-time curves in the browser,
start `python -m visdom.server` in the PyCharm terminal (default
`http://localhost:8097`). Training runs fine without it — there are simply no
live plots; the curve data always goes to `result/<run>/log.csv`.

### Reference timings

Measured on the CPU of the development container (`sps ≈ 66` decisions/s, one
episode ≈ 1100 decisions):

| Stage | Cost |
|---|---|
| Instance generation (§4) | seconds |
| Self-checks (§5) | ≈1 min |
| One training episode | ≈16 s CPU |
| One 2000-episode run (§6) | ≈9–12 h CPU; substantially less on a GPU |
| One evaluation of 3 instances × 6 methods | ≈2 min |

Read your own throughput from the `sps` and `wall_clock_s` columns of
`result/<run>/log.csv` before planning a full run matrix.

---

## 3. How to run

Find the script under `experiments/` in the project tree → **right-click → Run
'...'**. That is the whole procedure.

Every script takes **no arguments**; a clearly marked configuration block at the
top is what you edit in the IDE before running:

```python
# ==================== configuration (edit, then Run) ====================
RUNS     = 3      # independent repetitions; use >= 3 for an ablation
EPISODES = None   # None = the 2000 of configs/algo.yaml; try 20 for a quick check
METHODS  = ["SAPPO", "MQ-ND", "MQ-MinRQ", "MQ-MI", "MI-MinRQ", "MI-MI"]
TIERS    = ["main"]
# ========================================================================
```

What the configuration variables mean:

| Variable | Meaning |
|---|---|
| `RUNS` | independent repetitions; no random seed is fixed, so repeated training *is* independent repetition |
| `EPISODES` | training episodes. `None` uses the 2000 of `configs/algo.yaml`; 20 checks the whole pipeline in minutes |
| `METHODS` | methods to evaluate: `"SAPPO"` and the five dispatching rules |
| `TIERS` | instance tiers: `main` (the paper's 27 cases), `val`, `large` |
| `CONFIGS` | for multi-configuration experiments (E1/E2/E4/E6/E7): which configurations to run |
| `PATTERN` | in the statistics script: which runs to aggregate, e.g. `"e4_*"` |

Deeper parameters (warehouse size, resource counts, PPO hyperparameters) live in
the YAML files under `configs/`; per-experiment differences are in
`configs/exp/*.yaml` and are referenced by the scripts, so they rarely need
editing.

### Script index

| Script | Purpose | Comment addressed |
|---|---|---|
| `run_00_selfcheck.py` | the three correctness gates | — |
| `run_01_prepare_data.py` | materialise the fixed instances | — |
| `run_smoke.py` | tiny end-to-end check | — |
| `run_e0_baseline.py` | **baseline reproduction, case C18 (the gate)** | — |
| `run_e1_ratio.py` | picker : robot ratios (1,1)/(3,1)/(4,2) | R1.2 |
| `run_e2_scale.py` | larger fleets (8,16)/(10,20) | R1.6 |
| `run_e3_training_cost.py` | training-cost summary (no training) | R2.1 |
| `run_e4_gamma.py` | γ ∈ {0.95, 0.99, 1.0} ablation | R2.3 |
| `run_e5_state_channels.py` | state-channel ablation | R2.4 |
| `run_e6_picktime.py` | τ_pick ∈ {15, 20} sensitivity | R1.4 |
| `run_e7_capacity.py` | carrying capacity C ∈ {2, 3} | R1.5 |
| `run_e8_layout.py` | middle cross-aisle layout | R1.3 |
| `run_rules_only.py` | the five dispatching rules (no training) | comparison columns |
| `run_stats_and_plots.py` | aggregation and figures | — |
| `run_all.py` | batch E0→E8 from a switch list | — |

**Batch runs:** `run_all.py` opens with a switch list; comment out what you do
not want and Run. Each entry carries its expected cost — with everything enabled
it runs for days.

---

## 4. Data preparation

**Run:** `experiments/run_01_prepare_data.py` (once)

**Products:** `data/instances/{main,val,large}/*.csv` and `data/instances/index.csv`

Existing files are **never overwritten** — these instances, not a random seed,
are the reproduction baseline (no seed is fixed anywhere in this project). The
three `main` streams are the ones that produced the submitted results and are
tracked in git, so a fresh clone reuses them.

---

## 5. Correctness self-checks

**Run:** `experiments/run_00_selfcheck.py`

Three gates, all of which must pass before any experiment is trusted:

1. **Reward identity** — `Σ r_t = −F̄_final`, so the agent optimises the paper's
   objective and not a proxy.
2. **Equivalence with the submitted simulator** — driven by the same
   deterministic dispatching rule, this environment and
   `tools/reference/env_I_submitted.py` (the implementation that produced the
   submitted results, kept verbatim) are compared at every decision epoch:
   clock, chosen action, reward, state tensor and final `F̄`.
3. **Capacity degeneracy** — `C = 1` behaves exactly like the submitted
   one-order-per-cycle model, while `C = 2` really does change the schedule.

To check another instance or rule, edit `STREAM` and `RULE` at the top of the
script (for example `data/instances/main/lam20.csv` with `MI-MI`).

---

## 6. E0 — baseline reproduction (do this first)

Case C18 of the paper: `1/λ = 40`, `K = 3`, `R = 6`.

**Run:** `experiments/run_e0_baseline.py` (`RUNS` defaults to 3)

**Products:** `result/e0_run*/{log.csv, checkpoint_best.pt, checkpoint_last.pt,
training_cost.csv, eval_results.csv, config_snapshot.yaml}`

**Pass criterion.** The `F̄` reported for SAPPO on `lam40` should land in the
run-to-run range of the submitted value for C18 (Table 5: `F̄ = 1379.890`);
three runs show that range. If E0 does not reproduce, treat every supplementary
result below as provisional and find out why first.

> To check the pipeline first, set `EPISODES` to 20 in the script — one round
> then takes minutes.

---

## 7. Supplementary experiments

For ablations set `RUNS` to **at least 3** — no random seed is fixed, so repeated
training is independent repetition. Each script creates its own run directories
named `<name>_run<i>`; nothing has to be named by hand.

### E1 — picker : robot ratio scenarios (R1.2)

The 27 published cases already cover the ratios 1:2, 1:4, 1:6, 1:1 (K=2,R=2),
2:4, 2:6, 3:2, 3:4 and 3:6; this adds the missing extremes (1,1), (3,1), (4,2).

**Run:** `experiments/run_e1_ratio.py`
**Products:** `eval_results.csv` under `result/e1_k1r1_run*/`, `e1_k3r1_run*/`
and `e1_k4r2_run*/`, carrying `n_pickers` and `n_robots` so the results can be
re-tabulated by ratio

### E2 — larger fleets (R1.6)

**Run:** `experiments/run_e2_scale.py`
**Products:** `eval_results.csv` and `training_cost.csv` under
`result/e2_k8r16_run*/` and `e2_k10r20_run*/`

### E3 — training cost vs problem size (R2.1)

No separate training: every training run writes a `training_cost.csv`, and this
step only aggregates them. Run E0, E1 and E2 first.

**Run:** `experiments/run_e3_training_cost.py`
**Products:** `result/training_cost_summary.csv`

The `n_actions` column is also the evidence that each configuration is trained
from scratch with no cross-scale weight transfer.

### E4 — discount factor (R2.3)

With `r_t = F̄_{t-1} − F̄_t` the undiscounted return telescopes exactly to
`−F̄_final`; under discounting Abel summation leaves an `O(1−γ)` path term, so
the objectives stay aligned without being identical. This measures how much that
term matters.

**Run:** `experiments/run_e4_gamma.py` (each γ runs `RUNS` times, default 3)
**Products:** `eval_results.csv` (with a `gamma` column) under
`result/e4_gamma0.95_run*/`, `e4_gamma0.99_run*/`, `e4_gamma1.00_run*/`
**Then:** run `run_stats_and_plots.py` with `PATTERN = "e4_*"` and
`SENSITIVITY_COLUMN = "gamma"`

### E5 — state sufficiency (R2.4)

`plus_agent` adds two channels carrying exactly the per-resource information the
mask uses. The control group is E0, which does not need to be rerun.

**Run:** `experiments/run_e5_state_channels.py` (`RUNS` defaults to 3)
**Products:** `result/e5_plus_run*/eval_results.csv` (with a `state_channels`
column)
**Then:** run `run_stats_and_plots.py` with `PATTERN = "e[05]_*"` to see both
groups together

### E6 — picking-time sensitivity, proxy for rack levels (R1.4)

**Run:** `experiments/run_e6_picktime.py`
**Products:** `eval_results.csv` under `result/e6_tau15_run*/` and
`e6_tau20_run*/`
**Then:** run `run_stats_and_plots.py` with `SENSITIVITY_COLUMN = "pick_time"`

### E7 — AMR carrying capacity (R1.5)

**Run:** `experiments/run_e7_capacity.py`
**Products:** `eval_results.csv` under `result/e7_c2_run*/` and `e7_c3_run*/`
**Then:** run `run_stats_and_plots.py` with
`SENSITIVITY_COLUMN = "robot_capacity"`

A larger `C` lengthens each robot tour; without a companion routing improvement
`F̄` need **not** improve monotonically — for reference, under MQ-ND on lam40
with K=3/R=6, `C=1` gives 1892.17 and `C=2` gives 1948.98. Report what the runs
show rather than what is expected.

### E8 — layout with a middle cross-aisle (R1.3)

**Run:** `experiments/run_e8_layout.py`
**Products:** `result/e8_mid_run*/eval_results.csv` (with a `layout` column)

### Dispatching rules only (no training required)

The rules are deterministic, so one pass is enough; these results are the
comparison columns of every new table.

**Run:** `experiments/run_rules_only.py`
**Products:** `result/rules_main/eval_results.csv`

### Smoke test

**Run:** `experiments/run_smoke.py`
**Products:** the full set of files under `result/smoke_run1/`

The episode budget is far too small to learn anything — this only proves the
pipeline runs end to end.

---

## 8. Aggregation and figures

**Run:** `experiments/run_stats_and_plots.py`
**Products:** `result/stats_summary.csv` and `result/figures/*.pdf|.png`

Three switches at the top of the script:

| Variable | Effect |
|---|---|
| `PATTERN` | which runs to aggregate. `"*"` for everything, `"e4_*"` for the discount-factor ablation |
| `SENSITIVITY_COLUMN` | x-axis of the sensitivity curve, e.g. `"gamma"`, `"robot_capacity"`, `"pick_time"` |
| `DRAW_STATE_FIGURE` | whether to also draw the state-representation sketch of Fig. 5 |

What it reports: per method, mean ± standard deviation of `F̄` over the instances
and over the independent runs, then each method against SAPPO with a **paired**
t-test, a Wilcoxon signed-rank test and Cohen's d — the samples are paired
because every method solves the same fixed instances.

### Two decision-time columns — do not confuse them

| Column | Meaning | Typical value here |
|---|---|---|
| `decision_time_ms` | wall-clock **computation** per decision | rule ≈0.03 ms, SAPPO ≈3.4 ms on CPU |
| `sim_time_per_decision` | **simulated** seconds per decision epoch (makespan / #epochs) | ≈7 s |

`sim_time_per_decision` describes how coarse a method's decisions are, not how
fast it computes, and must never be reported as a computation time. Both methods
decide far faster than the ≈7 s between consecutive epochs, so both are
comfortably real-time.

---

## 9. Output file map

| File | Produced by | Serves |
|---|---|---|
| `data/instances/**/*.csv`, `index.csv` | `run_01_prepare_data.py` | fixed evaluation baseline |
| `result/e0_run*/log.csv` | `run_e0_baseline.py` | convergence curves; E0 gate |
| `result/e0_run*/eval_results.csv` | `run_e0_baseline.py` | reproduction of Table 5 / Table 7 |
| `result/*/training_cost.csv`, `result/training_cost_summary.csv` | training scripts + `run_e3_training_cost.py` | training cost vs problem size (R2.1) |
| `result/e1_*/eval_results.csv` | `run_e1_ratio.py` | picker : robot ratio table (R1.2) |
| `result/e2_*/eval_results.csv` | `run_e2_scale.py` | larger fleets (R1.6) |
| `result/e4_*/eval_results.csv` | `run_e4_gamma.py` | discount-factor ablation (R2.3) |
| `result/e5_*/eval_results.csv` | `run_e5_state_channels.py` | state-sufficiency ablation (R2.4) |
| `result/e6_*/eval_results.csv` | `run_e6_picktime.py` | picking-time sensitivity (R1.4) |
| `result/e7_*/eval_results.csv` | `run_e7_capacity.py` | carrying-capacity sensitivity (R1.5) |
| `result/e8_*/eval_results.csv` | `run_e8_layout.py` | layout variant (R1.3) |
| `result/rules_main/eval_results.csv` | `run_rules_only.py` | the rule comparison columns |
| `result/stats_summary.csv` | `run_stats_and_plots.py` | significance tests |
| `result/figures/*.pdf` | `run_stats_and_plots.py` | draft figures |

Run directories are not tracked in git (see `.gitignore`); copy the CSV files you
want to keep into the paper repository.

---

## 10. Known gaps

1. **The four RL baselines (AG-DQN, HSDDQN, SOA+A2C, DRLG) are not in this
   repository.** The paper's data-availability statement points here, so they
   must be archived under `baselines/rl/` before resubmission. Until then any
   new table can only carry SAPPO and the five dispatching rules.
2. **The dispatching-rule numbers do not match Table 5 exactly.** With this
   simulator MQ-ND on case C18 gives `F̄ = 1892.17` against the reported
   1922.517 (−1.6 %). The original dispatching script is not in the repository,
   so the tie-breaking details differ. Regenerating Table 5 with this
   implementation would put every method on one simulator.
3. **The `D̄` column of the paper is not a computation time.** The reported
   values equal makespan / #decision epochs — for C18/MQ-ND this simulator gives
   6.808 against the reported 6.827 (−0.3 %), while the actual computation time
   is ≈0.03 ms. See §8.

`docs/experiment-spec.md` is the contract these experiments are held to; read it
before changing anything about what gets measured or written.
