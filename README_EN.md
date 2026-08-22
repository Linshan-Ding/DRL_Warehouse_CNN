# SAPPO — Human-Robot Collaborative Order Picking

> 中文文档见 [README.md](README.md)

Code for *Spatially-Aware Deep Reinforcement Learning for Human-Robot Collaborative
Order Picking Optimization in Smart Warehouse Systems* (CAIE, second revision).

This file is a reproduction manual, not an overview: executed top to bottom it
produces every experiment data file the paper relies on. Each supplementary
experiment can be started in two ways — **right-click Run in PyCharm** or **copy
the command into a terminal** — both run the same code and write the same files.

```
configs/       parameter files — nothing is hard-coded in the code
data/          order-stream generation and the fixed evaluation instances
environment/   warehouse model, state, action mask, discrete-event simulation
agent/         CNN encoder, actor / critic, rollout buffer, PPO update
baselines/     priority dispatching rules (and the slot for the RL baselines)
experiments/   ready-to-run scripts, one per experiment
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

## 2. Environment setup

```bash
python -V                      # tested with Python 3.11
pip install -r requirements.txt
```

`visdom` is optional — without a running server, training just has no live
plots. To use it: `python -m visdom.server` (default `http://localhost:8097`).

### PyCharm

1. **File → Open** the repository root (not its parent).
2. **Settings → Project → Python Interpreter**: pick an interpreter with
   `requirements.txt` installed.
3. The working directory needs no attention: every script under `experiments/`
   starts with `import _bootstrap`, which puts the repository root on `sys.path`
   and switches the working directory, so the scripts run correctly no matter
   where they are launched from. (If you write your own run configuration,
   setting Working directory to the repository root is the simplest choice.)

### Reference timings

Measured on the CPU of the development container (`sps ≈ 66` decisions/s, one
episode ≈ 1100 decisions):

| Stage | Cost |
|---|---|
| Instance generation (§4) | seconds |
| Self-checks (§6) | ≈1 min |
| One training episode | ≈16 s CPU |
| One 2000-episode run (§5) | ≈9–12 h CPU; substantially less on a GPU |
| One evaluation of 3 instances × 6 methods (§7) | ≈2 min |

Read your own throughput from the `sps` and `wall_clock_s` columns of
`result/<run>/log.csv` before planning a full run matrix.

---

## 3. Two ways to run

### A. Right-click Run in PyCharm (recommended)

Find the script under `experiments/` → right-click → **Run '...'**.
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

Each script prints the **equivalent terminal command**, so one run in the IDE
tells you what the command line looks like.

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

### B. Commands in the PyCharm terminal

**View → Tool Windows → Terminal**, then copy the commands from §4–§8. All of
them assume the repository root as working directory. Both routes write exactly
the same files.

Command-line conventions:

* `--config` stacks several YAML files, later ones win;
* any single field can be overridden with a dotted key, e.g. `--algo.gamma 1.0`
  or `--env.robot_capacity 3`;
* `--run-name` decides the output directory `result/<run-name>/`.

---

## 4. Data preparation

* **PyCharm:** run `experiments/run_01_prepare_data.py`
* **Terminal:**

```bash
python -m data.dataset
```

Products: `data/instances/{main,val,large}/*.csv` and `data/instances/index.csv`.

Existing files are **never overwritten** — these instances, not a random seed,
are the reproduction baseline (no seed is fixed anywhere in this project). The
three `main` streams are the ones that produced the submitted results and are
tracked in git, so a fresh clone reuses them.

---

## 5. E0 — baseline reproduction (do this first)

Case C18 of the paper: `1/λ = 40`, `K = 3`, `R = 6`.

* **PyCharm:** run `experiments/run_e0_baseline.py` (`RUNS` defaults to 3)
* **Terminal:**

```bash
python train.py --config configs/exp/e0_baseline.yaml --run-name e0_run1
python eval.py  --config configs/exp/e0_baseline.yaml \
                --ckpt result/e0_run1/checkpoint_best.pt \
                --methods SAPPO MQ-ND MQ-MinRQ MQ-MI MI-MinRQ MI-MI \
                --tiers main --run-id 1 --run-name e0_run1
```

Products: `result/e0_run*/{log.csv,checkpoint_best.pt,checkpoint_last.pt,training_cost.csv,eval_results.csv,config_snapshot.yaml}`.

**Pass criterion.** The `F̄` reported for SAPPO on `lam40` should land in the
run-to-run range of the submitted value for C18 (Table 5: `F̄ = 1379.890`);
three runs show that range. If E0 does not reproduce, treat every supplementary
result below as provisional and find out why first.

> To check the pipeline first, set `EPISODES` to 20 in the script (or add
> `--algo.n_episodes 20` on the command line).

---

## 6. Correctness self-checks

* **PyCharm:** run `experiments/run_00_selfcheck.py`
* **Terminal:**

```bash
python -m tools.selfcheck
```

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

Other settings can be checked too:

```bash
python -m tools.selfcheck --stream data/instances/main/lam20.csv --rule MI-MI
```

---

## 7. Supplementary experiments

Every experiment is a config overlay plus one command. Ablations need **at least
three independent runs** — on the command line vary only `--run-name` and
`--run-id`; repetitions are independent because no seed is fixed. Evaluation
must always be given the *same* `--config` as training, otherwise the checkpoint
will not match the action space.

### E1 — picker : robot ratio scenarios (R1.2)

The 27 published cases already cover the ratios 1:2, 1:4, 1:6, 1:1 (K=2,R=2),
2:4, 2:6, 3:2, 3:4 and 3:6; this adds the missing extremes.

* **PyCharm:** run `experiments/run_e1_ratio.py`
* **Terminal:**

```bash
for cfg in e1_ratio_k1_r1 e1_ratio_k3_r1 e1_ratio_k4_r2; do
    python train.py --config configs/exp/$cfg.yaml --run-name ${cfg}_run1
    python eval.py  --config configs/exp/$cfg.yaml \
                    --ckpt result/${cfg}_run1/checkpoint_best.pt \
                    --methods SAPPO MQ-ND MQ-MinRQ MQ-MI MI-MinRQ MI-MI \
                    --tiers main --run-id 1 --run-name ${cfg}_run1
done
```

### E2 — larger fleets (R1.6)

* **PyCharm:** run `experiments/run_e2_scale.py`
* **Terminal:**

```bash
for cfg in e2_scale_k8_r16 e2_scale_k10_r20; do
    python train.py --config configs/exp/$cfg.yaml --run-name ${cfg}_run1
    python eval.py  --config configs/exp/$cfg.yaml \
                    --ckpt result/${cfg}_run1/checkpoint_best.pt \
                    --methods SAPPO MQ-ND --tiers main --run-id 1 --run-name ${cfg}_run1
done
```

### E3 — training cost vs problem size (R2.1)

No separate run: `training_cost.csv` is written by every `train.py` invocation.
Run E0, E1 and E2 first, then aggregate.

* **PyCharm:** run `experiments/run_e3_training_cost.py`
* **Terminal:**

```bash
python - <<'PY'
import glob, pandas as pd
table = pd.concat([pd.read_csv(p) for p in sorted(glob.glob("result/*/training_cost.csv"))],
                  ignore_index=True).sort_values("n_actions")
table.to_csv("result/training_cost_summary.csv", index=False)
print(table[["run_name","n_pickers","n_robots","n_actions","n_parameters",
             "n_episodes","wall_clock_s","decisions_per_second"]].to_string(index=False))
PY
```

The `n_actions` column is also the evidence that each configuration is trained
from scratch with no cross-scale weight transfer.

### E4 — discount factor (R2.3)

With `r_t = F̄_{t-1} − F̄_t` the undiscounted return telescopes exactly to
`−F̄_final`; under discounting Abel summation leaves an `O(1−γ)` path term, so
the objectives stay aligned without being identical. This measures how much that
term matters.

* **PyCharm:** run `experiments/run_e4_gamma.py` (`RUNS` defaults to 3)
* **Terminal:**

```bash
for g in 0.95 0.99 1.00; do
  for i in 1 2 3; do
    python train.py --config configs/exp/e4_gamma_$g.yaml --run-name e4_gamma${g}_run$i
    python eval.py  --config configs/exp/e4_gamma_$g.yaml \
                    --ckpt result/e4_gamma${g}_run$i/checkpoint_best.pt \
                    --methods SAPPO --tiers main --run-id $i --run-name e4_gamma${g}_run$i
  done
done
python -m result.stats --pattern "e4_*" --out result/e4_gamma_summary.csv
python -m result.plot  --pattern "e4_*" --sensitivity-column gamma
```

### E5 — state sufficiency (R2.4)

`plus_agent` adds two channels carrying exactly the per-resource information the
mask uses. Compare against E0 at the same budget.

* **PyCharm:** run `experiments/run_e5_state_channels.py` (`RUNS` defaults to 3)
* **Terminal:**

```bash
for i in 1 2 3; do
    python train.py --config configs/exp/e5_state_plus_agent.yaml --run-name e5_plus_run$i
    python eval.py  --config configs/exp/e5_state_plus_agent.yaml \
                    --ckpt result/e5_plus_run$i/checkpoint_best.pt \
                    --methods SAPPO --tiers main --run-id $i --run-name e5_plus_run$i
done
python -m result.stats --pattern "e[05]_*" --out result/e5_state_summary.csv
```

### E6 — picking-time sensitivity, proxy for rack levels (R1.4)

* **PyCharm:** run `experiments/run_e6_picktime.py`
* **Terminal:**

```bash
for cfg in e6_picktime_15 e6_picktime_20; do
    python train.py --config configs/exp/$cfg.yaml --run-name ${cfg}_run1
    python eval.py  --config configs/exp/$cfg.yaml \
                    --ckpt result/${cfg}_run1/checkpoint_best.pt \
                    --methods SAPPO MQ-ND MQ-MinRQ MQ-MI MI-MinRQ MI-MI \
                    --tiers main --run-id 1 --run-name ${cfg}_run1
done
python -m result.plot --pattern "e6_*" --sensitivity-column pick_time
```

### E7 — AMR carrying capacity (R1.5)

* **PyCharm:** run `experiments/run_e7_capacity.py`
* **Terminal:**

```bash
for cfg in e7_capacity_2 e7_capacity_3; do
    python train.py --config configs/exp/$cfg.yaml --run-name ${cfg}_run1
    python eval.py  --config configs/exp/$cfg.yaml \
                    --ckpt result/${cfg}_run1/checkpoint_best.pt \
                    --methods SAPPO MQ-ND MQ-MinRQ MQ-MI MI-MinRQ MI-MI \
                    --tiers main --run-id 1 --run-name ${cfg}_run1
done
python -m result.plot --pattern "e[07]_*" --sensitivity-column robot_capacity
```

A larger `C` lengthens each robot tour; without a companion routing improvement
`F̄` need **not** improve monotonically — for reference, under MQ-ND on lam40
with K=3/R=6, `C=1` gives 1892.17 and `C=2` gives 1948.98. Report what the runs
show rather than what is expected.

### E8 — layout with a middle cross-aisle (R1.3)

* **PyCharm:** run `experiments/run_e8_layout.py`
* **Terminal:**

```bash
python train.py --config configs/exp/e8_layout_mid_aisle.yaml --run-name e8_mid_run1
python eval.py  --config configs/exp/e8_layout_mid_aisle.yaml \
                --ckpt result/e8_mid_run1/checkpoint_best.pt \
                --methods SAPPO MQ-ND MQ-MinRQ MQ-MI MI-MinRQ MI-MI \
                --tiers main --run-id 1 --run-name e8_mid_run1
```

### Dispatching rules only (no training required)

* **PyCharm:** run `experiments/run_rules_only.py`
* **Terminal:**

```bash
python eval.py --methods MQ-ND MQ-MinRQ MQ-MI MI-MinRQ MI-MI \
               --tiers main --run-id 1 --run-name rules_main
```

### Smoke test

* **PyCharm:** run `experiments/run_smoke.py`
* **Terminal:**

```bash
python train.py --config configs/exp/smoke.yaml --run-name smoke
python eval.py  --ckpt result/smoke/checkpoint_best.pt --methods SAPPO MQ-ND \
                --tiers main --run-name smoke
```

The episode budget is far too small to learn anything — this only proves the
pipeline runs end to end.

---

## 8. Aggregation and figures

* **PyCharm:** run `experiments/run_stats_and_plots.py`
* **Terminal:**

```bash
python -m result.stats                       # -> result/stats_summary.csv
python -m result.plot                        # -> result/figures/*.pdf|.png
python -m result.figs.state_illustration --out result/figures
```

`result.stats` reports, per method, mean ± standard deviation of `F̄` over the
instances and over the independent runs, then compares each method against
SAPPO with a **paired** t-test, a Wilcoxon signed-rank test and Cohen's d — the
samples are paired because every method solves the same fixed instances.

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

| File | Command | Serves |
|---|---|---|
| `data/instances/**/*.csv`, `index.csv` | §4 | fixed evaluation baseline |
| `result/e0_run*/log.csv` | §5 | convergence curves; E0 gate |
| `result/e0_run*/eval_results.csv` | §5 | reproduction of Table 5 / Table 7 |
| `result/*/training_cost.csv`, `result/training_cost_summary.csv` | §5, §7 E3 | training cost vs problem size (R2.1) |
| `result/e1_*/eval_results.csv` | §7 E1 | picker : robot ratio table (R1.2) |
| `result/e2_*/eval_results.csv` | §7 E2 | larger fleets (R1.6) |
| `result/e4_*/eval_results.csv`, `result/e4_gamma_summary.csv` | §7 E4 | discount-factor ablation (R2.3) |
| `result/e5_*/eval_results.csv`, `result/e5_state_summary.csv` | §7 E5 | state-sufficiency ablation (R2.4) |
| `result/e6_*/eval_results.csv` | §7 E6 | picking-time sensitivity (R1.4) |
| `result/e7_*/eval_results.csv` | §7 E7 | carrying-capacity sensitivity (R1.5) |
| `result/e8_*/eval_results.csv` | §7 E8 | layout variant (R1.3) |
| `result/stats_summary.csv` | §8 | significance tests |
| `result/figures/*.pdf` | §8 | draft figures |

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
