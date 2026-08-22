# SAPPO — Human-Robot Collaborative Order Picking

Code for *Spatially-Aware Deep Reinforcement Learning for Human-Robot
Collaborative Order Picking Optimization in Smart Warehouse Systems*
(CAIE, second revision).

This README is a reproduction manual, not an overview: executed top to bottom it
produces every experiment data file the paper relies on. Each supplementary
experiment requested by the reviewers has its own command in Section 6, and
Section 8 maps every output file back to the command that creates it.

```
configs/       parameter files — nothing is hard-coded in the code
data/          order-stream generation and the fixed evaluation instances
environment/   warehouse model, state, action mask, discrete-event simulation
agent/         CNN encoder, actor / critic, rollout buffer, PPO update
baselines/     priority dispatching rules (and the slot for the RL baselines)
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
  `−F̄_final` exactly; `tools/selfcheck.py` asserts it.
* **Carrying capacity `C`.** `C = 1` is assumption (A1) of the submitted paper.
  `C > 1` lets an AMR take several orders per cycle and visit the union of their
  picking points. `C = 1` reproduces the submitted model event for event.
* **Travel distance** is rectilinear through the bottom or top cross-aisle
  (Eq. 2). `layout: three_cross_aisles` adds a middle cross-aisle.
* **Rack levels are not modelled.** Vertical retrieval is absorbed into
  `τ_pick`; experiment E6 varies it as a proxy for higher rack levels.

A trained policy is **specific to one `(N_w, N_l, K, R)` configuration**: the
actor head has `|A| = K·N_w·N_l + R·(N_w·N_l + 1)` outputs. Every configuration
is trained from scratch; checkpoints refuse to load into a different setting.

---

## 2. Environment setup

```bash
python -V                      # tested with Python 3.11
pip install -r requirements.txt
```

`visdom` is optional — without a running server, training just has no live
plots. To use it: `python -m visdom.server` (default `http://localhost:8097`).

Reference timings, measured on the CPU of the development container
(`sps ≈ 66` decisions/s, one episode ≈ 1100 decisions):

| Stage | Cost |
|---|---|
| Instance generation (§3) | seconds |
| Self-checks (§5) | ≈1 min |
| One training episode | ≈16 s CPU |
| One 2000-episode run (§4) | ≈9–12 h CPU; substantially less on a GPU |
| One evaluation of 3 instances × 6 methods (§6) | ≈2 min |

Read your own throughput from the `sps` and `wall_clock_s` columns of
`result/<run>/log.csv` before planning a full run matrix.

---

## 3. Data preparation

```bash
python -m data.dataset
```

Products: `data/instances/{main,val,large}/*.csv` and
`data/instances/index.csv`.

Existing files are **never overwritten** — these instances, not a random seed,
are the reproduction baseline (no seed is fixed anywhere in this project). The
three `main` streams are the ones that produced the submitted results and are
tracked in git, so a fresh clone reuses them rather than generating new ones.

---

## 4. E0 — baseline reproduction (do this first)

Case C18 of the paper: `1/λ = 40`, `K = 3`, `R = 6`.

```bash
python train.py --config configs/exp/e0_baseline.yaml --run-name e0_run1
python eval.py  --config configs/exp/e0_baseline.yaml \
                --ckpt result/e0_run1/checkpoint_best.pt \
                --methods SAPPO MQ-ND MQ-MinRQ MQ-MI MI-MinRQ MI-MI \
                --tiers main --run-id 1 --run-name e0_run1
```

Products: `result/e0_run1/{log.csv,checkpoint_best.pt,checkpoint_last.pt,training_cost.csv,eval_results.csv,config_snapshot.yaml}`.

**Pass criterion.** The `F̄` reported for SAPPO on `lam40` should land in the
run-to-run range of the submitted value for C18 (Table 5: `F̄ = 1379.890`).
Repeat with `--run-name e0_run2`, `e0_run3` to see that range. If E0 does not
reproduce, treat every supplementary result below as provisional and find out
why first.

---

## 5. Correctness self-checks

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

Run them on other settings too, e.g.
`python -m tools.selfcheck --stream data/instances/main/lam20.csv --rule MI-MI`.

---

## 6. Supplementary experiments

Every experiment is a config overlay plus one command. Ablations need **at least
three independent runs** — vary only `--run-name` and `--run-id`; repetitions are
independent because no seed is fixed. Evaluation must always be given the *same*
`--config` as training, otherwise the checkpoint will not match the action space.

### E1 — picker : robot ratio scenarios (Reviewer #1)

```bash
for cfg in e1_ratio_k1_r1 e1_ratio_k3_r1 e1_ratio_k4_r2; do
    python train.py --config configs/exp/$cfg.yaml --run-name ${cfg}_run1
    python eval.py  --config configs/exp/$cfg.yaml \
                    --ckpt result/${cfg}_run1/checkpoint_best.pt \
                    --methods SAPPO MQ-ND MQ-MinRQ MQ-MI MI-MinRQ MI-MI \
                    --tiers main --run-id 1 --run-name ${cfg}_run1
done
```

Products: `result/e1_*_run1/eval_results.csv` with `n_pickers`, `n_robots`.
Together with the 27 published cases these cover 1:1, 1:n and n:1 operation.

### E2 — larger fleets (Reviewer #1)

```bash
for cfg in e2_scale_k8_r16 e2_scale_k10_r20; do
    python train.py --config configs/exp/$cfg.yaml --run-name ${cfg}_run1
    python eval.py  --config configs/exp/$cfg.yaml \
                    --ckpt result/${cfg}_run1/checkpoint_best.pt \
                    --methods SAPPO MQ-ND --tiers main --run-id 1 --run-name ${cfg}_run1
done
```

### E3 — training cost vs problem size (Reviewer #2, comment 1)

No separate run: `training_cost.csv` is written by every `train.py` invocation.
Collect the ones from E0, E1 and E2:

```bash
python - <<'PY'
import glob, pandas as pd
frames = [pd.read_csv(p) for p in sorted(glob.glob("result/*/training_cost.csv"))]
table = pd.concat(frames, ignore_index=True)
table.to_csv("result/training_cost_summary.csv", index=False)
print(table[["run_name","n_pickers","n_robots","n_actions","n_parameters",
             "n_episodes","wall_clock_s","decisions_per_second"]].to_string(index=False))
PY
```

### E4 — discount factor (Reviewer #2, comment 3)

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

### E5 — state sufficiency (Reviewer #2, comment 4)

`plus_agent` adds two channels carrying exactly the per-resource information the
mask uses. Compare against E0 at the same budget.

```bash
for i in 1 2 3; do
    python train.py --config configs/exp/e5_state_plus_agent.yaml --run-name e5_plus_run$i
    python eval.py  --config configs/exp/e5_state_plus_agent.yaml \
                    --ckpt result/e5_plus_run$i/checkpoint_best.pt \
                    --methods SAPPO --tiers main --run-id $i --run-name e5_plus_run$i
done
python -m result.stats --pattern "e[05]_*" --out result/e5_state_summary.csv
```

### E6 — picking-time sensitivity, proxy for rack levels (Reviewer #1)

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

### E7 — AMR carrying capacity (Reviewer #1)

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
`F̄` need not improve monotonically in `C`. Report what the runs show.

### E8 — layout with a middle cross-aisle (Reviewer #1)

```bash
python train.py --config configs/exp/e8_layout_mid_aisle.yaml --run-name e8_midaisle_run1
python eval.py  --config configs/exp/e8_layout_mid_aisle.yaml \
                --ckpt result/e8_midaisle_run1/checkpoint_best.pt \
                --methods SAPPO MQ-ND MQ-MinRQ MQ-MI MI-MinRQ MI-MI \
                --tiers main --run-id 1 --run-name e8_midaisle_run1
```

### Dispatching rules only (no training required)

```bash
python eval.py --methods MQ-ND MQ-MinRQ MQ-MI MI-MinRQ MI-MI \
               --tiers main --run-id 1 --run-name rules_main
```

### Smoke test — check the whole chain in minutes

```bash
python train.py --config configs/exp/smoke.yaml --run-name smoke
python eval.py  --ckpt result/smoke/checkpoint_best.pt --methods SAPPO MQ-ND \
                --tiers main --run-name smoke
python -m result.stats --pattern "smoke" && python -m result.plot --pattern "smoke"
```

The episode budget is far too small to learn anything — this only proves the
pipeline runs end to end.

---

## 7. Aggregation and figures

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

## 8. Output file map

| File | Command | Serves |
|---|---|---|
| `data/instances/**/*.csv`, `index.csv` | §3 | fixed evaluation baseline |
| `result/e0_run*/log.csv` | §4 | convergence curves; E0 gate |
| `result/e0_run*/eval_results.csv` | §4 | reproduction of Table 5 / Table 7 |
| `result/*/training_cost.csv`, `result/training_cost_summary.csv` | §4, §6 E3 | training cost vs problem size (R2.1) |
| `result/e1_*/eval_results.csv` | §6 E1 | picker : robot ratio table (R1.2) |
| `result/e2_*/eval_results.csv` | §6 E2 | larger fleets (R1.6) |
| `result/e4_*/eval_results.csv`, `result/e4_gamma_summary.csv` | §6 E4 | discount-factor ablation (R2.3) |
| `result/e5_*/eval_results.csv`, `result/e5_state_summary.csv` | §6 E5 | state-sufficiency ablation (R2.4) |
| `result/e6_*/eval_results.csv` | §6 E6 | picking-time sensitivity (R1.4) |
| `result/e7_*/eval_results.csv` | §6 E7 | carrying-capacity sensitivity (R1.5) |
| `result/e8_*/eval_results.csv` | §6 E8 | layout variant (R1.3) |
| `result/stats_summary.csv` | §7 | significance tests |
| `result/figures/*.pdf` | §7 | draft figures |

Run directories are not tracked in git (see `.gitignore`); copy the CSV files you
want to keep into the paper repository.

---

## 9. Known gaps

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
   is ≈0.03 ms. See Section 7.

`docs/experiment-spec.md` is the contract these experiments are held to; read it
before changing anything about what gets measured or written.
