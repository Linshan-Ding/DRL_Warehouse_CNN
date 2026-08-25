# SAPPO — Human-Robot Collaborative Order Picking (full rebuild)

> 中文文档见 [README.md](README.md)

Code for *Spatially-Aware Deep Reinforcement Learning for Human-Robot Collaborative
Order Picking Optimization in Smart Warehouse Systems* (CAIE, second revision).

This is a reproduction manual: running the scripts under `experiments/` in the
order of Section 3 produces every experimental result of the paper (the data of
Tables 5–12 and Figs 8–14) and every result the reviewer responses rely on.
All scripts start with a **right-click Run in PyCharm** — no arguments; edit the
configuration block at the top when needed.

```
configs/       every parameter (YAML); nothing is hard-coded
data/          the 27 fixed cases (C01–C27) and validation streams
environment/   warehouse model, observation, action mask, discrete-event simulation
agent/         five algorithms: SAPPO + four baselines (shared encoder/head/mask)
parallel/      multi-process episode collector (shared-memory parameter sync)
baselines/     the five combined dispatching rules
experiments/   the ready-to-run scripts — the only entry point
models/        ONE parameter file per algorithm (training output, git-ignored)
result/        logs, evaluation CSVs, statistics, figures
paper_assets/  LaTeX tables generated from the CSVs
tools/         correctness gates + the reference implementation
```

---

## 1. Problem and design highlights

A grid warehouse of `N_w x N_l` picking points; `R` AMRs carry orders and queue,
`K` pickers serve whole queues sequentially; orders arrive Poisson; the
objective is the **mean order flow time F̄**.  Decision epochs are event driven,
one resource moves per epoch; order release at the depot is a FIFO system rule
(up to the carrying capacity `C`); the policy decides picker assignment and
robot routing only.

**One parameter file serves every scenario (the core of this rebuild):**

* the action head is sized for the **resource envelope** (K ≤ 10, R ≤ 20),
  |A| = (10+20)·180+20 = 5420; resources absent from a scenario are permanently
  masked;
* the observation = the paper's 4 spatial channels + 5 **configuration planes**
  (normalised constant planes for K, R, C, τ_pick and the layout flag), which
  is how a single network distinguishes scenarios;
* training samples a fresh scenario **every episode** from the parameter table
  (λ / K / R / capacity / picking time / layout; weights in
  `configs/train.yaml`) with a freshly sampled order stream — after training,
  each algorithm keeps exactly **one** policy file that serves all 27 cases and
  every sensitivity scenario zero-shot;
* the **warehouse grid (N_w, N_l) is the exception**: it changes the input
  geometry, so grids retrain from scratch (`run_32`) — precisely the qualified
  transferability answer the reviewer asked for.

**Two metrics (do not confuse):** `F̄` = mean order flow time; **`D̄` = true
wall-clock milliseconds per decision** (action computation only; evaluation is
single-process on purpose so the timing is clean).  The *simulated* seconds
between decision epochs (makespan / #decisions ≈ 7 s) are logged separately as
`sim_time_per_decision` and never reported as D̄.

**The five algorithms:** column names follow the paper; the four competitors
are the **base algorithms** of the cited methods adapted to this problem (the
original designs are tied to incompatible problem structures; a manuscript
footnote states this):

| Column | Implementation | Key settings (Table 6) |
|---|---|---|
| SAPPO | PPO (the paper's method) | full Table 4, γ = 0.99 |
| AG-DQN | DQN | replay 100k, target 2000, ε 1→0.05 |
| HSDDQN | Double DQN | as above + double-Q target |
| SOA+A2C | A2C | rollout 200 |
| DRLG | multi-worker short-rollout AC | rollout 20 |

**Evaluation protocol:** every RL method evaluates each case with **3
stochastic-policy samples** (policy-gradient methods sample the policy;
value-based methods use ε = 0.05 greedy) → mean ± std of F̄; the deterministic
rules run once.

---

## 2. Setup (PyCharm)

1. **File → Open** the repository root; pick a Python 3.11 interpreter.
2. Open `requirements.txt` and click **Install requirements** (or
   `pip install -r requirements.txt`).  A CUDA build of PyTorch speeds up
   training; evaluation is fine on CPU.
3. Working directory needs no attention — every script starts with
   `import _bootstrap`.
4. **Multiprocessing note:** training spawns `cpu_count − 2` workers (edit
   `WORKERS` in the script's configuration block).  Do not run two trainings
   at once on one machine unless you halve `WORKERS` on both.
5. Optional live curves: `python -m visdom.server` before training.

**Memory:** the DQN replay stores 100k float16 transitions ≈ 1.3 GB; reduce
`replay_size` in `configs/train.yaml` if needed.

---

## 3. Orchestration — what runs in parallel, what must be serial

Dependency DAG (`run_all.py` executes it serially; across several machines the
"independent" groups may run concurrently):

```
run_00 selfcheck ─┐
run_01 instances ─┴─ (serial, first, minutes)
      │
      ▼
run_10 SAPPO ─ run_11 AG-DQN ─ run_12 HSDDQN ─ run_13 SOA+A2C ─ run_14 DRLG
  (the five trainings are mutually independent → parallel across machines;
   on one machine run them serially — each saturates the CPU workers)
      │
      ▼ once the needed models/ files exist
run_20 main ─ run_21 scale ─ run_22 capacity ─ run_23 picktime ─ run_24 layout
  (the five evaluations are mutually independent and may run alongside any
   training — evaluation is single-process and occupies one core)
      │
run_30 gamma ─ run_31 state ─ run_32 grids   (extra trainings: mutually
  independent, but serial with any other training on one machine — CPU contention)
      │
      ▼ after everything
run_40 stats + tables + plots  (serial, last, ~1 minute)
```

**Single-machine recommended order** = the default `STAGES` of `run_all.py`.
**Cost yardstick:** one global training = 15,000 episodes; read your own
throughput from the `sps` column of `result/train_*/log.csv`.  Reference:
8 workers ≈ 1200–1600 decisions/s → 3–6 h per training; the main evaluation
1–2 h; each sensitivity evaluation 20–40 min.

**Rehearse before committing compute:** set `EPISODES = 20` in any training
script (or once in `run_all.py`) to walk the whole chain in minutes, then set
it back to `None`.

---

## 4. Script index and output map

| Script (right-click Run) | Purpose | Products | Serves |
|---|---|---|---|
| `run_00_selfcheck.py` | the three correctness gates | console PASS | credibility |
| `run_01_make_instances.py` | generate C01–C27 + val streams | `data/instances/**` | all evaluations |
| `run_10_train_sappo.py` | SAPPO global policy | `models/sappo.pt`, `result/train_sappo/` | every table/figure |
| `run_11..14_train_*.py` | four baseline global policies | `models/{ag_dqn,hsddqn,soa_a2c,drlg}.pt` | Tables 7, 10 |
| `run_20_eval_main.py` | 27 cases × 10 methods | `result/main/eval_results.csv` | **Tables 5, 7, 8, 9**; Figs 9/10/11 |
| `run_21_eval_scale.py` | zero-shot over 5 fleets | `result/scale/K*_R*/` | **Table 10** (R1.6 + R2.1) |
| `run_22_eval_capacity.py` | zero-shot C ∈ {2, 3} | `result/capacity/c*/` | capacity table (R1.5) |
| `run_23_eval_picktime.py` | zero-shot τ ∈ {15, 20} | `result/picktime/tau*/` | τ table (R1.4) |
| `run_24_eval_layout.py` | zero-shot middle cross-aisle | `result/layout/three/` | layout table (R1.3) |
| `run_30_gamma_ablation.py` | train + eval γ ∈ {0.95, 1.0} | `models/sappo_g*.pt`, `result/gamma/` | γ table (R2.3) |
| `run_31_state_ablation.py` | train + eval + per-robot channels | `models/sappo_plus.pt`, `result/state_plus/` | state table (R2.4) |
| `run_32_warehouse_grids.py` | retrain + eval 12×20 and 9×30 | `models/*_12x20.pt` etc., `result/grid/` | **Tables 11, 12** |
| `run_40_stats_tables_plots.py` | tests + LaTeX tables + draft figures | `result/stats_summary.csv`, `paper_assets/tables/*.tex`, `result/figures/` | Tables 8/9 and every table |
| `run_all.py` | serial batch per its STAGES list | all of the above | — |

Training curves (the Fig. 8 style): the `curve_C06/C13/C15/C24` columns of each
`result/train_*/log.csv` (periodic greedy evaluation on the representative
cases, logged only — checkpoint selection uses the validation streams).

Reviewer-comment → data map: R1.1 text-only; R1.2 → `tab_ratio`; R1.3 →
`tab_layout`; R1.4 → `tab_picktime`; R1.5 → `tab_capacity`; R1.6 + R2.1 →
`tab_scale` + `tab_training_cost` (one policy, zero-shot across fleets — the
direct answer to the transferability question); R2.2 text-only; R2.3 →
`tab_gamma`; R2.4 → `tab_state`.

---

## 5. Correctness gates (`run_00`)

1. **Reward identity** `Σ r_t = −F̄_final` — the agents optimise the paper's
   objective.
2. **Event equivalence with the submitted implementation** — the same
   deterministic rule drives this environment and
   `tools/reference/env_I_submitted.py`; clock, actions, rewards, the **first
   four (spatial) channels** and the final F̄ are compared at every epoch (the
   configuration planes are new and excluded from the physics comparison).
3. **Capacity degeneracy** — `C = 1` matches the submitted one-order-per-cycle
   model exactly; `C = 2` genuinely changes the schedule.

---

## 6. Known notes

1. **New and old numbers are not comparable one-to-one.**  This rebuild retrains
   everything (single global policy, fresh cases, new evaluation protocol, and
   D̄ redefined as true computation time); the paper's tables are replaced
   wholesale.
2. **Rules being faster is expected.**  Measured D̄: rules ≈ 0.02 ms, SAPPO
   ≈ 2–3 ms — both far below the ≈ 7 simulated seconds between consecutive
   decision epochs, so real-time operation is comfortable either way; the
   manuscript's real-time argument is rewritten on this basis.
3. **`models/` is git-ignored** (50–85 MB per file); archive via a release if
   needed.
4. `docs/experiment-spec.md` (v2.0) is the experiment contract; read it before
   changing what gets measured or written.
