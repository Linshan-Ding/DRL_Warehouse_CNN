# SAPPO —— 人机协同订单拣选优化

> English version: [README_EN.md](README_EN.md)

论文 *Spatially-Aware Deep Reinforcement Learning for Human-Robot Collaborative Order
Picking Optimization in Smart Warehouse Systems*（CAIE，第二轮修订）的配套代码。

本文件是**复现手册**，不是项目简介：从上到下照着做，就能跑出论文需要的全部实验数据。
每个补充实验都有两种启动方式——**在 PyCharm 里右键 Run**，或**把命令复制进终端**——
两者跑的是同一套代码、产出同样的文件。

```
configs/       参数文件，代码里没有任何硬编码数值
data/          订单流生成与固定评测算例
environment/   仓库模型、状态、动作掩码、离散事件仿真
agent/         CNN 编码器、策略网络与价值网络、经验缓存、PPO 更新
baselines/     组合排序规则（以及四个 RL 基线的接入位）
experiments/   可直接右键运行的实验脚本
result/        日志、指标、统计检验、绘图，以及各次运行的产出目录
tools/         正确性闸门与原实现参考副本
docs/          本仓库遵循的实验规格契约
```

---

## 1. 问题与建模假设

`N_w` 条巷道、每条 `N_l` 个拣货位的 parts-to-picker 仓库。`R` 台 AMR 承载订单并在拣货位
排队，`K` 名拣货员走到拣货位后**串行服务整个队列**。订单在线到达（泊松过程）。
优化目标是**平均订单流程时间 F**。

- **决策点是事件驱动的。** 仿真推进到"有空闲拣货员且某拣货位有机器人在等"或
  "有载着订单的空闲机器人需要下一个目的地"时才停下来决策，每个决策点只动一个资源。
- **订单分配是系统规则，不是策略动作。** 待分配订单在 depot 处按 FIFO 交给空闲机器人，
  最多交到载运容量 `C`。策略只决定拣货员派往哪里、机器人去哪个拣货位。
- **策略观测的是 `o_t = (s_t, M_t)`**：四通道张量（排队机器人数、拣货员在位、未拣件数、
  未分配订单件数）**加上可行性掩码**。掩码由每台机器人/每个拣货员的个体状态构造，
  是观测的一部分，不是事后过滤。
- **奖励** `r_t = F_{t-1} - F_t`。无折扣时一个 episode 的累计奖励恰等于 `-F_final`，
  `run_00_selfcheck.py` 会断言这一点。
- **载运容量 `C`。** `C = 1` 就是已投稿版本的假设 (A1)；`C > 1` 允许一台 AMR 单次承接多个
  订单并访问其必访点的并集。`C = 1` 与原实现逐事件完全一致。
- **行走距离**沿巷道并经底部或顶部横通道绕行（Eq. 2）。`layout: three_cross_aisles`
  会额外提供一条中部横通道。
- **货架层高未建模**，竖直取件被吸收进 `tau_pick`；实验 E6 通过调 `tau_pick` 作为
  "从更高层取货"的代理。

训练出来的策略**只对一个 `(N_w, N_l, K, R)` 配置有效**：actor 输出层维度是
`|A| = K·N_w·N_l + R·(N_w·N_l + 1)`。每个配置都从零单独训练，checkpoint 存了 `|A|`，
载入到不同配置会直接报错而不是给出错误结果。

---

## 2. 环境配置

```bash
python -V                      # 在 Python 3.11 上测试
pip install -r requirements.txt
```

`visdom` 是可选的：不启动服务只是没有实时曲线，训练照跑。要用就先
`python -m visdom.server`（默认 `http://localhost:8097`）。

### PyCharm 配置

1. **File → Open** 打开本仓库根目录（不要打开上一级）。
2. **Settings → Project → Python Interpreter** 选一个装好
   `requirements.txt` 的解释器。
3. 工作目录**不用管**：`experiments/` 下的脚本第一行会 `import _bootstrap`，
   它负责把仓库根加进 `sys.path` 并切换工作目录，所以从哪儿启动都对。
   （如果你自定义了运行配置，把 Working directory 设成仓库根目录最省事。）

### 参考耗时

在开发容器的 CPU 上实测（约 66 决策/秒，一个 episode 约 1100 个决策点）：

| 阶段 | 耗时 |
|---|---|
| 生成固定算例（§4） | 几秒 |
| 正确性自检（§6） | 约 1 分钟 |
| 一个训练 episode | 约 16 秒（CPU） |
| 一次 2000-episode 训练（§5） | 约 9–12 小时（CPU），GPU 上快很多 |
| 一次评测（3 算例 × 6 方法，§7） | 约 2 分钟 |

正式排 run 矩阵之前，先从 `result/<run>/log.csv` 的 `sps` 和 `wall_clock_s` 两列
读出你自己机器的吞吐。

---

## 3. 两种运行方式

### 方式 A：PyCharm 里右键 Run（推荐）

在项目树里找到 `experiments/` 下对应的脚本 → 右键 → **Run '...'**。
每个脚本都**不需要任何参数**，顶部有一段醒目的配置区，在 IDE 里改完再 Run 即可：

```python
# ==================== 配置区（改完右键 Run） ====================
RUNS     = 3      # 独立重复次数；论文级消融建议 >= 3
EPISODES = None   # None = 用 configs/algo.yaml 的 2000；先填 20 快速验证链路
METHODS  = ["SAPPO", "MQ-ND", "MQ-MinRQ", "MQ-MI", "MI-MinRQ", "MI-MI"]
TIERS    = ["main"]
# ==============================================================
```

脚本运行时会把**等价的终端命令**打印出来，所以在 IDE 里跑一次就知道对应的命令行长什么样。

| 脚本 | 作用 | 回应的意见 |
|---|---|---|
| `run_00_selfcheck.py` | 三道正确性闸门 | — |
| `run_01_prepare_data.py` | 生成各档固定算例 | — |
| `run_smoke.py` | 极小预算跑通全链路 | — |
| `run_e0_baseline.py` | **基线复现 C18（其余实验的门槛）** | — |
| `run_e1_ratio.py` | 人机配比 (1,1)/(3,1)/(4,2) | R1.2 |
| `run_e2_scale.py` | 更大规模 (8,16)/(10,20) | R1.6 |
| `run_e3_training_cost.py` | 训练成本汇总（不训练） | R2.1 |
| `run_e4_gamma.py` | gamma ∈ {0.95, 0.99, 1.0} 消融 | R2.3 |
| `run_e5_state_channels.py` | 状态通道消融 | R2.4 |
| `run_e6_picktime.py` | tau_pick ∈ {15, 20} 敏感性 | R1.4 |
| `run_e7_capacity.py` | 载运容量 C ∈ {2, 3} | R1.5 |
| `run_e8_layout.py` | 中部横通道布局变体 | R1.3 |
| `run_rules_only.py` | 五条规则基线（不训练） | 各表对比列 |
| `run_stats_and_plots.py` | 统计聚合与出图 | — |
| `run_all.py` | 按开关列表批跑 E0→E8 | — |

### 方式 B：PyCharm 终端里复制命令

**View → Tool Windows → Terminal**，把 §4–§8 里的命令复制进去执行。所有命令都以
仓库根目录为工作目录。两种方式产出的文件完全一致。

命令行的通用规则：

- `--config` 可以叠加多个 YAML，后面的覆盖前面的；
- 任何单项都能用点号覆盖，例如 `--algo.gamma 1.0`、`--env.robot_capacity 3`；
- `--run-name` 决定产出目录 `result/<run-name>/`。

---

## 4. 数据准备

- **PyCharm：** 右键运行 `experiments/run_01_prepare_data.py`
- **终端：**

```bash
python -m data.dataset
```

产物：`data/instances/{main,val,large}/*.csv` 与 `data/instances/index.csv`。

已存在的文件**永远不会被覆盖**——这些算例文件（而不是随机种子）才是复现基准，
本项目任何地方都不固定随机种子。其中 `main` 档的三条订单流就是产生已投稿结果的那三条，
随仓库一起提交，所以新克隆下来会直接复用。

---

## 5. E0 基线复现（先做这个）

论文的基准算例 C18：`1/lambda = 40`、`K = 3`、`R = 6`。

- **PyCharm：** 右键运行 `experiments/run_e0_baseline.py`（`RUNS` 默认 3）
- **终端：**

```bash
python train.py --config configs/exp/e0_baseline.yaml --run-name e0_run1
python eval.py  --config configs/exp/e0_baseline.yaml \
                --ckpt result/e0_run1/checkpoint_best.pt \
                --methods SAPPO MQ-ND MQ-MinRQ MQ-MI MI-MinRQ MI-MI \
                --tiers main --run-id 1 --run-name e0_run1
```

产物：`result/e0_run*/{log.csv, checkpoint_best.pt, checkpoint_last.pt, training_cost.csv,
eval_results.csv, config_snapshot.yaml}`。

**通过标准。** SAPPO 在 `lam40` 上报出的 F 应落在已投稿数值的多次运行波动范围内
（论文 Table 5 的 C18：`F = 1379.890`）。跑满 3 次就能看到这个范围。
E0 复现不了的话，先查清原因，别急着跑后面的实验。

> 想先确认链路通不通，把脚本里的 `EPISODES` 改成 20（或终端加
> `--algo.n_episodes 20`），几分钟就能跑完一轮。

---

## 6. 正确性自检

- **PyCharm：** 右键运行 `experiments/run_00_selfcheck.py`
- **终端：**

```bash
python -m tools.selfcheck
```

三道闸门，全部 PASS 才值得信任后面的实验：

1. **奖励恒等式** —— `Σ r_t = -F_final`，说明智能体优化的确实是论文的目标函数。
2. **与原实现逐事件等价** —— 用同一条确定性规则同时驱动本环境和
   `tools/reference/env_I_submitted.py`（产生已投稿结果的原实现，原样保留），
   在每个决策点比对时钟、所选动作、奖励、状态张量和最终 F。
3. **载运容量退化** —— `C = 1` 与原来的"一次一单"模型完全一致，`C = 2` 确实改变调度。

换别的配置检查也可以：

```bash
python -m tools.selfcheck --stream data/instances/main/lam20.csv --rule MI-MI
```

---

## 7. 补充实验

每个实验都是"一个配置叠加层 + 一条命令"。消融实验请跑**至少 3 次独立重复**——
终端方式只需改 `--run-name` 和 `--run-id`；本项目不固定随机种子，重复训练即为独立重复。
**评测时必须传和训练相同的 `--config`**，否则 checkpoint 的动作空间对不上。

### E1 人机配比场景（R1.2）

论文已有的 27 个算例其实已覆盖 1:2、1:4、1:6、1:1（K=2,R=2）、2:4、2:6、3:2、3:4、3:6，
只是没按"配比"组织过。这里补上缺的三个极端档。

- **PyCharm：** 右键运行 `experiments/run_e1_ratio.py`
- **终端**（三个配置各一遍）：

```bash
for cfg in e1_ratio_k1_r1 e1_ratio_k3_r1 e1_ratio_k4_r2; do
    python train.py --config configs/exp/$cfg.yaml --run-name ${cfg}_run1
    python eval.py  --config configs/exp/$cfg.yaml \
                    --ckpt result/${cfg}_run1/checkpoint_best.pt \
                    --methods SAPPO MQ-ND MQ-MinRQ MQ-MI MI-MinRQ MI-MI \
                    --tiers main --run-id 1 --run-name ${cfg}_run1
done
```

### E2 更大规模（R1.6）

- **PyCharm：** 右键运行 `experiments/run_e2_scale.py`
- **终端：**

```bash
for cfg in e2_scale_k8_r16 e2_scale_k10_r20; do
    python train.py --config configs/exp/$cfg.yaml --run-name ${cfg}_run1
    python eval.py  --config configs/exp/$cfg.yaml \
                    --ckpt result/${cfg}_run1/checkpoint_best.pt \
                    --methods SAPPO MQ-ND --tiers main --run-id 1 --run-name ${cfg}_run1
done
```

### E3 训练成本随规模变化（R2.1）

不需要单独训练：每次 `train.py` 都会写一份 `training_cost.csv`，这一步只是汇总。
先跑完 E0/E1/E2 再来。

- **PyCharm：** 右键运行 `experiments/run_e3_training_cost.py`
- **终端：**

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

表里的 `n_actions` 一列同时也是"每个配置都是从零单独训练、没有跨规模权重迁移"的证据。

### E4 折扣因子消融（R2.3）

奖励 `r_t = F_{t-1} - F_t` 只有在 `gamma = 1` 时才严格 telescoping 成 `-F_final`；
`gamma < 1` 时按 Abel 分部求和多出一个 `O(1-gamma)` 的路径项，方向一致但不恒等。
这个实验测的就是这个差别在实践中有多大。

- **PyCharm：** 右键运行 `experiments/run_e4_gamma.py`（`RUNS` 默认 3）
- **终端：**

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

### E5 状态通道消融（R2.4）

`plus_agent` 额外给出两个通道：待路径决策机器人的剩余必访点分布、空闲资源的位置分布，
也就是目前只有掩码看得到、状态张量里没有的那部分信息。和 E0 在同等预算下对比。

- **PyCharm：** 右键运行 `experiments/run_e5_state_channels.py`（`RUNS` 默认 3）
- **终端：**

```bash
for i in 1 2 3; do
    python train.py --config configs/exp/e5_state_plus_agent.yaml --run-name e5_plus_run$i
    python eval.py  --config configs/exp/e5_state_plus_agent.yaml \
                    --ckpt result/e5_plus_run$i/checkpoint_best.pt \
                    --methods SAPPO --tiers main --run-id $i --run-name e5_plus_run$i
done
python -m result.stats --pattern "e[05]_*" --out result/e5_state_summary.csv
```

### E6 拣选时间敏感性（R1.4，多层货架代理）

- **PyCharm：** 右键运行 `experiments/run_e6_picktime.py`
- **终端：**

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

### E7 机器人单次载运能力（R1.5）

- **PyCharm：** 右键运行 `experiments/run_e7_capacity.py`
- **终端：**

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

`C` 变大会拉长单台机器人的路径，若不配套路径优化，F **未必单调改善**——
参考数据：规则 MQ-ND、lam40、K=3/R=6 下 C=1 得 1892.17，C=2 得 1948.98。
结论方向不要预设，跑出来是什么就报什么。

### E8 中部横通道布局变体（R1.3）

- **PyCharm：** 右键运行 `experiments/run_e8_layout.py`
- **终端：**

```bash
python train.py --config configs/exp/e8_layout_mid_aisle.yaml --run-name e8_mid_run1
python eval.py  --config configs/exp/e8_layout_mid_aisle.yaml \
                --ckpt result/e8_mid_run1/checkpoint_best.pt \
                --methods SAPPO MQ-ND MQ-MinRQ MQ-MI MI-MinRQ MI-MI \
                --tiers main --run-id 1 --run-name e8_mid_run1
```

### 只跑规则基线（不需要训练，几分钟）

- **PyCharm：** 右键运行 `experiments/run_rules_only.py`
- **终端：**

```bash
python eval.py --methods MQ-ND MQ-MinRQ MQ-MI MI-MinRQ MI-MI \
               --tiers main --run-id 1 --run-name rules_main
```

### 冒烟测试（几分钟确认环境没装错）

- **PyCharm：** 右键运行 `experiments/run_smoke.py`
- **终端：**

```bash
python train.py --config configs/exp/smoke.yaml --run-name smoke
python eval.py  --ckpt result/smoke/checkpoint_best.pt --methods SAPPO MQ-ND \
                --tiers main --run-name smoke
```

训练轮数太少，学不到任何东西——这一步只证明链路通。

---

## 8. 统计聚合与绘图

- **PyCharm：** 右键运行 `experiments/run_stats_and_plots.py`
- **终端：**

```bash
python -m result.stats                       # -> result/stats_summary.csv
python -m result.plot                        # -> result/figures/*.pdf|.png
python -m result.figs.state_illustration --out result/figures
```

`result.stats` 按方法给出 F 在算例上和在多次独立运行上的均值 ± 标准差，
再把每个方法与 SAPPO 做**配对**检验（配对 t 检验、Wilcoxon 符号秩检验、Cohen's d）——
之所以配对，是因为所有方法解的是同一批固定算例。

### 两个"决策时间"列不要混淆

| 列名 | 含义 | 本仓库实测量级 |
|---|---|---|
| `decision_time_ms` | 每次决策的**计算**墙钟时间 | 规则约 0.03 ms，SAPPO 约 3.4 ms（CPU） |
| `sim_time_per_decision` | 每个决策点之间的**仿真**秒数（makespan / 决策点数） | 约 7 s |

`sim_time_per_decision` 描述的是决策粒度，不是算得快不快，**不能当作计算时间报告**。
两种方法的决策计算都远快于相邻决策点之间约 7 秒的间隔，因此都能实时运行。

---

## 9. 产物对照表

| 文件 | 生成方式 | 服务于 |
|---|---|---|
| `data/instances/**/*.csv`、`index.csv` | §4 | 固定评测基准 |
| `result/e0_run*/log.csv` | §5 | 收敛曲线；E0 门槛 |
| `result/e0_run*/eval_results.csv` | §5 | Table 5 / Table 7 的复现 |
| `result/*/training_cost.csv`、`result/training_cost_summary.csv` | §5、§7 E3 | 训练成本随规模变化（R2.1） |
| `result/e1_*/eval_results.csv` | §7 E1 | 人机配比表（R1.2） |
| `result/e2_*/eval_results.csv` | §7 E2 | 更大规模（R1.6） |
| `result/e4_*/eval_results.csv`、`result/e4_gamma_summary.csv` | §7 E4 | 折扣因子消融（R2.3） |
| `result/e5_*/eval_results.csv`、`result/e5_state_summary.csv` | §7 E5 | 状态充分性消融（R2.4） |
| `result/e6_*/eval_results.csv` | §7 E6 | 拣选时间敏感性（R1.4） |
| `result/e7_*/eval_results.csv` | §7 E7 | 载运容量敏感性（R1.5） |
| `result/e8_*/eval_results.csv` | §7 E8 | 布局变体（R1.3） |
| `result/stats_summary.csv` | §8 | 显著性检验 |
| `result/figures/*.pdf` | §8 | 论文用图草稿 |

运行目录不进 git（见 `.gitignore`），要留档的 CSV 自行复制到论文仓库。

---

## 10. 已知缺口

1. **四个 RL 基线（AG-DQN、HSDDQN、SOA+A2C、DRLG）不在本仓库里。** 论文的
   Data availability 指向这里，重投前必须把它们归档到 `baselines/rl/`。在此之前，
   新表只能有 SAPPO 与五条组合规则两类列。
2. **规则基线的数字与 Table 5 对不上。** 本仿真器下 MQ-ND 在 C18 得 `F = 1892.17`，
   论文报 1922.517（差 −1.6%）。原始的规则脚本不在仓库里，平局打破的细节无从对齐。
   用本实现重跑 Table 5 可以让所有方法跑在同一个仿真器上。
3. **论文的 `D` 列不是计算时间。** 报告值等于 makespan / 决策点数——本仿真器下
   C18/MQ-ND 得 6.808，论文报 6.827（差 −0.3%），而真实计算时间约 0.03 ms。见 §8。

`docs/experiment-spec.md` 是这些实验遵循的契约，改动"测什么、写什么"之前先读它。
