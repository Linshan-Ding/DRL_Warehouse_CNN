# SAPPO —— 人机协同订单拣选优化（全量重构版）

> English version: [README_EN.md](README_EN.md)

论文 *Spatially-Aware Deep Reinforcement Learning for Human-Robot Collaborative Order
Picking Optimization in Smart Warehouse Systems*（CAIE，第二轮修订）的配套代码。

本文件是**复现手册**：按第 3 节的顺序把 `experiments/` 下的脚本跑完，即可得到
论文实验部分（Tables 5–12、Figs 8–14 的数据）与全部审稿意见回应所需的所有结果。
所有脚本在 **PyCharm 里右键 Run** 启动，零参数，顶部配置区可改。

```
configs/       全部参数（YAML）；代码里没有硬编码数值
data/          27 个固定算例（C01–C27）与验证流
environment/   仓库模型、观测、动作掩码、离散事件仿真
agent/         五个算法：SAPPO 与四个基线（共享编码器/动作头/掩码）
parallel/      多进程 episode 采集器（共享内存参数同步）
baselines/     五条组合调度规则
experiments/   一键运行脚本 —— 唯一入口
models/        每算法一个策略参数文件（训练产出，不进 git）
result/        日志、评测 CSV、统计、图
paper_assets/  从 CSV 生成论文 LaTeX 表格
tools/         正确性闸门与产生初稿结果的原实现（参考副本）
```

---

## 1. 问题与设计要点

`N_w×N_l` 网格仓库；`R` 台 AMR 承载订单并排队，`K` 名拣货员到点后**串行服务整个
队列**；订单泊松到达；目标是**平均订单流程时间 F̄**。决策点事件驱动，每次只动一个
资源；depot 处订单按 FIFO 交给空闲机器人（系统规则，至多载运容量 `C` 单），策略只
决定拣货员派位与机器人路由。

**一个策略文件通吃所有场景（本次重构的核心）：**

- 动作头按**资源包络**（K≤10、R≤20）定尺寸，|A| = (10+20)·180+20 = 5420；
  场景里不存在的资源被掩码永久屏蔽。
- 观测 = 论文的 4 个空间通道 + 5 个**配置广播平面**（K、R、C、τ_pick、布局的归一化
  常值面），单一网络据此分辨场景。
- 训练时**每个 episode 从参数表随机采样场景**（λ/K/R/容量/拣货时间/布局，权重见
  `configs/train.yaml`）并新采订单流实例化环境 → 训练完每个算法**只保存一个**
  策略文件，零样本服务全部 27 算例与各敏感性场景。
- **仓库网格（N_w,N_l）例外**：改网格就改输入几何，须按网格从零重训
  （`run_32`）——这正是对审稿人"迁移性"质疑的限定性回答。

**两个指标（勿混）：** `F̄` = 平均订单流程时间；**`D̄` = 每次决策的真实墙钟毫秒**
（只计动作计算，评测单进程串行以保证计时干净）。每个决策点之间的**仿真**间隔
（makespan/决策数 ≈ 7 s）另存为 `sim_time_per_decision` 诊断列，不作为 D̄ 报告。

**五个算法：** 列名沿用论文；四个对比方法采用**其文献方法对应的基础算法**适配本
问题（原方法与本问题结构不兼容，正文脚注已说明）：

| 列名 | 实现 | 关键设置（Table 6 对齐） |
|---|---|---|
| SAPPO | PPO（本文方法） | 修订稿 Table 4 全套，γ=0.99 |
| AG-DQN | DQN | replay 100k、target 2000、ε 1→0.05 |
| HSDDQN | Double DQN | 同上 + double-Q |
| SOA+A2C | A2C | rollout 200 |
| DRLG | 多 worker 短 rollout AC | rollout 20 |

**评测协议：** 每个 RL 算法在每个算例上**随机策略采样评估 3 次**（策略梯度类按分布
采样；值函数类 ε=0.05 贪婪）→ F̄ 均值±标准差；调度规则确定性单次。

**SAPPO 全局策略稳定化（R2 修订，随修订稿 Table 4 更新）：** 混采场景的回报量级差
一个数量级，为此 (a) critic 在**归一化回报空间**训练（运行统计随参数广播给 worker）；
(b) 优势**按回合内标准化**（`algo.advantage_norm`，让轻载/重载场景在梯度中等权发声）；
(c) PPO epoch 上限 4，**KL 超过 `kl_target=0.02` 提前停止**；(d) 学习率与熵系数按训练
进度**线性衰减**（→10% / →0.002）；(e) minibatch 256；(f) 策略/价值网络正交初始化
（策略头 gain 0.01，初始近均匀策略）。checkpoint 按 **3 条 λ 验证流 × 3 档车队
(1,2)/(2,4)/(3,6) 共 9 个贪婪回合的均值**选取，覆盖各负载档而非只看中档。

---

## 2. 环境配置（PyCharm）

1. **File → Open** 打开仓库根目录；选 Python 3.11 解释器。
2. 打开 `requirements.txt`，点编辑器顶部的 **Install requirements**
   （或终端 `pip install -r requirements.txt`）。有 NVIDIA GPU 时装 CUDA 版
   PyTorch 训练更快；评测 CPU 即可。
3. 工作目录不用管：脚本第一行 `import _bootstrap` 会修好路径。
4. **多进程注意**：训练脚本会起 `CPU核数−2` 个 worker 进程（配置区 `WORKERS` 可改）。
   Windows/PyCharm 下正常；不要在训练同时跑第二个训练，除非把两边 WORKERS 各减半。
5. **实时训练监控（visdom，默认开启）**：先在终端起 `python -m visdom.server`，
   浏览器开 `http://localhost:8097`，面板下拉选 `sappo_<run_name>`（每次训练一个面板）。
   窗口内容：
   - **Representative cases**：四条固定代表算例 **C06/C13/C15/C24** 的贪婪 F̄ 曲线
     合并在一个多线窗口——这就是"训练效果"的直接监控，也是论文 Fig. 8 的数据来源
     （同步落盘在 `log.csv` 的 `curve_C*` 列）；
   - `eval_flow_mean`：验证小网格（3 条 λ 流 × 3 档车队 = 9 个贪婪回合）的 F̄ 均值
     （checkpoint 按它选，独立于上面四个算例）；
   - `mean_flow_time`：训练 episode（采样策略、随机场景）的 F̄，噪声大属正常；
   - SAPPO 另有 `ret_mean/ret_std`（critic 归一化统计）、`ppo_epochs_used`（KL 早停
     实际用的 epoch 数）、`actor_lr`/`entropy_coef`（衰减进度）；
   - loss/熵/KL/sps/显存等诊断曲线各自成窗。
   评估频率由 `configs/train.yaml` 的 `eval_interval` 控制（默认 10 轮 ≈ 8 worker 下
   每 80 episodes 一个点；每次评估 13 个贪婪回合约 1–2 分钟、期间 worker 空闲，
   调大省时间调小更密）。不起 visdom 服务训练照常，曲线数据始终写入 `log.csv`。

**内存**：DQN 系 replay 默认 10 万条 float16 转移 ≈ 1.3 GB；不足时在
`configs/train.yaml` 调小 `replay_size`。

---

## 3. 运行总控（哪些可并行、哪些必须串行）

依赖 DAG（`run_all.py` 即按此串行；多机可把"可并行"的项分开跑）：

```
run_00 自检 ─┐
run_01 算例 ─┴─(串行,最先,几分钟)
      │
      ▼
run_10 SAPPO ─┬ run_11 AG-DQN ─┬ run_12 HSDDQN ─┬ run_13 SOA+A2C ─┬ run_14 DRLG
  (五个训练互相独立 → 多台机器可并行；单机请串行 —— 每个都会吃满 CPU worker)
      │
      ▼ models/ 五个文件齐后
run_20 主评测 ── run_21 规模 ── run_22 容量 ── run_23 拣选时间 ── run_24 布局
  (五个评测互相独立，可与任何"训练"并行 —— 评测单进程、只占 1 核;
   但 21–24 只需 models 存在，20 跑完与否不影响)
      │
run_30 γ消融 ── run_31 状态消融 ── run_32 网格   (三个都是额外训练:
  互相独立、与 run_10..14 逻辑独立，但单机上与任何训练串行 —— 争 CPU)
      │
      ▼ 全部完成后
run_40 统计+表格+图  (串行,最后,1 分钟)
```

**单机推荐顺序** = `run_all.py` 的默认 STAGES（把不跑的注释掉即可）。
**耗时标尺**：一次全局训练 = 15000 episodes；吞吐看 `result/train_*/log.csv` 的
`sps` 列。参考：8 worker 约 1200–1600 决策/s → 单次训练约 3–6 小时；
主评测约 1–2 小时；每个敏感性评测 20–40 分钟；γ/状态/网格是额外训练，同训练量级。

**先演练再投产**：任何训练脚本把配置区 `EPISODES` 设为 20（或 `run_all.py` 里统一
设），几分钟跑通全链路后再改回 `None` 正式训练。

---

## 4. 脚本清单与产物对照表

| 脚本（右键 Run） | 做什么 | 产出 | 服务于 |
|---|---|---|---|
| `run_00_selfcheck.py` | 三道正确性闸门 | 控制台 PASS | 可信度基线 |
| `run_01_make_instances.py` | 生成 C01–C27 + 验证流 | `data/instances/**` | 全部评测 |
| `run_10_train_sappo.py` | SAPPO 全局策略 | `models/sappo.pt`、`result/train_sappo/` | 全部表图 |
| `run_11..14_train_*.py` | 四个基线全局策略 | `models/{ag_dqn,hsddqn,soa_a2c,drlg}.pt` | Table 7/10 |
| `run_20_eval_main.py` | 27 算例 × 10 方法 | `result/main/eval_results.csv` | **Table 5、7、8、9**；Fig 9/10/11 |
| `run_21_eval_scale.py` | 零样本评 5 档机队 | `result/scale/K*_R*/` | **Table 10**（R1.6+R2.1） |
| `run_22_eval_capacity.py` | 零样本评 C∈{2,3} | `result/capacity/c*/` | 容量表（R1.5） |
| `run_23_eval_picktime.py` | 零样本评 τ∈{15,20} | `result/picktime/tau*/` | τ 表（R1.4） |
| `run_24_eval_layout.py` | 零样本评三横通道 | `result/layout/three/` | 布局表（R1.3） |
| `run_30_gamma_ablation.py` | 训练+评 γ∈{0.95,1.0} | `models/sappo_g*.pt`、`result/gamma/` | γ 表（R2.3） |
| `run_31_state_ablation.py` | 训练+评 +个体通道 | `models/sappo_plus.pt`、`result/state_plus/` | 状态表（R2.4） |
| `run_32_warehouse_grids.py` | 12×20 与 9×30 重训+评 | `models/*_12x20.pt` 等、`result/grid/` | **Tables 11、12** |
| `run_40_stats_tables_plots.py` | 检验+LaTeX 表+草图 | `result/stats_summary.csv`、`paper_assets/tables/*.tex`、`result/figures/` | Table 8/9 与全部表图 |
| `run_all.py` | 按 STAGES 串行批跑 | 上述全部 | — |

训练曲线（Fig. 8 风格）：各 `result/train_*/log.csv` 的 `curve_C06/C13/C15/C24` 列
（训练中周期性在代表算例上贪婪评测，仅作曲线、不参与选点；选点用验证流）。

审稿意见 → 数据的完整映射：R1.1 纯写作；R1.2 由 `tab_ratio`（main+scale 重组）；
R1.3→`tab_layout`；R1.4→`tab_picktime`；R1.5→`tab_capacity`；R1.6+R2.1→`tab_scale`
+`tab_training_cost`（单策略零样本跨规模 = 迁移性质疑的直接回答）；R2.2 纯写作；
R2.3→`tab_gamma`；R2.4→`tab_state`。

---

## 5. 正确性自检（`run_00`）

1. **奖励恒等式** `Σ r_t = −F̄_final`：优化的确实是论文目标。
2. **与原实现逐事件等价**：同一条确定性规则驱动新环境与
   `tools/reference/env_I_submitted.py`（产生初稿结果的原实现，原样保留），逐决策点
   比对时钟/动作/奖励/**前 4 个空间通道**/最终 F̄（配置平面是新增的，不参与物理比对）。
3. **容量退化**：C=1 与初稿"一次一单"模型完全一致；C=2 确实改变调度。

---

## 6. 已知说明

1. **新旧数字不可直接对照。** 本版是全量重训（单一全局策略、新算例、新评测协议、
   D̄ 换为真实计算时间），所有表格整体替换，不与初稿逐数字比较。
2. **规则更快是预期的。** D̄ 实测规则约 0.02 ms、SAPPO 约 2–3 ms；两者都远小于相邻
   决策点之间约 7 s 的仿真间隔，实时性均无压力——论文据此改写实时性论证。
3. **models/ 不进 git**（单文件 50–85 MB）。要归档就发 Release 或网盘，README 注明。
4. `docs/experiment-spec.md` 是实验契约（v2.0），改"测什么/存什么"之前先读它。
