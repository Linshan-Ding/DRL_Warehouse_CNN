# 人机协同订单拣选优化：SAPPO-I 训练说明

本项目用于在人机协同仓储拣选场景中，基于 PPO（Proximal Policy Optimization）训练智能体，使机器人与拣货员在订单到达、任务分配、拣选点选择和返回 Depot 等过程中形成更优的协同决策策略。

本 README 只围绕 `agent/SAPPO_I.py` 及其直接依赖文件展开说明，不介绍项目中未使用的 `DQN.py`、`A3C.py`、`MAPPO.py` 等其他算法代码。

---

## 1. 核心功能

`SAPPO_I.py` 实现的是一个基于 CNN 状态编码和 PPO 策略更新的单智能体训练流程，主要包括：

1. 读取订单数据 `.pkl`；
2. 初始化仓储环境 `WarehouseEnv`；
3. 使用 CNN 提取仓库状态特征；
4. 根据当前环境状态构造合法动作集合；
5. 对非法动作进行 mask，只在合法动作中采样；
6. 通过 PPO clipped objective 更新策略网络和值函数网络；
7. 记录训练奖励、损失、完工时间、订单流经时间等指标；
8. 自动保存 `.npz` 训练数据和论文绘图用 `.png` 曲线。

---

## 2. 项目结构说明

与 `SAPPO_I.py` 直接相关的文件和目录如下：

```text
github_code/
├── agent/
│   ├── SAPPO_I.py          # 核心训练入口：PPO训练、评估、绘图保存
│   ├── CNN.py              # CNN状态特征提取器
│   └── conj.py             # 训练超参数配置，例如学习率、gamma、device等
│
├── env/ 或 environment/
│   ├── env_I.py            # 仓储环境主体，提供 reset() 和 step()
│   ├── class_object.py     # 订单、商品、机器人、拣货员等对象定义
│   └── class_config.py     # 环境参数配置
│
├── data/
│   ├── data/instances/
│   │   ├── orders_20.pkl   # 训练订单数据
│   │   ├── orders_40.pkl   # 订单数据示例
│   │   ├── orders_60.pkl   # 订单数据示例
│   │   └── orders_100.pkl  # 订单数据示例
│   ├── generat_order_data.py      # 推荐使用的订单生成脚本
│   └── generat_order_csv_pkl.py   # 早期订单生成脚本
│
└── results/
    └── SAPPO/              # 训练数据和图像输出目录
```

> 注意：项目中可能同时存在 `env` 和 `environment` 两套包名。`SAPPO_I.py`、订单生成脚本和 `.pkl` 订单文件必须使用同一套包名，否则会出现反序列化错误。

---

## 3. 运行环境

建议使用 Python 3.8 及以上版本。

主要依赖：

```bash
pip install torch numpy matplotlib visdom
```

其中：

- `torch`：神经网络训练；
- `numpy`：数值计算与训练数据保存；
- `matplotlib`：训练曲线绘制；
- `visdom`：训练过程可视化，可选安装；
- `pickle`、`os`、`math`：Python 标准库，无需额外安装。

如果不需要 Visdom，可不启动 Visdom 服务。代码会自动检测是否安装 `visdom`，未安装时会继续训练，只是不显示实时可视化窗口。

如需使用 Visdom，可在单独终端启动：

```bash
python -m visdom.server
```

然后在浏览器打开：

```text
http://localhost:8097
```

---

## 4. SAPPO-I 方法流程

### 4.1 状态输入

环境每一步返回的状态会被转换为 PyTorch tensor，并送入 CNN 特征提取器：

```python
CNNFeatureExtractor(4, cfg.cnn_output_dim)
```

这里的 `4` 表示输入状态通道数，`cfg.cnn_output_dim` 来自 `agent/conj.py` 中的配置。

### 4.2 策略网络

`PolicyNetwork` 由 CNN 和 MLP 两部分组成：

```text
state -> CNNFeatureExtractor -> MLP -> action scores
```

输出维度为完整动作空间大小：

```python
action_num = (env.N_robots + env.N_pickers) * env.N_l * env.N_w + env.N_robots
```

动作空间包括三类动作：

1. 拣货员选择拣选点；
2. 机器人选择拣选点；
3. 机器人返回 Depot。

### 4.3 值函数网络

`ValueNetwork` 同样使用 CNN 提取状态特征，然后通过 value head 输出当前状态价值：

```text
state -> CNNFeatureExtractor -> value head -> state value
```

该值函数用于计算 GAE advantage 和 PPO value loss。

### 4.4 合法动作 mask

在每个决策步，代码会根据机器人、拣货员、拣选点和订单状态动态生成合法动作集合。

非法动作会被赋值为极小值：

```python
masked_logits = logits.masked_fill(invalid_mask, -1e9)
```

然后只在合法动作中进行采样：

```python
probs = torch.softmax(masked_logits, dim=-1)
dist = torch.distributions.Categorical(probs=probs)
action_idx = dist.sample()
```

这样可以避免智能体选择当前环境中不可执行的动作。

### 4.5 PPO 更新

每个 episode 结束后，代码默认执行一次 PPO 更新：

```python
UPDATE_EVERY_EPISODES = 1
```

更新过程包括：

1. 根据 reward、value、done 计算 GAE；
2. 标准化 advantage；
3. 重新计算当前策略下的 log probability；
4. 使用 PPO clipped objective 计算 policy loss；
5. 使用 clipped value loss 计算 value loss；
6. 加入 entropy 项保持探索；
7. 使用梯度裁剪提高训练稳定性。

---

## 5. 数据准备

`SAPPO_I.py` 训练时默认读取：

```python
../data/data/instances/orders_20.pkl
```

也就是说，如果你在 `agent/` 目录下运行：

```bash
python SAPPO_I.py
```

则实际读取路径为：

```text
github_code/data/data/instances/orders_20.pkl
```

如果订单文件不在该位置，需要二选一：

### 方式一：把订单文件放到代码默认路径

```text
github_code/data/data/instances/orders_20.pkl
```

### 方式二：修改 `SAPPO_I.py` 中的订单路径

把训练函数中的路径：

```python
with open("../data/data/instances/orders_20.pkl", "rb") as f:
    orders = pickle.load(f)
```

改为你的实际路径。

评估阶段同理，需要检查 `evaluate_greedy()` 使用的订单路径是否存在。例如当前代码中评估可能读取：

```python
../data/data/instances/orders_40_w6l20.pkl
```

如果项目中没有该文件，需要改成已有文件，例如：

```python
../data/data/instances/orders_40.pkl
```

---

## 6. 生成订单数据

推荐使用：

```text
data/generat_order_data.py
```

该脚本会根据仓库环境中的商品信息随机生成订单，并保存为 `.csv` 和 `.pkl` 两种格式。

核心参数包括：

```python
generator = GenerateData(
    warehouse,
    total_orders=100,
    poisson_parameter=100,
    max_items_per_order=5
)
```

参数含义：

| 参数 | 含义 |
|---|---|
| `total_orders` | 生成订单总数 |
| `poisson_parameter` | 订单到达间隔参数，数值越大，订单到达越稀疏 |
| `max_items_per_order` | 单个订单最多包含的商品数量 |

运行：

```bash
cd github_code/data
python generat_order_data.py
```

生成后请确认 `.pkl` 文件保存位置与 `SAPPO_I.py` 中读取路径一致。

---

## 7. 训练运行方式

进入算法目录：

```bash
cd github_code/agent
```

运行训练脚本：

```bash
python SAPPO_I.py
```

程序入口位于文件末尾：

```python
if __name__ == '__main__':
    env = WarehouseEnv()
    agent = PPOAgent()
    train(agent, env)
```

默认训练设置：

```python
train(agent, env, n_episodes=2000, max_steps=5000)
```

含义：

| 参数 | 含义 |
|---|---|
| `n_episodes` | 训练 episode 数量 |
| `max_steps` | 每个 episode 最大决策步数 |
| `eval_interval` | 每隔多少 episode 执行一次 greedy 策略评估 |
| `n_eval_episodes` | 每次评估运行多少个 episode |

如需快速测试代码能否正常运行，可以先临时改小：

```python
train(agent, env, n_episodes=5, max_steps=500)
```

---

## 8. 输出结果

训练完成后，代码会保存训练数据和图像。

### 8.1 训练数据

默认保存到：

```text
results/SAPPO/SAPPO_lambda20/p2r6/SAPPO_I_p2r6_training_data.npz
```

包含指标：

| 指标 | 含义 |
|---|---|
| `episode_rewards` | 每个 episode 的总奖励 |
| `policy_losses` | 策略网络损失 |
| `value_losses` | 值函数网络损失 |
| `episode_makespan` | 每个 episode 的完工时间 |
| `episode_n_step` | 每个 episode 的决策步数 |
| `episode_avg_decision_time` | 平均单步决策时间 |
| `running_min_makespan` | 当前训练过程中出现过的最小完工时间 |
| `flow_mean_done` | 已完成订单的平均流经时间 |
| `eval_flow_mean` | greedy 评估阶段的平均订单流经时间 |
| `eval_flow_std` | greedy 评估阶段的订单流经时间标准差 |

### 8.2 训练曲线

默认保存到：

```text
results/SAPPO/SAPPO_lambda20/p2r6/
```

包括：

```text
reward_curve.png
makespan_curve.png
flow_time_curve.png
eval_flow_time_curve.png
```

---

## 9. 常见问题

### 9.1 ModuleNotFoundError: No module named 'env'

该错误通常不是 PPO 训练代码的问题，而是 `.pkl` 订单文件的问题。

原因是：

```text
pickle 文件会记录对象原始所属模块路径。
```

例如，你原来项目中的包名叫：

```text
env
```

而老师 GitHub 项目中的包名叫：

```text
environment
```

如果订单 `.pkl` 是在旧项目中生成的，里面可能记录了：

```text
env.class_object.Order
```

现在放到老师项目里读取时，Python 找不到 `env`，就会报错。

推荐解决方法：

1. 使用当前项目中的订单生成脚本重新生成 `.pkl` 文件；
2. 保证 `SAPPO_I.py`、环境包、订单生成脚本使用同一个包名；
3. 不要直接混用旧项目生成的 `.pkl` 文件。

临时兼容方法是在项目根目录创建 `env` 兼容包，将其转发到 `environment`，但更推荐重新生成订单文件。

### 9.2 FileNotFoundError: orders_20.pkl

说明订单路径不对。请检查运行目录。

如果在 `agent/` 目录下运行：

```bash
python SAPPO_I.py
```

则：

```python
../data/data/instances/orders_20.pkl
```

对应的是：

```text
github_code/data/data/instances/orders_20.pkl
```

如果从项目根目录运行，路径含义会变化。建议固定从 `agent/` 目录运行，或者把订单路径改成基于 `__file__` 的绝对路径。

### 9.3 Warning: visdom not installed

这是可视化工具缺失提示，不影响训练。

如需安装：

```bash
pip install visdom
```

如不需要实时可视化，可以忽略。

### 9.4 No valid actions available at this step

说明当前环境状态下没有可执行动作。可能原因包括：

1. 订单数据与环境设置不匹配；
2. 机器人、拣货员、拣选点状态更新异常；
3. 订单中的商品或拣选点编号与当前仓库环境不一致；
4. `.pkl` 文件来自旧版本环境，类结构或对象属性不兼容。

建议优先重新生成订单数据，并确认生成订单所用环境与训练环境一致。

---

## 10. 代码修改建议

为了减少路径和包名问题，建议后续将订单路径改为基于 `SAPPO_I.py` 文件位置的绝对路径，例如：

```python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ORDERS_20_PATH = os.path.abspath(os.path.join(
    BASE_DIR, "..", "data", "data", "instances", "orders_20.pkl"
))
```

然后读取：

```python
with open(ORDERS_20_PATH, "rb") as f:
    orders = pickle.load(f)
```

这样无论从项目根目录运行，还是从 `agent/` 目录运行，订单路径都不会因为当前工作目录变化而失效。

---

## 11. 项目运行流程总结

完整流程如下：

```text
准备仓库环境
    ↓
生成或检查订单 pkl 文件
    ↓
运行 agent/SAPPO_I.py
    ↓
环境 reset，载入订单
    ↓
每一步构造合法动作集合
    ↓
策略网络在合法动作中采样动作
    ↓
环境 step，返回 reward 和下一状态
    ↓
episode 结束后计算 GAE
    ↓
执行 PPO 策略更新和值函数更新
    ↓
记录 reward、loss、makespan、flow time
    ↓
保存 npz 数据和 png 曲线
```

---

## 12. 说明

本 README 仅服务于 `SAPPO_I.py` 对应的 SAPPO-I 训练流程。项目中的其他算法文件和实验版本未纳入说明范围。如果后续切换到 MAPPO、DQN、A3C 或其他版本，需要另行整理对应 README。
