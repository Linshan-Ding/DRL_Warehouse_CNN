import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
import math
import os
import matplotlib

try:
    matplotlib.use('TkAgg')
except Exception:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from CNN import CNNFeatureExtractor
from conj import config
from env.env_I import WarehouseEnv

# 添加 Visdom 导入
try:
    import visdom
    VISDOM_AVAILABLE = True
except ImportError:
    VISDOM_AVAILABLE = False
    print("Warning: visdom not installed. Install with: pip install visdom")

# =====================
# Output paths (paper)
# =====================
# NOTE: Use script-relative absolute paths so outputs don't depend on current working directory (CWD).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 训练数据保存路径（.npz）
TRAINING_DATA_PATH = os.path.abspath(os.path.join(
    BASE_DIR, "..", "results", "SAPPO", "SAPPO_lambda20", "p2r6", "SAPPO_I_p2r6_training_data.npz"
))
# 论文用图保存目录（PNG）
FIGURES_DIR = os.path.abspath(os.path.join(
    BASE_DIR, "..", "results", "SAPPO", "SAPPO_lambda20", "p2r6"
))

# # 训练数据保存路径（.npz）
# TRAINING_DATA_PATH = os.path.abspath(os.path.join(
#     BASE_DIR, "..", "revise","w6l20","PPO_w6l20.npz"
# ))
# # 论文用图保存目录（PNG）
# FIGURES_DIR = os.path.abspath(os.path.join(
#     BASE_DIR, "..", "revise","PPO_w6l20"
# ))

print("[PATH] CWD =", os.getcwd())
print("[PATH] __file__ =", os.path.abspath(__file__))
print("[PATH] TRAINING_DATA_PATH =", TRAINING_DATA_PATH)
print("[PATH] FIGURES_DIR =", FIGURES_DIR)

# Update policy every N episodes by accumulating rollout in memory
UPDATE_EVERY_EPISODES = 1  # update every episode (original PPO_I behavior)


env = WarehouseEnv()
device = config.device
action_num = (env.N_robots + env.N_pickers) * env.N_l * env.N_w + env.N_robots

class PolicyNetwork(nn.Module):
    def __init__(self, cfg=config):
        super().__init__()
        self.device = device

        self.cnn = CNNFeatureExtractor(4, cfg.cnn_output_dim).to(self.device)

        self.mlp = nn.Sequential(
            nn.Linear(cfg.cnn_output_dim, 2048), # 原先256
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, action_num)  # 输出所有动作的得分
        )

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        # 提取状态特征
        if state.dim() == 3:
            state = state.unsqueeze(0)
        state_feat = self.cnn(state)  # [B, cnn_feat_dim]

        # MLP处理
        scores = self.mlp(state_feat)  # [B, action_num]

        return scores


class ValueNetwork(nn.Module):
    def __init__(self, cfg=config):
        super(ValueNetwork, self).__init__()
        self.device = device

        # 使用和PolicyNetwork相同的CNN结构（但不共享权重）
        self.cnn = CNNFeatureExtractor(4, cfg.cnn_output_dim).to(self.device)

        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(cfg.cnn_output_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

    def forward(self, state):  # state_tensor:[B,4,H,W]
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        if state.dim() == 3:
            state = state.unsqueeze(0)
        # 提取特征
        features = self.cnn(state)  # [B, cnn_output_dim]
        # 计算价值
        value = self.value_head(features)  # [B, 1]

        return value

class PPOAgent:
    def __init__(self, cfg=config):
        self.cfg = cfg
        self.device = cfg.device

        # 策略网络（包含ATN）
        self.policy_net = PolicyNetwork(cfg).to(self.device)
        self.value_net = ValueNetwork(cfg).to(self.device)
        # 优化器
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.actor_lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=cfg.critic_lr)

        # 经验缓存
        self.memory = {
            'states': [],  # 状态张量
            'values': [],  # 价值标量
            'rewards': [],  # 奖励
            'dones': [],  # 是否完成
            'log_prob': [],
            'selected_action_index':[],
            'legal_action_index':[]
        }

    def clear_memory(self):
        """清空经验缓存"""
        for key in self.memory:
            self.memory[key] = []

    def get_avaliable_action(self, env):
        picker_available_action = []
        robot_available_action = []
        # picker候选动作
        #思路：总共picker动作空间3*10 + robot动作空间5*10
        for pkr in env.pickers:
            if pkr.state == 'idle':
                p_index = env.pickers.index(pkr) # 我们以环境中picker列表的索引计算
                for pp in env.pick_points_list:
                    if pp.is_idle:  # 有机器人排队且无拣货员
                        j_index = env.pick_points_list.index(pp)
                        picker_available_action.append((p_index,j_index))

        for r in env.robots:
            r_index = env.robots.index(r)
            if r.state == 'idle' and len(r.item_pick_order) > 0 and r.position == env.depot_object.position: # 从depot点出发的robot
                for i in r.item_pick_order:
                    p = i.position
                    pp = env.pick_point_dict[p]
                    t_index = env.pick_points_list.index(pp)
                    robot_available_action.append((r_index,t_index))

            elif r.state == 'idle' and r.order is not None and len(r.item_pick_order) > 0: # 已经在拣选路上的robot
                for i in r.item_pick_order:
                    p = i.position
                    pp = env.pick_point_dict[p]
                    t_index = env.pick_points_list.index(pp)
                    robot_available_action.append((r_index, t_index))

            elif r.state == 'idle' and r.order is not None and len(r.item_pick_order) == 0 and r.pick_point is not None: # 完成拣货将要回到depot点的robot
                robot_available_action.append((r_index, -1))

        return picker_available_action,robot_available_action

    def total_action_index_pair(self,env,state):
        picker_available_action,robot_available_action = self.get_avaliable_action(env)
        # 获取全部robot，picker动作和mlp对应的索引，及具体动作
        total_action = [] # 先picker，后robot,最后几项是回depot点
        # 整体空间：
        # picker->pick_point
        # robot->pick_point
        # robot->depot
        for picker in env.pickers:
            for pick_point in env.pick_points_list:
                total_action.append((picker,pick_point))
        for robot in env.robots:
            for pick_point in env.pick_points_list:
                total_action.append((robot, pick_point))
        for robot in env.robots:
            d = env.depot_object
            total_action.append((robot,d))

        action_to_index = {action: idx for idx, action in enumerate(total_action)} # 创建映射字典

        legal_action_index = []
        # picker 合法动作
        for p_index, pp_index in picker_available_action:
            picker = env.pickers[p_index]
            pp = env.pick_points_list[pp_index]
            legal_action_index.append(action_to_index[(picker, pp)])

        # robot 合法动作
        for r_index, t_index in robot_available_action:
            robot = env.robots[r_index]
            if t_index != -1:
                target = env.pick_points_list[t_index]
            else:
                target = env.depot_object
            legal_action_index.append(action_to_index[(robot, target)])

        if len(legal_action_index) == 0:
            raise RuntimeError("No valid actions available at this step!Simulation error!")

        # 处理 state
        if isinstance(state, np.ndarray):
            state_tensor = torch.from_numpy(state).float().to(self.device)
        else:
            state_tensor = state.to(self.device)

        if state_tensor.dim() == 3:
            state_tensor = state_tensor.unsqueeze(0)

        with torch.no_grad():
            value = self.value_net(state_tensor)
            # 1. 得到动作分数
            logits = self.policy_net(state_tensor)
            # 2. 构造 mask
            invalid_mask = torch.ones_like(logits, dtype=torch.bool) # 全部置true
            invalid_mask[:,legal_action_index] = False

            if len(legal_action_index) == 0:
                raise RuntimeError("No legal actions at this step!")

            # 3. mask 无效动作
            masked_logits = logits.masked_fill(invalid_mask, -1e9)

            # 4. 建立分布 & 采样
            probs = torch.softmax(masked_logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)
            selected_action = total_action[action_idx.item()]

            # 判断是picker_action还是robot_action
            if action_idx < env.N_pickers * env.N_w * env.N_l:
                picker_action = selected_action
                robot_action = None
                action = (picker_action,robot_action)
            else:
                robot_action = selected_action
                picker_action = None
                action = (picker_action, robot_action)

        self.memory['states'].append(state if isinstance(state, np.ndarray) else state.cpu().numpy())
        self.memory['values'].append(value.item())
        self.memory['log_prob'].append(log_prob.item())
        self.memory['selected_action_index'].append(action_idx.item())
        self.memory['legal_action_index'].append(legal_action_index)

        return action

    def select_action_greedy(self, env, state):
        """评估用：贪心选动作（argmax），不写 memory，不采样"""
        picker_available_action, robot_available_action = self.get_avaliable_action(env)

        total_action = []
        for picker in env.pickers:
            for pick_point in env.pick_points_list:
                total_action.append((picker, pick_point))
        for robot in env.robots:
            for pick_point in env.pick_points_list:
                total_action.append((robot, pick_point))
        for robot in env.robots:
            total_action.append((robot, env.depot_object))

        action_to_index = {action: idx for idx, action in enumerate(total_action)}

        legal_action_index = []
        for p_index, pp_index in picker_available_action:
            picker = env.pickers[p_index]
            pp = env.pick_points_list[pp_index]
            legal_action_index.append(action_to_index[(picker, pp)])

        for r_index, t_index in robot_available_action:
            robot = env.robots[r_index]
            target = env.pick_points_list[t_index] if t_index != -1 else env.depot_object
            legal_action_index.append(action_to_index[(robot, target)])

        if len(legal_action_index) == 0:
            raise RuntimeError("No valid actions available at this step! (eval)")

        # state -> tensor
        if isinstance(state, np.ndarray):
            state_tensor = torch.from_numpy(state).float().to(self.device)
        else:
            state_tensor = state.to(self.device)
        if state_tensor.dim() == 3:
            state_tensor = state_tensor.unsqueeze(0)

        with torch.no_grad():
            logits = self.policy_net(state_tensor)

            invalid_mask = torch.ones_like(logits, dtype=torch.bool)
            invalid_mask[:, legal_action_index] = False
            masked_logits = logits.masked_fill(invalid_mask, -1e9)

            # greedy: 直接 argmax
            action_idx = torch.argmax(masked_logits, dim=1).item()
            selected_action = total_action[action_idx]

        if action_idx < env.N_pickers * env.N_w * env.N_l:
            return (selected_action, None)
        else:
            return (None, selected_action)

    def compute_returns_and_advantages(self, last_value=0.0):
        """计算GAE和returns"""
        rewards = self.memory['rewards']
        values = self.memory['values']
        dones = self.memory['dones']

        T = len(rewards)
        returns = [0.0] * T
        advantages = [0.0] * T

        gae = 0.0
        next_value = last_value

        for t in reversed(range(T)):
            delta = rewards[t] + self.cfg.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.cfg.gamma * self.cfg.lam * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            next_value = values[t]

        # 标准化advantages
        advantages = np.array(advantages, dtype=np.float32)
        # print(
        #     "[GAE raw] mean:", advantages.mean(),
        #     "std:", advantages.std(),
        #     "min:", advantages.min(),
        #     "max:", advantages.max()
        # )
        if advantages.std() > 0:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantages = np.clip(advantages, -5.0, 5.0) # 改

        # print(
        #     "[GAE raw] mean:", advantages.mean(),
        #     "std:", advantages.std(),
        #     "min:", advantages.min(),
        #     "max:", advantages.max()
        # )

        print(f"reward mean: {sum(rewards) / len(rewards)}")

        return returns, advantages.tolist()

    def update(self):
        if len(self.memory['rewards']) == 0:
            return 0.0, 0.0

        # ===== 1) returns & advantages =====
        returns, advantages = self.compute_returns_and_advantages()
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)

        # old values / old log_probs / actions
        old_values = torch.tensor(self.memory['values'], dtype=torch.float32, device=self.device)

        old_log_probs = torch.tensor(self.memory['log_prob'], dtype=torch.float32, device=self.device)

        actions = torch.tensor(self.memory['selected_action_index'], dtype=torch.long, device=self.device)

        T = len(returns)
        idxs = np.arange(T)

        PPO_EPOCHS = getattr(self.cfg, "epochs", 2)
        BATCH_SIZE = getattr(self.cfg, "batch_size", 32)
        ENTROPY_COEF = getattr(self.cfg, "entropy_coef", 0.01)

        last_policy_loss = 0.0
        last_value_loss = 0.0

        kl_list = []
        clipfrac_list = []
        entropy_list = []
        ratio_list = []

        for _ in range(PPO_EPOCHS):
            np.random.shuffle(idxs)

            for start in range(0, T, BATCH_SIZE):
                mb = idxs[start:start + BATCH_SIZE]

                policy_losses = []
                value_losses = []
                entropies = []

                for t in mb:
                    state = torch.from_numpy(self.memory['states'][t]).float().to(self.device)
                    if state.dim() == 3:
                        state = state.unsqueeze(0)  # [1,4,H,W]

                    # ===== 2) 重新算 logits，并做 mask（必须重建 mask）=====
                    # 这里要用当前状态下的合法动作集合
                    # 先不管该动作是否合法，对所有动作评分，再mask非法动作
                    # 所以必须在采样时把 legal_action_index 也存下来，否则 update 无法重建 mask。

                    logits = self.policy_net(state)  # [1, 1445] 之类

                    # 1) 先查 logits 是否 NaN/Inf（定位根因）
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        raise RuntimeError(f"[update] logits has NaN/Inf at step t={t}")

                    legal_action_index = self.memory['legal_action_index'][t]
                    if len(legal_action_index) == 0:
                        raise RuntimeError(f"[update] legal_action_index empty at t={t}")

                    invalid_mask = torch.ones_like(logits, dtype=torch.bool)
                    invalid_mask[:, legal_action_index] = False
                    masked_logits = logits.masked_fill(invalid_mask, -1e9)

                    # 2) 再查 masked_logits
                    if torch.isnan(masked_logits).any() or torch.isinf(masked_logits).any():
                        raise RuntimeError(f"[update] masked_logits has NaN/Inf at step t={t}")

                    dist = torch.distributions.Categorical(logits=masked_logits)

                    new_log_prob = dist.log_prob(actions[t])  # 标量tensor
                    entropy = dist.entropy()

                    # ratio = torch.exp(new_log_prob - old_log_probs[t])
                    ratio_cap = getattr(self.cfg, "ratio_cap", 5.0)  # 建议先 5.0（更稳），或 8.0（更不保守）
                    cap = math.log(ratio_cap)  # ln(5)=1.609, ln(8)=2.079

                    log_ratio = new_log_prob - old_log_probs[t]
                    log_ratio = torch.clamp(log_ratio, -cap, cap)  # 真正“限幅”
                    ratio = torch.exp(log_ratio)
                    # approx KL（旧策略 vs 新策略）
                    kl_list.append((old_log_probs[t] - new_log_prob).detach())

                    # clip fraction
                    # clipped = (torch.abs(ratio - 1.0) > self.cfg.ppo_clip).float()
                    clipped = ((ratio < 1.0 - self.cfg.ppo_clip) | (ratio > 1.0 + self.cfg.ppo_clip)).float()

                    clipfrac_list.append(clipped.detach())

                    entropy_list.append(entropy.detach())
                    ratio_list.append(ratio.detach())

                    surr1 = ratio * advantages[t]
                    surr2 = torch.clamp(
                        ratio,
                        1.0 - self.cfg.ppo_clip,
                        1.0 + self.cfg.ppo_clip
                    ) * advantages[t]
                    policy_losses.append(-torch.min(surr1, surr2))
                    entropies.append(entropy)

                    # ===== 3) value + value clip（更稳）=====
                    value_pred = self.value_net(state).squeeze()
                    v_pred_clipped = old_values[t] + torch.clamp(
                        value_pred - old_values[t],
                        -self.cfg.ppo_clip,
                        self.cfg.ppo_clip
                    )
                    v_loss1 = (value_pred - returns[t]).pow(2)
                    v_loss2 = (v_pred_clipped - returns[t]).pow(2)
                    value_losses.append(torch.max(v_loss1, v_loss2))

                policy_loss = torch.mean(torch.stack(policy_losses))
                value_loss = torch.mean(torch.stack(value_losses))
                value_loss = torch.clamp(value_loss, max=1e6)

                entropy_mean = torch.mean(torch.stack(entropies))

                total_loss = policy_loss + self.cfg.value_coef * value_loss - ENTROPY_COEF * entropy_mean

                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.cfg.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.cfg.max_grad_norm)

                self.policy_optimizer.step()
                self.value_optimizer.step()

                last_policy_loss = policy_loss.item()
                last_value_loss = value_loss.item()

        if len(kl_list) > 0:
            kl = torch.mean(torch.stack(kl_list)).item()
            clipfrac = torch.mean(torch.stack(clipfrac_list)).item()
            ent = torch.mean(torch.stack(entropy_list)).item()
            r_mean = torch.mean(torch.stack(ratio_list)).item()
            r_min = torch.min(torch.stack(ratio_list)).item()
            r_max = torch.max(torch.stack(ratio_list)).item()
            print(
                f"[PPO] approx_kl={kl:.6f} clip_frac={clipfrac:.3f} entropy={ent:.3f} ratio(mean/min/max)={r_mean:.3f}/{r_min:.3f}/{r_max:.3f}")

        self.clear_memory()
        return last_policy_loss, last_value_loss

def episode_flow_time(env):
    """计算本 episode 的订单流经时间(Flow time)统计量（只统计已完成订单）。

    返回: (mean, n_completed, n_total)
    """
    flows = []
    # 已完成订单
    for o in getattr(env, 'orders_completed', []):
        ct = getattr(o, 'complete_time', None)
        at = getattr(o, 'arrive_time', None)
        if ct is not None and at is not None:
            flows.append(float(ct - at))

    n_completed = len(getattr(env, 'orders_completed', []))
    n_total = n_completed + len(getattr(env, 'orders_uncompleted', []))

    if len(flows) == 0:
        return 0.0, n_completed, n_total

    arr = __import__('numpy').asarray(flows, dtype=__import__('numpy').float32)
    return float(arr.mean()), n_completed, n_total


def evaluate_greedy(agent, n_eval_episodes=3, max_steps=5000, orders_path="../data/data/instances/orders_20.pkl"):
    """评估：greedy policy，不采样；返回每个episode的完成订单 mean flow time 列表"""
    eval_env = WarehouseEnv()
    agent.policy_net.eval()
    agent.value_net.eval()

    order_processing_times = []
    with torch.no_grad():
        for _ in range(n_eval_episodes):
            with open(orders_path, "rb") as f:
                orders = pickle.load(f)

            state = eval_env.reset(orders)

            for t in range(max_steps):
                action = agent.select_action_greedy(eval_env, state)
                next_state, reward, done, truncated, info = eval_env.step(action)
                state = next_state
                if done or truncated:
                    break

            m_done, n_c, n_tot = episode_flow_time(eval_env)
            order_processing_times.append(m_done)

    agent.policy_net.train()
    agent.value_net.train()

    return order_processing_times

def train(agent, env, n_episodes=2000, max_steps=5000):
    """训练函数（记录并保存论文用数据与PNG图）。"""

    # 初始化Visdom
    viz = None
    if VISDOM_AVAILABLE:
        try:
            viz = visdom.Visdom(env='WarehousePPO_I')
            print("Visdom connected successfully!")
        except:
            print("Warning: Could not connect to visdom server. Running without visualization.")
            viz = None

    # =============================
    # 训练数据记录（会保存到 .npz）
    # =============================
    episode_rewards = []
    policy_losses = []
    value_losses = []

    # 订单流经时间（只统计已完成订单）
    flow_mean_done = []

    # 你需要的 3 个指标：
    # - 最小完工时间 min_makespan（全局最小）
    # - 决策点数量 n_step
    # - 平均决策时间 makespan / n_step
    episode_makespan = []
    episode_n_step = []
    episode_avg_decision_time = []
    running_min_makespan = []

    # ===== 可选：定期 greedy eval（若不需要可把 eval_interval 设为 0） =====
    eval_interval = 10
    n_eval_episodes = 3
    eval_flow_mean = []
    eval_flow_std = []

    interrupted = False  # set True if Ctrl+C

    for ep in range(n_episodes):
        try:
            with open("../data/data/instances/orders_20.pkl", "rb") as f: # 改
                orders = pickle.load(f)

            total_step = -1
            state = env.reset(orders)
            ep_reward = 0.0
            legal_cnts = []
            r_pick_list = []
            r_complete_list = []
            done_bonus_list = []
            picked_delta_list = []
            completed_delta_list = []

            # 防止 max_steps==0 导致 done/truncated 未定义
            done = False
            truncated = False

            for t in range(max_steps):
                # 1. 选择动作
                action = agent.total_action_index_pair(env, state)
                # 2. 执行动作
                next_state, reward, done, truncated, info = env.step(action)
                # 统计本步合法动作数（total_action_index_pair 已经把 legal_action_index 存到 memory 末尾）
                legal_cnts.append(len(agent.memory["legal_action_index"][-1]))
                rp = info.get("reward_parts", {})
                r_pick_list.append(rp.get("r_pick", 0.0))
                r_complete_list.append(rp.get("r_complete", 0.0))
                done_bonus_list.append(rp.get("done_bonus", 0.0))
                picked_delta_list.append(rp.get("picked_delta", 0.0))
                completed_delta_list.append(rp.get("completed_delta", 0.0))
                real_done = done or truncated

                # 3. 存 reward / done
                agent.memory['rewards'].append(reward)
                agent.memory['dones'].append(real_done)

                ep_reward += reward
                state = next_state
                total_step = t

                if real_done:
                    break
            # ===== Episode 结束：按 UPDATE_EVERY_EPISODES 触发更新（=1 表示每集更新） =====
            do_update = ((ep + 1) % UPDATE_EVERY_EPISODES == 0)
            if do_update:
                policy_loss, value_loss = agent.update()
            else:
                # 不更新时保持上一次loss（方便画图/日志）；如果还没有则用 0
                policy_loss = policy_losses[-1] if len(policy_losses) > 0 else 0.0
                value_loss = value_losses[-1] if len(value_losses) > 0 else 0.0

            # ===== 统计 makespan / n_step / avg decision time / running min makespan =====
            makespan = float(getattr(env, 'current_time', 0.0))
            n_step = int(total_step + 1) if total_step >= 0 else 0
            avg_decision_time = (makespan / n_step) if n_step > 0 else 0.0

            episode_makespan.append(makespan)
            episode_n_step.append(n_step)
            episode_avg_decision_time.append(avg_decision_time)

            cur_min_ms = min(episode_makespan) if len(episode_makespan) > 0 else makespan
            running_min_makespan.append(float(cur_min_ms))

            # episode 结束原因（done / truncated / max_steps / break_early）
            end_reason = (
                "done" if done else
                ("truncated" if truncated else
                 ("max_steps" if (total_step == max_steps - 1) else "break_early"))
            )

            def _stat(x):
                if len(x) == 0:
                    return (0.0, 0.0, 0.0)
                return (float(np.mean(x)), float(np.min(x)), float(np.max(x)))

            legal_mean, legal_min, legal_max = _stat(legal_cnts)
            print(
                f"[EP{ep + 1}] end={end_reason} steps={n_step} makespan={makespan:.2f} "
                f"reward={ep_reward:.2f} | "
                f"legal(mean/min/max)={legal_mean:.1f}/{legal_min:.0f}/{legal_max:.0f} | "
                f"r_pick_sum={sum(r_pick_list):.2f} "
                f"r_complete_sum={sum(r_complete_list):.2f} done_bonus_sum={sum(done_bonus_list):.2f} | "
                f"picked_delta_sum={sum(picked_delta_list):.0f} completed_delta_sum={sum(completed_delta_list):.0f}"
            )

            # ===== 定期评估（greedy）=====
            if (ep + 1) % eval_interval == 0:
                flows = evaluate_greedy(
                    agent,
                    n_eval_episodes=n_eval_episodes,
                    max_steps=max_steps,
                    orders_path="../data/data/instances/orders_40_w6l20.pkl"
                )
                m = float(np.mean(flows))
                s = float(np.std(flows))
                eval_flow_mean.append(m)
                eval_flow_std.append(s)

                print(f"[EVAL] ep={ep+1} greedy mean_flow(completed) mean={m:.2f} std={s:.2f}")

                if viz is not None:
                    viz.line(
                        Y=[m],
                        X=[ep + 1],
                        win='eval_flow_mean',
                        update='append' if len(eval_flow_mean) > 1 else None,
                        opts=dict(
                            title='Eval Mean Flow Time (greedy)',
                            xlabel='Episode',
                            ylabel='Time',
                            legend=['Eval Mean']
                        )
                    )

            # ===== 记录 =====
            episode_rewards.append(ep_reward)
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)

            m_done, n_c, n_tot = episode_flow_time(env)
            flow_mean_done.append(m_done)

            # ===== 打印信息 =====
            print(
                f"Episode {ep + 1}/{n_episodes} | "
                f"Reward: {ep_reward:.2f} | "
                f"Policy Loss: {policy_loss:.4f} | "
                f"Value Loss: {value_loss:.4f} | "
                f"n_step:{n_step} | "
                f"makespan:{makespan:.3f} | min_makespan:{running_min_makespan[-1]:.3f} | "
                f"avg_dt:{avg_decision_time:.4f} | mean_flow_done:{m_done:.2f}"
            )

            # 使用Visdom绘制图表
            if viz is not None:
                # 1. 绘制奖励曲线
                viz.line(
                    Y=[ep_reward],
                    X=[ep + 1],
                    win='reward',
                    update='append' if ep > 0 else None,
                    opts=dict(
                        title='Episode Reward',
                        xlabel='Episode',
                        ylabel='Total Reward',
                        legend=['Reward']
                    )
                )

                # 1.5 绘制训练阶段的平均订单流经时间（只统计已完成订单）
                viz.line(
                    Y=[m_done],
                    X=[ep + 1],
                    win='flow_time',
                    update='append' if ep > 0 else None,
                    opts=dict(
                        title='Mean Flow Time',
                        xlabel='Episode',
                        ylabel='Time',
                        legend=['Mean Flow']
                    )
                )


                # 2. 绘制策略损失曲线
                viz.line(
                    Y=[policy_loss],
                    X=[ep + 1],
                    win='policy_loss',
                    update='append' if ep > 0 else None,
                    opts=dict(
                        title='Policy Loss',
                        xlabel='Episode',
                        ylabel='Loss',
                        legend=['Policy Loss']
                    )
                )
                # 3. 绘制价值损失曲线
                viz.line(
                    Y=[value_loss],
                    X=[ep + 1],
                    win='value_loss',
                    update='append' if ep > 0 else None,
                    opts=dict(
                        title='Value Loss',
                        xlabel='Episode',
                        ylabel='Loss',
                        legend=['Value Loss']
                    )
                )

                # 4. 可选：绘制移动平均奖励（窗口大小为10）
                if len(episode_rewards) >= 10:
                    avg_reward = np.mean(episode_rewards[-10:])
                    viz.line(
                        Y=[avg_reward],
                        X=[ep + 1],
                        win='avg_reward',
                        update='append' if ep >= 10 else None,
                        opts=dict(
                            title='Moving Average Reward (window=10)',
                            xlabel='Episode',
                            ylabel='Average Reward',
                            legend=['Avg Reward']
                        )
                    )

        except KeyboardInterrupt:
            print("\n[INTERRUPT] Caught Ctrl+C. Saving training data/figures/checkpoints...\n")
            interrupted = True
            break
    # 训练结束后保存训练曲线数据
    if episode_rewards:
        print(f"\nTraining completed!")
        print(f"Average reward: {np.mean(episode_rewards):.2f}")
        print(f"Max reward: {np.max(episode_rewards):.2f}")
        print(f"Min reward: {np.min(episode_rewards):.2f}")

        # 保存训练数据到文件（路径见 TRAINING_DATA_PATH）
        training_data = {
            'episode_rewards': episode_rewards,
            'policy_losses': policy_losses,
            'value_losses': value_losses,
            'episode_makespan': episode_makespan,
            'episode_n_step': episode_n_step,
            'episode_avg_decision_time': episode_avg_decision_time,
            'running_min_makespan': running_min_makespan,
            'min_makespan': (min(episode_makespan) if len(episode_makespan) > 0 else 0.0),
            'flow_mean_done': flow_mean_done,
            'eval_flow_mean': eval_flow_mean,
            'eval_flow_std': eval_flow_std,
            'eval_interval': eval_interval,
        }

        try:
            os.makedirs(os.path.dirname(TRAINING_DATA_PATH), exist_ok=True)
            np.savez(TRAINING_DATA_PATH, **training_data)
            print(f"Training data saved to '{TRAINING_DATA_PATH}'")
        except Exception:
            import traceback
            traceback.print_exc()

        # 训练结束后自动生成论文用图（PNG 输出目录见 FIGURES_DIR）
        try:
            plot_training_curves(training_data, out_dir=FIGURES_DIR)
            print(f"Figures saved to {FIGURES_DIR}")
        except Exception:
            import traceback
            traceback.print_exc()


def plot_training_curves(training_data, out_dir=FIGURES_DIR):
    """用 matplotlib 画论文要用的图，并保存为 PNG。"""
    # 图像保存目录：out_dir（默认 FIGURES_DIR）
    os.makedirs(out_dir, exist_ok=True)

    # --- Reward ---
    rewards = training_data.get('episode_rewards', [])
    if len(rewards) > 0:
        x = np.arange(1, len(rewards) + 1)
        plt.figure()
        plt.plot(x, rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Episode Reward')
        plt.tight_layout()
        # 保存路径：{out_dir}/reward_curve.png
        try:
            _p = os.path.abspath(os.path.join(out_dir, 'reward_curve.png'))
            plt.savefig(_p, dpi=300, bbox_inches='tight')
            print(f"[plot] saved reward_curve.png -> {_p}")
        except Exception as e:
            print(f"[plot] Failed to save reward_curve.png: {e}")
        try:
            plt.show(block=False)
            plt.pause(0.001)
        except Exception:
            pass


    # --- Makespan ---
    makespans = training_data.get('episode_makespan', [])
    if len(makespans) > 0:
        x = np.arange(1, len(makespans) + 1)
        plt.figure()
        plt.plot(x, makespans)
        plt.xlabel('Episode')
        plt.ylabel('Makespan')
        plt.title('Episode Makespan')
        plt.tight_layout()
        # 保存路径：{out_dir}/makespan_curve.png
        try:
            _p = os.path.abspath(os.path.join(out_dir, 'makespan_curve.png'))
            plt.savefig(_p, dpi=300, bbox_inches='tight')
            print(f"[plot] saved makespan_curve.png -> {_p}")
        except Exception as e:
            print(f"[plot] Failed to save makespan_curve.png: {e}")
        try:
            plt.show(block=False)
            plt.pause(0.001)
        except Exception:
            pass


    # --- Flow time (completed orders) ---
    flow_mean = training_data.get('flow_mean_done', [])
    if len(flow_mean) > 0:
        x = np.arange(1, len(flow_mean) + 1)
        plt.figure()
        plt.plot(x, flow_mean)
        plt.xlabel('Episode')
        plt.ylabel('Flow Time')
        plt.title('Flow Time (completed orders)')
        plt.tight_layout()
        # 保存路径：{out_dir}/flow_time_curve.png
        try:
            _p = os.path.abspath(os.path.join(out_dir, 'flow_time_curve.png'))
            plt.savefig(_p, dpi=300, bbox_inches='tight')
            print(f"[plot] saved flow_time_curve.png -> {_p}")
        except Exception as e:
            print(f"[plot] Failed to save flow_time_curve.png: {e}")
        try:
            plt.show(block=False)
            plt.pause(0.001)
        except Exception:
            pass


    # --- Greedy eval flow time (optional) ---
    eval_mean = training_data.get('eval_flow_mean', [])
    eval_std = training_data.get('eval_flow_std', [])
    eval_interval = training_data.get('eval_interval', None)
    if len(eval_mean) > 0 and eval_interval is not None and int(eval_interval) > 0:
        x = np.arange(1, len(eval_mean) + 1) * int(eval_interval)
        plt.figure()
        plt.plot(x, eval_mean)
        if len(eval_std) == len(eval_mean):
            plt.errorbar(x, eval_mean, yerr=eval_std, fmt='none', capsize=3)
        plt.xlabel('Episode')
        plt.ylabel('Flow Time')
        plt.title('Greedy Eval Flow Time (completed orders)')
        plt.tight_layout()
        # 保存路径：{out_dir}/eval_flow_time_curve.png
        try:
            _p = os.path.abspath(os.path.join(out_dir, 'eval_flow_time_curve.png'))
            plt.savefig(_p, dpi=300, bbox_inches='tight')
            print(f"[plot] saved eval_flow_time_curve.png -> {_p}")
        except Exception as e:
            print(f"[plot] Failed to save eval_flow_time_curve.png: {e}")
        try:
            plt.show(block=False)
            plt.pause(0.001)
        except Exception:
            pass



if __name__ == '__main__':
    env = WarehouseEnv()
    agent = PPOAgent()
    train(agent, env)