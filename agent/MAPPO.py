"""
多智能体PPO算法：机器人路径规划智能体、拣货员任务分配智能体

输入：环境状态
输出：智能体动作、即时奖励、训练曲线、训练完成的模型参数

主要模块：
演员网络、评论家网络
动作选择函数、优势计算函数、网络参数更新函数
主函数（多周期训练和测试）
"""
'''
标题：第一次完成并上传
提示：Multi-Agent在train中建立
'''
# MAPPO.py (修正版)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from CNN import CNNFeatureExtractor
from ATN import ActionTransformer
from conj import config

device = config.device

class Memory:
    def __init__(self, n_amrs, n_pickers):
        self.n_amrs = n_amrs
        self.n_pickers = n_pickers
        self.clear()

    def clear(self):
        # rollout-collected data
        self.states = []           # list of (4,H,W) tensors (detached cpu)
        self.rewards = []          # list of floats
        self.amr_pairs = []        # list of cpu tensors (n_amrs, n_cand, d)
        self.picker_pairs = []     # list of cpu tensors (n_pickers, n_cand, d)
        self.amr_actions = []      # list of lists of ints
        self.picker_actions = []   # list of lists of ints
        self.amr_log_probs = []    # list of lists of floats (old log probs)
        self.picker_log_probs = [] # list of lists of floats
        self.amr_values = []       # list of lists of floats (critic at sampling time)
        self.picker_values = []    # list of lists of floats
        # computed after rollout (pure float lists)
        self.returns = None
        self.advantages = None

class PPOActor(nn.Module):
    def __init__(self, input_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

class PPOCritic(nn.Module):
    def __init__(self, input_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

class MAPPO:
    def __init__(self, cfg=config):
        self.cfg = cfg
        self.device = cfg.device

        # 1. 先定义所有网络
        self.cnn = CNNFeatureExtractor(4, cfg.cnn_feat_dim).to(self.device)
        self.atn_amr = ActionTransformer(cfg.action_pair_dim, cfg.atn_embed_dim).to(self.device)
        self.atn_picker = ActionTransformer(cfg.action_pair_dim, cfg.atn_embed_dim).to(self.device)

        actor_input_dim = cfg.cnn_feat_dim + cfg.atn_embed_dim
        self.amr_actors = nn.ModuleList([PPOActor(actor_input_dim).to(self.device) for _ in range(cfg.n_amrs)])
        self.picker_actors = nn.ModuleList([PPOActor(actor_input_dim).to(self.device) for _ in range(cfg.n_pickers)])
        self.critic = PPOCritic(cfg.cnn_feat_dim).to(self.device)

        # 2. 然后收集参数（此时所有网络都已定义）
        all_parameters = (
            list(self.cnn.parameters()) +
            list(self.atn_amr.parameters()) +
            list(self.atn_picker.parameters()) +
            list(self.critic.parameters()) +
            [p for actor in list(self.amr_actors) + list(self.picker_actors) for p in actor.parameters()]
        )

        # 3. 最后定义优化器
        self.optimizer = torch.optim.Adam(all_parameters, lr=cfg.actor_lr)

    # --------------------- sampling ---------------------
    def encode_state(self, state):
        # accept (4,H,W) or (B,4,H,W)
        if state.dim() == 3:
            state = state.unsqueeze(0)
        state = state.to(self.device)
        return self.cnn(state)  # (B, feat_dim) 将四通道输入装变成AC能理解的向量

    def select_actions(self, state, amr_pairs, picker_pairs):
        """
        在 rollout（ rollout 阶段） 期间调用。
        此时允许 ATN、Actor 和 Critic 构建计算图。
        返回 Python 原生的 float / int 类型，以便存入 Memory。
        """
        amr_pairs = amr_pairs.to(self.device) # 候选动作特征
        picker_pairs = picker_pairs.to(self.device)

        state_feat = self.encode_state(state)  # (1, feat)，这就是通过CNN提取的状态
        state_feat0 = state_feat.squeeze(0)    # (feat) 直接整成一维张量

        # produce ATN embeddings (with grad during sampling)
        amr_feats = self.atn_amr(amr_pairs)         # (n_amrs, n_cand, atn_embed_dim)
        picker_feats = self.atn_picker(picker_pairs) # (n_pickers, n_cand, atn_embed_dim)

        state_value = self.critic(state_feat).squeeze(0)
        state_value_float = float(state_value.detach().cpu().item())

        res = {
            'amr_actions': [],
            'amr_log_probs': [],
            'amr_values': [],
            'picker_actions': [],
            'picker_log_probs': [],
            'picker_values': []
        }

        # AMRs
        for i in range(self.cfg.n_amrs):
            cand_feats = amr_feats[i]  # (n_cand, atn_embed_dim)
            n_cand = cand_feats.shape[0] # 有几个候选动作
            fused = torch.cat([state_feat0.unsqueeze(0).repeat(n_cand, 1), cand_feats], dim=-1)  # (n_cand, input_dim)
            '''
            矩阵形式，第0维是行，repeat(n_cand, 1)将行复制n_cand次，列不变，这样才能跟cand_feats同维度
            '''
            logits = self.amr_actors[i](fused)  # (n_cand,) 调用第 i 个 Actor 网络的 forward(fused)
            dist = Categorical(logits=logits)
            a = dist.sample()
            logp = dist.log_prob(a)

            res['amr_actions'].append(int(a.item()))
            res['amr_log_probs'].append(float(logp.detach().cpu().item()))
            res['amr_values'].append(state_value_float)

        # Pickers
        for j in range(self.cfg.n_pickers):
            cand_feats = picker_feats[j]
            n_cand = cand_feats.shape[0]
            fused = torch.cat([state_feat0.unsqueeze(0).repeat(n_cand,1), cand_feats], dim=-1)
            logits = self.picker_actors[j](fused)
            dist = Categorical(logits=logits)
            a = dist.sample()
            logp = dist.log_prob(a)

            res['picker_actions'].append(int(a.item()))
            res['picker_log_probs'].append(float(logp.detach().cpu().item()))
            res['picker_values'].append(state_value_float)

        return res

    # --------------------- GAE returns (pure python/numpy) ---------------------
    def compute_gae_returns(self, rewards, values, last_value=0.0): # 计算广义优势估计GAE
        """
        rewards: list floats length T
        values: list floats length T (baseline per timestep)
        returns, advs: lists floats length T (pure python/numpy)
        """
        T = len(rewards)
        returns = [0.0] * T
        advs = [0.0] * T
        gae = 0.0
        next_value = last_value
        for t in reversed(range(T)):
            delta = rewards[t] + self.cfg.gamma * next_value - values[t]
            gae = delta + self.cfg.gamma * self.cfg.lam * gae
            advs[t] = float(gae)
            returns[t] = float(advs[t] + values[t])
            next_value = values[t]
        # normalize advs
        import numpy as _np
        advs_arr = _np.array(advs, dtype=_np.float32)
        if advs_arr.std() > 0:
            advs_arr = (advs_arr - advs_arr.mean()) / (advs_arr.std() + 1e-8)
        return returns, advs_arr.tolist()

    # --------------------- update ---------------------
    def update(self, memory: Memory):
        """
        Precondition: memory.returns and memory.advantages are pure python lists of floats (length T).
        Update actors and critic. Use logits & Categorical(logits=...) for numerical stability,
        ensure no graph reuse (cand_feats/state_feats detached appropriately).
        """
        T = len(memory.rewards)
        if T == 0:
            return

        states = torch.stack(memory.states).to(self.device)  # (T,4,H,W)
        state_feats = self.cnn(states)  # (T, feat_dim)

        returns = torch.tensor(memory.returns, dtype=torch.float32, device=self.device)
        advantages = torch.tensor(memory.advantages, dtype=torch.float32, device=self.device)

        # 累积所有损失
        total_loss = 0
        batch_count = 0

        # update AMR actors
        for i in range(self.cfg.n_amrs):
            old_logps = [float(memory.amr_log_probs[t][i]) for t in range(T)] # 提出old.pi[s|a]

            for epoch in range(self.cfg.epochs): # 对同一批数据训练多少遍
                for t in range(T):
                    # candidate features - 移除detach，允许ATN训练
                    cand_feats = memory.amr_pairs[t][i].to(self.device)  # (n_cand, d)，从 memory 里取出第 t 步、第 i 个 AMR 的候选动作特征矩阵，并把它搬到训练设备
                    # n_cand = 候选动作数量，d = 单个候选动作的特征维度。
                    n_cand = cand_feats.shape[0] # 读取候选动作个数，后面要用来对状态重复或对输出做索引。

                    cand_embed = self.atn_amr(cand_feats.unsqueeze(0)).squeeze(0)  # (n_cand, atn_dim)
                    # 把状态特征与每个候选动作的embedding拼接（concatenate）在一起，作为actor的输入。
                    fused = torch.cat([state_feats[t].unsqueeze(0).repeat(n_cand, 1), cand_embed], dim=-1)
                    logits = self.amr_actors[i](fused)  # (n_cand,) 得到每个动作的logits
                    dist = Categorical(logits=logits)
                    # compute new log prob at chosen action
                    chosen_idx = int(memory.amr_actions[t][i]) # 取出当时在 rollout 时为第 i 个 AMR 在时间步 t 选择的动作索引（整数）
                    # dist.log_prob expects a tensor (scalar)
                    chosen_tensor = torch.tensor(chosen_idx, dtype=torch.long, device=self.device)
                    new_logp = dist.log_prob(chosen_tensor)
                    old_logp_tensor = torch.tensor(old_logps[t], dtype=torch.float32, device=self.device)

                    ratio = torch.exp(new_logp - old_logp_tensor)
                    adv = advantages[t] # 在train里面

                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1.0 - self.cfg.ppo_clip, 1.0 + self.cfg.ppo_clip) * adv # PPO策略损失函数

                    entropy = dist.entropy().mean()  # entropy scalar
                    loss_t = -torch.min(surr1, surr2) - self.cfg.entropy_coef * entropy
                    total_loss = total_loss + loss_t
                    batch_count += 1

        # update Picker actors
        for j in range(self.cfg.n_pickers):
            old_logps = [float(memory.picker_log_probs[t][j]) for t in range(T)]

            for epoch in range(self.cfg.epochs):
                for t in range(T):
                    cand_feats = memory.picker_pairs[t][j].to(self.device)  # 移除detach
                    n_cand = cand_feats.shape[0]

                    # 移除no_grad，允许ATN训练
                    cand_embed = self.atn_picker(cand_feats.unsqueeze(0)).squeeze(0)

                    fused = torch.cat([state_feats[t].unsqueeze(0).repeat(n_cand, 1), cand_embed], dim=-1)
                    logits = self.picker_actors[j](fused)
                    dist = Categorical(logits=logits)
                    chosen_idx = int(memory.picker_actions[t][j])
                    chosen_tensor = torch.tensor(chosen_idx, dtype=torch.long, device=self.device)
                    new_logp = dist.log_prob(chosen_tensor)
                    old_logp_tensor = torch.tensor(old_logps[t], dtype=torch.float32, device=self.device)

                    ratio = torch.exp(new_logp - old_logp_tensor)
                    adv = advantages[t]

                    entropy = dist.entropy().mean()
                    loss_t = -torch.min(ratio * adv, torch.clamp(ratio, 1.0 - self.cfg.ppo_clip,
                                                                 1.0 + self.cfg.ppo_clip) * adv) - self.cfg.entropy_coef * entropy
                    total_loss = total_loss + loss_t
                    batch_count += 1

        # Critic update (use returns as target)
        critic_values = self.critic(state_feats).squeeze(-1)  # 直接使用state_feats，不需要detach
        critic_loss = F.mse_loss(critic_values, returns)
        total_loss += critic_loss * self.cfg.value_coef  # 使用value_coef加权critic损失

        # 统一更新所有参数
        self.optimizer.zero_grad()
        if batch_count > 0:
            total_loss = total_loss / batch_count  #

        total_loss.backward()  # 直接反向传播

        # 收集所有参数进行梯度裁剪
        all_params = []
        for g in self.optimizer.param_groups:
            all_params.extend(g['params'])
        torch.nn.utils.clip_grad_norm_(all_params, self.cfg.max_grad_norm)

        self.optimizer.step()

        #为了显示visdom，添加状态字典
        # 在 MAPPO 类中添加以下方法

    def state_dict(self):
        """返回所有网络的状态字典"""
        state = {
            'cnn': self.cnn.state_dict(),
            'atn_amr': self.atn_amr.state_dict(),
            'atn_picker': self.atn_picker.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

        # 保存所有AMR actor的状态
        state['amr_actors'] = []
        for i, actor in enumerate(self.amr_actors):
            state['amr_actors'].append(actor.state_dict())

        # 保存所有Picker actor的状态
        state['picker_actors'] = []
        for j, actor in enumerate(self.picker_actors):
            state['picker_actors'].append(actor.state_dict())

        return state

    def load_state_dict(self, state_dict):
        """加载所有网络的状态字典"""
        self.cnn.load_state_dict(state_dict['cnn'])
        self.atn_amr.load_state_dict(state_dict['atn_amr'])
        self.atn_picker.load_state_dict(state_dict['atn_picker'])
        self.critic.load_state_dict(state_dict['critic'])
        self.optimizer.load_state_dict(state_dict['optimizer'])

        # 加载AMR actors
        for i, actor_state in enumerate(state_dict['amr_actors']):
            self.amr_actors[i].load_state_dict(actor_state)

        # 加载Picker actors
        for j, actor_state in enumerate(state_dict['picker_actors']):
            self.picker_actors[j].load_state_dict(actor_state)
