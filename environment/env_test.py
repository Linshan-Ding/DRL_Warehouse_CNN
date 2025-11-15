
# Chatgpt_New_env
# warehouse_env.py

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from agent.conj import config

class WarehouseEnv(gym.Env):
    """
    一个非常精简的仓库环境，可直接用于测试 MAPPO 训练流程。
    状态：4 通道矩阵
      0: 空地
      1: AMR 位置
      2: Picker 位置
      3: 订单点位置
    动作：每个 agent 会有 action_pairs（由 MAPPO 生成），环境只接收 "动作索引"
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.H = config.grid_h
        self.W = config.grid_w

        self.n_amrs = config.n_amrs
        self.n_pickers = config.n_pickers

        # 对于每个 agent，action 是一个整数 (对应于 action_pairs 选择哪一个)
        self.action_space = spaces.MultiDiscrete([5] * (self.n_amrs + self.n_pickers))
        # 5 个动作对应：停 / 上 / 下 / 左 / 右
        self.action_vectors = np.array([
            [0, 0],
            [-1, 0],
            [1, 0],
            [0, -1],
            [0, 1]
        ])

        # 状态空间 4×H×W
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(4, self.H, self.W),
                                            dtype=np.float32)

        self.reset()

    # --------------------------------------
    # 辅助函数：生成状态图
    # --------------------------------------
    def _get_obs(self):
        obs = np.zeros((4, self.H, self.W), dtype=np.float32)
        for ax, ay in self.amr_pos:
            obs[1, ax, ay] = 1
        for px, py in self.picker_pos:
            obs[2, px, py] = 1
        ox, oy = self.order_point
        obs[3, ox, oy] = 1
        return obs

    # --------------------------------------
    # reset
    # --------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 初始化 AMR 和 Picker 随机位置
        self.amr_pos = []
        self.picker_pos = []

        all_positions = set()

        # AMRs
        while len(self.amr_pos) < self.n_amrs:
            x = np.random.randint(0, self.H)
            y = np.random.randint(0, self.W)
            if (x, y) not in all_positions:
                all_positions.add((x, y))
                self.amr_pos.append([x, y])

        # Pickers
        while len(self.picker_pos) < self.n_pickers:
            x = np.random.randint(0, self.H)
            y = np.random.randint(0, self.W)
            if (x, y) not in all_positions:
                all_positions.add((x, y))
                self.picker_pos.append([x, y])

        # 订单随机位置
        self.order_point = (
            np.random.randint(0, self.H),
            np.random.randint(0, self.W)
        )

        obs = self._get_obs()
        return obs, {}

    # --------------------------------------
    # step
    # --------------------------------------
    def step(self, action_list):
        """
        action_list: [amr_0, amr_1, ..., picker_0, picker_1, ...]
        """

        reward = 0

        # AMR 移动
        for i, a in enumerate(action_list[:self.n_amrs]):
            dx, dy = self.action_vectors[a]
            x, y = self.amr_pos[i]
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.H and 0 <= ny < self.W:
                self.amr_pos[i] = [nx, ny]

            # → AMR 接近订单点奖励
            ox, oy = self.order_point
            reward += -0.01 * (abs(nx - ox) + abs(ny - oy))

        # Picker 移动
        for j, a in enumerate(action_list[self.n_amrs:]):
            dx, dy = self.action_vectors[a]
            x, y = self.picker_pos[j]
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.H and 0 <= ny < self.W:
                self.picker_pos[j] = [nx, ny]

            ox, oy = self.order_point
            reward += -0.01 * (abs(nx - ox) + abs(ny - oy))

        # 完成订单奖励
        for (ax, ay) in self.amr_pos:
            if (ax, ay) == self.order_point:
                reward += 5
        for (px, py) in self.picker_pos:
            if (px, py) == self.order_point:
                reward += 5

        obs = self._get_obs()
        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, {}

    # --------------------------------------
    # action_pairs 生成（给 MAPPO）
    # --------------------------------------
    def get_action_pairs(self):
        """
        生成每个 agent 的 5 个候选动作的特征（4 维）
        结构：
        amr_pairs: (n_amrs, 5, 4)
        picker_pairs: (n_pickers, 5, 4)
        """
        amr_pairs = []
        for x, y in self.amr_pos:
            feats = []
            for dx, dy in self.action_vectors:
                feats.append([dx, dy, x, y])  # 你可按真实项目修改
            amr_pairs.append(feats)

        picker_pairs = []
        for x, y in self.picker_pos:
            feats = []
            for dx, dy in self.action_vectors:
                feats.append([dx, dy, x, y])
            picker_pairs.append(feats)

        return (
            torch.tensor(amr_pairs, dtype=torch.float32),
            torch.tensor(picker_pairs, dtype=torch.float32)
        )
