# # env/warehouse_env.py
# import numpy as np
# import random
#
#
# class WarehouseEnv:
#     def __init__(self, grid_h=20, grid_w=20, n_robots=8, n_pickers=5, max_time=2000):
#         self.grid_h = grid_h
#         self.grid_w = grid_w
#         self.n_cells = grid_h * grid_w
#         self.n_robots = n_robots
#         self.n_pickers = n_pickers
#         self.max_time = max_time
#         self.current_time = 0
#
#         # 智能体状态
#         self.robots = []  # 每个机器人的位置和状态
#         self.pickers = []  # 每个拣货员的位置和状态
#         self.orders = []
#
#         # 订单处理参数
#         self.single_item_processing_time = 2.0
#         self.completed_orders = []
#         self.order_history = []
#
#         self.reset()
#
#     def reset(self):
#         self.current_time = 0
#         self.completed_orders = []
#         self.order_history = []
#
#         # 初始化机器人位置
#         self.robots = []
#         for i in range(self.n_robots):
#             r = random.randrange(self.grid_h)
#             c = random.randrange(self.grid_w)
#             self.robots.append({
#                 'id': i,
#                 'position': [r, c],
#                 'target': None,
#                 'carrying_order': None,
#                 'status': 'idle'  # idle, moving, working, arrived
#             })
#
#         # 初始化拣货员位置
#         self.pickers = []
#         for i in range(self.n_pickers):
#             r = random.randrange(self.grid_h)
#             c = random.randrange(self.grid_w)
#             self.pickers.append({
#                 'id': i,
#                 'position': [r, c],
#                 'target': None,
#                 'current_task': None,
#                 'status': 'idle'  # idle, moving, picking
#             })
#
#         self.orders = []
#         return self._get_state()
#
#     def _get_state(self):
#         # 构建状态矩阵
#         C0 = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)  # 机器人位置
#         C1 = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)  # 拣货员位置
#         C2 = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)  # 需求热度
#         C3 = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)  # 订单存在性
#
#         # 填充机器人位置
#         for robot in self.robots:
#             r, c = robot['position']
#             C0[r, c] += 1.0
#
#         # 填充拣货员位置
#         for picker in self.pickers:
#             r, c = picker['position']
#             C1[r, c] += 1.0
#
#         # 填充订单信息
#         for order in self.orders:
#             r = order['cell'] // self.grid_w
#             c = order['cell'] % self.grid_w
#             C2[r, c] += order.get('qty', 1)
#             C3[r, c] = 1.0
#
#         return np.stack([C0, C1, C2, C3], axis=0)
#
#
#     def _position_to_cell(self, position):
#         """将位置转换为单元格索引"""
#         return position[0] * self.grid_w + position[1]
#
#     def _cell_to_position(self, cell):
#         """将单元格索引转换为位置"""
#         return [cell // self.grid_w, cell % self.grid_w]
#
#     def step(self, actions):
#         """
#         执行多智能体动作
#         actions: 字典，包含'amr_actions'和'picker_actions'
#         每个动作是网格索引 (0 到 grid_h*grid_w-1)
#         """
#         self.current_time += 1
#
#         total_reward = 0.0
#         done = False
#         info = {
#             'individual_rewards': [],
#             'completed_orders': [],
#             'active_orders_count': len(self.orders)
#         }
#
#         # 执行AMR动作 - 直接使用网格索引
#         amr_actions = actions.get('amr_actions', [])
#         for i, action in enumerate(amr_actions):
#             if i < len(self.robots):
#                 reward = self._execute_amr_action(i, action)
#                 total_reward += reward
#                 info['individual_rewards'].append(reward)
#
#         # 执行Picker动作 - 直接使用网格索引
#         picker_actions = actions.get('picker_actions', [])
#         for i, action in enumerate(picker_actions):
#             if i < len(self.pickers):
#                 reward = self._execute_picker_action(i, action)
#                 total_reward += reward
#                 info['individual_rewards'].append(reward)
#
#         # 处理拣货完成
#         self._process_picking()
#
#         # 时间惩罚
#         total_reward -= 0.01
#
#         # 检查是否结束
#         if self.current_time >= self.max_time or len(self.orders) == 0:
#             done = True
#
#         next_state = self._get_state()
#         return next_state, total_reward, done, info
#
#     def _execute_amr_action(self, amr_id, target_cell):
#         """执行单个AMR的动作"""
#         robot = self.robots[amr_id]
#
#         # 将单元格索引转换为位置
#         target_r = target_cell // self.grid_w
#         target_c = target_cell % self.grid_w
#
#         # 设置目标
#         robot['target'] = [target_r, target_c]
#         robot['status'] = 'moving'
#
#         # 移动机器人
#         current_r, current_c = robot['position']
#
#         # 简单移动逻辑：直接移动到相邻位置
#         if current_r < target_r:
#             current_r += 1
#         elif current_r > target_r:
#             current_r -= 1
#         elif current_c < target_c:
#             current_c += 1
#         elif current_c > target_c:
#             current_c -= 1
#
#         robot['position'] = [current_r, current_c]
#
#         # 检查是否到达目标
#         reward = 0.0
#         if current_r == target_r and current_c == target_c:
#             robot['status'] = 'arrived'
#             # 到达目标点奖励
#             reward += 0.5
#
#             # 检查是否有订单
#             cell_idx = current_r * self.grid_w + current_c
#             for order in self.orders:
#                 if order['cell'] == cell_idx:
#                     reward += 1.0  # 找到订单奖励
#
#         return reward
#
#     def _execute_picker_action(self, picker_id, target_cell):
#         """执行单个Picker的动作"""
#         picker = self.pickers[picker_id]
#
#         # 将单元格索引转换为位置
#         target_r = target_cell // self.grid_w
#         target_c = target_cell % self.grid_w
#
#         # 设置目标
#         picker['target'] = [target_r, target_c]
#         picker['status'] = 'moving'
#
#         # 移动拣货员
#         current_r, current_c = picker['position']
#
#         # 简单移动逻辑
#         if current_r < target_r:
#             current_r += 1
#         elif current_r > target_r:
#             current_r -= 1
#         elif current_c < target_c:
#             current_c += 1
#         elif current_c > target_c:
#             current_c -= 1
#
#         picker['position'] = [current_r, current_c]
#
#         # 检查是否到达目标
#         reward = 0.0
#         if current_r == target_r and current_c == target_c:
#             picker['status'] = 'picking'
#             reward += 0.3  # 到达目标点奖励
#
#         return reward
#
#     def _process_arrivals(self):
#         """处理AMR和Picker到达目标点的情况"""
#         # 移动AMR
#         for robot in self.robots:
#             if robot['status'] == 'moving' and robot['target'] is not None:
#                 current_pos = robot['position']
#                 target_pos = robot['target']
#
#                 # 移动一步
#                 if current_pos[0] < target_pos[0]:
#                     current_pos[0] += 1
#                 elif current_pos[0] > target_pos[0]:
#                     current_pos[0] -= 1
#                 elif current_pos[1] < target_pos[1]:
#                     current_pos[1] += 1
#                 elif current_pos[1] > target_pos[1]:
#                     current_pos[1] -= 1
#
#                 # 检查是否到达目标
#                 if current_pos[0] == target_pos[0] and current_pos[1] == target_pos[1]:
#                     robot['status'] = 'arrived'
#
#         # 移动Picker
#         for picker in self.pickers:
#             if picker['status'] == 'moving' and picker['target'] is not None:
#                 current_pos = picker['position']
#                 target_pos = picker['target']
#
#                 # 移动一步
#                 if current_pos[0] < target_pos[0]:
#                     current_pos[0] += 1
#                 elif current_pos[0] > target_pos[0]:
#                     current_pos[0] -= 1
#                 elif current_pos[1] < target_pos[1]:
#                     current_pos[1] += 1
#                 elif current_pos[1] > target_pos[1]:
#                     current_pos[1] -= 1
#
#                 # 检查是否到达目标
#                 if current_pos[0] == target_pos[0] and current_pos[1] == target_pos[1]:
#                     picker['status'] = 'picking'
#
#     def _process_picking(self):
#         """处理拣货完成逻辑"""
#         # 检查Picker是否在AMR位置进行拣货
#         for picker in self.pickers:
#             if picker['status'] == 'picking':
#                 for robot in self.robots:
#                     if (robot['status'] == 'arrived' and
#                             robot['position'][0] == picker['position'][0] and
#                             robot['position'][1] == picker['position'][1]):
#
#                         # 完成拣货
#                         cell_idx = self._position_to_cell(robot['position'])
#                         for order in list(self.orders):
#                             if order['cell'] == cell_idx:
#                                 order['qty'] -= 1
#
#                                 if order['qty'] <= 0:
#                                     # 订单完成
#                                     completed_order = order.copy()
#                                     completed_order['completion_time'] = self.current_time
#                                     completed_order['total_processing_time'] = self.current_time - completed_order[
#                                         'created']
#                                     self.completed_orders.append(completed_order)
#                                     self.orders.remove(order)
#
#                         # 重置状态
#                         robot['status'] = 'idle'
#                         robot['target'] = None
#                         picker['status'] = 'idle'
#                         picker['target'] = None
#                         break
#
#     def insert_order(self, cell_idx, qty=1, deadline=None):
#         order = {
#             'cell': int(cell_idx),
#             'qty': int(qty),
#             'deadline': deadline,
#             'created': self.current_time,
#             'remaining_qty': int(qty)
#         }
#         self.orders.append(order)
#         self.order_history.append(order.copy())
#
#     # 保持现有的统计方法
#     def get_completion_times(self):
#         completion_times = []
#         for order in self.orders:
#             remaining_time = order['qty'] * self.single_item_processing_time
#             target_r = order['cell'] // self.grid_w
#             target_c = order['cell'] % self.grid_w
#
#             min_robot_distance = float('inf')
#             for robot in self.robots:
#                 r_pos, c_pos = robot['position']
#                 distance = abs(r_pos - target_r) + abs(c_pos - target_c)
#                 if distance < min_robot_distance:
#                     min_robot_distance = distance
#
#             estimated_completion = self.current_time + min_robot_distance + remaining_time
#             completion_times.append(estimated_completion)
#
#         return completion_times
#
#     def get_max_completion_time(self):
#         completion_times = self.get_completion_times()
#         return max(completion_times) if completion_times else 0.0
#
#     def get_total_remaining_items(self):
#         total = 0
#         for order in self.orders:
#             total += order['qty']
#         return total
#
#     def get_single_item_processing_time(self):
#         return self.single_item_processing_time
#
#     def get_order_statistics(self):
#         total_orders = len(self.order_history)
#         completed_orders = len(self.completed_orders)
#         active_orders = len(self.orders)
#
#         avg_completion_time = 0
#         if completed_orders > 0:
#             avg_completion_time = sum(
#                 order['total_processing_time'] for order in self.completed_orders) / completed_orders
#
#         return {
#             'total_orders': total_orders,
#             'completed_orders': completed_orders,
#             'active_orders': active_orders,
#             'avg_completion_time': avg_completion_time,
#             'current_max_completion': self.get_max_completion_time()
#         }


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
