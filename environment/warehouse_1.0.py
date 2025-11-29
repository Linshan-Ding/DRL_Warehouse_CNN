"""
动作：（拣货员，拣货位）
"""
import numpy as np
import random
import pickle
import gymnasium as gym
import sys

# ==========================================
# 1. 配置与基础类定义 (整合 Config & Class_Object)
# ==========================================

class Config:
    def __init__(self):
        self.parameters = self.parameter()

    def parameter(self):
        return {
            "warehouse": {
                "shelf_capacity": 20,  # 单层货架储货位数量 (N_l)
                "shelf_levels": 3,
                "aisle_num": 9,  # 巷道数量 (N_w)
                "shelf_length": 1.0,
                "shelf_width": 1.0,
                "aisle_width": 2.0,
                "entrance_width": 2.0,
                "depot_position": (18, 0)
            },
            "robot": {
                "robot_speed": 3.0,
                "robot_num": 5
            },
            "picker": {
                "picker_speed": 0.75,
                "picker_num": 3
            },
            "order": {
                "pack_time": 20,
                "unit_delay_cost": 0.1,
            },
            "item": {
                "pick_time": 10
            }
        }


# --- 基础实体类 ---

class Depot:
    def __init__(self, position):
        self.position = position


class StorageBin:
    def __init__(self, bin_id, position, item_id=None, pick_point_id=None):
        self.bin_id = bin_id
        self.position = position
        self.item_id = item_id
        self.pick_point_id = pick_point_id
        self.robot_queue = []
        self.picker = None


class Item:
    def __init__(self, item_id, bin_id, position, pick_point_id):
        self.item_id = item_id
        self.bin_id = bin_id
        self.position = position
        self.pick_point_id = pick_point_id
        self.pick_time = 10  # 默认值，实际从 Config 获取


class Order:
    def __init__(self, order_id, items, arrive_time=0, due_time=None):
        self.order_id = order_id
        self.items = items
        self.arrive_time = arrive_time
        self.due_time = due_time
        self.complete_time = None
        self.unpicked_items = list(items)  # 浅拷贝列表
        self.picked_items = []


class PickPoint:
    def __init__(self, point_id, position, item_ids, storage_bin_ids):
        self.point_id = point_id
        self.position = position
        self.item_ids = item_ids
        self.storage_bin_ids = storage_bin_ids
        self.robot_queue = []
        self.picker = None
        self.unpicked_items = []  # 用于状态统计

    @property
    def is_idle(self):
        # 只有当有机器人排队 且 没有拣货员时，才需要分配拣货员
        return len(self.robot_queue) > 0 and self.picker is None


# --- 智能体类 ---

class Robot(Config):
    def __init__(self, position):
        super().__init__()
        self.param = self.parameters["robot"]
        self.position = position
        self.speed = self.param["robot_speed"]
        self.state = 'idle'

        self.order = None
        self.item_pick_order = []
        self.pick_point = None

        # 时间状态
        self.move_to_pick_point_time = float('inf')
        self.pick_point_complete_time = float('inf')
        self.move_to_depot_time = float('inf')
        self.working_time = 0

        # 拣货策略: 2=最近
        self.pick_point_selection_rule = 2

    def assign_order(self, order):
        self.order = order
        self.item_pick_order = list(order.items)

    def next_pick_point(self, pick_points_dict):
        if not self.item_pick_order:
            return None

        # 简单策略：找最近的
        current_pos = self.position
        best_point = None
        min_dist = float('inf')

        # 找出当前订单涉及的所有 PickPoint
        target_point_ids = list(set([item.pick_point_id for item in self.item_pick_order]))

        for pid in target_point_ids:
            point = pick_points_dict[pid]
            dist = abs(point.position[0] - current_pos[0]) + abs(point.position[1] - current_pos[1])
            if dist < min_dist:
                min_dist = dist
                best_point = point

        return best_point

    @property
    def items(self):
        # 返回当前拣货位需要拣选的商品
        if self.order and self.pick_point:
            return [item for item in self.order.items if item.pick_point_id == self.pick_point.point_id]
        return []


class Picker(Config):
    def __init__(self):
        super().__init__()
        self.param = self.parameters["picker"]
        self.speed = self.param["picker_speed"]
        self.state = 'idle'
        self.position = (0, 0)  # 初始位置由环境设定

        self.pick_point = None
        self.pick_start_time = float('inf')
        self.pick_end_time = float('inf')
        self.working_time = 0


# ==========================================
# 2. 仓库环境类 (WarehouseEnv) - 核心逻辑
# ==========================================

class WarehouseEnv(gym.Env, Config):
    def __init__(self):
        super().__init__()
        self.wh_param = self.parameters["warehouse"]

        # 尺寸参数
        self.N_l = self.wh_param["shelf_capacity"]
        self.N_w = self.wh_param["aisle_num"]
        self.S_l = self.wh_param["shelf_length"]
        self.S_w = self.wh_param["shelf_width"]
        self.S_b = self.wh_param["aisle_width"]
        self.S_d = self.wh_param["entrance_width"]
        self.S_a = self.wh_param["aisle_width"]
        self.depot_position = self.wh_param["depot_position"]
        self.pack_time = self.parameters["order"]["pack_time"]

        # 资源数量
        self.N_robots = self.parameters["robot"]["robot_num"]
        self.N_pickers = self.parameters["picker"]["picker_num"]

        # 核心容器
        self.pick_points = {}
        self.pick_points_list = []
        self.storage_bins = {}
        self.items = {}
        self.depot_object = Depot(self.depot_position)

        # 构建地图
        self.create_warehouse_graph()

        # 状态变量
        self.robots = []
        self.pickers = []
        self.orders = []
        self.orders_not_arrived = []
        self.orders_unassigned = []
        self.orders_uncompleted = []

        self.current_time = 0
        self.done = False

    def create_warehouse_graph(self):
        """构建仓库拓扑结构"""
        for nw in range(1, self.N_w + 1):
            for nl in range(1, self.N_l + 1):
                # 坐标计算
                x = self.S_d + (2 * nw - 1) * self.S_w + (2 * nw - 1) / 2 * self.S_a
                y = self.S_b + (2 * nl - 1) / 2 * self.S_l
                position = (x, y)

                point_id = f"{nw}-{nl}"

                # 模拟左右货位及商品
                items_ids = []
                bin_ids = []
                for side in ['left', 'right']:
                    bin_id = f"{point_id}-{side}"
                    item_id = f"{bin_id}-item"

                    # 创建对象
                    self.storage_bins[bin_id] = StorageBin(bin_id, position, item_id, point_id)
                    item = Item(item_id, bin_id, position, point_id)
                    item.pick_time = self.parameters["item"]["pick_time"]  # 设置拣货时间
                    self.items[item_id] = item

                    items_ids.append(item_id)
                    bin_ids.append(bin_id)

                # 创建 PickPoint
                pick_point = PickPoint(point_id, position, items_ids, bin_ids)
                self.pick_points[point_id] = pick_point
                self.pick_points_list.append(pick_point)

    def shortest_path_between_pick_points(self, entity, target):
        """曼哈顿距离计算 (简化版)"""
        p1 = entity.position
        p2 = target.position if hasattr(target, 'position') else target
        # 简单假设：只能走直角
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def adjust_resources(self):
        """初始化机器人和拣货员"""
        self.robots = [Robot(self.depot_position) for _ in range(self.N_robots)]
        self.pickers = []
        for i in range(self.N_pickers):
            p = Picker()
            # 初始位置均匀分布或在Depot
            p.position = self.pick_points_list[i * 5].position if i * 5 < len(
                self.pick_points_list) else self.depot_position
            self.pickers.append(p)

    def reset(self, orders):
        """重置环境"""
        self.current_time = 0
        self.done = False
        self.adjust_resources()

        # 重置订单状态
        self.orders = orders
        # 必须按到达时间排序
        self.orders_not_arrived = sorted(orders, key=lambda x: x.arrive_time)
        self.orders_unassigned = []
        self.orders_uncompleted = []

        # 清空拣货点队列
        for pp in self.pick_points.values():
            pp.robot_queue = []
            pp.picker = None
            pp.unpicked_items = []

        # 初始推进
        self.time_to_next_decision_point()
        return self.state_extractor()

    def time_to_next_decision_point(self):
        """
        核心事件循环：推进时间直到需要 RL 介入决策
        RL 介入条件：有空闲 Picker 且 有待处理 PickPoint
        """
        while not self.done:
            # 检查是否满足决策条件
            if len(self.idle_pickers) > 0 and len(self.idle_pick_points) > 0:
                return

            # 收集所有未来事件的时间
            future_events = []

            # 1. 订单到达
            if self.orders_not_arrived:
                future_events.append(self.orders_not_arrived[0].arrive_time)

            # 2. 机器人事件
            for r in self.robots:
                if r.move_to_pick_point_time > self.current_time:
                    future_events.append(r.move_to_pick_point_time)
                if r.pick_point_complete_time > self.current_time:
                    future_events.append(r.pick_point_complete_time)
                if r.move_to_depot_time > self.current_time:
                    future_events.append(r.move_to_depot_time)

            # 3. 拣货员事件
            for p in self.pickers:
                if p.pick_start_time > self.current_time:
                    future_events.append(p.pick_start_time)
                if p.pick_end_time > self.current_time:
                    future_events.append(p.pick_end_time)

            # 过滤无效时间 (inf)
            valid_events = [t for t in future_events if t != float('inf')]

            # 如果没有未来事件
            if not valid_events:
                if not self.orders_not_arrived and not self.orders_unassigned and not self.orders_uncompleted:
                    self.done = True
                    print(f"仿真结束。总耗时: {self.current_time:.2f}")
                    return
                else:
                    # 异常保护：还有订单但没有事件（可能是逻辑卡死），强制结束
                    print("Warning: No future events but orders pending. Breaking loop.")
                    self.done = True
                    return

            # 推进时间
            next_time = min(valid_events)
            self.current_time = next_time

            # === 处理事件 ===

            # A. 订单到达
            while self.orders_not_arrived and self.current_time >= self.orders_not_arrived[0].arrive_time:
                order = self.orders_not_arrived.pop(0)
                self.orders_unassigned.append(order)
                self.orders_uncompleted.append(order)

            # 尝试分配订单给空闲机器人 (自动策略，无需 RL)
            self.assign_order_to_robot()

            # B. 机器人到达拣货点
            for r in self.robots:
                if self.current_time == r.move_to_pick_point_time:
                    pp = r.pick_point
                    pp.robot_queue.append(r)
                    r.position = pp.position
                    r.move_to_pick_point_time = float('inf')

                    # 如果该点已有 Picker 正在工作，需加入其工作流 (Requirement 1 叠加时间)
                    # 这里简化处理：只有当 Picker 分配任务时才计算时间。
                    # 如果 Picker 已经在忙，新的机器人只能排队等待 Picker 这一轮结束并释放后，重新分配。
                    # 或者：实时动态延长 Picker 的 pick_end_time。
                    # 为保持 Step 逻辑清晰，采用“Picker 完成后释放 -> 重新 Step 分配”的模式。

            # C. 拣货员到达并开始作业
            for p in self.pickers:
                if self.current_time == p.pick_start_time:
                    p.pick_start_time = float('inf')
                    # 此时状态仍为 busy，等待 pick_end_time

            # D. 拣货完成 (Picker & Robot)
            for r in self.robots:
                if self.current_time == r.pick_point_complete_time:
                    r.pick_point_complete_time = float('inf')
                    pp = r.pick_point
                    if r in pp.robot_queue:
                        pp.robot_queue.remove(r)

                    # 更新订单商品状态
                    for item in r.items:
                        r.order.picked_items.append(item)
                        if item in r.item_pick_order:
                            r.item_pick_order.remove(item)
                        if item in r.order.unpicked_items:
                            r.order.unpicked_items.remove(item)

                    # 规划下一步
                    if r.item_pick_order:
                        next_pp = r.next_pick_point(self.pick_points)
                        dist = self.shortest_path_between_pick_points(r, next_pp)
                        r.move_to_pick_point_time = self.current_time + dist / r.speed
                        r.pick_point = next_pp
                        r.state = 'busy'
                    else:
                        dist = self.shortest_path_between_pick_points(r, self.depot_object)
                        r.move_to_depot_time = self.current_time + dist / r.speed + self.pack_time

            for p in self.pickers:
                if self.current_time == p.pick_end_time:
                    p.pick_end_time = float('inf')
                    p.state = 'idle'
                    if p.pick_point:
                        p.pick_point.picker = None
                        p.pick_point = None

            # E. 机器人回库
            for r in self.robots:
                if self.current_time == r.move_to_depot_time:
                    r.move_to_depot_time = float('inf')
                    r.state = 'idle'
                    r.position = self.depot_position
                    if r.order in self.orders_uncompleted:
                        self.orders_uncompleted.remove(r.order)
                    r.order = None
                    # 机器人释放后尝试分配新订单
                    self.assign_order_to_robot()

    def step(self, action):
        """
        执行 RL 动作：分配 [Picker, PickPoint]
        """
        picker, pick_point = action

        # 1. 绑定
        picker.state = 'busy'
        picker.pick_point = pick_point
        pick_point.picker = picker

        # 2. 移动时间
        dist = self.shortest_path_between_pick_points(picker, pick_point)
        travel_time = dist / picker.speed
        picker.pick_start_time = self.current_time + travel_time
        picker.position = pick_point.position  # 逻辑更新

        # 3. 计算作业时间 (Requirement 1: 叠加时间)
        cumulative_pick_time = 0
        for robot in pick_point.robot_queue:
            # 当前机器人所需拣货时间
            robot_items = [i for i in robot.items if i.pick_point_id == pick_point.point_id]
            job_time = sum(item.pick_time for item in robot_items)

            cumulative_pick_time += job_time

            # 设定机器人完成时刻 = Picker到达 + 累积作业时间
            robot.pick_point_complete_time = picker.pick_start_time + cumulative_pick_time

        picker.pick_end_time = picker.pick_start_time + cumulative_pick_time

        # 4. 推进
        self.time_to_next_decision_point()

        # 5. 返回
        state = self.state_extractor()
        reward = 0  # 需自行实现奖励函数
        return state, reward, self.done, False, {}

    def assign_order_to_robot(self):
        """自动分配策略"""
        while self.idle_robots and self.orders_unassigned:
            robot = self.idle_robots.pop(0)
            order = self.orders_unassigned.pop(0)

            robot.assign_order(order)
            robot.state = 'busy'

            target = robot.next_pick_point(self.pick_points)
            if target:
                robot.pick_point = target
                dist = self.shortest_path_between_pick_points(robot, target)
                robot.move_to_pick_point_time = self.current_time + dist / robot.speed
            else:
                # 空订单或异常
                robot.state = 'idle'

    def state_extractor(self):
        # 简单提取：每个 PickPoint 的排队长度
        # RL 可用的 State 通常是 Tensor，这里返回 numpy
        return np.array([len(pp.robot_queue) for pp in self.pick_points.values()], dtype=np.float32)

    @property
    def idle_robots(self):
        return [r for r in self.robots if r.state == 'idle']

    @property
    def idle_pickers(self):
        return [p for p in self.pickers if p.state == 'idle']

    @property
    def idle_pick_points(self):
        return [pp for pp in self.pick_points_list if pp.is_idle]


# ==========================================
# 3. 主程序入口 (测试与执行)
# ==========================================

if __name__ == "__main__":
    # 初始化仓库环境
    warehouse = WarehouseEnv()

    # 读取订单数据，orders.pkl文件中
    with open("../data/data/instances/orders_100.pkl", "rb") as f:
        orders = pickle.load(f)

    # 2. 初始化环境
    env = WarehouseEnv()
    state = env.reset(orders)

    print("开始仿真...")
    step_count = 0

    # 3. 仿真循环
    while not env.done:
        # 获取合法动作空间
        avail_pickers = env.idle_pickers
        avail_points = env.idle_pick_points

        if not avail_pickers or not avail_points:
            break  # 理论上不会发生，除非 done 状态延迟

        # === 模拟 RL Agent 动作 ===
        # 策略：随机选择一个 Picker，去服务排队最长的 Point
        picker = random.choice(avail_pickers)
        # 按排队长度降序排列
        avail_points.sort(key=lambda p: len(p.robot_queue), reverse=True)
        pick_point = avail_points[0]

        action = [picker, pick_point]

        # 执行 Step
        next_state, reward, done, truncated, info = env.step(action)

        step_count += 1
        print(f"Step {step_count}: Time={env.current_time:.2f}s | "
              f"Assigned Picker {env.pickers.index(picker)} -> Point {pick_point.point_id} "
              f"(Queue: {len(pick_point.robot_queue)})")

    print("仿真结束。")