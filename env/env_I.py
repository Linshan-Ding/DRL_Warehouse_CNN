import numpy as np
import random
import pickle
import gymnasium as gym


# ==========================================
# 1. 配置与基础类定义
# ==========================================

class Config:
    def __init__(self):
        self.parameters = self.parameter()

    def parameter(self):
        return {
            "warehouse": {
                "shelf_capacity": 30,
                "shelf_levels": 3,
                "aisle_num": 9,
                "shelf_length": 1.0,
                "shelf_width": 1.0,
                "aisle_width": 2.0,
                "entrance_width": 2.0,
                "depot_position": (18, 0)
            },
            "robot": {
                "robot_speed": 3.0,
                "robot_num": 6
            },
            "picker": {
                "picker_speed": 0.75,
                "picker_num": 2
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
        self.item_id = item_id  # nw--nl-left/right-item
        self.bin_id = bin_id  # nw--nl-left/right
        self.position = position  # (x,y)
        self.pick_point_id = pick_point_id  # nw--nl
        self.pick_time = 10


class Order:
    def __init__(self, order_id, items, arrive_time=0, due_time=None):
        self.order_id = order_id
        self.items = items
        self.arrive_time = arrive_time
        self.due_time = due_time
        self.complete_time = None
        self.unpicked_items = list(items)
        self.picked_items = []
        # 添加单位延迟成本
        self.unit_delay_cost = 0.1  # 可以根据需要调整


class PickPoint:
    def __init__(self, point_id, position, item_ids, storage_bin_ids):
        self.point_id = point_id
        self.position = position
        self.item_ids = item_ids
        self.storage_bin_ids = storage_bin_ids
        self.robot_queue = []
        self.picker = None

    @property
    def is_idle(self):
        # 任务分配智能体关注：有机器人排队 且 无拣货员
        return len(self.robot_queue) > 0 and self.picker is None


# --- 智能体类 ---

class Robot(Config):
    def __init__(self, robot_id, position):
        super().__init__()
        self.param = self.parameters["robot"]
        self.robot_id = robot_id
        self.position = position
        self.speed = self.param["robot_speed"]
        self.state = 'idle'  # idle, busy

        self.order = None
        self.item_pick_order = []  # 当前订单中剩余未规划的商品
        self.pick_point = None  # 当前所在或前往的拣货位

        # 时间状态
        self.move_to_pick_point_time = float('inf')
        self.pick_point_complete_time = float('inf')
        self.move_to_depot_time = float('inf')

    def assign_order(self, order):
        self.order = order
        self.item_pick_order = list(order.items)

    @property
    def items(self):
        # 返回当前所在拣货位需要拣选的商品
        if self.order and self.pick_point:
            return [item for item in self.order.items if item.pick_point_id == self.pick_point.point_id]
        return []


class Picker(Config):
    def __init__(self, picker_id):
        super().__init__()
        self.param = self.parameters["picker"]
        self.picker_id = picker_id
        self.speed = self.param["picker_speed"]
        self.state = 'idle'  # idle, busy
        self.position = (0, 0)

        self.pick_point = None
        self.pick_start_time = float('inf')
        self.pick_end_time = float('inf')


# ==========================================
# 2. 仓库环境类 (WarehouseEnv)
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

        self.N_robots = self.parameters["robot"]["robot_num"]
        self.N_pickers = self.parameters["picker"]["picker_num"]

        # 核心容器
        self.pick_points = {}
        self.pick_points_list = []
        self.pick_point_dict = {}  # (x, y) -> PickPoint
        self.storage_bins = {}
        self.items = {}
        self.depot_object = Depot(self.depot_position)

        # 构建地图
        self.create_warehouse_graph()

        # 状态变量
        self.robots = []
        self.robot_dict = {}
        self.pickers = []
        self.picker_dict = {}
        self.orders = []
        self.orders_not_arrived = []
        self.orders_unassigned = []
        self.orders_uncompleted = []
        self.orders_completed = []

        self.current_time = 0.0
        self.last_decision_time = 0.0
        self.order_handle_time = 0.0
        self.done = False

        # 统计订单数
        self.order_counter = 0

    def create_warehouse_graph(self):
        for nw in range(1, self.N_w + 1):
            for nl in range(1, self.N_l + 1):
                x = self.S_d + (2 * nw - 1) * self.S_w + (2 * nw - 1) / 2 * self.S_a
                y = self.S_b + (2 * nl - 1) / 2 * self.S_l
                position = (x, y)
                point_id = f"{nw}-{nl}"
                items_ids = []
                bin_ids = []
                for side in ['left', 'right']:
                    bin_id = f"{point_id}-{side}"
                    item_id = f"{bin_id}-item"
                    self.storage_bins[bin_id] = StorageBin(bin_id, position, item_id, point_id)
                    item = Item(item_id, bin_id, position, point_id)
                    item.pick_time = self.parameters["item"]["pick_time"]
                    self.items[item_id] = item
                    items_ids.append(item_id)
                    bin_ids.append(bin_id)

                pick_point = PickPoint(point_id, position, items_ids, bin_ids)
                self.pick_points[point_id] = pick_point
                self.pick_points_list.append(pick_point)
                self.pick_point_dict[position] = pick_point

    def shortest_path_between_pick_points(self, point1, point2):
        x1, y1 = point1.position
        x2, y2 = point2.position
        # 如果两个拣货位在同一巷道，则返回两个拣货位之间的直线路径长度
        if x1 == x2:
            return abs(y1 - y2)
        # 计算从上部绕过和从下部绕过的路径，选择最短路径，并返回路径长度
        else:
            path1 = abs(y1 - self.S_b / 2) + abs(y2 - self.S_b / 2) + abs(x1 - x2)
            path2 = (abs(y1 - (self.S_b * 1.5 + self.N_l * self.S_l)) + abs(y2 - (self.S_b * 1.5 + self.N_l * self.S_l))
                     + abs(x1 - x2))
            return min(path1, path2)

    def adjust_resources(self):
        self.robots = [Robot(i, self.depot_position) for i in range(self.N_robots)]
        for robot in self.robots:
            self.robot_dict[robot.robot_id] = robot

        self.pickers = []
        for i in range(self.N_pickers):
            p = Picker(i)
            # 初始位置均匀分布
            p.position = self.pick_points_list[i * 5].position if i * 5 < len(
                self.pick_points_list) else self.depot_position
            p.pick_point = self.pick_points_list[i * 5]
            self.pickers.append(p)
            self.picker_dict[p.picker_id] = p
            # print(f"picker{p.picker_id}位于{p.pick_point.item_ids}")

    def reset(self, orders):
        self.current_time = 0
        self.last_decision_time = 0
        self.order_handle_time = 0
        self.done = False
        self.adjust_resources()

        self.orders = orders
        self.orders_not_arrived = sorted(orders, key=lambda x: x.arrive_time)
        self.orders_unassigned = []
        self.orders_uncompleted = []
        self.orders_completed = []

        for pp in self.pick_points.values():
            pp.robot_queue = []
            pp.picker = None

        self.time_to_next_decision_point()

        # === reward shaping trackers ===
        # 统计当前剩余未拣货 item 数（orders_uncompleted 里未完成订单）
        self.prev_total_unpicked = sum(len(o.unpicked_items) for o in self.orders_uncompleted)
        # 已完成订单数
        self.prev_completed_orders = len(self.orders_completed)

        return self.state_extractor()

    def time_to_next_decision_point(self):
        """
        事件推进逻辑：
        当满足以下任意条件时停止（返回RL进行决策）：
        1. 存在空闲Picker 且有需要服务的PickPoint（任务分配决策点）
        2. 存在空闲Robot（路径规划决策点：Robot在Depot分到了订单，或者Robot在货架拣完货要去下一站）
        """
        while not self.done:
            # --- 检查是否满足决策点条件 ---
            # 检查是否有机器人需要订单分配 (在Depot)
            robots_needing_order = [r for r in self.robots if
                                    r.state == 'idle' and r.order is None and r.position == self.depot_position]  # 改

            if len(robots_needing_order) > 0 and len(self.orders_unassigned) > 0:
                for r in self.robots:
                    if r.order is None and self.orders_unassigned:
                        # 默认取第一个订单 (或者 RL 可以指定 Order，这里简化处理)
                        order = self.orders_unassigned.pop(0)
                        r.assign_order(order)
                        # print(f"robot{r.robot_id}接到订单{order.order_id}")

            # 1. 任务分配决策点
            if len(self.idle_pickers) > 0 and len(self.idle_pick_points) > 0:
                # print(f"{self.current_time},有picker需要行动")
                return

            # 2. 路径规划决策点
            # 机器人空闲的定义：
            # a. 在Depot，且有未分配订单 (需要在step中分配订单并决定去哪) -> 实际上这里简化为：有空闲机器人且有待分配订单
            # b. 在PickPoint拣货完成，且订单未完成 (需要决定去下一个哪个点)

            # 检查是否有机器人需要路径决策
            robots_needing_path = [
                r for r in self.robots
                if r.state == 'idle'
                   and r.order is not None
            ]
            if len(robots_needing_path) > 0:
                # print(f"{self.current_time},有robot需要行动")
                return

            # --- 如果不需要决策，推进时间 ---

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

            valid_events = [t for t in future_events if t != float('inf')]

            if not valid_events:
                if not self.orders_not_arrived and not self.orders_unassigned and not self.orders_uncompleted:
                    self.done = True
                    return
                else:
                    print("Warning: Simulation stuck. Breaking.")
                    print(len(self.orders_not_arrived), len(self.orders_unassigned), len(self.orders_uncompleted),
                          self.current_time)
                    self.done = True
                    return

            next_time = min(valid_events)
            self.current_time = next_time

            # --- 处理事件 ---

            # A. 订单到达
            while self.orders_not_arrived and self.current_time >= self.orders_not_arrived[0].arrive_time:
                order = self.orders_not_arrived.pop(0)
                self.orders_unassigned.append(order)
                self.orders_uncompleted.append(order)
                # 新订单到达，可能有闲置机器人在Depot等待，循环会再次检查决策条件

            # B. 机器人到达拣货位
            for r in self.robots:
                if self.current_time == r.move_to_pick_point_time:
                    pp = r.pick_point
                    pp.robot_queue.append(r)
                    r.position = pp.position
                    r.move_to_pick_point_time = float('inf')
                    # 进入排队状态，等待Picker来处理（Picker逻辑在Step中）

            # C. 拣货员到达位置 (开始拣货)
            for p in self.pickers:
                if self.current_time == p.pick_start_time:
                    p.pick_start_time = float('inf')
                    # 此时Picker到位，等待pick_end_time结束

            # D. 拣货完成
            # D1. 机器人完成部分
            for r in self.robots:
                if self.current_time == r.pick_point_complete_time:
                    r.pick_point_complete_time = float('inf')
                    pp = r.pick_point
                    if r in pp.robot_queue: pp.robot_queue.remove(r)

                    # 结算物品
                    items_picked = [i for i in r.items]  # 当前拣货位的物品
                    for item in items_picked:
                        r.order.picked_items.append(item)
                        # 从待规划列表中移除 (注意：这里假设items在robot.items中是唯一的)
                        if item in r.item_pick_order:
                            r.item_pick_order.remove(item)
                        if item in r.order.unpicked_items:
                            r.order.unpicked_items.remove(item)

                    # 状态置为 Idle，等待 RL 给出下一个去向 (Next Pick Point or Depot)
                    r.state = 'idle'
                    # 循环将在下一次迭代通过 robots_needing_path 捕获此状态

            # D2. 拣货员完成部分
            for p in self.pickers:
                if self.current_time == p.pick_end_time:
                    p.pick_end_time = float('inf')
                    p.state = 'idle'
                    if p.pick_point:
                        p.pick_point.picker = None
                        p.pick_point = None
                    # 状态置为 Idle，等待 RL 分配下一个 PickPoint

            # E. 机器人回到 Depot (完成订单)
            for r in self.robots:
                if self.current_time == r.move_to_depot_time:
                    r.move_to_depot_time = float('inf')
                    r.state = 'idle'
                    r.position = self.depot_position
                    if r.order is not None:
                        r.order.complete_time = self.current_time
                        if r.order in self.orders_uncompleted:
                            self.orders_uncompleted.remove(r.order)
                            self.orders_completed.append(r.order)
                        r.order = None
                        r.pick_point = None
                        # 状态置为 Idle，等待 RL 分配新订单

    def step(self, action):
        """
        执行多智能体联合动作
        action = (picker_action, robot_action)
        picker_action: (PickerObj, PickPointObj) 或 None
        robot_action: (RobotObj, TargetObj) 或 None.
                      TargetObj 可以是 PickPoint (路径规划) 或 Order (分配订单，暂简化为直接下一步Target)
        """

        picker_act, robot_act = action

        # --- 1. 执行拣货员动作 (任务分配) ---
        if picker_act is not None:
            picker, pick_point = picker_act

            # 绑定
            picker.state = 'busy'
            picker.pick_point = pick_point
            pick_point.picker = picker

            # 计算移动
            dist = self.shortest_path_between_pick_points(picker, pick_point)
            travel_time = dist / picker.speed
            picker.pick_start_time = self.current_time + travel_time
            picker.position = pick_point.position

            # 计算叠加拣选时间 (Requirement 1)
            cumulative_pick_time = 0
            for robot in pick_point.robot_queue:
                robot_items = [i for i in robot.items]  # Robot.items属性已过滤为当前PickPoint的商品
                job_time = sum(item.pick_time for item in robot_items)
                cumulative_pick_time += job_time
                # 同步机器人完成时间
                robot.pick_point_complete_time = picker.pick_start_time + cumulative_pick_time

            picker.pick_end_time = picker.pick_start_time + cumulative_pick_time
            # print(f"picker{picker.picker_id}前往{pick_point.item_ids}")

        # --- 2. 执行机器人动作 (路径规划/订单获取) ---
        if robot_act is not None:
            robot, target = robot_act

            # 情况 A: 机器人在 Depot，且 Target 是 Order (这里假设 Target 是分配了Order后的第一个PickPoint)
            # 为了简化 RL 接口，假设外部 Agent 已经把 Order 分配好了，或者 Target 就是 PickPoint
            # 这里实现逻辑：如果 Robot 没有 Order，说明 Action 隐含了“分配一个 Order 并去 Target”

            if robot.order is not None:
                # 情况 B: 机器人已有 Order，Target 是下一个 PickPoint (或者 Depot)
                robot.state = 'busy'

                if isinstance(target, Depot):
                    # 去 Depot
                    dist = self.shortest_path_between_pick_points(robot, target)
                    robot.move_to_depot_time = self.current_time + dist / robot.speed + self.pack_time
                    robot.pick_point = None
                    # print(f"robot{robot.robot_id}回到depot点")
                elif isinstance(target, PickPoint):
                    # 去 PickPoint
                    robot.pick_point = target
                    dist = self.shortest_path_between_pick_points(robot, target)
                    robot.move_to_pick_point_time = self.current_time + dist / robot.speed
                    # print(f"robot{robot.robot_id}前往{target.item_ids}")
                else:
                    # 异常或空动作
                    pass
        # 3. 推进环境
        self.time_to_next_decision_point()

        # time_to_next_decision_point 会推进 current_time；dt 表示两个决策点之间经过的真实时间
        dt = self.current_time - self.last_decision_time

        reward = self.compute_reward()

        # 更新状态
        state = self.state_extractor()
        done = self.done

        info = {
            "dt": float(dt),
            "makespan": float(self.current_time),
        }
        info["reward_parts"] = getattr(self, "last_reward_parts", {})

        return state, reward, done, False, info

    def state_extractor(self):

        H = self.N_w
        W = self.N_l

        M_queue = np.zeros((H, W), dtype=np.float32)
        M_picker = np.zeros((H, W), dtype=np.float32)
        M_unpicked = np.zeros((H, W), dtype=np.float32)
        M_unassigned = np.zeros((H, W), dtype=np.float32)

        # 计算 unpicked 和 unassigned 计数
        for pp in self.pick_points_list:
            w, l = map(int, pp.point_id.split('-'))
            i, j = w - 1, l - 1

            M_queue[i, j] = len(pp.robot_queue)
            M_picker[i, j] = 0 if pp.picker is None else 1

            # 计算未拣选商品数
            unpicked = 0
            for order in self.orders_uncompleted:
                for item in order.unpicked_items:
                    if item and item.pick_point_id == pp.point_id:
                        unpicked += 1
            M_unpicked[i, j] = unpicked

            # 计算未分配订单的商品数
            unassigned = 0
            for order in self.orders_unassigned:
                for item in order.unpicked_items:
                    if item and item.pick_point_id == pp.point_id:
                        unassigned += 1
            M_unassigned[i, j] = unassigned

        return np.stack(
            [M_queue, M_picker, M_unpicked, M_unassigned],
            axis=0
        )  # (4,H,W)

    def compute_reward(self):
        """
        计算奖励：reward = 上一时刻所有订单的总流经时间 - 当前时刻所有订单的总流经时间
        已完成订单的流经时间 = 完工时间 - 到达时间
        未完成订单的流经时间 = 当前时间 - 到达时间
        """
        # 计算当前时刻所有订单的流经时间
        current_total_time = sum(
            (o.complete_time - o.arrive_time)
            for o in self.orders_completed
        )
        current_total_time += sum(
            (self.current_time - o.arrive_time)  # 未完成订单的流经时间
            for o in self.orders_uncompleted
        )

        num_orders = len(self.orders_completed) + len(self.orders_uncompleted)
        average_order_handle_time = current_total_time / max(1, num_orders)

        # 奖励为总流经时间的差
        reward = self.order_handle_time - average_order_handle_time

        self.last_decision_time = self.current_time
        self.order_handle_time = average_order_handle_time

        return reward

    # --- 辅助属性 ---

    @property
    def idle_robots(self):
        # 返回在 Depot 等待订单的机器人
        return [r for r in self.robots if r.state == 'idle' and r.order is None]

    @property
    def idle_pickers(self):
        return [p for p in self.pickers if p.state == 'idle']

    @property
    def idle_pick_points(self):
        return [pp for pp in self.pick_points_list if pp.is_idle]

    @property
    def robots_needing_planning(self):
        # 返回需要规划下一个点的机器人 (已有订单且Idle)
        return [r for r in self.robots if r.state == 'idle' and r.order is not None]

    @property
    def unpicked_count(self):
        count = {pp.point_id: 0 for pp in self.pick_points_list}
        for order in self.orders_uncompleted:
            for item in order.unpicked_items:
                if item is None:
                    continue
                pid = item.pick_point_id
                count[pid] += 1
        return count

    @property
    def unassigned_count(self):
        count = {pp.point_id: 0 for pp in self.pick_points_list}
        for order in self.orders_unassigned:
            for item in order.unpicked_items:
                if item is None:
                    continue
                pid = item.pick_point_id
                count[pid] += 1
        return count
# ==========================================
# 3. 测试入口
# ==========================================

if __name__ == "__main__":
    # 初始化仓库环境
    warehouse = WarehouseEnv()

    # 读取订单数据，orders.pkl文件中
    with open("../data/orders_100.pkl", "rb") as f:
        orders = pickle.load(f)

    env = WarehouseEnv()
    state = env.reset(orders)

    print("仿真开始...")
    step_i = 0

    while not env.done:
        picker_act = None
        robot_act = None

        # --- 模拟 任务分配 Agent ---
        avail_pickers = env.idle_pickers
        avail_pps = env.idle_pick_points
        if avail_pickers and avail_pps:
            # 策略: 随机分配
            p = random.choice(avail_pickers)
            pp = random.choice(avail_pps)
            picker_act = (p, pp)
            print(f"[任务分配] Picker {p.picker_id} -> Point {pp.point_id}")

        # --- 模拟 路径规划 Agent ---
        # 1. 在 Depot 等订单的机器人
        idle_robots_depot = env.idle_robots
        if idle_robots_depot and env.orders_unassigned:
            r = idle_robots_depot[0]
            # 假设 Agent 决定让它去订单的第一个商品位置
            # 注意：这里只是模拟，实际 RL 会输出具体的 PickPoint
            # 为了让代码跑通，我们需要预先偷看一眼即将分配的 Order 的内容
            next_order = env.orders_unassigned[0]
            first_item_pid = next_order.items[0].pick_point_id
            target = env.pick_points[first_item_pid]

            robot_act = (r, target)
            print(f"[路径规划] Robot {r.robot_id} (Depot) -> Point {target.point_id}")

        # 2. 在 PickPoint 完成任务，需要去下一个点的机器人
        if robot_act is None:  # 简单起见，每步只处理一个机器人动作
            idle_robots_field = env.robots_needing_planning
            if idle_robots_field:
                r = idle_robots_field[0]
                if r.item_pick_order:
                    # 去下一个商品点
                    next_pid = r.item_pick_order[0].pick_point_id
                    target = env.pick_points[next_pid]
                    print(f"[路径规划] Robot {r.robot_id} -> Point {target.point_id}")
                else:
                    # 去 Depot
                    target = env.depot_object
                    print(f"[路径规划] Robot {r.robot_id} -> Depot")

                robot_act = (r, target)

        # 如果两个 Agent 都没有动作，且环境未结束，可能是因为资源都被占用了
        # 但 time_to_next_decision_point 保证了只有在有决策需求时才返回
        if picker_act is None and robot_act is None:
            # 强制推进防止死循环 (理论上不应进入此分支，除非逻辑有遗漏)
            # 在真实 RL 中，这对应 "No-Op"
            pass

        action = (picker_act, robot_act)
        state, reward, done, _, _ = env.step(action)
        step_i += 1
        print("当前时间：", env.current_time)

    print(f"仿真结束。耗时: {env.current_time:.2f}")