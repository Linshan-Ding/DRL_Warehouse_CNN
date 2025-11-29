import numpy as np
import random
import pickle
import gymnasium as gym
import sys
import pygame
import math


# ==========================================
# 1. 配色与配置 (Scientific Style Colors)
# ==========================================

class ColorPalette:
    # 背景
    BACKGROUND = (250, 250, 250)  # 极浅灰/白
    # 静态设施
    SHELF = (119, 136, 153)  # LightSlateGray (储货位)
    SHELF_BORDER = (47, 79, 79)  # DarkSlateGray
    PICK_POINT = (105, 105, 105)  # DimGray (拣货点位)
    DEPOT = (60, 179, 113)  # MediumSeaGreen (Depot)
    # 动态智能体
    ROBOT_IDLE = (100, 149, 237)  # CornflowerBlue
    ROBOT_BUSY = (25, 25, 112)  # MidnightBlue
    PICKER_IDLE = (255, 160, 122)  # LightSalmon
    PICKER_BUSY = (178, 34, 34)  # Firebrick
    # UI
    TEXT = (0, 0, 0)
    UI_BG = (230, 230, 230)
    PAUSE_OVERLAY = (0, 0, 0, 100)  # 半透明遮罩


class Config:
    def __init__(self):
        self.parameters = self.parameter()

    def parameter(self):
        return {
            "warehouse": {
                "shelf_capacity": 20,
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
            },
            "vis": {
                "screen_width": 1200,
                "screen_height": 800,
                "scale_factor": 20,  # 1 meter = 20 pixels
                "margin_x": 60,
                "margin_y": 80,
                "default_fps": 30,  # 默认仿真速度
                "min_fps": 1,
                "max_fps": 120
            }
        }


# ==========================================
# 2. 基础实体类
# ==========================================

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
        return len(self.robot_queue) > 0 and self.picker is None


# --- 智能体类 ---

class Robot(Config):
    def __init__(self, robot_id, position):
        super().__init__()
        self.param = self.parameters["robot"]
        self.robot_id = robot_id
        self.position = position
        self.speed = self.param["robot_speed"]
        self.state = 'idle'

        self.order = None
        self.item_pick_order = []
        self.pick_point = None

        self.move_to_pick_point_time = float('inf')
        self.pick_point_complete_time = float('inf')
        self.move_to_depot_time = float('inf')

    def assign_order(self, order):
        self.order = order
        self.item_pick_order = list(order.items)

    @property
    def items(self):
        if self.order and self.pick_point:
            return [item for item in self.order.items if item.pick_point_id == self.pick_point.point_id]
        return []


class Picker(Config):
    def __init__(self, picker_id):
        super().__init__()
        self.param = self.parameters["picker"]
        self.picker_id = picker_id
        self.speed = self.param["picker_speed"]
        self.state = 'idle'
        self.position = (0, 0)

        self.pick_point = None
        self.pick_start_time = float('inf')
        self.pick_end_time = float('inf')


# ==========================================
# 3. 仓库环境类 (WarehouseEnv)
# ==========================================

class WarehouseEnv(gym.Env, Config):
    def __init__(self, render_mode=None):
        super().__init__()
        self.wh_param = self.parameters["warehouse"]
        self.vis_param = self.parameters["vis"]

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
        self.orders_completed = []

        self.current_time = 0
        self.done = False

        # 可视化与控制初始化
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None
        self.large_font = None

        # 速度控制状态
        self.target_fps = self.vis_param["default_fps"]
        self.paused = False

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
                    # 调整 bin 坐标用于可视化
                    bin_x = position[0] - self.S_w if side == 'left' else position[0] + self.S_w
                    bin_pos = (bin_x, position[1])

                    self.storage_bins[bin_id] = StorageBin(bin_id, bin_pos, item_id, point_id)
                    item = Item(item_id, bin_id, position, point_id)
                    item.pick_time = self.parameters["item"]["pick_time"]
                    self.items[item_id] = item
                    items_ids.append(item_id)
                    bin_ids.append(bin_id)

                pick_point = PickPoint(point_id, position, items_ids, bin_ids)
                self.pick_points[point_id] = pick_point
                self.pick_points_list.append(pick_point)

    def shortest_path_between_pick_points(self, entity, target):
        p1 = entity.position
        p2 = target.position if hasattr(target, 'position') else target
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def adjust_resources(self):
        self.robots = [Robot(i, self.depot_position) for i in range(self.N_robots)]
        self.pickers = []
        for i in range(self.N_pickers):
            p = Picker(i)
            # 均匀分布初始位置
            if i * 5 < len(self.pick_points_list):
                p.position = self.pick_points_list[i * 5].position
            else:
                p.position = self.depot_position
            self.pickers.append(p)

    def reset(self, orders):
        self.current_time = 0
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
        return self.state_extractor()

    def time_to_next_decision_point(self):
        while not self.done:
            # 决策点判定
            if len(self.idle_pickers) > 0 and len(self.idle_pick_points) > 0:
                return

            robots_needing_path = [r for r in self.robots if
                                   r.state == 'idle' and r.order is not None and len(r.item_pick_order) > 0]
            robots_needing_order = [r for r in self.robots if r.state == 'idle' and r.order is None]

            if len(robots_needing_path) > 0:
                return
            if len(robots_needing_order) > 0 and len(self.orders_unassigned) > 0:
                return

            # 事件列表
            future_events = []
            if self.orders_not_arrived:
                future_events.append(self.orders_not_arrived[0].arrive_time)

            for r in self.robots:
                if r.move_to_pick_point_time > self.current_time:
                    future_events.append(r.move_to_pick_point_time)
                if r.pick_point_complete_time > self.current_time:
                    future_events.append(r.pick_point_complete_time)
                if r.move_to_depot_time > self.current_time:
                    future_events.append(r.move_to_depot_time)

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
                    print("Warning: Simulation stuck (No future events). Ending simulation.")
                    self.done = True
                    return

            next_time = min(valid_events)
            self.current_time = next_time

            # 事件处理
            # A. 订单到达
            while self.orders_not_arrived and self.current_time >= self.orders_not_arrived[0].arrive_time:
                order = self.orders_not_arrived.pop(0)
                self.orders_unassigned.append(order)
                self.orders_uncompleted.append(order)

            # B. 机器人到达
            for r in self.robots:
                if self.current_time == r.move_to_pick_point_time:
                    pp = r.pick_point
                    pp.robot_queue.append(r)
                    r.position = pp.position
                    r.move_to_pick_point_time = float('inf')

            # C. 拣货员到达
            for p in self.pickers:
                if self.current_time == p.pick_start_time:
                    p.pick_start_time = float('inf')

            # D. 拣货完成
            for r in self.robots:
                if self.current_time == r.pick_point_complete_time:
                    r.pick_point_complete_time = float('inf')
                    pp = r.pick_point
                    if r in pp.robot_queue: pp.robot_queue.remove(r)

                    items_picked = [i for i in r.items]
                    for item in items_picked:
                        r.order.picked_items.append(item)
                        if item in r.item_pick_order: r.item_pick_order.remove(item)
                        if item in r.order.unpicked_items: r.order.unpicked_items.remove(item)
                    r.state = 'idle'

            for p in self.pickers:
                if self.current_time == p.pick_end_time:
                    p.pick_end_time = float('inf')
                    p.state = 'idle'
                    if p.pick_point:
                        p.pick_point.picker = None
                        p.pick_point = None

            # E. 回到 Depot
            for r in self.robots:
                if self.current_time == r.move_to_depot_time:
                    r.move_to_depot_time = float('inf')
                    r.state = 'idle'
                    r.position = self.depot_position
                    if r.order in self.orders_uncompleted:
                        self.orders_uncompleted.remove(r.order)
                        self.orders_completed.append(r.order)
                    r.order = None
                    r.pick_point = None

    def step(self, action):
        """
        action = ((picker, pick_point), (robot, target))
        """
        picker_act, robot_act = action

        # 1. 拣货员指派
        if picker_act is not None:
            picker, pick_point = picker_act
            picker.state = 'busy'
            picker.pick_point = pick_point
            pick_point.picker = picker

            dist = self.shortest_path_between_pick_points(picker, pick_point)
            travel_time = dist / picker.speed
            picker.pick_start_time = self.current_time + travel_time
            picker.position = pick_point.position

            cumulative_pick_time = 0
            for robot in pick_point.robot_queue:
                robot_items = [i for i in robot.items]
                job_time = sum(item.pick_time for item in robot_items)
                cumulative_pick_time += job_time
                robot.pick_point_complete_time = picker.pick_start_time + cumulative_pick_time

            picker.pick_end_time = picker.pick_start_time + cumulative_pick_time

        # 2. 机器人指派
        if robot_act is not None:
            robot, target = robot_act

            if robot.order is None and self.orders_unassigned:
                order = self.orders_unassigned.pop(0)
                robot.assign_order(order)

            if robot.order is not None:
                robot.state = 'busy'
                if isinstance(target, Depot):
                    dist = self.shortest_path_between_pick_points(robot, target)
                    robot.move_to_depot_time = self.current_time + dist / robot.speed + self.pack_time
                    robot.pick_point = None
                elif isinstance(target, PickPoint):
                    robot.pick_point = target
                    dist = self.shortest_path_between_pick_points(robot, target)
                    robot.move_to_pick_point_time = self.current_time + dist / robot.speed

        self.time_to_next_decision_point()
        return self.state_extractor(), 0, self.done, False, {}

    def state_extractor(self):
        queue_lengths = np.array([len(pp.robot_queue) for pp in self.pick_points.values()], dtype=np.float32)
        return queue_lengths

    # --- 可视化方法 ---

    def _transform_coord(self, pos):
        """逻辑坐标 -> 屏幕像素坐标"""
        scale = self.vis_param["scale_factor"]
        mx = self.vis_param["margin_x"]
        my = self.vis_param["margin_y"]
        x_px = int(pos[0] * scale) + mx
        y_px = int(pos[1] * scale) + my
        return (x_px, y_px)

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.vis_param["screen_width"], self.vis_param["screen_height"]))
            pygame.display.set_caption("Intelligent Warehouse Simulation (Press SPACE to Pause, Up/Down for Speed)")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 12)
            self.large_font = pygame.font.SysFont("Arial", 18, bold=True)
            self.title_font = pygame.font.SysFont("Arial", 32, bold=True)

        # === 事件处理 (速度控制) ===
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_UP or event.key == pygame.K_w:
                    self.target_fps = min(self.vis_param["max_fps"], self.target_fps + 5)
                elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                    self.target_fps = max(self.vis_param["min_fps"], self.target_fps - 5)

        # === 绘制背景 ===
        self.screen.fill(ColorPalette.BACKGROUND)

        # === 1. 绘制静态设施 ===
        # 储货位
        for bin_id, s_bin in self.storage_bins.items():
            center = self._transform_coord(s_bin.position)
            w = self.wh_param["shelf_width"] * self.vis_param["scale_factor"]
            h = self.wh_param["shelf_length"] * self.vis_param["scale_factor"]
            rect_x = center[0] - w / 2
            rect_y = center[1] - h / 2
            pygame.draw.rect(self.screen, ColorPalette.SHELF, (rect_x, rect_y, w, h))
            pygame.draw.rect(self.screen, ColorPalette.SHELF_BORDER, (rect_x, rect_y, w, h), 1)

        # 拣货点
        for pp in self.pick_points.values():
            center = self._transform_coord(pp.position)
            pygame.draw.circle(self.screen, ColorPalette.PICK_POINT, center, 3)

        # Depot
        depot_center = self._transform_coord(self.depot_position)
        size = 20
        points = [
            (depot_center[0], depot_center[1] - size),
            (depot_center[0] + size, depot_center[1]),
            (depot_center[0], depot_center[1] + size),
            (depot_center[0] - size, depot_center[1])
        ]
        pygame.draw.polygon(self.screen, ColorPalette.DEPOT, points)

        # === 2. 绘制动态智能体 ===
        # 拣货员 (方形)
        for p in self.pickers:
            pos = self._transform_coord(p.position)
            color = ColorPalette.PICKER_BUSY if p.state == 'busy' else ColorPalette.PICKER_IDLE
            size = 16
            rect = (pos[0] - size / 2, pos[1] - size / 2, size, size)
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, (50, 50, 50), rect, 1)  # 边框
            text_surf = self.font.render(f"P{p.picker_id}", True, ColorPalette.TEXT)
            self.screen.blit(text_surf, (pos[0] - size, pos[1] - size * 1.5))

        # 机器人 (圆形)
        for r in self.robots:
            pos = self._transform_coord(r.position)
            color = ColorPalette.ROBOT_BUSY if r.state == 'busy' else ColorPalette.ROBOT_IDLE
            radius = 8
            pygame.draw.circle(self.screen, color, pos, radius)
            pygame.draw.circle(self.screen, (0, 0, 0), pos, radius, 1)
            text_surf = self.font.render(f"R{r.robot_id}", True, ColorPalette.TEXT)
            self.screen.blit(text_surf, (pos[0] + radius, pos[1] - radius * 2))

        # === 3. 绘制 UI 面板 ===
        # 顶部信息栏
        pygame.draw.rect(self.screen, ColorPalette.UI_BG, (0, 0, self.vis_param["screen_width"], 60))
        pygame.draw.line(self.screen, (200, 200, 200), (0, 60), (self.vis_param["screen_width"], 60))

        info_text = f"Time: {self.current_time:.1f} s  |  Orders: {len(self.orders_completed)}/{len(self.orders)}  |  FPS: {self.target_fps}"
        info_surf = self.large_font.render(info_text, True, ColorPalette.TEXT)

        controls_text = "Controls: SPACE=Pause/Resume, UP/W=Speed Up, DOWN/S=Slow Down"
        controls_surf = self.font.render(controls_text, True, (100, 100, 100))

        self.screen.blit(info_surf, (20, 10))
        self.screen.blit(controls_surf, (20, 35))

        # 暂停遮罩
        if self.paused:
            s = pygame.Surface((self.vis_param["screen_width"], self.vis_param["screen_height"]), pygame.SRCALPHA)
            s.fill(ColorPalette.PAUSE_OVERLAY)
            self.screen.blit(s, (0, 0))
            pause_text = self.title_font.render("PAUSED", True, (255, 255, 255))
            text_rect = pause_text.get_rect(
                center=(self.vis_param["screen_width"] / 2, self.vis_param["screen_height"] / 2))
            self.screen.blit(pause_text, text_rect)

        pygame.display.flip()

    def close(self):
        if self.screen:
            pygame.quit()
            self.screen = None

    # --- 辅助属性 ---
    @property
    def idle_robots(self):
        return [r for r in self.robots if r.state == 'idle' and r.order is None]

    @property
    def idle_pickers(self):
        return [p for p in self.pickers if p.state == 'idle']

    @property
    def idle_pick_points(self):
        return [pp for pp in self.pick_points_list if pp.is_idle]

    @property
    def robots_needing_planning(self):
        return [r for r in self.robots if r.state == 'idle' and r.order is not None]


# ==========================================
# 4. 测试入口
# ==========================================

if __name__ == "__main__":
    # 1. 加载数据
    try:
        with open("../data/data/instances/orders_100.pkl", "rb") as f:
            orders = pickle.load(f)
        print(f"已加载 {len(orders)} 个订单")
    except FileNotFoundError:
        print("未找到文件，生成模拟数据...")
        orders = []
        for i in range(20):
            items = []
            for j in range(2):
                pid = f"{random.randint(1, 3)}-{random.randint(1, 5)}"
                items.append(Item(f"i{i}_{j}", "bin", (0, 0), pid))
            orders.append(Order(i, items, i * 2))

    # 开启渲染模式
    env = WarehouseEnv(render_mode="human")
    state = env.reset(orders)

    print("仿真开始...")
    step_count = 0
    clock = pygame.time.Clock()

    while not env.done:
        # 速度控制逻辑
        if env.render_mode == "human":
            # 始终调用 render 以响应事件（如调整速度）
            env.render()
            # 如果暂停，则跳过 step 更新，只维持界面刷新
            if env.paused:
                clock.tick(30)  # 暂停时保持低帧率刷新 UI
                continue

            # 正常运行时的帧率控制
            clock.tick(env.target_fps)

        # === 智能体决策模拟 ===
        picker_act = None
        robot_act = None

        # 任务分配 Agent
        avail_pickers = env.idle_pickers
        avail_pps = env.idle_pick_points
        if avail_pickers and avail_pps:
            p = random.choice(avail_pickers)
            pp = random.choice(avail_pps)
            picker_act = (p, pp)

        # 路径规划 Agent
        idle_robots_depot = env.idle_robots
        if idle_robots_depot and env.orders_unassigned:
            r = idle_robots_depot[0]
            next_order = env.orders_unassigned[0]
            first_item_pid = next_order.items[0].pick_point_id
            target = env.pick_points[first_item_pid]
            robot_act = (r, target)

        if robot_act is None:
            idle_robots_field = env.robots_needing_planning
            if idle_robots_field:
                r = idle_robots_field[0]
                if r.item_pick_order:
                    next_pid = r.item_pick_order[0].pick_point_id
                    target = env.pick_points[next_pid]
                else:
                    target = env.depot_object
                robot_act = (r, target)

        # 执行动作
        action = (picker_act, robot_act)
        state, reward, done, _, _ = env.step(action)
        step_count += 1

    print(f"仿真结束。耗时: {env.current_time:.2f}")
    # 保持结束画面
    while True:
        env.render()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                sys.exit()