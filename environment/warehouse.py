"""
智能仓库人机协同拣选系统仿真环境
1、同一拣货位的不同商品的拣选时间需要叠加
2、机器人移动到depot_position后，进行打包操作，打包时间为定值
"""

import numpy as np
import random
import pickle
from environment.class_config import Config
from environment.class_object import Robot, Picker, PickPoint, StorageBin, Item, Depot


# -------------------------仓库环境类---------------------------
# 包括机器人、拣货员、拣货位、储货位和商品
# 步进函数step()实现仓库环境的仿真
# 动作为每间隔24个小时调整每个区域的拣货员和仓库中总的机器人的数量
class WarehouseEnv(Config):
    def __init__(self, N_l, N_w, S_l, S_w, S_b, S_d, S_a, N_robots, N_pickers, depot_position):
        # 仓库环境参数
        self.N_l = N_l  # 单个货架中储货位的数量
        self.N_w = N_w  # 巷道的数量
        self.S_l = S_l  # 储货位的长度
        self.S_w = S_w  # 储货位的宽度
        self.S_b = S_b  # 底部通道的宽度
        self.S_d = S_d  # 仓库的出入口处的宽度
        self.S_a = S_a  # 巷道的宽度
        self.depot_position = depot_position  # 机器人的起始位置
        self.pack_time = 10  # 机器人拣完订单后的打包时间
        self.total_orders = None  # 仿真总时间
        self.N_robots = N_robots  # 仓库中机器人的数量
        self.N_pickers = N_pickers  # 仓库中拣货员的数量

        # 仓库固定属性
        self.pick_points = {}  # 拣货位字典
        self.storage_bins = {}  # 储货位字典
        self.items = {}  # 商品字典
        self.pick_points_list = []  # 拣货位列表字典
        self.depot_object = Depot(depot_position)  # 仓库起始点对象
        # 构建仓库图
        self.create_warehouse_graph()

        # 为仓库添加机器人数量和拣货员数量
        self.add_pickers = self.N_pickers  # 拣货员数量
        self.add_robots = self.N_robots  # 机器人数量

        # 仓库强化学习环境属性
        self.state = None  # 当前状态
        self.action = None  # 当前动作
        self.next_state = None  # 下一个状态
        self.reward = None  # 当前奖励
        self.total_reward = 0  # 累计奖励
        self.done = False  # 是否结束标志
        self.current_time = 0  # 当前时间

        # 仓库中的拣货员对象信息
        self.pickers = []  # 拣货员列表
        self.pickers_list = []  # 拣货员列表字典
        # 仓库中的机器人对象信息
        self.robots = []  # 机器人列表
        self.robots_at_depot = []  # depot_position位置的机器人列表
        self.robots_assigned = []  # 已分配订单的机器人列表
        # 仓库中的订单对象信息
        self.orders = []  # 整个仿真过程所有订单对象列表
        self.orders_not_arrived = []  # 未到达的订单对象列表
        self.orders_unassigned = []  # 已到达但未分配机器人的订单对象列表
        self.orders_uncompleted = []  # 已到达未拣选完成的订单对象列表

    def create_warehouse_graph(self):
        # 创建仓库图, 包括货架、巷道、储货位和商品
        for nw in range(1, self.N_w + 1):
            for nl in range(1, self.N_l + 1):
                x = self.S_d + (2 * nw - 1) * self.S_w + (2 * nw - 1) / 2 * self.S_a
                y = self.S_b + (2 * nl - 1) / 2 * self.S_l
                # 计算拣货位的位置
                position = (x, y)

                # 创建该拣货位左侧储货位对象
                bin_id_left = f"{nw}-{nl}-left"
                storage_bin = StorageBin(bin_id_left, position,None, None)
                self.storage_bins[bin_id_left] = storage_bin
                # 创建该储货位存储的商品对象
                item_id_left = f"{nw}-{nl}-left-item"
                item = Item(item_id_left, bin_id_left, position, None)
                self.items[item_id_left] = item
                # 将商品放入储货位
                storage_bin.item_id = item_id_left

                # 创建该拣货位右侧储货位对象
                bin_id_right = f"{nw}-{nl}-right"
                storage_bin = StorageBin(bin_id_right, position, None, None)
                self.storage_bins[bin_id_right] = storage_bin
                # 创建该储货位存储的商品对象
                item_id_right = f"{nw}-{nl}-right-item"
                item = Item(item_id_right, bin_id_right, position, None)
                self.items[item_id_right] = item
                # 将商品放入储货位
                storage_bin.item_id = item_id_right

                # 创建拣货位对象
                point_id = f"{nw}-{nl}"
                pick_point = PickPoint(point_id, position, [item_id_left, item_id_right], [bin_id_left, bin_id_right])
                self.pick_points[point_id] = pick_point  # 将拣货位加入到拣货位字典中
                self.pick_points_list.append(pick_point)  # 将拣货位加入到对应区域的拣货位列表中

                # 将拣货位和储货位+商品关联
                self.storage_bins[bin_id_left].pick_point_id = point_id
                self.storage_bins[bin_id_right].pick_point_id = point_id
                self.items[item_id_left].pick_point_id = point_id
                self.items[item_id_right].pick_point_id = point_id

    # 两个拣货位之间的最短路径长度（若不在一个巷道，则需要从上部或下部绕过储货位）
    def shortest_path_between_pick_points(self, point1, point2):
        x1, y1 = point1.position
        x2, y2 = point2.position
        # 如果两个拣货位在同一巷道，则返回两个拣货位之间的直线路径长度
        if x1 == x2:
            return abs(y1 - y2)
        # 计算从上部绕过和从下部绕过的路径，选择最短路径，并返回路径长度
        else:
            path1 = abs(y1 - self.S_b) + abs(y2 - self.S_b) + abs(x1 - x2)
            path2 = abs(y1 - (self.S_b + self.N_l * self.S_l)) + abs(y2 - (self.S_b + self.N_l * self.S_l)) + abs(x1 - x2)
            return min(path1, path2)

    def adjust_robots_and_pickers(self, n_robots, n_pickers):
        """为仓库中添加机器人和拣货员"""
        # 实例化拣货员对象并添加到仓库中
        for i in range(n_pickers):
            picker = Picker(picker_id=i+1)  # 实例化拣货员对象
            picker.pick_points = self.pick_points_list  # 拣货员负责的拣货位列表
            picker.position = picker.initial_position  # 根据负责的拣货位列表中的拣货位的坐标计算拣货员的初始位置
            self.pickers_list.append(picker)  # 将拣货员加入到拣货员列表中
            self.pickers.append(picker)  # 将拣货员加入到拣货员列表中

        # 实例化机器人对象并添加到仓库中
        for i in range(n_robots):
            robot = Robot(position=self.depot_position)
            self.robots.append(robot)  # 将机器人加入到机器人列表中
            self.robots_at_depot.append(robot)  # 将机器人加入到depot_position位置的机器人列表中

    def reset(self, orders):
        """重置仓库环境"""
        # 重置仓库中的机器人和拣货员对象信息
        self.robots = []  # 机器人列表
        self.pickers = []  # 拣货员列表
        self.pickers_list = []  # 每个区域的拣货员列表字典
        # 重置仓库强化学习环境属性
        self.state = None  # 当前状态
        self.action = None  # 当前动作
        self.next_state = None  # 下一个状态
        self.reward = None  # 当前奖励
        self.total_reward = 0  # 累计奖励
        self.done = False  # 是否结束标志
        # 重置仓库仿真环境时钟和订单对象属性
        self.current_time = 0  # 当前时间
        self.orders = orders  # 整个仿真过程所有订单对象列表
        self.orders_not_arrived = orders  # 未到达的订单对象列表
        self.orders_unassigned = []  # 已到达未分配机器人的订单对象列表
        self.orders_uncompleted = []  # 已到达未拣选完成的订单对象列表
        self.robots_at_depot = []  # depot_position位置的机器人列表
        # 为仓库添加机器人数量和拣货员数量
        self.add_pickers = self.N_pickers  # 拣货员数量
        self.add_robots = self.N_robots  # 机器人数量
        # 为仓库添加机器人和拣货员
        self.adjust_robots_and_pickers(self.add_robots, self.add_pickers)
        # 时间点移动到第一个决策点
        self.time_to_next_decision_point()  # 移动到下一个决策点
        # 提取初始状态
        self.state = self.state_extractor()
        return self.state

    def time_to_next_decision_point(self):
        """
        当前时间移动到决策点：仓库总存在空闲拣货员和待分配拣货员的拣货位
        """
        while len(self.idle_pickers) == 0 or len(self.idle_pick_points) == 0:
            """选择下一个离散点时刻"""
            # 判断是否移动时钟到下一个离散点:若当前离散点[新订单到达（为机器人分配订单），拣货员拣货完成，机器人移动到拣货点，机器人拣完商品，机器人移动到depot_position]
            # 中的最小值大于当前时间，则移动时钟到下一个离散点
            if len(self.orders_not_arrived) > 0:
                new_order_arrival_time = self.orders_not_arrived[0].arrive_time  # 新订单到达时刻
            else:
                new_order_arrival_time = 0  # 如果没有新订单到达，则设置0
            pickers_pick_complete_time = [picker.pick_end_time for picker in self.pickers]  # 所有拣货员拣货完成时刻
            pickers_move_to_pick_point_time = [picker.pick_start_time for picker in self.pickers]  # 所有拣货员移动到拣货位开始拣货时刻
            robots_pick_complete_time = [robot.pick_point_complete_time for robot in self.robots]  # 所有机器人在拣货点拣完商品时刻
            robots_move_to_pick_point_time = [robot.move_to_pick_point_time for robot in self.robots]  # 所有机器人移动到拣货点时刻
            robots_move_to_depot_time = [robot.move_to_depot_time for robot in self.robots]  # 所有机器人移动到depot_position时刻

            # 所有离散时刻
            discrete_times = ([new_order_arrival_time] + pickers_pick_complete_time + pickers_move_to_pick_point_time +
                              robots_pick_complete_time + robots_move_to_pick_point_time + robots_move_to_depot_time)
            # 下一个离散点时刻
            next_discrete_time = min([time for time in discrete_times if time > self.current_time])
            # 更新当前时间
            self.current_time = next_discrete_time

            """更新新离散点各对象的属性"""
            # 1、若当前时间等于新订单到达时刻，则将新订单加入到待分配订单列表中
            while self.current_time == new_order_arrival_time:
                order = self.orders_not_arrived.pop(0)
                self.orders_unassigned.append(order)
                self.orders_uncompleted.append(order)
                if len(self.orders_not_arrived) == 0:
                    break
                new_order_arrival_time = self.orders_not_arrived[0].arrive_time  # 新订单到达时刻
                # print("添加新到达订单", order.order_id)
                # 若存在待分配订单和空闲机器人，则为机器人分配订单
                self.assign_order_to_robot()

            # 2、若当前时间等于机器人移动到拣货点时刻，则更新机器人所在拣货位的机器人队列
            for robot in self.robots:
                if self.current_time == robot.move_to_pick_point_time:
                    # print("机器人移动到新拣货位")
                    # 更新机器人所在拣货位
                    pick_point = robot.next_pick_point(self.pick_points)  # 机器人当前拣货位
                    pick_point.robot_queue.append(robot)  # 更新拣货位的机器人队列
                    robot.position = pick_point.position  # 更新机器人的位置
                    robot.pick_point = pick_point  # 更新机器人所在拣货位
                    # 若当前时间该拣货点有拣货员，则更新拣货员的拣货完成时间和该机器人的拣完商品时间
                    if pick_point.picker is not None:
                        # 更新拣货员拣货完工时间和机器人在该拣货位拣完商品时间
                        for item in robot.items:
                            pick_point.picker.pick_end_time += item.pick_time  # 更新拣货员拣货完成时间
                        robot.pick_point_complete_time = pick_point.picker.pick_end_time  # 机器人在该拣货位的拣货完成时间

            # 3、若当前时间等于拣货员拣货完成时刻，则更新拣货员的状态，重置拣货位的拣货员对象; 若等于拣货员移动到拣货位开始拣货时刻，则更新拣货员的状态
            for picker in self.pickers:
                if self.current_time == picker.pick_start_time:
                    # print("拣货员移动到拣货位开始拣货")
                    picker.state = 'busy'
                    pick_point = picker.pick_point  # 拣货员所在拣货位
                    pick_point.picker = picker  # 更新拣货位的拣货员对象
                    picker.pick_point = pick_point  # 更新拣货员的拣货位对象
                    picker.position = pick_point.position  # 更新拣货员的位置

                if self.current_time == picker.pick_end_time:
                    # print("拣货员拣货完成")
                    picker.state = 'idle'  # 更新拣货员的状态
                    pick_point = picker.pick_point  # 拣货员所在拣货位
                    pick_point.picker = None  # 重置拣货位的拣货员对象
                    picker.pick_point = None  # 重置拣货员的拣货位对象

            # 4、若当前时间等于机器人在拣货位拣完商品时刻：更新拣货位的机器人队列，更新机器人所属订单拣选的商品列表，更新机器人所属订单未拣选完成的商品列表
            for robot in self.robots:
                """更新机器人属性和所属订单属性"""
                if self.current_time == robot.pick_point_complete_time:
                    # print("机器人拣完当前拣货位商品")
                    # 机器人所在拣货位
                    pick_point = robot.pick_point
                    # 从拣货位的机器人队列中移除该机器人
                    pick_point.robot_queue.remove(robot)
                    # 更新机器人所属订单拣选完成的商品列表, 机器人所属订单未拣选完成的商品列表, 机器人所属订单拣选的商品列表
                    for item in robot.items:
                        robot.order.picked_items.append(item)   # 更新机器人所属订单拣选完成的商品列表
                        robot.order.unpicked_items.remove(item)  # 更新机器人所属订单未拣选完成的商品列表
                        robot.item_pick_order.remove(item)  # 更新机器人所属订单拣选的商品列表
                    # 若机器人所有商品未拣货完成, 则更新机器人移动到拣货点的时间，更新机器人待拣货或正在拣货的商品
                    if len(robot.item_pick_order) > 0:
                        # print("机器人未拣货完成")
                        # 更新机器人下一个拣货位
                        next_pick_point = robot.next_pick_point(self.pick_points)
                        # 计算机器人移动到拣货点的时间
                        shortest_path_length = self.shortest_path_between_pick_points(robot, next_pick_point)
                        move_time = shortest_path_length / robot.speed
                        robot.move_to_pick_point_time = self.current_time + move_time
                    # 若机器人所有商品拣货完成，则更新机器人移动到depot_position的时间
                    else:
                        # print("机器人拣货完成")
                        # 更新机器人移动到depot_position的时间
                        shortest_path_length = self.shortest_path_between_pick_points(robot, self.depot_object)
                        move_time = shortest_path_length / robot.speed
                        robot.move_to_depot_time = self.current_time + move_time + self.pack_time  # 机器人移动到depot_position的时间

            # 5、若当前时间等于机器人移动到depot_position时刻，则更新机器人的状态，重置机器人的订单对象
            for robot in self.robots:
                # 若机器人移动到depot_position时刻
                if self.current_time == robot.move_to_depot_time:
                    robot.state = 'idle'  # 更新机器人的状态
                    self.orders_uncompleted.remove(robot.order)  # 从未拣选完成的订单列表中移除该订单
                    robot.order = None  # 重置机器人的订单对象
                    robot.position = self.depot_position  # 更新机器人的位置
                    # print("机器人移动到depot_position")
                    # 若存在待分配订单和空闲机器人，则为机器人分配订单
                    self.assign_order_to_robot()

            """判断是否结束仿真"""
            if len(self.orders_unassigned) == 0 and len(self.orders_uncompleted) == 0 and len(self.orders_not_arrived) == 0:
                self.done = True
                print("仿真结束时刻", self.current_time)
                break

    def step(self, action):
        """
        仓库环境的仿真步进函数：每个决策点执行一次step()函数
        决策点：仓库总存在空闲拣货员和待分配拣货员的拣货位
        离散点：新订单到达时刻、拣货员空闲时刻、机器人移动到拣货点时刻，机器人空闲时刻
        action: “空闲拣货员和待分配拣货员的拣货位” 动作对
        """
        """更具动作（选择的空闲拣货员和待分配拣货员的拣货位）更新仓库环境"""
        pick_point = action[1]  # 待分配拣货位
        picker = action[0]  # 空闲拣货员
        # 为拣货员分配拣货位
        picker.pick_point = pick_point
        # 为拣货位分配拣货员
        pick_point.picker = picker
        # 计算拣货员移动到拣货位的最短路径长度
        shortest_path_length = self.shortest_path_between_pick_points(picker, pick_point)
        # 计算拣货员移动到拣货位的时间
        move_time = shortest_path_length / picker.speed
        # 更新拣货员的状态
        picker.state = 'busy'
        # 更新拣货员的工作时间
        picker.working_time += move_time
        # 拣货员在该拣货位拣货开始时间
        picker.pick_start_time = self.current_time + move_time  # 拣货员在该拣货位拣货开始时间
        picker.pick_end_time = picker.pick_start_time  # 初始化拣货员在该拣货位拣货结束时间
        # 更新该拣货位的机器人对象队列中的所有机器人该商品的拣货完成时间，按机器人队列顺序完成拣货
        for robot in pick_point.robot_queue:
            # 更新拣货员拣货完工时间和机器人在该拣货位拣完商品时间
            for item in robot.items:
                picker.pick_end_time += item.pick_time  # 更新拣货员拣货完成时间
            robot.pick_point_complete_time = picker.pick_end_time  # 机器人在该拣货位的拣货完成时间
        # 更新拣货员的位置
        picker.position = pick_point.position

        # 移动到下一个决策点
        self.time_to_next_decision_point()

        # 提取当前状态，计算回报值
        self.state = self.state_extractor()  # 提取当前状态

        return self.state, self.reward, self.done

    def assign_order_to_robot(self):
        """若存在待分配订单和空闲机器人，则为机器人分配订单"""
        while len(self.orders_unassigned) > 0 and len(self.idle_robots) > 0:
            # 选择一个空闲机器人
            robot = self.idle_robots.pop(0)
            # 选择一个待分配订单
            order = self.orders_unassigned.pop(0)
            # 为机器人分配订单
            robot.assign_order(order)
            # 机器人下一个拣货位
            next_pick_point = robot.next_pick_point(self.pick_points)
            # 计算机器人移动到订单中首个商品的最短路径
            shortest_path_length = self.shortest_path_between_pick_points(robot, next_pick_point)
            # 计算机器人移动时间
            move_time = shortest_path_length / robot.speed
            # 更新机器人的状态
            robot.state = 'busy'
            # 更新机器人的工作时间
            robot.working_time += move_time
            # 机器人移动到拣货位时间
            robot.move_to_pick_point_time = self.current_time + move_time

    def state_extractor(self):
        """提取仓库的当前状态"""
        # 每个拣货位的排队机器人数量
        robot_queue_list = [len(point.robot_queue) for point in self.pick_points.values()]
        # 根据仓库拣货位的布局转为把robot_queue_list转为二维numpy数组
        robot_queue_list = np.array(robot_queue_list).reshape((self.N_w, self.N_l))
        # 每个拣货位是否有拣货员拣货员，有的话为1，没有的话为0
        picker_list = [0 if point.picker is None else 1 for point in self.pick_points.values()]
        # 根据仓库拣货位的布局转为把picker_list转为二维numpy数组
        picker_list = np.array(picker_list).reshape((self.N_w, self.N_l))
        # 每个拣货位待拣货商品数量
        unpicked_items_list = self.pick_point_unpicked_items
        # 根据仓库拣货位的布局转为把unpicked_items_list转为二维numpy数组
        unpicked_items_list = np.array(unpicked_items_list).reshape((self.N_w, self.N_l))
        # depot_position位置的机器人数量，即空闲机器人数量
        n_robots_at_depot = len([robot for robot in self.robots if robot.state == 'idle'])
        # 机器人总数
        n_robots = len(self.robots)
        # 拣货员总数
        n_pickers = len(self.pickers)
        # 所有状态特征组合成state字典
        state = {'robot_queue_list': robot_queue_list, 'picker_list': picker_list, 'unpicked_items_list': unpicked_items_list,
                 'n_robots_at_depot': n_robots_at_depot, 'n_robots': n_robots, 'n_pickers': n_pickers}
        return state

    def compute_reward(self):
        """计算当前奖励"""
        pass

    # 当前离散点空闲机器人列表
    @ property
    def idle_robots(self):
        return [robot for robot in self.robots if robot.state == 'idle']

    # 当前离散点每个区域的空闲拣货员列表
    @ property
    def idle_pickers(self):
        # 每个区域的空闲拣货员列表字典
        idle_pickers = [picker for picker in self.pickers_list if picker.state == 'idle']
        return idle_pickers

    # 当前离散点每个区域待分配拣货员的拣货位列表
    @ property
    def idle_pick_points(self):
        # 每个区域待分配拣货员的拣货位列表字典
        idle_pick_points = [point for point in self.pick_points_list if point.is_idle]
        return idle_pick_points

    # 当前离散点每个拣货位未拣货商品数量（基于未拣货完成订单中的未拣货完成商品计算对应拣货位的未拣货商品数量）
    @ property
    def pick_point_unpicked_items(self):
        # 重置拣货位中的未拣货商品列表
        for point in self.pick_points.values():
            point.unpicked_items = []
        # 计算拣货位中的未拣货商品数量
        for order in self.orders_uncompleted:
            for item in order.unpicked_items:
                pick_point_id = item.pick_point_id
                self.pick_points[pick_point_id].unpicked_items.append(item)
        # 每个拣货位未拣货商品数量列表
        unpicked_items_list = [len(point.unpicked_items) for point in self.pick_points.values()]
        return unpicked_items_list


if __name__ == "__main__":
    # 初始化仓库环境
    N_l = 10  # 单个货架中储货位的数量
    N_w = 6  # 巷道的数量
    S_l = 1  # 储货位的长度
    S_w = 1  # 储货位的宽度
    S_b = 2  # 底部通道的宽度
    S_d = 2  # 仓库的出入口处的宽度
    S_a = 2  # 巷道的宽度
    depot_position = (0, 0)  # 机器人的起始位置
    # 机器人数量
    N_robots = 10
    # 拣货员数量
    N_pickers = 5

    # 初始化仓库环境
    warehouse = WarehouseEnv(N_l, N_w, S_l, S_w, S_b, S_d, S_a, N_robots, N_pickers, depot_position)

    # # 基于仓库中的商品创建一个订单对象，每个订单包含多个商品，订单到达时间服从泊松分布
    # total_orders = 100  # 订单总数
    # # 订单到达泊松分布参数
    # poisson_parameter = 60  # 泊松分布参数, 60秒一个订单到达
    # # 生成一个月内的订单数据，并保存到orders.pkl和orders.csv文件中
    # generate_orders = GenerateData(warehouse, total_orders, poisson_parameter)  # 生成订单数据对象
    # generate_orders.generate_orders()  # 生成一个月内的订单数据

    # 读取订单数据，orders.pkl文件中
    with open("../data/orders.pkl", "rb") as f:
        orders = pickle.load(f)

    # 基于订单数据和仓库环境数据，实现仓库环境的仿真
    warehouse.reset(orders)  # 重置仓库环境
    # 最后一个订单到达时间
    last_order_arrival_time = orders[-1].arrive_time

    # 仿真程序
    while not warehouse.done:
        # 随机选择一个待分配拣货位
        pick_point = random.choice(warehouse.idle_pick_points)
        # 选择距离该拣货位最近的空闲拣货员
        picker = min(warehouse.idle_pickers, key=lambda picker: warehouse.shortest_path_between_pick_points(picker, pick_point))
        action = [picker, pick_point]  # 动作对
        # 仓库环境的仿真步进函数
        state, reward, done = warehouse.step(action)
        # 输出当前状态
        print(f"Current state: {warehouse.state}")
        print(f"Current time: {warehouse.current_time}")  # 当前时间
        print("-----------------------------------------------------------------------------------------------------------------------------------------------")