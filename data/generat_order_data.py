"""
生成仿真订单数据和算例数据
"""
import random
import csv
import pickle
import os
import copy
from environment.class_object import Order
from environment.warehouse import WarehouseEnv


class GenerateData:
    """
    生成仿真订单
    """

    def __init__(self, warehouse, total_orders, poisson_parameter, max_items_per_order=10):
        """
        :param warehouse: 仓库环境对象
        :param total_orders: 需要生成的订单总数
        :param poisson_parameter: 泊松分布参数（用于生成到达间隔时间）
        :param max_items_per_order: 每个订单包含的最大商品数量限制
        """
        self.warehouse = warehouse
        self.total_orders = total_orders
        self.poisson_parameter = poisson_parameter
        self.max_items_per_order = max_items_per_order

        # 预先将商品对象列表化，避免在循环中重复转换，提高性能
        # 注意：这里假设订单引用的是仓库中的商品信息，不需要深拷贝整个库存
        self.all_items = list(self.warehouse.items.values())

        # 确保数据目录存在
        self.data_dir = 'data/instances'
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def generate_orders(self):
        orders = []  # 订单对象列表
        order_id = 0
        arrival_time = 0

        # 仓库总商品数
        total_inventory_size = len(self.all_items)

        # 确定CSV文件路径
        csv_path = os.path.join(self.data_dir, f'orders_{self.poisson_parameter}.csv')
        pkl_path = os.path.join(self.data_dir, f'orders_{self.poisson_parameter}.pkl')

        print(f"Start generating {self.total_orders} orders...")

        with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
            fieldnames = ['order_id', 'arrival_time', 'item_id', 'pick_point_id']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for _ in range(self.total_orders):
                order_id += 1

                # 1. 生成到达时间
                # random.expovariate(lambd): lambd是频率 (1/平均间隔)。
                # 如果 poisson_parameter 代表平均间隔时间，则参数应为 1/poisson_parameter
                interval = int(random.expovariate(1.0 / self.poisson_parameter))
                arrival_time += interval

                # 2. 确定该订单的商品数量 (限制最大数量，符合实际逻辑)
                # 如果仓库商品少于最大限制，则以仓库总数为上限
                current_max = min(self.max_items_per_order, total_inventory_size)
                order_n_items = random.randint(1, current_max)

                # 3. 随机选择商品
                # 使用 random.sample 进行无放回抽样（一个订单不包含重复的商品ID，视具体业务需求而定）
                selected_items = random.sample(self.all_items, order_n_items)

                # 如果需要模拟每个订单的商品是独立的实体（不影响仓库库存对象），可以使用浅拷贝
                # items_for_order = [copy.copy(item) for item in selected_items]
                # 这里暂时直接使用引用

                # 4. 创建订单对象 (假设Order类支持 due_time，通常会设置一个截至时间)
                # 这里为了兼容原始代码，只传参 order_id, items, arrival_time
                order = Order(order_id, selected_items, arrival_time)
                orders.append(order)

                # 5. 写入CSV
                for item in selected_items:
                    writer.writerow({
                        'order_id': order_id,
                        'arrival_time': arrival_time,
                        'item_id': item.item_id,
                        'pick_point_id': item.pick_point_id
                    })

        # 保存订单列表到pickle文件
        with open(pkl_path, "wb") as f:
            pickle.dump(orders, f)

        print(f"Success! Generated {len(orders)} orders.")
        print(f"Saved to: {pkl_path}")


if __name__ == '__main__':
    # 1. 实例化仓库环境
    warehouse = WarehouseEnv()

    # 2. 检查仓库是否有商品，避免报错
    if not warehouse.items:
        print("Error: Warehouse has no items. Please initialize warehouse inventory first.")
    else:
        # 3. 实例化生成器并运行
        # 设置泊松参数为100（即平均间隔100秒一个订单），限制每个订单最多5个商品
        generator = GenerateData(warehouse, total_orders=100, poisson_parameter=100, max_items_per_order=5)
        generator.generate_orders()