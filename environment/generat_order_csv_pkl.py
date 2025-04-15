"""
生成仿真订单
"""
from class_public import Order
import random
import copy
import csv
import pickle
from datetime import datetime

class GenerateData:
    """
    生成仿真订单
    """
    def __init__(self, warehouse, total_orders, poisson_parameter):
        self.warehouse = warehouse  # 仓库对象
        self.total_orders = total_orders  # 订单总数
        self.poisson_parameter = poisson_parameter  # 泊松分布参数

    def generate_orders(self):
        orders = []  # 订单列表
        order_id = 0  # 订单ID
        arrival_time = 0  # 到达时间
        n_items = len(self.warehouse.items)  # 商品类型数量

        with open('orders.csv', 'w', newline='') as csv_file:
            # 修正 fieldnames 以匹配 writerow 中的键
            fieldnames = ['order_id', 'arrival_time', 'item_id', 'pick_point_id']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for _ in range(self.total_orders):
                order_n_items = random.randint(1, n_items)  # 随机选择订单中商品的数量
                items_list = copy.deepcopy(list(self.warehouse.items.values()))  # 深拷贝商品列表
                if order_n_items > len(items_list):
                    order_n_items = len(items_list)
                items = random.sample(items_list, order_n_items)  # 随机选择商品
                arrival_time += int(random.expovariate(1 / self.poisson_parameter))  # 根据泊松分布生成到达时间
                order_id += 1  # 订单ID自增
                order = Order(order_id, items, arrival_time)  # 创建订单对象
                orders.append(order)  # 将订单添加到订单列表

                # 将订单信息写入CSV文件
                for item in items:
                    writer.writerow({
                        'order_id': order_id,
                        'arrival_time': arrival_time,
                        'item_id': item.item_id,
                        'pick_point_id': item.pick_point_id
                    })

        # 保存订单列表到pickle文件
        with open("orders.pkl", "wb") as f:
            pickle.dump(orders, f)

        print(f"Total number of orders: {len(orders)}")