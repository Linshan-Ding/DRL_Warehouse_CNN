"""训练 SAPPO 的全局策略 —— 唯一产出 models/ 下的一个参数文件。

每个 episode 按参数表随机采样场景（λ/K/R/容量/拣货时间/布局，权重见
configs/train.yaml）并实例化新环境、新订单流；多进程 worker 并行采集整回合，
GPU 上批量更新。训练出的单一策略零样本服务全部 27 算例与各敏感性场景。

与 run_10..run_14 的其它训练互相独立：多台机器可并行；单机建议串行，
或最多双开并把 WORKERS 设为核数的一半。

产出: models/ 下的策略文件、result/train_*/ 下的 log.csv 与 training_cost.csv
耗时: 默认 15000 episodes；按你机器的 sps 折算（8 worker 参考 3-6 小时）。
"""
import _bootstrap  # noqa: F401

from _runner import train_algo

# ==================== 配置区（改完右键 Run） ====================
EPISODES = None   # None = configs/train.yaml 的 15000；填 20 可几分钟验证链路
WORKERS = None    # None = 自动（CPU 核数 - 2）；双开训练时减半
# ==============================================================


def main(episodes=EPISODES, workers=WORKERS):
    return train_algo("SAPPO", episodes=episodes, workers=workers)


if __name__ == "__main__":
    main()
