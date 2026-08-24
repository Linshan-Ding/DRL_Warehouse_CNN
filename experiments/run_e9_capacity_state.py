"""E9 容量 x 状态通道联合实验 —— 加固对 Reviewer #1 载运容量意见的回应。

动机来自 E5 和 E7 两个结果的对照:
    E5 说明 C=1 下给网络补上 per-robot 通道没有显著收益（p=0.317）——
        聚合状态 + 掩码已经充分;
    E7 说明 C>1 下 SAPPO 没吃到批量的收益（C=3 时 1912.0，而 MQ-MinRQ 达 1485.8），
        且六条方法里只有队列均衡型的 MQ-MinRQ 从批量中获益。

假说: 批量运载让每台机器人的剩余任务变得更长、更异质，"哪台机器人还差哪些点"
这类个体信息在 C>1 下才真正变得重要。本实验把 E5 的 plus_agent 通道叠加到
E7 的 C=2/3 配置上验证这一点:

    结果变好  -> 完整的故事: 聚合状态在 C=1 下充分（E5），批量场景需要个体通道（E9），
                 两条意见的回应互相印证;
    结果没变  -> 如实报告，批量感知的策略设计列为 future work。

训练轮数默认 3000（比其它实验多 1000）: E7 的训练曲线显示 C=3 在 2000 轮时仍在缓降，
多给一点预算排除"欠训练"这个混淆因素。

产出: result/e9_c2plus_run*/、result/e9_c3plus_run*/ 下的 eval_results.csv
      （robot_capacity 与 state_channels 两列都有值，可与 E0/E5/E7 直接拼表）
随后: 右键运行 run_stats_and_plots.py，把 PATTERN 设为 "e[579]_*" 三组一起看
耗时: 两个配置各一次训练，3000 轮约 5-7 小时/个（GPU，按你机器 160-200 决策/秒估）。
"""
import _bootstrap  # noqa: F401  必须最先导入

from _runner import run_experiment

# ==================== 配置区（改完右键 Run） ====================
CONFIGS = [
    ("e9_c2plus", ["configs/exp/e7_capacity_2.yaml", "configs/exp/e5_state_plus_agent.yaml"]),
    ("e9_c3plus", ["configs/exp/e7_capacity_3.yaml", "configs/exp/e5_state_plus_agent.yaml"]),
]
RUNS = 1
EPISODES = 3000                   # C>1 收敛更慢，比默认的 2000 多给 1000 轮
METHODS = ["SAPPO", "MQ-MinRQ"]   # 对照只留批量下最强的规则，省评测时间
TIERS = ["main"]
# ==============================================================


def main(configs=CONFIGS, runs=RUNS, episodes=EPISODES, methods=METHODS, tiers=TIERS):
    run_dirs = []
    for name, overlays in configs:
        run_dirs += run_experiment(name=name, overlays=overlays, runs=runs,
                                   episodes=episodes, methods=methods, tiers=tiers)
    return run_dirs


if __name__ == "__main__":
    main()
