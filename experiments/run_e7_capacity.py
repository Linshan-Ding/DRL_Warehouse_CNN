"""E7 机器人单次载运能力 —— 回应 Reviewer #1，本轮唯一的真实建模改动。

已投稿版本的假设 (A1) 把 AMR 锁死为"一次一单"，全文和代码里都没有任何容量参数。
现在引入单次载运订单数上限 C: depot 处一次绑定至多 C 个待分配订单，必访点取并集，
返回时按订单数结算打包时间。C = 1 与原实现逐事件完全等价（run_00_selfcheck 第 3 项
就在断言这件事），所以这是模型的推广而不是替换。

结论方向不可预设。参考数据: 在规则 MQ-ND、lam40、K=3/R=6 下，
C=1 时 F = 1892.17，C=2 时 F = 1948.98，反而变差 3%——C 变大会拉长单台机器人的路径，
若不配套路径优化，F 未必单调改善。这本身是有价值的发现，但写回复信时不能提前
假定"性能随 C 提升"，要等这个实验的 SAPPO 结果出来再定调。

耗时: 两个配置各一次训练，约 18-24 小时（CPU）。
等价的终端命令（对每个配置）:
    python train.py --config configs/exp/e7_capacity_2.yaml --run-name e7_c2_run1
    python eval.py --config configs/exp/e7_capacity_2.yaml \
                   --ckpt result/e7_c2_run1/checkpoint_best.pt \
                   --methods SAPPO MQ-ND MQ-MinRQ MQ-MI MI-MinRQ MI-MI \
                   --tiers main --run-id 1 --run-name e7_c2_run1
"""
import _bootstrap  # noqa: F401  必须最先导入

from _runner import DEFAULT_METHODS, run_experiment

# ==================== 配置区（改完右键 Run） ====================
CONFIGS = [
    ("e7_c2", "configs/exp/e7_capacity_2.yaml"),
    ("e7_c3", "configs/exp/e7_capacity_3.yaml"),
]
RUNS = 1
EPISODES = None
METHODS = list(DEFAULT_METHODS)
TIERS = ["main"]
# ==============================================================


def main(configs=CONFIGS, runs=RUNS, episodes=EPISODES, methods=METHODS, tiers=TIERS):
    run_dirs = []
    for name, overlay in configs:
        run_dirs += run_experiment(name=name, overlays=[overlay], runs=runs,
                                   episodes=episodes, methods=methods, tiers=tiers)
    return run_dirs


if __name__ == "__main__":
    main()
