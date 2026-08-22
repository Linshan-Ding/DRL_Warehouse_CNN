"""E6 拣选时间敏感性 —— 回应 Reviewer #1 关于货架第一层以上的意见。

论文的仓库是二维的，没有层高这个维度。完整建模多层货架（层高相关的取件时间、
可达性、举升设备）超出本文范围，因此改为新增假设 (A8): 货架为多层，但每个拣货位的
竖直取件动作被吸收进 tau_pick，即 tau_pick 视为跨层的平均取件时间。

把 tau_pick 从 10 s 提到 15 s、20 s，就是"从更高层取货"的代理实验，用来说明
取件时间上升时 SAPPO 相对基线的优势是否还保持得住。

耗时: 两个配置各一次训练，约 18-24 小时（CPU）。
等价的终端命令（对每个配置）:
    python train.py --config configs/exp/e6_picktime_15.yaml --run-name e6_tau15_run1
    python eval.py --config configs/exp/e6_picktime_15.yaml \
                   --ckpt result/e6_tau15_run1/checkpoint_best.pt \
                   --methods SAPPO MQ-ND MQ-MinRQ MQ-MI MI-MinRQ MI-MI \
                   --tiers main --run-id 1 --run-name e6_tau15_run1
"""
import _bootstrap  # noqa: F401  必须最先导入

from _runner import DEFAULT_METHODS, run_experiment

# ==================== 配置区（改完右键 Run） ====================
CONFIGS = [
    ("e6_tau15", "configs/exp/e6_picktime_15.yaml"),
    ("e6_tau20", "configs/exp/e6_picktime_20.yaml"),
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
