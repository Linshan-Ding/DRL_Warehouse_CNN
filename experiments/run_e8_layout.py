"""E8 中部横通道布局变体 —— 回应 Reviewer #1 关于"货架之间的移动"。

这条意见其实是误读: 论文 Eq.(2) 早就是标准的双横通道最短路，同巷道取纵向距离，
跨巷道从底部或顶部横通道绕行取短者，"货架之间（跨巷道）的移动"是建模了的，
只是不能穿过货架。误读的责任在图和文字都没画出跨巷道路径。

真正没建模的是中部横向通道。这个实验把它加进距离函数，证明结论不依赖
"只有两条横通道"这个布局假设——而且换布局只需要换距离函数，MDP、状态、网络都不动。

参考数据: 规则 MQ-ND、lam40、K=3/R=6 下，加中部横通道后 F 从 1892.17 降到 1216.66，
说明布局会显著影响绝对值，但方法本身不受影响。

耗时: 一次训练，约 9-12 小时（CPU）。
等价的终端命令:
    python train.py --config configs/exp/e8_layout_mid_aisle.yaml --run-name e8_mid_run1
    python eval.py --config configs/exp/e8_layout_mid_aisle.yaml \
                   --ckpt result/e8_mid_run1/checkpoint_best.pt \
                   --methods SAPPO MQ-ND MQ-MinRQ MQ-MI MI-MinRQ MI-MI \
                   --tiers main --run-id 1 --run-name e8_mid_run1
"""
import _bootstrap  # noqa: F401  必须最先导入

from _runner import DEFAULT_METHODS, run_experiment

# ==================== 配置区（改完右键 Run） ====================
RUNS = 1
EPISODES = None
METHODS = list(DEFAULT_METHODS)
TIERS = ["main"]
# ==============================================================


def main(runs=RUNS, episodes=EPISODES, methods=METHODS, tiers=TIERS):
    return run_experiment(name="e8_mid",
                          overlays=["configs/exp/e8_layout_mid_aisle.yaml"],
                          runs=runs, episodes=episodes, methods=methods, tiers=tiers)


if __name__ == "__main__":
    main()
