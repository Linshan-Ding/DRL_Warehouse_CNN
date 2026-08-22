"""冒烟测试 —— 用极小预算把整条链路跑一遍，确认环境装对了。

这不是实验: 训练轮数太少，学不到任何东西。它只回答一个问题——
"data -> train -> eval -> CSV 这条链路在我这台机器上通不通"。

产出: result/smoke_run1/{log.csv, eval_results.csv, training_cost.csv, checkpoint_best.pt}
耗时: 约 2-3 分钟。
"""
import _bootstrap  # noqa: F401  必须最先导入

from _runner import run_experiment

# ==================== 配置区（改完右键 Run） ====================
EPISODES = 3                        # 训练轮数，冒烟用 3 轮足够
METHODS = ["SAPPO", "MQ-ND"]        # 参与评测的方法
TIERS = ["main"]                    # 评测算例档位
# ==============================================================


def main(episodes=EPISODES, methods=METHODS, tiers=TIERS):
    return run_experiment(name="smoke",
                          overlays=["configs/exp/smoke.yaml"],
                          runs=1, episodes=episodes,
                          methods=methods, tiers=tiers)


if __name__ == "__main__":
    main()
