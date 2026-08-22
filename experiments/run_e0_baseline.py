"""E0 基线复现 —— 其余所有补充实验的前置门槛。

复现论文的基准算例 C18: 1/lambda = 40、K = 3 个拣货员、R = 6 台 AMR。

通过标准: SAPPO 在 lam40 上报出的 F 应落在已投稿数值的多次运行波动范围内
（论文 Table 5 的 C18: F = 1379.890）。把 RUNS 设为 3 就能看到这个范围。
E0 复现不了的话，先查清原因，别急着跑后面的实验。

产出: result/e0_run*/{log.csv, eval_results.csv, training_cost.csv,
      checkpoint_best.pt, config_snapshot.yaml}
耗时: 单次约 9-12 小时（CPU），GPU 上快很多。想先确认链路，把 EPISODES 设成 20。
"""
import _bootstrap  # noqa: F401  必须最先导入

from _runner import DEFAULT_METHODS, run_experiment

# ==================== 配置区（改完右键 Run） ====================
RUNS = 3                          # 独立重复次数；要看波动范围至少 3 次
EPISODES = None                   # None = 用 configs/algo.yaml 的 2000；先填 20 可快速验证
METHODS = list(DEFAULT_METHODS)   # SAPPO + 五条组合规则
TIERS = ["main"]                  # 评测算例档位
# ==============================================================


def main(runs=RUNS, episodes=EPISODES, methods=METHODS, tiers=TIERS):
    return run_experiment(name="e0", overlays=["configs/exp/e0_baseline.yaml"],
                          runs=runs, episodes=episodes, methods=methods, tiers=tiers)


if __name__ == "__main__":
    main()
