"""E2 更大规模 —— 回应 Reviewer #1 关于"机器人和人员数量还要更多"的意见。

上一轮修订已经做到 K=6 / R=12（论文 Table 10），审稿人显然觉得还不够。
这里再往上加两档: (K,R) = (8,16) 和 (10,20)。

配合 E3 产出的训练成本数据，可以用"决策时间与训练时长随规模的增长曲线"来回答
可扩展性，而不是只多两行数字。

产出: result/e2_k8r16_run*/、result/e2_k10r20_run*/ 下的 eval_results.csv
      与 training_cost.csv
耗时: 每个配置一次训练；规模越大单步越慢，两个配置约 20-30 小时（CPU）。
"""
import _bootstrap  # noqa: F401  必须最先导入

from _runner import run_experiment

# ==================== 配置区（改完右键 Run） ====================
CONFIGS = [
    ("e2_k8r16", "configs/exp/e2_scale_k8_r16.yaml"),
    ("e2_k10r20", "configs/exp/e2_scale_k10_r20.yaml"),
]
RUNS = 1
EPISODES = None
METHODS = ["SAPPO", "MQ-ND"]      # 大规模档只留一条规则做对照，省时间
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
