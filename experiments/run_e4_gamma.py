"""E4 折扣因子消融 —— 回应 Reviewer #2 第 3 条。

审稿人指出的问题成立: 奖励 r_t = F_{t-1} - F_t 只有在 gamma = 1 时才严格
telescoping 成 -F_final。gamma < 1 时按 Abel 分部求和有

    sum_t gamma^{t-1} r_t = F_0 - gamma^{T-1} F_T - (1-gamma) sum_t gamma^{t-1} F_t

多出来的那个 O(1-gamma) 项额外惩罚"轨迹上持续偏高的运行均值"，方向与目标一致，
所以目标对齐但不恒等。这个实验测的就是这个差别在实践中到底有多大。

三个 gamma 各跑 RUNS 次（本项目不固定随机种子，重复训练即为独立重复）。
如果三者最终 F 的差异落在噪声内，这条意见就被证据关掉了；
如果 gamma = 1.0 明显更好，那是个可以写进正文的正面发现。

产出: result/e4_gamma0.95_run*/、e4_gamma0.99_run*/、e4_gamma1.00_run*/ 下的
      eval_results.csv（含 gamma 一列）
随后: 右键运行 run_stats_and_plots.py，把 PATTERN 设为 "e4_*"、
      SENSITIVITY_COLUMN 设为 "gamma"
耗时: 3 x RUNS 次训练，默认 9 次约 3-4 天（CPU）。算力紧张就把 RUNS 降到 2。
"""
import _bootstrap  # noqa: F401  必须最先导入

from _runner import run_experiment

# ==================== 配置区（改完右键 Run） ====================
CONFIGS = [
    ("e4_gamma0.95", "configs/exp/e4_gamma_0.95.yaml"),
    ("e4_gamma0.99", "configs/exp/e4_gamma_0.99.yaml"),
    ("e4_gamma1.00", "configs/exp/e4_gamma_1.00.yaml"),
]
RUNS = 3                          # 每个 gamma 的独立重复次数，消融建议 >= 3
EPISODES = None
METHODS = ["SAPPO"]               # 消融只比 SAPPO 自己
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
