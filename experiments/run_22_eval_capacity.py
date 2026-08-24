"""载运容量敏感性 —— 回应 R1.5（C=1 行直接取自 result/main）。

同一批全局策略零样本评 C∈{2,3}（训练时按 train.yaml 的扰动权重见过这些场景）。
前置: run_10..run_14（至少 SAPPO）与 run_20。
产出: result/capacity/c<C>/eval_results.csv
耗时: 每档约 20-40 分钟。
"""
import _bootstrap  # noqa: F401

from _runner import ALL_METHODS, LAMBDA40_CASES, eval_methods, summarise

# ==================== 配置区（改完右键 Run） ====================
CAPACITIES = [2, 3]
METHODS = list(ALL_METHODS)
SAMPLES = 3
CASES = list(LAMBDA40_CASES)
# ==============================================================


def main(capacities=CAPACITIES, methods=METHODS, samples=SAMPLES, cases=CASES):
    outs = [eval_methods(methods, f"result/capacity/c{c}", cases, samples,
                         env_overrides={"robot_capacity": c})
            for c in capacities]
    summarise(outs)
    return outs


if __name__ == "__main__":
    main()
