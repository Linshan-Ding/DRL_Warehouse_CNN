"""规则基线的载运容量扫描 —— 不需要训练，几分钟画出"批量收益曲线"。

E7 只有 C∈{1,2,3} 三个点，而且揭示了一个值得展开的现象: 批量的系统级收益
只有配上队列均衡型路由（MQ-MinRQ）才能兑现，其余规则反而变差。这个脚本把
五条规则在 C∈{1..5} 上全部扫一遍——规则是确定性的，整个扫描只要几分钟——
让"批量收益曲线"有足够的点来画图和下结论。

产出: result/rules_capacity_c{1..5}/eval_results.csv（robot_capacity 列区分）
随后: 右键运行 run_stats_and_plots.py，把 PATTERN 设为 "rules_capacity_*"、
      SENSITIVITY_COLUMN 设为 "robot_capacity"
耗时: 约 10-15 分钟（5 个容量档 x 5 条规则 x 3 个实例，无训练）。
"""
import _bootstrap  # noqa: F401  必须最先导入

from _runner import RULES_ONLY, run_experiment

# ==================== 配置区（改完右键 Run） ====================
CAPACITIES = [1, 2, 3, 4, 5]     # 扫描的载运容量档
METHODS = list(RULES_ONLY)       # 五条组合规则
TIERS = ["main"]
# ==============================================================


def main(capacities=CAPACITIES, methods=METHODS, tiers=TIERS):
    run_dirs = []
    for capacity in capacities:
        run_dirs += run_experiment(name=f"rules_capacity_c{capacity}",
                                   overlays=[], runs=1,
                                   methods=methods, tiers=tiers, train_first=False,
                                   overrides={"env.robot_capacity": capacity})
    return run_dirs


if __name__ == "__main__":
    main()
