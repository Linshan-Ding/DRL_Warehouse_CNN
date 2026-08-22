"""五条组合排序规则的评测 —— 不需要训练，几分钟出结果。

规则是确定性的，所以跑一次就够。这份结果同时也是所有新表的对比列来源。

对应论文: 5.3 节的 MQ-ND / MQ-MinRQ / MQ-MI / MI-MinRQ / MI-MI

产出: result/rules_main/eval_results.csv
耗时: 约 2 分钟。
"""
import _bootstrap  # noqa: F401  必须最先导入

from _runner import RULES_ONLY, run_experiment

# ==================== 配置区（改完右键 Run） ====================
METHODS = list(RULES_ONLY)   # 五条组合规则
TIERS = ["main"]             # 想一起评大规模档就写 ["main", "large"]
OVERLAYS = []                # 想在别的资源配置下评规则就填 configs/exp/xxx.yaml
# ==============================================================


def main(methods=METHODS, tiers=TIERS, overlays=OVERLAYS):
    return run_experiment(name="rules_main", overlays=overlays, runs=1,
                          methods=methods, tiers=tiers, train_first=False)


if __name__ == "__main__":
    main()
