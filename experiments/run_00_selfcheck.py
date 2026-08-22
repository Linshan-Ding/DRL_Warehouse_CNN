"""正确性自检 —— 跑任何实验之前先跑这个。

三道闸门（全部必须 PASS）:

1. 奖励恒等式    一个 episode 的无折扣累计奖励恰等于 -F_final，
                 说明智能体优化的确实是论文的目标函数而不是某个代理量。
2. 与原实现等价  用同一条确定性规则同时驱动重构后的环境和
                 tools/reference/env_I_submitted.py（产生已投稿结果的原实现，
                 原样保留），逐个决策点比对时钟、所选动作、奖励、状态张量和最终 F。
3. 容量退化      C=1 与原来的"一次一单"模型完全一致，C=2 确实改变调度。

产出: 不写文件，结果直接打印在控制台，三行都是 PASS 才算通过。
耗时: 约 1 分钟。
"""
import _bootstrap  # noqa: F401  必须最先导入

from configs.config import load_config
from data.generator import load_stream_csv
from tools.selfcheck import (check_capacity_degeneracy, check_legacy_equivalence,
                             check_reward_identity)
from _runner import banner

# ==================== 配置区（改完右键 Run） ====================
STREAM = "data/instances/main/lam40.csv"   # 用哪条订单流做自检
RULE = "MQ-ND"                             # 驱动自检的确定性规则
OVERLAYS = []                              # 想检查别的配置就填 configs/exp/xxx.yaml
# ==============================================================


def main(stream=STREAM, rule=RULE, overlays=OVERLAYS):
    cfg = load_config(overlays)
    records = load_stream_csv(stream)
    banner("正确性自检",
           f"订单流: {stream}\n"
           f"配置: N_w={cfg.env.n_aisles} N_l={cfg.env.n_positions} "
           f"K={cfg.env.n_pickers} R={cfg.env.n_robots} C={cfg.env.robot_capacity} "
           f"规则={rule}")

    results = {
        "奖励恒等式": check_reward_identity(cfg, records, rule),
        "与原实现逐事件等价": check_legacy_equivalence(cfg, records, rule),
        "载运容量退化": check_capacity_degeneracy(cfg, records, rule),
    }

    banner("自检结果")
    for item, ok in results.items():
        print(f"  {item:<24} {'PASS' if ok else 'FAIL'}")
    failed = [item for item, ok in results.items() if not ok]
    print("\n全部通过，可以开始跑实验" if not failed else f"\n未通过: {', '.join(failed)}")
    return not failed


if __name__ == "__main__":
    main()
