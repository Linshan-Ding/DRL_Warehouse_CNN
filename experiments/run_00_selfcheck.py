"""正确性自检 —— 跑任何实验之前先跑这个（串行，最先）。

三道闸门（全部必须 PASS）:
1. 奖励恒等式    无折扣累计奖励恰等于 -F_final，智能体优化的确实是论文目标。
2. 与原实现等价  用同一条确定性规则驱动重构环境与 tools/reference/ 里
                 产生已投稿结果的原实现，逐决策点比对时钟/动作/奖励/前 4 个
                 空间通道/最终 F（新观测追加的配置平面不参与物理比对）。
3. 容量退化      C=1 与"一次一单"模型完全一致，C=2 确实改变调度。

产出: 控制台三行 PASS。耗时: 约 1 分钟。
"""
import _bootstrap  # noqa: F401

from configs.config import load_config
from data.dataset import make_instances
from data.generator import load_stream_csv
from tools.selfcheck import (check_capacity_degeneracy, check_legacy_equivalence,
                             check_reward_identity)
from _runner import banner

# ==================== 配置区（改完右键 Run） ====================
STREAM = "data/instances/cases/C18.csv"
RULE = "MQ-ND"
# ==============================================================


def main(stream=STREAM, rule=RULE):
    cfg = load_config()
    make_instances(cfg)
    records = load_stream_csv(stream)
    banner("正确性自检", f"订单流: {stream} | 规则: {rule}")
    results = {"奖励恒等式": check_reward_identity(cfg, records, rule),
               "与原实现逐事件等价": check_legacy_equivalence(cfg, records, rule),
               "载运容量退化": check_capacity_degeneracy(cfg, records, rule)}
    banner("自检结果")
    for item, ok in results.items():
        print(f"  {item:<24} {'PASS' if ok else 'FAIL'}")
    failed = [k for k, ok in results.items() if not ok]
    print("\n全部通过" if not failed else f"\n未通过: {failed}")
    return not failed


if __name__ == "__main__":
    main()
