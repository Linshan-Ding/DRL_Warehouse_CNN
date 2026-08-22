"""按顺序批跑多个实验 —— 一键把 run 矩阵跑完。

先改 EXPERIMENTS 这个列表，把不想跑的注释掉，再右键 Run。注意下面标的耗时是
开发容器 CPU（约 66 决策/秒）的量级，全开会跑好几天，GPU 上快很多。

建议的顺序和优先级:
    1. selfcheck + data + smoke   先确认环境没问题（分钟级）
    2. e0                         基线复现，是其余实验的门槛（必跑）
    3. e4 + e5                    直接关掉 Reviewer #2 的第 3、4 条（性价比最高）
    4. e1 + e7                    Reviewer #1 的配比与载运容量
    5. e2 + e6 + e8               规模、拣选时间、布局，视时间加码
    6. e3 + stats                 汇总，几秒钟
"""
import _bootstrap  # noqa: F401  必须最先导入

import time

from _runner import banner

# ==================== 配置区（改完右键 Run） ====================
EXPERIMENTS = [
    "selfcheck",     # 约 1 分钟   正确性自检，强烈建议保留
    "data",          # 几秒        生成固定算例
    # "smoke",       # 约 3 分钟   极小预算跑通全链路
    "e0",            # 约 27-36 h  基线复现 x3（门槛，必跑）
    # "e1",          # 约 27-36 h  人机配比 (1,1)/(3,1)/(4,2)
    # "e2",          # 约 20-30 h  更大规模 (8,16)/(10,20)
    "e4",            # 约 3-4 天   gamma 消融 3 档 x3 次
    "e5",            # 约 1-1.5 天 状态通道消融 x3 次
    # "e6",          # 约 18-24 h  tau_pick 敏感性
    # "e7",          # 约 18-24 h  载运容量 C=2/3
    # "e8",          # 约 9-12 h   中部横通道布局
    "rules",         # 约 2 分钟   五条规则基线
    "e3",            # 几秒        训练成本汇总
    "stats",         # 几秒        统计聚合与出图
]
EPISODES = None      # None = 用配置文件里的 2000；先填 20 可端到端演练整条流水线
# ==============================================================

_STEPS = {
    "selfcheck": ("正确性自检", "run_00_selfcheck"),
    "data": ("生成固定算例", "run_01_prepare_data"),
    "smoke": ("冒烟测试", "run_smoke"),
    "e0": ("E0 基线复现", "run_e0_baseline"),
    "e1": ("E1 人机配比", "run_e1_ratio"),
    "e2": ("E2 更大规模", "run_e2_scale"),
    "e3": ("E3 训练成本汇总", "run_e3_training_cost"),
    "e4": ("E4 gamma 消融", "run_e4_gamma"),
    "e5": ("E5 状态通道消融", "run_e5_state_channels"),
    "e6": ("E6 拣选时间敏感性", "run_e6_picktime"),
    "e7": ("E7 载运容量", "run_e7_capacity"),
    "e8": ("E8 布局变体", "run_e8_layout"),
    "rules": ("规则基线", "run_rules_only"),
    "stats": ("统计聚合与出图", "run_stats_and_plots"),
}

# 接受 EPISODES 参数的实验（训练类）
_TRAINING_STEPS = {"smoke", "e0", "e1", "e2", "e4", "e5", "e6", "e7", "e8"}


def main(experiments=EXPERIMENTS, episodes=EPISODES):
    unknown = [key for key in experiments if key not in _STEPS]
    if unknown:
        raise ValueError(f"未知的实验代号: {unknown}；可选: {sorted(_STEPS)}")

    banner("批量运行", "计划: " + " -> ".join(_STEPS[key][0] for key in experiments))
    started = time.time()

    for position, key in enumerate(experiments, start=1):
        title, module_name = _STEPS[key]
        banner(f"[{position}/{len(experiments)}] {title}")
        module = __import__(module_name)
        if episodes and key in _TRAINING_STEPS:
            module.main(episodes=episodes)
        else:
            module.main()

    banner("全部完成", f"总耗时 {(time.time() - started) / 3600:.2f} 小时")


if __name__ == "__main__":
    main()
