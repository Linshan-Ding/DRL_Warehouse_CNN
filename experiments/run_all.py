"""按顺序批跑 —— 一键执行整个 run 矩阵。

改 STAGES 列表（注释掉不想跑的）再右键 Run。依赖关系与并行性见 README 第 3 节：
01 最先；10-14 互相独立（多机可并行，单机串行）；20-24 需要对应的 models；
30/31/32 是额外训练；40 最后。单机上本脚本就是推荐的串行顺序。
"""
import _bootstrap  # noqa: F401

import time

from _runner import banner

# ==================== 配置区（改完右键 Run） ====================
STAGES = [
    "selfcheck",     # ~1 分钟
    "instances",     # 几秒
    "train_sappo",   # 一次全局训练
    "train_agdqn",
    "train_hsddqn",
    "train_soa_a2c",
    "train_drlg",
    "eval_main",     # ~1-2 小时
    "eval_scale",
    "eval_capacity",
    "eval_picktime",
    "eval_layout",
    "gamma",         # 2 次额外训练
    "state",         # 1 次额外训练
    "grids",         # 每网格每算法 1 次训练
    "finalize",      # 统计+表格+图
]
EPISODES = None      # 统一覆盖训练轮数；填 20 可整链路演练
# ==============================================================

_STEPS = {
    "selfcheck": ("正确性自检", "run_00_selfcheck", False),
    "instances": ("生成固定算例", "run_01_make_instances", False),
    "train_sappo": ("训练 SAPPO", "run_10_train_sappo", True),
    "train_agdqn": ("训练 AG-DQN", "run_11_train_agdqn", True),
    "train_hsddqn": ("训练 HSDDQN", "run_12_train_hsddqn", True),
    "train_soa_a2c": ("训练 SOA+A2C", "run_13_train_soa_a2c", True),
    "train_drlg": ("训练 DRLG", "run_14_train_drlg", True),
    "eval_main": ("主对比评测", "run_20_eval_main", False),
    "eval_scale": ("规模零样本评测", "run_21_eval_scale", False),
    "eval_capacity": ("容量敏感性", "run_22_eval_capacity", False),
    "eval_picktime": ("拣选时间敏感性", "run_23_eval_picktime", False),
    "eval_layout": ("布局变体", "run_24_eval_layout", False),
    "gamma": ("γ 消融", "run_30_gamma_ablation", True),
    "state": ("状态通道消融", "run_31_state_ablation", True),
    "grids": ("仓库网格敏感性", "run_32_warehouse_grids", True),
    "finalize": ("统计+表格+图", "run_40_stats_tables_plots", False),
}


def main(stages=STAGES, episodes=EPISODES):
    unknown = [s for s in stages if s not in _STEPS]
    if unknown:
        raise ValueError(f"未知阶段: {unknown}")
    banner("批量运行", " -> ".join(_STEPS[s][0] for s in stages))
    started = time.time()
    for i, stage in enumerate(stages, 1):
        title, module_name, takes_episodes = _STEPS[stage]
        banner(f"[{i}/{len(stages)}] {title}")
        module = __import__(module_name)
        if episodes and takes_episodes:
            module.main(episodes=episodes)
        else:
            module.main()
    banner("全部完成", f"总耗时 {(time.time() - started) / 3600:.2f} 小时")


if __name__ == "__main__":
    main()
