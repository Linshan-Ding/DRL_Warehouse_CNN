"""E3 训练成本汇总 —— 回应 Reviewer #2 第 1 条关于"训练成本随规模如何变化"。

不需要单独训练: 每次 train.py 都会写一份 training_cost.csv，这个脚本只是把
result/ 下所有的汇总成一张表，列出动作空间大小、参数量、训练轮数、墙钟时长和
决策吞吐。先跑 E0 / E1 / E2，再跑这个。

顺便回答了同一条意见的另一半: actor 输出层维度是 K*N_w*N_l + R*(N_w*N_l+1)，
所以每个配置都是从零单独训练的，不存在跨规模的权重迁移。表里的 n_actions 一列
就是这句话的证据。

产出: result/training_cost_summary.csv
耗时: 几秒。
"""
import _bootstrap  # noqa: F401  必须最先导入

import glob
import os

import pandas as pd

from _runner import banner

# ==================== 配置区（改完右键 Run） ====================
PATTERN = "result/*/training_cost.csv"                 # 汇总哪些运行
OUT = "result/training_cost_summary.csv"               # 汇总表写到哪
COLUMNS = ["run_name", "n_pickers", "n_robots", "robot_capacity", "state_channels",
           "gamma", "n_actions", "n_parameters", "n_episodes", "total_decisions",
           "wall_clock_s", "decisions_per_second", "device"]
# ==============================================================


def main(pattern=PATTERN, out=OUT, columns=COLUMNS):
    paths = sorted(glob.glob(pattern))
    banner("训练成本汇总", f"匹配到 {len(paths)} 个运行")
    if not paths:
        print("没有找到 training_cost.csv —— 先跑 E0 / E1 / E2 再来")
        return None

    table = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    table = table.sort_values("n_actions").reset_index(drop=True)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    table.to_csv(out, index=False)

    present = [c for c in columns if c in table.columns]
    print(table[present].to_string(index=False))
    print(f"\n汇总表: {os.path.abspath(out)}")
    return out


if __name__ == "__main__":
    main()
