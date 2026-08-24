"""主对比评测 —— 论文 Table 5 / Table 7 / Table 8 / Table 9 与 Fig. 9-11 的数据。

前置: 五个训练脚本 run_10..run_14 全部完成（models/ 下五个文件齐）。
协议: 每个 RL 算法在 27 个固定算例上各做 3 次随机策略采样评估（策略梯度类按
策略分布采样，值函数类用 ε=0.05 的 ε-贪婪）→ F̄ 均值±标准差；五条调度规则
确定性单次。D̄ = 每次决策的真实墙钟毫秒（只计动作计算）。
本评测单进程串行执行——并行会污染 D̄ 的计时，属有意设计。

产出: result/main/eval_results.csv
耗时: 约 1-2 小时（27 算例 x (5 RL x 3 + 5 规则)）。
"""
import _bootstrap  # noqa: F401

from _runner import ALL_METHODS, eval_methods, summarise

# ==================== 配置区（改完右键 Run） ====================
METHODS = list(ALL_METHODS)   # 5 个 RL + 5 条规则
SAMPLES = 3                   # 每个 RL 方法每算例的随机评估次数
CASES = None                  # None = 全部 27；调试可填 ["C18"]
# ==============================================================


def main(methods=METHODS, samples=SAMPLES, cases=CASES):
    out = eval_methods(methods, "result/main", cases, samples)
    summarise([out])
    return out


if __name__ == "__main__":
    main()
