"""资源规模零样本评测 —— 论文 Table 10（回应 R1.6 与 R2.1）。

同一批全局策略（不重训、不换文件）直接在更大的机队上评测：
(4,8) (5,10) (6,12) (8,16) (10,20)，订单流用九条 1/λ=40 的算例流。
这正是对审稿人"是否跨规模迁移"质疑的直接回答——一个策略文件通吃全部规模。

前置: run_10..run_14。产出: result/scale/K<k>_R<r>/eval_results.csv
耗时: 每档约 20-40 分钟。
"""
import _bootstrap  # noqa: F401

from _runner import ALL_METHODS, LAMBDA40_CASES, eval_methods, summarise

# ==================== 配置区（改完右键 Run） ====================
FLEETS = [(4, 8), (5, 10), (6, 12), (8, 16), (10, 20)]
METHODS = list(ALL_METHODS)
SAMPLES = 3
CASES = list(LAMBDA40_CASES)   # 九条 1/λ=40 锚点流
# ==============================================================


def main(fleets=FLEETS, methods=METHODS, samples=SAMPLES, cases=CASES):
    outs = []
    for k, r in fleets:
        outs.append(eval_methods(
            methods, f"result/scale/K{k}_R{r}", cases, samples,
            env_overrides={"n_pickers": k, "n_robots": r}, fleet_from_index=False))
    summarise(outs)
    return outs


if __name__ == "__main__":
    main()
