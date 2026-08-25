"""拣选时间敏感性（多层货架代理）—— 回应 R1.4（τ=10 行取自 result/main）。

产出: result/picktime/tau<t>/eval_results.csv  耗时: 每档约 20-40 分钟。
"""
import _bootstrap  # noqa: F401

from _runner import ALL_METHODS, LAMBDA40_CASES, eval_methods, summarise

# ==================== 配置区（改完右键 Run） ====================
PICK_TIMES = [15.0, 20.0]
METHODS = list(ALL_METHODS)
SAMPLES = 3
CASES = list(LAMBDA40_CASES)
# ==============================================================


def main(pick_times=PICK_TIMES, methods=METHODS, samples=SAMPLES, cases=CASES):
    outs = [eval_methods(methods, f"result/picktime/tau{t:g}", cases, samples,
                         env_overrides={"pick_time": t})
            for t in pick_times]
    summarise(outs)
    return outs


if __name__ == "__main__":
    main()
