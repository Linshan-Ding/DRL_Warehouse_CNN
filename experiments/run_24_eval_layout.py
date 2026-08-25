"""中部横通道布局变体 —— 回应 R1.3（双横通道行取自 result/main）。

产出: result/layout/three/eval_results.csv  耗时: 约 20-40 分钟。
"""
import _bootstrap  # noqa: F401

from _runner import ALL_METHODS, LAMBDA40_CASES, eval_methods, summarise

# ==================== 配置区（改完右键 Run） ====================
METHODS = list(ALL_METHODS)
SAMPLES = 3
CASES = list(LAMBDA40_CASES)
# ==============================================================


def main(methods=METHODS, samples=SAMPLES, cases=CASES):
    out = eval_methods(methods, "result/layout/three", cases, samples,
                       env_overrides={"layout": "three_cross_aisles"})
    summarise([out])
    return out


if __name__ == "__main__":
    main()
