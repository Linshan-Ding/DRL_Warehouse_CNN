"""状态通道消融 —— 回应 R2.4（基线观测的行取自主训练与 result/main）。

plus_agent 观测在配置平面之外再加两个通道：待路径决策机器人的剩余必访点分布、
空闲资源位置分布——正是掩码在用、基线观测看不到的个体信息。训练一个 SAPPO
变体并评测，与主结果同预算对比。

产出: models/sappo_plus.pt、result/state_plus/eval_results.csv
耗时: 1 次完整训练 + 1 次评测。
"""
import _bootstrap  # noqa: F401

import os
import shutil

from agent.registry import model_path
from configs.config import load_config
from _runner import eval_methods, summarise, train_algo

# ==================== 配置区（改完右键 Run） ====================
OVERLAY = "configs/exp/state_plus_agent.yaml"
EPISODES = None
WORKERS = None
SAMPLES = 3
CASES = None
# ==============================================================


def main(overlay=OVERLAY, episodes=EPISODES, workers=WORKERS, samples=SAMPLES, cases=CASES):
    cfg = load_config([overlay])
    main_model = model_path(cfg, "SAPPO")
    tagged_model = model_path(cfg, "SAPPO", suffix="plus")
    keep = main_model + ".keep"
    if os.path.exists(main_model):
        shutil.move(main_model, keep)
    try:
        train_algo("SAPPO", overlays=[overlay], run_name="train_sappo_plus",
                   episodes=episodes, workers=workers)
        shutil.move(main_model, tagged_model)
        shutil.copy(tagged_model, main_model)
        out = eval_methods(["SAPPO"], "result/state_plus", cases, samples,
                           overlays=[overlay])
        os.remove(main_model)
    finally:
        if os.path.exists(keep):
            shutil.move(keep, main_model)
    summarise([out])
    return out


if __name__ == "__main__":
    main()
