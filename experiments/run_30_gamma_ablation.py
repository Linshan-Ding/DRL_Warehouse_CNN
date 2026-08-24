"""折扣因子消融 —— 回应 R2.3（γ=0.99 的行直接取自主训练与 result/main）。

r_t = F̄_{t-1} - F̄_t 只有 γ=1 时严格 telescoping 成 -F̄_final；γ<1 时按 Abel
分部求和多一个 O(1-γ) 路径项，方向一致但不恒等。本实验训练 γ∈{0.95, 1.0}
两个额外的 SAPPO 全局策略（模型文件带后缀，不覆盖主模型），再各自评测。

与其它训练脚本互相独立，但同样吃满 CPU——单机与 run_10..14 串行执行。
产出: models/sappo_g0.95.pt、models/sappo_g1.00.pt、result/gamma/g*/eval_results.csv
耗时: 2 次完整训练 + 2 次评测。
"""
import _bootstrap  # noqa: F401

import os
import shutil

from agent.registry import model_path
from configs.config import load_config
from _runner import banner, eval_methods, summarise, train_algo

# ==================== 配置区（改完右键 Run） ====================
GAMMAS = [("0.95", "configs/exp/gamma_0.95.yaml"),
          ("1.00", "configs/exp/gamma_1.00.yaml")]
EPISODES = None    # None = 15000；填 20 可快速验证链路
WORKERS = None
SAMPLES = 3
CASES = None       # None = 全部 27 算例
# ==============================================================


def main(gammas=GAMMAS, episodes=EPISODES, workers=WORKERS, samples=SAMPLES, cases=CASES):
    outs = []
    for tag, overlay in gammas:
        cfg = load_config([overlay])
        main_model = model_path(cfg, "SAPPO")
        tagged_model = model_path(cfg, "SAPPO", suffix=f"g{tag}")
        keep = main_model + ".keep"
        if os.path.exists(main_model):
            shutil.move(main_model, keep)     # 防止消融训练覆盖主模型
        try:
            train_algo("SAPPO", overlays=[overlay], run_name=f"train_sappo_g{tag}",
                       episodes=episodes, workers=workers)
            shutil.move(main_model, tagged_model)
            banner(f"γ={tag} 模型 -> {tagged_model}")
            shutil.copy(tagged_model, main_model)
            outs.append(eval_methods(["SAPPO"], f"result/gamma/g{tag}", cases, samples,
                                     overlays=[overlay]))
            os.remove(main_model)
        finally:
            if os.path.exists(keep):
                shutil.move(keep, main_model)
    summarise(outs)
    return outs


if __name__ == "__main__":
    main()
