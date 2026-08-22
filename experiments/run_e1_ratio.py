"""E1 人机配比场景 —— 回应 Reviewer #1 关于 1:1 / 1:n / n:1 的意见。

论文已有的 27 个算例（K in {1,2,3} x R in {2,4,6}）其实已经覆盖了
1:2、1:4、1:6、1:1（K=2,R=2）、2:4、2:6、3:2、3:4、3:6 九种配比，
只是从来没有按"配比"这个角度组织过。这里补上缺的三个极端档:

    (K,R) = (1,1)   纯 1:1
    (K,R) = (3,1)   拣货员远多于机器人（n:1）
    (K,R) = (4,2)   n:1 的另一档

注意每个 (K,R) 都要单独训练: actor 输出层维度是 K*N_w*N_l + R*(N_w*N_l+1)，
换了资源配置就换了网络结构，权重不可能跨配置复用。

产出: result/e1_k1r1_run*/、result/e1_k3r1_run*/、result/e1_k4r2_run*/ 下的
      eval_results.csv（含 n_pickers、n_robots 两列，可直接按配比重排成表）
耗时: 每个配置一次训练，三个配置约 27-36 小时（CPU）。
"""
import _bootstrap  # noqa: F401  必须最先导入

from _runner import DEFAULT_METHODS, run_experiment

# ==================== 配置区（改完右键 Run） ====================
CONFIGS = [                       # (运行名, 配置文件)
    ("e1_k1r1", "configs/exp/e1_ratio_k1_r1.yaml"),
    ("e1_k3r1", "configs/exp/e1_ratio_k3_r1.yaml"),
    ("e1_k4r2", "configs/exp/e1_ratio_k4_r2.yaml"),
]
RUNS = 1                          # 每个配置的重复次数
EPISODES = None                   # None = 用配置文件里的 2000
METHODS = list(DEFAULT_METHODS)
TIERS = ["main"]
# ==============================================================


def main(configs=CONFIGS, runs=RUNS, episodes=EPISODES, methods=METHODS, tiers=TIERS):
    run_dirs = []
    for name, overlay in configs:
        run_dirs += run_experiment(name=name, overlays=[overlay], runs=runs,
                                   episodes=episodes, methods=methods, tiers=tiers)
    return run_dirs


if __name__ == "__main__":
    main()
