"""E5 状态通道消融 —— 回应 Reviewer #2 第 4 条关于状态充分性。

审稿人指出的不对称真实存在: 四个基础通道全部是按拣货位聚合的，而动作掩码用的是
每台机器人的剩余必访点、每个拣货员是否空闲这类个体信息。也就是说"哪台机器人还差
哪些点"根本无法从状态张量里还原，只通过掩码进入决策。

plus_agent 通道集就是把这部分信息也摆到网络面前:
    第 5 通道  待路径决策机器人的剩余必访点分布
    第 6 通道  空闲资源（待决策机器人 + 空闲拣货员）的位置分布

和 E0 在同等训练预算下对比:
    提升很小   -> 支撑"四通道 + 掩码已经充分"的论证
    提升明显   -> 据实报告并把它写进正文（结论被推翻也是合法产出，而且是更好的论文）

耗时: RUNS 次训练，默认 3 次约 1-1.5 天（CPU）。对照组直接用 E0 的结果，不必重跑。
等价的终端命令:
    python train.py --config configs/exp/e5_state_plus_agent.yaml --run-name e5_plus_run1
    python eval.py --config configs/exp/e5_state_plus_agent.yaml \
                   --ckpt result/e5_plus_run1/checkpoint_best.pt \
                   --methods SAPPO --tiers main --run-id 1 --run-name e5_plus_run1
"""
import _bootstrap  # noqa: F401  必须最先导入

from _runner import run_experiment

# ==================== 配置区（改完右键 Run） ====================
RUNS = 3                          # 独立重复次数，消融建议 >= 3
EPISODES = None
METHODS = ["SAPPO"]
TIERS = ["main"]
# ==============================================================


def main(runs=RUNS, episodes=EPISODES, methods=METHODS, tiers=TIERS):
    return run_experiment(name="e5_plus",
                          overlays=["configs/exp/e5_state_plus_agent.yaml"],
                          runs=runs, episodes=episodes, methods=methods, tiers=tiers)


if __name__ == "__main__":
    main()
