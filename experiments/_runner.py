"""实验脚本的公共执行逻辑。

这里不重复实现任何东西，只是把各模块串起来：

* ``data.dataset.make_eval_instances`` 生成固定算例
* ``train.train``                     训练并存 checkpoint
* ``eval.evaluate``                   在固定算例上评测并写 eval_results.csv

各实验脚本只负责声明"用哪个配置、跑几次、评哪些方法"，具体流程都在这里。
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import _bootstrap  # noqa: F401  必须最先导入：修好 sys.path 与工作目录

from baselines.rules import PAPER_RULES
from configs.config import load_config
from data.dataset import make_eval_instances
from eval import evaluate
from train import train

DEFAULT_METHODS: Sequence[str] = ("SAPPO",) + tuple(PAPER_RULES)
RULES_ONLY: Sequence[str] = tuple(PAPER_RULES)

_LINE = "=" * 78


def banner(title: str, detail: str = "") -> None:
    print(f"\n{_LINE}\n{title}" + (f"\n{detail}" if detail else "") + f"\n{_LINE}")


def run_experiment(name: str,
                   overlays: Sequence[str] = (),
                   runs: int = 1,
                   episodes: Optional[int] = None,
                   methods: Sequence[str] = DEFAULT_METHODS,
                   tiers: Sequence[str] = ("main",),
                   train_first: bool = True,
                   overrides: Optional[Dict[str, object]] = None) -> List[str]:
    """跑一个实验：按需训练，再在固定算例上评测。

    :param name:        运行目录前缀，实际目录是 ``result/<name>_run<i>``
    :param overlays:    叠加在默认配置之上的 YAML（``configs/exp/*.yaml``）
    :param runs:        独立重复次数；论文级消融建议 >= 3
    :param episodes:    覆盖 ``algo.n_episodes``；``None`` 表示用配置文件里的值
    :param methods:     参与评测的方法；含 "SAPPO" 时必须先训练
    :param tiers:       评测算例档位，main / val / large
    :param train_first: ``False`` 表示只评测（用于纯规则基线）
    :param overrides:   额外的点号覆盖，例如 ``{"env.pick_time": 15.0}``
    :return:            本次产出的 run 目录列表
    """
    overlays = list(overlays)
    methods = list(methods)
    if not train_first and "SAPPO" in methods:
        raise ValueError("只评测模式下不能包含 SAPPO —— 它需要先训练出 checkpoint")

    run_dirs: List[str] = []
    for run_id in range(1, runs + 1):
        run_name = f"{name}_run{run_id}" if runs > 1 or train_first else name

        merged: Dict[str, object] = dict(overrides or {})
        if episodes:
            merged["algo.n_episodes"] = episodes
        cfg = load_config(overlays, merged)
        cfg.run.run_name = run_name

        banner(f"[{name}] 第 {run_id}/{runs} 次运行 -> result/{run_name}",
               f"配置文件: {' + '.join(overlays) if overlays else '默认'}\n"
               f"资源设置: K={cfg.env.n_pickers} R={cfg.env.n_robots} "
               f"C={cfg.env.robot_capacity} 通道={cfg.env.state_channels}\n"
               f"算法设置: gamma={cfg.algo.gamma} tau_pick={cfg.env.pick_time} "
               f"布局={cfg.env.layout} 训练轮数={cfg.algo.n_episodes}\n"
               f"评测方法: {', '.join(methods)}   档位: {', '.join(tiers)}")

        make_eval_instances(cfg)

        checkpoint = None
        if train_first:
            run_dir = train(cfg)
            checkpoint = os.path.join(run_dir, "checkpoint_best.pt")
        else:
            run_dir = os.path.join(cfg.run.result_dir, run_name)

        evaluate(cfg, methods, tiers, checkpoint, run_id,
                 os.path.join(run_dir, "eval_results.csv"))
        run_dirs.append(run_dir)

    summarise(name, run_dirs)
    return run_dirs


def summarise(name: str, run_dirs: Sequence[str]) -> None:
    """打印产出文件的绝对路径，PyCharm 控制台里可以直接点开。"""
    banner(f"[{name}] 完成，共 {len(run_dirs)} 个运行目录")
    for run_dir in run_dirs:
        print(f"  {os.path.abspath(run_dir)}")
        for filename in ("log.csv", "eval_results.csv", "training_cost.csv",
                         "checkpoint_best.pt", "config_snapshot.yaml"):
            path = os.path.join(run_dir, filename)
            if os.path.exists(path):
                print(f"    - {filename}")
    print("\n下一步: 右键运行 experiments/run_stats_and_plots.py 做统计聚合与出图")
