"""实验脚本的公共执行逻辑。

这里不重复实现任何东西，只是把各模块串起来：`train.train` 训练全局策略、
`eval.evaluate` 在固定算例上评测、`data.dataset.make_instances` 生成算例。
各实验脚本只声明"训练哪个算法 / 评哪些方法 / 用什么场景覆盖"。
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import _bootstrap  # noqa: F401  必须最先导入：修好 sys.path 与工作目录

from agent.registry import ALGORITHMS, model_path
from baselines.rules import PAPER_RULES
from configs.config import load_config
from data.dataset import make_instances
from eval import evaluate
from train import train

ALL_METHODS: Sequence[str] = tuple(ALGORITHMS) + tuple(PAPER_RULES)
LAMBDA40_CASES = [f"C{i:02d}" for i in range(10, 19)]   # 中等负载锚点流（1/λ=40）

_LINE = "=" * 78


def banner(title: str, detail: str = "") -> None:
    print(f"\n{_LINE}\n{title}" + (f"\n{detail}" if detail else "") + f"\n{_LINE}")


def train_algo(algo: str, overlays: Sequence[str] = (), run_name: Optional[str] = None,
               episodes: Optional[int] = None, workers: Optional[int] = None) -> str:
    overrides: Dict[str, object] = {}
    if episodes:
        overrides["train.n_episodes"] = episodes
    if workers:
        overrides["train.n_workers"] = workers
    cfg = load_config(list(overlays), overrides)
    if run_name:
        cfg.run.run_name = run_name
    make_instances(cfg)
    banner(f"训练 {algo}",
           f"配置: {' + '.join(overlays) if overlays else '默认'} | "
           f"episodes={cfg.train.n_episodes} | 产出: {model_path(cfg, algo)}")
    return train(cfg, algo)


def eval_methods(methods: Sequence[str], out_dir: str,
                 cases: Optional[Sequence[str]] = None, samples: int = 3,
                 overlays: Sequence[str] = (),
                 env_overrides: Optional[dict] = None,
                 fleet_from_index: bool = True) -> str:
    cfg = load_config(list(overlays))
    make_instances(cfg)
    banner(f"评测 -> {out_dir}",
           f"方法: {', '.join(methods)} | 算例: {'全部27' if not cases else ','.join(cases)} | "
           f"采样次数: {samples}（规则确定性单次）")
    return evaluate(cfg, list(methods), out_dir, cases, samples,
                    env_overrides, fleet_from_index)


def summarise(paths: Sequence[str]) -> None:
    banner("完成")
    for path in paths:
        print(f"  {os.path.abspath(path)}")
    print("\n下一步: 全部实验跑完后右键运行 experiments/run_40_stats_tables_plots.py")
