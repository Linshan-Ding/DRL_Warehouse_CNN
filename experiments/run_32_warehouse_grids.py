"""仓库网格敏感性 —— 论文 Tables 11-12（12x20 与 9x30 两个非基准网格）。

网格尺寸改变输入几何（CNN 网格与动作数），因此**按网格从零重训**——这正是
对 R2.1 的限定性回答：资源规模零样本泛化（单策略），仓库几何需重训。
每个网格：生成该网格自己的 3 条 1/λ=40 订单流（固定、落盘）→ 训练 →
评测。默认只训 SAPPO 并配五条规则；ALGOS 加入其它算法即可补全 RL 列。

产出: data/instances/grid_<AxB>/、models/*_<AxB>.pt、result/grid/<AxB>/eval_results.csv
耗时: 每网格每算法一次完整训练。
"""
import _bootstrap  # noqa: F401

import os
import random
import shutil

from agent.registry import model_path
from configs.config import load_config
from data.generator import sample_order_records, save_stream_csv
from environment.problem import Warehouse
from eval import evaluate
from _runner import banner, summarise, train_algo

# ==================== 配置区（改完右键 Run） ====================
GRIDS = [("12x20", "configs/exp/grid_12x20.yaml"),
         ("9x30", "configs/exp/grid_9x30.yaml")]
ALGOS = ["SAPPO"]                 # 补全 Table 11/12 的 RL 列时加入其余四个
RULES = ["MQ-ND", "MQ-MinRQ", "MQ-MI", "MI-MinRQ", "MI-MI"]
N_STREAMS = 3                     # 每网格的固定订单流条数（1/λ=40）
EPISODES = None
WORKERS = None
SAMPLES = 3
# ==============================================================


def _grid_streams(cfg, tag: str, n_streams: int):
    """该网格自己的固定订单流（商品分布覆盖整个网格，与 9x20 流不同）。"""
    directory = os.path.join(cfg.instance.instances_dir, f"grid_{tag}")
    os.makedirs(directory, exist_ok=True)
    warehouse = Warehouse(cfg.env)
    rng = random.Random()
    paths = []
    for i in range(n_streams):
        path = os.path.join(directory, f"stream{i:02d}.csv")
        if not os.path.exists(path):
            save_stream_csv(sample_order_records(warehouse, cfg.instance,
                                                 cfg.instance.n_orders, 40.0, rng), path)
            print(f"[grid {tag}] generated {path}")
        paths.append(path)
    return paths


def main(grids=GRIDS, algos=ALGOS, rules=RULES, n_streams=N_STREAMS,
         episodes=EPISODES, workers=WORKERS, samples=SAMPLES):
    import csv as csv_mod

    outs = []
    for tag, overlay in grids:
        cfg = load_config([overlay])
        streams = _grid_streams(cfg, tag, n_streams)

        # index rows for evaluate(): reuse the case mechanism with a local index
        index_dir = os.path.join(cfg.instance.instances_dir)
        for algo in algos:
            main_model = model_path(cfg, algo)
            tagged = model_path(cfg, algo, suffix=tag)
            keep = main_model + ".keep"
            if os.path.exists(main_model):
                shutil.move(main_model, keep)
            try:
                train_algo(algo, overlays=[overlay], run_name=f"train_{algo.lower()}_{tag}",
                           episodes=episodes, workers=workers)
                shutil.move(main_model, tagged)
            finally:
                if os.path.exists(keep):
                    shutil.move(keep, main_model)

        # evaluation over the grid streams, methods loaded from tagged models
        from configs.config import Config
        from eval import ModelMethod, RuleMethod, solve
        rows = []
        for algo in algos + rules:
            if algo in rules:
                method = RuleMethod(algo)
            else:
                method = ModelMethod(cfg, algo, checkpoint=model_path(cfg, algo, suffix=tag))
            for si, stream in enumerate(streams):
                n = 1 if method.deterministic else samples
                for sample in range(1, n + 1):
                    metrics = solve(cfg, method, stream, {})
                    rows.append({"case": f"{tag}-s{si}", "mean_interarrival": 40.0,
                                 "case_id": "", "method": method.name, "sample_id": sample,
                                 "n_aisles": cfg.env.n_aisles,
                                 "n_positions": cfg.env.n_positions,
                                 "n_pickers": cfg.env.n_pickers,
                                 "n_robots": cfg.env.n_robots,
                                 "robot_capacity": cfg.env.robot_capacity,
                                 "state_channels": cfg.env.state_channels,
                                 "layout": cfg.env.layout, "pick_time": cfg.env.pick_time,
                                 "gamma": cfg.algo.gamma, "grid": tag,
                                 **{k: metrics[k] for k in (
                                     "mean_flow_time", "makespan", "n_completed",
                                     "n_orders", "n_decisions", "decision_time_ms",
                                     "sim_time_per_decision", "solve_wall_clock_s")}})
                print(f"  [{tag}] {method.name:<9} stream{si} 完成")
        out_dir = os.path.join("result", "grid", tag)
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, "eval_results.csv")
        with open(out, "w", newline="", encoding="utf-8") as fh:
            writer = csv_mod.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        banner(f"[grid {tag}] {len(rows)} rows -> {out}")
        outs.append(out)
    summarise(outs)
    return outs


if __name__ == "__main__":
    main()
