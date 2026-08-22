"""生成各档固定评测算例 —— 只需要跑一次。

产物: data/instances/{main,val,large}/*.csv 与 data/instances/index.csv

已存在的文件永远不会被覆盖: 这些算例文件（而不是随机种子）才是复现基准，
本项目任何地方都不固定随机种子。其中 main 档的三条订单流就是产生已投稿结果的
那三条，随仓库一起提交，所以新克隆下来会直接复用而不是重新生成。

耗时: 几秒。
"""
import _bootstrap  # noqa: F401  必须最先导入

import os

from configs.config import load_config
from data.dataset import make_eval_instances
from _runner import banner

# ==================== 配置区（改完右键 Run） ====================
OVERLAYS = []      # 想按别的算例参数表生成就填 configs/exp/xxx.yaml
# ==============================================================


def main(overlays=OVERLAYS):
    cfg = load_config(overlays)
    banner("生成固定评测算例")
    index_path = make_eval_instances(cfg)
    print(f"\n索引文件: {os.path.abspath(index_path)}")
    return index_path


if __name__ == "__main__":
    main()
