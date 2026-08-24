"""生成 27 个固定算例与验证流 —— 只需跑一次（串行，最先）。

27 算例 = R∈{2,4,6} x K∈{1,2,3} x 1/λ∈{20,40,60}，编号 C01..C27
（λ 外层、K 中层、R 内层，与论文 Table 5 一致），每个算例一条独立订单流。
已存在的文件永不覆盖——这些 CSV（而非随机种子）是复现基准。

产出: data/instances/cases/C01..C27.csv、data/instances/val/*.csv、index.csv
耗时: 几秒。
"""
import _bootstrap  # noqa: F401

import os

from configs.config import load_config
from data.dataset import make_instances
from _runner import banner


def main():
    banner("生成固定算例")
    path = make_instances(load_config())
    print(f"\n索引: {os.path.abspath(path)}")
    return path


if __name__ == "__main__":
    main()
