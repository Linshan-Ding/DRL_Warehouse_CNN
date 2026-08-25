"""统计检验 + 论文表格 + 图 —— 全部实验跑完后的收尾（串行，最后）。

1. result.stats: 按方法汇总 + 每方法 vs SAPPO 的配对检验（t / Wilcoxon /
   Cohen's d，跨 27 算例配对）→ result/stats_summary.csv（Table 8/9 数据）。
2. paper_assets.make_tables: 从全部 CSV 生成修订稿与回复信用的 LaTeX 表格片段
   → paper_assets/tables/*.tex（数字不手抄）。
3. result.plot: 训练曲线（代表算例）、方法箱线图等草图 → result/figures/。

产出见上。耗时: 约 1 分钟。
"""
import _bootstrap  # noqa: F401

from paper_assets.make_tables import main as make_tables
from result.plot import render_all
from result.stats import write_summary
from _runner import banner

# ==================== 配置区（改完右键 Run） ====================
STATS_DIRS = ["result/main"]     # 配对检验只看主对比
# ==============================================================


def main(stats_dirs=STATS_DIRS):
    banner("统计检验")
    try:
        write_summary(stats_dirs)
    except FileNotFoundError as error:
        print(error)
    banner("LaTeX 表格")
    make_tables()
    banner("草图")
    render_all("result", "train_*")
    banner("完成")


if __name__ == "__main__":
    main()
