"""统计聚合与出图 —— 跑完实验之后收尾用。

做三件事:
1. 把匹配到的运行的 eval_results.csv 汇总，按方法给出 F 的均值/标准差，
   并对每个方法与 SAPPO 做配对检验（配对 t 检验 + Wilcoxon 符号秩 + Cohen's d）。
   之所以是"配对"，是因为所有方法解的是同一批固定算例。
2. 画训练曲线（多次独立运行画均值线 + 标准差带）与方法对比箱线图，
   矢量 PDF 加 300 dpi PNG 各一份。
3. 画论文 Fig. 5 的状态表示示意图。

只想看某一组实验就改 PATTERN，例如 "e4_*" 只看折扣因子消融。

耗时: 几秒到一分钟。
等价的终端命令:
    python -m result.stats --pattern "*"
    python -m result.plot --pattern "*"
    python -m result.figs.state_illustration --out result/figures
"""
import _bootstrap  # noqa: F401  必须最先导入

import os

from result.plot import render_all
from result.stats import write_summary
from _runner import banner

# ==================== 配置区（改完右键 Run） ====================
PATTERN = "*"                 # 匹配哪些运行目录: "*" 全部, "e4_*" 只看 E4
REFERENCE = "SAPPO"           # 配对检验的参照方法
SENSITIVITY_COLUMN = None     # 画敏感性曲线的横轴列, 例如 "gamma"
                              # "robot_capacity" / "pick_time" / "n_robots"
SUMMARY_OUT = None            # None = result/stats_summary.csv
DRAW_STATE_FIGURE = True      # 是否顺便画状态表示示意图
# ==============================================================


def main(pattern=PATTERN, reference=REFERENCE, sensitivity_column=SENSITIVITY_COLUMN,
         summary_out=SUMMARY_OUT, draw_state_figure=DRAW_STATE_FIGURE):
    banner("统计聚合", f"匹配模式: {pattern}")
    try:
        out = write_summary("result", pattern, reference, summary_out)
    except FileNotFoundError as error:
        print(error)
        return None

    banner("绘图")
    figures_dir = render_all("result", pattern, None, sensitivity_column)

    if draw_state_figure:
        from result.figs.state_illustration import CHANNELS, plot_channel
        banner("状态表示示意图")
        os.makedirs(figures_dir, exist_ok=True)
        for index, (title, matrix, cmap) in enumerate(CHANNELS, start=1):
            plot_channel(matrix, title, cmap,
                         os.path.join(figures_dir, f"state_channel_{index}"))

    banner("完成")
    print(f"  统计表: {os.path.abspath(out)}")
    print(f"  图目录: {os.path.abspath(figures_dir)}")
    return out


if __name__ == "__main__":
    main()
