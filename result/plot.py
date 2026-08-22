"""Draft figures from the CSV files -- vector PDF plus a 300 dpi PNG.

The style follows the publication conventions used across the project: a closed
rectangular frame, inward ticks, Times-like serif labels at readable size, a
colour-blind safe palette and, wherever several independent runs exist, a mean
line with a standard-deviation band.

Run::

    python -m result.plot                       # every run under result/
    python -m result.plot --pattern "e4_*"      # one experiment family
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Okabe-Ito, colour-blind safe.
PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000")
MARKERS = ("o", "s", "^", "D", "v", "P", "X")
LINESTYLES = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1)))

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 13,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.dpi": 110,
})


def save(fig, out_stem: str) -> None:
    os.makedirs(os.path.dirname(out_stem) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"{out_stem}.pdf")
    fig.savefig(f"{out_stem}.png", dpi=300)
    plt.close(fig)
    print(f"  -> {out_stem}.pdf / .png")


def _smooth(values: Sequence[float], window: int = 20) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < window or window <= 1:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_training_curves(run_dirs: Sequence[str], out_stem: str,
                         column: str = "mean_flow_time") -> None:
    """Mean line with a standard-deviation band across independent runs."""
    series: List[np.ndarray] = []
    for run_dir in run_dirs:
        path = os.path.join(run_dir, "log.csv")
        if not os.path.exists(path):
            continue
        frame = pd.read_csv(path)
        if column not in frame.columns:
            continue
        series.append(_smooth(frame[column].to_numpy(dtype=float)))
    if not series:
        print(f"  no '{column}' column found under the selected runs; skipped")
        return

    length = min(len(s) for s in series)
    stack = np.vstack([s[:length] for s in series])
    steps = np.arange(1, length + 1)
    mean = stack.mean(axis=0)
    std = stack.std(axis=0, ddof=1) if stack.shape[0] > 1 else np.zeros_like(mean)

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(steps, mean, color=PALETTE[0], linewidth=1.6,
            label=f"mean of {stack.shape[0]} run(s)")
    if stack.shape[0] > 1:
        ax.fill_between(steps, mean - std, mean + std, color=PALETTE[0], alpha=0.20,
                        linewidth=0)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Mean order flow time (s)")
    ax.legend(frameon=False)
    save(fig, out_stem)


def plot_method_comparison(frame: pd.DataFrame, out_stem: str) -> None:
    """Box plot of F-bar per method over the evaluation instances."""
    methods = sorted(frame["method"].unique())
    data = [frame.loc[frame["method"] == m, "mean_flow_time"].to_numpy() for m in methods]

    fig, ax = plt.subplots(figsize=(1.1 * len(methods) + 2.0, 3.4))
    box = ax.boxplot(data, patch_artist=True, widths=0.6, tick_labels=methods)
    for patch, colour in zip(box["boxes"], PALETTE * 4):
        patch.set_facecolor(colour)
        patch.set_alpha(0.35)
        patch.set_edgecolor(colour)
    for median in box["medians"]:
        median.set_color("black")
    ax.set_ylabel("Mean order flow time (s)")
    ax.tick_params(axis="x", rotation=20)
    save(fig, out_stem)


def plot_sensitivity(frame: pd.DataFrame, x_column: str, out_stem: str) -> None:
    """F-bar against a swept configuration variable, one line per method."""
    if x_column not in frame.columns:
        print(f"  column '{x_column}' absent; skipped")
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for i, (method, group) in enumerate(frame.groupby("method")):
        curve = group.groupby(x_column)["mean_flow_time"].agg(["mean", "std"]).sort_index()
        ax.plot(curve.index, curve["mean"], color=PALETTE[i % len(PALETTE)],
                marker=MARKERS[i % len(MARKERS)], linestyle=LINESTYLES[i % len(LINESTYLES)],
                linewidth=1.5, markersize=5, label=method)
        if curve["std"].notna().any():
            ax.fill_between(curve.index, curve["mean"] - curve["std"].fillna(0),
                            curve["mean"] + curve["std"].fillna(0),
                            color=PALETTE[i % len(PALETTE)], alpha=0.15, linewidth=0)
    ax.set_xlabel(x_column.replace("_", " "))
    ax.set_ylabel("Mean order flow time (s)")
    ax.legend(frameon=False)
    save(fig, out_stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", default="result")
    parser.add_argument("--pattern", default="*")
    parser.add_argument("--figures-dir", default=None)
    parser.add_argument("--sensitivity-column", default=None,
                        help="e.g. robot_capacity, n_robots, pick_time, gamma")
    args = parser.parse_args()

    run_dirs = sorted(d for d in glob.glob(os.path.join(args.result_dir, args.pattern))
                      if os.path.isdir(d))
    figures_dir = args.figures_dir or os.path.join(args.result_dir, "figures")
    print(f"{len(run_dirs)} run directory(ies) matched")

    print("training curves:")
    plot_training_curves(run_dirs, os.path.join(figures_dir, "training_curve"))

    frames = [pd.read_csv(os.path.join(d, "eval_results.csv")) for d in run_dirs
              if os.path.exists(os.path.join(d, "eval_results.csv"))]
    if frames:
        frame = pd.concat(frames, ignore_index=True)
        print("method comparison:")
        plot_method_comparison(frame, os.path.join(figures_dir, "method_comparison"))
        if args.sensitivity_column:
            print(f"sensitivity to {args.sensitivity_column}:")
            plot_sensitivity(frame, args.sensitivity_column,
                             os.path.join(figures_dir,
                                          f"sensitivity_{args.sensitivity_column}"))
    else:
        print("no eval_results.csv found; only training curves were drawn")


if __name__ == "__main__":
    main()
