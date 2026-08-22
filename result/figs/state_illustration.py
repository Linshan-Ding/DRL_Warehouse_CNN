"""Schematic of the four-channel state representation (manuscript Fig. 5).

A deliberately tiny 3 x 3 warehouse is used so that every cell can be labelled;
the matrices below are illustrative, not measured.  Run::

    python -m result.figs.state_illustration --out result/figures

Products: one vector PDF plus a 300 dpi PNG per channel.
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 14,
})

# Channels of the state tensor for a 3 x 3 toy grid (rows: aisles N_w,
# columns: picking points per aisle N_l).
CHANNELS = (
    ("M_r  robots queueing", np.array([[2, 0, 1], [1, 3, 0], [0, 0, 2]]), "Blues"),
    ("M_k  picker present", np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), "Greens"),
    ("M_u  unpicked items", np.array([[5, 0, 3], [2, 7, 0], [0, 0, 4]]), "Oranges"),
    ("M_q  unassigned order items", np.array([[3, 0, 1], [0, 4, 0], [2, 0, 0]]), "Purples"),
)


def plot_channel(matrix: np.ndarray, title: str, cmap: str, out_stem: str) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    image = ax.imshow(matrix, cmap=cmap)
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, str(int(value)), ha="center", va="center",
                color="black", fontsize=13)
    ax.set_xlabel(r"Picking point index $n_l$")
    ax.set_ylabel(r"Aisle index $n_w$")
    ax.set_xticks(range(matrix.shape[1]), [str(i + 1) for i in range(matrix.shape[1])])
    ax.set_yticks(range(matrix.shape[0]), [str(i + 1) for i in range(matrix.shape[0])])
    fig.colorbar(image, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(f"{out_stem}.pdf")
    fig.savefig(f"{out_stem}.png", dpi=300)
    plt.close(fig)
    print(f"  {title} -> {out_stem}.pdf / .png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="result/figures")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for index, (title, matrix, cmap) in enumerate(CHANNELS, start=1):
        stem = os.path.join(args.out, f"state_channel_{index}")
        plot_channel(matrix, title, cmap, stem)


if __name__ == "__main__":
    main()
