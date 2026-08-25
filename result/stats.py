"""Aggregate evaluation results and test the differences for significance.

Input: one or more ``eval_results.csv`` produced by ``eval.py`` (one row per
case x method x stochastic sample).  Output: per-method summary plus, for every
method, a paired comparison against SAPPO over the shared cases -- paired
t-test, Wilcoxon signed-rank test and Cohen's d, exactly the battery of
Section 5.5 (Tables 8-9).  The samples are paired because every method solves
the same fixed case streams.

Run: right-click ``experiments/run_40_stats_tables_plots.py`` (or
``python -m result.stats --dirs result/main``).
"""
from __future__ import annotations

import argparse
import glob
import math
import os
from typing import List, Sequence

import pandas as pd

try:
    from scipy import stats as scipy_stats
except ImportError:  # pragma: no cover
    scipy_stats = None

REFERENCE_METHOD = "SAPPO"


def load_results(dirs: Sequence[str]) -> pd.DataFrame:
    frames = []
    for pattern in dirs:
        for path in sorted(glob.glob(os.path.join(pattern, "**", "eval_results.csv"),
                                     recursive=True)):
            frame = pd.read_csv(path)
            frame["source"] = os.path.relpath(path)
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no eval_results.csv under {list(dirs)}")
    return pd.concat(frames, ignore_index=True)


def per_case_means(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean over the stochastic samples -> one row per (case, method)."""
    return (frame.groupby(["case", "method"], as_index=False)
            .agg(mean_flow_time=("mean_flow_time", "mean"),
                 flow_std=("mean_flow_time", "std"),
                 decision_time_ms=("decision_time_ms", "mean"),
                 n_samples=("sample_id", "count")))


def summarise_methods(frame: pd.DataFrame) -> pd.DataFrame:
    cases = per_case_means(frame)
    rows = []
    for method, group in cases.groupby("method"):
        rows.append({"method": method,
                     "n_cases": int(group["case"].nunique()),
                     "flow_mean": float(group["mean_flow_time"].mean()),
                     "flow_std_within_case": float(group["flow_std"].mean()),
                     "decision_time_ms_mean": float(group["decision_time_ms"].mean())})
    return pd.DataFrame(rows).sort_values("flow_mean").reset_index(drop=True)


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    diff = [x - y for x, y in zip(a, b)]
    n = len(diff)
    if n < 2:
        return float("nan")
    mean = sum(diff) / n
    sd = math.sqrt(sum((d - mean) ** 2 for d in diff) / (n - 1))
    return mean / sd if sd > 0 else float("nan")


def paired_tests(frame: pd.DataFrame, reference: str = REFERENCE_METHOD) -> pd.DataFrame:
    pivot = per_case_means(frame).pivot(index="case", columns="method",
                                        values="mean_flow_time")
    if reference not in pivot.columns:
        return pd.DataFrame()
    rows: List[dict] = []
    for method in pivot.columns:
        if method == reference:
            continue
        pair = pivot[[reference, method]].dropna()
        if len(pair) < 2:
            continue
        ours, other = pair[reference].tolist(), pair[method].tolist()
        row = {"comparison": f"{reference} vs {method}",
               "n_paired_cases": len(pair),
               "mean_reference": sum(ours) / len(ours),
               "mean_other": sum(other) / len(other),
               "avg_gap": (sum(ours) / len(ours) - sum(other) / len(other))
                          / (sum(other) / len(other)),
               "cohens_d": cohens_d(ours, other)}
        if scipy_stats is not None:
            t_stat, t_p = scipy_stats.ttest_rel(ours, other)
            row["t_statistic"], row["p_value_t"] = float(t_stat), float(t_p)
            try:
                w_stat, w_p = scipy_stats.wilcoxon(ours, other)
                row["w_statistic"], row["p_value_wilcoxon"] = float(w_stat), float(w_p)
            except ValueError:
                row["w_statistic"] = row["p_value_wilcoxon"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def write_summary(dirs: Sequence[str], reference: str = REFERENCE_METHOD,
                  out: str | None = None) -> str:
    frame = load_results(dirs)
    summary = summarise_methods(frame)
    tests = paired_tests(frame, reference)
    out = out or "result/stats_summary.csv"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    pd.concat([summary.assign(block="per_method"),
               tests.assign(block="paired_test")], ignore_index=True).to_csv(out, index=False)
    print(summary.to_string(index=False))
    if not tests.empty:
        print(); print(tests.to_string(index=False))
    print(f"\nwritten -> {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dirs", nargs="+", default=["result/main"])
    parser.add_argument("--reference", default=REFERENCE_METHOD)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    write_summary(args.dirs, args.reference, args.out)


if __name__ == "__main__":
    main()
