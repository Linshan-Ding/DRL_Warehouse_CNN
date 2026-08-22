"""Aggregate evaluation results and test the differences for significance.

Reads every ``eval_results.csv`` under ``result/`` and writes
``result/stats_summary.csv``.

Two levels of aggregation, matching Section 5.5 of the manuscript:

* **per method**: mean +/- standard deviation of F-bar over the evaluation
  instances, and over the independent runs of the same configuration;
* **method vs SAPPO**: paired tests over the shared instances -- a paired
  t-test, a Wilcoxon signed-rank test and Cohen's d.  The samples are paired
  because every method solves the very same fixed instances.

Run::

    python -m result.stats                      # everything under result/
    python -m result.stats --pattern "e4_*"     # one experiment family
"""
from __future__ import annotations

import argparse
import glob
import math
import os
from typing import Dict, List, Sequence

import pandas as pd

try:
    from scipy import stats as scipy_stats
except ImportError:  # pragma: no cover
    scipy_stats = None

REFERENCE_METHOD = "SAPPO"
KEY_COLUMNS = ("instance_id", "tier", "mean_interarrival", "n_pickers", "n_robots",
               "robot_capacity", "case_id")


def load_results(result_dir: str = "result", pattern: str = "*") -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(result_dir, pattern, "eval_results.csv")))
    if not paths:
        raise FileNotFoundError(
            f"no eval_results.csv under {result_dir}/{pattern}/ -- run eval.py first")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source"] = os.path.relpath(path, result_dir)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Paired Cohen's d: mean difference over the standard deviation of the differences."""
    diff = [x - y for x, y in zip(a, b)]
    n = len(diff)
    if n < 2:
        return float("nan")
    mean = sum(diff) / n
    variance = sum((d - mean) ** 2 for d in diff) / (n - 1)
    sd = math.sqrt(variance)
    return mean / sd if sd > 0 else float("nan")


def summarise_methods(frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for method, group in frame.groupby("method"):
        per_run = group.groupby("run_id")["mean_flow_time"].mean()
        rows.append({
            "method": method,
            "n_instances": int(group["instance_id"].nunique()),
            "n_runs": int(group["run_id"].nunique()),
            "flow_mean": float(group["mean_flow_time"].mean()),
            "flow_std_over_instances": float(group["mean_flow_time"].std(ddof=1))
                if len(group) > 1 else 0.0,
            "flow_std_over_runs": float(per_run.std(ddof=1)) if len(per_run) > 1 else 0.0,
            "decision_time_ms_mean": float(group["decision_time_ms"].mean()),
            "sim_time_per_decision_mean": float(group["sim_time_per_decision"].mean()),
        })
    return pd.DataFrame(rows).sort_values("flow_mean").reset_index(drop=True)


def paired_tests(frame: pd.DataFrame, reference: str = REFERENCE_METHOD) -> pd.DataFrame:
    keys = [c for c in KEY_COLUMNS if c in frame.columns]
    pivot = (frame.groupby(keys + ["method"])["mean_flow_time"].mean()
             .unstack("method"))
    if reference not in pivot.columns:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for method in pivot.columns:
        if method == reference:
            continue
        pair = pivot[[reference, method]].dropna()
        if len(pair) < 2:
            continue
        ours = pair[reference].tolist()
        other = pair[method].tolist()
        gap = (sum(ours) / len(ours) - sum(other) / len(other)) / (sum(other) / len(other))

        row: Dict[str, object] = {
            "comparison": f"{reference} vs {method}",
            "n_paired_instances": len(pair),
            "mean_reference": sum(ours) / len(ours),
            "mean_other": sum(other) / len(other),
            "avg_gap": gap,
            "cohens_d": cohens_d(ours, other),
        }
        if scipy_stats is not None:
            t_stat, t_p = scipy_stats.ttest_rel(ours, other)
            row["t_statistic"] = float(t_stat)
            row["p_value_t"] = float(t_p)
            try:
                w_stat, w_p = scipy_stats.wilcoxon(ours, other)
                row["w_statistic"] = float(w_stat)
                row["p_value_wilcoxon"] = float(w_p)
            except ValueError:
                row["w_statistic"] = float("nan")
                row["p_value_wilcoxon"] = float("nan")
        else:
            row["note"] = "scipy not installed; significance tests skipped"
        rows.append(row)
    return pd.DataFrame(rows)


def write_summary(result_dir: str = "result", pattern: str = "*",
                  reference: str = REFERENCE_METHOD,
                  out: str | None = None) -> str:
    """Aggregate, test, print and write the summary; returns the output path."""
    frame = load_results(result_dir, pattern)
    summary = summarise_methods(frame)
    tests = paired_tests(frame, reference)

    out = out or os.path.join(result_dir, "stats_summary.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    pd.concat([summary.assign(block="per_method"),
               tests.assign(block="paired_test")], ignore_index=True).to_csv(out, index=False)

    print(summary.to_string(index=False))
    if not tests.empty:
        print()
        print(tests.to_string(index=False))
    print(f"\nwritten -> {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", default="result")
    parser.add_argument("--pattern", default="*",
                        help="glob over run directories, e.g. 'e4_*'")
    parser.add_argument("--reference", default=REFERENCE_METHOD)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    write_summary(args.result_dir, args.pattern, args.reference, args.out)


if __name__ == "__main__":
    main()
