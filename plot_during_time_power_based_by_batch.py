import argparse
import re
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def find_csv_files(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*.csv") if p.is_file())


def sanitize_name(name: str) -> str:
    name = re.sub(r"[^\w\-]+", "_", name)
    return name.strip("_") or "unknown"


def load_data(log_dir: Path) -> pd.DataFrame:
    csv_files = find_csv_files(log_dir)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found: {log_dir}")

    dfs = []
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            df["__source_file"] = str(csv_path.relative_to(log_dir))
            dfs.append(df)
        except Exception as exc:
            print(f"Skip unreadable CSV: {csv_path} ({exc})")

    if not dfs:
        raise RuntimeError("No readable CSV files.")

    return pd.concat(dfs, ignore_index=True)


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = [
        "batch_size",
        "input_len",
        "kv_cache_lens",
        "model_name",
        "sm_clock",
        "index",
        "length",
        "during_time",
        "repeat_count",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    numeric_cols = [
        "batch_size",
        "input_len",
        "kv_cache_lens",
        "sm_clock",
        "index",
        "length",
        "during_time",
        "repeat_count",
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    key_cols = ["model_name", "batch_size", "input_len", "kv_cache_lens", "sm_clock"]

    kv_counts = (
        df.groupby(["model_name", "batch_size", "input_len", "sm_clock"])["kv_cache_lens"]
        .nunique()
        .reset_index(name="kv_unique_cnt")
    )
    valid_keys = kv_counts[kv_counts["kv_unique_cnt"] > 20][
        ["model_name", "batch_size", "input_len", "sm_clock"]
    ]
    df = df.merge(valid_keys, on=["model_name", "batch_size", "input_len", "sm_clock"])

    df = df.copy()
    df["idx_ratio"] = df["index"] / df["length"]
    df["prompt_gap"] = df["kv_cache_lens"] - df["input_len"]
    df["during_time_per_repeat"] = df["during_time"] / df["repeat_count"]

    filtered = df[(df["idx_ratio"] >= 0.5) & (df["prompt_gap"] >= 5)]

    agg = (
        filtered.groupby(key_cols, as_index=False)
        .agg(
            avg_during_time=("during_time_per_repeat", "mean"),
            avg_repeat=("repeat_count", "mean"),
        )
    )

    agg["during_time_metric"] = agg["avg_during_time"]
    agg = agg[np.isfinite(agg["during_time_metric"])].copy()
    return agg


def plot_during_time_by_batch(metrics: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_sizes = sorted(metrics["batch_size"].dropna().unique())
    model_names = sorted(metrics["model_name"].dropna().unique())

    if len(batch_sizes) == 0:
        raise RuntimeError("No batch_size data to plot.")

    for bs in batch_sizes:
        plt.figure(figsize=(10, 6))

        has_data = False
        for model_name in model_names:
            sub = metrics[
                (metrics["batch_size"] == bs) & (metrics["model_name"] == model_name)
            ]
            if sub.empty:
                continue

            line = (
                sub.groupby("sm_clock", as_index=False)["during_time_metric"]
                .mean()
                .sort_values("sm_clock")
            )
            if line.empty:
                continue

            has_data = True
            plt.plot(
                line["sm_clock"],
                line["during_time_metric"],
                marker="o",
                linewidth=1.8,
                label=str(model_name),
            )

        if not has_data:
            plt.close()
            print(f"No data, skip batch_size={bs}")
            continue

        plt.xlabel("SM clock frequency (MHz)")
        plt.ylabel("During time per repeat (sec)")
        plt.title(f"During time by model (batch_size={int(bs)})")
        plt.grid(True, alpha=0.3)
        plt.legend(title="model_name")
        plt.tight_layout()

        file_name = f"during_time_bs{int(bs)}.png"
        out_path = output_dir / file_name
        plt.savefig(out_path)
        plt.close()
        print(f"Saved: {out_path}")


def summarize_change_as_batch_increases(metrics: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = (
        metrics.groupby(["model_name", "sm_clock", "batch_size"], as_index=False)["during_time_metric"]
        .mean()
        .sort_values(["model_name", "sm_clock", "batch_size"])
    )

    grouped["prev_batch_size"] = grouped.groupby(["model_name", "sm_clock"])["batch_size"].shift(1)
    grouped["prev_during_time"] = grouped.groupby(["model_name", "sm_clock"])["during_time_metric"].shift(1)

    grouped["abs_change"] = grouped["during_time_metric"] - grouped["prev_during_time"]
    grouped["pct_change"] = grouped["abs_change"] / grouped["prev_during_time"] * 100.0

    summary = grouped.dropna(subset=["prev_batch_size", "prev_during_time"]).copy()
    summary = summary.rename(
        columns={
            "during_time_metric": "curr_during_time",
            "batch_size": "curr_batch_size",
        }
    )

    summary_path = output_dir / "during_time_change_by_batch_step.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    for model_name in sorted(summary["model_name"].dropna().unique()):
        sub_model = summary[summary["model_name"] == model_name]
        if sub_model.empty:
            continue

        plt.figure(figsize=(10, 6))
        has_data = False

        for sm_clock in sorted(sub_model["sm_clock"].dropna().unique()):
            sub_clock = sub_model[sub_model["sm_clock"] == sm_clock].sort_values("curr_batch_size")
            if sub_clock.empty:
                continue

            has_data = True
            plt.plot(
                sub_clock["curr_batch_size"],
                sub_clock["pct_change"],
                marker="o",
                linewidth=1.6,
                label=f"sm_clock={int(sm_clock)}",
            )

        if not has_data:
            plt.close()
            continue

        plt.xlabel("Current batch_size")
        plt.ylabel("Change vs previous batch (%)")
        plt.title(f"Batch-size effect on during time by SM clock ({model_name})")
        plt.grid(True, alpha=0.3)
        plt.legend(title="sm_clock")
        plt.tight_layout()

        model_safe = sanitize_name(str(model_name))
        out_path = output_dir / f"change_pct_by_batch_during_time_{model_safe}.png"
        plt.savefig(out_path)
        plt.close()
        print(f"Saved: {out_path}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge all CSV files and create batch-wise during_time line plots with model_name legend."
        )
    )
    parser.add_argument(
        "log_dir",
        type=str,
        help="Root directory that contains CSV files (searched recursively)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: <log_dir>/analysis_output_during_time)",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {log_dir}")

    output_dir = (
        Path(args.output_dir) if args.output_dir else (log_dir / "analysis_output_during_time")
    )

    df = load_data(log_dir)
    metrics = compute_metrics(df)
    plot_during_time_by_batch(metrics, output_dir)
    summarize_change_as_batch_increases(metrics, output_dir)


if __name__ == "__main__":
    main()