"""
지정 폴더 내 gpu_profile_*.csv 파일들을 로드하여
6가지 메트릭을 계산하고, 각각 CSV로 저장한 뒤 sm_clock별 히스토그램을 그린다.

사용법:
  python plot_6metrics_histogram.py <log_dir> [--output-dir <out>]

메트릭:
  1) total_energy_based_j_per_token
  2) power_based_j_per_token
  3) latency  (sec/token)
  4) throughput  (tokens/sec)
  5) total_energy_based_avg_power  (W)
  6) power_based_avg_power  (W)
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_KEY = ["cudagraph_mode", "batch_size", "sm_clock", "input_len", "model_name"]
ITER_KEY = DATA_KEY + ["kv_cache_lens"]


# ── 유틸 ──────────────────────────────────────────────
def sanitize(name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", name).strip("_") or "unknown"


# ── 로드 ──────────────────────────────────────────────
def load_csvs(log_dir: Path) -> pd.DataFrame:
    csv_files = sorted(log_dir.glob("gpu_profile_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"gpu_profile_*.csv 파일을 찾을 수 없습니다: {log_dir}")

    dfs = []
    for p in csv_files:
        print(f"   로드: {p.name}")
        dfs.append(pd.read_csv(p))
    df = pd.concat(dfs, ignore_index=True)

    num_cols = [
        "batch_size", "input_len", "kv_cache_lens", "sm_clock",
        "index", "length", "power", "during_time", "repeat_count",
        "total_energy", "gpu_util", "memory_util",
    ]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    return df


# ── iteration 단위 중간 집계 ──────────────────────────
def compute_iteration_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    iteration = (batch_size, sm_clock, input_len, model_name, kv_cache_lens) 그룹.
    각 iteration에 대해:
      - delta_total_energy_mJ : total_energy[index==length] - total_energy[index==1]
      - decoding_tokens       : repeat_count * batch_size
      - during_time           : 해당 iteration의 during_time (첫 행 값, 상수)
      - avg_power_saturated   : index/length >= 0.5 인 행의 평균 power
      - energy_power_based_J  : avg_power_saturated * during_time
    """
    rows = []
    for keys, grp in df.groupby(ITER_KEY, sort=False):
        grp_sorted = grp.sort_values("index")
        idx_1 = grp_sorted[grp_sorted["index"] == 1]
        idx_max = grp_sorted[grp_sorted["index"] == grp_sorted["length"].iloc[0]]

        if idx_1.empty or idx_max.empty:
            continue

        te_start = idx_1["total_energy"].iloc[0]
        te_end = idx_max["total_energy"].iloc[0]
        delta_te_mJ = te_end - te_start

        if delta_te_mJ <= 0:
            continue

        during = grp_sorted["during_time"].iloc[0]
        repeat = grp_sorted["repeat_count"].iloc[0]
        bs = grp_sorted["batch_size"].iloc[0]
        tokens = repeat * bs

        grp_sorted = grp_sorted.copy()
        grp_sorted["idx_ratio"] = grp_sorted["index"] / grp_sorted["length"]
        saturated = grp_sorted[grp_sorted["idx_ratio"] > 0.5]
        avg_pwr = saturated["power"].mean() if not saturated.empty else np.nan
        measured_sm = int(saturated["sm_clock"].mean()) if not saturated.empty else int(grp_sorted["sm_clock"].mean())
        avg_gpu_util = saturated["gpu_util"].mean() if not saturated.empty else np.nan
        avg_mem_util = saturated["memory_util"].mean() if not saturated.empty else np.nan

        energy_pwr_J = avg_pwr * during if np.isfinite(avg_pwr) else np.nan

        row = dict(zip(ITER_KEY, keys))
        row["sm_clock"] = measured_sm
        row["delta_total_energy_mJ"] = delta_te_mJ
        row["decoding_tokens"] = tokens
        row["during_time_iter"] = during
        row["repeat_count"] = repeat
        row["avg_power_saturated"] = avg_pwr
        row["energy_power_based_J"] = energy_pwr_J
        row["avg_gpu_util"] = avg_gpu_util
        row["avg_memory_util"] = avg_mem_util
        rows.append(row)

    return pd.DataFrame(rows)


# ── data 단위 최종 메트릭 집계 ────────────────────────
def aggregate_metrics(it: pd.DataFrame, min_iterations: int = 0) -> pd.DataFrame:
    """
    data = (batch_size, sm_clock, input_len, model_name) 그룹.
    iteration 수가 min_iterations 이하인 그룹은 drop한다.
    """
    records = []
    dropped = []
    for keys, grp in it.groupby(DATA_KEY, sort=False):
        if len(grp) <= min_iterations:
            dropped.append(dict(zip(DATA_KEY, keys)) | {"iterations": len(grp)})
            continue
        sum_delta_te = grp["delta_total_energy_mJ"].sum()
        sum_tokens = grp["decoding_tokens"].sum()
        sum_dur = grp["during_time_iter"].sum()

        # (1) total energy based J/token  (mJ -> J : /1000)
        te_j_per_tok = (sum_delta_te / 1000.0) / sum_tokens if sum_tokens else np.nan

        # (2) power based J/token
        sum_energy_pwr = grp["energy_power_based_J"].sum()
        pwr_j_per_tok = sum_energy_pwr / sum_tokens if sum_tokens else np.nan

        # (3) latency (sec/token)
        latency = sum_dur / sum_tokens if sum_tokens else np.nan

        # (4) throughput (tokens/sec)
        throughput = 1.0 / latency if latency and latency > 0 else np.nan

        # (5) total energy based avg power (W)
        te_avg_power = (sum_delta_te / 1000.0) / sum_dur if sum_dur else np.nan

        # (6) power based avg power (W)
        pwr_avg_power = grp["avg_power_saturated"].mean()

        # (7) gpu_util (%)
        gpu_util = grp["avg_gpu_util"].mean()

        # (8) memory_util (%)
        mem_util = grp["avg_memory_util"].mean()

        row = dict(zip(DATA_KEY, keys))
        row["total_energy_based_j_per_token"] = te_j_per_tok
        row["power_based_j_per_token"] = pwr_j_per_tok
        row["latency_sec_per_token"] = latency
        row["throughput_tokens_per_sec"] = throughput
        row["total_energy_based_avg_power_W"] = te_avg_power
        row["power_based_avg_power_W"] = pwr_avg_power
        row["gpu_util_pct"] = gpu_util
        row["memory_util_pct"] = mem_util
        records.append(row)

    if dropped:
        print(f"   drop된 그룹 (iterations <= {min_iterations}):")
        for d in dropped:
            print(f"     {d}")

    return pd.DataFrame(records)


# ── 히스토그램 플롯 ───────────────────────────────────
METRIC_COLS = [
    ("total_energy_based_j_per_token", "Total Energy Based J/token", "J/token"),
    ("power_based_j_per_token", "Power Based J/token", "J/token"),
    ("latency_sec_per_token", "Latency", "sec/token"),
    ("throughput_tokens_per_sec", "Throughput", "tokens/sec"),
    ("total_energy_based_avg_power_W", "Total Energy Based Avg Power", "W"),
    ("power_based_avg_power_W", "Power Based Avg Power", "W"),
    ("gpu_util_pct", "GPU Utilization", "%"),
    ("memory_util_pct", "Memory Utilization", "%"),
]

LABEL_FMT = {
    "total_energy_based_avg_power_W": ".2f",
    "power_based_avg_power_W": ".2f",
    "gpu_util_pct": ".1f",
    "memory_util_pct": ".1f",
}

MODEL_COLORS_DARK = {
    "attn": "#1565C0",
    "mlp": "#C62828",
}
MODEL_COLORS_LIGHT = {
    "attn": "#BBDEFB",
    "mlp": "#FFCDD2",
}


MAX_COLOR = "#222222"


def _bar_colors(model_name: str, vals: np.ndarray, sm_clocks: np.ndarray) -> list[str]:
    """min 값 → 진한 모델색, max sm_clock → 검정, 나머지 → 연한색."""
    dark = MODEL_COLORS_DARK.get(str(model_name), "#607D8B")
    light = MODEL_COLORS_LIGHT.get(str(model_name), "#CFD8DC")
    finite_mask = np.isfinite(vals)
    if not finite_mask.any():
        return [light] * len(vals)
    max_sm_idx = int(np.argmax(sm_clocks))
    min_val_idx = int(np.nanargmin(vals))
    colors = [light] * len(vals)
    colors[max_sm_idx] = MAX_COLOR
    colors[min_val_idx] = dark
    return colors


def plot_histograms(metrics: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_sizes = sorted(metrics["batch_size"].dropna().unique())
    input_lens = sorted(metrics["input_len"].dropna().unique())
    model_names = sorted(metrics["model_name"].dropna().unique())

    for col, title, ylabel in METRIC_COLS:
        for bs in batch_sizes:
            sub_bs = metrics[metrics["batch_size"] == bs]
            if sub_bs.empty:
                continue

            combos = []
            for il in input_lens:
                for mn in model_names:
                    s = sub_bs[(sub_bs["input_len"] == il) & (sub_bs["model_name"] == mn)]
                    if not s.empty:
                        combos.append((il, mn))

            if not combos:
                continue

            ncols = min(len(combos), 3)
            nrows = math.ceil(len(combos) / ncols)
            fig_w = 6.5 * ncols + 1
            fig_h = 4.5 * nrows + 1
            fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
            ax_flat = axes.ravel()

            for idx, (il, mn) in enumerate(combos):
                ax = ax_flat[idx]
                sub = sub_bs[(sub_bs["input_len"] == il) & (sub_bs["model_name"] == mn)]
                agg = sub.groupby("sm_clock", as_index=False)[col].mean().sort_values("sm_clock")

                sm_labels = [str(int(s)) for s in agg["sm_clock"]]
                vals = agg[col].values
                x_pos = np.arange(len(sm_labels))

                colors = _bar_colors(mn, vals, agg["sm_clock"].values)
                bars = ax.bar(x_pos, vals, color=colors, edgecolor="white", width=0.7)

                fmt = LABEL_FMT.get(col, ".5f")
                for bar, v in zip(bars, vals):
                    if np.isfinite(v):
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height(),
                            f"{v:{fmt}}",
                            ha="center", va="bottom", fontsize=7, rotation=45,
                            fontweight="bold",
                        )

                ax.set_xticks(x_pos)
                ax.set_xticklabels(sm_labels, rotation=45, ha="right", fontsize=8, fontweight="bold")
                ax.set_xlabel("SM Clock (MHz)", fontweight="bold")
                ax.set_ylabel(ylabel, fontweight="bold")
                ax.set_title(f"{mn}  input_len={int(il)}", fontsize=10, fontweight="bold")
                ax.tick_params(axis="y", labelsize=8)
                for label in ax.get_yticklabels():
                    label.set_fontweight("bold")
                ax.grid(axis="y", alpha=0.3)


            for i in range(len(combos), len(ax_flat)):
                ax_flat[i].set_visible(False)

            fig.suptitle(f"{title}  (batch_size={int(bs)})", fontsize=13, fontweight="bold", y=1.01)
            fig.tight_layout()

            fname = f"hist_{sanitize(col)}_bs{int(bs)}.png"
            fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  저장: {output_dir / fname}")


# ── CSV 저장 ──────────────────────────────────────────
def save_metric_csvs(metrics: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for col, _, _ in METRIC_COLS:
        out = metrics[DATA_KEY + [col]].copy()
        path = output_dir / f"{col}.csv"
        out.to_csv(path, index=False)
        print(f"  CSV 저장: {path}")


# ── main ──────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="gpu_profile_*.csv 파일들로 6가지 메트릭 히스토그램을 생성한다."
    )
    parser.add_argument(
        "log_dir",
        type=str,
        help="gpu_profile_*.csv 파일이 있는 디렉터리 경로",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="출력 디렉터리 (기본값: <log_dir>/analysis_6metrics)",
    )
    parser.add_argument(
        "--min",
        type=int,
        default=0,
        help="data 그룹 내 iteration 수가 이 값 이하이면 drop (기본값: 0, drop 안 함)",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        raise NotADirectoryError(f"디렉터리가 아닙니다: {log_dir}")
    output_dir = Path(args.output_dir) if args.output_dir else (log_dir / "analysis_6metrics")

    print("1. CSV 로드...")
    df = load_csvs(log_dir)
    print(f"   전체 행: {len(df):,}")

    print("2. iteration 단위 중간 집계...")
    it = compute_iteration_stats(df)
    print(f"   iteration 수: {len(it):,}")

    print(f"3. data 단위 최종 메트릭 집계 (min_iterations={args.min})...")
    metrics = aggregate_metrics(it, min_iterations=args.min)
    print(f"   data 수: {len(metrics):,}")
    print(metrics.to_string(index=False))

    output_dir.mkdir(parents=True, exist_ok=True)
    it.to_csv(output_dir / "iteration_stats.csv", index=False)

    print("\n4. 메트릭 CSV 저장...")
    save_metric_csvs(metrics, output_dir)

    print("\n5. 히스토그램 플롯 생성...")
    plot_histograms(metrics, output_dir)

    print("\n완료!")


if __name__ == "__main__":
    main()
