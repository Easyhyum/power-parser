"""
지정 폴더 내 gpu_profile_*.csv 파일들을 로드하여
6가지 메트릭을 계산하고, 각각 CSV로 저장한 뒤 sm_clock별 히스토그램을 그린다.

사용법:
  python plot_6metrics_histogram.py <log_dir> [--output-dir <out>] [--line-sm-clock-xmax 2500]

메트릭:
  1) total_energy_based_j_per_token
  2) power_based_j_per_token
  3) latency  (sec/token)
  4) throughput  (tokens/sec)
  5) total_energy_based_avg_power  (W)
  6) power_based_avg_power  (W)
  7) gpu_util_pct, memory_util_pct, avg_temperature_C (°C, saturated 구간 평균)
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_KEY = ["cudagraph_mode", "batch_size", "target_sm_clock", "input_len", "model_name"]
ITER_KEY = DATA_KEY + ["kv_cache_lens", "iteration"]


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
        "total_energy", "gpu_util", "memory_util", "iteration",
        "temperature",
    ]
    cols_ok = [c for c in num_cols if c in df.columns]
    if cols_ok:
        df[cols_ok] = df[cols_ok].apply(pd.to_numeric, errors="coerce")

    if "target_sm_clock" not in df.columns:
        df["target_sm_clock"] = df["sm_clock"]
    else:
        df["target_sm_clock"] = pd.to_numeric(df["target_sm_clock"], errors="coerce")

    return df


# ── iteration 단위 중간 집계 ──────────────────────────
def compute_iteration_stats(df: pd.DataFrame, start_idx: int = 1) -> pd.DataFrame:
    """
    iteration = (batch_size, target_sm_clock, input_len, model_name, kv_cache_lens, iteration) 그룹.
    각 iteration에 대해:
      - delta_total_energy_mJ : total_energy[index==length] - total_energy[index==start_idx]
      - decoding_tokens       : repeat_count * batch_size
      - during_time           : 해당 iteration의 during_time (첫 행 값, 상수)
      - avg_power_saturated   : index/length >= 0.5 인 행의 평균 power
      - energy_power_based_J  : avg_power_saturated * during_time
    """
    rows = []
    for keys, grp in df.groupby(ITER_KEY, sort=False):
        grp_sorted = grp.sort_values("index")
        length_val = grp_sorted["length"].iloc[0]
        idx_start = grp_sorted[grp_sorted["index"] == start_idx]
        idx_max = grp_sorted[grp_sorted["index"] == length_val]

        if idx_start.empty or idx_max.empty:
            continue

        te_start = idx_start["total_energy"].iloc[0]
        te_end = idx_max["total_energy"].iloc[0]
        delta_te_mJ = te_end - te_start

        if delta_te_mJ <= 0:
            continue

        during = grp_sorted["during_time"].iloc[0]
        repeat = grp_sorted["repeat_count"].iloc[0]
        bs = grp_sorted["batch_size"].iloc[0]
        model = grp_sorted["model_name"].iloc[0]
        inp_len = grp_sorted["input_len"].iloc[0]
        tokens = repeat * bs * inp_len if "prefill" in str(model).lower() else repeat * bs

        grp_sorted = grp_sorted.copy()
        grp_sorted["idx_ratio"] = grp_sorted["index"] / grp_sorted["length"]
        saturated = grp_sorted[grp_sorted["idx_ratio"] > 0.5]
        avg_pwr = saturated["power"].mean() if not saturated.empty else np.nan
        measured_sm = int(saturated["sm_clock"].mean()) if not saturated.empty else int(grp_sorted["sm_clock"].mean())
        avg_gpu_util = saturated["gpu_util"].mean() if not saturated.empty else np.nan
        avg_mem_util = saturated["memory_util"].mean() if not saturated.empty else np.nan
        if "temperature" in grp_sorted.columns and not saturated.empty:
            avg_temp = saturated["temperature"].mean()
        else:
            avg_temp = np.nan

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
        row["avg_temperature"] = avg_temp
        rows.append(row)

    return pd.DataFrame(rows)


# ── data 단위 최종 메트릭 집계 ────────────────────────
def aggregate_metrics(it: pd.DataFrame) -> pd.DataFrame:
    """
    data = (batch_size, sm_clock, input_len, model_name) 그룹.
    """
    records = []
    for keys, grp in it.groupby(DATA_KEY, sort=False):
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

        # (9) temperature (°C, saturated 구간 평균의 data 그룹 평균)
        temp_c = grp["avg_temperature"].mean()

        # measured sm_clock (actual average)
        measured_sm = int(grp["sm_clock"].mean()) if "sm_clock" in grp.columns else np.nan

        row = dict(zip(DATA_KEY, keys))
        row["sm_clock"] = measured_sm
        row["total_energy_based_j_per_token"] = te_j_per_tok
        row["power_based_j_per_token"] = pwr_j_per_tok
        row["latency_sec_per_token"] = latency
        row["throughput_tokens_per_sec"] = throughput
        row["total_energy_based_avg_power_W"] = te_avg_power
        row["power_based_avg_power_W"] = pwr_avg_power
        row["gpu_util_pct"] = gpu_util
        row["memory_util_pct"] = mem_util
        row["avg_temperature_C"] = temp_c
        records.append(row)

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
    ("avg_temperature_C", "Temperature", "°C"),
]

LABEL_FMT = {
    "total_energy_based_avg_power_W": ".2f",
    "power_based_avg_power_W": ".2f",
    "gpu_util_pct": ".1f",
    "memory_util_pct": ".1f",
    "throughput_tokens_per_sec": ".2f",
    "avg_temperature_C": ".1f",
}

SHARED_YLIM_COLS = {"memory_util_pct", "avg_temperature_C"}


def _hist_combo_sort_key(combo: tuple) -> tuple:
    """히스토그램 서브플롯 순서: input_len → prefill/decoding → attn/mlp."""
    il, mn = combo
    mn_lower = str(mn).lower()
    phase = 0 if "prefill" in mn_lower else (1 if "decoding" in mn_lower else 2)
    comp = 0 if "attn" in mn_lower else (1 if "mlp" in mn_lower else 2)
    return (int(il), phase, comp, str(mn))


MAX_COLOR = "#222222"
# 히스토그램: prefill=빨강 계열, decoding=파랑 계열, max sm_clock=검정 유지
HIST_PREFILL_LIGHT = "#FFCDD2"
HIST_PREFILL_DARK = "#C62828"
HIST_DECODE_LIGHT = "#BBDEFB"
HIST_DECODE_DARK = "#1565C0"
HIST_NEUTRAL_LIGHT = "#ECEFF1"
HIST_NEUTRAL_DARK = "#607D8B"


def _hist_bar_phase_colors(model_name: str) -> tuple[str, str]:
    """(min용 진한색, 나머지 막대 연한색). prefill/decoding 기준."""
    s = str(model_name).lower()
    if "prefill" in s:
        return HIST_PREFILL_DARK, HIST_PREFILL_LIGHT
    if "decoding" in s:
        return HIST_DECODE_DARK, HIST_DECODE_LIGHT
    return HIST_NEUTRAL_DARK, HIST_NEUTRAL_LIGHT


def _bar_colors(model_name: str, vals: np.ndarray, sm_clocks: np.ndarray) -> list[str]:
    """max sm_clock → 검정, min 값 → phase별 진한색, 나머지 → phase별 연한색."""
    dark, light = _hist_bar_phase_colors(model_name)
    finite_mask = np.isfinite(vals)
    if not finite_mask.any():
        return [light] * len(vals)
    max_sm_idx = int(np.argmax(sm_clocks))
    min_val_idx = int(np.nanargmin(vals))
    colors = [light] * len(vals)
    colors[max_sm_idx] = MAX_COLOR
    if min_val_idx != max_sm_idx:
        colors[min_val_idx] = dark
    return colors


def _plot_one(metrics: pd.DataFrame, output_dir: Path, mode: str) -> None:
    """mode: 'raw' 또는 'norm' (max target_sm_clock 기준 정규화)."""
    normalize = mode == "norm"

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
            combos.sort(key=_hist_combo_sort_key)

            if not combos:
                continue

            ncols = min(len(combos), 2)
            nrows = math.ceil(len(combos) / ncols)
            fig_w = 6.5 * ncols + 1
            fig_h = 4.5 * nrows + 1
            fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
            ax_flat = axes.ravel()

            for idx, (il, mn) in enumerate(combos):
                ax = ax_flat[idx]
                sub = sub_bs[(sub_bs["input_len"] == il) & (sub_bs["model_name"] == mn)]
                agg = sub.groupby("target_sm_clock", as_index=False).agg(
                    val=(col, "mean"), sm_clock=("sm_clock", "mean")
                ).sort_values("target_sm_clock")

                sm_labels = [str(int(s)) for s in agg["sm_clock"]]
                vals = agg["val"].values.copy()

                if normalize:
                    max_tsm_idx = int(np.argmax(agg["target_sm_clock"].values))
                    ref_val = vals[max_tsm_idx]
                    if ref_val and np.isfinite(ref_val) and ref_val != 0:
                        vals = vals / ref_val
                    else:
                        vals = np.full_like(vals, np.nan)

                x_pos = np.arange(len(sm_labels))
                colors = _bar_colors(mn, vals, agg["sm_clock"].values)
                bars = ax.bar(x_pos, vals, color=colors, edgecolor="white", width=0.7)

                fmt = ".3f" if normalize else LABEL_FMT.get(col, ".5f")
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
                ax.set_ylabel("ratio (max_sm=1)" if normalize else ylabel, fontweight="bold")
                ax.set_title(f"{mn}  input_len={int(il)}", fontsize=10, fontweight="bold")
                ax.tick_params(axis="y", labelsize=8)
                for label in ax.get_yticklabels():
                    label.set_fontweight("bold")
                ax.grid(axis="y", alpha=0.3)

            if normalize or col in SHARED_YLIM_COLS:
                y_max = max(ax_flat[i].get_ylim()[1] for i in range(len(combos)))
                for i in range(len(combos)):
                    ax_flat[i].set_ylim(0, y_max)

            for i in range(len(combos), len(ax_flat)):
                ax_flat[i].set_visible(False)

            suffix = " [normalized]" if normalize else ""
            fig.suptitle(f"{title}{suffix}  (batch_size={int(bs)})", fontsize=13, fontweight="bold", y=1.01)
            fig.tight_layout()

            tag = "norm" if normalize else "raw"
            fname = f"hist_{sanitize(col)}_bs{int(bs)}_{tag}.png"
            fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  저장: {output_dir / fname}")


COMPARE_SM_COLS = {
    "memory_util_pct",
    "throughput_tokens_per_sec",
    "power_based_avg_power_W",
    "total_energy_based_avg_power_W",
    "avg_temperature_C",
}

SM_RATIO_COLOR = "#AAAAAA"
DIFF_COLOR = "#FFCDD2"


def _plot_compare_sm(metrics: pd.DataFrame, output_dir: Path) -> None:
    """metric(normalized)와 sm_clock/max_sm_clock 비율을 나란히 비교하는 PNG."""
    batch_sizes = sorted(metrics["batch_size"].dropna().unique())
    input_lens = sorted(metrics["input_len"].dropna().unique())
    model_names = sorted(metrics["model_name"].dropna().unique())

    for col, title, ylabel in METRIC_COLS:
        if col not in COMPARE_SM_COLS:
            continue
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

            combos.sort(key=_hist_combo_sort_key)

            if not combos:
                continue

            ncols = min(len(combos), 2)
            nrows = math.ceil(len(combos) / ncols)
            fig_w = 7.5 * ncols + 1
            fig_h = 4.5 * nrows + 1
            fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
            ax_flat = axes.ravel()

            bar_w = 0.25

            for idx, (il, mn) in enumerate(combos):
                ax = ax_flat[idx]
                sub = sub_bs[(sub_bs["input_len"] == il) & (sub_bs["model_name"] == mn)]
                agg = sub.groupby("target_sm_clock", as_index=False).agg(
                    val=(col, "mean"), sm_clock=("sm_clock", "mean")
                ).sort_values("target_sm_clock")

                sm_labels = [str(int(s)) for s in agg["sm_clock"]]
                vals = agg["val"].values.copy()
                sm_vals = agg["sm_clock"].values.copy()

                max_tsm_idx = int(np.argmax(agg["target_sm_clock"].values))
                ref_val = vals[max_tsm_idx]
                max_sm = sm_vals[max_tsm_idx]

                if ref_val and np.isfinite(ref_val) and ref_val != 0:
                    metric_norm = vals / ref_val
                else:
                    metric_norm = np.full_like(vals, np.nan)

                sm_norm = sm_vals / max_sm if max_sm else np.full_like(sm_vals, np.nan)
                diff = metric_norm - sm_norm

                x_pos = np.arange(len(sm_labels))

                colors = _bar_colors(mn, metric_norm, sm_vals)
                bars_m = ax.bar(x_pos - bar_w, metric_norm, bar_w,
                                color=colors, edgecolor="white", label=title)
                bars_s = ax.bar(x_pos, sm_norm, bar_w,
                                color=SM_RATIO_COLOR, edgecolor="white", label="SM Clock ratio")
                bars_d = ax.bar(x_pos + bar_w, diff, bar_w,
                                color=DIFF_COLOR, edgecolor="white", label="Difference")

                for bar, v in zip(bars_m, metric_norm):
                    if np.isfinite(v):
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                                f"{v:.3f}", ha="center", va="bottom", fontsize=5,
                                rotation=45, fontweight="bold")
                for bar, v in zip(bars_s, sm_norm):
                    if np.isfinite(v):
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                                f"{v:.3f}", ha="center", va="bottom", fontsize=5,
                                rotation=45, fontweight="bold", color="#666666")
                for bar, v in zip(bars_d, diff):
                    if np.isfinite(v):
                        y = bar.get_height() if v >= 0 else 0
                        va = "bottom" if v >= 0 else "top"
                        ax.text(bar.get_x() + bar.get_width() / 2, y,
                                f"{v:+.3f}", ha="center", va=va, fontsize=5,
                                rotation=45, fontweight="bold", color="#C62828")

                ax.set_xticks(x_pos)
                ax.set_xticklabels(sm_labels, rotation=45, ha="right", fontsize=8, fontweight="bold")
                ax.set_xlabel("SM Clock (MHz)", fontweight="bold")
                ax.set_ylabel("ratio (max_sm=1)", fontweight="bold")
                ax.set_title(f"{mn}  input_len={int(il)}", fontsize=10, fontweight="bold")
                ax.tick_params(axis="y", labelsize=8)
                for label in ax.get_yticklabels():
                    label.set_fontweight("bold")
                ax.grid(axis="y", alpha=0.3)
                if idx == 0:
                    ax.legend(fontsize=7, loc="upper left")

            y_max = max(ax_flat[i].get_ylim()[1] for i in range(len(combos)))
            for i in range(len(combos)):
                ax_flat[i].set_ylim(0, y_max)

            for i in range(len(combos), len(ax_flat)):
                ax_flat[i].set_visible(False)

            fig.suptitle(f"{title} vs SM Clock ratio  (batch_size={int(bs)})",
                         fontsize=13, fontweight="bold", y=1.01)
            fig.tight_layout()

            fname = f"hist_{sanitize(col)}_bs{int(bs)}_compare.png"
            fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  저장: {output_dir / fname}")


def plot_histograms(metrics: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_one(metrics, output_dir, mode="raw")
    _plot_one(metrics, output_dir, mode="norm")
    _plot_compare_sm(metrics, output_dir)


# ── SM clock 라인 차트 (analysis_6metrics_line) ─────────
LINE_Y_METRICS = [
    ("total_energy_based_j_per_token", "Energy/token (total energy)", "J/token"),
    ("power_based_j_per_token", "Energy/token (power-based)", "J/token"),
    ("memory_util_pct", "Memory Utilization", "%"),
    ("power_based_avg_power_W", "Avg Power (power-based)", "W"),
    ("throughput_tokens_per_sec", "Throughput", "tokens/sec"),
    ("avg_temperature_C", "Temperature", "°C"),
]

COLOR_PREFILL = "#C62828"
COLOR_DECODING = "#1565C0"


def _line_infer_phase(model_name: str) -> str:
    s = str(model_name).lower()
    if "prefill" in s:
        return "prefill"
    if "decoding" in s:
        return "decoding"
    return "other"


def _line_infer_component(model_name: str) -> str:
    """attn / mlp / other (attn·mlp 동시 포함은 mlp로 분류)."""
    s = str(model_name).lower()
    if "mlp" in s:
        return "mlp"
    if "attn" in s:
        return "attn"
    return "other"


def _should_split_attn_mlp(metrics: pd.DataFrame) -> bool:
    comps = metrics["model_name"].map(_line_infer_component)
    return (comps == "attn").any() and (comps == "mlp").any()


def _apply_sm_clock_xlim(fig, xmax: float) -> None:
    """sm_clock x축: figure 내 보이는 모든 axes에 xlim (0, xmax)."""
    if not math.isfinite(xmax) or xmax <= 0:
        return
    for ax in fig.axes:
        if ax.get_visible():
            ax.set_xlim(0.0, xmax)


INPUT_LEN_X_MAX = 10_000


def _apply_input_len_xlim_10k(fig) -> None:
    """input_len x축 라인 차트: figure 내 보이는 axes xlim (0, 10K) 고정."""
    for ax in fig.axes:
        if ax.get_visible():
            ax.set_xlim(0.0, float(INPUT_LEN_X_MAX))


def _unify_batch_size_xlim(fig, df: pd.DataFrame) -> None:
    """batch_size x축: figure 내 보이는 axes xlim을 데이터 batch_size 범위(+여백)로 통일."""
    if "batch_size" not in df.columns:
        return
    xs = pd.to_numeric(df["batch_size"], errors="coerce").dropna()
    if xs.empty:
        return
    lo, hi = float(xs.min()), float(xs.max())
    pad = max(1.0, (hi - lo) * 0.05) if hi > lo else max(1.0, abs(lo) * 0.1)
    lo2 = max(0.0, lo - pad)
    hi2 = hi + pad
    for ax in fig.axes:
        if ax.get_visible():
            ax.set_xlim(lo2, hi2)


def _input_len_alphas(input_lens: list[int]) -> dict[int, float]:
    ils = sorted(set(int(x) for x in input_lens))
    if not ils:
        return {}
    n = len(ils)
    if n == 1:
        return {ils[0]: 1.0}
    lo, hi = 0.35, 1.0
    return {il: lo + (hi - lo) * i / (n - 1) for i, il in enumerate(ils)}


def _plot_line_axes(
    ax,
    sub: pd.DataFrame,
    ycol: str,
    ytitle: str,
    yunit: str,
    alpha_map: dict[int, float],
    phase_filter: str,
    x_col: str = "sm_clock",
    x_axis_label: str = "GPU frequency (SM clock, MHz)",
    y_log_scale: bool = False,
) -> None:
    """
    sub: metrics subset (단일 component). phase_filter는 "prefill" 또는 "decoding".
    x_col: "sm_clock" | "memory_util_pct" (0~100) | "input_len" (0~10K) | "batch_size".
    input_len/batch_size x: target_sm_clock 기준 집계, 범례 avg(sm_clock)=….
    y_log_scale: True면 y>0인 점만 이어 그리고 log y (0 이하·비유한은 구간 끊김).
    sm_clock/memory: 범례=input_len. 패널마다 독립 y축.
    """
    if sub.empty:
        ax.set_visible(False)
        return

    work = sub.copy()
    work["_phase"] = work["model_name"].map(_line_infer_phase)
    work = work[work["_phase"] == phase_filter].copy()

    if work.empty:
        ax.text(
            0.5, 0.5, f"No {phase_filter} rows", ha="center", va="center",
            transform=ax.transAxes,
        )
        if x_col == "memory_util_pct":
            ax.set_xlim(0, 100)
        elif x_col == "input_len":
            ax.set_xlim(0, INPUT_LEN_X_MAX)
        return

    if x_col not in work.columns:
        ax.text(
            0.5, 0.5, f"No {x_col} column", ha="center", va="center",
            transform=ax.transAxes,
        )
        if x_col == "memory_util_pct":
            ax.set_xlim(0, 100)
        elif x_col == "input_len":
            ax.set_xlim(0, INPUT_LEN_X_MAX)
        return

    ylabel = f"{ytitle} ({yunit})" if yunit else ytitle

    if x_col == "batch_size":
        if "target_sm_clock" not in work.columns:
            ax.text(
                0.5, 0.5, "No target_sm_clock column",
                ha="center", va="center", transform=ax.transAxes,
            )
            return
        w = work.dropna(subset=["batch_size", "target_sm_clock", "sm_clock", ycol])
        if w.empty:
            ax.text(
                0.5, 0.5, "No batch_size / target_sm_clock data",
                ha="center", va="center", transform=ax.transAxes,
            )
            return
        g = (
            w.groupby(["target_sm_clock", "batch_size"], as_index=False)
            .agg({ycol: "mean", "sm_clock": "mean"})
            .sort_values("batch_size")
        )
        cmap = plt.get_cmap("tab10")
        tgt_groups = sorted(
            g.groupby("target_sm_clock", sort=False), key=lambda kv: float(kv[0])
        )
        for i, (_tgt, grp) in enumerate(tgt_groups):
            grp = grp.sort_values("batch_size")
            x = grp["batch_size"].values.astype(float)
            y = grp[ycol].values.astype(float)
            if y_log_scale:
                y = np.where((y > 0) & np.isfinite(y), y, np.nan)
            if len(x) < 1:
                continue
            color = cmap(i % 10)
            lw = 1.5
            avg_sm = float(grp["sm_clock"].mean())
            lbl = int(round(avg_sm)) if math.isfinite(avg_sm) else 0
            ax.plot(
                x, y, marker="o", markersize=4, linewidth=lw, color=color,
                label=f"avg(sm_clock)={lbl}",
            )
    elif x_col == "input_len":
        if "target_sm_clock" not in work.columns:
            ax.text(
                0.5, 0.5, "No target_sm_clock column",
                ha="center", va="center", transform=ax.transAxes,
            )
            ax.set_xlim(0, INPUT_LEN_X_MAX)
            return
        w = work.dropna(subset=["input_len", "target_sm_clock", "sm_clock", ycol])
        if w.empty:
            ax.text(
                0.5, 0.5, "No input_len / target_sm_clock data",
                ha="center", va="center", transform=ax.transAxes,
            )
            ax.set_xlim(0, INPUT_LEN_X_MAX)
            return
        g = (
            w.groupby(["target_sm_clock", "input_len"], as_index=False)
            .agg({ycol: "mean", "sm_clock": "mean"})
            .sort_values("input_len")
        )
        cmap = plt.get_cmap("tab10")
        tgt_groups = sorted(
            g.groupby("target_sm_clock", sort=False), key=lambda kv: float(kv[0])
        )
        for i, (_tgt, grp) in enumerate(tgt_groups):
            grp = grp.sort_values("input_len")
            x = grp["input_len"].values.astype(float)
            y = grp[ycol].values.astype(float)
            if y_log_scale:
                y = np.where((y > 0) & np.isfinite(y), y, np.nan)
            if len(x) < 1:
                continue
            color = cmap(i % 10)
            lw = 1.5
            avg_sm = float(grp["sm_clock"].mean())
            lbl = int(round(avg_sm)) if math.isfinite(avg_sm) else 0
            ax.plot(
                x, y, marker="o", markersize=4, linewidth=lw, color=color,
                label=f"avg(sm_clock)={lbl}",
            )
    elif x_col == "sm_clock":
        g = (
            work.groupby(["input_len", "sm_clock"], as_index=False)[ycol]
            .mean()
            .sort_values("sm_clock")
        )
        sort_col = "sm_clock"
        grouped = {int(il): grp for il, grp in g.groupby("input_len", sort=False)}
        sorted_ils = sorted(grouped.keys())
        color = COLOR_PREFILL if phase_filter == "prefill" else COLOR_DECODING
        for il in sorted_ils:
            grp = grouped[il].sort_values(sort_col)
            x = grp[x_col].values.astype(float)
            y = grp[ycol].values.astype(float)
            if y_log_scale:
                y = np.where((y > 0) & np.isfinite(y), y, np.nan)
            if len(x) < 1:
                continue
            alpha = alpha_map.get(il, 0.7)
            lw = 1.2 + 0.8 * alpha
            ax.plot(
                x, y, marker="o", markersize=4, linewidth=lw, color=color, alpha=alpha,
                label=f"input_len={il}",
            )
    elif x_col == "memory_util_pct":
        w = work.dropna(subset=["memory_util_pct", ycol])
        if w.empty:
            ax.text(0.5, 0.5, "No memory_util data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlim(0, 100)
            return
        g = (
            w.groupby(["input_len", "memory_util_pct"], as_index=False)[ycol]
            .mean()
            .sort_values("memory_util_pct")
        )
        sort_col = "memory_util_pct"
        grouped = {int(il): grp for il, grp in g.groupby("input_len", sort=False)}
        sorted_ils = sorted(grouped.keys())
        color = COLOR_PREFILL if phase_filter == "prefill" else COLOR_DECODING
        for il in sorted_ils:
            grp = grouped[il].sort_values(sort_col)
            x = grp[x_col].values.astype(float)
            y = grp[ycol].values.astype(float)
            if y_log_scale:
                y = np.where((y > 0) & np.isfinite(y), y, np.nan)
            if len(x) < 1:
                continue
            alpha = alpha_map.get(il, 0.7)
            lw = 1.2 + 0.8 * alpha
            ax.plot(
                x, y, marker="o", markersize=4, linewidth=lw, color=color, alpha=alpha,
                label=f"input_len={il}",
            )
    else:
        ax.text(
            0.5, 0.5, f"Unknown x_col: {x_col}", ha="center", va="center",
            transform=ax.transAxes,
        )
        return

    ax.set_xlabel(x_axis_label, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    if y_log_scale:
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
    else:
        ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=8)
    for lb in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lb.set_fontweight("bold")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=6, loc="best", ncol=2)

    if x_col == "memory_util_pct":
        ax.set_xlim(0, 100)
    elif x_col == "input_len":
        ax.set_xlim(0, INPUT_LEN_X_MAX)


def _plot_line_bundle(
    metrics: pd.DataFrame,
    line_dir: Path,
    batch_sizes: list,
    alpha_map: dict[int, float],
    split: bool,
    x_col: str,
    x_axis_label: str,
    fname_suffix: str,
    supt_mid: str,
    y_log_scale: bool = False,
    line_sm_clock_xmax: float = 2500.0,
) -> None:
    """한 종류의 x축에 대해 LINE_Y_METRICS 전부 저장."""
    if x_col == "batch_size":
        plot_slices: list[tuple[pd.DataFrame, int | None]] = [(metrics, None)]
    else:
        plot_slices = [
            (metrics[metrics["batch_size"] == bs], bs)
            for bs in batch_sizes
            if not metrics[metrics["batch_size"] == bs].empty
        ]

    for ycol, ytitle, yunit in LINE_Y_METRICS:
        if x_col == "memory_util_pct" and ycol == "memory_util_pct":
            continue
        for sub_bs, bs in plot_slices:
            if sub_bs.empty:
                continue

            if bs is None:
                supt = f"{ytitle} {supt_mid}  (all batch_size)"
            else:
                supt = f"{ytitle} {supt_mid}  (batch_size={int(bs)})"
            if y_log_scale:
                supt += " · y: log"

            if split:
                # 행: prefill → decoding, 열: attn → mlp
                fig, axes = plt.subplots(2, 2, figsize=(14, 9), squeeze=False)
                sub_a = sub_bs[sub_bs["model_name"].map(_line_infer_component) == "attn"]
                sub_m = sub_bs[sub_bs["model_name"].map(_line_infer_component) == "mlp"]
                _plot_line_axes(
                    axes[0, 0], sub_a, ycol, ytitle, yunit, alpha_map, "prefill",
                    x_col=x_col, x_axis_label=x_axis_label, y_log_scale=y_log_scale,
                )
                _plot_line_axes(
                    axes[0, 1], sub_m, ycol, ytitle, yunit, alpha_map, "prefill",
                    x_col=x_col, x_axis_label=x_axis_label, y_log_scale=y_log_scale,
                )
                _plot_line_axes(
                    axes[1, 0], sub_a, ycol, ytitle, yunit, alpha_map, "decoding",
                    x_col=x_col, x_axis_label=x_axis_label, y_log_scale=y_log_scale,
                )
                _plot_line_axes(
                    axes[1, 1], sub_m, ycol, ytitle, yunit, alpha_map, "decoding",
                    x_col=x_col, x_axis_label=x_axis_label, y_log_scale=y_log_scale,
                )
                axes[0, 0].set_title("prefill · attn", fontsize=11, fontweight="bold")
                axes[0, 1].set_title("prefill · mlp", fontsize=11, fontweight="bold")
                axes[1, 0].set_title("decoding · attn", fontsize=11, fontweight="bold")
                axes[1, 1].set_title("decoding · mlp", fontsize=11, fontweight="bold")
                fig.suptitle(supt, fontsize=13, fontweight="bold", y=1.01)
            else:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5), squeeze=False)
                ax0, ax1 = axes[0, 0], axes[0, 1]
                _plot_line_axes(
                    ax0, sub_bs, ycol, ytitle, yunit, alpha_map, "prefill",
                    x_col=x_col, x_axis_label=x_axis_label, y_log_scale=y_log_scale,
                )
                _plot_line_axes(
                    ax1, sub_bs, ycol, ytitle, yunit, alpha_map, "decoding",
                    x_col=x_col, x_axis_label=x_axis_label, y_log_scale=y_log_scale,
                )
                ax0.set_title("prefill", fontsize=11, fontweight="bold")
                ax1.set_title("decoding", fontsize=11, fontweight="bold")
                fig.suptitle(supt, fontsize=13, fontweight="bold", y=1.02)

            if x_col == "sm_clock":
                _apply_sm_clock_xlim(fig, line_sm_clock_xmax)
            elif x_col == "input_len":
                _apply_input_len_xlim_10k(fig)
            elif x_col == "batch_size":
                _unify_batch_size_xlim(fig, sub_bs)

            fig.tight_layout()
            if x_col == "batch_size":
                fname = f"line_{sanitize(ycol)}{fname_suffix}.png"
            else:
                fname = f"line_{sanitize(ycol)}_bs{int(bs)}{fname_suffix}.png"
            fig.savefig(line_dir / fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  저장 (line): {line_dir / fname}")


TP_COL = "throughput_tokens_per_sec"
PWR_COL = "power_based_avg_power_W"
EPT_COL = "power_based_j_per_token"

COMBO_COLOR_THROUGHPUT = "#1565C0"
COMBO_COLOR_POWER = "#C62828"
COMBO_COLOR_ENERGY = "#424242"

COMBO_NORM_YLIM = (0.3, 1.1)


def _set_combo_ept_ylim(ept_ax, y_ept: np.ndarray) -> None:
    """Energy/token y축: 양수 max가 있으면 [0.3*max, 1.1*max], 그 외는 데이터 min~max."""
    ye = np.asarray(y_ept, dtype=float)
    ye = ye[np.isfinite(ye)]
    if not ye.size:
        return
    ept_hi = float(np.max(ye))
    ept_lo = float(np.min(ye))
    if ept_hi > 0:
        ept_ax.set_ylim(ept_hi * 0.3, ept_hi * 1.1)
    elif ept_hi == 0 and ept_lo == 0:
        ept_ax.set_ylim(-0.05, 0.05)
    else:
        ept_ax.set_ylim(ept_lo, ept_hi)


def _hide_combo_power_y_axis(ax3) -> None:
    """3메트릭: power twin의 y축 눈금·라벨·오른쪽 spine 제거."""
    ax3.set_ylabel("")
    ax3.set_yticks([])
    ax3.tick_params(axis="y", which="both", left=False, right=False, labelleft=False, labelright=False)
    if "right" in ax3.spines:
        ax3.spines["right"].set_visible(False)


def _normalize_by_value_at_max_target_sm_clock(x_target: np.ndarray, y: np.ndarray) -> np.ndarray:
    """각 metric별로 target_sm_clock이 최대인 그룹의 y를 1로 두는 비율."""
    x_target = np.asarray(x_target, dtype=float)
    y = np.asarray(y, dtype=float)
    if x_target.size == 0:
        return y
    j = int(np.nanargmax(x_target))
    ref = float(y[j])
    if not math.isfinite(ref) or ref == 0:
        return np.full_like(y, np.nan, dtype=float)
    return y.astype(float) / ref


def _annotate_line_points(ax, x: np.ndarray, y: np.ndarray, fmt: str) -> None:
    """마커 옆에 수치 표기 (raw 플롯용)."""
    for xi, yi in zip(np.asarray(x, dtype=float), np.asarray(y, dtype=float)):
        if not (math.isfinite(xi) and math.isfinite(yi)):
            continue
        ax.annotate(
            f"{yi:{fmt}}",
            (xi, yi),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=4,
            color="#333333",
        )


def _combo_panel_has_phase_component_rows(
    metrics: pd.DataFrame, phase: str, component: str | None
) -> bool:
    """콤보 그리드용: 해당 phase(및 선택 시 attn/mlp)에 target/sm_clock 유효 행이 있는지."""
    work = metrics.copy()
    work["_phase"] = work["model_name"].map(_line_infer_phase)
    work = work[work["_phase"] == phase]
    if component is not None:
        work = work[work["model_name"].map(_line_infer_component) == component]
    return not work.dropna(subset=["target_sm_clock", "sm_clock"]).empty


def _plot_sm_clock_throughput_avg_power_grid(
    metrics: pd.DataFrame,
    line_dir: Path,
    line_sm_clock_xmax: float,
) -> None:
    """
    x = 그룹별 평균 sm_clock (그룹 키는 target_sm_clock).
    3메트릭(TP+power+EPT)과 2메트릭(TP+power) figure를 각각 저장; 2메트릭 파일명에 _tp_pw.
    서브플롯 = (batch_size, input_len). phase(prefill/decoding)별 raw·norm.
    attn·mlp가 데이터에 둘 다 있으면 phase별 전체 합산 + _attn + _mlp PNG.
    normalized: TP·power는 ylim (0.3, 1.1) 통일; energy/token은 패널별 [0.3·max, 1.1·max].
    raw: E/T 축은 [0.3·max, 1.1·max](양수 max).     3메트릭: E/T는 안쪽 오른쪽 축(구 power 자리); power는 바깥 twin에 곡선만 표시(y축 미표시).
    power ylim은 표시가 있을 때와 동일: norm은 COMBO_NORM_YLIM, raw는 matplotlib autoscale(y).
    """
    need = {"batch_size", "input_len", "target_sm_clock", "sm_clock", "model_name", TP_COL, PWR_COL, EPT_COL}
    if not need <= set(metrics.columns):
        print("  (line combo) 스킵: 필요한 컬럼이 없습니다.")
        return

    pairs_df = metrics.dropna(subset=["batch_size", "input_len"])[["batch_size", "input_len"]].drop_duplicates()
    pairs_df = pairs_df.sort_values(["batch_size", "input_len"])
    if pairs_df.empty:
        return
    pairs = [(int(row.batch_size), int(row.input_len)) for row in pairs_df.itertuples(index=False)]
    n = len(pairs)
    ncols = min(4, max(2, int(math.ceil(math.sqrt(n)))))
    nrows = int(math.ceil(n / ncols))
    fig_w = 4.8 * ncols + 1.2
    fig_h = 3.4 * nrows + 1.2
    line_dir.mkdir(parents=True, exist_ok=True)

    split_am = _should_split_attn_mlp(metrics)

    for phase in ("prefill", "decoding"):
        variants: list[tuple[str | None, str]] = [(None, "")]
        if split_am:
            if _combo_panel_has_phase_component_rows(metrics, phase, "attn"):
                variants.append(("attn", "_attn"))
            if _combo_panel_has_phase_component_rows(metrics, phase, "mlp"):
                variants.append(("mlp", "_mlp"))

        for component, comp_suffix in variants:
            for include_ept in (True, False):
                for normalized in (False, True):
                    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
                    ax_flat = axes.ravel()

                    for idx, (bs, il) in enumerate(pairs):
                        ax = ax_flat[idx]
                        sub = metrics[(metrics["batch_size"] == bs) & (metrics["input_len"] == il)]
                        if sub.empty:
                            ax.set_visible(False)
                            continue
                        work = sub.copy()
                        work["_phase"] = work["model_name"].map(_line_infer_phase)
                        work = work[work["_phase"] == phase].copy()
                        if component is not None:
                            work = work[work["model_name"].map(_line_infer_component) == component].copy()
                        if work.empty:
                            msg = f"No {phase} data" if component is None else f"No {phase} · {component}"
                            ax.text(
                                0.5, 0.5, msg, ha="center", va="center",
                                transform=ax.transAxes,
                            )
                            ax.set_xlim(0, line_sm_clock_xmax)
                            continue
                        w = work.dropna(subset=["target_sm_clock", "sm_clock"])
                        if w.empty:
                            ax.text(
                                0.5, 0.5, "No target/sm_clock data", ha="center", va="center",
                                transform=ax.transAxes,
                            )
                            ax.set_xlim(0, line_sm_clock_xmax)
                            continue
                        g = (
                            w.groupby("target_sm_clock", as_index=False)
                            .agg(
                                {
                                    TP_COL: "mean",
                                    PWR_COL: "mean",
                                    EPT_COL: "mean",
                                    "sm_clock": "mean",
                                }
                            )
                            .sort_values("target_sm_clock")
                        )
                        if g.empty:
                            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                            ax.set_xlim(0, line_sm_clock_xmax)
                            continue

                        ax2 = ax.twinx()
                        ax3 = None
                        if include_ept:
                            ax3 = ax.twinx()
                            ax3.spines["right"].set_position(("outward", 44))
                            ax3.patch.set_visible(False)
                            for sp in ("top", "left", "bottom"):
                                ax3.spines[sp].set_visible(False)

                        x = g["sm_clock"].values.astype(float)
                        x_tgt = g["target_sm_clock"].values.astype(float)
                        y_tp = g[TP_COL].values.astype(float)
                        y_pw = g[PWR_COL].values.astype(float)
                        y_ept = g[EPT_COL].values.astype(float)
                        if normalized:
                            y_tp = _normalize_by_value_at_max_target_sm_clock(x_tgt, y_tp)
                            y_pw = _normalize_by_value_at_max_target_sm_clock(x_tgt, y_pw)
                            y_ept = _normalize_by_value_at_max_target_sm_clock(x_tgt, y_ept)

                        (ln_tp,) = ax.plot(
                            x, y_tp, marker="o", markersize=3, linewidth=1.5,
                            color=COMBO_COLOR_THROUGHPUT, label="throughput",
                        )
                        ln_pw = None
                        ln_ept = None
                        if include_ept and ax3 is not None:
                            (ln_ept,) = ax2.plot(
                                x, y_ept, marker="^", markersize=3, linewidth=1.5,
                                color=COMBO_COLOR_ENERGY, linestyle=":", label="energy/token (J)",
                            )
                            (ln_pw,) = ax3.plot(
                                x, y_pw, marker="s", markersize=3, linewidth=1.5,
                                color=COMBO_COLOR_POWER, linestyle="--", label="avg power (W)",
                            )
                        else:
                            (ln_pw,) = ax2.plot(
                                x, y_pw, marker="s", markersize=3, linewidth=1.5,
                                color=COMBO_COLOR_POWER, linestyle="--", label="avg power (W)",
                            )

                        if normalized:
                            _annotate_line_points(ax, x, y_tp, ".3f")
                            if include_ept and ax3 is not None:
                                _annotate_line_points(ax2, x, y_ept, ".3f")
                                _annotate_line_points(ax3, x, y_pw, ".3f")
                            else:
                                _annotate_line_points(ax2, x, y_pw, ".3f")
                        else:
                            _annotate_line_points(ax, x, y_tp, ".4g")
                            if include_ept and ax3 is not None:
                                _annotate_line_points(ax2, x, y_ept, ".4g")
                                _annotate_line_points(ax3, x, y_pw, ".3f")
                            else:
                                _annotate_line_points(ax2, x, y_pw, ".3f")

                        ax.set_xlabel("SM clock (MHz)", fontweight="bold", fontsize=8)
                        if normalized:
                            ax.set_ylabel(
                                "Throughput (norm.)", fontweight="bold", fontsize=8,
                                color=COMBO_COLOR_THROUGHPUT,
                            )
                            if include_ept and ax3 is not None:
                                ax2.set_ylabel(
                                    "Energy/token (norm.)", fontweight="bold", fontsize=8,
                                    color=COMBO_COLOR_ENERGY,
                                )
                            else:
                                ax2.set_ylabel(
                                    "Avg power (norm.)", fontweight="bold", fontsize=8,
                                    color=COMBO_COLOR_POWER,
                                )
                        else:
                            ax.set_ylabel(
                                "Throughput (tokens/s)", fontweight="bold", fontsize=8,
                                color=COMBO_COLOR_THROUGHPUT,
                            )
                            if include_ept and ax3 is not None:
                                ax2.set_ylabel(
                                    "Energy/token (J, power-based)", fontweight="bold", fontsize=8,
                                    color=COMBO_COLOR_ENERGY,
                                )
                            else:
                                ax2.set_ylabel(
                                    "Avg power (W, power-based)", fontweight="bold", fontsize=8,
                                    color=COMBO_COLOR_POWER,
                                )
                        ax.set_title(f"batch_size={bs}, input_len={il}", fontsize=9, fontweight="bold")
                        ax.set_xlim(0, line_sm_clock_xmax)
                        if normalized:
                            lo, hi = COMBO_NORM_YLIM
                            ax.set_ylim(lo, hi)
                            if include_ept and ax3 is not None:
                                ax3.set_ylim(lo, hi)
                                _set_combo_ept_ylim(ax2, y_ept)
                            else:
                                ax2.set_ylim(lo, hi)
                        elif include_ept and ax3 is not None:
                            ax3.relim(visible_only=True)
                            ax3.autoscale_view(scalex=False, scaley=True)
                            _set_combo_ept_ylim(ax2, y_ept)
                        ax.grid(True, alpha=0.3)
                        ax.tick_params(axis="both", labelsize=7)
                        ax.tick_params(axis="y", labelcolor=COMBO_COLOR_THROUGHPUT)
                        if include_ept and ax3 is not None:
                            ax2.tick_params(axis="y", labelsize=7, labelcolor=COMBO_COLOR_ENERGY)
                            _hide_combo_power_y_axis(ax3)
                        else:
                            ax2.tick_params(axis="y", labelsize=7, labelcolor=COMBO_COLOR_POWER)
                        for lb in ax.get_xticklabels():
                            lb.set_fontweight("bold")
                        for lb in ax.get_yticklabels():
                            lb.set_fontweight("bold")
                        for lb in ax2.get_yticklabels():
                            lb.set_fontweight("bold")
                        if include_ept and ax3 is not None:
                            h_leg = [ln_tp, ln_pw, ln_ept]
                            l_leg = ["throughput", "avg power (W)", "energy/token (J)"]
                        else:
                            h_leg = [ln_tp, ln_pw]
                            l_leg = ["throughput", "avg power (W)"]
                        ax.legend(
                            h_leg,
                            l_leg,
                            fontsize=10,
                            loc="upper left",
                            ncol=1,
                            markerscale=1.6,
                            framealpha=0.92,
                        )

                    for j in range(len(pairs), len(ax_flat)):
                        ax_flat[j].set_visible(False)

                    comp_part = f" · {component}" if component else ""
                    norm_note = (
                        " [normalized, max target SM=1/metric; TP·power y 0.3–1.1; E/T y 0.3·max–1.1·max]"
                        if normalized
                        else " [raw values]"
                    )
                    if include_ept:
                        st = (
                            f"Throughput, avg power, energy/token vs SM clock — {phase}{comp_part}{norm_note}  "
                            f"(panels: batch_size × input_len)"
                        )
                    else:
                        st = (
                            f"Throughput & avg power vs SM clock — {phase}{comp_part}{norm_note}  "
                            f"(panels: batch_size × input_len)"
                        )
                    fig.suptitle(st, fontsize=11, fontweight="bold", y=1.0)
                    fig.tight_layout()
                    tp_pw = "_tp_pw" if not include_ept else ""
                    norm_s = "_norm" if normalized else ""
                    fname = (
                        f"line_throughput_and_avg_power_W_vs_sm_clock_bs_il_{phase}"
                        f"{comp_suffix}{tp_pw}{norm_s}.png"
                    )
                    fig.savefig(line_dir / fname, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    print(f"  저장 (line): {line_dir / fname}")


def plot_line_charts(
    metrics: pd.DataFrame,
    line_dir: Path,
    line_sm_clock_xmax: float = 2500.0,
) -> None:
    """
    (1) x = sm_clock 선형 y, (2) x = memory_util_pct 선형 y,
    (3) x = sm_clock log y (_ylog), (4) x = memory log y (_xmem_ylog),
    (5) x = input_len (_xin / _xin_ylog), (6) x = batch_size 전 배치 한 figure (_xbs / _xbs_ylog),
    (7) x=target 그룹의 평균 sm_clock, 패널=batch×input_len. phase(prefill/decoding)별 raw·norm.
        3메트릭(TP+avg power+energy/token) PNG와 2메트릭(TP+avg power) PNG를 각각 저장(후자 파일명에 _tp_pw).
        3메트릭: 왼쪽 throughput, 안쪽 오른쪽 축에 E/T 제목·눈금; power는 바깥 twin에 곡선만(y축 없음).
        power 스케일은 y축이 보일 때와 동일(norm: 0.3–1.1, raw: autoscale). E/T ylim [0.3·max, 1.1·max].
        attn·mlp 동시 존재 시 …_{phase}_attn*, …_{phase}_mlp* 추가.
    sm_clock x축은 xlim (0, line_sm_clock_xmax) 로 figure 내 패널 통일.
    attn·mlp 동시 존재 시 2×2: 행=prefill/decoding, 열=attn/mlp. 아니면 1×2 prefill|decoding.
    패널별 독립 y축.
    """
    line_dir.mkdir(parents=True, exist_ok=True)
    split = _should_split_attn_mlp(metrics)

    all_lens = metrics["input_len"].dropna().unique().tolist()
    alpha_map = _input_len_alphas(all_lens)

    batch_sizes = sorted(metrics["batch_size"].dropna().unique())

    _plot_line_bundle(
        metrics, line_dir, batch_sizes, alpha_map, split,
        x_col="sm_clock",
        x_axis_label="GPU frequency (SM clock, MHz)",
        fname_suffix="",
        supt_mid="vs GPU frequency",
        line_sm_clock_xmax=line_sm_clock_xmax,
    )
    _plot_line_bundle(
        metrics, line_dir, batch_sizes, alpha_map, split,
        x_col="memory_util_pct",
        x_axis_label="Memory utilization (%)",
        fname_suffix="_xmem",
        supt_mid="vs memory utilization",
    )
    _plot_line_bundle(
        metrics, line_dir, batch_sizes, alpha_map, split,
        x_col="sm_clock",
        x_axis_label="GPU frequency (SM clock, MHz)",
        fname_suffix="_ylog",
        supt_mid="vs GPU frequency",
        y_log_scale=True,
        line_sm_clock_xmax=line_sm_clock_xmax,
    )
    _plot_line_bundle(
        metrics, line_dir, batch_sizes, alpha_map, split,
        x_col="memory_util_pct",
        x_axis_label="Memory utilization (%)",
        fname_suffix="_xmem_ylog",
        supt_mid="vs memory utilization",
        y_log_scale=True,
    )
    _plot_line_bundle(
        metrics, line_dir, batch_sizes, alpha_map, split,
        x_col="input_len",
        x_axis_label=f"Input length (0-{INPUT_LEN_X_MAX // 1000}k)",
        fname_suffix="_xin",
        supt_mid="vs input length",
    )
    _plot_line_bundle(
        metrics, line_dir, batch_sizes, alpha_map, split,
        x_col="input_len",
        x_axis_label=f"Input length (0-{INPUT_LEN_X_MAX // 1000}k)",
        fname_suffix="_xin_ylog",
        supt_mid="vs input length",
        y_log_scale=True,
    )
    _plot_line_bundle(
        metrics, line_dir, batch_sizes, alpha_map, split,
        x_col="batch_size",
        x_axis_label="Batch size",
        fname_suffix="_xbs",
        supt_mid="vs batch size",
    )
    _plot_line_bundle(
        metrics, line_dir, batch_sizes, alpha_map, split,
        x_col="batch_size",
        x_axis_label="Batch size",
        fname_suffix="_xbs_ylog",
        supt_mid="vs batch size",
        y_log_scale=True,
    )
    _plot_sm_clock_throughput_avg_power_grid(metrics, line_dir, line_sm_clock_xmax)


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
        "--idx",
        type=int,
        default=1,
        help="total_energy 시작 index (기본값: 1)",
    )
    parser.add_argument(
        "--line-sm-clock-xmax",
        type=float,
        default=2500.0,
        help="라인 차트(sm_clock x축) x축 상한 MHz, xlim=0~이 값 (기본: 2500)",
    )
    args = parser.parse_args()

    if args.line_sm_clock_xmax <= 0:
        parser.error("--line-sm-clock-xmax 는 0보다 커야 합니다.")

    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        raise NotADirectoryError(f"디렉터리가 아닙니다: {log_dir}")
    output_dir = Path(args.output_dir) if args.output_dir else (log_dir / "analysis_6metrics")

    print("1. CSV 로드...")
    df = load_csvs(log_dir)
    print(f"   전체 행: {len(df):,}")

    print(f"2. iteration 단위 중간 집계... (start_idx={args.idx})")
    it = compute_iteration_stats(df, start_idx=args.idx)
    print(f"   iteration 수: {len(it):,}")

    print("3. data 단위 최종 메트릭 집계...")
    metrics = aggregate_metrics(it)
    print(f"   data 수: {len(metrics):,}")
    print(metrics.to_string(index=False))

    output_dir.mkdir(parents=True, exist_ok=True)
    it.to_csv(output_dir / "iteration_stats.csv", index=False)

    print("\n4. 메트릭 CSV 저장...")
    save_metric_csvs(metrics, output_dir)

    print("\n5. 히스토그램 플롯 생성...")
    plot_histograms(metrics, output_dir)

    line_dir = log_dir / "analysis_6metrics_line"
    print("\n6. SM clock 라인 차트 (analysis_6metrics_line)...")
    plot_line_charts(metrics, line_dir, line_sm_clock_xmax=args.line_sm_clock_xmax)

    print("\n완료!")


if __name__ == "__main__":
    main()
