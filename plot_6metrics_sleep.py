"""
sleep_ns / sleep_freq / kernel 컬럼이 있는 gpu_profile_*.csv 를 로드하여 라인 PNG(에너지·지연·처리량·전력 + GPU/Memory util %) 를 둘 다 저장한다.
(1) x=sleep_ns: (sleep_freq, kernel, 메트릭)당 1장, sf=0 파일 생략, x=-1 에 sf=0 기준 default(검정·legend).
(2) x=sleep_freq: (sleep_ns, kernel, 메트릭)당 1장, (sns=0,sf=0) 은 라인 제외 후 x=-1 default(검정·legend).

사용법:
  python plot_6metrics_sleep.py <log_dir> [--output-dir <out>] [--idx 1]

메트릭 (라인 차트):
  1–6) 에너지·지연·처리량·전력 (기존과 동일)
  7) gpu_util_pct  8) memory_util_pct
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_KEY = [
    "cudagraph_mode",
    "batch_size",
    "target_sm_clock",
    "input_len",
    "model_name",
    "sleep_freq",
    "kernel",
    "sleep_ns",
]
ITER_KEY = [
    "cudagraph_mode",
    "batch_size",
    "target_sm_clock",
    "input_len",
    "model_name",
    "kv_cache_lens",
    "iteration",
    "sleep_ns",
    "sleep_freq",
    "kernel",
]


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
        "temperature", "sleep_ns", "sleep_freq",
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

# sleep_ns / sleep_freq 라인 차트용 (전력·에너지 6종 + GPU/Memory util)
SLEEP_LINE_METRICS = METRIC_COLS[:6] + METRIC_COLS[6:8]
SLEEP_LINE_PREFILL_COLOR = "#C62828"
SLEEP_LINE_DECODE_COLOR = "#1565C0"


def _phase_from_model_name(name: str) -> str | None:
    s = str(name).lower()
    if "prefill" in s:
        return "prefill"
    if "decoding" in s:
        return "decoding"
    return None


def _sleep_freq_is_zero(v) -> bool:
    try:
        return float(v) == 0.0
    except (TypeError, ValueError):
        return False


def _sleep_ns_is_zero(v) -> bool:
    try:
        return float(v) == 0.0
    except (TypeError, ValueError):
        return False


def _sf0_baseline_y(metrics: pd.DataFrame, kern: str, phase: str, col: str) -> float:
    """sleep_freq==0 행에서 phase·kernel 일치 시 해당 메트릭 평균 (default 참조점)."""
    sfn = pd.to_numeric(metrics["sleep_freq"], errors="coerce")
    b = metrics[(sfn == 0) & (metrics["kernel"].astype(str) == str(kern))].copy()
    if b.empty:
        return float("nan")
    b["_ph"] = b["model_name"].map(_phase_from_model_name)
    bp = b[b["_ph"] == phase]
    if bp.empty:
        return float("nan")
    v = pd.to_numeric(bp[col], errors="coerce").mean()
    return float(v) if np.isfinite(v) else float("nan")


def _sns0_sf0_baseline_y(metrics: pd.DataFrame, kern: str, phase: str, col: str) -> float:
    """sleep_ns==0 & sleep_freq==0 (phase·kernel 일치) 메트릭 평균. x=-1 default 점용."""
    sns = pd.to_numeric(metrics["sleep_ns"], errors="coerce")
    sff = pd.to_numeric(metrics["sleep_freq"], errors="coerce")
    b = metrics[
        (sns == 0) & (sff == 0) & (metrics["kernel"].astype(str) == str(kern))
    ].copy()
    if b.empty:
        return float("nan")
    b["_ph"] = b["model_name"].map(_phase_from_model_name)
    bp = b[b["_ph"] == phase]
    if bp.empty:
        return float("nan")
    v = pd.to_numeric(bp[col], errors="coerce").mean()
    return float(v) if np.isfinite(v) else float("nan")


def plot_sleep_ns_line_charts(metrics: pd.DataFrame, output_dir: Path) -> None:
    """
    (sleep_freq, kernel, metric) 조합마다 PNG 1장. sleep_freq==0 은 파일을 만들지 않음.
    2행 1열: 위 Prefill(빨강), 아래 Decoding(파랑). x축 공유(sleep_ns).
    sleep_freq!=0 인 경우 x=-1 에 sleep_freq=0 기준 점(검정), 범례 항목 default. x축 눈금은 정수(-1 포함).
    """
    need = {"sleep_ns", "sleep_freq", "kernel", "model_name"}
    if not need.issubset(metrics.columns):
        print("  sleep_ns 라인 차트 생략: 필수 컬럼 없음", need - set(metrics.columns))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    freqs = sorted(metrics["sleep_freq"].dropna().unique(), key=lambda x: (isinstance(x, str), x))
    kernels = sorted(metrics["kernel"].dropna().astype(str).unique())

    for sf in freqs:
        if _sleep_freq_is_zero(sf):
            continue
        for kern in kernels:
            sub = metrics[
                (metrics["sleep_freq"] == sf) & (metrics["kernel"].astype(str) == str(kern))
            ]
            if sub.empty:
                continue

            sub = sub.copy()
            sub["_phase"] = sub["model_name"].map(_phase_from_model_name)
            sub = sub[sub["_phase"].notna()]
            if sub.empty:
                continue

            sf_tag = sanitize(str(sf))
            kern_tag = sanitize(str(kern))

            for col, title, ylabel in SLEEP_LINE_METRICS:
                fig, (ax_pre, ax_dec) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
                any_drew = False
                line_x_union: set[float] = set()
                show_default_axis = False
                legend_pre = False
                legend_dec = False

                for phase, ax, color, row_title in (
                    ("prefill", ax_pre, SLEEP_LINE_PREFILL_COLOR, "Prefill"),
                    ("decoding", ax_dec, SLEEP_LINE_DECODE_COLOR, "Decoding"),
                ):
                    g = sub[sub["_phase"] == phase]
                    plotted = False
                    if not g.empty:
                        agg = (
                            g.groupby("sleep_ns", as_index=False)[col]
                            .mean()
                            .sort_values("sleep_ns")
                        )
                        if not agg.empty and np.isfinite(agg[col]).any():
                            ax.plot(
                                agg["sleep_ns"],
                                agg[col],
                                marker="o",
                                linewidth=1.5,
                                markersize=5,
                                color=color,
                            )
                            line_x_union.update(
                                float(x) for x in agg["sleep_ns"] if pd.notna(x)
                            )
                            plotted = True
                            any_drew = True
                    def_y = _sf0_baseline_y(metrics, str(kern), phase, col)
                    if np.isfinite(def_y):
                        ax.plot(
                            [-1],
                            [def_y],
                            "o",
                            color="black",
                            markersize=7,
                            zorder=10,
                            clip_on=False,
                            label="default",
                        )
                        show_default_axis = True
                        any_drew = True
                        if phase == "prefill":
                            legend_pre = True
                        else:
                            legend_dec = True
                    if not plotted:
                        ax.text(
                            0.5,
                            0.5,
                            "no data",
                            transform=ax.transAxes,
                            ha="center",
                            va="center",
                            fontsize=10,
                            color="gray",
                        )
                    ax.set_ylabel(ylabel, fontweight="bold")
                    ax.set_title(row_title, fontsize=10, fontweight="bold", color=color)
                    ax.grid(alpha=0.3)
                    ax.tick_params(axis="both", labelsize=9)

                if not any_drew:
                    plt.close(fig)
                    continue

                xs_sorted = sorted(line_x_union)
                if show_default_axis:
                    tick_pos = sorted({-1.0} | set(xs_sorted))
                else:
                    tick_pos = xs_sorted
                if tick_pos:
                    tick_lbl = [
                        str(int(t)) if float(t).is_integer() else str(t) for t in tick_pos
                    ]
                    ax_dec.set_xticks(tick_pos)
                    ax_dec.set_xticklabels(tick_lbl, fontsize=8)
                if legend_pre:
                    ax_pre.legend(loc="best", fontsize=8, framealpha=0.9)
                if legend_dec:
                    ax_dec.legend(loc="best", fontsize=8, framealpha=0.9)
                ax_dec.set_xlabel("sleep_ns", fontweight="bold")
                fig.suptitle(
                    f"{title}  |  sleep_freq={sf}, kernel={kern}",
                    fontsize=11,
                    fontweight="bold",
                    y=1.01,
                )
                fig.tight_layout()
                col_tag = sanitize(col)
                fname = f"line_sleep_ns_{col_tag}_sf{sf_tag}_kern_{kern_tag}.png"
                out_path = output_dir / fname
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"  저장: {out_path}")


def plot_sleep_freq_line_charts(metrics: pd.DataFrame, output_dir: Path) -> None:
    """
    (sleep_ns, kernel, metric) 조합마다 PNG 1장. x축 sleep_freq, 위·아래 Prefill/Decoding.
    (sleep_ns=0, sleep_freq=0) 값은 라인에서 빼고 x=-1·검정·legend default 로만 표시.
    sleep_ns==0 인 PNG 에서는 sleep_freq==0 점을 라인에 넣지 않음(중복 방지).
    """
    need = {"sleep_ns", "sleep_freq", "kernel", "model_name"}
    if not need.issubset(metrics.columns):
        print("  sleep_freq 라인 차트 생략: 필수 컬럼 없음", need - set(metrics.columns))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    sns_all = pd.to_numeric(metrics["sleep_ns"], errors="coerce").dropna().unique()
    sns_list = sorted({float(x) for x in sns_all})
    kernels = sorted(metrics["kernel"].dropna().astype(str).unique())
    m_sns = pd.to_numeric(metrics["sleep_ns"], errors="coerce")

    for sns in sns_list:
        for kern in kernels:
            sub = metrics[
                (m_sns == sns) & (metrics["kernel"].astype(str) == str(kern))
            ]
            if sub.empty:
                continue

            sub = sub.copy()
            sub["_phase"] = sub["model_name"].map(_phase_from_model_name)
            sub = sub[sub["_phase"].notna()]
            if sub.empty:
                continue

            sns_tag = sanitize(str(int(sns)) if float(sns).is_integer() else str(sns))
            kern_tag = sanitize(str(kern))

            for col, title, ylabel in SLEEP_LINE_METRICS:
                fig, (ax_pre, ax_dec) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
                any_drew = False
                line_x_union: set[float] = set()
                show_default_axis = False
                legend_pre = False
                legend_dec = False

                for phase, ax, color, row_title in (
                    ("prefill", ax_pre, SLEEP_LINE_PREFILL_COLOR, "Prefill"),
                    ("decoding", ax_dec, SLEEP_LINE_DECODE_COLOR, "Decoding"),
                ):
                    g = sub[sub["_phase"] == phase]
                    plotted = False
                    if not g.empty:
                        agg = (
                            g.groupby("sleep_freq", as_index=False)[col]
                            .mean()
                            .sort_values("sleep_freq")
                        )
                        if _sleep_ns_is_zero(sns) and not agg.empty:
                            sff = pd.to_numeric(agg["sleep_freq"], errors="coerce")
                            agg = agg[sff != 0]
                        if not agg.empty and np.isfinite(agg[col]).any():
                            ax.plot(
                                agg["sleep_freq"],
                                agg[col],
                                marker="o",
                                linewidth=1.5,
                                markersize=5,
                                color=color,
                            )
                            line_x_union.update(
                                float(x) for x in agg["sleep_freq"] if pd.notna(x)
                            )
                            plotted = True
                            any_drew = True
                    def_y = _sns0_sf0_baseline_y(metrics, str(kern), phase, col)
                    if np.isfinite(def_y):
                        ax.plot(
                            [-1],
                            [def_y],
                            "o",
                            color="black",
                            markersize=7,
                            zorder=10,
                            clip_on=False,
                            label="default",
                        )
                        show_default_axis = True
                        any_drew = True
                        if phase == "prefill":
                            legend_pre = True
                        else:
                            legend_dec = True
                    if not plotted:
                        ax.text(
                            0.5,
                            0.5,
                            "no data",
                            transform=ax.transAxes,
                            ha="center",
                            va="center",
                            fontsize=10,
                            color="gray",
                        )
                    ax.set_ylabel(ylabel, fontweight="bold")
                    ax.set_title(row_title, fontsize=10, fontweight="bold", color=color)
                    ax.grid(alpha=0.3)
                    ax.tick_params(axis="both", labelsize=9)

                if not any_drew:
                    plt.close(fig)
                    continue

                xs_sorted = sorted(line_x_union)
                if show_default_axis:
                    tick_pos = sorted({-1.0} | set(xs_sorted))
                else:
                    tick_pos = xs_sorted
                if tick_pos:
                    tick_lbl = [
                        str(int(t)) if float(t).is_integer() else str(t) for t in tick_pos
                    ]
                    ax_dec.set_xticks(tick_pos)
                    ax_dec.set_xticklabels(tick_lbl, fontsize=8)
                if legend_pre:
                    ax_pre.legend(loc="best", fontsize=8, framealpha=0.9)
                if legend_dec:
                    ax_dec.legend(loc="best", fontsize=8, framealpha=0.9)
                ax_dec.set_xlabel("sleep_freq", fontweight="bold")
                fig.suptitle(
                    f"{title}  |  sleep_ns={sns}, kernel={kern}",
                    fontsize=11,
                    fontweight="bold",
                    y=1.01,
                )
                fig.tight_layout()
                col_tag = sanitize(col)
                fname = f"line_sleep_freq_{col_tag}_sns_{sns_tag}_kern_{kern_tag}.png"
                out_path = output_dir / fname
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"  저장: {out_path}")


def _hist_combo_sort_key(combo: tuple) -> tuple:
    """히스토그램 서브플롯 순서: input_len → prefill/decoding → attn/mlp."""
    il, mn = combo
    mn_lower = str(mn).lower()
    phase = 0 if "prefill" in mn_lower else (1 if "decoding" in mn_lower else 2)
    comp = 0 if "attn" in mn_lower else (1 if "mlp" in mn_lower else 2)
    return (int(il), phase, comp, str(mn))


def _hist_combo_sort_key_bs(combo: tuple) -> tuple:
    """히스토그램 서브플롯 순서: batch_size → prefill/decoding → attn/mlp."""
    bs, mn = combo
    mn_lower = str(mn).lower()
    phase = 0 if "prefill" in mn_lower else (1 if "decoding" in mn_lower else 2)
    comp = 0 if "attn" in mn_lower else (1 if "mlp" in mn_lower else 2)
    return (int(bs), phase, comp, str(mn))


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


def _plot_one_by_input_len(metrics: pd.DataFrame, output_dir: Path, mode: str) -> None:
    """input_len 고정, 패널 = (batch_size, model). 파일명 hist_*_il{il}_*.png"""
    normalize = mode == "norm"

    batch_sizes = sorted(metrics["batch_size"].dropna().unique())
    input_lens = sorted(metrics["input_len"].dropna().unique())
    model_names = sorted(metrics["model_name"].dropna().unique())

    for col, title, ylabel in METRIC_COLS:
        for il in input_lens:
            sub_il = metrics[metrics["input_len"] == il]
            if sub_il.empty:
                continue

            combos = []
            for bs in batch_sizes:
                for mn in model_names:
                    s = sub_il[(sub_il["batch_size"] == bs) & (sub_il["model_name"] == mn)]
                    if not s.empty:
                        combos.append((bs, mn))
            combos.sort(key=_hist_combo_sort_key_bs)

            if not combos:
                continue

            ncols = min(len(combos), 2)
            nrows = math.ceil(len(combos) / ncols)
            fig_w = 6.5 * ncols + 1
            fig_h = 4.5 * nrows + 1
            fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
            ax_flat = axes.ravel()

            for idx, (bs, mn) in enumerate(combos):
                ax = ax_flat[idx]
                sub = sub_il[(sub_il["batch_size"] == bs) & (sub_il["model_name"] == mn)]
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
                ax.set_title(f"{mn}  batch_size={int(bs)}", fontsize=10, fontweight="bold")
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
            fig.suptitle(f"{title}{suffix}  (input_len={int(il)})", fontsize=13, fontweight="bold", y=1.01)
            fig.tight_layout()

            tag = "norm" if normalize else "raw"
            fname = f"hist_{sanitize(col)}_il{int(il)}_{tag}.png"
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


def _plot_compare_sm_by_input_len(metrics: pd.DataFrame, output_dir: Path) -> None:
    """input_len 고정, 패널 = (batch_size, model). 파일명 hist_*_il{il}_compare.png"""
    batch_sizes = sorted(metrics["batch_size"].dropna().unique())
    input_lens = sorted(metrics["input_len"].dropna().unique())
    model_names = sorted(metrics["model_name"].dropna().unique())

    for col, title, ylabel in METRIC_COLS:
        if col not in COMPARE_SM_COLS:
            continue
        for il in input_lens:
            sub_il = metrics[metrics["input_len"] == il]
            if sub_il.empty:
                continue

            combos = []
            for bs in batch_sizes:
                for mn in model_names:
                    s = sub_il[(sub_il["batch_size"] == bs) & (sub_il["model_name"] == mn)]
                    if not s.empty:
                        combos.append((bs, mn))

            combos.sort(key=_hist_combo_sort_key_bs)

            if not combos:
                continue

            ncols = min(len(combos), 2)
            nrows = math.ceil(len(combos) / ncols)
            fig_w = 7.5 * ncols + 1
            fig_h = 4.5 * nrows + 1
            fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
            ax_flat = axes.ravel()

            bar_w = 0.25

            for idx, (bs, mn) in enumerate(combos):
                ax = ax_flat[idx]
                sub = sub_il[(sub_il["batch_size"] == bs) & (sub_il["model_name"] == mn)]
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
                ax.set_title(f"{mn}  batch_size={int(bs)}", fontsize=10, fontweight="bold")
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

            fig.suptitle(f"{title} vs SM Clock ratio  (input_len={int(il)})",
                         fontsize=13, fontweight="bold", y=1.01)
            fig.tight_layout()

            fname = f"hist_{sanitize(col)}_il{int(il)}_compare.png"
            fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  저장: {output_dir / fname}")


def plot_histograms(metrics: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_one(metrics, output_dir, mode="raw")
    _plot_one(metrics, output_dir, mode="norm")
    _plot_compare_sm(metrics, output_dir)
    _plot_one_by_input_len(metrics, output_dir, mode="raw")
    _plot_one_by_input_len(metrics, output_dir, mode="norm")
    _plot_compare_sm_by_input_len(metrics, output_dir)


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
    """Energy/token y축: 양수 max이면 [0.3*max, 1.1*max], 그 외 min~max 또는 소구간."""
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


def _combo_try_get_plot_series(
    metrics: pd.DataFrame,
    phase: str,
    component: str | None,
    bs: int,
    il: int,
    normalized: bool,
    include_ept: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """콤보 패널용 집계 시리즈. 유효하면 (x, x_tgt, y_tp, y_pw, y_ept), 아니면 None.
    normalized: 세 메트릭 모두 max target SM 대비 비율. raw+include_ept: E/T만 동일 비율, TP·PW는 raw.
    """
    sub = metrics[(metrics["batch_size"] == bs) & (metrics["input_len"] == il)]
    if sub.empty:
        return None
    work = sub.copy()
    work["_phase"] = work["model_name"].map(_line_infer_phase)
    work = work[work["_phase"] == phase].copy()
    if component is not None:
        work = work[work["model_name"].map(_line_infer_component) == component].copy()
    if work.empty:
        return None
    w = work.dropna(subset=["target_sm_clock", "sm_clock"])
    if w.empty:
        return None
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
        return None
    x = g["sm_clock"].values.astype(float)
    x_tgt = g["target_sm_clock"].values.astype(float)
    y_tp = g[TP_COL].values.astype(float)
    y_pw = g[PWR_COL].values.astype(float)
    y_ept = g[EPT_COL].values.astype(float)
    if normalized:
        y_tp = _normalize_by_value_at_max_target_sm_clock(x_tgt, y_tp)
        y_pw = _normalize_by_value_at_max_target_sm_clock(x_tgt, y_pw)
        y_ept = _normalize_by_value_at_max_target_sm_clock(x_tgt, y_ept)
    elif include_ept:
        y_ept = _normalize_by_value_at_max_target_sm_clock(x_tgt, y_ept)
    return x, x_tgt, y_tp, y_pw, y_ept


def _combo_figure_unified_ylim_bounds(chunks: list[np.ndarray]) -> tuple[float, float]:
    """한 figure 안 해당 메트릭 전 패널 값 합쳐 ymin=0.9·min, ymax=1.1·max."""
    parts: list[np.ndarray] = []
    for c in chunks:
        v = np.asarray(c, dtype=float).ravel()
        v = v[np.isfinite(v)]
        if v.size:
            parts.append(v)
    if not parts:
        return (0.0, 1.0)
    a = np.concatenate(parts)
    lo, hi = float(np.min(a)), float(np.max(a))
    ymin, ymax = lo * 0.9, hi * 1.1
    if ymax <= ymin or not math.isfinite(ymin) or not math.isfinite(ymax):
        pad = max(abs(hi), abs(lo), 1e-12) * 0.05
        ymin, ymax = lo - pad, hi + pad
    return (ymin, ymax)


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
    *_norm.png: 세 메트릭 모두 max target SM 정규화; TP·power y (0.3,1.1), E/T는 패널별 [0.3·max,1.1·max].
    raw: TP·power는 figure 전역 y(0.9·min–1.1·max); E/T만 정규화·y∈[0.1,1.2]. 3메트릭 축 배치 동일.
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

                    tp_chunks: list[np.ndarray] = []
                    pw_chunks: list[np.ndarray] = []
                    if not normalized:
                        for bs0, il0 in pairs:
                            ser0 = _combo_try_get_plot_series(
                                metrics, phase, component, bs0, il0, False, include_ept,
                            )
                            if ser0 is None:
                                continue
                            _, _, yt0, yp0, _ = ser0
                            tp_chunks.append(yt0)
                            pw_chunks.append(yp0)
                        ylim_tp = _combo_figure_unified_ylim_bounds(tp_chunks)
                        ylim_pw = _combo_figure_unified_ylim_bounds(pw_chunks)
                        ylim_ept = (0.1, 1.2) if include_ept else (0.0, 1.0)

                    for idx, (bs, il) in enumerate(pairs):
                        ax = ax_flat[idx]
                        ser = _combo_try_get_plot_series(
                            metrics, phase, component, bs, il, normalized, include_ept,
                        )
                        if ser is None:
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
                            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                            ax.set_xlim(0, line_sm_clock_xmax)
                            continue

                        x, x_tgt, y_tp, y_pw, y_ept = ser

                        ax2 = ax.twinx()
                        ax3 = None
                        if include_ept:
                            ax3 = ax.twinx()
                            ax3.spines["right"].set_position(("outward", 44))
                            ax3.patch.set_visible(False)
                            for sp in ("top", "left", "bottom"):
                                ax3.spines[sp].set_visible(False)

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
                                _annotate_line_points(ax2, x, y_ept, ".3f")
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
                                    "Energy/token(J/token)", fontweight="bold", fontsize=8,
                                    color=COMBO_COLOR_ENERGY,
                                )
                            else:
                                ax2.set_ylabel(
                                    "Avg power (W, power)", fontweight="bold", fontsize=8,
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
                        else:
                            ax.set_ylim(ylim_tp[0], ylim_tp[1])
                            if include_ept and ax3 is not None:
                                ax2.set_ylim(ylim_ept[0], ylim_ept[1])
                                ax3.set_ylim(ylim_pw[0], ylim_pw[1])
                            else:
                                ax2.set_ylim(ylim_pw[0], ylim_pw[1])
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
                            l_leg = (
                                ["throughput", "avg power (W)", "energy/token (J)"]
                                if normalized
                                else ["throughput", "avg power (W)", "energy/token(Norm)"]
                            )
                        else:
                            h_leg = [ln_tp, ln_pw]
                            l_leg = ["throughput", "avg power (W)"]
                        ax.legend(
                            h_leg,
                            l_leg,
                            fontsize=8,
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
                        else (
                            " [raw; TP·power y unified/fig; E/T vs max target SM, y∈[0.1,1.2]]"
                            if include_ept
                            else " [raw; TP·power y unified/fig]"
                        )
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


def _plot_sm_clock_throughput_avg_power_grid_by_input_len(
    metrics: pd.DataFrame,
    line_dir: Path,
    line_sm_clock_xmax: float,
) -> None:
    """
    input_len 고정, 한 PNG 안에 batch_size별 패널만 배치 (3메트릭·2메트릭 규칙은 bs×il 그리드와 동일).
    파일명: line_throughput_and_avg_power_W_vs_sm_clock_il{il}_bs_{phase}...
    """
    need = {"batch_size", "input_len", "target_sm_clock", "sm_clock", "model_name", TP_COL, PWR_COL, EPT_COL}
    if not need <= set(metrics.columns):
        print("  (line combo by input_len) 스킵: 필요한 컬럼이 없습니다.")
        return

    batch_sizes = sorted(metrics["batch_size"].dropna().unique())
    input_lens = sorted(metrics["input_len"].dropna().unique())
    line_dir.mkdir(parents=True, exist_ok=True)

    split_am = _should_split_attn_mlp(metrics)

    for il in input_lens:
        pairs: list[tuple[int, int]] = []
        for bs in batch_sizes:
            sub = metrics[(metrics["batch_size"] == bs) & (metrics["input_len"] == il)]
            if sub.empty:
                continue
            pairs.append((int(bs), int(il)))
        if not pairs:
            continue

        n = len(pairs)
        ncols = min(4, max(2, int(math.ceil(math.sqrt(n)))))
        nrows = int(math.ceil(n / ncols))
        fig_w = 4.8 * ncols + 1.2
        fig_h = 3.4 * nrows + 1.2

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

                        tp_chunks: list[np.ndarray] = []
                        pw_chunks: list[np.ndarray] = []
                        if not normalized:
                            for bs0, il0 in pairs:
                                ser0 = _combo_try_get_plot_series(
                                    metrics, phase, component, bs0, il0, False, include_ept,
                                )
                                if ser0 is None:
                                    continue
                                _, _, yt0, yp0, _ = ser0
                                tp_chunks.append(yt0)
                                pw_chunks.append(yp0)
                            ylim_tp = _combo_figure_unified_ylim_bounds(tp_chunks)
                            ylim_pw = _combo_figure_unified_ylim_bounds(pw_chunks)
                            ylim_ept = (0.1, 1.2) if include_ept else (0.0, 1.0)

                        for idx, (bs, il_one) in enumerate(pairs):
                            ax = ax_flat[idx]
                            ser = _combo_try_get_plot_series(
                                metrics, phase, component, bs, il_one, normalized, include_ept,
                            )
                            if ser is None:
                                sub = metrics[(metrics["batch_size"] == bs) & (metrics["input_len"] == il_one)]
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
                                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                                ax.set_xlim(0, line_sm_clock_xmax)
                                continue

                            x, x_tgt, y_tp, y_pw, y_ept = ser

                            ax2 = ax.twinx()
                            ax3 = None
                            if include_ept:
                                ax3 = ax.twinx()
                                ax3.spines["right"].set_position(("outward", 44))
                                ax3.patch.set_visible(False)
                                for sp in ("top", "left", "bottom"):
                                    ax3.spines[sp].set_visible(False)

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
                                    _annotate_line_points(ax2, x, y_ept, ".3f")
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
                                        "Energy/token(J/token)", fontweight="bold", fontsize=8,
                                        color=COMBO_COLOR_ENERGY,
                                    )
                                else:
                                    ax2.set_ylabel(
                                        "Avg power (W, power)", fontweight="bold", fontsize=8,
                                        color=COMBO_COLOR_POWER,
                                    )
                            ax.set_title(f"batch_size={bs}, input_len={il_one}", fontsize=9, fontweight="bold")
                            ax.set_xlim(0, line_sm_clock_xmax)
                            if normalized:
                                lo, hi = COMBO_NORM_YLIM
                                ax.set_ylim(lo, hi)
                                if include_ept and ax3 is not None:
                                    ax3.set_ylim(lo, hi)
                                    _set_combo_ept_ylim(ax2, y_ept)
                                else:
                                    ax2.set_ylim(lo, hi)
                            else:
                                ax.set_ylim(ylim_tp[0], ylim_tp[1])
                                if include_ept and ax3 is not None:
                                    ax2.set_ylim(ylim_ept[0], ylim_ept[1])
                                    ax3.set_ylim(ylim_pw[0], ylim_pw[1])
                                else:
                                    ax2.set_ylim(ylim_pw[0], ylim_pw[1])
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
                                l_leg = (
                                    ["throughput", "avg power (W)", "energy/token (J)"]
                                    if normalized
                                    else ["throughput", "avg power (W)", "energy/token(Norm)"]
                                )
                            else:
                                h_leg = [ln_tp, ln_pw]
                                l_leg = ["throughput", "avg power (W)"]
                            ax.legend(
                                h_leg,
                                l_leg,
                                fontsize=8,
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
                            else (
                                " [raw; TP·power y unified/fig; E/T vs max target SM, y∈[0.1,1.2]]"
                                if include_ept
                                else " [raw; TP·power y unified/fig]"
                            )
                        )
                        if include_ept:
                            st = (
                                f"Throughput, avg power, energy/token vs SM clock — {phase}{comp_part}{norm_note}  "
                                f"(input_len={il}, panels: batch_size)"
                            )
                        else:
                            st = (
                                f"Throughput & avg power vs SM clock — {phase}{comp_part}{norm_note}  "
                                f"(input_len={il}, panels: batch_size)"
                            )
                        fig.suptitle(st, fontsize=11, fontweight="bold", y=1.0)
                        fig.tight_layout()
                        tp_pw = "_tp_pw" if not include_ept else ""
                        norm_s = "_norm" if normalized else ""
                        fname = (
                            f"line_throughput_and_avg_power_W_vs_sm_clock_il{il}_bs_{phase}"
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
    (8) 동일 3·2메트릭이나 input_len 고정·패널=batch_size만 모은 PNG(파일명 il{input_len}_bs_…).
        3메트릭(TP+avg power+energy/token) PNG와 2메트릭(TP+avg power) PNG를 각각 저장(후자 파일명에 _tp_pw).
        3메트릭: 왼쪽 throughput, 안쪽 E/T, 바깥 power(곡선만).
        *_norm.png: 세 메트릭 정규화·ylim v1 규칙. raw: TP·power figure 전역 y; E/T만 비율·y∈[0.1,1.2].
        attn·mlp 동시 존재 시 …_{phase}_attn*, …_{phase}_mlp* 추가.
    sm_clock x축은 xlim (0, line_sm_clock_xmax) 로 figure 내 패널 통일.
    attn·mlp 동시 존재 시 2×2: 행=prefill/decoding, 열=attn/mlp. 아니면 1×2 prefill|decoding.
    raw 콤보는 TP·power만 figure 내 y 공유; norm 콤보는 *_norm 규칙(패널별 E/T y 등).
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
    _plot_sm_clock_throughput_avg_power_grid_by_input_len(metrics, line_dir, line_sm_clock_xmax)


# ── CSV 저장 ──────────────────────────────────────────
def save_metric_csvs(metrics: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    extra = [c for c in ("sm_clock",) if c in metrics.columns]
    for col, _, _ in METRIC_COLS:
        out = metrics[DATA_KEY + extra + [col]].copy()
        path = output_dir / f"{col}.csv"
        out.to_csv(path, index=False)
        print(f"  CSV 저장: {path}")


# ── main ──────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="gpu_profile_*.csv (sleep_ns/sleep_freq/kernel) 로 sleep 라인 차트(에너지·전력·util 등)를 생성한다."
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
    args = parser.parse_args()

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

    line_sleep_dir = output_dir / "analysis_sleep_ns_line"
    print("\n5. sleep_ns 라인 차트 (sleep_freq × kernel, util 포함)...")
    plot_sleep_ns_line_charts(metrics, line_sleep_dir)

    line_sleep_freq_dir = output_dir / "analysis_sleep_freq_line"
    print("\n6. sleep_freq 라인 차트 (sleep_ns × kernel, util 포함)...")
    plot_sleep_freq_line_charts(metrics, line_sleep_freq_dir)

    print("\n완료!")


if __name__ == "__main__":
    main()
