"""
지정 폴더 내 gpu_profile_*.csv 파일들을 로드하여
6가지 메트릭을 계산하고, 각각 CSV로 저장한 뒤 sm_clock별 히스토그램을 그린다.

사용법:
  python plot_6metrics_histogram_v0.py <log_dir> [--output-dir <out>] [--iteration-surrogate during_time]

iteration 컬럼이 없는 gpu_profile(예: gpu_profile_1427)은 기본 segment 모드:
DATA_KEY 별로 during_time|length|kv_cache_lens 시그니처를 factorize. --iteration-surrogate during_time 은 구형(그룹당 dt|len 만).

메트릭:
  1) total_energy_based_j_per_token
  2) power_based_j_per_token
  3) latency  (sec/token)
  4) throughput  (tokens/sec)
  5) total_energy_based_avg_power  (W)
  6) power_based_avg_power  (W)
  7) gpu_util_pct, 8) memory_util_pct

히스토그램 PNG (배치별):
  - hist_<metric>_bs<N>.png : (input_len×model) 격자, 원시값
  - hist_<metric>_bs<N>_norm_maxsm1.png : 각 서브플롯에서 max SM clock 값을 1로 두고 비율
  - hist_<metric>_bs<N>_norm_maxsm_pct.png : 동일, 퍼센트(100%=max SM)
  - hist_<metric>_combined_bs<N>.png : 한 축에 모든 (input_len, model) 그룹 막대, 원시값
  - hist_<metric>_combined_bs<N>_norm_maxsm1.png / _norm_maxsm_pct.png : combined 정규화
  - hist_<metric>_model_<name>.png : model_name별 (batch×input_len) 격자, 파일 하나에 저장
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
ITER_KEY = DATA_KEY + ["kv_cache_lens", "iteration"]


# ── 유틸 ──────────────────────────────────────────────
def sanitize(name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", name).strip("_") or "unknown"


# ── 로드 ──────────────────────────────────────────────
def ensure_iteration_column(df: pd.DataFrame, surrogate: str = "segment") -> pd.DataFrame:
    """
    iteration 컬럼이 없을 때 surrogate 모드:

    - segment (기본): DATA_KEY 그룹 안에서 during_time|length|kv_cache_lens 를 factorize.
      gpu_profile_1427 처럼 kv 가 바뀌는 구간마다 0,1,2… (같은 kv 블록 안은 동일 iteration).
    - during_time: (DATA_KEY+kv_cache_lens) 그룹에서 during_time|length 만 factorize
      (한 블록에 (dt,length) 가 한 종류면 전부 0이 됨 — 구형 로그용).
    - index1 또는 빈 문자열: (DATA_KEY+kv_cache_lens) 에서 index==1 마다 iteration 증가.
    """
    out = df.copy()
    if "iteration" in out.columns:
        out["iteration"] = (
            pd.to_numeric(out["iteration"], errors="coerce").fillna(0).astype(np.int64)
        )
        return out

    out["_row_ord"] = np.arange(len(out), dtype=np.int64)
    out = out.sort_values("_row_ord", kind="mergesort")
    base_data = [c for c in DATA_KEY if c in out.columns]
    base_kv = [c for c in (DATA_KEY + ["kv_cache_lens"]) if c in out.columns]
    if not base_kv:
        out["iteration"] = np.int64(0)
        return out.drop(columns=["_row_ord"])

    sur = (surrogate or "").strip().lower()
    if sur in ("", "index1", "index_1"):
        def _iter_from_index1(ser: pd.Series) -> np.ndarray:
            m = pd.to_numeric(ser, errors="coerce").fillna(0) == 1
            return (m.cumsum() - 1).clip(lower=0).astype(np.int64)

        out["iteration"] = out.groupby(base_kv, sort=False, group_keys=False)["index"].transform(
            _iter_from_index1
        )
    elif sur == "during_time":
        if "during_time" not in out.columns:
            raise ValueError("during_time 모드인데 during_time 컬럼이 없습니다.")
        s = pd.to_numeric(out["during_time"], errors="coerce")
        lg = (
            pd.to_numeric(out["length"], errors="coerce")
            if "length" in out.columns
            else pd.Series(0, index=out.index)
        )
        out["_sig"] = s.astype(str) + "|" + lg.astype(str)

        def _fac_codes(series: pd.Series) -> np.ndarray:
            codes, _ = pd.factorize(series, sort=False)
            return codes.astype(np.int64, copy=False)

        out["iteration"] = out.groupby(base_kv, sort=False, group_keys=False)["_sig"].transform(
            _fac_codes
        )
        out = out.drop(columns=["_sig"])
    else:
        # segment (default): DATA_KEY 만으로 그룹, 시그니처에 kv 포함 → kv/dt/len 변화마다 증가
        if "during_time" not in out.columns:
            raise ValueError("segment 모드인데 during_time 컬럼이 없습니다.")
        s = pd.to_numeric(out["during_time"], errors="coerce")
        lg = (
            pd.to_numeric(out["length"], errors="coerce")
            if "length" in out.columns
            else pd.Series(0, index=out.index)
        )
        kv = (
            pd.to_numeric(out["kv_cache_lens"], errors="coerce")
            if "kv_cache_lens" in out.columns
            else pd.Series(np.nan, index=out.index)
        )
        out["_sig"] = s.astype(str) + "|" + lg.astype(str) + "|" + kv.astype(str)

        def _fac_codes2(series: pd.Series) -> np.ndarray:
            codes, _ = pd.factorize(series, sort=False)
            return codes.astype(np.int64, copy=False)

        if not base_data:
            out["iteration"] = _fac_codes2(out["_sig"])
        else:
            out["iteration"] = out.groupby(base_data, sort=False, group_keys=False)[
                "_sig"
            ].transform(_fac_codes2)
        out = out.drop(columns=["_sig"])

    return out.drop(columns=["_row_ord"])


def load_csvs(log_dir: Path, iteration_surrogate: str = "segment") -> pd.DataFrame:
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
    cols_ok = [c for c in num_cols if c in df.columns]
    if cols_ok:
        df[cols_ok] = df[cols_ok].apply(pd.to_numeric, errors="coerce")

    had_iteration = "iteration" in df.columns
    df = ensure_iteration_column(df, surrogate=iteration_surrogate)
    if not had_iteration:
        print(f"   iteration 컬럼 없음 → 모드={iteration_surrogate!r} 로 구간 ID 부여")
    return df


# ── iteration 단위 중간 집계 ──────────────────────────
def compute_iteration_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    iteration = (DATA_KEY, kv_cache_lens, iteration) 그룹.
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


def _normalize_by_max_sm_clock(vals: np.ndarray, sm_clocks: np.ndarray) -> tuple[np.ndarray, bool]:
    """최대 sm_clock 위치의 값을 기준(1.0)으로 나눈 비율. 기준이 0이거나 비유효면 전부 nan."""
    if len(vals) == 0:
        return vals, False
    imax = int(np.argmax(sm_clocks))
    baseline = vals[imax]
    if not np.isfinite(baseline) or baseline == 0:
        return np.full_like(vals, np.nan, dtype=float), False
    return (vals.astype(float) / baseline), True


def _collect_il_mn_combos(
    sub_bs: pd.DataFrame, input_lens: list, model_names: list
) -> list[tuple[float, str]]:
    combos: list[tuple[float, str]] = []
    for il in input_lens:
        for mn in model_names:
            s = sub_bs[(sub_bs["input_len"] == il) & (sub_bs["model_name"] == mn)]
            if not s.empty:
                combos.append((float(il), str(mn)))
    return combos


def _sm_series_mean(sub: pd.DataFrame, col: str) -> pd.Series:
    return sub.groupby("sm_clock", sort=True)[col].mean().sort_index()


def _plot_one_subplot_bars(
    ax,
    sub: pd.DataFrame,
    col: str,
    mn: str,
    il: float,
    ylabel: str,
    norm_mode: str | None,
) -> str:
    agg = sub.groupby("sm_clock", as_index=False)[col].mean().sort_values("sm_clock")
    sm_labels = [str(int(s)) for s in agg["sm_clock"]]
    vals_raw = agg[col].values.astype(float)
    sm_arr = agg["sm_clock"].values.astype(float)

    if norm_mode is None:
        vals = vals_raw
        fmt = LABEL_FMT.get(col, ".5f")
        ylab = ylabel
        supt_note = ""
    elif norm_mode == "ratio":
        vals, ok = _normalize_by_max_sm_clock(vals_raw, sm_arr)
        fmt = ".4f"
        ylab = f"ratio ({ylabel} @ max SM = 1)"
        supt_note = " · norm: max SM clock = 1"
        if not ok:
            ax.set_title(
                f"{mn}  input_len={int(il)}\n(no valid baseline)", fontsize=9, fontweight="bold"
            )
            ax.set_xlabel("SM Clock (MHz)", fontweight="bold")
            ax.set_ylabel(ylab, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)
            return supt_note
    else:  # pct
        r, ok = _normalize_by_max_sm_clock(vals_raw, sm_arr)
        vals = r * 100.0
        fmt = ".2f"
        ylab = f"% of {ylabel} at max SM clock"
        supt_note = " · norm: % @ max SM clock"
        if not ok:
            ax.set_title(
                f"{mn}  input_len={int(il)}\n(no valid baseline)", fontsize=9, fontweight="bold"
            )
            ax.set_xlabel("SM Clock (MHz)", fontweight="bold")
            ax.set_ylabel(ylab, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)
            return supt_note

    x_pos = np.arange(len(sm_labels))
    colors = _bar_colors(mn, vals, sm_arr)
    bars = ax.bar(x_pos, np.nan_to_num(vals, nan=0.0), color=colors, edgecolor="white", width=0.7)

    for bar, v in zip(bars, vals):
        if np.isfinite(v):
            if norm_mode == "pct":
                txt = f"{v:{fmt}}%"
            else:
                txt = f"{v:{fmt}}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                txt,
                ha="center", va="bottom", fontsize=7, rotation=45,
                fontweight="bold",
            )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(sm_labels, rotation=45, ha="right", fontsize=8, fontweight="bold")
    ax.set_xlabel("SM Clock (MHz)", fontweight="bold")
    ax.set_ylabel(ylabel if norm_mode is None else ylab, fontweight="bold")
    ax.set_title(f"{mn}  input_len={int(il)}", fontsize=10, fontweight="bold")
    ax.tick_params(axis="y", labelsize=8)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.grid(axis="y", alpha=0.3)
    if norm_mode == "pct" and np.isfinite(vals).any():
        ax.axhline(100.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    return supt_note


def _plot_bs_grid(
    sub_bs: pd.DataFrame,
    bs: float,
    combos: list[tuple[float, str]],
    col: str,
    title: str,
    ylabel: str,
    output_dir: Path,
    norm_mode: str | None,
    fname_suffix: str,
) -> None:
    ncols = min(len(combos), 3)
    nrows = math.ceil(len(combos) / ncols)
    fig_w = 6.5 * ncols + 1
    fig_h = 4.5 * nrows + 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    ax_flat = axes.ravel()
    extra = ""

    for idx, (il, mn) in enumerate(combos):
        ax = ax_flat[idx]
        sub = sub_bs[(sub_bs["input_len"] == il) & (sub_bs["model_name"] == mn)]
        sn = _plot_one_subplot_bars(ax, sub, col, mn, il, ylabel, norm_mode)
        if sn:
            extra = sn  # 동일 문구로 충분 (첫 비어있지 않은 서브플롯)

    for i in range(len(combos), len(ax_flat)):
        ax_flat[i].set_visible(False)

    supt = f"{title}  (batch_size={int(bs)}){extra}"
    fig.suptitle(supt, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fname = f"hist_{sanitize(col)}_bs{int(bs)}{fname_suffix}.png"
    fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {output_dir / fname}")


def _combined_tab_color(j: int) -> str:
    cmap = plt.get_cmap("tab20")
    return cmap(j % 20)


def _plot_combined_single_ax(
    metrics: pd.DataFrame,
    sub_bs: pd.DataFrame,
    bs: float,
    col: str,
    title: str,
    ylabel: str,
    output_dir: Path,
    norm_mode: str | None,
    fname_suffix: str,
    input_lens: list,
    model_names: list,
) -> None:
    series_rows: list[tuple[float, str, pd.Series, str]] = []
    for il in input_lens:
        for mn in model_names:
            sub = sub_bs[(sub_bs["input_len"] == il) & (sub_bs["model_name"] == mn)]
            if sub.empty:
                continue
            raw_s = _sm_series_mean(sub, col)
            if raw_s.empty:
                continue
            if norm_mode == "ratio":
                v = raw_s.values.astype(float)
                smx = raw_s.index.astype(float).values
                r, ok = _normalize_by_max_sm_clock(v, smx)
                if not ok:
                    continue
                s = pd.Series(r, index=raw_s.index)
                leg_yl = "ratio (max SM = 1)"
            elif norm_mode == "pct":
                v = raw_s.values.astype(float)
                smx = raw_s.index.astype(float).values
                r, ok = _normalize_by_max_sm_clock(v, smx)
                if not ok:
                    continue
                s = pd.Series(r * 100.0, index=raw_s.index)
                leg_yl = "% @ max SM clock"
            else:
                s = raw_s
                leg_yl = ylabel

            mshort = str(mn) if len(str(mn)) <= 28 else str(mn)[:25] + "..."
            leg = f"il={int(il)} | {mshort}"
            series_rows.append((float(il), mn, s, leg))

    if not series_rows:
        return

    all_clocks = sorted({float(k) for _, _, s, _ in series_rows for k in s.index})
    n_c = len(all_clocks)
    n_m = len(series_rows)
    fig_w = float(max(11.0, min(26.0, 0.45 * n_c + 2.0 + 0.35 * n_m)))
    fig_h = 7.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    x = np.arange(n_c, dtype=float)
    cluster_w = 0.82
    bar_w = cluster_w / max(n_m, 1)
    ymax = 0.0

    for j, (_il, _mn, ser, leg) in enumerate(series_rows):
        val_map = {float(k): float(v) for k, v in ser.items()}
        offsets = x + (j - (n_m - 1) / 2.0) * bar_w
        heights_arr = np.array([val_map.get(float(c), np.nan) for c in all_clocks])
        heights_plot = np.where(np.isfinite(heights_arr), heights_arr, 0.0)
        if np.any(np.isfinite(heights_arr)):
            ymax = max(ymax, float(np.nanmax(heights_arr[np.isfinite(heights_arr)])))

        color = _combined_tab_color(j)
        bars = ax.bar(
            offsets,
            heights_plot,
            width=bar_w * 0.92,
            color=color,
            edgecolor="black",
            linewidth=0.35,
            label=leg,
        )
        fs = 4.5 if n_m > 8 else 5.5
        rfmt = LABEL_FMT.get(col, ".5g")
        for bar, h_raw in zip(bars, heights_arr):
            if np.isfinite(h_raw):
                if norm_mode == "pct":
                    t = f"{h_raw:.2f}%"
                elif norm_mode == "ratio":
                    t = f"{h_raw:.4f}"
                else:
                    t = f"{h_raw:{rfmt}}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    float(bar.get_height()),
                    t,
                    ha="center",
                    va="bottom",
                    fontsize=fs,
                    rotation=90,
                )

    ax.set_xticks(x)
    def _clk_lbl(c: float) -> str:
        cf = float(c)
        return str(int(cf)) if cf == int(cf) else str(cf)

    ax.set_xticklabels([_clk_lbl(c) for c in all_clocks], rotation=45, ha="right")
    ax.set_xlabel("SM Clock (MHz)", fontweight="bold")
    ax.set_ylabel(leg_yl, fontweight="bold")
    mode_txt = {
        None: "raw · all (input_len, model) groups",
        "ratio": "norm max SM=1 · combined",
        "pct": "norm % @ max SM · combined",
    }[norm_mode]
    ax.set_title(f"{title}  (batch_size={int(bs)}) — {mode_txt}", fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    if norm_mode == "pct":
        ax.axhline(100.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_ylim(0, max(105.0, ymax * 1.12) if ymax > 0 else (0, 1))
    elif norm_mode == "ratio":
        ax.set_ylim(0, max(1.08, ymax * 1.12) if ymax > 0 else (0, 1))
    else:
        ax.set_ylim(0, ymax * 1.12 if ymax > 0 else 1.0)
    ax.legend(
        title="input_len | model",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=5.5,
        framealpha=0.92,
    )
    fig.tight_layout()
    fname = f"hist_{sanitize(col)}_combined_bs{int(bs)}{fname_suffix}.png"
    fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {output_dir / fname}")


def _plot_per_model_figure(
    metrics: pd.DataFrame,
    mn: str,
    col: str,
    title: str,
    ylabel: str,
    output_dir: Path,
    batch_sizes: list,
    input_lens: list,
) -> None:
    combos_bm: list[tuple[float, float]] = []
    for bs in batch_sizes:
        for il in input_lens:
            sub = metrics[
                (metrics["batch_size"] == bs)
                & (metrics["input_len"] == il)
                & (metrics["model_name"] == mn)
            ]
            if not sub.empty:
                combos_bm.append((float(bs), float(il)))

    if not combos_bm:
        return

    ncols = min(len(combos_bm), 3)
    nrows = math.ceil(len(combos_bm) / ncols)
    fig_w = 6.5 * ncols + 1
    fig_h = 4.5 * nrows + 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    ax_flat = axes.ravel()

    for idx, (bs, il) in enumerate(combos_bm):
        ax = ax_flat[idx]
        sub = metrics[
            (metrics["batch_size"] == bs)
            & (metrics["input_len"] == il)
            & (metrics["model_name"] == mn)
        ]
        _plot_one_subplot_bars(ax, sub, col, mn, il, ylabel, None)

    for i in range(len(combos_bm), len(ax_flat)):
        ax_flat[i].set_visible(False)

    short_m = str(mn) if len(str(mn)) <= 60 else str(mn)[:57] + "..."
    fig.suptitle(f"{title}  ·  {short_m}", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fname = f"hist_{sanitize(col)}_model_{sanitize(mn)}.png"
    fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {output_dir / fname}")


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

            combos = _collect_il_mn_combos(sub_bs, input_lens, model_names)
            if not combos:
                continue

            _plot_bs_grid(sub_bs, bs, combos, col, title, ylabel, output_dir, None, "")
            _plot_bs_grid(sub_bs, bs, combos, col, title, ylabel, output_dir, "ratio", "_norm_maxsm1")
            _plot_bs_grid(sub_bs, bs, combos, col, title, ylabel, output_dir, "pct", "_norm_maxsm_pct")

            _plot_combined_single_ax(
                metrics, sub_bs, bs, col, title, ylabel, output_dir, None, "",
                input_lens, model_names,
            )
            _plot_combined_single_ax(
                metrics,
                sub_bs,
                bs,
                col,
                title,
                ylabel,
                output_dir,
                "ratio",
                "_norm_maxsm1",
                input_lens,
                model_names,
            )
            _plot_combined_single_ax(
                metrics,
                sub_bs,
                bs,
                col,
                title,
                ylabel,
                output_dir,
                "pct",
                "_norm_maxsm_pct",
                input_lens,
                model_names,
            )

        for mn in model_names:
            _plot_per_model_figure(
                metrics, mn, col, title, ylabel, output_dir, batch_sizes, input_lens
            )


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
    parser.add_argument(
        "--iteration-surrogate",
        type=str,
        default="segment",
        help=(
            "iteration 없을 때: segment(기본)=DATA_KEY 그룹에서 dt|len|kv factorize, "
            "during_time=(DATA_KEY+kv)에서 dt|len 만, index1=index==1 마다 (그룹 DATA_KEY+kv)"
        ),
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        raise NotADirectoryError(f"디렉터리가 아닙니다: {log_dir}")
    output_dir = Path(args.output_dir) if args.output_dir else (log_dir / "analysis_6metrics")

    print("1. CSV 로드...")
    sur = (args.iteration_surrogate or "").strip()
    df = load_csvs(log_dir, iteration_surrogate=sur if sur else "")
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
