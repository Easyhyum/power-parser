"""
CSV 로그를 읽어 energy / throughput / during_time 지표를
batch_size별로 (input_len 오름차순 × model_name) 조합마다 막대(히스토그램 스타일)로 그린다.

각 (model_name, batch_size, input_len)에서 최대 sm_clock(MHz) 지표를 100%로 두고,
다른 sm_clock은 그 기준 대비 퍼센트(소수 둘째 자리)로 y축에 표시한다.
막대 위에 동일한 수치를 라벨로 붙인다.

추가로, max sm_clock 정규화 없이 동일 레이아웃의 원시값 막대그래프(raw) PNG를 생성한다.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_csv_files(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*.csv") if p.is_file())


def sanitize_name(name: str) -> str:
    name = re.sub(r"[^\w\-]+", "_", name)
    return name.strip("_") or "unknown"


def load_data(log_dir: Path) -> pd.DataFrame:
    csv_files = find_csv_files(log_dir)
    if not csv_files:
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {log_dir}")

    dfs = []
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            df["__source_file"] = str(csv_path.relative_to(log_dir))
            dfs.append(df)
        except Exception as exc:
            print(f"CSV 로드 실패, 건너뜀: {csv_path} ({exc})")

    if not dfs:
        raise RuntimeError("읽을 수 있는 CSV가 없습니다.")

    return pd.concat(dfs, ignore_index=True)


def compute_all_metrics(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = [
        "batch_size",
        "input_len",
        "kv_cache_lens",
        "model_name",
        "sm_clock",
        "index",
        "length",
        "power",
        "during_time",
        "repeat_count",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"필수 컬럼이 없습니다: {col}")

    numeric_cols = [
        "batch_size",
        "input_len",
        "kv_cache_lens",
        "sm_clock",
        "index",
        "length",
        "power",
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
    valid_keys = kv_counts[kv_counts["kv_unique_cnt"] > 10][
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
            avg_power=("power", "mean"),
            avg_during_time=("during_time_per_repeat", "mean"),
            avg_repeat=("repeat_count", "mean"),
        )
    )

    agg["total_energy_power_based"] = (
        agg["avg_power"] * agg["avg_during_time"]
    )
    denom = agg["avg_repeat"] * agg["batch_size"]
    agg["energy_per_token_power_based"] = agg["total_energy_power_based"] / denom

    total_tokens = agg["avg_repeat"] * agg["batch_size"]
    total_time = agg["avg_during_time"] * agg["avg_repeat"]
    agg["throughput"] = np.where(total_time > 0, total_tokens / total_time, np.nan)

    agg["during_time_metric"] = agg["avg_during_time"]

    agg = agg[
        np.isfinite(agg["energy_per_token_power_based"])
        & np.isfinite(agg["throughput"])
        & np.isfinite(agg["during_time_metric"])
    ].copy()

    return agg


def sort_model_names_for_plot(names: List[str]) -> List[str]:
    """
    플롯 서브플롯 순서:
    1) 이름에 FP/Int(대소문자 무관)가 없는 모델
    2) FP8 → NVFP4 → Int8 → Int4
    3) 그 외 (알파벳 순)
    """

    def sort_key(name: str) -> tuple:
        s = str(name)
        lo = s.lower()
        if "fp" not in lo and "int" not in lo:
            return (0, s)
        if "fp8" in lo:
            return (1, s)
        if "nvfp4" in lo:
            return (2, s)
        if "int8" in lo:
            return (3, s)
        if "int4" in lo:
            return (4, s)
        return (5, s)

    return sorted(names, key=sort_key)


def model_name_bar_color(name: str) -> str:
    """
    범례/막대 색: FP·Int 없음 회색, FP8 진한 빨강, NVFP4 연한 빨강,
    Int8 진한 파랑, Int4 연한 파랑, 그 외는 청회색.
    """

    s = str(name)
    lo = s.lower()
    if "fp" not in lo and "int" not in lo:
        return "#808080"
    if "fp8" in lo:
        return "#C62828"
    if "nvfp4" in lo:
        return "#FFAB91"
    if "int8" in lo:
        return "#1565C0"
    if "int4" in lo:
        return "#90CAF9"
    return "#78909C"


def _ilen_label(input_len: float) -> str:
    f = float(input_len)
    return str(int(f)) if f.is_integer() else str(f)


def _sm_clock_series_for_batch_model_input_len(
    metrics: pd.DataFrame,
    batch_size: float,
    model_name: str,
    input_len: float,
    value_col: str,
) -> pd.Series:
    sub = metrics[
        (metrics["batch_size"] == batch_size)
        & (metrics["model_name"] == model_name)
        & (metrics["input_len"] == input_len)
    ]
    if sub.empty:
        return pd.Series(dtype=float)
    return sub.groupby("sm_clock", as_index=True)[value_col].mean().sort_index()


def _normalized_pct_by_sm_clock(
    metrics: pd.DataFrame,
    batch_size: float,
    model_name: str,
    value_col: str,
    input_len: float,
) -> Optional[pd.Series]:
    s = _sm_clock_series_for_batch_model_input_len(
        metrics, batch_size, model_name, input_len, value_col
    )
    if s.empty:
        return None
    sm_max = float(s.index.max())
    baseline = float(s.loc[sm_max])
    if not np.isfinite(baseline) or baseline == 0:
        return None
    return (s / baseline) * 100.0


def _ylabel_for_raw_value_col(value_col: str) -> str:
    if value_col == "throughput":
        return "Throughput (tokens/sec)"
    if value_col == "during_time_metric":
        return "During time per repeat (sec)"
    return "Energy per token (J/token, power-based)"


MIN_BAR_LIGHT_BLUE = "#B3E5FC"


def _raw_bar_text(v: float) -> str:
    return f"{v:.5f}"


def _raw_bar_colors_min_light_blue(
    base_color: str, heights: np.ndarray
) -> List[str]:
    """유한값 중 최소 높이 막대만 연한 파랑, 나머지는 base_color."""
    colors = [base_color] * len(heights)
    valid = np.isfinite(heights)
    if not valid.any():
        return colors
    idxs = np.flatnonzero(valid)
    rel_min = int(np.argmin(heights[idxs]))
    colors[int(idxs[rel_min])] = MIN_BAR_LIGHT_BLUE
    return colors


def plot_raw_bars_by_batch(
    metrics: pd.DataFrame,
    value_col: str,
    output_dir: Path,
    file_prefix: str,
    plot_title_metric: str,
    ncols: int = 2,
) -> None:
    """
    batch_size별 figure. 서브플롯 = (input_len 오름차순 × model_name 정렬) 조합.
    y축은 max sm_clock 정규화 없이 sm_clock별 지표 원시값(kv는 동일 키 내 평균).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ylabel = _ylabel_for_raw_value_col(value_col)

    batch_sizes = sorted(metrics["batch_size"].dropna().unique())
    input_lens = sorted(metrics["input_len"].dropna().unique())
    model_names = sort_model_names_for_plot(
        list(metrics["model_name"].dropna().unique())
    )

    if not batch_sizes:
        raise RuntimeError("plot할 batch_size 데이터가 없습니다.")

    for bs in batch_sizes:
        pairs_with_data: List[tuple[float, str]] = []
        for ilen in input_lens:
            for mn in model_names:
                s = _sm_clock_series_for_batch_model_input_len(
                    metrics, bs, mn, float(ilen), value_col
                )
                if not s.empty:
                    pairs_with_data.append((float(ilen), mn))

        if not pairs_with_data:
            print(f"데이터 없음, 건너뜀 (raw): {file_prefix} batch_size={bs}")
            continue

        n = len(pairs_with_data)
        nrows = int(math.ceil(n / ncols))
        fig_w = 5.5 * ncols + 1
        fig_h = 3.8 * nrows + 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
        ax_flat = axes.ravel()

        for idx, (input_len, model_name) in enumerate(pairs_with_data):
            ax = ax_flat[idx]
            s = _sm_clock_series_for_batch_model_input_len(
                metrics, bs, model_name, input_len, value_col
            )
            clocks = s.index.to_numpy()
            heights = s.to_numpy(dtype=float)
            x = np.arange(len(clocks))
            bar_colors = _raw_bar_colors_min_light_blue(
                model_name_bar_color(model_name), heights
            )
            bars = ax.bar(
                x,
                heights,
                color=bar_colors,
                edgecolor="black",
                linewidth=0.6,
            )

            ax.set_xticks(x)
            ax.set_xticklabels([str(int(c)) for c in clocks], rotation=45, ha="right")
            ax.set_xlabel("SM clock (MHz)")
            ax.set_ylabel(ylabel)
            short = str(model_name)
            if len(short) > 40:
                short = short[:37] + "..."
            ax.set_title(
                f"input_len={_ilen_label(input_len)}\n{short}",
                fontsize=9,
            )
            ax.grid(True, axis="y", alpha=0.3)
            if len(heights) and np.isfinite(heights).any():
                ax.set_ylim(
                    0,
                    float(np.nanmax(heights[np.isfinite(heights)])) * 1.12,
                )

            for bar, h in zip(bars, heights):
                if np.isfinite(h):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bar.get_height(),
                        _raw_bar_text(float(h)),
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        for j in range(len(pairs_with_data), len(ax_flat)):
            ax_flat[j].set_visible(False)

        fig.suptitle(
            f"{plot_title_metric} — raw by SM clock (batch_size={int(bs)}); "
            f"subplots: input_len ↑ then model; min bar = light blue",
            fontsize=12,
            y=1.02,
        )
        fig.tight_layout()
        out_path = output_dir / f"{file_prefix}_raw_bs{int(bs)}.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"저장 완료: {out_path}")


def plot_raw_bars_combined_by_batch(
    metrics: pd.DataFrame,
    value_col: str,
    output_dir: Path,
    file_prefix: str,
    plot_title_metric: str,
) -> None:
    """
    한 figure에 (input_len ↑ × model) 순서의 그룹 막대. y는 원시값.
    특정 sm_clock에 값이 없으면 막대 높이 0, 라벨 생략.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ylabel = _ylabel_for_raw_value_col(value_col)

    batch_sizes = sorted(metrics["batch_size"].dropna().unique())
    input_lens = sorted(metrics["input_len"].dropna().unique())
    model_names = sort_model_names_for_plot(
        list(metrics["model_name"].dropna().unique())
    )

    if not batch_sizes:
        raise RuntimeError("plot할 batch_size 데이터가 없습니다.")

    for bs in batch_sizes:
        series_rows: List[tuple[float, str, pd.Series, str]] = []
        for ilen in input_lens:
            for mn in model_names:
                s = _sm_clock_series_for_batch_model_input_len(
                    metrics, bs, mn, float(ilen), value_col
                )
                if not s.empty:
                    mshort = str(mn) if len(str(mn)) <= 28 else str(mn)[:25] + "..."
                    leg = f"ilen {_ilen_label(float(ilen))} | {mshort}"
                    series_rows.append((float(ilen), mn, s, leg))

        if not series_rows:
            print(f"데이터 없음, 건너뜀 (raw combined): {file_prefix} batch_size={bs}")
            continue

        all_clocks_set: set[float] = set()
        for _, _, s, _ in series_rows:
            all_clocks_set.update(float(k) for k in s.index)
        all_clocks = sorted(all_clocks_set)

        n_c = len(all_clocks)
        n_m = len(series_rows)

        fig_w = float(max(11.0, min(26.0, 0.45 * n_c + 2.0 + 0.35 * n_m)))
        fig_h = 7.5
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        x = np.arange(n_c, dtype=float)
        cluster_w = 0.82
        bar_w = cluster_w / max(n_m, 1)

        ymax = 0.0
        for j, (_ilen, mn, raw_s, leg) in enumerate(series_rows):
            val_map = {float(k): float(v) for k, v in raw_s.items()}
            offsets = x + (j - (n_m - 1) / 2.0) * bar_w
            heights_arr = np.array([val_map.get(float(c), np.nan) for c in all_clocks])
            heights_plot = np.where(np.isfinite(heights_arr), heights_arr, 0.0)
            if np.any(np.isfinite(heights_arr)):
                ymax = max(
                    ymax,
                    float(np.nanmax(heights_arr[np.isfinite(heights_arr)])),
                )

            base_color = model_name_bar_color(mn)
            bar_colors = _raw_bar_colors_min_light_blue(base_color, heights_arr)
            bars = ax.bar(
                offsets,
                heights_plot,
                width=bar_w * 0.92,
                color=bar_colors,
                edgecolor="black",
                linewidth=0.35,
                label=leg,
            )
            for bar, h_raw in zip(bars, heights_arr):
                if np.isfinite(h_raw):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        float(h_raw),
                        _raw_bar_text(float(h_raw)),
                        ha="center",
                        va="bottom",
                        fontsize=4.5,
                        rotation=90,
                    )

        ax.set_xticks(x)
        tick_labels = []
        for c in all_clocks:
            cf = float(c)
            tick_labels.append(str(int(cf)) if cf.is_integer() else str(cf))
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.set_xlabel("SM clock (MHz)")
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"{plot_title_metric} — raw (batch_size={int(bs)}); "
            f"groups: input_len ↑ then model; min bar per group = light blue"
        )
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(0, ymax * 1.12 if ymax > 0 else 1.0)
        ax.legend(
            title="input_len | model_name",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=5.5,
            framealpha=0.92,
        )
        fig.tight_layout()
        out_path = output_dir / f"{file_prefix}_raw_combined_bs{int(bs)}.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"저장 완료: {out_path}")


def plot_normalized_bars_by_batch(
    metrics: pd.DataFrame,
    value_col: str,
    output_dir: Path,
    file_prefix: str,
    plot_title_metric: str,
    ncols: int = 2,
) -> None:
    """
    각 batch_size마다 한 figure. 서브플롯 = (input_len 오름차순 × model_name 정렬).
    y = 해당 (배치, input_len, 모델)에서 max(sm_clock) 대비 %.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_sizes = sorted(metrics["batch_size"].dropna().unique())
    input_lens = sorted(metrics["input_len"].dropna().unique())
    model_names = sort_model_names_for_plot(
        list(metrics["model_name"].dropna().unique())
    )

    if not batch_sizes:
        raise RuntimeError("plot할 batch_size 데이터가 없습니다.")

    for bs in batch_sizes:
        pairs_with_data: List[tuple[float, str]] = []
        for ilen in input_lens:
            for mn in model_names:
                s = _sm_clock_series_for_batch_model_input_len(
                    metrics, bs, mn, float(ilen), value_col
                )
                if not s.empty:
                    pairs_with_data.append((float(ilen), mn))

        if not pairs_with_data:
            print(f"데이터 없음, 건너뜀: {file_prefix} batch_size={bs}")
            continue

        n = len(pairs_with_data)
        nrows = int(math.ceil(n / ncols))
        fig_w = 5.5 * ncols + 1
        fig_h = 3.8 * nrows + 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
        ax_flat = axes.ravel()

        for idx, (input_len, model_name) in enumerate(pairs_with_data):
            ax = ax_flat[idx]
            s = _sm_clock_series_for_batch_model_input_len(
                metrics, bs, model_name, input_len, value_col
            )
            sm_max = float(s.index.max())
            baseline = float(s.loc[sm_max])
            if not np.isfinite(baseline) or baseline == 0:
                ax.set_title(
                    f"ilen={_ilen_label(input_len)}\n{model_name}\n(no valid baseline)"
                )
                ax.set_visible(True)
                continue

            pct = (s / baseline) * 100.0
            clocks = s.index.to_numpy()
            heights = pct.to_numpy()
            x = np.arange(len(clocks))
            bars = ax.bar(
                x,
                heights,
                color=model_name_bar_color(model_name),
                edgecolor="black",
                linewidth=0.6,
            )

            ax.set_xticks(x)
            ax.set_xticklabels([str(int(c)) for c in clocks], rotation=45, ha="right")
            ax.set_xlabel("SM clock (MHz)")
            ax.set_ylabel("% of metric at max SM clock")
            ax.axhline(100.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
            short = str(model_name)
            if len(short) > 40:
                short = short[:37] + "..."
            ax.set_title(
                f"input_len={_ilen_label(input_len)}\n{short}\n"
                f"Baseline: max sm_clock={int(sm_max)} MHz (=100.00%)",
                fontsize=9,
            )
            ax.grid(True, axis="y", alpha=0.3)
            ymax = max(105.0, float(np.nanmax(heights)) * 1.12 if len(heights) else 105.0)
            ax.set_ylim(0, ymax)

            for bar, h in zip(bars, heights):
                label = f"{h:.2f}%"
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height(),
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        for j in range(len(pairs_with_data), len(ax_flat)):
            ax_flat[j].set_visible(False)

        fig.suptitle(
            f"{plot_title_metric} — normalized vs max SM clock (batch_size={int(bs)}); "
            f"subplots: input_len ↑ then model",
            fontsize=12,
            y=1.02,
        )
        fig.tight_layout()
        out_path = output_dir / f"{file_prefix}_bs{int(bs)}.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"저장 완료: {out_path}")


def plot_normalized_bars_combined_by_batch(
    metrics: pd.DataFrame,
    value_col: str,
    output_dir: Path,
    file_prefix: str,
    plot_title_metric: str,
) -> None:
    """
    batch_size마다 한 figure. 그룹 막대 순서: input_len 오름차순 × model_name 정렬.
    각 (input_len, model) 조합은 자기 max sm_clock 기준 100% 정규화.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_sizes = sorted(metrics["batch_size"].dropna().unique())
    input_lens = sorted(metrics["input_len"].dropna().unique())
    model_names = sort_model_names_for_plot(
        list(metrics["model_name"].dropna().unique())
    )

    if not batch_sizes:
        raise RuntimeError("plot할 batch_size 데이터가 없습니다.")

    for bs in batch_sizes:
        series_rows: List[tuple[float, str, pd.Series, str]] = []
        for ilen in input_lens:
            for mn in model_names:
                pct = _normalized_pct_by_sm_clock(
                    metrics, bs, mn, value_col, float(ilen)
                )
                if pct is not None:
                    mshort = str(mn) if len(str(mn)) <= 28 else str(mn)[:25] + "..."
                    leg = f"ilen {_ilen_label(float(ilen))} | {mshort}"
                    series_rows.append((float(ilen), mn, pct, leg))

        if not series_rows:
            print(f"데이터 없음, 건너뜀 (combined): {file_prefix} batch_size={bs}")
            continue

        all_clocks_set: set[float] = set()
        for _, _, s, _ in series_rows:
            all_clocks_set.update(float(k) for k in s.index)
        all_clocks = sorted(all_clocks_set)

        n_c = len(all_clocks)
        n_m = len(series_rows)

        fig_w = float(max(11.0, min(26.0, 0.45 * n_c + 2.0 + 0.35 * n_m)))
        fig_h = 7.5
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        x = np.arange(n_c, dtype=float)
        cluster_w = 0.82
        bar_w = cluster_w / max(n_m, 1)

        ymax = 105.0
        for j, (_ilen, mn, pct_s, leg) in enumerate(series_rows):
            pct_map = {float(k): float(v) for k, v in pct_s.items()}
            offsets = x + (j - (n_m - 1) / 2.0) * bar_w
            heights_arr = np.array([pct_map.get(float(c), 0.0) for c in all_clocks])
            if np.any(np.isfinite(heights_arr)):
                ymax = max(
                    ymax,
                    float(np.nanmax(heights_arr[np.isfinite(heights_arr)])) * 1.12,
                )

            color = model_name_bar_color(mn)
            bars = ax.bar(
                offsets,
                heights_arr,
                width=bar_w * 0.92,
                color=color,
                edgecolor="black",
                linewidth=0.35,
                label=leg,
            )
            for bar, h in zip(bars, heights_arr):
                if h > 0.05 and np.isfinite(h):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        h,
                        f"{h:.2f}%",
                        ha="center",
                        va="bottom",
                        fontsize=4.5,
                        rotation=90,
                    )

        ax.set_xticks(x)
        tick_labels = []
        for c in all_clocks:
            cf = float(c)
            tick_labels.append(str(int(cf)) if cf.is_integer() else str(cf))
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.set_xlabel("SM clock (MHz)")
        ax.set_ylabel("% of metric at max SM clock")
        ax.set_title(
            f"{plot_title_metric} — combined (batch_size={int(bs)}); "
            f"groups: input_len ↑ then model; each group max sm_clock = 100%"
        )
        ax.axhline(100.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(0, ymax)
        ax.legend(
            title="input_len | model_name",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=5.5,
            framealpha=0.92,
        )
        fig.tight_layout()
        out_path = output_dir / f"{file_prefix}_combined_bs{int(bs)}.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"저장 완료: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "에너지/처리량/during_time을 batch_size별 막대그래프로 출력. "
            "모델 내 최고 sm_clock(MHz)에서의 지표를 100%로 정규화. "
            "동일 레이아웃의 원시값(raw) PNG도 추가 생성."
        )
    )
    parser.add_argument(
        "log_dir",
        type=str,
        help="CSV 루트 디렉터리 (하위 폴더까지 검색)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="출력 디렉터리 (기본: <log_dir>/analysis_output_hist_max_sm_clock)",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        raise NotADirectoryError(f"디렉터리가 아닙니다: {log_dir}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (log_dir / "analysis_output_hist_max_sm_clock")
    )

    df = load_data(log_dir)
    metrics = compute_all_metrics(df)

    plot_normalized_bars_by_batch(
        metrics,
        "energy_per_token_power_based",
        output_dir,
        "hist_energy_per_token_power_based",
        "Energy per token (power-based, J/token)",
    )
    plot_normalized_bars_by_batch(
        metrics,
        "throughput",
        output_dir,
        "hist_throughput",
        "Throughput (tokens/sec)",
    )
    plot_normalized_bars_by_batch(
        metrics,
        "during_time_metric",
        output_dir,
        "hist_during_time",
        "During time per repeat (sec)",
    )

    plot_normalized_bars_combined_by_batch(
        metrics,
        "energy_per_token_power_based",
        output_dir,
        "hist_energy_per_token_power_based",
        "Energy per token (power-based, J/token)",
    )
    plot_normalized_bars_combined_by_batch(
        metrics,
        "throughput",
        output_dir,
        "hist_throughput",
        "Throughput (tokens/sec)",
    )
    plot_normalized_bars_combined_by_batch(
        metrics,
        "during_time_metric",
        output_dir,
        "hist_during_time",
        "During time per repeat (sec)",
    )

    plot_raw_bars_by_batch(
        metrics,
        "energy_per_token_power_based",
        output_dir,
        "hist_energy_per_token_power_based",
        "Energy per token (power-based, J/token)",
    )
    plot_raw_bars_by_batch(
        metrics,
        "throughput",
        output_dir,
        "hist_throughput",
        "Throughput (tokens/sec)",
    )
    plot_raw_bars_by_batch(
        metrics,
        "during_time_metric",
        output_dir,
        "hist_during_time",
        "During time per repeat (sec)",
    )
    plot_raw_bars_combined_by_batch(
        metrics,
        "energy_per_token_power_based",
        output_dir,
        "hist_energy_per_token_power_based",
        "Energy per token (power-based, J/token)",
    )
    plot_raw_bars_combined_by_batch(
        metrics,
        "throughput",
        output_dir,
        "hist_throughput",
        "Throughput (tokens/sec)",
    )
    plot_raw_bars_combined_by_batch(
        metrics,
        "during_time_metric",
        output_dir,
        "hist_during_time",
        "During time per repeat (sec)",
    )


if __name__ == "__main__":
    main()
