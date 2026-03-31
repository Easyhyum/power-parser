import argparse
import re
from pathlib import Path
from typing import List, Tuple

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


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
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
            avg_power=("power", "mean"),
            avg_during_time=("during_time_per_repeat", "mean"),
            avg_repeat=("repeat_count", "mean"),
        )
    )

    agg["total_energy_power_based"] = (
        agg["avg_power"] * agg["avg_during_time"] * agg["avg_repeat"]
    )

    denom = agg["avg_repeat"] * agg["batch_size"]
    agg["energy_per_token_power_based"] = agg["total_energy_power_based"] / denom

    total_tokens = agg["avg_repeat"] * agg["batch_size"]
    total_time = agg["avg_during_time"] * agg["avg_repeat"]
    agg["throughput"] = np.where(total_time > 0, total_tokens / total_time, np.nan)
    return agg


def model_sort_key(model_name: str) -> Tuple[int, str]:
    """
    범례/그리기 순서: FP·Int 미포함 → FP8 → NVFP4 → Int8 → Int4 → 기타.
    """
    s = str(model_name)
    if "FP" not in s and "Int" not in s:
        return (0, s)
    if "FP8" in s:
        return (1, s)
    if "NVFP4" in s:
        return (2, s)
    if "Int8" in s:
        return (3, s)
    if "Int4" in s:
        return (4, s)
    return (5, s)


def model_color(model_name: str) -> str:
    s = str(model_name)
    if "FP" not in s and "Int" not in s:
        return "#808080"
    if "FP8" in s:
        return "#c62828"
    if "NVFP4" in s:
        return "#ef9a9a"
    if "Int8" in s:
        return "#1565c0"
    if "Int4" in s:
        return "#90caf9"
    return "#a0a0a0"


def _sm_clock_slug(sm_clock_min: int, sm_clock_max: int) -> str:
    """파일명용: 단일 클럭이면 max_N, 구간이면 min_max."""
    imin, imax = int(sm_clock_min), int(sm_clock_max)
    if imin == imax:
        return f"max_{imin}"
    return f"{imin}_{imax}"


def _sm_clock_title_range(sm_clock_min: int, sm_clock_max: int) -> str:
    """제목용: 단일 클럭이면 SM clock = N MHz."""
    imin, imax = int(sm_clock_min), int(sm_clock_max)
    if imin == imax:
        return f"SM clock = {imin} MHz"
    return f"SM clock {imin}–{imax} MHz"


def plot_energy_per_token_by_batch_sm_clock_band(
    metrics: pd.DataFrame,
    output_dir: Path,
    sm_clock_min: int = 1200,
    sm_clock_max: int = 1250,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    band = metrics[
        (metrics["sm_clock"] >= sm_clock_min) & (metrics["sm_clock"] <= sm_clock_max)
    ]
    if band.empty:
        raise RuntimeError(
            f"sm_clock {sm_clock_min}~{sm_clock_max} MHz 구간에 해당하는 행이 없습니다."
        )

    model_names = sorted(band["model_name"].dropna().unique(), key=model_sort_key)

    plt.figure(figsize=(10, 6))
    has_data = False

    for model_name in model_names:
        sub = band[band["model_name"] == model_name]
        if sub.empty:
            continue

        line = (
            sub.groupby("batch_size", as_index=False)["energy_per_token_power_based"]
            .mean()
            .sort_values("batch_size")
        )
        if line.empty:
            continue

        has_data = True
        plt.plot(
            line["batch_size"],
            line["energy_per_token_power_based"],
            marker="o",
            linewidth=1.8,
            color=model_color(model_name),
            label=str(model_name),
        )

    if not has_data:
        plt.close()
        raise RuntimeError("플롯할 데이터가 없습니다.")

    plt.xlabel("batch_size")
    plt.ylabel("Energy per token (J/token, power-based)")
    plt.yscale("log")
    clk = _sm_clock_title_range(sm_clock_min, sm_clock_max)
    plt.title(f"Energy per token (power-based) by model ({clk}, averaged)")
    plt.grid(True, alpha=0.3)
    plt.legend(title="model_name")
    plt.tight_layout()

    out_path = (
        output_dir
        / f"energy_per_token_power_based_sm_clock_{_sm_clock_slug(sm_clock_min, sm_clock_max)}_x_batch.png"
    )
    plt.savefig(out_path)
    plt.close()
    print(f"저장 완료: {out_path}")


def plot_throughput_by_batch_sm_clock_band(
    metrics: pd.DataFrame,
    output_dir: Path,
    sm_clock_min: int = 1200,
    sm_clock_max: int = 1250,
) -> None:
    """plot_throughput_power_based_by_batch.py 와 동일 정의의 throughput, 동일 스타일(구간·색·정렬)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    band = metrics[
        (metrics["sm_clock"] >= sm_clock_min) & (metrics["sm_clock"] <= sm_clock_max)
    ]
    if band.empty:
        raise RuntimeError(
            f"sm_clock {sm_clock_min}~{sm_clock_max} MHz 구간에 해당하는 행이 없습니다."
        )

    model_names = sorted(band["model_name"].dropna().unique(), key=model_sort_key)

    plt.figure(figsize=(10, 6))
    has_data = False

    for model_name in model_names:
        sub = band[band["model_name"] == model_name]
        if sub.empty:
            continue

        line = (
            sub.groupby("batch_size", as_index=False)["throughput"]
            .mean()
            .sort_values("batch_size")
        )
        line = line[np.isfinite(line["throughput"])]
        if line.empty:
            continue

        has_data = True
        plt.plot(
            line["batch_size"],
            line["throughput"],
            marker="o",
            linewidth=1.8,
            color=model_color(model_name),
            label=str(model_name),
        )

    if not has_data:
        plt.close()
        raise RuntimeError("Throughput 플롯할 데이터가 없습니다.")

    plt.xlabel("batch_size")
    plt.ylabel("Throughput (tokens/sec)")
    clk = _sm_clock_title_range(sm_clock_min, sm_clock_max)
    plt.title(f"Throughput by model ({clk}, averaged)")
    plt.grid(True, alpha=0.3)
    plt.legend(title="model_name")
    plt.tight_layout()

    out_path = (
        output_dir
        / f"throughput_sm_clock_{_sm_clock_slug(sm_clock_min, sm_clock_max)}_x_batch.png"
    )
    plt.savefig(out_path)
    plt.close()
    print(f"저장 완료: {out_path}")


def plot_avg_power_by_batch_sm_clock_band(
    metrics: pd.DataFrame,
    output_dir: Path,
    sm_clock_min: int = 1200,
    sm_clock_max: int = 1250,
) -> None:
    """sm_clock 밴드 내 평균 소비전력(W), x=batch_size, 범례=model_name."""
    output_dir.mkdir(parents=True, exist_ok=True)

    band = metrics[
        (metrics["sm_clock"] >= sm_clock_min) & (metrics["sm_clock"] <= sm_clock_max)
    ]
    if band.empty:
        raise RuntimeError(
            f"sm_clock {sm_clock_min}~{sm_clock_max} MHz 구간에 해당하는 행이 없습니다."
        )

    band = band[np.isfinite(band["avg_power"])].copy()
    if band.empty:
        raise RuntimeError("avg_power 데이터가 없습니다.")

    model_names = sorted(band["model_name"].dropna().unique(), key=model_sort_key)

    plt.figure(figsize=(10, 6))
    has_data = False

    for model_name in model_names:
        sub = band[band["model_name"] == model_name]
        if sub.empty:
            continue

        line = (
            sub.groupby("batch_size", as_index=False)["avg_power"]
            .mean()
            .sort_values("batch_size")
        )
        if line.empty:
            continue

        has_data = True
        plt.plot(
            line["batch_size"],
            line["avg_power"],
            marker="o",
            linewidth=1.8,
            color=model_color(model_name),
            label=str(model_name),
        )

    if not has_data:
        plt.close()
        raise RuntimeError("평균 전력 플롯할 데이터가 없습니다.")

    plt.xlabel("batch_size")
    plt.ylabel("Average power (W)")
    clk = _sm_clock_title_range(sm_clock_min, sm_clock_max)
    plt.title(f"Average power by model ({clk}, averaged)")
    plt.grid(True, alpha=0.3)
    plt.legend(title="model_name")
    plt.tight_layout()

    out_path = (
        output_dir
        / f"avg_power_sm_clock_{_sm_clock_slug(sm_clock_min, sm_clock_max)}_x_batch.png"
    )
    plt.savefig(out_path)
    plt.close()
    print(f"저장 완료: {out_path}")


def plot_time_per_batch_generate_by_batch_sm_clock_band(
    metrics: pd.DataFrame,
    output_dir: Path,
    sm_clock_min: int = 1200,
    sm_clock_max: int = 1250,
) -> None:
    """
    repeat 1회당 wall time(avg_during_time) = batch_size 토큰을 한 번에 생성하는 구간 소요 시간(초).
    x=batch_size, 범례=model_name.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    band = metrics[
        (metrics["sm_clock"] >= sm_clock_min) & (metrics["sm_clock"] <= sm_clock_max)
    ]
    if band.empty:
        raise RuntimeError(
            f"sm_clock {sm_clock_min}~{sm_clock_max} MHz 구간에 해당하는 행이 없습니다."
        )

    band = band[np.isfinite(band["avg_during_time"])].copy()
    band = band[band["avg_during_time"] > 0].copy()
    if band.empty:
        raise RuntimeError("avg_during_time 데이터가 없습니다.")

    model_names = sorted(band["model_name"].dropna().unique(), key=model_sort_key)

    plt.figure(figsize=(10, 6))
    has_data = False

    for model_name in model_names:
        sub = band[band["model_name"] == model_name]
        if sub.empty:
            continue

        line = (
            sub.groupby("batch_size", as_index=False)["avg_during_time"]
            .mean()
            .sort_values("batch_size")
        )
        if line.empty:
            continue

        has_data = True
        plt.plot(
            line["batch_size"],
            line["avg_during_time"],
            marker="o",
            linewidth=1.8,
            color=model_color(model_name),
            label=str(model_name),
        )

    if not has_data:
        plt.close()
        raise RuntimeError("batch 생성 소요 시간 플롯할 데이터가 없습니다.")

    plt.xlabel("batch_size")
    plt.ylabel("Wall time per repeat (s)")
    clk = _sm_clock_title_range(sm_clock_min, sm_clock_max)
    plt.title(
        f"Time to generate batch_size tokens (per repeat) by model ({clk}, averaged)"
    )
    plt.grid(True, alpha=0.3)
    plt.legend(title="model_name")
    plt.tight_layout()

    out_path = (
        output_dir
        / f"time_per_batch_generate_sm_clock_{_sm_clock_slug(sm_clock_min, sm_clock_max)}_x_batch.png"
    )
    plt.savefig(out_path)
    plt.close()
    print(f"저장 완료: {out_path}")


def plot_energy_hist_ratio_to_baseline_model_sm_clock_band(
    metrics: pd.DataFrame,
    output_dir: Path,
    sm_clock_min: int = 1200,
    sm_clock_max: int = 1250,
    baseline_model_query: str = "Qwen/Qwen3-8B",
    hist_bins: int = 30,
) -> None:
    """
    sm_clock 지정 구간(기본 1200~1250 MHz)에서 energy_per_token_power_based를
    model_name x batch_size 별로 평균 낸 뒤,
    baseline_model_query(Qwen/Qwen3-8B)를 batch_size별로 100%로 두고
    model_name별 비율(%)을 x축=batch_size 막대(히스토그램 스타일)로 그립니다.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    band = metrics[
        (metrics["sm_clock"] >= sm_clock_min) & (metrics["sm_clock"] <= sm_clock_max)
    ].copy()
    if band.empty:
        raise RuntimeError(
            f"sm_clock {sm_clock_min}~{sm_clock_max} MHz 구간에 해당하는 행이 없습니다."
        )

    band = band[np.isfinite(band["energy_per_token_power_based"])].copy()
    band = band[band["energy_per_token_power_based"] > 0].copy()
    if band.empty:
        raise RuntimeError("양수 energy_per_token_power_based 데이터가 없습니다.")

    batch_sizes = sorted(band["batch_size"].dropna().unique())
    if not batch_sizes:
        raise RuntimeError("plot할 batch_size 데이터가 없습니다.")

    model_names = sorted(band["model_name"].dropna().unique(), key=model_sort_key)

    grouped = (
        band.groupby(["model_name", "batch_size"], as_index=False)[
            "energy_per_token_power_based"
        ]
        .mean()
    )

    model_str = grouped["model_name"].astype(str)
    baseline_mask = model_str.str.contains(baseline_model_query, regex=False, na=False)
    baseline_df = grouped.loc[baseline_mask].copy()
    if baseline_df.empty:
        raise RuntimeError(
            f"baseline 모델 '{baseline_model_query}'를 포함한 model_name 데이터를 찾을 수 없습니다."
        )

    baseline_per_batch = (
        baseline_df.groupby("batch_size", as_index=True)["energy_per_token_power_based"]
        .mean()
    )
    baseline_per_batch = baseline_per_batch.replace(0, np.nan)

    # baseline이 없는 batch_size는 ratio 계산에서 제외
    valid_batch_sizes = [bs for bs in batch_sizes if bs in baseline_per_batch.index]
    if not valid_batch_sizes:
        raise RuntimeError("baseline이 존재하는 batch_size가 없습니다.")

    x = np.arange(len(valid_batch_sizes), dtype=float)
    n_models = 0
    model_ratios: list[tuple[str, np.ndarray]] = []

    for model_name in model_names:
        sub = grouped[grouped["model_name"] == model_name]
        if sub.empty:
            continue
        series = sub.set_index("batch_size")["energy_per_token_power_based"]
        ratio_pct = np.array(
            [
                (series.loc[bs] / baseline_per_batch.loc[bs]) * 100.0
                if (bs in series.index) and np.isfinite(baseline_per_batch.loc[bs])
                else np.nan
                for bs in valid_batch_sizes
            ],
            dtype=float,
        )
        if np.isfinite(ratio_pct).any():
            n_models += 1
            model_ratios.append((str(model_name), ratio_pct))

    if n_models == 0:
        plt.close()
        raise RuntimeError("baseline 대비 비율 막대 플롯할 데이터가 없습니다.")

    fig, ax = plt.subplots(figsize=(12, 6))
    bar_total_width = 0.85
    bar_width = bar_total_width / max(n_models, 1)

    all_finite_heights: list[float] = []
    for j, (model_name, ratio_pct) in enumerate(model_ratios):
        offsets = (j - (n_models - 1) / 2.0) * bar_width
        bars = ax.bar(
            x + offsets,
            ratio_pct,
            width=bar_width * 0.95,
            color=model_color(model_name),
            edgecolor="black",
            linewidth=0.5,
            label=model_name,
            alpha=0.95,
        )
        for bar, h in zip(bars, ratio_pct):
            if np.isfinite(h):
                all_finite_heights.append(float(h))
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height(),
                    f"{h:.2f}%",
                    ha="center",
                    va="bottom",
                    fontsize=5.5,
                    rotation=0,
                )

    ax.axhline(100.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xlabel("batch_size")
    ax.set_ylabel("Energy per token ratio to baseline (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [str(int(bs)) if float(bs).is_integer() else str(bs) for bs in valid_batch_sizes],
        rotation=0,
    )
    clk = _sm_clock_title_range(sm_clock_min, sm_clock_max)
    ax.set_title(
        f"Energy per token ratio to baseline ({baseline_model_query}) by model ({clk})"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="model_name", fontsize=8)
    if all_finite_heights:
        ymax = max(105.0, max(all_finite_heights) * 1.12)
        ax.set_ylim(0, ymax)
    fig.tight_layout()

    safe_baseline = sanitize_name(baseline_model_query)
    out_path = (
        output_dir
        / f"hist_energy_per_token_power_based_ratio_to_{safe_baseline}_sm_clock_{_sm_clock_slug(sm_clock_min, sm_clock_max)}_x_batch.png"
    )
    fig.savefig(out_path)
    plt.close(fig)
    print(f"저장 완료: {out_path}")


def plot_throughput_hist_ratio_to_baseline_model_sm_clock_band(
    metrics: pd.DataFrame,
    output_dir: Path,
    sm_clock_min: int = 1200,
    sm_clock_max: int = 1250,
    baseline_model_query: str = "Qwen/Qwen3-8B",
) -> None:
    """
    sm_clock 지정 구간(기본 1200~1250 MHz)에서 throughput을
    model_name x batch_size 별로 평균 낸 뒤,
    baseline_model_query(Qwen/Qwen3-8B)를 batch_size별로 100%로 두고
    model_name별 비율(%)을 x축=batch_size 막대(히스토그램 스타일)로 그립니다.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    band = metrics[
        (metrics["sm_clock"] >= sm_clock_min) & (metrics["sm_clock"] <= sm_clock_max)
    ].copy()
    if band.empty:
        raise RuntimeError(
            f"sm_clock {sm_clock_min}~{sm_clock_max} MHz 구간에 해당하는 행이 없습니다."
        )

    band = band[np.isfinite(band["throughput"])].copy()
    band = band[band["throughput"] > 0].copy()
    if band.empty:
        raise RuntimeError("양수 throughput 데이터가 없습니다.")

    batch_sizes = sorted(band["batch_size"].dropna().unique())
    if not batch_sizes:
        raise RuntimeError("plot할 batch_size 데이터가 없습니다.")

    model_names = sorted(band["model_name"].dropna().unique(), key=model_sort_key)

    grouped = (
        band.groupby(["model_name", "batch_size"], as_index=False)["throughput"]
        .mean()
    )

    model_str = grouped["model_name"].astype(str)
    baseline_mask = model_str.str.contains(baseline_model_query, regex=False, na=False)
    baseline_df = grouped.loc[baseline_mask].copy()
    if baseline_df.empty:
        raise RuntimeError(
            f"baseline 모델 '{baseline_model_query}'를 포함한 model_name 데이터를 찾을 수 없습니다."
        )

    baseline_per_batch = (
        baseline_df.groupby("batch_size", as_index=True)["throughput"].mean()
    )
    baseline_per_batch = baseline_per_batch.replace(0, np.nan)

    valid_batch_sizes = [bs for bs in batch_sizes if bs in baseline_per_batch.index]
    if not valid_batch_sizes:
        raise RuntimeError("baseline이 존재하는 batch_size가 없습니다.")

    x = np.arange(len(valid_batch_sizes), dtype=float)
    n_models = 0
    model_ratios: list[tuple[str, np.ndarray]] = []

    for model_name in model_names:
        sub = grouped[grouped["model_name"] == model_name]
        if sub.empty:
            continue
        series = sub.set_index("batch_size")["throughput"]
        ratio_pct = np.array(
            [
                (series.loc[bs] / baseline_per_batch.loc[bs]) * 100.0
                if (bs in series.index) and np.isfinite(baseline_per_batch.loc[bs])
                else np.nan
                for bs in valid_batch_sizes
            ],
            dtype=float,
        )
        if np.isfinite(ratio_pct).any():
            n_models += 1
            model_ratios.append((str(model_name), ratio_pct))

    if n_models == 0:
        plt.close()
        raise RuntimeError("baseline 대비 비율 막대 플롯할 throughput 데이터가 없습니다.")

    fig, ax = plt.subplots(figsize=(12, 6))
    bar_total_width = 0.85
    bar_width = bar_total_width / max(n_models, 1)

    all_finite_heights: list[float] = []
    for j, (model_name, ratio_pct) in enumerate(model_ratios):
        offsets = (j - (n_models - 1) / 2.0) * bar_width
        bars = ax.bar(
            x + offsets,
            ratio_pct,
            width=bar_width * 0.95,
            color=model_color(model_name),
            edgecolor="black",
            linewidth=0.5,
            label=model_name,
            alpha=0.95,
        )
        for bar, h in zip(bars, ratio_pct):
            if np.isfinite(h):
                all_finite_heights.append(float(h))
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height(),
                    f"{h:.2f}%",
                    ha="center",
                    va="bottom",
                    fontsize=5.5,
                    rotation=0,
                )

    ax.axhline(100.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xlabel("batch_size")
    ax.set_ylabel("Throughput ratio to baseline (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [str(int(bs)) if float(bs).is_integer() else str(bs) for bs in valid_batch_sizes],
        rotation=0,
    )
    clk = _sm_clock_title_range(sm_clock_min, sm_clock_max)
    ax.set_title(
        f"Throughput ratio to baseline ({baseline_model_query}) by model ({clk})"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="model_name", fontsize=8)
    if all_finite_heights:
        ymax = max(105.0, max(all_finite_heights) * 1.12)
        ax.set_ylim(0, ymax)
    fig.tight_layout()

    safe_baseline = sanitize_name(baseline_model_query)
    out_path = (
        output_dir
        / f"hist_throughput_ratio_to_{safe_baseline}_sm_clock_{_sm_clock_slug(sm_clock_min, sm_clock_max)}_x_batch.png"
    )
    fig.savefig(out_path)
    plt.close(fig)
    print(f"저장 완료: {out_path}")


def _hist_ratio_by_batch_sm_clock_band(
    metrics: pd.DataFrame,
    output_dir: Path,
    sm_clock_min: int,
    sm_clock_max: int,
    value_col: str,
    baseline_model_query: str,
    hist_file_tag: str,
    ylabel_pct: str,
    title_metric: str,
    empty_band_msg: str,
) -> None:
    """
    model_name×batch_size 평균(value_col) 대비 baseline(batch별 100%) 비율(%) 막대 플롯.
    hist_file_tag 예: hist_avg_power → hist_avg_power_ratio_to_{baseline}_sm_clock_..._x_batch.png
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    band = metrics[
        (metrics["sm_clock"] >= sm_clock_min) & (metrics["sm_clock"] <= sm_clock_max)
    ].copy()
    if band.empty:
        raise RuntimeError(
            f"sm_clock {sm_clock_min}~{sm_clock_max} MHz 구간에 해당하는 행이 없습니다."
        )

    band = band[np.isfinite(band[value_col])].copy()
    band = band[band[value_col] > 0].copy()
    if band.empty:
        raise RuntimeError(empty_band_msg)

    batch_sizes = sorted(band["batch_size"].dropna().unique())
    if not batch_sizes:
        raise RuntimeError("plot할 batch_size 데이터가 없습니다.")

    model_names = sorted(band["model_name"].dropna().unique(), key=model_sort_key)

    grouped = band.groupby(["model_name", "batch_size"], as_index=False)[value_col].mean()

    model_str = grouped["model_name"].astype(str)
    baseline_mask = model_str.str.contains(baseline_model_query, regex=False, na=False)
    baseline_df = grouped.loc[baseline_mask].copy()
    if baseline_df.empty:
        raise RuntimeError(
            f"baseline 모델 '{baseline_model_query}'를 포함한 model_name 데이터를 찾을 수 없습니다."
        )

    baseline_per_batch = baseline_df.groupby("batch_size", as_index=True)[value_col].mean()
    baseline_per_batch = baseline_per_batch.replace(0, np.nan)

    valid_batch_sizes = [bs for bs in batch_sizes if bs in baseline_per_batch.index]
    if not valid_batch_sizes:
        raise RuntimeError("baseline이 존재하는 batch_size가 없습니다.")

    x = np.arange(len(valid_batch_sizes), dtype=float)
    n_models = 0
    model_ratios: list[tuple[str, np.ndarray]] = []

    for model_name in model_names:
        sub = grouped[grouped["model_name"] == model_name]
        if sub.empty:
            continue
        series = sub.set_index("batch_size")[value_col]
        ratio_pct = np.array(
            [
                (series.loc[bs] / baseline_per_batch.loc[bs]) * 100.0
                if (bs in series.index) and np.isfinite(baseline_per_batch.loc[bs])
                else np.nan
                for bs in valid_batch_sizes
            ],
            dtype=float,
        )
        if np.isfinite(ratio_pct).any():
            n_models += 1
            model_ratios.append((str(model_name), ratio_pct))

    if n_models == 0:
        plt.close()
        raise RuntimeError(f"baseline 대비 비율 막대 플롯할 {value_col} 데이터가 없습니다.")

    fig, ax = plt.subplots(figsize=(12, 6))
    bar_total_width = 0.85
    bar_width = bar_total_width / max(n_models, 1)

    all_finite_heights: list[float] = []
    for j, (model_name, ratio_pct) in enumerate(model_ratios):
        offsets = (j - (n_models - 1) / 2.0) * bar_width
        bars = ax.bar(
            x + offsets,
            ratio_pct,
            width=bar_width * 0.95,
            color=model_color(model_name),
            edgecolor="black",
            linewidth=0.5,
            label=model_name,
            alpha=0.95,
        )
        for bar, h in zip(bars, ratio_pct):
            if np.isfinite(h):
                all_finite_heights.append(float(h))
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height(),
                    f"{h:.2f}%",
                    ha="center",
                    va="bottom",
                    fontsize=5.5,
                    rotation=0,
                )

    ax.axhline(100.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xlabel("batch_size")
    ax.set_ylabel(ylabel_pct)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [str(int(bs)) if float(bs).is_integer() else str(bs) for bs in valid_batch_sizes],
        rotation=0,
    )
    clk = _sm_clock_title_range(sm_clock_min, sm_clock_max)
    ax.set_title(
        f"{title_metric} ratio to baseline ({baseline_model_query}) by model ({clk})"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="model_name", fontsize=8)
    if all_finite_heights:
        ymax = max(105.0, max(all_finite_heights) * 1.12)
        ax.set_ylim(0, ymax)
    fig.tight_layout()

    safe_baseline = sanitize_name(baseline_model_query)
    out_path = (
        output_dir
        / f"{hist_file_tag}_ratio_to_{safe_baseline}_sm_clock_{_sm_clock_slug(sm_clock_min, sm_clock_max)}_x_batch.png"
    )
    fig.savefig(out_path)
    plt.close(fig)
    print(f"저장 완료: {out_path}")


def plot_avg_power_hist_ratio_to_baseline_model_sm_clock_band(
    metrics: pd.DataFrame,
    output_dir: Path,
    sm_clock_min: int = 1200,
    sm_clock_max: int = 1250,
    baseline_model_query: str = "Qwen/Qwen3-8B",
) -> None:
    _hist_ratio_by_batch_sm_clock_band(
        metrics,
        output_dir,
        sm_clock_min,
        sm_clock_max,
        value_col="avg_power",
        baseline_model_query=baseline_model_query,
        hist_file_tag="hist_avg_power",
        ylabel_pct="Average power ratio to baseline (%)",
        title_metric="Average power",
        empty_band_msg="양수 avg_power 데이터가 없습니다.",
    )


def plot_time_per_batch_generate_hist_ratio_to_baseline_model_sm_clock_band(
    metrics: pd.DataFrame,
    output_dir: Path,
    sm_clock_min: int = 1200,
    sm_clock_max: int = 1250,
    baseline_model_query: str = "Qwen/Qwen3-8B",
) -> None:
    _hist_ratio_by_batch_sm_clock_band(
        metrics,
        output_dir,
        sm_clock_min,
        sm_clock_max,
        value_col="avg_during_time",
        baseline_model_query=baseline_model_query,
        hist_file_tag="hist_time_per_batch_generate",
        ylabel_pct="Wall time per repeat ratio to baseline (%)",
        title_metric="Time to generate batch_size tokens (per repeat)",
        empty_band_msg="양수 avg_during_time 데이터가 없습니다.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "sm_clock이 지정 구간인 데이터만 사용해, x축 batch_size·범례 model_name인 "
            "에너지/토큰·Throughput·평균전력·batch 토큰 생성시간·"
            "각 지표 baseline 대비 히스토그램(막대) 비율 플롯을 생성합니다. "
            "데이터 전역 최대 sm_clock에 대해서도 동일 플롯을 추가 생성합니다."
        )
    )
    parser.add_argument(
        "log_dir",
        type=str,
        help="CSV 파일이 있는 루트 디렉터리 (하위 폴더까지 검색)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="출력 디렉터리 (기본값: <log_dir>/analysis_output_sm_clock_band)",
    )
    parser.add_argument(
        "--sm-clock-min",
        type=int,
        default=1200,
        help="포함할 최소 sm_clock (MHz)",
    )
    parser.add_argument(
        "--sm-clock-max",
        type=int,
        default=1250,
        help="포함할 최대 sm_clock (MHz)",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        raise NotADirectoryError(f"디렉터리가 아닙니다: {log_dir}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (log_dir / "analysis_output_sm_clock_band")
    )

    df = load_data(log_dir)
    metrics = compute_metrics(df)
    plot_energy_per_token_by_batch_sm_clock_band(
        metrics,
        output_dir,
        sm_clock_min=args.sm_clock_min,
        sm_clock_max=args.sm_clock_max,
    )
    plot_throughput_by_batch_sm_clock_band(
        metrics,
        output_dir,
        sm_clock_min=args.sm_clock_min,
        sm_clock_max=args.sm_clock_max,
    )
    plot_avg_power_by_batch_sm_clock_band(
        metrics,
        output_dir,
        sm_clock_min=args.sm_clock_min,
        sm_clock_max=args.sm_clock_max,
    )
    plot_time_per_batch_generate_by_batch_sm_clock_band(
        metrics,
        output_dir,
        sm_clock_min=args.sm_clock_min,
        sm_clock_max=args.sm_clock_max,
    )
    plot_energy_hist_ratio_to_baseline_model_sm_clock_band(
        metrics,
        output_dir,
        sm_clock_min=args.sm_clock_min,
        sm_clock_max=args.sm_clock_max,
    )
    plot_throughput_hist_ratio_to_baseline_model_sm_clock_band(
        metrics,
        output_dir,
        sm_clock_min=args.sm_clock_min,
        sm_clock_max=args.sm_clock_max,
    )
    plot_avg_power_hist_ratio_to_baseline_model_sm_clock_band(
        metrics,
        output_dir,
        sm_clock_min=args.sm_clock_min,
        sm_clock_max=args.sm_clock_max,
    )
    plot_time_per_batch_generate_hist_ratio_to_baseline_model_sm_clock_band(
        metrics,
        output_dir,
        sm_clock_min=args.sm_clock_min,
        sm_clock_max=args.sm_clock_max,
    )

    sm_peak = metrics["sm_clock"].max()
    if np.isfinite(sm_peak):
        metrics_at_peak = metrics[metrics["sm_clock"] == sm_peak].copy()
        if not metrics_at_peak.empty:
            sm_int = int(round(float(sm_peak)))
            max_plotters = [
                plot_energy_per_token_by_batch_sm_clock_band,
                plot_throughput_by_batch_sm_clock_band,
                plot_avg_power_by_batch_sm_clock_band,
                plot_time_per_batch_generate_by_batch_sm_clock_band,
                plot_energy_hist_ratio_to_baseline_model_sm_clock_band,
                plot_throughput_hist_ratio_to_baseline_model_sm_clock_band,
                plot_avg_power_hist_ratio_to_baseline_model_sm_clock_band,
                plot_time_per_batch_generate_hist_ratio_to_baseline_model_sm_clock_band,
            ]
            for plot_fn in max_plotters:
                try:
                    plot_fn(
                        metrics_at_peak,
                        output_dir,
                        sm_clock_min=sm_int,
                        sm_clock_max=sm_int,
                    )
                except RuntimeError as exc:
                    print(f"[sm_clock=max {sm_int} MHz] {plot_fn.__name__} 건너뜀: {exc}")


if __name__ == "__main__":
    main()
