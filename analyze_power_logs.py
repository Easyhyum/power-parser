import argparse
import os
import re
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pandas as pd


def find_csv_files(root: Path) -> List[Path]:
    return sorted(p for p in root.glob("*.csv") if p.is_file())


def sanitize_name(name: str) -> str:
    name = re.sub(r"[^\w\-]+", "_", name)
    return name.strip("_") or "unknown"


def load_data(log_dir: Path) -> pd.DataFrame:
    csv_files = find_csv_files(log_dir)
    if not csv_files:
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {log_dir}")

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            df["__source_file"] = f.name
            dfs.append(df)
        except Exception as e:
            print(f"CSV 로드 실패, 건너뜀: {f} ({e})")
    if not dfs:
        raise RuntimeError("CSV 파일을 모두 읽지 못했습니다.")

    df_all = pd.concat(dfs, ignore_index=True)
    return df_all


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    # 필요한 컬럼 형 변환
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
        "total_energy",
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
        "total_energy",
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # 기본 키
    key_cols = ["model_name", "batch_size", "input_len", "kv_cache_lens", "sm_clock"]

    # sm_clock을 포함한 그룹에서 kv_cache_lens의 unique 개수가 5개 이하면 해당 데이터는 버림
    kv_counts = (
        df.groupby(["model_name", "batch_size", "input_len", "sm_clock"])[
            "kv_cache_lens"
        ]
        .nunique()
        .reset_index(name="kv_unique_cnt")
    )
    valid_keys = kv_counts[kv_counts["kv_unique_cnt"] > 20][
        ["model_name", "batch_size", "input_len", "sm_clock"]
    ]
    df = df.merge(valid_keys, on=["model_name", "batch_size", "input_len", "sm_clock"])

    # 1) base total_energy: 그룹 내 첫 행과 마지막 행의 total_energy 차이
    df_sorted = df.sort_values(key_cols + ["index"])

    def _energy_diff(g: pd.DataFrame) -> float:
        first = g.iloc[0]["total_energy"]
        last = g.iloc[-1]["total_energy"]
        # CSV의 total_energy 단위는 mJ 이므로 J로 변환
        return (last - first) * 1e-3

    base_energy = (
        df_sorted.groupby(key_cols, as_index=False)
        .apply(_energy_diff)
        .rename(columns={None: "total_energy_base"})
    )

    # 2) 필터링
    df = df.copy()
    df["idx_ratio"] = df["index"] / df["length"]
    df["prompt_gap"] = df["kv_cache_lens"] - df["input_len"]
    # repeat_count 기준으로 during_time을 나눈 per-repeat 시간
    df["during_time_per_repeat"] = df["during_time"] / df["repeat_count"]

    # 요구 사항(최종):
    # - index / length >= 0.5 인 항목만 사용
    # - input_len - kv_cache_lens 값이 5 이상인 항목만 사용
    filtered = df[(df["idx_ratio"] >= 0.5) & (df["prompt_gap"] >= 5)]

    # 필터링된 행에서 평균 power, during_time_per_repeat, repeat_count
    agg = (
        filtered.groupby(key_cols, as_index=False)
        .agg(
            avg_power=("power", "mean"),
            avg_during_time=("during_time_per_repeat", "mean"),
            avg_repeat=("repeat_count", "mean"),
        )
    )

    # power 기반 total energy (비교군) - J 단위
    # per-repeat 평균 시간과 평균 repeat 수를 곱해, 전체 구간 에너지로 맞춰줌
    agg["total_energy_power_based"] = (
        agg["avg_power"] * agg["avg_during_time"] * agg["avg_repeat"]
    )

    # base total_energy 와 조인
    metrics = pd.merge(agg, base_energy, on=key_cols, how="inner")

    # energy per token: total_energy(J) / (avg(repeat_count) * batch_size)
    denom = metrics["avg_repeat"] * metrics["batch_size"]
    # J 단위 그대로 사용
    metrics["energy_per_token_base"] = metrics["total_energy_base"] / denom
    metrics["energy_per_token_power_based"] = (
        metrics["total_energy_power_based"] / denom
    )

    # throughput: during_time(초)에서 batch_size로 나눈 값의 역수 개념
    # 사용자 설명에 따라 during_time으로부터 batch_size를 나누어 사용
    # 여기서는 per-repeat 평균 시간(avg_during_time)을 사용하므로,
    # throughput ≈ batch_size / avg_during_time (samples / second)
    metrics["throughput"] = metrics["batch_size"] / metrics["avg_during_time"]

    return metrics


def plot_for_model(metrics: pd.DataFrame, output_dir: Path, model_name: str) -> None:
    m = metrics[metrics["model_name"] == model_name].copy()
    if m.empty:
        return

    model_safe = sanitize_name(model_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_sizes = sorted(m["batch_size"].unique())

    def _plot(y_col: str, ylabel: str, title_suffix: str, filename_suffix: str):
        plt.figure(figsize=(8, 5))
        for bs in batch_sizes:
            sub = m[m["batch_size"] == bs]
            # 같은 sm_clock 내에서는 평균 값만 사용 (점 수 줄이기)
            if sub.empty:
                continue
            sub = (
                sub.groupby("sm_clock", as_index=False)[y_col]
                .mean()
                .sort_values("sm_clock")
            )
            if sub.empty:
                continue
            plt.plot(
                sub["sm_clock"],
                sub[y_col],
                marker="o",
                label=f"batch_size={int(bs)}",
            )
        plt.xlabel("SM clock frequency (MHz)")
        plt.ylabel(ylabel)
        plt.title(f"{model_name} - {title_suffix}")
        plt.legend()
        # 에너지 관련 값들은 소수점 5자리까지 표시
        if "energy" in y_col:
            plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%.5f"))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fname = output_dir / f"{model_safe}_{filename_suffix}.png"
        plt.savefig(fname)
        plt.close()
        print(f"저장 완료: {fname}")

    # 1) total_energy base 기준
    _plot(
        "total_energy_base",
        "Total energy (J, base)",
        "Total energy (base)",
        "total_energy_base",
    )

    # 2) power base 기준 (평균 소비전력)
    _plot(
        "avg_power",
        "Average power (W, base)",
        "Average power (base)",
        "avg_power_base",
    )

    # 3) during_time
    _plot(
        "avg_during_time",
        "Average during_time (s)",
        "Average during_time",
        "avg_during_time",
    )

    # 4) energy per token - total_energy base 기준
    _plot(
        "energy_per_token_base",
        "Energy per token (J/token, base total_energy)",
        "Energy per token (base total_energy)",
        "energy_per_token_base",
    )

    # 5) energy per token - power 기반 total energy 기준
    _plot(
        "energy_per_token_power_based",
        "Energy per token (J/token, power-based)",
        "Energy per token (power-based)",
        "energy_per_token_power_based",
    )

    # 6) throughput (samples / second, per repeat)
    _plot(
        "throughput",
        "Throughput (samples/s, per repeat)",
        "Throughput (per repeat)",
        "throughput",
    )


def plot_for_model_per_batch(
    metrics: pd.DataFrame, output_dir: Path, model_name: str
) -> None:
    """
    model_name 별로, batch_size 단위의 개별 PNG를 생성하고
    각 점에 y값(또는 간단한 텍스트)을 라벨로 붙인다.
    """
    m = metrics[metrics["model_name"] == model_name].copy()
    if m.empty:
        return

    model_safe = sanitize_name(model_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_sizes = sorted(m["batch_size"].unique())

    def _plot_batch(y_col: str, ylabel: str, title_suffix: str, filename_suffix: str):
        for bs in batch_sizes:
            sub = m[m["batch_size"] == bs]
            if sub.empty:
                continue

            # 같은 sm_clock 내에서는 평균 값만 사용
            sub = (
                sub.groupby("sm_clock", as_index=False)[y_col]
                .mean()
                .sort_values("sm_clock")
            )
            if sub.empty:
                continue

            plt.figure(figsize=(8, 5))
            x = sub["sm_clock"]
            y = sub[y_col]

            plt.scatter(x, y, marker="o")
            # 각 점에 값 라벨 붙이기 (소수점 5자리)
            for xi, yi in zip(x, y):
                plt.annotate(
                    f"{yi:.5f}",
                    (xi, yi),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha="center",
                    fontsize=8,
                )

            plt.xlabel("SM clock frequency (MHz)")
            plt.ylabel(ylabel)
            plt.title(f"{model_name} (batch_size={int(bs)}) - {title_suffix}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            fname = (
                output_dir
                / f"{model_safe}_bs{int(bs)}_{filename_suffix}.png"
            )
            plt.savefig(fname)
            plt.close()
            print(f"저장 완료: {fname}")

    # total_energy_base
    _plot_batch(
        "total_energy_base",
        "Total energy (J, base)",
        "Total energy (base)",
        "total_energy_base",
    )

    # avg_power
    _plot_batch(
        "avg_power",
        "Average power (W, base)",
        "Average power (base)",
        "avg_power_base",
    )

    # avg_during_time
    _plot_batch(
        "avg_during_time",
        "Average during_time (s, per repeat)",
        "Average during_time (per repeat)",
        "avg_during_time",
    )

    # energy_per_token_base
    _plot_batch(
        "energy_per_token_base",
        "Energy per token (J/token, base total_energy)",
        "Energy per token (base total_energy)",
        "energy_per_token_base",
    )

    # energy_per_token_power_based
    _plot_batch(
        "energy_per_token_power_based",
        "Energy per token (J/token, power-based)",
        "Energy per token (power-based)",
        "energy_per_token_power_based",
    )

    # throughput
    _plot_batch(
        "throughput",
        "Throughput (samples/s, per repeat)",
        "Throughput (per repeat)",
        "throughput",
    )


def summarize_optimal_clocks(metrics: pd.DataFrame, out_path: Path) -> None:
    records = []
    grouped = metrics.groupby(["model_name", "batch_size"], as_index=False)

    for (model_name, batch_size), g in grouped:
        g_sorted = g.sort_values("energy_per_token_base")
        best_base = g_sorted.iloc[0]

        g_sorted2 = g.sort_values("energy_per_token_power_based")
        best_power = g_sorted2.iloc[0]

        records.append(
            {
                "model_name": model_name,
                "batch_size": batch_size,
                "best_sm_clock_base": best_base["sm_clock"],
                "min_energy_per_token_base": best_base["energy_per_token_base"],
                "best_sm_clock_power_based": best_power["sm_clock"],
                "min_energy_per_token_power_based": best_power[
                    "energy_per_token_power_based"
                ],
            }
        )

    summary = pd.DataFrame(records)
    summary.to_csv(out_path, index=False)
    print(f"최적 sm_clock 요약 저장: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="GPU profile CSV를 분석하여 batch_size별 최적 SM clock과 energy per token을 계산하고 플롯합니다."
    )
    parser.add_argument(
        "log_dir",
        type=str,
        help="CSV 로그가 들어있는 디렉터리 (예: log_pro6000_quant)",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        raise NotADirectoryError(f"디렉터리가 아닙니다: {log_dir}")

    df = load_data(log_dir)

    metrics = compute_metrics(df)

    # 출력 디렉터리: log_dir / analysis_output
    output_dir = log_dir / "analysis_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # model_name 별 플롯
    for model_name in sorted(metrics["model_name"].unique()):
        plot_for_model(metrics, output_dir, model_name)
        # batch_size별 개별 PNG (각 점에 값 라벨 포함)
        plot_for_model_per_batch(metrics, output_dir, model_name)

    # 최적 sm_clock 요약
    summary_path = output_dir / "optimal_sm_clock_summary.csv"
    summarize_optimal_clocks(metrics, summary_path)


if __name__ == "__main__":
    main()

