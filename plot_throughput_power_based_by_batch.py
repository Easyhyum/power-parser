import argparse
import re
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_csv_files(root: Path) -> List[Path]:
    # 하위 폴더까지 포함해서 모든 CSV를 수집
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
        "during_time",
        "repeat_count",
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    key_cols = ["model_name", "batch_size", "input_len", "kv_cache_lens", "sm_clock"]

    # 기존 스크립트와 동일한 유효 그룹 필터
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

    total_tokens = agg["avg_repeat"] * agg["batch_size"]
    total_time = agg["avg_during_time"] * agg["avg_repeat"]

    # throughput: token/sec
    agg["throughput"] = np.where(total_time > 0, total_tokens / total_time, np.nan)
    agg = agg[np.isfinite(agg["throughput"])].copy()

    return agg


def plot_throughput_by_batch(metrics: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_sizes = sorted(metrics["batch_size"].dropna().unique())
    model_names = sorted(metrics["model_name"].dropna().unique())

    if len(batch_sizes) == 0:
        raise RuntimeError("plot할 batch_size 데이터가 없습니다.")

    for bs in batch_sizes:
        plt.figure(figsize=(10, 6))

        has_data = False
        for model_name in model_names:
            sub = metrics[
                (metrics["batch_size"] == bs) & (metrics["model_name"] == model_name)
            ]
            if sub.empty:
                continue

            # 여러 input_len/kv 조합이 있을 수 있으므로 sm_clock별 평균
            line = sub.groupby("sm_clock", as_index=False)["throughput"].mean().sort_values("sm_clock")
            if line.empty:
                continue

            has_data = True
            plt.plot(
                line["sm_clock"],
                line["throughput"],
                marker="o",
                linewidth=1.8,
                label=str(model_name),
            )

        if not has_data:
            plt.close()
            print(f"데이터 없음, 건너뜀: batch_size={bs}")
            continue

        plt.xlabel("SM clock frequency (MHz)")
        plt.ylabel("Throughput (tokens/sec)")
        plt.title(f"Throughput by model (batch_size={int(bs)})")
        plt.grid(True, alpha=0.3)
        plt.legend(title="model_name")
        plt.tight_layout()

        file_name = f"throughput_bs{int(bs)}.png"
        out_path = output_dir / file_name
        plt.savefig(out_path)
        plt.close()
        print(f"저장 완료: {out_path}")


def summarize_increase_as_batch_increases(metrics: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    model_name, sm_clock 단위로 batch_size 증가에 따른
    throughput 증가량/증가율을 계산한다.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = (
        metrics.groupby(["model_name", "sm_clock", "batch_size"], as_index=False)["throughput"]
        .mean()
        .sort_values(["model_name", "sm_clock", "batch_size"])
    )

    grouped["prev_batch_size"] = grouped.groupby(["model_name", "sm_clock"])["batch_size"].shift(1)
    grouped["prev_throughput"] = grouped.groupby(["model_name", "sm_clock"])["throughput"].shift(1)

    grouped["abs_increase"] = grouped["throughput"] - grouped["prev_throughput"]
    grouped["pct_increase"] = grouped["abs_increase"] / grouped["prev_throughput"] * 100.0

    summary = grouped.dropna(subset=["prev_batch_size", "prev_throughput"]).copy()
    summary = summary.rename(
        columns={
            "throughput": "curr_throughput",
            "batch_size": "curr_batch_size",
        }
    )

    summary_path = output_dir / "throughput_increase_by_batch_step.csv"
    summary.to_csv(summary_path, index=False)
    print(f"저장 완료: {summary_path}")

    # 모델별로 증가율(직전 batch 대비 %) 플롯 생성
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
                sub_clock["pct_increase"],
                marker="o",
                linewidth=1.6,
                label=f"sm_clock={int(sm_clock)}",
            )

        if not has_data:
            plt.close()
            continue

        plt.xlabel("Current batch_size")
        plt.ylabel("Increase vs previous batch (%)")
        plt.title(f"Batch increase effect on throughput by SM clock ({model_name})")
        plt.grid(True, alpha=0.3)
        plt.legend(title="sm_clock")
        plt.tight_layout()

        model_safe = sanitize_name(str(model_name))
        out_path = output_dir / f"increase_pct_by_batch_throughput_{model_safe}.png"
        plt.savefig(out_path)
        plt.close()
        print(f"저장 완료: {out_path}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "모든 CSV를 합쳐서 batch_size별 throughput 라인차트(모든 model_name 범례 포함)를 생성합니다."
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
        help="출력 디렉터리 (기본값: <log_dir>/analysis_output_throughput)",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        raise NotADirectoryError(f"디렉터리가 아닙니다: {log_dir}")

    output_dir = (
        Path(args.output_dir) if args.output_dir else (log_dir / "analysis_output_throughput")
    )

    df = load_data(log_dir)
    metrics = compute_metrics(df)
    plot_throughput_by_batch(metrics, output_dir)
    summarize_increase_as_batch_increases(metrics, output_dir)


if __name__ == "__main__":
    main()