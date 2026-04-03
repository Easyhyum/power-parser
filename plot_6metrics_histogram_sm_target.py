"""
plot_6metrics_histogram.py 와 동일하게 gpu_profile 로드 → iteration_stats → aggregate_metrics.
플롯: x축=target_sm_clock, 범례·막대 순서 BF16→FP8→NVFP4→INT8→INT4.
메트릭·batch_size 마다 PNG 2종: *_raw.png(원값), *_norm.png(값÷기준버킷, BF16 우선·없으면 FP8→…).
레이아웃: 열 왼쪽=prefill, 오른쪽=decoding / 행=input_len 오름차순.

사용법:
  python plot_6metrics_histogram_sm_target.py <log_dir> [--output-dir <out>]
  python plot_6metrics_histogram_sm_target.py <log_dir> --plot-only \\
      --metrics-from <path/to/metrics.parquet 또는 metrics가 있는 csv 경로의 상위 폴더>

기본 플롯 출력: <log_dir>/analysis_6metrics_sm_target/
  하위 폴더 line_sm_clock_by_model/: x=sm_clock, 범례=input_len, 모델(베이스)별 라인
  · *_raw.png  · *_norm_ref.png(÷기준버킷/클럭)  · *_norm_maxsm.png(각 input_len 시리즈에서 max sm_clock y=1)

(--plot-only 이면 <metrics 상위>/analysis_6metrics_sm_target/ 또는 --output-dir)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_6metrics_histogram import (
    DATA_KEY,
    LABEL_FMT,
    METRIC_COLS,
    aggregate_metrics,
    compute_iteration_stats,
    load_csvs,
    save_metric_csvs,
    sanitize,
)


def _model_base(model_name: str) -> str:
    """nvidia/Qwen3-8B-NVFP4_prefill-all → nvidia/Qwen3-8B-NVFP4"""
    s = str(model_name)
    for suf in ("_prefill-all", "_decoding-all"):
        if s.endswith(suf):
            return s[: -len(suf)]
    return s


def _infer_phase(model_name: str) -> str | None:
    s = str(model_name).lower()
    if "prefill" in s:
        return "prefill"
    if "decoding" in s:
        return "decoding"
    return None


# 범례: BF16을 맨 앞, 이어서 FP8, NVFP4, INT8, INT4
QUANT_LEGEND_ORDER = ["BF16", "FP8", "NVFP4", "INT8", "INT4"]

QUANT_BUCKET_COLORS = {
    "BF16": "#424242",
    "FP8": "#1565C0",
    "NVFP4": "#2E7D32",
    "INT8": "#6A1B9A",
    "INT4": "#C62828",
}


def _quantize_bucket(model_base: str) -> str:
    """
    FP8 / NVFP4 / INT8 / INT4 에 해당하지 않으면 BF16.
    매칭 순서: 명시 BF16 → NVFP4 → FP8 → INT8 → INT4.
    """
    sl = str(model_base).lower()
    if "bf16" in sl:
        return "BF16"
    if "nvfp4" in sl:
        return "NVFP4"
    if "fp8" in sl:
        return "FP8"
    if "int8" in sl or "gptq-int8" in sl:
        return "INT8"
    if "int4" in sl or "gptq-int4" in sl:
        return "INT4"
    return "BF16"


def _buckets_in_legend_order(present: set[str]) -> list[str]:
    return [b for b in QUANT_LEGEND_ORDER if b in present]


def _pick_reference_bucket(present: set[str]) -> str:
    """데이터에 BF16이 없으면 FP8 → NVFP4 → INT8 → INT4 순으로 분모 후보."""
    for b in QUANT_LEGEND_ORDER:
        if b in present:
            return b
    return "BF16"


COLOR_LINE_PREFILL = "#C62828"
COLOR_LINE_DECODING = "#1565C0"


def _model_names_for_base(df: pd.DataFrame, base: str) -> tuple[str | None, str | None]:
    """같은 model_base 에 대해 prefill / decoding model_name 각 하나."""
    pre, dec = None, None
    for n in df["model_name"].dropna().unique():
        s = str(n)
        if _model_base(s) != base:
            continue
        ph = _infer_phase(s)
        if ph == "prefill":
            pre = s if pre is None else pre
        elif ph == "decoding":
            dec = s if dec is None else dec
    return pre, dec


def _series_vs_sm_clock(
    metrics: pd.DataFrame,
    model_name: str,
    col: str,
    batch_size: int,
    input_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """측정 sm_clock 오름차순 (x, y)."""
    sub = metrics[
        (metrics["batch_size"] == batch_size)
        & (metrics["model_name"] == model_name)
        & (metrics["input_len"] == input_len)
    ].dropna(subset=["sm_clock", col])
    if sub.empty:
        return np.array([]), np.array([])
    g = sub.groupby("sm_clock", as_index=False)[col].mean().sort_values("sm_clock")
    x = g["sm_clock"].values.astype(float)
    y = g[col].values.astype(float)
    return x, y


def _metrics_with_buckets(metrics: pd.DataFrame) -> pd.DataFrame:
    m = metrics.copy()
    m["model_base"] = m["model_name"].map(_model_base)
    m["q_bucket"] = m["model_base"].map(_quantize_bucket)
    m["_phase"] = m["model_name"].map(_infer_phase)
    return m


def _ref_value_at_sm_clock(
    m_b: pd.DataFrame,
    col: str,
    batch_size: int,
    input_len: int,
    phase: str,
    sm_clock: float,
    ref_bucket: str,
) -> float:
    sub = m_b[
        (m_b["batch_size"] == batch_size)
        & (m_b["input_len"] == input_len)
        & (m_b["_phase"] == phase)
        & (m_b["q_bucket"] == ref_bucket)
    ]
    if sub.empty:
        return float("nan")
    smf = sub["sm_clock"].astype(float)
    mask = np.isclose(smf, float(sm_clock), rtol=0.0, atol=1e-3)
    sub2 = sub[mask]
    if sub2.empty:
        return float("nan")
    return float(sub2[col].mean())


def _normalize_y_by_max_sm_clock(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """각 시리즈에서 sm_clock 이 최대인 지점의 y 를 1로."""
    if x.size == 0:
        return y
    mask = np.isfinite(x) & np.isfinite(y)
    if not mask.any():
        return np.full_like(y, np.nan, dtype=float)
    xm = float(np.max(x[mask]))
    ref_mask = mask & np.isclose(x, xm, rtol=0.0, atol=1e-3)
    if not ref_mask.any():
        return np.full_like(y, np.nan, dtype=float)
    ref = float(np.nanmean(y[ref_mask]))
    if not math.isfinite(ref) or ref == 0:
        return np.full_like(y, np.nan, dtype=float)
    out = np.full_like(y, np.nan, dtype=float)
    out[mask] = y[mask] / ref
    return out


def plot_line_charts_sm_clock_by_model(
    metrics: pd.DataFrame,
    line_dir: Path,
    line_sm_clock_xmax: float = 2500.0,
) -> None:
    """
    모델 베이스(model_base)마다 figure: 왼쪽 prefill, 오른쪽 decoding.
    x=측정 sm_clock, 범례=input_len.
    raw / norm_ref(클럭·input_len·phase 마다 ref_bucket 메트릭으로 나눔) /
    norm_maxsm(각 input_len 라인에서 max sm_clock 의 y=1).
    """
    if "sm_clock" not in metrics.columns:
        print("  (line sm_clock) 스킵: sm_clock 컬럼 없음")
        return

    line_dir.mkdir(parents=True, exist_ok=True)
    m_b = _metrics_with_buckets(metrics)

    batch_sizes = sorted(metrics["batch_size"].dropna().unique())
    bases = sorted(m_b["model_base"].dropna().unique().tolist(), key=str)

    for col, title, ylabel in METRIC_COLS:
        for bs in batch_sizes:
            sub_bs = m_b[m_b["batch_size"] == int(bs)]
            if sub_bs.empty:
                continue
            present = set(sub_bs["q_bucket"].dropna().unique().tolist())
            ref_bucket = _pick_reference_bucket(present)
            ref_note = (
                f"{ref_bucket}=1"
                if ref_bucket == "BF16"
                else f"{ref_bucket}=1 (BF16 없음)"
            )
            input_lens = sorted(sub_bs["input_len"].dropna().unique().tolist())
            if not input_lens:
                continue

            for base in bases:
                pre_mn, dec_mn = _model_names_for_base(
                    metrics[metrics["batch_size"] == int(bs)], base
                )
                if pre_mn is None and dec_mn is None:
                    continue

                tag_base = sanitize(base)[:120]

                for mode, y_axis, supt_suff, fname_tag in (
                    ("raw", ylabel, "raw", "raw"),
                    (
                        "norm_ref",
                        f"{ylabel} (÷ {ref_bucket} @ sm_clock)",
                        f"normalized {ref_note} @ each sm_clock",
                        "norm_ref",
                    ),
                    (
                        "norm_maxsm",
                        f"{ylabel} (ratio, max sm_clock y=1 / line)",
                        "max sm_clock → y=1 per input_len",
                        "norm_maxsm",
                    ),
                ):
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)
                    ax_p, ax_d = axes[0, 0], axes[0, 1]
                    n_il = max(len(input_lens), 1)

                    for ax, mn, phase, color in (
                        (ax_p, pre_mn, "prefill", COLOR_LINE_PREFILL),
                        (ax_d, dec_mn, "decoding", COLOR_LINE_DECODING),
                    ):
                        if mn is None:
                            ax.text(
                                0.5,
                                0.5,
                                f"No {phase} model",
                                ha="center",
                                va="center",
                                transform=ax.transAxes,
                            )
                            ax.set_xlabel("SM clock (MHz, measured)", fontweight="bold")
                            ax.set_ylabel(y_axis, fontweight="bold")
                            ax.set_title(phase, fontweight="bold")
                            continue

                        for j, il in enumerate(input_lens):
                            x, y = _series_vs_sm_clock(metrics, mn, col, int(bs), int(il))
                            if x.size == 0:
                                continue
                            if mode == "norm_ref":
                                y_plot = np.empty_like(y, dtype=float)
                                for k, (sv, v) in enumerate(zip(x, y)):
                                    r = _ref_value_at_sm_clock(
                                        m_b,
                                        col,
                                        int(bs),
                                        int(il),
                                        phase,
                                        float(sv),
                                        ref_bucket,
                                    )
                                    if (
                                        math.isfinite(float(v))
                                        and math.isfinite(r)
                                        and r != 0
                                    ):
                                        y_plot[k] = float(v) / r
                                    else:
                                        y_plot[k] = np.nan
                            elif mode == "norm_maxsm":
                                y_plot = _normalize_y_by_max_sm_clock(x, y)
                            else:
                                y_plot = y

                            lw = 1.2 + 0.6 * (j / max(n_il - 1, 1))
                            ax.plot(
                                x,
                                y_plot,
                                marker="o",
                                markersize=4,
                                linewidth=lw,
                                color=color,
                                alpha=0.75 + 0.25 * (j / max(n_il - 1, 1)),
                                label=f"input_len={int(il)}",
                            )

                        ax.set_xlabel("SM clock (MHz, measured)", fontweight="bold")
                        ax.set_ylabel(y_axis, fontweight="bold")
                        ax.set_title(f"{phase}\n{mn}", fontsize=9, fontweight="bold")
                        ax.grid(True, alpha=0.3)
                        ax.tick_params(axis="both", labelsize=8)
                        if ax.get_legend_handles_labels()[0]:
                            ax.legend(fontsize=7, loc="best")
                        if math.isfinite(line_sm_clock_xmax) and line_sm_clock_xmax > 0:
                            ax.set_xlim(0.0, float(line_sm_clock_xmax))

                    fig.suptitle(
                        f"{title}  (batch_size={int(bs)}, {supt_suff})\nbase={base}",
                        fontsize=11,
                        fontweight="bold",
                        y=1.03,
                    )
                    fig.tight_layout()
                    fname = f"line_sm_{sanitize(col)}_bs{int(bs)}_{tag_base}_{fname_tag}.png"
                    out_path = line_dir / fname
                    fig.savefig(out_path, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    print(f"  저장 (line): {out_path}")


def _plot_panel_grouped_by_target_sm(
    ax,
    metrics: pd.DataFrame,
    col: str,
    ylabel_axis: str,
    phase: str,
    batch_size: int,
    input_len: int,
    row_title_left: str,
    row_title_right: str,
    ref_bucket: str,
    normalize: bool,
) -> None:
    sub = metrics[
        (metrics["batch_size"] == batch_size) & (metrics["input_len"] == input_len)
    ].copy()
    sub["_phase"] = sub["model_name"].map(_infer_phase)
    sub = sub[sub["_phase"] == phase]
    if sub.empty:
        ax.text(
            0.5,
            0.5,
            f"No {phase} data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xlabel("Target SM clock (MHz)", fontweight="bold")
        ax.set_ylabel(ylabel_axis, fontweight="bold")
        ax.set_title(row_title_left if phase == "prefill" else row_title_right, fontsize=10, fontweight="bold")
        return

    sub["model_base"] = sub["model_name"].map(_model_base)
    sub["q_bucket"] = sub["model_base"].map(_quantize_bucket)
    g = (
        sub.groupby(["target_sm_clock", "q_bucket"], as_index=False)[col]
        .mean()
        .sort_values("target_sm_clock")
    )
    target_sms = sorted(g["target_sm_clock"].dropna().unique().tolist())
    buckets = _buckets_in_legend_order(set(g["q_bucket"].dropna().unique()))
    if not target_sms or not buckets:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Target SM clock (MHz)", fontweight="bold")
        ax.set_ylabel(ylabel_axis, fontweight="bold")
        ax.set_title(row_title_left if phase == "prefill" else row_title_right, fontsize=10, fontweight="bold")
        return

    x = np.arange(len(target_sms))
    n = len(buckets)
    width = min(0.8 / max(n, 1), 0.15)

    for bi, bucket in enumerate(buckets):
        heights: list[float] = []
        for tsm in target_sms:
            row = g[(g["target_sm_clock"] == tsm) & (g["q_bucket"] == bucket)]
            raw = float(row[col].iloc[0]) if not row.empty else np.nan
            if not normalize:
                heights.append(raw)
                continue
            ref_rows = g[(g["target_sm_clock"] == tsm) & (g["q_bucket"] == ref_bucket)]
            ref = float(ref_rows[col].iloc[0]) if not ref_rows.empty else np.nan
            if np.isfinite(raw) and np.isfinite(ref) and ref != 0:
                heights.append(raw / ref)
            else:
                heights.append(np.nan)

        offset = (bi - (n - 1) / 2.0) * width
        color = QUANT_BUCKET_COLORS.get(bucket, "#757575")
        bars = ax.bar(x + offset, heights, width, label=bucket, color=color, edgecolor="white")

        fmt = ".3f" if normalize else LABEL_FMT.get(col, ".5g")
        for bar, v in zip(bars, heights):
            if np.isfinite(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{v:{fmt}}",
                    ha="center",
                    va="bottom",
                    fontsize=5,
                    rotation=90,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(t)) for t in target_sms], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Target SM clock (MHz)", fontweight="bold")
    ax.set_ylabel(ylabel_axis, fontweight="bold")
    ax.set_title(row_title_left if phase == "prefill" else row_title_right, fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="y", labelsize=8)


def plot_histograms_by_target_sm(metrics: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_sizes = sorted(metrics["batch_size"].dropna().unique())
    for col, title, ylabel in METRIC_COLS:
        for bs in batch_sizes:
            sub_bs = metrics[metrics["batch_size"] == bs]
            if sub_bs.empty:
                continue
            input_lens = sorted(sub_bs["input_len"].dropna().unique().tolist())
            if not input_lens:
                continue

            present_buckets: set[str] = set()
            work_bs = sub_bs.copy()
            work_bs["model_base"] = work_bs["model_name"].map(_model_base)
            work_bs["q_bucket"] = work_bs["model_base"].map(_quantize_bucket)
            present_buckets.update(work_bs["q_bucket"].dropna().unique().tolist())
            ref_bucket = _pick_reference_bucket(present_buckets)
            ref_note = (
                f", {ref_bucket}=1"
                if ref_bucket == "BF16"
                else f", {ref_bucket}=1 (BF16 없음)"
            )
            ylabel_norm = f"{ylabel} (÷ {ref_bucket})"

            for normalize, tag, y_axis, supt_extra in (
                (False, "raw", ylabel, "raw"),
                (True, "norm", ylabel_norm, f"normalized{ref_note}"),
            ):
                nrows = len(input_lens)
                ncols = 2
                fig_w = max(11.0, 5.2 * ncols + 1)
                fig_h = max(6.0, 3.4 * nrows + 1.2)
                fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

                for r, il in enumerate(input_lens):
                    tl = f"prefill · input_len={int(il)}"
                    tr = f"decoding · input_len={int(il)}"
                    _plot_panel_grouped_by_target_sm(
                        axes[r, 0],
                        metrics,
                        col,
                        y_axis,
                        "prefill",
                        int(bs),
                        int(il),
                        tl,
                        tr,
                        ref_bucket,
                        normalize,
                    )
                    _plot_panel_grouped_by_target_sm(
                        axes[r, 1],
                        metrics,
                        col,
                        y_axis,
                        "decoding",
                        int(bs),
                        int(il),
                        tl,
                        tr,
                        ref_bucket,
                        normalize,
                    )

                y_max = max(
                    axes[i, j].get_ylim()[1]
                    for i in range(nrows)
                    for j in range(ncols)
                    if axes[i, j].get_visible()
                )
                for i in range(nrows):
                    for j in range(ncols):
                        if axes[i, j].get_visible():
                            axes[i, j].set_ylim(0, y_max)

                handles, labels = [], []
                seen = set()
                for i in range(nrows):
                    for j in range(ncols):
                        h, lab = axes[i, j].get_legend_handles_labels()
                        for hi, li in zip(h, lab):
                            if li not in seen:
                                seen.add(li)
                                handles.append(hi)
                                labels.append(li)

                order = _buckets_in_legend_order(set(labels))
                if order:
                    perm = sorted(
                        range(len(labels)),
                        key=lambda k: order.index(labels[k]) if labels[k] in order else 99,
                    )
                    handles = [handles[k] for k in perm]
                    labels = [labels[k] for k in perm]

                if handles:
                    fig.legend(
                        handles,
                        labels,
                        loc="lower center",
                        ncol=min(len(labels), 5),
                        fontsize=8,
                        frameon=True,
                        bbox_to_anchor=(0.5, -0.02),
                    )

                fig.suptitle(
                    f"{title}  (batch_size={int(bs)}, {supt_extra})",
                    fontsize=13,
                    fontweight="bold",
                    y=1.02,
                )
                fig.tight_layout(rect=(0, 0.06, 1, 0.98))
                fname = f"hist_target_sm_{sanitize(col)}_bs{int(bs)}_{tag}.png"
                out_path = output_dir / fname
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"  저장: {out_path}")


def _load_metrics_for_plot_only(path: Path) -> pd.DataFrame:
    """aggregate_metrics 결과와 동일한 컬럼을 가진 CSV/단일 파일에서 metrics 로드."""
    if path.is_file():
        return pd.read_csv(path)
    merged = None
    for col, _, _ in METRIC_COLS:
        p = path / f"{col}.csv"
        if not p.is_file():
            continue
        part = pd.read_csv(p)
        if merged is None:
            merged = part
        else:
            merged = merged.merge(part, on=DATA_KEY, how="outer")
    if merged is None:
        raise FileNotFoundError(
            f"메트릭 CSV를 찾을 수 없습니다: {path} (또는 단일 통합 csv 경로를 지정하세요)"
        )
    if "sm_clock" not in merged.columns and "target_sm_clock" in merged.columns:
        merged = merged.copy()
        merged["sm_clock"] = pd.to_numeric(merged["target_sm_clock"], errors="coerce")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "히스토그램(target_sm_clock)·라인(sm_clock×input_len, 모델 베이스별 "
            "raw / norm_ref / norm_maxsm)."
        )
    )
    parser.add_argument(
        "log_dir",
        type=str,
        nargs="?",
        default=None,
        help="gpu_profile_*.csv 디렉터리 (--plot-only 일 때는 생략)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="기본: <log_dir>/analysis_6metrics_sm_target",
    )
    parser.add_argument(
        "--idx",
        type=int,
        default=1,
        help="total_energy 시작 index (plot_6metrics_histogram.py 와 동일)",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="gpu_profile 로드/집계 생략, --metrics-from 만 사용",
    )
    parser.add_argument(
        "--metrics-from",
        type=str,
        default=None,
        help="plot-only: metrics 통합 csv 경로 또는 per-metric csv가 있는 디렉터리",
    )
    parser.add_argument(
        "--skip-metric-csvs",
        action="store_true",
        help="메트릭 개별 CSV 저장 생략",
    )
    parser.add_argument(
        "--line-sm-clock-xmax",
        type=float,
        default=2500.0,
        help="라인 차트(sm_clock x축) 상한 MHz (기본 2500, 0 이하면 자동)",
    )
    parser.add_argument(
        "--skip-line-sm-clock",
        action="store_true",
        help="line_sm_clock_by_model 라인 차트 생략",
    )
    args = parser.parse_args()

    if args.plot_only:
        if not args.metrics_from:
            parser.error("--plot-only 는 --metrics-from 가 필요합니다.")
        mpath = Path(args.metrics_from)
        metrics = _load_metrics_for_plot_only(mpath)
        root = mpath.parent if mpath.is_file() else mpath
        base_out = (
            Path(args.output_dir)
            if args.output_dir
            else root.parent / "analysis_6metrics_sm_target"
        )
    else:
        if not args.log_dir:
            parser.error("log_dir 인자가 필요합니다 (또는 --plot-only --metrics-from 사용).")
        log_dir = Path(args.log_dir)
        if not log_dir.is_dir():
            raise NotADirectoryError(f"디렉터리가 아닙니다: {log_dir}")
        analysis_dir = log_dir / "analysis_6metrics"
        base_out = Path(args.output_dir) if args.output_dir else (log_dir / "analysis_6metrics_sm_target")

        print("1. CSV 로드...")
        df = load_csvs(log_dir)
        print(f"   전체 행: {len(df):,}")

        print(f"2. iteration 단위 집계... (start_idx={args.idx})")
        it = compute_iteration_stats(df, start_idx=args.idx)
        print(f"   iteration 수: {len(it):,}")

        print("3. data 단위 메트릭 집계...")
        metrics = aggregate_metrics(it)
        print(f"   data 수: {len(metrics):,}")

        analysis_dir.mkdir(parents=True, exist_ok=True)
        it.to_csv(analysis_dir / "iteration_stats.csv", index=False)
        print(f"   저장: {analysis_dir / 'iteration_stats.csv'}")

        if not args.skip_metric_csvs:
            print("\n4. 메트릭 CSV 저장...")
            save_metric_csvs(metrics, analysis_dir)

    plot_dir = base_out
    print(f"\n5. target_sm_clock 히스토그램 → {plot_dir}")
    plot_histograms_by_target_sm(metrics, plot_dir)
    if not args.skip_line_sm_clock:
        line_dir = plot_dir / "line_sm_clock_by_model"
        xmax = args.line_sm_clock_xmax
        if xmax <= 0:
            xmax = float("nan")
        print(f"\n6. sm_clock 라인 차트 (모델 베이스별) → {line_dir}")
        print(
            "   · *_raw.png 원값 | *_norm_ref.png 는 각 (sm_clock,input_len,phase)에서 "
            "기준 정량 버킷(BF16 우선…) 메트릭으로 나눈 비율 | "
            "*_norm_maxsm.png 는 input_len 마다 최대 sm_clock 의 y 를 1 로"
        )
        plot_line_charts_sm_clock_by_model(
            metrics,
            line_dir,
            line_sm_clock_xmax=xmax,
        )
    print("\n완료!")


if __name__ == "__main__":
    main()
