"""
plot_6metrics_histogram.py 와 동일 파이프라인(로드 → iteration_stats → aggregate_metrics).
plot_6metrics_histogram_sm_target.py 와 유사 레이아웃이나, 정량 버킷 없이 model_name 전체로 구분.

- x축: target_sm_clock
- y축: 비율 — clock 마다 동일 베이스의 model_name 에 "ascending" 이 들어간 행을 1로 두고 나눔
  (max/min target_sm_clock 기준 정규화 없음).
- 열: 왼쪽 prefill / 오른쪽 decoding
- 행: input_len 오름차순
- 메트릭마다 PNG 1개

기본 출력: <log_dir>/analysis_6metrics_sm_target_by_model/
  (--plot-only 시 metrics 상위 폴더 기준 동일 이름 서브폴더 또는 --output-dir)

사용법:
  python plot_6metrics_histogram_sm_target_by_model.py <log_dir>
  python plot_6metrics_histogram_sm_target_by_model.py --plot-only \\
      --metrics-from <analysis_6metrics 디렉터리>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_6metrics_histogram import (
    DATA_KEY,
    METRIC_COLS,
    aggregate_metrics,
    compute_iteration_stats,
    load_csvs,
    save_metric_csvs,
    sanitize,
)


def _infer_phase(model_name: str) -> str | None:
    s = str(model_name).lower()
    if "prefill" in s:
        return "prefill"
    if "decoding" in s:
        return "decoding"
    return None


# model_name 예: ..._prefill-all_ascending / ..._descending / ..._even
_SCHEDULE_SUFFIXES = ("_ascending", "_descending", "_even")


def _schedule_base_key(model_name: str) -> str:
    """접미사 _ascending / _descending / _even 제거한 베이스 키 (스케줄 묶음)."""
    s = str(model_name)
    low = s.lower()
    for suf in _SCHEDULE_SUFFIXES:
        if low.endswith(suf):
            return s[: len(s) - len(suf)]
    return s


def _ascending_peer_in_names(names: list[str], base: str) -> str | None:
    """같은 베이스에 대해 이름에 ascending 이 포함된 model_name 하나."""
    for n in names:
        nlow = str(n).lower()
        if "ascending" in nlow and _schedule_base_key(n) == base:
            return str(n)
    return None


def _ratio_vs_ascending_schedule(
    target_sms: list,
    g: pd.DataFrame,
    col: str,
    model_name: str,
    names: list[str],
) -> list[float]:
    """
    각 target_sm_clock 마다 동일 베이스의 ascending 포함 모델 값을 분모(=1).
    해당 베이스에 기준 ascending 행이 없으면 전부 NaN.
    """
    base = _schedule_base_key(model_name)
    asc_mn = _ascending_peer_in_names(names, base)
    if asc_mn is None:
        return [np.nan] * len(target_sms)

    out: list[float] = []
    for tsm in target_sms:
        ref_row = g[(g["target_sm_clock"] == tsm) & (g["model_name"] == asc_mn)]
        raw_row = g[(g["target_sm_clock"] == tsm) & (g["model_name"] == model_name)]
        ref = float(ref_row[col].iloc[0]) if not ref_row.empty else np.nan
        raw = float(raw_row[col].iloc[0]) if not raw_row.empty else np.nan
        if np.isfinite(ref) and ref != 0 and np.isfinite(raw):
            out.append(raw / ref)
        else:
            out.append(np.nan)
    return out


def _plot_panel_by_model_name(
    ax,
    metrics: pd.DataFrame,
    col: str,
    ylabel: str,
    phase: str,
    batch_size: int,
    input_len: int,
    row_title_left: str,
    row_title_right: str,
    cmap,
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
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(
            row_title_left if phase == "prefill" else row_title_right,
            fontsize=10,
            fontweight="bold",
        )
        return

    g = (
        sub.groupby(["target_sm_clock", "model_name"], as_index=False)[col]
        .mean()
        .sort_values("target_sm_clock")
    )
    target_sms = sorted(g["target_sm_clock"].dropna().unique().tolist())
    names = sorted(g["model_name"].dropna().unique().tolist(), key=str)
    if not target_sms or not names:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Target SM clock (MHz)", fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(
            row_title_left if phase == "prefill" else row_title_right,
            fontsize=10,
            fontweight="bold",
        )
        return

    x = np.arange(len(target_sms))
    n = len(names)
    width = min(0.8 / max(n, 1), 0.12)
    n_colors = getattr(cmap, "N", 10)

    ratio_fmt = ".3f"
    for bi, mn in enumerate(names):
        heights = _ratio_vs_ascending_schedule(target_sms, g, col, mn, names)
        offset = (bi - (n - 1) / 2.0) * width
        color = cmap(bi % n_colors)
        bars = ax.bar(x + offset, heights, width, label=str(mn), color=color, edgecolor="white")

        for bar, v in zip(bars, heights):
            if np.isfinite(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{v:{ratio_fmt}}",
                    ha="center",
                    va="bottom",
                    fontsize=4,
                    rotation=90,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(t)) for t in target_sms], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Target SM clock (MHz)", fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_title(
        row_title_left if phase == "prefill" else row_title_right,
        fontsize=10,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="y", labelsize=8)


def plot_histograms_by_target_sm_by_model(metrics: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("tab20")

    batch_sizes = sorted(metrics["batch_size"].dropna().unique())
    for col, title, ylabel in METRIC_COLS:
        ylabel_ratio = f"{ylabel} (÷ ascending @ same clock)"
        for bs in batch_sizes:
            sub_bs = metrics[metrics["batch_size"] == bs]
            if sub_bs.empty:
                continue
            input_lens = sorted(sub_bs["input_len"].dropna().unique().tolist())
            if not input_lens:
                continue

            nrows = len(input_lens)
            ncols = 2
            fig_w = max(11.0, 5.2 * ncols + 1)
            fig_h = max(6.0, 3.4 * nrows + 1.2)
            fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

            for r, il in enumerate(input_lens):
                tl = f"prefill · input_len={int(il)}"
                tr = f"decoding · input_len={int(il)}"
                _plot_panel_by_model_name(
                    axes[r, 0],
                    metrics,
                    col,
                    ylabel_ratio,
                    "prefill",
                    int(bs),
                    int(il),
                    tl,
                    tr,
                    cmap,
                )
                _plot_panel_by_model_name(
                    axes[r, 1],
                    metrics,
                    col,
                    ylabel_ratio,
                    "decoding",
                    int(bs),
                    int(il),
                    tl,
                    tr,
                    cmap,
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
            seen: set[str] = set()
            for i in range(nrows):
                for j in range(ncols):
                    h, lab = axes[i, j].get_legend_handles_labels()
                    for hi, li in zip(h, lab):
                        if li not in seen:
                            seen.add(li)
                            handles.append(hi)
                            labels.append(li)
            labels_sorted = sorted(labels, key=str)
            perm = [labels.index(lb) for lb in labels_sorted]
            handles = [handles[k] for k in perm]
            labels = labels_sorted

            if handles:
                fig.legend(
                    handles,
                    labels,
                    loc="lower center",
                    ncol=1,
                    fontsize=6,
                    frameon=True,
                    bbox_to_anchor=(0.5, -0.12),
                )

            fig.suptitle(
                f"{title}  (batch_size={int(bs)}, by model_name; "
                f"per target SM: ascending-in-name = 1)",
                fontsize=13,
                fontweight="bold",
                y=1.02,
            )
            fig.tight_layout(rect=(0, 0.14, 1, 0.98))
            fname = f"hist_target_sm_{sanitize(col)}_bs{int(bs)}.png"
            out_path = output_dir / fname
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  저장: {out_path}")


def _load_metrics_for_plot_only(path: Path) -> pd.DataFrame:
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
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="target_sm_clock x축·model_name 범례·비율(클럭마다 동일베이스 ascending포함=1)·행=input_len 열=prefill|decoding.",
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
        help="기본: <log_dir>/analysis_6metrics_sm_target_by_model",
    )
    parser.add_argument(
        "--idx",
        type=int,
        default=1,
        help="total_energy 시작 index",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="gpu_profile 로드/집계 생략",
    )
    parser.add_argument(
        "--metrics-from",
        type=str,
        default=None,
        help="plot-only: 통합 csv 또는 per-metric csv 디렉터리",
    )
    parser.add_argument(
        "--skip-metric-csvs",
        action="store_true",
        help="메트릭 개별 CSV 저장 생략",
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
            else root.parent / "analysis_6metrics_sm_target_by_model"
        )
    else:
        if not args.log_dir:
            parser.error("log_dir 인자가 필요합니다 (또는 --plot-only --metrics-from 사용).")
        log_dir = Path(args.log_dir)
        if not log_dir.is_dir():
            raise NotADirectoryError(f"디렉터리가 아닙니다: {log_dir}")
        analysis_dir = log_dir / "analysis_6metrics"
        base_out = (
            Path(args.output_dir)
            if args.output_dir
            else (log_dir / "analysis_6metrics_sm_target_by_model")
        )

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
    print(f"\n5. target_sm_clock (by model_name) → {plot_dir}")
    plot_histograms_by_target_sm_by_model(metrics, plot_dir)
    print("\n완료!")


if __name__ == "__main__":
    main()
