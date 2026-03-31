"""
sm_clock × (prefill | decoding)별 선형 회귀:
  Power = a * gpu_util + b * memory_util + c

- b는 비음수 제약: 무제약 최소제곱 후 b<0이면 b=0으로 두고 a,c만 재피팅.
- prefill / decoding은 model_name으로 분리하여 각각 별도 회귀.

iteration_stats.csv를 로드하여 계수를 구하고,
실제 vs 예측 비교 플롯 및 계수 분포 플롯을 생성한다.

사용법:
  python fit_power_model.py <iteration_stats.csv> [--output-dir <out>]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sanitize(name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", name).strip("_") or "unknown"


def infer_phase(model_name: str) -> str:
    s = str(model_name).lower()
    if "prefill" in s:
        return "prefill"
    if "decoding" in s:
        return "decoding"
    return "other"


def _fit_one_group(gpu: np.ndarray, mem: np.ndarray, pwr: np.ndarray) -> tuple[float, float, float]:
    """
    Power = a*gpu + b*mem + c, b >= 0.
    볼록 최소제곱 + 반공간 b>=0 → 무제약 해가 b>=0이면 채택, 아니면 b=0 경계에서 a,c만 피팅.
    """
    gpu = gpu.astype(float)
    mem = mem.astype(float)
    y = pwr.astype(float)
    A = np.column_stack([gpu, mem, np.ones(len(y))])
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b, c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
    if b >= 0:
        return a, b, c
    A2 = np.column_stack([gpu, np.ones(len(y))])
    ac, *_ = np.linalg.lstsq(A2, y, rcond=None)
    return float(ac[0]), 0.0, float(ac[1])


def fit_per_sm_clock_phase(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    (sm_clock, phase)별로 Power = a*gpu_util + b*memory_util + c, b>=0.
    """
    df = df.copy()
    df["_phase"] = df["model_name"].map(infer_phase)

    coeff_records: list[dict] = []
    detail_records: list[dict] = []

    for (sm, phase), grp in df.groupby(["sm_clock", "_phase"], sort=True):
        gpu = grp["avg_gpu_util"].values
        mem = grp["avg_memory_util"].values
        pwr = grp["avg_power_saturated"].values

        a, b, c = _fit_one_group(gpu, mem, pwr)
        pred = a * gpu + b * mem + c
        ss_res = float(np.sum((pwr - pred) ** 2))
        ss_tot = float(np.sum((pwr - pwr.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        coeff_records.append({
            "sm_clock": sm,
            "phase": phase,
            "a": a,
            "b": b,
            "c": c,
            "r2": r2,
            "n": len(grp),
        })

        for i, row in grp.iterrows():
            d = row.to_dict()
            d.pop("_phase", None)
            d["phase"] = phase
            d["predicted_power"] = (
                a * row["avg_gpu_util"] + b * row["avg_memory_util"] + c
            )
            d["a"] = a
            d["b"] = b
            d["c"] = c
            detail_records.append(d)

    coeffs_df = pd.DataFrame(coeff_records)
    detail_df = pd.DataFrame(detail_records)
    return coeffs_df, detail_df


def plot_actual_vs_pred(detail: pd.DataFrame, output_dir: Path) -> None:
    """sm_clock × phase별 실제 power vs 예측 power."""
    sm_clocks = sorted(detail["sm_clock"].unique())
    phases = sorted(detail["phase"].unique())

    for sm in sm_clocks:
        for phase in phases:
            sub = detail[(detail["sm_clock"] == sm) & (detail["phase"] == phase)].copy()
            if sub.empty:
                continue

            sub["label"] = sub["model_name"].str.replace("Qwen/Qwen3-8B_", "", regex=False)
            sub["label"] = sub["label"] + "\nil=" + sub["input_len"].astype(int).astype(str)
            sub = sub.sort_values(["model_name", "input_len"])

            grp = sub.groupby("label", sort=False).agg(
                actual=("avg_power_saturated", "mean"),
                predicted=("predicted_power", "mean"),
            ).reset_index()

            x_pos = np.arange(len(grp))
            bar_w = 0.35

            fig, ax = plt.subplots(figsize=(max(8, len(grp) * 0.8), 5))
            bars_a = ax.bar(
                x_pos - bar_w / 2, grp["actual"], bar_w,
                color="#1565C0", edgecolor="white", label="Actual",
            )
            bars_p = ax.bar(
                x_pos + bar_w / 2, grp["predicted"], bar_w,
                color="#FFCDD2", edgecolor="white", label="Predicted",
            )

            for bar, v in zip(bars_a, grp["actual"]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.1f}", ha="center", va="bottom", fontsize=7,
                    fontweight="bold", rotation=45,
                )
            for bar, v in zip(bars_p, grp["predicted"]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.1f}", ha="center", va="bottom", fontsize=7,
                    fontweight="bold", rotation=45, color="#C62828",
                )

            a_val = sub["a"].iloc[0]
            b_val = sub["b"].iloc[0]
            c_val = sub["c"].iloc[0]

            ax.set_xticks(x_pos)
            ax.set_xticklabels(grp["label"], rotation=45, ha="right", fontsize=7, fontweight="bold")
            ax.set_ylabel("Power (W)", fontweight="bold")
            ax.set_title(
                f"sm_clock={sm}  [{phase}]  P = {a_val:.3f}·gpu + {b_val:.3f}·mem + {c_val:.1f}  (b≥0)",
                fontsize=10, fontweight="bold",
            )
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)
            ax.tick_params(axis="y", labelsize=8)
            for label in ax.get_yticklabels():
                label.set_fontweight("bold")

            fig.tight_layout()
            fname = f"fit_sm{sm}_{phase}.png"
            fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  저장: {output_dir / fname}")


def plot_coefficients_by_phase(coeffs: pd.DataFrame, output_dir: Path) -> None:
    """phase별로 a, b, c를 sm_clock 축에 바 플롯."""
    for phase in sorted(coeffs["phase"].unique()):
        sub = coeffs[coeffs["phase"] == phase].sort_values("sm_clock")
        if sub.empty:
            continue

        sm_labels = [str(int(s)) for s in sub["sm_clock"]]
        x_pos = np.arange(len(sm_labels))

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for ax, col, label, color in zip(
            axes,
            ["a", "b", "c"],
            ["a (gpu_util coeff)", "b (memory_util coeff, ≥0)", "c (intercept)"],
            ["#1565C0", "#C62828", "#2E7D32"],
        ):
            vals = sub[col].values
            bars = ax.bar(x_pos, vals, color=color, edgecolor="white", width=0.7, alpha=0.8)
            for bar, v in zip(bars, vals):
                y = bar.get_height() if v >= 0 else 0
                va = "bottom" if v >= 0 else "top"
                ax.text(
                    bar.get_x() + bar.get_width() / 2, y,
                    f"{v:.3f}", ha="center", va=va, fontsize=7,
                    rotation=45, fontweight="bold",
                )
            ax.set_xticks(x_pos)
            ax.set_xticklabels(sm_labels, rotation=45, ha="right", fontsize=8, fontweight="bold")
            ax.set_xlabel("SM Clock (MHz)", fontweight="bold")
            ax.set_ylabel(label, fontweight="bold")
            ax.set_title(label, fontsize=11, fontweight="bold")
            ax.tick_params(axis="y", labelsize=8)
            for lb in ax.get_yticklabels():
                lb.set_fontweight("bold")
            ax.grid(axis="y", alpha=0.3)
            ax.axhline(0, color="black", linewidth=0.5)
            if col == "b":
                ax.axhline(0, color="red", linewidth=1, linestyle="--", alpha=0.5)

        fig.suptitle(
            f"Coefficients by SM Clock  [{phase}]  (b constrained ≥ 0)\n"
            f"P = a·gpu_util + b·memory_util + c",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout()
        fname = f"coefficients_abc_{phase}.png"
        fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  저장: {output_dir / fname}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Power = a*gpu_util + b*mem_util + c (b≥0), prefill/decoding 분리",
    )
    parser.add_argument("csv_path", type=str, help="iteration_stats.csv 경로")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent / "power_model"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("1. CSV 로드...")
    df = pd.read_csv(csv_path)
    print(f"   행 수: {len(df)}")

    print("2. (sm_clock × phase)별 회귀 (b≥0, prefill/decoding 분리)...")
    coeffs, detail = fit_per_sm_clock_phase(df)
    print(coeffs.sort_values(["phase", "sm_clock"]).to_string(index=False))

    coeffs.to_csv(output_dir / "coefficients.csv", index=False)
    detail.to_csv(output_dir / "detail_predictions.csv", index=False)

    print("\n3. 실제 vs 예측 플롯...")
    plot_actual_vs_pred(detail, output_dir)

    print("\n4. 계수 분포 플롯 (phase별)...")
    plot_coefficients_by_phase(coeffs, output_dir)

    print("\n완료!")


if __name__ == "__main__":
    main()
