"""
batch_size + kv_cache_lens 고정 조건에서 sm_clock 축 비교 그래프 생성

- x축: sm_clock
- y축: avg power, energy per token
- 같은 (batch_size, kv_group)마다 PNG 1개 생성
- legend: graph_mode
"""

from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def get_kv_group(kv_len: float):
    kv_cache_groups = {
        "128": (128 + 10, 128 + 29),
        "1024": (1024 + 10, 1024 + 29),
        "4096": (4096 + 10, 4096 + 29),
        "8192": (8192 + 10, 8192 + 29),
        "16384": (16384 + 10, 16384 + 29),
    }
    for group_name, (lower, upper) in kv_cache_groups.items():
        if lower <= kv_len < upper:
            return group_name
    return None


def main():
    parser = argparse.ArgumentParser(description="Plot sm_clock comparison for each batch_size and kv_cache_lens.")
    parser.add_argument("folder", nargs="?", default="log_v0", help="Folder name under C:\\sourceCode\\2026\\power")
    args = parser.parse_args()

    root = Path(r"C:\sourceCode\2026\power")
    log_dir = root / args.folder
    out_dir = root / "plots" / args.folder / "smclock_by_batch_kv"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(log_dir.glob("gpu_profile_*.csv"))
    print(f"CSV files: {len(csv_files)}")
    if not csv_files:
        print("No CSV files found.")
        return

    df = pd.concat([pd.read_csv(f).assign(source_file=f.name) for f in csv_files], ignore_index=True)
    print(f"Total rows: {len(df):,}")

    # 기존 규칙 유지
    df["kv_group"] = df["kv_cache_lens"].apply(get_kv_group)
    df = df[df["kv_group"].notna()].copy()

    sm_counts = df.groupby("sm_clock").size().sort_values(ascending=False)
    sm_threshold = len(df) * 0.01
    sm_keep = sm_counts[sm_counts >= sm_threshold].index.tolist()
    df = df[df["sm_clock"].isin(sm_keep)].copy()

    # index 후반부만 사용
    df = df[df["index"] >= df["length"] * 0.5].copy()
    print(f"Rows after filters: {len(df):,}")
    print(f"SM clocks used: {sm_keep}")

    # 실험 단위 키 구성 후 energy per token 계산
    df["exp_key"] = (
        df["graph_mode"].astype(str)
        + "_"
        + df["batch_size"].astype(str)
        + "_"
        + df["sm_clock"].astype(str)
        + "_"
        + df["kv_group"].astype(str)
        + "_"
        + df["kv_cache_lens"].astype(str)
        + "_"
        + df["repeat_count"].astype(str)
        + "_"
        + df["during_time"].astype(str)
        + "_"
        + df["source_file"].astype(str)
    )

    def calc_exp_metrics(group: pd.DataFrame) -> pd.Series:
        avg_power = group["power"].mean()
        during_time = group["during_time"].iloc[0]
        repeat_count = group["repeat_count"].iloc[0]
        batch_size = group["batch_size"].iloc[0]
        token_count = repeat_count * batch_size
        energy_per_token = (avg_power * during_time / token_count) if token_count > 0 else 0.0
        return pd.Series(
            {
                "batch_size": group["batch_size"].iloc[0],
                "kv_group": group["kv_group"].iloc[0],
                "kv_cache_lens": group["kv_cache_lens"].iloc[0],
                "graph_mode": group["graph_mode"].iloc[0],
                "sm_clock": group["sm_clock"].iloc[0],
                "avg_power": avg_power,
                "energy_per_token": energy_per_token,
                "points": len(group),
            }
        )

    exp_metrics = df.groupby("exp_key", as_index=False).apply(calc_exp_metrics).reset_index(drop=True)

    # 집계: 같은 조건 내 평균 power / energy_per_token
    agg = (
        exp_metrics.groupby(
            ["batch_size", "kv_group", "graph_mode", "sm_clock"], as_index=False
        )
        .agg(
            avg_power=("avg_power", "mean"),
            avg_energy_per_token=("energy_per_token", "mean"),
            points=("points", "sum"),
        )
    )

    # 모든 그래프 y축 고정
    y_min = agg["avg_power"].min()
    y_max = agg["avg_power"].max()
    y_pad = max((y_max - y_min) * 0.05, 1.0)
    fixed_ylim = (y_min - y_pad, y_max + y_pad)
    print(f"Fixed power y-range: {fixed_ylim[0]:.2f} ~ {fixed_ylim[1]:.2f}")

    e_min = agg["avg_energy_per_token"].min()
    e_max = agg["avg_energy_per_token"].max()
    e_pad = max((e_max - e_min) * 0.05, 0.01)
    fixed_e_ylim = (e_min - e_pad, e_max + e_pad)
    print(f"Fixed energy/token y-range: {fixed_e_ylim[0]:.3f} ~ {fixed_e_ylim[1]:.3f}")

    colors = {"all": "#1f77b4", "mani": "#ff7f0e", "seg": "#2ca02c"}
    mode_order = ["all", "mani", "seg"]

    file_count = 0
    # 요청 반영: kvlens를 그룹값(128/1024/4096/8192)으로 묶어서 처리
    target_kv_groups = ["128", "1024", "4096", "8192"]
    agg = agg[agg["kv_group"].isin(target_kv_groups)].copy()

    combos = agg[["batch_size", "kv_group"]].drop_duplicates().sort_values(["batch_size", "kv_group"])
    print(f"Target combinations: {len(combos):,}")

    for _, row in combos.iterrows():
        batch_size = int(row["batch_size"])
        kv_group = str(row["kv_group"])
        sub = agg[(agg["batch_size"] == batch_size) & (agg["kv_group"] == kv_group)]
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        for mode in mode_order:
            m = sub[sub["graph_mode"] == mode].sort_values("sm_clock")
            if m.empty:
                continue
            ax.plot(
                m["sm_clock"],
                m["avg_power"],
                "o-",
                color=colors.get(mode, "gray"),
                label=mode,
                markersize=6,
                linewidth=1.8,
            )

        ax.set_xlabel("SM Clock (MHz)")
        ax.set_ylabel("Average Power (W)")
        ax.set_title(f"Batch Size={batch_size}, KV Group={kv_group}\nPower vs SM Clock")
        ax.set_ylim(*fixed_ylim)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Graph Mode")

        plt.tight_layout()
        out_file = out_dir / f"batch_{batch_size}_kvgroup_{kv_group}_smclock_power.png"
        plt.savefig(out_file, dpi=150)
        plt.close(fig)
        file_count += 1

        # Energy per token plot
        fig, ax = plt.subplots(figsize=(10, 6))
        for mode in mode_order:
            m = sub[sub["graph_mode"] == mode].sort_values("sm_clock")
            if m.empty:
                continue
            ax.plot(
                m["sm_clock"],
                m["avg_energy_per_token"],
                "o-",
                color=colors.get(mode, "gray"),
                label=mode,
                markersize=6,
                linewidth=1.8,
            )

        ax.set_xlabel("SM Clock (MHz)")
        ax.set_ylabel("Energy per Token (J/token)")
        ax.set_title(f"Batch Size={batch_size}, KV Group={kv_group}\nEnergy per Token vs SM Clock")
        ax.set_ylim(*fixed_e_ylim)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Graph Mode")

        plt.tight_layout()
        out_file = out_dir / f"batch_{batch_size}_kvgroup_{kv_group}_smclock_energy_per_token.png"
        plt.savefig(out_file, dpi=150)
        plt.close(fig)
        file_count += 1

    agg.to_csv(out_dir / "smclock_by_batch_kv_source.csv", index=False)
    print(f"Done. Generated {file_count} PNG files.")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()

