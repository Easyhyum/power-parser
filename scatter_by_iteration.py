"""
Decoding 기준(kv_cache_lens)별 scatter plot 생성 스크립트

규칙:
- kv_cache_lens 그룹: 각 기준값 +10 ~ +28
- sm_clock: 전체 데이터의 1% 이상인 값만 사용
- index >= length * 0.5 데이터만 사용

출력:
- decoding(kv_cache_lens)별, sm_clock별, kv_group별 scatter PNG
  - x축: batch_size
  - y축(좌): 평균 소비전력(power mean)
  - y축(우): total_energy 기반 총 소비량(실험 시작점 대비 delta)
"""

from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def get_kv_group(kv_len: float):
    kv_cache_groups = {
        "128": (128 + 10, 128 + 29),       # 138 ~ 156
        "1024": (1024 + 10, 1024 + 29),    # 1034 ~ 1052
        "4096": (4096 + 10, 4096 + 29),    # 4106 ~ 4124
        "8192": (8192 + 10, 8192 + 29),    # 8202 ~ 8220
        "16384": (16384 + 10, 16384 + 29), # 16394 ~ 16412
    }
    for group_name, (lower, upper) in kv_cache_groups.items():
        if lower <= kv_len < upper:
            return group_name
    return None


def main():
    parser = argparse.ArgumentParser(description="Generate scatter plots by decoding from a folder.")
    parser.add_argument("folder", nargs="?", default="log_v0", help="Folder name under C:\\sourceCode\\2026\\power")
    args = parser.parse_args()

    root = Path(r"C:\sourceCode\2026\power")
    folder = args.folder
    log_dir = root / folder
    out_dir = root / "plots" / folder / "scatter_by_iteration"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(log_dir.glob("gpu_profile_*.csv"))
    print(f"CSV files: {len(csv_files)}")
    if not csv_files:
        print("No CSV files found.")
        return

    df = pd.concat(
        [pd.read_csv(f).assign(source_file=f.name) for f in csv_files],
        ignore_index=True,
    )
    print(f"Total rows: {len(df):,}")

    # 기존 규칙 적용(기본 필터)
    df["kv_group"] = df["kv_cache_lens"].apply(get_kv_group)
    df = df[df["kv_group"].notna()].copy()

    sm_clock_counts = df.groupby("sm_clock").size().sort_values(ascending=False)
    sm_threshold = len(df) * 0.01
    sm_keep = sm_clock_counts[sm_clock_counts >= sm_threshold].index.tolist()
    df = df[df["sm_clock"].isin(sm_keep)].copy()

    print(f"Rows after base filters: {len(df):,}")
    print(f"SM clocks: {sm_keep}")

    # 실험 키: 동일 실행 단위
    df["exp_key"] = (
        df["graph_mode"].astype(str)
        + "_"
        + df["batch_size"].astype(str)
        + "_"
        + df["sm_clock"].astype(str)
        + "_"
        + df["kv_group"].astype(str)
        + "_"
        + df["repeat_count"].astype(str)
        + "_"
        + df["during_time"].astype(str)
        + "_"
        + df["source_file"].astype(str)
    )

    # index 필터 이전 원본(=kv/sm 및 validation만 적용된 상태)을 energy endpoint 계산 소스로 사용
    df_energy_source = df.copy()

    # total_energy endpoint delta 계산:
    # - 시작점: index==0 우선, 없으면 index==1 사용
    # - 종료점: index==length
    start_at_0 = (
        df_energy_source[df_energy_source["index"] == 0][["exp_key", "total_energy"]]
        .groupby("exp_key", as_index=False)
        .first()
        .rename(columns={"total_energy": "start_total_energy"})
    )
    start_at_1 = (
        df_energy_source[df_energy_source["index"] == 1][["exp_key", "total_energy"]]
        .groupby("exp_key", as_index=False)
        .first()
        .rename(columns={"total_energy": "start_total_energy"})
    )

    end_at_length = (
        df_energy_source[df_energy_source["index"] == df_energy_source["length"]][["exp_key", "total_energy"]]
        .groupby("exp_key", as_index=False)
        .first()
        .rename(columns={"total_energy": "end_total_energy"})
    )

    # index==0이 실제로 없으면 fallback(index==1)을 사용
    if start_at_0.empty:
        print("[energy] index==0 rows not found. Falling back to index==1 as start point.")
        start_points = start_at_1
    else:
        start_points = start_at_0

    exp_energy = start_points.merge(end_at_length, on="exp_key", how="inner")
    exp_energy["total_energy_delta_endpoint"] = (
        exp_energy["end_total_energy"] - exp_energy["start_total_energy"]
    )
    print(
        f"[energy] endpoint pairs: {len(exp_energy):,} "
        f"(start candidates={len(start_points):,}, end candidates={len(end_at_length):,})"
    )

    # plotting용 데이터는 기존 규칙대로 후반 50%만 사용 (delta 계산은 이미 그 전에 완료)
    df_plot = df[df["index"] >= df["length"] * 0.5].copy()
    print(f"Rows after index>=50% filter (for plotting): {len(df_plot):,}")

    # endpoint delta를 plotting 데이터에 병합
    df_plot = df_plot.merge(
        exp_energy[["exp_key", "total_energy_delta_endpoint"]],
        on="exp_key",
        how="inner",
    )
    print(f"Rows after endpoint merge: {len(df_plot):,}")

    # decoding 기준(kv_cache_lens) 단위로 집계
    agg = (
        df_plot.groupby(
            ["sm_clock", "kv_group", "kv_cache_lens", "graph_mode", "batch_size"], as_index=False
        )
        .agg(
            avg_power=("power", "mean"),
            avg_total_energy_delta=("total_energy_delta_endpoint", "mean"),
            avg_print_index=("index", "mean"),  # 참고용: 샘플링 순번 평균
            points=("power", "size"),
        )
    )

    print(f"Aggregated rows: {len(agg):,}")
    print("Generating scatter plots (one file per sm_clock + kv_group) ...")

    colors = {"all": "#1f77b4", "mani": "#ff7f0e", "seg": "#2ca02c"}
    kv_order = ["128", "1024", "4096", "8192", "16384"]
    mode_order = ["all", "mani", "seg"]

    # 모든 PNG에서 동일한 power 범위 사용
    power_min = agg["avg_power"].min()
    power_max = agg["avg_power"].max()
    power_pad = max((power_max - power_min) * 0.05, 1.0)
    fixed_power_ylim = (power_min - power_pad, power_max + power_pad)
    print(f"Fixed power y-range: {fixed_power_ylim[0]:.2f} ~ {fixed_power_ylim[1]:.2f}")

    total_files = 0
    sm_list = sorted(agg["sm_clock"].unique())

    for sm in sm_list:
        for kv_group in [k for k in kv_order if k in agg["kv_group"].unique()]:
            sub = agg[(agg["kv_group"] == kv_group) & (agg["sm_clock"] == sm)]
            if sub.empty:
                continue

            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            batch_sizes = sorted(sub["batch_size"].unique())
            x_map = {
                (bs, mode): (i * len(mode_order) + j)
                for i, bs in enumerate(batch_sizes)
                for j, mode in enumerate(mode_order)
            }
            xticks = [x_map[(bs, mode)] for bs in batch_sizes for mode in mode_order]
            xlabels = [f"{bs}\n{mode}" for bs in batch_sizes for mode in mode_order]

        # power scatter only
            for mode in mode_order:
                m = sub[sub["graph_mode"] == mode]
                if m.empty:
                    continue
                x_values = [x_map[(bs, mode)] for bs in m["batch_size"]]
                ax.scatter(
                    x_values,
                    m["avg_power"],
                    color=colors.get(mode, "gray"),
                    marker="o",
                    alpha=0.6,
                    s=28,
                )
            ax.set_xlabel("Batch Size / Graph Mode")
            ax.set_ylabel("Average Power (W)")
            ax.set_title("Power Scatter")
            ax.set_ylim(*fixed_power_ylim)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xlabels, fontsize=8)
            ax.grid(alpha=0.3)

            # batch_size 그룹(all/mani/seg) 경계에 진한 구분선 추가
            for i in range(len(batch_sizes) - 1):
                group_end = (i + 1) * len(mode_order) - 0.5
                ax.axvline(x=group_end, color="black", linewidth=1.8, alpha=0.75)

        # 범례 2개: color=mode, marker=sm_clock
            mode_handles = [
                plt.Line2D([0], [0], marker="o", color="w", label=mode,
                           markerfacecolor=colors[mode], markersize=8)
                for mode in mode_order
            ]
            ax.legend(handles=mode_handles, title="Graph Mode", loc="upper left")

            fig.suptitle(
                f"SM {sm} | KV Group {kv_group} (+10~+28)",
                fontsize=13,
            )
            fig.tight_layout()

            out_file = out_dir / f"sm_{sm}_kv_{kv_group}_scatter.png"
            fig.savefig(out_file, dpi=150)
            plt.close(fig)
            total_files += 1

    # 집계 CSV 저장
    agg.to_csv(out_dir / "decoding_scatter_source.csv", index=False)
    print(f"Done. Generated {total_files} PNG files.")
    print(f"Output: {out_dir}")
    print(f"Source table: {out_dir / 'decoding_scatter_source.csv'}")


if __name__ == "__main__":
    main()

