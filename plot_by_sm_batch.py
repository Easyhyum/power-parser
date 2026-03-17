"""
SM Clock별 + Batch Size 그룹별 그래프 생성
- Time per Token: Line Chart
- Power Avg: Bar Chart with values
- Energy per Token: Bar Chart with values
- Temperature Avg: Bar Chart with values
- x축: input_len
- legend: graph_mode
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 폰트 설정
plt.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 인자 파싱
parser = argparse.ArgumentParser(description="Plot GPU metrics by SM clock/batch group from a folder.")
parser.add_argument("folder", nargs="?", default="log_v0", help="Folder name under C:\\code\\power_plot")
args = parser.parse_args()

root_dir = Path(r"C:\code\power_plot")
target_folder = args.folder

# 데이터 로드
log_dir = root_dir / target_folder
csv_files = list(log_dir.glob("gpu_profile_*.csv"))

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in: {log_dir}")

print(f"Loading {len(csv_files)} CSV files...")

dfs = []
for f in csv_files:
    df = pd.read_csv(f)
    df['source_file'] = f.name
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)
print(f"Total rows: {len(df_all):,}")

# sm_clock 필터링 (상위 1% 이상 데이터만)
sm_clock_counts = df_all.groupby('sm_clock').size().sort_values(ascending=False)
threshold = len(df_all) * 0.01
sm_clocks_to_keep = sm_clock_counts[sm_clock_counts >= threshold].index.tolist()
print(f"SM Clocks to analyze: {sm_clocks_to_keep}")

# 필터링
df_base = df_all.copy()
df_base = df_base[df_base['sm_clock'].isin(sm_clocks_to_keep)]

# during_time 이상치 필터: 같은 조건(graph_mode, batch_size, sm_clock, input_len)에서 중앙값 대비 ±30% 벗어나는 실험 제거
group_cols = ['graph_mode', 'batch_size', 'sm_clock', 'input_len']
run_key_cols = ['graph_mode', 'batch_size', 'kv_cache_lens', 'sm_clock']
exp_during = df_base.drop_duplicates(subset=run_key_cols)[group_cols + ['kv_cache_lens', 'during_time']]
median_during = exp_during.groupby(group_cols)['during_time'].transform('median')
exp_during = exp_during.copy()
exp_during['is_valid'] = (exp_during['during_time'] >= median_during * 0.7) & (exp_during['during_time'] <= median_during * 1.3)
before_count = len(exp_during)
removed_count = (~exp_during['is_valid']).sum()
invalid_df = exp_during[~exp_during['is_valid']][run_key_cols]
df_base = df_base.merge(invalid_df, on=run_key_cols, how='left', indicator=True)
df_base = df_base[df_base['_merge'] == 'left_only'].drop(columns=['_merge'])
print(f"During time outlier filter: {before_count} experiments, {removed_count} removed")

# exp_key 생성
df_base['exp_key'] = (
    df_base['graph_mode'].astype(str) + '_'
    + df_base['batch_size'].astype(str) + '_'
    + df_base['sm_clock'].astype(str) + '_'
    + df_base['kv_cache_lens'].astype(str) + '_'
    + df_base['input_len'].astype(str) + '_'
    + df_base['repeat_count'].astype(str) + '_'
    + df_base['during_time'].astype(str) + '_'
    + df_base['source_file'].astype(str)
)


def calculate_energy(group):
    idx_sorted = group.sort_values('index')
    first_total_energy = idx_sorted['total_energy'].iloc[0]
    last_total_energy = idx_sorted['total_energy'].iloc[-1]
    total_energy_diff_mj = last_total_energy - first_total_energy
    total_energy_diff_j = total_energy_diff_mj / 1000.0
    repeat_count = idx_sorted['repeat_count'].iloc[0]
    during_time = idx_sorted['during_time'].iloc[0]

    power_from_total_w = total_energy_diff_j / during_time if during_time > 0 else 0
    batch_size = idx_sorted['batch_size'].iloc[0]
    token_count = repeat_count * batch_size
    energy_per_token_j = total_energy_diff_j / token_count if token_count > 0 else 0

    return pd.Series(
        {
            'total_energy_diff_mj': total_energy_diff_mj,
            'total_energy_diff_j': total_energy_diff_j,
            'power_from_total_w': power_from_total_w,
            'energy_per_token_j': energy_per_token_j,
            'during_time': during_time,
            'repeat_count': repeat_count,
            'sm_clock': idx_sorted['sm_clock'].iloc[0],
            'batch_size': idx_sorted['batch_size'].iloc[0],
            'graph_mode': idx_sorted['graph_mode'].iloc[0],
            'input_len': idx_sorted['input_len'].iloc[0],
        }
    )


energy_df = df_base.groupby('exp_key').apply(calculate_energy).reset_index()
print(f"Total experiments: {len(energy_df)}")

# index 50~90%, kv_cache_lens > input_len + 15 구간 평균 파워/온도
df_filtered = df_base[
    (df_base['index'] >= df_base['length'] * 0.5)
    & (df_base['index'] <= df_base['length'] * 0.9)
    & (df_base['kv_cache_lens'] > df_base['input_len'] + 15)
].copy()
print(f"After index >= 50%, index <= 90%, kv_cache_lens > input_len + 15 filter: {len(df_filtered):,} rows")


def calculate_avg_metrics(group):
    avg_power = group['power'].mean()
    during_time = group['during_time'].iloc[0]
    repeat_count = group['repeat_count'].iloc[0]
    batch_size = group['batch_size'].iloc[0]
    token_count = repeat_count * batch_size
    calculated_energy = avg_power * during_time
    energy_per_token_from_avg = calculated_energy / token_count if token_count > 0 else 0
    return pd.Series(
        {
            'avg_power': avg_power,
            'avg_temperature': group['temperature'].mean(),
            'energy_per_token_from_avg': energy_per_token_from_avg,
        }
    )


avg_df = df_filtered.groupby('exp_key').apply(calculate_avg_metrics).reset_index()

# 두 결과 merge
metrics_df = energy_df.merge(avg_df, on='exp_key', how='left')
metrics_df['time_per_token_s'] = metrics_df['during_time'] / metrics_df['repeat_count'].replace(0, np.nan)
metrics_df['time_per_token_s'] = metrics_df['time_per_token_s'].fillna(0)

print("\n=== Energy Statistics ===")
print(f"total_energy_diff (mJ): min={metrics_df['total_energy_diff_mj'].min():.2f}, max={metrics_df['total_energy_diff_mj'].max():.2f}")
print(f"total_energy_diff (J):  min={metrics_df['total_energy_diff_j'].min():.2f}, max={metrics_df['total_energy_diff_j'].max():.2f}")
print(f"power_from_total (W):   min={metrics_df['power_from_total_w'].min():.2f}, max={metrics_df['power_from_total_w'].max():.2f}")
print(f"energy_per_token (J):   min={metrics_df['energy_per_token_j'].min():.2f}, max={metrics_df['energy_per_token_j'].max():.2f}")
print(
    f"energy_per_token_from_avg: min={metrics_df['energy_per_token_from_avg'].min():.2f}, "
    f"max={metrics_df['energy_per_token_from_avg'].max():.2f}"
)
print(f"avg_temperature: min={metrics_df['avg_temperature'].min():.2f}, max={metrics_df['avg_temperature'].max():.2f}")

# 출력 디렉토리
output_dir = root_dir / "plots" / target_folder / "by_sm_batch"
output_dir.mkdir(parents=True, exist_ok=True)

colors = {'all': '#1f77b4', 'mani': '#ff7f0e', 'seg': '#2ca02c'}
graph_modes = ['all', 'mani', 'seg']
batch_sizes_available = sorted(metrics_df['batch_size'].unique())

print(f"Batch sizes available: {batch_sizes_available}")

for sm_clock in sorted(sm_clocks_to_keep):
    print(f"\n[SM Clock: {sm_clock} MHz]")
    df_sm = metrics_df[metrics_df['sm_clock'] == sm_clock]

    for batch_size in batch_sizes_available:
        df_batch = df_sm[df_sm['batch_size'] == batch_size]

        if len(df_batch) == 0:
            print(f"  BatchSize {batch_size}: No data")
            continue

        input_lens = sorted(df_batch['input_len'].unique())
        x = np.arange(len(input_lens))
        width = 0.25

        # 1) Time per Token (line)
        fig, ax = plt.subplots(figsize=(12, 7))
        for gmode in graph_modes:
            df_mode = df_batch[df_batch['graph_mode'] == gmode]
            if len(df_mode) > 0:
                avg_by_input = df_mode.groupby('input_len')['time_per_token_s'].mean().sort_index()
                x_vals = avg_by_input.index.tolist()
                y_vals = avg_by_input.values
                ax.plot(x_vals, y_vals, 'o-', label=gmode, color=colors[gmode], markersize=10, linewidth=2.5)

        ax.set_xlabel('Input Length', fontsize=14)
        ax.set_ylabel('Time per Token (s/token)', fontsize=14)
        ax.set_title(
            f'SM Clock: {sm_clock} MHz | Batch Size: {batch_size}\nTime per Token by Input Length',
            fontsize=14,
            fontweight='bold',
        )
        ax.legend(title='Graph Mode', fontsize=12, title_fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(input_lens)
        ax.tick_params(axis='both', labelsize=11)
        plt.tight_layout()
        plt.savefig(output_dir / f'sm_{sm_clock}_batch_{batch_size}_time_per_token.png', dpi=150)
        plt.close()

        # 2) Average Power (bar)
        fig, ax = plt.subplots(figsize=(14, 8))
        for i, gmode in enumerate(graph_modes):
            df_mode = df_batch[df_batch['graph_mode'] == gmode]
            if len(df_mode) > 0:
                avg_by_input = df_mode.groupby('input_len')['avg_power'].mean()
                values = [avg_by_input.get(il, 0) for il in input_lens]
                bars = ax.bar(x + i * width, values, width, label=gmode, color=colors[gmode])
                for bar, val in zip(bars, values):
                    if val > 0:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 1,
                            f'{val:.1f}',
                            ha='center',
                            va='bottom',
                            fontsize=8,
                            rotation=45,
                        )

        ax.set_xlabel('Input Length', fontsize=14)
        ax.set_ylabel('Average Power (W)', fontsize=14)
        ax.set_title(
            f'SM Clock: {sm_clock} MHz | Batch Size: {batch_size}\nAverage Power by Input Length',
            fontsize=14,
            fontweight='bold',
        )
        ax.set_xticks(x + width)
        ax.set_xticklabels(input_lens)
        ax.legend(title='Graph Mode', fontsize=12, title_fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='both', labelsize=11)
        ymax = ax.get_ylim()[1]
        ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1)
        plt.tight_layout()
        plt.savefig(output_dir / f'sm_{sm_clock}_batch_{batch_size}_power.png', dpi=150)
        plt.close()

        # 3) Energy per Token from avg_power (bar)
        fig, ax = plt.subplots(figsize=(14, 8))
        for i, gmode in enumerate(graph_modes):
            df_mode = df_batch[df_batch['graph_mode'] == gmode]
            if len(df_mode) > 0:
                avg_by_input = df_mode.groupby('input_len')['energy_per_token_from_avg'].mean()
                values = [avg_by_input.get(il, 0) for il in input_lens]
                bars = ax.bar(x + i * width, values, width, label=gmode, color=colors[gmode])
                for bar, val in zip(bars, values):
                    if val > 0:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.05,
                            f'{val:.3f}',
                            ha='center',
                            va='bottom',
                            fontsize=8,
                            rotation=45,
                        )

        ax.set_xlabel('Input Length', fontsize=14)
        ax.set_ylabel('Energy per Token (J/token)', fontsize=14)
        ax.set_title(
            f'SM Clock: {sm_clock} MHz | Batch Size: {batch_size}\nEnergy per Token by Input Length\n(Power x Time / Repeat Count)',
            fontsize=14,
            fontweight='bold',
        )
        ax.set_xticks(x + width)
        ax.set_xticklabels(input_lens)
        ax.legend(title='Graph Mode', fontsize=12, title_fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='both', labelsize=11)
        ymax = ax.get_ylim()[1]
        ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1)
        plt.tight_layout()
        plt.savefig(output_dir / f'sm_{sm_clock}_batch_{batch_size}_energy_per_token.png', dpi=150)
        plt.close()

        # 4) Power from total_energy (bar)
        fig, ax = plt.subplots(figsize=(14, 8))
        for i, gmode in enumerate(graph_modes):
            df_mode = df_batch[df_batch['graph_mode'] == gmode]
            if len(df_mode) > 0:
                avg_by_input = df_mode.groupby('input_len')['power_from_total_w'].mean()
                values = [avg_by_input.get(il, 0) for il in input_lens]
                bars = ax.bar(x + i * width, values, width, label=gmode, color=colors[gmode])
                for bar, val in zip(bars, values):
                    if val > 0:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 1,
                            f'{val:.1f}',
                            ha='center',
                            va='bottom',
                            fontsize=8,
                            rotation=45,
                        )

        ax.set_xlabel('Input Length', fontsize=14)
        ax.set_ylabel('Power (W)', fontsize=14)
        ax.set_title(
            f'SM Clock: {sm_clock} MHz | Batch Size: {batch_size}\nPower from total_energy by Input Length\n(total_energy_diff mJ / 1000 / during_time)',
            fontsize=14,
            fontweight='bold',
        )
        ax.set_xticks(x + width)
        ax.set_xticklabels(input_lens)
        ax.legend(title='Graph Mode', fontsize=12, title_fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='both', labelsize=11)
        ymax = ax.get_ylim()[1]
        ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1)
        plt.tight_layout()
        plt.savefig(output_dir / f'sm_{sm_clock}_batch_{batch_size}_power_from_total.png', dpi=150)
        plt.close()

        # 5) Energy per Token from total_energy (bar)
        fig, ax = plt.subplots(figsize=(14, 8))
        for i, gmode in enumerate(graph_modes):
            df_mode = df_batch[df_batch['graph_mode'] == gmode]
            if len(df_mode) > 0:
                avg_by_input = df_mode.groupby('input_len')['energy_per_token_j'].mean()
                values = [avg_by_input.get(il, 0) for il in input_lens]
                bars = ax.bar(x + i * width, values, width, label=gmode, color=colors[gmode])
                for bar, val in zip(bars, values):
                    if val > 0:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.05,
                            f'{val:.3f}',
                            ha='center',
                            va='bottom',
                            fontsize=8,
                            rotation=45,
                        )

        ax.set_xlabel('Input Length', fontsize=14)
        ax.set_ylabel('Energy per Token (J/token)', fontsize=14)
        ax.set_title(
            f'SM Clock: {sm_clock} MHz | Batch Size: {batch_size}\nEnergy per Token from total_energy by Input Length\n(total_energy_diff mJ / 1000 / (repeat_count * batch_size))',
            fontsize=14,
            fontweight='bold',
        )
        ax.set_xticks(x + width)
        ax.set_xticklabels(input_lens)
        ax.legend(title='Graph Mode', fontsize=12, title_fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='both', labelsize=11)
        ymax = ax.get_ylim()[1]
        ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1)
        plt.tight_layout()
        plt.savefig(output_dir / f'sm_{sm_clock}_batch_{batch_size}_energy_per_token_total.png', dpi=150)
        plt.close()

        # 6) Average Temperature (bar)
        fig, ax = plt.subplots(figsize=(14, 8))
        for i, gmode in enumerate(graph_modes):
            df_mode = df_batch[df_batch['graph_mode'] == gmode]
            if len(df_mode) > 0:
                avg_by_input = df_mode.groupby('input_len')['avg_temperature'].mean()
                values = [avg_by_input.get(il, 0) for il in input_lens]
                bars = ax.bar(x + i * width, values, width, label=gmode, color=colors[gmode])
                for bar, val in zip(bars, values):
                    if val > 0:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.05,
                            f'{val:.1f}',
                            ha='center',
                            va='bottom',
                            fontsize=8,
                            rotation=45,
                        )

        ax.set_xlabel('Input Length', fontsize=14)
        ax.set_ylabel('Average Temperature (C)', fontsize=14)
        ax.set_title(
            f'SM Clock: {sm_clock} MHz | Batch Size: {batch_size}\nAverage Temperature by Input Length',
            fontsize=14,
            fontweight='bold',
        )
        ax.set_xticks(x + width)
        ax.set_xticklabels(input_lens)
        ax.legend(title='Graph Mode', fontsize=12, title_fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='both', labelsize=11)
        ymax = ax.get_ylim()[1]
        ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1)
        plt.tight_layout()
        plt.savefig(output_dir / f'sm_{sm_clock}_batch_{batch_size}_temperature.png', dpi=150)
        plt.close()

        print(f"  BatchSize {batch_size}: Saved (data: {len(df_batch)})")

print(f"\n{'=' * 80}")
print("Energy per Token Summary (J/token)")
print(f"{'=' * 80}")

summary_data = []
for sm_clock in sorted(sm_clocks_to_keep):
    for batch_size in batch_sizes_available:
        for gmode in graph_modes:
            df_subset = metrics_df[
                (metrics_df['sm_clock'] == sm_clock)
                & (metrics_df['batch_size'] == batch_size)
                & (metrics_df['graph_mode'] == gmode)
            ]
            if len(df_subset) > 0:
                summary_data.append(
                    {
                        'sm_clock': sm_clock,
                        'batch_size': batch_size,
                        'graph_mode': gmode,
                        'avg_power_mean': df_subset['avg_power'].mean(),
                        'avg_temperature_mean': df_subset['avg_temperature'].mean(),
                        'during_time_mean': df_subset['during_time'].mean(),
                        'repeat_count': df_subset['repeat_count'].iloc[0],
                        'energy_per_token_mean': df_subset['energy_per_token_from_avg'].mean(),
                        'experiment_count': len(df_subset),
                    }
                )

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(output_dir / 'energy_summary.csv', index=False)
print(f"\nSummary saved to: {output_dir / 'energy_summary.csv'}")

files = sorted(output_dir.glob("*.png"))
print(f"\nGenerated {len(files)} PNG files")

metrics_df.to_csv(output_dir / 'metrics_with_energy.csv', index=False)
print(f"Metrics saved to: {output_dir / 'metrics_with_energy.csv'}")
print(f"Output directory: {output_dir}")
