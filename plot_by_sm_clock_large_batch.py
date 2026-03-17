"""
SM Clock별 + KV Cache 그룹별 그래프 생성
- Time per Token: Line Chart
- Power Avg: Bar Chart with values
- Energy per Token: Bar Chart with values
- Temperature Avg: Bar Chart with values
- x축: batch_size
- legend: graph_mode
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import os
# 폰트 설정
plt.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 인자 파싱
parser = argparse.ArgumentParser(description="Plot GPU metrics by SM clock/KV group from a folder.")
parser.add_argument("folder", nargs="?", default="log_v0", help="Folder name under C:\\sourceCode\\2026\\power")
args = parser.parse_args()

root_dir = Path(os.getcwd())
target_folder = args.folder

# 데이터 로드
log_dir = root_dir / target_folder
csv_files = list(log_dir.glob("gpu_profile_*.csv"))

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
sm_clocks_to_keep = [630,930, 1230, 1530, 1830, 2130, 2422]
print(f"SM Clocks to analyze: {sm_clocks_to_keep}")

# 필터링
df_base = df_all.copy()
df_base = df_base[df_base['sm_clock'].isin(sm_clocks_to_keep)]
# df_base = df_base[df_base['batch_size']]
# during_time 이상치 필터: 같은 조건(graph_mode, batch_size, sm_clock, input_len)에서
# 중앙값 대비 ±50% 벗어나는 실험 제거
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

# exp_key 생성 (graph_mode + batch_size + kv_cache_lens + sm_clock이 유니크)
df_base['exp_key'] = (
    df_base['graph_mode'].astype(str) + '_' +
    df_base['batch_size'].astype(str) + '_' +
    df_base['sm_clock'].astype(str) + '_' +
    df_base['kv_cache_lens'].astype(str) + '_' +
    df_base['input_len'].astype(str) + '_' +
    df_base['repeat_count'].astype(str) + '_' +
    df_base['during_time'].astype(str) + '_' +
    df_base['source_file'].astype(str)
)

# 1) 전체 구간에서 total_energy diff 계산 (0.5 필터 전)
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

    return pd.Series({
        'total_energy_diff_mj': total_energy_diff_mj,
        'total_energy_diff_j': total_energy_diff_j,
        'power_from_total_w': power_from_total_w,
        'energy_per_token_j': energy_per_token_j,
        'during_time': during_time,
        'repeat_count': repeat_count,
        'sm_clock': idx_sorted['sm_clock'].iloc[0],
        'batch_size': idx_sorted['batch_size'].iloc[0],
        'graph_mode': idx_sorted['graph_mode'].iloc[0],
        'input_len': idx_sorted['input_len'].iloc[0]
    })

energy_df = df_base.groupby('exp_key').apply(calculate_energy).reset_index()
print(f"Total experiments: {len(energy_df)}")

# 2) index >= length * 0.5, index <= length * 0.9, kv_cache_lens > input_len + 15 필터 후 avg_power, avg_temperature 계산
df_filtered = df_base[(df_base['index'] >= df_base['length'] * 0.5) & (df_base['index'] <= df_base['length'] * 0.9) & (df_base['kv_cache_lens'] > df_base['input_len'] + 15)].copy()
print(f"After index >= 50%, index <= 90%, kv_cache_lens > input_len + 15 filter: {len(df_filtered):,} rows")

def calculate_avg_metrics(group):
    avg_power = group['power'].mean()
    during_time = group['during_time'].iloc[0]
    repeat_count = group['repeat_count'].iloc[0]
    batch_size = group['batch_size'].iloc[0]
    token_count = repeat_count * batch_size
    calculated_energy = avg_power * during_time
    energy_per_token_from_avg = calculated_energy / token_count if token_count > 0 else 0
    return pd.Series({
        'avg_power': avg_power,
        'avg_temperature': group['temperature'].mean(),
        'energy_per_token_from_avg': energy_per_token_from_avg,
    })

avg_df = df_filtered.groupby('exp_key').apply(calculate_avg_metrics).reset_index()

# 3) 두 결과 merge
metrics_df = energy_df.merge(avg_df, on='exp_key', how='left')
metrics_df['time_per_token_s'] = metrics_df['during_time'] / metrics_df['repeat_count'].replace(0, np.nan)
metrics_df['time_per_token_s'] = metrics_df['time_per_token_s'].fillna(0)

# Energy 단위 확인
print(f"\n=== Energy Statistics ===")
print(f"total_energy_diff (mJ): min={metrics_df['total_energy_diff_mj'].min():.2f}, max={metrics_df['total_energy_diff_mj'].max():.2f}")
print(f"total_energy_diff (J):  min={metrics_df['total_energy_diff_j'].min():.2f}, max={metrics_df['total_energy_diff_j'].max():.2f}")
print(f"power_from_total (W):   min={metrics_df['power_from_total_w'].min():.2f}, max={metrics_df['power_from_total_w'].max():.2f}")
print(f"energy_per_token (J):   min={metrics_df['energy_per_token_j'].min():.2f}, max={metrics_df['energy_per_token_j'].max():.2f}")
print(f"energy_per_token_from_avg: min={metrics_df['energy_per_token_from_avg'].min():.2f}, max={metrics_df['energy_per_token_from_avg'].max():.2f}")
print(f"avg_temperature: min={metrics_df['avg_temperature'].min():.2f}, max={metrics_df['avg_temperature'].max():.2f}")

# 출력 디렉토리
output_dir = root_dir / "plots" / target_folder / "by_sm_input"
output_dir.mkdir(parents=True, exist_ok=True)

# 색상
colors = {'all': '#1f77b4', 'mani': '#ff7f0e', 'seg': '#2ca02c'}
graph_modes = ['all', 'mani', 'seg']
input_lens_available = sorted(metrics_df['input_len'].unique())

print(f"Input Lens available: {input_lens_available}")

# ============================================
# Input Len별 + SM Clock 비교 그래프 생성 (legend=SM Clock)
# ============================================

for input_len in input_lens_available:
    print(f"\n[InputLen: {input_len}]")
    df_len = metrics_df[metrics_df['input_len'] == input_len]
    if len(df_len) == 0:
        print("  No data")
        continue

    # input_len 기준 전체 batch_size를 사용 (batch_size=256 포함)
    batch_sizes = sorted(df_len['batch_size'].unique())
    if 256 in df_len['batch_size'].values:
        print("  batch_size=256 detected")

    # graph_mode는 평균내고, SM Clock 기준으로 비교
    compare_df = (
        df_len.groupby(['sm_clock', 'batch_size'], as_index=False)
        .agg(
            time_per_token_s=('time_per_token_s', 'mean'),
            avg_power=('avg_power', 'mean'),
            energy_per_token_from_avg=('energy_per_token_from_avg', 'mean'),
            power_from_total_w=('power_from_total_w', 'mean'),
            energy_per_token_j=('energy_per_token_j', 'mean'),
            avg_temperature=('avg_temperature', 'mean'),
        )
    )

    def plot_metric(metric_col, ylabel, title, file_suffix, yfmt=None):
        fig, ax = plt.subplots(figsize=(12, 7))
        sm_list = [sm for sm in sorted(sm_clocks_to_keep) if len(compare_df[compare_df['sm_clock'] == sm]) > 0]
        n_sm = len(sm_list)
        x = np.arange(len(batch_sizes))
        width = min(0.8 / max(n_sm, 1), 0.18)

        for i, sm_clock in enumerate(sm_list):
            sm_sub = compare_df[compare_df['sm_clock'] == sm_clock]
            yvals = sm_sub.set_index('batch_size')[metric_col].reindex(batch_sizes).fillna(0)
            pos = x + (i - (n_sm - 1) / 2) * width
            bars = ax.bar(pos, yvals.values, width=width, label=f'SM {sm_clock}')

            if yfmt is not None:
                for bar, y in zip(bars, yvals.values):
                    if y > 0:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height(),
                            format(y, yfmt),
                            fontsize=7,
                            ha='center',
                            va='bottom',
                        )

        ax.set_xlabel('Batch Size', fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(f'Input Len: {input_len}\n{title}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(batch_sizes)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(title='SM Clock', fontsize=10, title_fontsize=11)
        ax.tick_params(axis='both', labelsize=11)
        plt.tight_layout()
        plt.savefig(output_dir / f'input_{input_len}_{file_suffix}.png', dpi=150)
        plt.close()

    # 1) Time per token
    plot_metric('time_per_token_s', 'Time per Token (s/token)', 'Time per Token by Batch Size', 'time_per_token')

    # 2) Avg power
    plot_metric('avg_power', 'Average Power (W)', 'Average Power by Batch Size', 'power', yfmt='.4f')

    # 3) Energy per token (avg power 기반)
    plot_metric(
        'energy_per_token_from_avg',
        'Energy per Token (J/token)',
        'Energy per Token by Batch Size (Power x Time / Repeat Count)',
        'energy_per_token',
        yfmt='.4f'
    )

    # 3-2) Power from total_energy
    plot_metric(
        'power_from_total_w',
        'Power (W)',
        'Power from total_energy by Batch Size\n(total_energy_diff mJ / 1000 / during_time)',
        'power_from_total',
        yfmt='.4f'
    )

    # 3-3) Energy per token from total_energy
    plot_metric(
        'energy_per_token_j',
        'Energy per Token (J/token)',
        'Energy per Token from total_energy by Batch Size\n(total_energy_diff mJ / 1000 / (repeat_count * batch_size))',
        'energy_per_token_total',
        yfmt='.4f'
    )

    # 4) Temperature
    plot_metric('avg_temperature', 'Average Temperature (C)', 'Average Temperature by Batch Size', 'temperature', yfmt='.4f')
    print(f"  Saved (rows: {len(df_len)})")

# ============================================
# 요약 테이블 출력
# ============================================
print(f"\n{'='*80}")
print("Energy per Token Summary (J/token)")
print(f"{'='*80}")

summary_data = []
for sm_clock in sorted(sm_clocks_to_keep):
    for input_len in input_lens_available:
        for gmode in graph_modes:
            df_subset = metrics_df[(metrics_df['sm_clock'] == sm_clock) & 
                                   (metrics_df['input_len'] == input_len) &
                                   (metrics_df['graph_mode'] == gmode)]
            if len(df_subset) > 0:
                summary_data.append({
                    'sm_clock': sm_clock,
                    'input_len': input_len,
                    'graph_mode': gmode,
                    'avg_power_mean': df_subset['avg_power'].mean(),
                    'avg_temperature_mean': df_subset['avg_temperature'].mean(),
                    'during_time_mean': df_subset['during_time'].mean(),
                    'repeat_count': df_subset['repeat_count'].iloc[0],
                    'energy_per_token_mean': df_subset['energy_per_token_from_avg'].mean(),
                    'experiment_count': len(df_subset)
                })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(output_dir / 'energy_summary.csv', index=False)
print(f"\nSummary saved to: {output_dir / 'energy_summary.csv'}")

# 샘플 출력
print(f"\nSample (SM 1830 MHz, InputLen 1024):")
sample = summary_df[(summary_df['sm_clock'] == 1830) & (summary_df['input_len'] == 1024)]
for _, row in sample.iterrows():
    print(f"  {row['graph_mode']}: Power={row['avg_power_mean']:.4f}W, "
          f"Time={row['during_time_mean']:.2f}s, "
          f"Repeat={row['repeat_count']}, "
          f"Energy/Token={row['energy_per_token_mean']:.4f}J")

print(f"\n=== Complete ===")
print(f"Output directory: {output_dir}")

# 파일 목록 출력
files = sorted(output_dir.glob("*.png"))
print(f"\nGenerated {len(files)} PNG files")

# 메트릭 데이터 저장
metrics_df.to_csv(output_dir / 'metrics_with_energy.csv', index=False)
print(f"Metrics saved to: {output_dir / 'metrics_with_energy.csv'}")
