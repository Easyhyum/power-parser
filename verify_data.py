"""
데이터 검증 스크립트
- graph_mode별 power 값 직접 확인
- 1830, 2130 MHz에서 all vs seg 비교
"""

import pandas as pd
from pathlib import Path

# 데이터 로드
log_dir = Path(r"C:\sourceCode\2026\power\log_v0")
csv_files = list(log_dir.glob("gpu_profile_*.csv"))

dfs = []
for f in csv_files:
    df = pd.read_csv(f)
    df['source_file'] = f.name
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)
print(f"Total rows: {len(df_all):,}")

# kv_cache_lens 그룹 정의
kv_cache_groups = {
    '128': (128 + 10, 128 + 29),
    '1024': (1024 + 10, 1024 + 29),
    '4096': (4096 + 10, 4096 + 29),
    '8192': (8192 + 10, 8192 + 29),
    '16384': (16384 + 10, 16384 + 29)
}

def get_kv_group(kv_len):
    for group_name, (min_val, max_val) in kv_cache_groups.items():
        if min_val <= kv_len < max_val:
            return group_name
    return None

df_all['kv_group'] = df_all['kv_cache_lens'].apply(get_kv_group)

# 필터링
df_filtered = df_all[df_all['kv_group'].notna()].copy()
df_filtered = df_filtered[df_filtered['index'] >= df_filtered['length'] * 0.5]

print("\n" + "="*80)
print("검증 1: SM Clock별, graph_mode별 원본 power 평균")
print("="*80)

for sm_clock in [1230, 1530, 1830, 2130, 2422]:
    df_sm = df_filtered[df_filtered['sm_clock'] == sm_clock]
    print(f"\n[SM Clock: {sm_clock} MHz]")
    print(f"  데이터 수: {len(df_sm):,}")
    
    for gmode in ['all', 'mani', 'seg']:
        df_mode = df_sm[df_sm['graph_mode'] == gmode]
        if len(df_mode) > 0:
            print(f"  {gmode}: 평균={df_mode['power'].mean():.2f}W, "
                  f"중앙값={df_mode['power'].median():.2f}W, "
                  f"표준편차={df_mode['power'].std():.2f}W, "
                  f"개수={len(df_mode):,}")

print("\n" + "="*80)
print("검증 2: batch_size=8, SM Clock=1830에서 상세 확인")
print("="*80)

for sm_clock in [1830, 2130]:
    print(f"\n[SM Clock: {sm_clock} MHz, Batch Size: 8]")
    
    df_check = df_filtered[(df_filtered['sm_clock'] == sm_clock) & 
                            (df_filtered['batch_size'] == 8)]
    
    for gmode in ['all', 'mani', 'seg']:
        df_mode = df_check[df_check['graph_mode'] == gmode]
        if len(df_mode) > 0:
            print(f"\n  {gmode}:")
            print(f"    데이터 수: {len(df_mode)}")
            print(f"    power 평균: {df_mode['power'].mean():.2f}W")
            print(f"    power 범위: {df_mode['power'].min():.2f} ~ {df_mode['power'].max():.2f}W")
            
            # 샘플 데이터 출력
            sample = df_mode[['graph_mode', 'batch_size', 'sm_clock', 'power', 'kv_group', 'source_file']].head(5)
            print(f"    샘플 데이터:")
            for _, row in sample.iterrows():
                print(f"      power={row['power']:.1f}W, kv={row['kv_group']}, file={row['source_file']}")

print("\n" + "="*80)
print("검증 3: 각 source_file별 graph_mode 분포 확인")
print("="*80)

for f in df_filtered['source_file'].unique():
    df_file = df_filtered[df_filtered['source_file'] == f]
    print(f"\n[{f}]")
    mode_counts = df_file.groupby('graph_mode').size()
    for mode, count in mode_counts.items():
        df_mode = df_file[df_file['graph_mode'] == mode]
        print(f"  {mode}: {count:,} rows, power avg={df_mode['power'].mean():.2f}W")

print("\n" + "="*80)
print("검증 4: graph_mode 컬럼의 고유값 및 분포")
print("="*80)
print(df_filtered['graph_mode'].value_counts())

print("\n" + "="*80)
print("검증 5: cudagraph_mode vs graph_mode 관계")
print("="*80)
print(pd.crosstab(df_filtered['cudagraph_mode'], df_filtered['graph_mode']))
