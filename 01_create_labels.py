#!/usr/bin/env python3
"""
01_create_labels.py - 建立自然產 PPH 標籤

PPH 定義（4個診斷名稱）：
1. Other immediate postpartum hemorrhage
2. Other immediate postpartum haemorrhage
3. Delayed and secondary postpartum hemorrhage
4. Delayed and secondary postpartum haemorrhage

輸出：vaginal_delivery_labels.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 70)
print("【步驟 1】建立自然產 PPH 標籤")
print("=" * 70)

# 讀取資料
print("\n讀取資料...")
patient = pd.read_csv('1.F2025627_2024產婦基本資料N1239.csv')
diag = pd.read_csv('3.F2025627_疾病診斷(2023-2024).csv')
surgery = pd.read_csv('16.F2025627_產婦手術術式(2024).csv')

print(f"基本資料: {len(patient)} 筆")
print(f"診斷資料: {len(diag)} 筆")
print(f"手術資料: {len(surgery)} 筆")

# 清理空格
surgery['ordproced_clean'] = surgery['ordproced'].str.strip().str.upper()
diag['condition_clean'] = diag['condition_concept_name'].str.strip()

# ===== 定義自然產 =====
vaginal_keywords = [
    'VAGINAL DELIVERY SINGLETOM',
    'VAGINAL DELIVERY IN COMPLICATED PREGNANCY',
    'VAGINAL DELIVERY MULTIPLE',
    'VAGINAL DELIVERY TWIN',
    'VAGINAL DELIVERY,SINGLETOM,PREVIOUS C/S',
    'VAGINAL DELIVERY,TWIN,PREVIOUS C/S'
]

# ===== PPH 定義 =====
pph_keywords = [
    'Other immediate postpartum hemorrhage',
    'Other immediate postpartum haemorrhage',
    'Delayed and secondary postpartum hemorrhage',
    'Delayed and secondary postpartum haemorrhage'
]

# 找出自然產病患及其分娩時間
vaginal_surgery = surgery[surgery['ordproced_clean'].isin(vaginal_keywords)].copy()
print(f"\n自然產紀錄數: {len(vaginal_surgery)}")
print(f"自然產病患數: {vaginal_surgery['PID'].nunique()}")

# 每位病患取最早的分娩時間（如果有多次）
vaginal_surgery['ordbgndttm'] = pd.to_datetime(vaginal_surgery['ordbgndttm'])
delivery_time = vaginal_surgery.groupby('PID')['ordbgndttm'].min().reset_index()
delivery_time.columns = ['PID', 'delivery_datetime']

# 找出 PPH 病患
pph_pids = set(diag[diag['condition_clean'].isin(pph_keywords)]['PID'].unique())

# 建立標籤資料
labels = delivery_time.copy()
labels['pph'] = labels['PID'].isin(pph_pids).astype(int)

# 加入年齡（從 patient 資料）
labels = labels.merge(patient[['PID', 'year_of_birth']], on='PID', how='left')
labels['delivery_year'] = labels['delivery_datetime'].dt.year
labels['age'] = labels['delivery_year'] - labels['year_of_birth']

# 建立時間截止點：分娩當天 00:00
labels['time_cutoff'] = labels['delivery_datetime'].dt.normalize()

# ===== 輸出統計 =====
print("\n" + "=" * 70)
print("【標籤統計】")
print("=" * 70)

n_total = len(labels)
n_pph = labels['pph'].sum()
n_no_pph = n_total - n_pph

print(f"自然產總人數: {n_total}")
print(f"PPH+ (陽性): {n_pph} ({n_pph/n_total*100:.1f}%)")
print(f"PPH- (陰性): {n_no_pph} ({n_no_pph/n_total*100:.1f}%)")
print(f"類別比例: 1:{n_no_pph/n_pph:.1f}")

print(f"\n年齡統計:")
print(f"  平均: {labels['age'].mean():.1f} 歲")
print(f"  標準差: {labels['age'].std():.1f}")
print(f"  範圍: {labels['age'].min()}-{labels['age'].max()} 歲")

# 各 PPH 類型的細節
print("\n" + "=" * 70)
print("【PPH 診斷類型分布】")
print("=" * 70)

vaginal_pids = set(labels['PID'])
for kw in pph_keywords:
    pids_with_this_diag = set(diag[diag['condition_clean'] == kw]['PID'].unique())
    n_vaginal_with_this = len(pids_with_this_diag & vaginal_pids)
    print(f"  {kw}: {n_vaginal_with_this} 人")

# 儲存
output_cols = ['PID', 'delivery_datetime', 'time_cutoff', 'age', 'pph']
labels[output_cols].to_csv('vaginal_delivery_labels.csv', index=False)
print(f"\n已儲存標籤檔案: vaginal_delivery_labels.csv")

# 額外輸出 PPH 定義文件
with open('pph_definition.txt', 'w') as f:
    f.write("PPH (產後出血) 定義\n")
    f.write("=" * 50 + "\n\n")
    f.write("包含以下 condition_concept_name:\n")
    for i, kw in enumerate(pph_keywords, 1):
        f.write(f"  {i}. {kw}\n")

print("已儲存 PPH 定義文件: pph_definition.txt")
print("\n完成！")
