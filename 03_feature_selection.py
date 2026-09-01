#!/usr/bin/env python3
"""
03_feature_selection_WARNING_threshold60pct_NOT_used_in_final_model.py - 特徵選擇（Bootstrap + LASSO）
⚠️ WARNING: 此腳本使用 SELECTION_THRESHOLD = 0.6（≥60% 門檻篩選），但最終模型並非以此方式選擇特徵。
   實際做法見 train_RF_SMOTEENN_Top30.py 第 211 行：freq_df.head(30)（依排名取 Top N）。

方法論（文獻支持）：

1. 第一階段：單變量檢定篩選（已在 02 完成）
   - 二元特徵：Chi-square test (p < 0.05)
   - 連續特徵：t-test (p < 0.05)

2. 第二階段：Bootstrap + LASSO (Stability Selection)
   - 文獻：Meinshausen & Bühlmann (2010), JRSS-B
   - 重複 bootstrap 抽樣 + LASSO
   - 選擇穩定被選中的特徵（選中頻率 > 閾值）

輸出：
- selected_features.json：最終選擇的特徵清單
- bootstrap_selection_frequency.csv：各特徵的選擇頻率
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import json
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("【步驟 3】特徵選擇（Bootstrap + LASSO）")
print("=" * 70)

# =====================================================================
# 讀取資料
# =====================================================================
print("\n讀取資料...")
features = pd.read_csv('all_features.csv')

with open('significant_features.json', 'r') as f:
    sig_features = json.load(f)

print(f"總樣本數: {len(features)}")
print(f"PPH+: {features['pph'].sum()} ({features['pph'].mean()*100:.1f}%)")
print(f"PPH-: {len(features) - features['pph'].sum()} ({(1-features['pph'].mean())*100:.1f}%)")

# =====================================================================
# 第一階段結果：單變量顯著特徵
# =====================================================================
print("\n" + "=" * 70)
print("【第一階段】單變量檢定結果")
print("=" * 70)

print(f"\n顯著二元特徵 ({len(sig_features['binary_features'])} 個):")
for feat in sig_features['binary_features']:
    print(f"  - {feat}")

print(f"\n顯著連續特徵 ({len(sig_features['continuous_features'])} 個):")
for feat in sig_features['continuous_features'][:10]:
    print(f"  - {feat}")
if len(sig_features['continuous_features']) > 10:
    print(f"  ... 及其他 {len(sig_features['continuous_features']) - 10} 個")

# =====================================================================
# 準備資料
# =====================================================================
candidate_features = sig_features['all_significant']
print(f"\n候選特徵數: {len(candidate_features)}")

X = features[candidate_features].copy()
y = features['pph'].values

# 處理缺失值
print("\n處理缺失值...")
missing_before = X.isnull().sum().sum()
for col in X.columns:
    if X[col].isnull().any():
        X[col].fillna(X[col].median(), inplace=True)
print(f"  填補前缺失值: {missing_before}")
print(f"  填補後缺失值: {X.isnull().sum().sum()}")

# =====================================================================
# Bootstrap + LASSO (Stability Selection)
# =====================================================================
print("\n" + "=" * 70)
print("【第二階段】Bootstrap + LASSO (Stability Selection)")
print("=" * 70)

# 參數設定
N_BOOTSTRAP = 100  # Bootstrap 次數
SELECTION_THRESHOLD = 0.6  # 選擇頻率閾值（Meinshausen & Bühlmann 2010: 0.6 對應 E(V)≤2.5）
SAMPLE_FRACTION = 0.8  # 每次抽樣比例

print(f"\n參數設定:")
print(f"  Bootstrap 次數: {N_BOOTSTRAP}")
print(f"  選擇頻率閾值: {SELECTION_THRESHOLD * 100}%")
print(f"  每次抽樣比例: {SAMPLE_FRACTION * 100}%")

# 記錄每個特徵被選中的次數
selection_counts = {feat: 0 for feat in candidate_features}

# 定義 LASSO 正則化參數（使用中等強度）
C_values = [0.01, 0.1, 0.5, 1.0]

print(f"\n執行 Bootstrap + LASSO...")
print(f"  測試的 C 值: {C_values}")

for i in range(N_BOOTSTRAP):
    if (i + 1) % 20 == 0:
        print(f"  進度: {i + 1}/{N_BOOTSTRAP}")

    # Bootstrap 抽樣（分層抽樣保持類別比例）
    n_samples = int(len(X) * SAMPLE_FRACTION)

    # 分層 bootstrap
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]

    n_pos = int(len(idx_pos) * SAMPLE_FRACTION)
    n_neg = int(len(idx_neg) * SAMPLE_FRACTION)

    boot_idx_pos = resample(idx_pos, n_samples=n_pos, random_state=i)
    boot_idx_neg = resample(idx_neg, n_samples=n_neg, random_state=i)
    boot_idx = np.concatenate([boot_idx_pos, boot_idx_neg])

    X_boot = X.iloc[boot_idx]
    y_boot = y[boot_idx]

    # 標準化
    scaler = StandardScaler()
    X_boot_scaled = scaler.fit_transform(X_boot)

    # 對每個 C 值執行 LASSO
    for C in C_values:
        lasso = LogisticRegression(
            penalty='l1',
            solver='saga',
            C=C,
            max_iter=2000,
            random_state=i,
            class_weight='balanced'
        )

        try:
            lasso.fit(X_boot_scaled, y_boot)

            # 記錄非零係數的特徵
            selected_idx = np.where(lasso.coef_[0] != 0)[0]
            for idx in selected_idx:
                selection_counts[candidate_features[idx]] += 1
        except:
            pass

# 計算選擇頻率
total_iterations = N_BOOTSTRAP * len(C_values)
selection_freq = {feat: count / total_iterations for feat, count in selection_counts.items()}

# 按頻率排序
freq_df = pd.DataFrame({
    'feature': list(selection_freq.keys()),
    'selection_frequency': list(selection_freq.values()),
    'selection_count': list(selection_counts.values())
}).sort_values('selection_frequency', ascending=False)

print("\n" + "=" * 70)
print("【特徵選擇頻率】")
print("=" * 70)

print(f"\n特徵選擇頻率（Top 20）:")
print("-" * 60)
print(f"{'特徵':<45} {'頻率':<10} {'次數':<10}")
print("-" * 60)

for _, row in freq_df.head(20).iterrows():
    bar = "█" * int(row['selection_frequency'] * 20)
    status = "✓" if row['selection_frequency'] >= SELECTION_THRESHOLD else ""
    print(f"{row['feature']:<45} {row['selection_frequency']:.1%}  {bar} {status}")

# 選擇穩定特徵
stable_features = freq_df[freq_df['selection_frequency'] >= SELECTION_THRESHOLD]['feature'].tolist()

print(f"\n穩定選擇的特徵（頻率 ≥ {SELECTION_THRESHOLD*100}%）: {len(stable_features)} 個")

# =====================================================================
# 最終特徵選擇
# =====================================================================
print("\n" + "=" * 70)
print("【最終特徵選擇】")
print("=" * 70)

final_binary = [f for f in stable_features if f in sig_features['binary_features']]
final_continuous = [f for f in stable_features if f in sig_features['continuous_features']]

print(f"\n最終選擇的二元特徵 ({len(final_binary)} 個):")
for feat in final_binary:
    freq = selection_freq[feat]
    print(f"  - {feat} (選擇頻率: {freq:.1%})")

print(f"\n最終選擇的連續特徵 ({len(final_continuous)} 個):")
for feat in final_continuous:
    freq = selection_freq[feat]
    print(f"  - {feat} (選擇頻率: {freq:.1%})")

# 如果沒有特徵通過閾值，降低閾值
if len(stable_features) == 0:
    print("\n警告: 沒有特徵通過 50% 閾值，降低至 30%")
    SELECTION_THRESHOLD = 0.3
    stable_features = freq_df[freq_df['selection_frequency'] >= SELECTION_THRESHOLD]['feature'].tolist()
    final_binary = [f for f in stable_features if f in sig_features['binary_features']]
    final_continuous = [f for f in stable_features if f in sig_features['continuous_features']]
    print(f"降低閾值後選擇的特徵: {len(stable_features)} 個")

# =====================================================================
# 儲存結果
# =====================================================================
print("\n" + "=" * 70)
print("【儲存結果】")
print("=" * 70)

selected_features = {
    'binary_features': final_binary,
    'continuous_features': final_continuous,
    'all_selected': stable_features,
    'selection_threshold': SELECTION_THRESHOLD,
    'n_bootstrap': N_BOOTSTRAP,
    'feature_count': len(stable_features)
}

with open('selected_features.json', 'w') as f:
    json.dump(selected_features, f, indent=2, ensure_ascii=False)
print(f"已儲存: selected_features.json")

freq_df.to_csv('bootstrap_selection_frequency.csv', index=False)
print(f"已儲存: bootstrap_selection_frequency.csv")

# =====================================================================
# 摘要
# =====================================================================
print("\n" + "=" * 70)
print("【摘要】")
print("=" * 70)

print(f"""
特徵選擇流程:
  第一階段（單變量檢定）: {len(candidate_features)} 個顯著特徵
  第二階段（Bootstrap + LASSO）: {len(stable_features)} 個穩定特徵

Bootstrap 參數:
  - 抽樣次數: {N_BOOTSTRAP}
  - 選擇閾值: {SELECTION_THRESHOLD*100}%
  - 總迭代次數: {total_iterations}

最終選擇:
  - 二元特徵: {len(final_binary)} 個
  - 連續特徵: {len(final_continuous)} 個
  - 總計: {len(stable_features)} 個

文獻支持:
  - Meinshausen & Bühlmann (2010), JRSS-B: Stability Selection
  - 此方法比單次 LASSO 更穩健，減少假陽性特徵
""")

print("完成！")
