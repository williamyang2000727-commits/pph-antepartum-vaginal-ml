#!/usr/bin/env python3
"""
03_feature_selection.py - Bootstrap LASSO stability selection.

WARNING:
  This script uses SELECTION_THRESHOLD = 0.6 (selects features with
  >=60% selection frequency), but the FINAL MODEL does NOT use that
  approach. The final model takes the top 30 features by selection
  frequency rank (see 04_train_RF_SMOTEENN_Top30.py around line 211:
  freq_df.head(30)).

Methodology:
  Stage 1 - univariate filtering (in 02_feature_engineering.py):
    - binary features: Chi-square test, p < 0.05
    - continuous features: t-test, p < 0.05

  Stage 2 - Bootstrap LASSO stability selection (this script):
    - Reference: Meinshausen & Bühlmann (2010), J R Stat Soc B
    - Repeated bootstrap resampling + L1-penalized logistic regression
    - Select features by selection frequency

Output:
  - selected_features.json: final selected feature list
  - bootstrap_selection_frequency.csv: per-feature selection frequency
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
print("【step 3】feature selection(Bootstrap + LASSO)")
print("=" * 70)

# =====================================================================
# Loading data
# =====================================================================
print("\nLoading data...")
features = pd.read_csv('all_features.csv')

with open('significant_features.json', 'r') as f:
    sig_features = json.load(f)

print(f"Totalsamples數: {len(features)}")
print(f"PPH+: {features['pph'].sum()} ({features['pph'].mean()*100:.1f}%)")
print(f"PPH-: {len(features) - features['pph'].sum()} ({(1-features['pph'].mean())*100:.1f}%)")

# =====================================================================
# Stage 1result: 單change量顯著features
# =====================================================================
print("\n" + "=" * 70)
print("【Stage 1】單change量檢定result")
print("=" * 70)

print(f"\n顯著binaryfeatures ({len(sig_features['binary_features'])} ):")
for feat in sig_features['binary_features']:
    print(f"  - {feat}")

print(f"\n顯著continuousfeatures ({len(sig_features['continuous_features'])} ):")
for feat in sig_features['continuous_features'][:10]:
    print(f"  - {feat}")
if len(sig_features['continuous_features']) > 10:
    print(f"  ... andits他 {len(sig_features['continuous_features']) - 10} ")

# =====================================================================
# 準備data
# =====================================================================
candidate_features = sig_features['all_significant']
print(f"\ncandidatefeatures數: {len(candidate_features)}")

X = features[candidate_features].copy()
y = features['pph'].values

# process缺失值
print("\nprocess缺失值...")
missing_before = X.isnull().sum().sum()
for col in X.columns:
    if X[col].isnull().any():
        X[col].fillna(X[col].median(), inplace=True)
print(f"  填補before缺失值: {missing_before}")
print(f"  填補after缺失值: {X.isnull().sum().sum()}")

# =====================================================================
# Bootstrap + LASSO (Stability Selection)
# =====================================================================
print("\n" + "=" * 70)
print("【Stage 2】Bootstrap + LASSO (Stability Selection)")
print("=" * 70)

# parameters設定
N_BOOTSTRAP = 100  # Bootstrap times數
SELECTION_THRESHOLD = 0.6  # selection frequencythreshold(Meinshausen & Bühlmann 2010: 0.6 對應 E(V)<=2.5)
SAMPLE_FRACTION = 0.8  # eachtimesresampling比例

print(f"\nparameters設定:")
print(f"  Bootstrap times數: {N_BOOTSTRAP}")
print(f"  selection frequencythreshold: {SELECTION_THRESHOLD * 100}%")
print(f"  eachtimesresampling比例: {SAMPLE_FRACTION * 100}%")

# 記錄eachfeaturesisselected的times數
selection_counts = {feat: 0 for feat in candidate_features}

# 定義 LASSO 正則化parameters(Usemiddle等強度)
C_values = [0.01, 0.1, 0.5, 1.0]

print(f"\n執line Bootstrap + LASSO...")
print(f"  測試的 C 值: {C_values}")

for i in range(N_BOOTSTRAP):
    if (i + 1) % 20 == 0:
        print(f"  Progress: {i + 1}/{N_BOOTSTRAP}")

    # Bootstrap resampling(split層resampling保持Class ratio)
    n_samples = int(len(X) * SAMPLE_FRACTION)

    # split層 bootstrap
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]

    n_pos = int(len(idx_pos) * SAMPLE_FRACTION)
    n_neg = int(len(idx_neg) * SAMPLE_FRACTION)

    boot_idx_pos = resample(idx_pos, n_samples=n_pos, random_state=i)
    boot_idx_neg = resample(idx_neg, n_samples=n_neg, random_state=i)
    boot_idx = np.concatenate([boot_idx_pos, boot_idx_neg])

    X_boot = X.iloc[boot_idx]
    y_boot = y[boot_idx]

    # mark準化
    scaler = StandardScaler()
    X_boot_scaled = scaler.fit_transform(X_boot)

    # 對each C 值執line LASSO
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

            # 記錄非零係數的features
            selected_idx = np.where(lasso.coef_[0] != 0)[0]
            for idx in selected_idx:
                selection_counts[candidate_features[idx]] += 1
        except:
            pass

# Computeselection frequency
total_iterations = N_BOOTSTRAP * len(C_values)
selection_freq = {feat: count / total_iterations for feat, count in selection_counts.items()}

# byfrequencysorting
freq_df = pd.DataFrame({
    'feature': list(selection_freq.keys()),
    'selection_frequency': list(selection_freq.values()),
    'selection_count': list(selection_counts.values())
}).sort_values('selection_frequency', ascending=False)

print("\n" + "=" * 70)
print("【feature selectionfrequency】")
print("=" * 70)

print(f"\nfeature selectionfrequency(Top 20):")
print("-" * 60)
print(f"{'features':<45} {'frequency':<10} {'times數':<10}")
print("-" * 60)

for _, row in freq_df.head(20).iterrows():
    bar = "█" * int(row['selection_frequency'] * 20)
    status = "✓" if row['selection_frequency'] >= SELECTION_THRESHOLD else ""
    print(f"{row['feature']:<45} {row['selection_frequency']:.1%}  {bar} {status}")

# select穩定features
stable_features = freq_df[freq_df['selection_frequency'] >= SELECTION_THRESHOLD]['feature'].tolist()

print(f"\n穩定select的features(frequency >= {SELECTION_THRESHOLD*100}%): {len(stable_features)} ")

# =====================================================================
# finalfeature selection
# =====================================================================
print("\n" + "=" * 70)
print("【finalfeature selection】")
print("=" * 70)

final_binary = [f for f in stable_features if f in sig_features['binary_features']]
final_continuous = [f for f in stable_features if f in sig_features['continuous_features']]

print(f"\nfinally selected的binaryfeatures ({len(final_binary)} ):")
for feat in final_binary:
    freq = selection_freq[feat]
    print(f"  - {feat} (selection frequency: {freq:.1%})")

print(f"\nfinally selected的continuousfeatures ({len(final_continuous)} ):")
for feat in final_continuous:
    freq = selection_freq[feat]
    print(f"  - {feat} (selection frequency: {freq:.1%})")

# e.g.果missingfeatures通過threshold, 降lowthreshold
if len(stable_features) == 0:
    print("\nWARNING: missingfeatures通過 50% threshold, 降low至 30%")
    SELECTION_THRESHOLD = 0.3
    stable_features = freq_df[freq_df['selection_frequency'] >= SELECTION_THRESHOLD]['feature'].tolist()
    final_binary = [f for f in stable_features if f in sig_features['binary_features']]
    final_continuous = [f for f in stable_features if f in sig_features['continuous_features']]
    print(f"降lowthresholdafterselect的features: {len(stable_features)} ")

# =====================================================================
# Saveresult
# =====================================================================
print("\n" + "=" * 70)
print("【Saveresult】")
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
print(f"Save: selected_features.json")

freq_df.to_csv('bootstrap_selection_frequency.csv', index=False)
print(f"Save: bootstrap_selection_frequency.csv")

# =====================================================================
# 摘to
# =====================================================================
print("\n" + "=" * 70)
print("【摘to】")
print("=" * 70)

print(f"""
feature selectionpipeline:
  Stage 1(單change量檢定): {len(candidate_features)} 顯著features
  Stage 2(Bootstrap + LASSO): {len(stable_features)} 穩定features

Bootstrap parameters:
  - resamplingtimes數: {N_BOOTSTRAP}
  - selectthreshold: {SELECTION_THRESHOLD*100}%
  - Total迭代times數: {total_iterations}

finally selected:
  - binaryfeatures: {len(final_binary)} 
  - continuousfeatures: {len(final_continuous)} 
  - Totalcount: {len(stable_features)} 

Literature support:
  - Meinshausen & Bühlmann (2010), JRSS-B: Stability Selection
  - thismethod比單times LASSO more穩健, 減less假positivefeatures
""")

print("Done!")
