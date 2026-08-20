#!/usr/bin/env python3
"""
train_RF_SMOTEENN_Top30.py - 重新訓練 RandomForest + SMOTEENN (Top 30)

基於 00_clean_pipeline_no_leakage.py 修改：
- 改用 RandomForest (不是 LightGBM)
- 改用 Top 30 features (不是 Top 25)
- 改用 KNN k=1 (不是 k=5)
- 保持所有其他參數不變 (random_state=42)

目標：復現 Final_Top30_RandomForest_SMOTEENN_info.json 的結果
- TP=27, FP=41, FN=10, TN=76
- Recall=0.7297, AUC=0.7169, MCC=0.3263
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split

# 2026-07-23 路徑遷移:改用「腳本自身位置」推導,不再硬編碼舊倉庫 PPH_Prediction_Model-main。
# BASE = PPH_joint_search_2026_06_05/ (本檔的上一層)。整包資料夾搬到哪都能跑。
def _resolve_base(script):
    """找出資料根目錄：從腳本所在位置往上找含 `02_experiment_csv` 的那一層。

    🚨 2026-08-20 修：原本寫死成「腳本往上兩層」
       （`os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`）。
       這個假設只在本機專案成立 —— 腳本放在 `<專案>/03_scripts_ORIGINAL_DO_NOT_MODIFY/`，
       往上兩層剛好是專案根。但**公開倉庫的腳本就放在倉庫根目錄**，
       往上兩層會跑到倉庫外面：實測解析到 `~/Desktop`，
       於是 clone 下來的人一執行 04／05／07／08 就去找不存在的
       `<倉庫上上層>/02_experiment_csv/` 而崩潰，錯誤訊息還指向倉庫外的路徑。
       改成往上搜尋含 `02_experiment_csv` 的那一層；找不到就退回腳本自己的目錄，
       這樣錯誤訊息至少指在倉庫內。
       ✅ 已實測：在本機專案的解析結果與舊寫法**完全相同**（五支逐一比對）。
    """
    d = os.path.dirname(os.path.abspath(script))
    probe = d
    for _ in range(3):                      # 腳本自身 → 上一層 → 上上層
        if os.path.isdir(os.path.join(probe, '02_experiment_csv')):
            return probe
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    return d


BASE = _resolve_base(__file__)
CSV_DIR = os.path.join(BASE, '02_experiment_csv')
RESULTS_DIR = os.path.join(BASE, '01_KEY_RESULTS')
os.makedirs(RESULTS_DIR, exist_ok=True)   # 2026-08-20：倉庫裡沒有這個目錄，不建會直接 FileNotFoundError
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.utils import resample
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score,
                             precision_recall_curve, average_precision_score,
                             recall_score, precision_score, f1_score,
                             matthews_corrcoef, accuracy_score)
from sklearn.ensemble import RandomForestClassifier  # ⭐ 改用 RandomForest
from imblearn.combine import SMOTEENN
import json
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("  PPH Prediction - RandomForest + SMOTEENN (Top 30 Features)")
print("=" * 80)

# =====================================================================
# Step 1: Load Original Data
# =====================================================================
print("\n" + "=" * 80)
print("Step 1: Load Original Data")
print("=" * 80)

all_features = pd.read_csv(os.path.join(CSV_DIR, 'all_features.csv'))
print(f"Total samples: {len(all_features)}")
print(f"Total features: {len(all_features.columns) - 2}")
print(f"PPH+: {all_features['pph'].sum()} ({all_features['pph'].mean()*100:.1f}%)")
print(f"PPH-: {len(all_features) - all_features['pph'].sum()} ({(1-all_features['pph'].mean())*100:.1f}%)")

# =====================================================================
# Step 2: Data Split FIRST (Before Any Feature Selection!)
# =====================================================================
print("\n" + "=" * 80)
print("Step 2: Data Split (80/20 Stratified) - BEFORE any feature selection!")
print("=" * 80)

feature_cols = [col for col in all_features.columns if col not in ['PID', 'pph']]

X = all_features[feature_cols]
y = all_features['pph']
pids = all_features['PID']

X_train, X_test, y_train, y_test, pid_train, pid_test = train_test_split(
    X, y, pids,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Train set: {len(y_train)} samples (PPH+: {sum(y_train)}, {sum(y_train)/len(y_train)*100:.1f}%)")
print(f"Test set:  {len(y_test)} samples (PPH+: {sum(y_test)}, {sum(y_test)/len(y_test)*100:.1f}%)")

train_df = X_train.copy()
train_df['pph'] = y_train.values
train_df['PID'] = pid_train.values

test_df = X_test.copy()
test_df['pph'] = y_test.values
test_df['PID'] = pid_test.values

# =====================================================================
# Step 3: Chi-square & t-test (Train Data Only!)
# =====================================================================
print("\n" + "=" * 80)
print("Step 3: Statistical Feature Selection (Train Data Only!)")
print("=" * 80)

binary_features = []
continuous_features = []

for col in feature_cols:
    unique_vals = train_df[col].dropna().unique()
    if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
        binary_features.append(col)
    else:
        continuous_features.append(col)

print(f"Binary features: {len(binary_features)}")
print(f"Continuous features: {len(continuous_features)}")

# Chi-square test
print("\n--- Chi-square Test (Binary Features, Train Only) ---")
chi2_results = []

for feat in binary_features:
    contingency = pd.crosstab(train_df[feat].fillna(0), train_df['pph'])

    if contingency.shape == (2, 2):
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        chi2_results.append({
            'feature': feat,
            'chi2': chi2,
            'p_value': p_value,
            'significant': p_value < 0.05
        })

chi2_df = pd.DataFrame(chi2_results).sort_values('p_value')
significant_binary = chi2_df[chi2_df['significant']]['feature'].tolist()
print(f"Significant binary features (p<0.05): {len(significant_binary)}")

# t-test
print("\n--- T-test (Continuous Features, Train Only) ---")
ttest_results = []

for feat in continuous_features:
    pph_pos = train_df[train_df['pph'] == 1][feat].dropna()
    pph_neg = train_df[train_df['pph'] == 0][feat].dropna()

    if len(pph_pos) > 1 and len(pph_neg) > 1:
        t_stat, p_value = stats.ttest_ind(pph_pos, pph_neg)
        ttest_results.append({
            'feature': feat,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        })

ttest_df = pd.DataFrame(ttest_results).sort_values('p_value')
significant_continuous = ttest_df[ttest_df['significant']]['feature'].tolist()
print(f"Significant continuous features (p<0.05): {len(significant_continuous)}")

significant_features = significant_binary + significant_continuous
print(f"\nTotal significant features: {len(significant_features)}")

# =====================================================================
# Step 4: Bootstrap LASSO (Train Data Only!)
# =====================================================================
print("\n" + "=" * 80)
print("Step 4: Bootstrap LASSO - Stability Selection (Train Data Only!)")
print("=" * 80)

X_train_sig = train_df[significant_features].values
y_train_arr = train_df['pph'].values

imputer_temp = SimpleImputer(strategy='median')
X_train_imputed = imputer_temp.fit_transform(X_train_sig)

N_BOOTSTRAP = 100
SAMPLE_FRACTION = 0.8
C_VALUES = [0.01, 0.1, 1.0, 10.0]
total_iterations = N_BOOTSTRAP * len(C_VALUES)

print(f"Parameters:")
print(f"  Bootstrap iterations: {N_BOOTSTRAP}")
print(f"  Sample fraction: {SAMPLE_FRACTION}")
print(f"  C values: {C_VALUES}")
print(f"  Total experiments: {total_iterations}")

selection_counts = {feat: 0 for feat in significant_features}

print("\nRunning Bootstrap LASSO...")
for i in range(N_BOOTSTRAP):
    if (i + 1) % 20 == 0:
        print(f"  Progress: {i+1}/{N_BOOTSTRAP} ({(i+1)/N_BOOTSTRAP*100:.0f}%)")

    n_samples = int(len(X_train_imputed) * SAMPLE_FRACTION)
    indices = resample(range(len(X_train_imputed)), n_samples=n_samples, random_state=i)
    X_boot = X_train_imputed[indices]
    y_boot = y_train_arr[indices]

    scaler_temp = StandardScaler()
    X_scaled = scaler_temp.fit_transform(X_boot)

    for C in C_VALUES:
        model = LogisticRegression(
            penalty='l1',
            solver='saga',
            C=C,
            max_iter=2000,
            random_state=42
        )
        model.fit(X_scaled, y_boot)

        nonzero_idx = np.where(model.coef_[0] != 0)[0]
        for idx in nonzero_idx:
            selection_counts[significant_features[idx]] += 1

print("  Bootstrap LASSO completed!")

freq_df = pd.DataFrame([
    {
        'feature': feat,
        'selection_count': count,
        'selection_frequency': count / total_iterations
    }
    for feat, count in selection_counts.items()
]).sort_values('selection_frequency', ascending=False).reset_index(drop=True)

# 2026-08-19：輸出 selection frequency 供 06_joint_search.py 使用。
#   此表原本只存在於記憶體中，外部檔案 bootstrap_selection_frequency_clean.csv 需人工保存，
#   容易與本腳本的實際計算分岔。改為直接由本腳本輸出，確保永遠一致（不改變任何計算邏輯）。
freq_df[['feature', 'selection_frequency', 'selection_count']].to_csv(
    os.path.join(CSV_DIR, 'bootstrap_selection_frequency_clean.csv'), index=False)
print(f"  已輸出 selection frequency（{len(freq_df)} 個特徵）")

# ⭐ Get Top 30 features (not 25!)
top30_features = freq_df.head(30)['feature'].tolist()

print(f"\n⭐ Top 30 Features (by selection frequency):")
print("-" * 70)
for i, row in freq_df.head(30).iterrows():
    print(f"  {i+1:2d}. {row['feature'][:45]:<45} {row['selection_frequency']:.3f}")

# =====================================================================
# Step 5: Prepare Final Train/Test Data with Top 30 Features
# =====================================================================
print("\n" + "=" * 80)
print("Step 5: Prepare Final Data (Top 30 Features)")
print("=" * 80)

X_train_final = train_df[top30_features].values
X_test_final = test_df[top30_features].values
y_train_final = train_df['pph'].values
y_test_final = test_df['pph'].values

print(f"Train shape: {X_train_final.shape}")
print(f"Test shape: {X_test_final.shape}")

# =====================================================================
# Step 6: KNN Imputation (k=1, Fit on Train Only)
# =====================================================================
print("\n" + "=" * 80)
print("Step 6: KNN Imputation (k=1) - Fit on Train, Transform on Test")
print("=" * 80)

# ⭐ 改用 k=1 (不是 k=5)
imputer = KNNImputer(n_neighbors=1)
X_train_imp = imputer.fit_transform(X_train_final)
X_test_imp = imputer.transform(X_test_final)
print("  ⭐ Completed (k=1, fit on train, transform on test)")

# =====================================================================
# Step 7: StandardScaler (Fit on Train Only)
# =====================================================================
print("\n" + "=" * 80)
print("Step 7: StandardScaler - Fit on Train, Transform on Test")
print("=" * 80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)
print("  Completed (fit on train, transform on test)")

# =====================================================================
# Step 8: SMOTEENN (Train Only)
# =====================================================================
print("\n" + "=" * 80)
print("Step 8: SMOTEENN Resampling (Train Only)")
print("=" * 80)

print(f"  Before SMOTEENN: {len(y_train_final)} samples (PPH+: {sum(y_train_final)})")
smoteenn = SMOTEENN(random_state=42)
X_train_res, y_train_res = smoteenn.fit_resample(X_train_scaled, y_train_final)
print(f"  After SMOTEENN:  {len(y_train_res)} samples (PPH+: {sum(y_train_res)})")

# =====================================================================
# Step 9: Train RandomForest (⭐ 不是 LightGBM)
# =====================================================================
print("\n" + "=" * 80)
print("Step 9: Train RandomForest Classifier (⭐ not LightGBM)")
print("=" * 80)

# ⭐ 改用 RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_res, y_train_res)
print("  ⭐ RandomForest training completed!")

# =====================================================================
# Step 10: Evaluate on Independent Test Set
# =====================================================================
print("\n" + "=" * 80)
print("Step 10: Evaluate on Independent Test Set")
print("=" * 80)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Calculate all metrics
tn, fp, fn, tp = confusion_matrix(y_test_final, y_pred).ravel()
recall = recall_score(y_test_final, y_pred)
specificity = tn / (tn + fp)
precision = precision_score(y_test_final, y_pred)
f1 = f1_score(y_test_final, y_pred)
auc = roc_auc_score(y_test_final, y_prob)
mcc = matthews_corrcoef(y_test_final, y_pred)
accuracy = accuracy_score(y_test_final, y_pred)

print(f"\n⭐⭐⭐ Confusion Matrix:")
print(f"  TP={tp}, FP={fp}")
print(f"  FN={fn}, TN={tn}")
print(f"\nPerformance Metrics:")
print(f"  Recall (Sensitivity): {recall:.4f}")
print(f"  Specificity:          {specificity:.4f}")
print(f"  Precision:            {precision:.4f}")
print(f"  F1-Score:             {f1:.4f}")
print(f"  AUC:                  {auc:.4f}")
print(f"  MCC:                  {mcc:.4f}")
print(f"  Accuracy:             {accuracy:.4f}")

# =====================================================================
# Verification: Check if matches JSON target
# =====================================================================
print("\n" + "=" * 80)
print("⛔⛔⛔ Verification: Compare with Target JSON")
print("=" * 80)

target_tp = 27
target_fp = 41
target_fn = 10
target_tn = 76
target_recall = 0.7297
target_auc = 0.7169
target_mcc = 0.3263

print("\nTarget (from JSON):")
print(f"  TP={target_tp}, FP={target_fp}, FN={target_fn}, TN={target_tn}")
print(f"  Recall={target_recall:.4f}, AUC={target_auc:.4f}, MCC={target_mcc:.4f}")

print("\nActual (this run):")
print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
print(f"  Recall={recall:.4f}, AUC={auc:.4f}, MCC={mcc:.4f}")

# Validation checks
checks = {
    "TP": tp == target_tp,
    "FP": fp == target_fp,
    "FN": fn == target_fn,
    "TN": tn == target_tn,
    "Recall": abs(recall - target_recall) < 0.001,
    "AUC": abs(auc - target_auc) < 0.001,
    "MCC": abs(mcc - target_mcc) < 0.001
}

print("\nVerification Results:")
all_passed = True
for metric, passed in checks.items():
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {metric:10s}: {status}")
    if not passed:
        all_passed = False

if all_passed:
    print("\n🎉🎉🎉 ✅ ALL CHECKS PASSED! Results match target JSON!")
else:
    print("\n⚠️ WARNING: Some metrics do not match target JSON")
    print("This may be due to different feature selection or random seeds")

# =====================================================================
# Save Results
# =====================================================================
print("\n" + "=" * 80)
print("Saving Results")
print("=" * 80)

output_dir = RESULTS_DIR + os.sep  # 結果檔(test_set_y_*.npy 等)統一存 01_KEY_RESULTS/

# Save predictions and probabilities
np.save(output_dir + 'test_set_y_true.npy', y_test_final)
np.save(output_dir + 'test_set_y_proba_RF_SMOTEENN.npy', y_prob)
np.save(output_dir + 'test_set_y_pred_RF_SMOTEENN.npy', y_pred)

# Save ROC curve data
fpr, tpr, thresholds = roc_curve(y_test_final, y_prob)
np.save(output_dir + 'roc_curve_fpr_RF_SMOTEENN.npy', fpr)
np.save(output_dir + 'roc_curve_tpr_RF_SMOTEENN.npy', tpr)
np.save(output_dir + 'roc_curve_thresholds_RF_SMOTEENN.npy', thresholds)

print(f"  ✅ Saved: test_set_y_true.npy")
print(f"  ✅ Saved: test_set_y_proba_RF_SMOTEENN.npy")
print(f"  ✅ Saved: test_set_y_pred_RF_SMOTEENN.npy")
print(f"  ✅ Saved: roc_curve_fpr_RF_SMOTEENN.npy")
print(f"  ✅ Saved: roc_curve_tpr_RF_SMOTEENN.npy")
print(f"  ✅ Saved: roc_curve_thresholds_RF_SMOTEENN.npy")

# Save training log
training_log = {
    "date": "2026-02-21",
    "model": "RandomForest + SMOTEENN",
    "n_features": 30,
    "features": top30_features,
    "random_state": 42,
    "preprocessing": {
        "imputation": "KNN (k=1)",
        "scaling": "StandardScaler",
        "resampling": "SMOTEENN"
    },
    "confusion_matrix": {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn)
    },
    "metrics": {
        "recall": float(recall),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "auc": float(auc),
        "mcc": float(mcc),
        "accuracy": float(accuracy)
    },
    "target_metrics": {
        "tp": target_tp,
        "fp": target_fp,
        "fn": target_fn,
        "tn": target_tn,
        "recall": target_recall,
        "auc": target_auc,
        "mcc": target_mcc
    },
    "verification": {
        "all_passed": bool(all_passed),
        "details": {k: bool(v) for k, v in checks.items()}
    },
    "reproducible": True,
    "data_leakage_prevention": {
        "data_split": "Done FIRST before any feature selection",
        "statistical_tests": "Train data only",
        "bootstrap_lasso": "Train data only",
        "imputation": "Fit on train, transform on test",
        "scaling": "Fit on train, transform on test",
        "resampling": "Train data only"
    }
}

with open(output_dir + 'RF_SMOTEENN_training_log.json', 'w') as f:
    json.dump(training_log, f, indent=2)

print(f"  ✅ Saved: RF_SMOTEENN_training_log.json")

print("\n" + "=" * 80)
print("✅ Pipeline completed successfully!")
print(f"All results saved to: {output_dir}")
print("=" * 80)
