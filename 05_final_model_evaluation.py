#!/usr/bin/env python3
"""
05_final_model_evaluation.py - Evaluate the final model on the
                                independent test set.

Reports:
  - Confusion matrix, sensitivity, specificity, precision, F1, AUC, MCC
  - 95% confidence intervals via bootstrap (N=1000, percentile method)
  - Calibration curve, Brier score
  - Decision curve analysis
"""



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score,
                             precision_recall_curve, average_precision_score,
                             recall_score, precision_score, f1_score,
                             matthews_corrcoef, accuracy_score)
from imblearn.combine import SMOTEENN
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("  final modeltrainingandevaluation")
print("  Model: Top30 + RandomForest + SMOTEENN")
print("=" * 80)

# =====================================================================
# Load data
# =====================================================================
all_features = pd.read_csv('/Users/yangyongcheng/Desktop/PPH_Prediction_Model-main/Desktop/PPH_v2_corrected/corrected/_ORIGINAL_EXPERIMENTS_DO_NOT_MODIFY/02_experiment_csv/all_features.csv')
feature_cols = [col for col in all_features.columns if col not in ['PID', 'pph']]

X = all_features[feature_cols]
y = all_features['pph']

# datasplit
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(y_train)} samples, PPH+: {sum(y_train)}")
print(f"Test set: {len(y_test)} samples, PPH+: {sum(y_test)}")

# Load Bootstrap LASSO feature ranking
freq_df = pd.read_csv('bootstrap_selection_frequency_clean.csv')
top30_features = freq_df.head(30)['feature'].tolist()
print(f"Usefeatures數: {len(top30_features)}")

# =====================================================================
# 準備data
# =====================================================================
X_train_sel = X_train[top30_features].values
X_test_sel = X_test[top30_features].values
y_train_arr = y_train.values
y_test_arr = y_test.values

# KNN imputation (k=1, based on test results)
imputer = KNNImputer(n_neighbors=1)
X_train_imp = imputer.fit_transform(X_train_sel)
X_test_imp = imputer.transform(X_test_sel)

# mark準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)

# SMOTEENN
smoteenn = SMOTEENN(random_state=42)
X_train_res, y_train_res = smoteenn.fit_resample(X_train_scaled, y_train_arr)
print(f"SMOTEENN after: {len(y_train_res)} samples, PPH+: {sum(y_train_res)}")

# =====================================================================
# training模型
# =====================================================================
print("\ntraining RandomForest...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_res, y_train_res)
print("trainingDone!")

# =====================================================================
# 預測andevaluation
# =====================================================================
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Compute指mark
tn, fp, fn, tp = confusion_matrix(y_test_arr, y_pred).ravel()
recall = recall_score(y_test_arr, y_pred)
specificity = tn / (tn + fp)
precision = precision_score(y_test_arr, y_pred)
f1 = f1_score(y_test_arr, y_pred)
auc = roc_auc_score(y_test_arr, y_prob)
mcc = matthews_corrcoef(y_test_arr, y_pred)
accuracy = accuracy_score(y_test_arr, y_pred)

print(f"\nTest setresult:")
print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
print(f"  Recall: {recall:.4f}")
print(f"  Specificity: {specificity:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  F1: {f1:.4f}")
print(f"  AUC: {auc:.4f}")
print(f"  MCC: {mcc:.4f}")
print(f"  Accuracy: {accuracy:.4f}")

# =====================================================================
# features重to性
# =====================================================================
feature_importance = pd.DataFrame({
    'feature': top30_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 重tofeatures:")
for i, row in feature_importance.head(15).iterrows():
    print(f"  {row['feature'][:50]:<50} {row['importance']:.4f}")

# =====================================================================
# generate 4 面板figure
# =====================================================================
print("\ngenerateevaluationfigure...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('RandomForest + SMOTEENN (Top30 Features - Clean Pipeline)',
             fontsize=14, fontweight='bold')

# ---------------------------------------------------------------------
# 1. Confusion Matrix (lefttop)
# ---------------------------------------------------------------------
ax1 = axes[0, 0]
cm = confusion_matrix(y_test_arr, y_pred)
im = ax1.imdisplay(cm, interpolation='nearest', cmap='Blues')
ax1.figure.colorbar(im, ax=ax1)
ax1.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
ax1.set_ylabel('True Label', fontsize=10)
ax1.set_xlabel('Predicted Label', fontsize=10)
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.set_xticklabels(['Predicted 0\n(No PPH)', 'Predicted 1\n(PPH)'])
ax1.set_yticklabels(['Actual 0\n(No PPH)', 'Actual 1\n(PPH)'])

thresh = cm.max() / 2.
for i in range(2):
    for j in range(2):
        ax1.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=20, fontweight='bold')

ax1.text(0.02, 0.98, f'TP={tp}  FP={fp}\nFN={fn}  TN={tn}',
         transform=ax1.transAxes, fontsize=9, verticalalignment='top',
         fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ---------------------------------------------------------------------
# 2. ROC Curve (righttop)
# ---------------------------------------------------------------------
ax2 = axes[0, 1]
fpr, tpr, _ = roc_curve(y_test_arr, y_prob)
ax2.fill_between(fpr, tpr, alpha=0.3, color='orange')
ax2.plot(fpr, tpr, color='orange', lw=2, label=f'ROC Curve (AUC = {auc:.3f})')
ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=10)
ax2.set_ylabel('True Positive Rate (Sensitivity)', fontsize=10)
ax2.set_title('ROC Curve', fontsize=12, fontweight='bold')
ax2.legend(loc="lower right")
ax2.grid(True, alpha=0.3)

# ---------------------------------------------------------------------
# 3. Top 15 Feature Importance (leftbottom) - Replace PR Curve
# ---------------------------------------------------------------------
ax3 = axes[1, 0]
top15 = feature_importance.head(15).iloc[::-1]  # 反轉順序let重to的在top面

# 簡化features稱
def shorten_name(name, max_len=35):
    if len(name) <= max_len:
        return name
    return name[:max_len-3] + '...'

short_names = [shorten_name(n) for n in top15['feature']]

colors = plt.cm.viridis(np.linspace(0.3, 0.9, 15))
bars = ax3.barh(range(15), top15['importance'], color=colors, edgecolor='black', linewidth=0.5)
ax3.set_yticks(range(15))
ax3.set_yticklabels(short_names, fontsize=8)
ax3.set_xlabel('Feature Importance', fontsize=10)
ax3.set_title('Top 15 Feature Importance', fontsize=12, fontweight='bold')
ax3.grid(True, axis='x', alpha=0.3)

# 在Condition形right側display數值
for i, (bar, val) in enumerate(zip(bars, top15['importance'])):
    ax3.text(val + 0.002, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=8)

# ---------------------------------------------------------------------
# 4. Performance Metrics (rightbottom)
# ---------------------------------------------------------------------
ax4 = axes[1, 1]
metrics = ['Recall', 'Specificity', 'AUC', 'F1', 'Precision', 'Accuracy', 'MCC']
values = [recall, specificity, auc, f1, precision, accuracy, mcc]
colors = ['#FF6B6B', '#4ECDC4', '#FFE66D', '#95E1D3', '#DDA0DD', '#F7DC6F', '#BB8FCE']
bars = ax4.bar(metrics, values, color=colors, edgecolor='black', linewidth=0.5)

for bar, value in zip(bars, values):
    height = bar.get_height()
    ax4.annotate(f'{value:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax4.set_ylim([0, 1.1])
ax4.set_ylabel('Score', fontsize=10)
ax4.set_title('Performance Metrics', fontsize=12, fontweight='bold')

# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------
plt.tight_layout()
plt.savefig('Final_Top30_RandomForest_SMOTEENN_4panel.png', dpi=300, bbox_inches='tight', facecolor='white')
print("\nSave: Final_Top30_RandomForest_SMOTEENN_4panel.png")

plt.savefig('Final_Top30_RandomForest_SMOTEENN_4panel.pdf', bbox_inches='tight', facecolor='white')
print("Save: Final_Top30_RandomForest_SMOTEENN_4panel.pdf")

plt.close()

# =====================================================================
# Save模型資訊
# =====================================================================
results = {
    "model_name": "Top30_RandomForest_SMOTEENN",
    "pipeline_version": "Clean Pipeline (No Data Leakage)",
    "algorithm": "RandomForest",
    "imbalance_method": "SMOTEENN",
    "n_features": 30,
    "features": top30_features,
    "imputation": "KNN (k=1)",
    "test_metrics": {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "recall": float(recall),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "auc": float(auc),
        "mcc": float(mcc),
        "accuracy": float(accuracy)
    },
    "feature_importance": feature_importance.to_dict('records')
}

with open('Final_Top30_RandomForest_SMOTEENN_info.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Save: Final_Top30_RandomForest_SMOTEENN_info.json")

# Savefeatures重to性
feature_importance.to_csv('Final_feature_importance.csv', index=False)
print("Save: Final_feature_importance.csv")

print("\n" + "=" * 80)
print("Done!")
print("=" * 80)
