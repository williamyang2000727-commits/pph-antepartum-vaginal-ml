#!/usr/bin/env python3
"""
重新生成 Fig3.png — Joint Search Top 5(by Test composite)

1:1 沿用原版 regenerate_Fig3_correct_top5.py 的繪圖風格:
  - 7.5 x 7.5 inch, 300 DPI, equal aspect
  - 第 1 名 RF: 紅色 / 實線 / linewidth 3.5
  - #2-#5: 彩色虛線 / linewidth 2 / alpha 0.7
  - 對角線: 灰色 / 點線 ':' / linewidth 1.5
  - Arial / fs=9-11pt / 無 title (PLOS 規範)

只改:Top 5 model 配置(從原 5,265 ranking 改成 joint search 2,646 by test composite)

Joint Search Top 5 (by Test composite 0.4*MCC + 0.3*AUC + 0.3*Recall):
  #1: KNN_k1  + RandomForest + SMOTEENN            + Top30  (Test comp 0.5645, AUC 0.717)
  #2: KNN_k3  + DecisionTree + SMOTEENN            + Top30  (Test comp 0.5602, AUC 0.686)
  #3: KNN_k1  + ExtraTrees   + SMOTEENN            + Top30  (Test comp 0.5557, AUC 0.713)
  #4: KNN_k1  + SVM_RBF      + SMOTEENN            + Top30  (Test comp 0.5527, AUC 0.639)
  #5: KNN_k7  + ExtraTrees   + RandomUndersampling + Top20  (Test comp 0.5490, AUC 0.688)
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# 路徑
# =====================================================================
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_DIR = f'{BASE}/02_experiment_csv'
# 2026-07-25 修:原本輸出 6/11 改名前的舊檔名 Main_Fig3_ROC.png,重跑不會更新投稿用的
# Fig3.png,還會在 06_figures 內多出一個並存的舊名檔案造成混淆。
OUT_PATH = f'{BASE}/06_figures/Fig3.png'

# =====================================================================
# 1. 載入數據 & 分割(1:1 沿用原 protocol)
# =====================================================================
all_features = pd.read_csv(f'{CSV_DIR}/all_features.csv')
feature_cols = [c for c in all_features.columns if c not in ['PID', 'pph']]
X = all_features[feature_cols]
y = all_features['pph']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(y_train)} (PPH+ {sum(y_train)})")
print(f"Test:  {len(y_test)} (PPH+ {sum(y_test)})")

# =====================================================================
# 2. Bootstrap LASSO 排名 → feature subsets
# =====================================================================
freq_df = pd.read_csv(f'{CSV_DIR}/bootstrap_selection_frequency_clean.csv')
top30_features = freq_df.head(30)['feature'].tolist()
top20_features = freq_df.head(20)['feature'].tolist()

# =====================================================================
# 3. Joint Search Top 5 配置(by Test composite)
# =====================================================================
model_configs = [
    {  # #1 冠軍 — 紅色實線(原版同樣風格)
        'name': 'RF + SMOTEENN + Top30 + KNN k=1',
        'features': top30_features,
        'imputer_k': 1,
        'imbalance': 'SMOTEENN',
        'model': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'color': 'red',
        'linewidth': 3.5,
        'linestyle': '-',
        'alpha': 1.0,
    },
    {  # #2
        'name': 'DecisionTree + SMOTEENN + Top30 + KNN k=3',
        'features': top30_features,
        'imputer_k': 3,
        'imbalance': 'SMOTEENN',
        'model': DecisionTreeClassifier(random_state=42),
        'color': '#FF8C00',  # dark orange
        'linewidth': 2,
        'linestyle': '--',
        'alpha': 0.7,
    },
    {  # #3
        'name': 'ExtraTrees + SMOTEENN + Top30 + KNN k=1',
        'features': top30_features,
        'imputer_k': 1,
        'imbalance': 'SMOTEENN',
        'model': ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'color': '#8B008B',  # dark magenta
        'linewidth': 2,
        'linestyle': '--',
        'alpha': 0.7,
    },
    {  # #4
        'name': 'SVM_RBF + SMOTEENN + Top30 + KNN k=1',
        'features': top30_features,
        'imputer_k': 1,
        'imbalance': 'SMOTEENN',
        'model': SVC(kernel='rbf', probability=True, random_state=42),
        'color': '#1E90FF',  # dodger blue
        'linewidth': 2,
        'linestyle': '--',
        'alpha': 0.7,
    },
    {  # #5
        'name': 'ExtraTrees + RandomUndersampling + Top20 + KNN k=7',
        'features': top20_features,
        'imputer_k': 7,
        'imbalance': 'RandomUndersampling',
        'model': ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'color': '#228B22',  # forest green
        'linewidth': 2,
        'linestyle': '--',
        'alpha': 0.7,
    },
]

# =====================================================================
# 4. 每個 model:選特徵 → impute → scale → resample → train → ROC
# =====================================================================
roc_results = []

for cfg in model_configs:
    name = cfg['name']
    features = cfg['features']
    print(f"\n--- {name} ---")

    X_tr = X_train[features].values
    X_te = X_test[features].values
    y_tr = y_train.values
    y_te = y_test.values

    # KNN imputation (k 變化)
    imputer = KNNImputer(n_neighbors=cfg['imputer_k'])
    X_tr_imp = imputer.fit_transform(X_tr)
    X_te_imp = imputer.transform(X_te)

    # StandardScaler
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr_imp)
    X_te_sc = scaler.transform(X_te_imp)

    # Imbalance
    if cfg['imbalance'] == 'SMOTEENN':
        resampler = SMOTEENN(random_state=42)
    elif cfg['imbalance'] == 'RandomUndersampling':
        resampler = RandomUnderSampler(random_state=42)

    X_tr_res, y_tr_res = resampler.fit_resample(X_tr_sc, y_tr)
    print(f"  After {cfg['imbalance']}: {len(y_tr_res)} samples (PPH+ {sum(y_tr_res)})")

    # Train
    model = cfg['model']
    model.fit(X_tr_res, y_tr_res)

    # Predict on test
    y_prob = model.predict_proba(X_te_sc)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    auc = roc_auc_score(y_te, y_prob)
    print(f"  AUC = {auc:.4f}")

    roc_results.append({
        'name': name,
        'fpr': fpr,
        'tpr': tpr,
        'auc': auc,
        'color': cfg['color'],
        'linewidth': cfg['linewidth'],
        'linestyle': cfg['linestyle'],
        'alpha': cfg['alpha'],
    })

# =====================================================================
# 5. 驗證 AUC(對照 joint search test eval csv)
# =====================================================================
expected = [0.717, 0.686, 0.713, 0.639, 0.688]
print("\n" + "=" * 60)
print("AUC 驗證:")
for i, r in enumerate(roc_results):
    match = "✅" if abs(r['auc'] - expected[i]) < 0.005 else "⚠️"
    print(f"  #{i+1} {r['name'][:50]:<50s} AUC={r['auc']:.4f}  (expect {expected[i]:.3f}) [{match}]")

# =====================================================================
# 6. 繪圖 — 1:1 沿用原版 PLOS 規格
# =====================================================================
fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=300)

# 排序 by AUC(高→低),用於 legend 排序
sorted_results = sorted(roc_results, key=lambda r: r['auc'], reverse=True)

# 先畫 4 個非 RF model(虛線),低 AUC 先畫,高 AUC 蓋上去
non_rf = [r for r in sorted_results if not r['name'].startswith('RF +')]
for r in reversed(non_rf):
    ax.plot(r['fpr'], r['tpr'],
            color=r['color'], linewidth=r['linewidth'],
            linestyle=r['linestyle'], alpha=r['alpha'])

# RF 最後畫(實線,蓋最上層)
rf = sorted_results[0] if sorted_results[0]['name'].startswith('RF +') else \
     next(r for r in roc_results if r['name'].startswith('RF +'))
ax.plot(rf['fpr'], rf['tpr'],
        color=rf['color'], linewidth=rf['linewidth'],
        linestyle=rf['linestyle'], alpha=rf['alpha'])

# Random classifier 對角線
ax.plot([0, 1], [0, 1], color='gray', linewidth=1.5, linestyle=':', alpha=0.7)

# 手動 legend handles(按 AUC 高→低排)
from matplotlib.lines import Line2D
# 重新 sort(放 RF 第一)
legend_order = [r for r in sorted_results]
# 確保 RF 在第一
legend_order = [rf] + [r for r in sorted_results if r['name'] != rf['name']]

legend_handles = []
for r in legend_order:
    if r['name'].startswith('RF +'):
        lbl = f"{r['name']} (AUC = {r['auc']:.3f})"
    else:
        lbl = f"{r['name']} (AUC = {r['auc']:.3f})"
    legend_handles.append(
        Line2D([0], [0], color=r['color'], linewidth=r['linewidth'],
               linestyle=r['linestyle'], alpha=r['alpha'], label=lbl)
    )
legend_handles.append(
    Line2D([0], [0], color='gray', linewidth=1.5, linestyle=':', alpha=0.7,
           label='Random Classifier')
)

# 軸設定(沿用原版)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11)
ax.tick_params(axis='both', labelsize=10)
ax.legend(handles=legend_handles, loc='lower right', fontsize=8.5, framealpha=0.9)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# RGB flatten + 規格檢查
from PIL import Image
img = Image.open(OUT_PATH).convert('RGB')
img.save(OUT_PATH, dpi=(300, 300))
w, h = img.width, img.height
print(f"\nFinal: {w}x{h} px")
print(f"Width ≤2250: {'✅' if w <= 2250 else '❌'}")
print(f"Height ≤2625: {'✅' if h <= 2625 else '❌'}")
print(f"Saved: {OUT_PATH}")
