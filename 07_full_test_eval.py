#!/usr/bin/env python3
"""
07_full_test_eval.py - Evaluate all 2,646 unique pipelines from the
                        joint search on the independent test set.

For each pipeline, the model is refit on the full training set (n=615)
and predictions are evaluated on the held-out test set (n=154).
Composite score: 0.4 x MCC + 0.3 x AUC + 0.3 x Recall.
"""


import pandas as pd
import numpy as np
import warnings
import time
import sys
import os

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (recall_score, precision_score, f1_score,
                             roc_auc_score, matthews_corrcoef, confusion_matrix,
                             accuracy_score)

sys.path.insert(0, '/Users/yangyongcheng/Desktop/PPH_Prediction_Model-main/PPH_v2_corrected/corrected/joint_search_2026_06_05/')
from joint_search import (get_imputer, get_model, get_imbalance,
                          IMPUTERS, MODELS, IMBALANCES, FEATURE_SUBSETS)

# Load data
csv_path = '/Users/yangyongcheng/Desktop/PPH_Prediction_Model-main/PPH_v2_corrected/corrected/_ORIGINAL_EXPERIMENTS_DO_NOT_MODIFY/02_experiment_csv/all_features.csv'
all_features = pd.read_csv(csv_path)
feature_cols = [c for c in all_features.columns if c not in ['PID', 'pph']]
X = all_features[feature_cols]
y = all_features['pph']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
y_test_arr = y_test.values
y_train_arr = y_train.values

freq_path = '/Users/yangyongcheng/Desktop/PPH_Prediction_Model-main/PPH_v2_corrected/corrected/_ORIGINAL_EXPERIMENTS_DO_NOT_MODIFY/02_experiment_csv/bootstrap_selection_frequency_clean.csv'
freq_df = pd.read_csv(freq_path)

feature_subsets = {}
for name, kind, val in FEATURE_SUBSETS:
    if kind == 'top':
        feats = freq_df.head(val)['feature'].tolist()
    elif kind == 'threshold':
        feats = freq_df[freq_df['selection_frequency'] >= val]['feature'].tolist()
    feature_subsets[name] = feats

# Build all 2,646 combinations
combos = []
for imp in IMPUTERS:
    for mod in MODELS:
        for imb in IMBALANCES:
            for fs_name, _, _ in FEATURE_SUBSETS:
                combos.append((imp, mod, imb, fs_name))

print(f'Total combinations: {len(combos)}')
print(f'Test set: {len(y_test_arr)} samples, PPH+: {y_test_arr.sum()}')
print('=' * 80)

results = []
output_dir = '/Users/yangyongcheng/Desktop/PPH_Prediction_Model-main/PPH_v2_corrected/corrected/joint_search_2026_06_05/'
csv_out = os.path.join(output_dir, 'full_test_eval_results.csv')
log_out = os.path.join(output_dir, 'full_test_eval_progress.log')

# Clear old log
if os.path.exists(log_out):
    os.remove(log_out)

start = time.time()
for i, (imp_name, mod_name, imb_name, fs_name) in enumerate(combos):
    try:
        feats = feature_subsets[fs_name]
        X_tr_sel = X_train[feats].values
        X_te_sel = X_test[feats].values

        # Imputation: fit_transform on full train (same as the original experiment)
        imputer = get_imputer(imp_name)
        X_tr_imp = imputer.fit_transform(X_tr_sel)
        X_te_imp = imputer.transform(X_te_sel)

        # Scaler
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_imp)
        X_te_sc = scaler.transform(X_te_imp)

        # Imbalance (on full train)
        imb = get_imbalance(imb_name)
        class_weight = None
        if imb is None:
            X_res, y_res = X_tr_sc, y_train_arr
        elif imb == 'balanced':
            X_res, y_res = X_tr_sc, y_train_arr
            class_weight = 'balanced'
        else:
            try:
                X_res, y_res = imb.fit_resample(X_tr_sc, y_train_arr)
            except Exception:
                X_res, y_res = X_tr_sc, y_train_arr

        # Model
        model = get_model(mod_name)
        if class_weight is not None and hasattr(model, 'class_weight'):
            model.set_params(class_weight=class_weight)

        model.fit(X_res, y_res)
        y_pred = model.predict(X_te_sc)

        # Metrics
        if hasattr(model, 'predict_proba'):
            try:
                y_prob = model.predict_proba(X_te_sc)[:, 1]
                auc = roc_auc_score(y_test_arr, y_prob)
            except Exception:
                auc = np.nan
        elif hasattr(model, 'decision_function'):
            try:
                y_score = model.decision_function(X_te_sc)
                auc = roc_auc_score(y_test_arr, y_score)
            except Exception:
                auc = np.nan
        else:
            auc = np.nan

        tn, fp, fn, tp = confusion_matrix(y_test_arr, y_pred, labels=[0,1]).ravel()
        recall = recall_score(y_test_arr, y_pred, zero_division=0)
        precision = precision_score(y_test_arr, y_pred, zero_division=0)
        f1 = f1_score(y_test_arr, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test_arr, y_pred)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        acc = accuracy_score(y_test_arr, y_pred)

        results.append({
            'imputer': imp_name, 'model': mod_name, 'imbalance': imb_name,
            'feature_subset': fs_name,
            'test_recall': recall, 'test_spec': spec, 'test_precision': precision,
            'test_f1': f1, 'test_auc': auc, 'test_mcc': mcc, 'test_acc': acc,
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        })
    except Exception as e:
        results.append({
            'imputer': imp_name, 'model': mod_name, 'imbalance': imb_name,
            'feature_subset': fs_name, 'error': str(e),
        })

    # Progress
    if (i + 1) % 50 == 0 or i == len(combos) - 1:
        el = time.time() - start
        eta = (el / (i + 1)) * (len(combos) - i - 1) / 60
        msg = f'[{i+1}/{len(combos)}] {imp_name:8s} | {mod_name:18s} | {imb_name:22s} | {fs_name:12s} | ETA {eta:.1f} min'
        print(msg)
        with open(log_out, 'a') as f:
            f.write(msg + '\n')
        # write csv
        pd.DataFrame(results).to_csv(csv_out, index=False)

# Cleanup
df = pd.DataFrame(results)
df.to_csv(csv_out, index=False)

# Compute the true composite score
df_valid = df[df['test_recall'].notna() & df['test_auc'].notna()].copy()
df_valid['test_composite'] = (
    0.4 * df_valid['test_mcc'] +
    0.3 * df_valid['test_auc'] +
    0.3 * df_valid['test_recall']
)
df_valid = df_valid.sort_values('test_composite', ascending=False).reset_index(drop=True)

ranked_out = os.path.join(output_dir, 'full_test_eval_ranked.csv')
df_valid.to_csv(ranked_out, index=False)

total_min = (time.time() - start) / 60
print('\n' + '=' * 80)
print(f'✅ Done!Totalelapsed {total_min:.1f} min')
print(f'  Valid results: {len(df_valid)} / {len(combos)}')
print('=' * 80)
print('\nTop 10 by test composite:')
print(df_valid.head(10)[['imputer','model','imbalance','feature_subset',
                          'test_recall','test_auc','test_mcc','test_composite']].to_string(index=True))

# original championranking
champ = df_valid[(df_valid['imputer']=='KNN_k1') & (df_valid['model']=='RandomForest') &
                  (df_valid['imbalance']=='SMOTEENN') & (df_valid['feature_subset']=='Top30')]
if len(champ) > 0:
    rank = champ.index[0] + 1
    print(f'\n🎯 original champion RF+SMOTEENN+Top30+KNN k=1 across all {len(df_valid)}  ranking in the test eval: {rank}')
    print(champ[['test_recall','test_auc','test_mcc','test_composite']].to_string())
