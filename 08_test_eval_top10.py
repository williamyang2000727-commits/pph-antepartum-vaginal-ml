#!/usr/bin/env python3
"""
08_test_eval_top10.py - Sanity-check test-set evaluation across the
                        top 10 pipelines from the joint search.

Each pipeline follows the same fit/transform protocol as the original
experiment (Imputation and Imbalance applied on the full training set,
not inside CV).
"""


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (recall_score, precision_score, f1_score,
                             roc_auc_score, matthews_corrcoef, confusion_matrix,
                             accuracy_score)

import sys
sys.path.insert(0, '/Users/yangyongcheng/Desktop/PPH_Prediction_Model-main/PPH_v2_corrected/corrected/joint_search_2026_06_05/')
from joint_search import get_imputer, get_model, get_imbalance, FEATURE_SUBSETS

# Load data
csv_path = '/Users/yangyongcheng/Desktop/PPH_Prediction_Model-main/PPH_v2_corrected/corrected/_ORIGINAL_EXPERIMENTS_DO_NOT_MODIFY/02_experiment_csv/all_features.csv'
all_features = pd.read_csv(csv_path)
feature_cols = [c for c in all_features.columns if c not in ['PID', 'pph']]
X = all_features[feature_cols]
y = all_features['pph']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
y_test_arr = y_test.values

freq_path = '/Users/yangyongcheng/Desktop/PPH_Prediction_Model-main/PPH_v2_corrected/corrected/_ORIGINAL_EXPERIMENTS_DO_NOT_MODIFY/02_experiment_csv/bootstrap_selection_frequency_clean.csv'
freq_df = pd.read_csv(freq_path)

feature_subsets = {}
for name, kind, val in FEATURE_SUBSETS:
    if kind == 'top':
        feats = freq_df.head(val)['feature'].tolist()
    elif kind == 'threshold':
        feats = freq_df[freq_df['selection_frequency'] >= val]['feature'].tolist()
    feature_subsets[name] = feats

# Load ranked result
ranked = pd.read_csv('/Users/yangyongcheng/Desktop/PPH_Prediction_Model-main/PPH_v2_corrected/corrected/joint_search_2026_06_05/joint_search_ranked.csv')

# take Top 10 plus original champion
top10 = ranked.head(10).copy()
champ = ranked[(ranked['imputer']=='KNN_k1') &
               (ranked['model']=='RandomForest') &
               (ranked['imbalance']=='SMOTEENN') &
               (ranked['feature_subset']=='Top30')].copy()

eval_list = pd.concat([top10, champ]).drop_duplicates(subset=['imputer','model','imbalance','feature_subset'])
print(f'toevaluate {len(eval_list)} pipelines on the test set')
print('=' * 100)

results = []
for idx, row in eval_list.iterrows():
    imp_name = row['imputer']
    mod_name = row['model']
    imb_name = row['imbalance']
    fs_name = row['feature_subset']

    feats = feature_subsets[fs_name]
    X_tr_sel = X_train[feats].values
    X_te_sel = X_test[feats].values
    y_tr_arr = y_train.values

    # Imputation (matches the original experiment:fit_transform on the full training set)
    imputer = get_imputer(imp_name)
    X_tr_imp = imputer.fit_transform(X_tr_sel)
    X_te_imp = imputer.transform(X_te_sel)

    # Scaler
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr_imp)
    X_te_sc = scaler.transform(X_te_imp)

    # Imbalance(on the full training set,because this is the final test,instead of CV)
    imb = get_imbalance(imb_name)
    class_weight = None
    if imb is None:
        X_res, y_res = X_tr_sc, y_tr_arr
    elif imb == 'balanced':
        X_res, y_res = X_tr_sc, y_tr_arr
        class_weight = 'balanced'
    else:
        try:
            X_res, y_res = imb.fit_resample(X_tr_sc, y_tr_arr)
        except Exception:
            X_res, y_res = X_tr_sc, y_tr_arr

    # Model
    model = get_model(mod_name)
    if class_weight is not None and hasattr(model, 'class_weight'):
        model.set_params(class_weight=class_weight)

    try:
        model.fit(X_res, y_res)
        y_pred = model.predict(X_te_sc)

        # Probabilistic predictions
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
            'cv_rank': int(idx) + 1,
            'imputer': imp_name,
            'model': mod_name,
            'imbalance': imb_name,
            'feature_subset': fs_name,
            'cv_recall': row['cv_recall_mean'],
            'cv_auc': row['cv_auc_mean'],
            'cv_mcc': row['cv_mcc_mean'],
            'cv_composite': row['composite_score'],
            'test_recall': recall,
            'test_spec': spec,
            'test_precision': precision,
            'test_f1': f1,
            'test_auc': auc,
            'test_mcc': mcc,
            'test_acc': acc,
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        })

        marker = '🎯 original champion' if (mod_name=='RandomForest' and imb_name=='SMOTEENN' and fs_name=='Top30' and imp_name=='KNN_k1') else f'#{int(idx)+1}'
        print(f'{marker:12s} {imp_name:8s} | {mod_name:18s} | {imb_name:22s} | {fs_name:12s} | '
              f'Test recall={recall:.3f} AUC={auc:.3f} MCC={mcc:.3f} F1={f1:.3f} Spec={spec:.3f}')
    except Exception as e:
        print(f'  {imp_name} | {mod_name} | {imb_name} | {fs_name}: ERROR {e}')

# Save
df = pd.DataFrame(results)
out = '/Users/yangyongcheng/Desktop/PPH_Prediction_Model-main/PPH_v2_corrected/corrected/joint_search_2026_06_05/top10_test_eval.csv'
df.to_csv(out, index=False)
print(f'\nSaved: {out}')

# Compare CV vs test rank changes
print('\n' + '=' * 100)
print('rankingchange(CV composite vs Test composite):')
print('=' * 100)
df['test_composite'] = (
    df['test_recall'].rank(pct=True) +
    df['test_auc'].rank(pct=True) +
    df['test_mcc'].rank(pct=True)
) / 3
df_sorted_test = df.sort_values('test_composite', ascending=False).reset_index(drop=True)
print(df_sorted_test[['cv_rank', 'imputer', 'model', 'imbalance', 'feature_subset',
                       'cv_composite', 'test_composite', 'test_recall', 'test_auc', 'test_mcc']].to_string(index=True))
