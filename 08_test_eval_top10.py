"""
Test set evaluation of Top 10 from joint search + 原冠軍
(事後 sanity check,不用於 model selection)
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
import os

# 2026-07-23 路徑遷移:改用「腳本自身位置」推導,不再硬編碼舊倉庫 PPH_Prediction_Model-main。
# BASE = PPH_joint_search_2026_06_05/ (本檔的上一層)。整包資料夾搬到哪都能跑。
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE, '02_experiment_csv')

# 原 `from joint_search import ...` 依賴的 joint_search.py 已更名為 06_joint_search.py,
# 檔名以數字開頭無法直接 import,改用 importlib 由路徑載入(等價於原 import)。
import importlib.util
_spec = importlib.util.spec_from_file_location(
    'joint_search', os.path.join(SCRIPT_DIR, '06_joint_search.py'))
joint_search = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(joint_search)
get_imputer, get_model, get_imbalance, FEATURE_SUBSETS = (joint_search.get_imputer,
                                                          joint_search.get_model,
                                                          joint_search.get_imbalance,
                                                          joint_search.FEATURE_SUBSETS)

# 載入 data
csv_path = os.path.join(CSV_DIR, 'all_features.csv')
all_features = pd.read_csv(csv_path)
feature_cols = [c for c in all_features.columns if c not in ['PID', 'pph']]
X = all_features[feature_cols]
y = all_features['pph']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
y_test_arr = y_test.values

freq_path = os.path.join(CSV_DIR, 'bootstrap_selection_frequency_clean.csv')
freq_df = pd.read_csv(freq_path)

feature_subsets = {}
for name, kind, val in FEATURE_SUBSETS:
    if kind == 'top':
        feats = freq_df.head(val)['feature'].tolist()
    elif kind == 'threshold':
        feats = freq_df[freq_df['selection_frequency'] >= val]['feature'].tolist()
    feature_subsets[name] = feats

# 載入 ranked 結果
ranked = pd.read_csv(os.path.join(CSV_DIR, 'joint_search_ranked.csv'))

# 取 Top 10 + 原冠軍
top10 = ranked.head(10).copy()
champ = ranked[(ranked['imputer']=='KNN_k1') &
               (ranked['model']=='RandomForest') &
               (ranked['imbalance']=='SMOTEENN') &
               (ranked['feature_subset']=='Top30')].copy()

eval_list = pd.concat([top10, champ]).drop_duplicates(subset=['imputer','model','imbalance','feature_subset'])
print(f'要評估 {len(eval_list)} 個 pipeline 在 test set 上的表現')
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

    # Imputation (跟原實驗一致:對 train 全體 fit_transform)
    imputer = get_imputer(imp_name)
    X_tr_imp = imputer.fit_transform(X_tr_sel)
    X_te_imp = imputer.transform(X_te_sel)

    # Scaler
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr_imp)
    X_te_sc = scaler.transform(X_te_imp)

    # Imbalance(對 train 全體,因為這次是 final test,不是 CV)
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

        marker = '🎯 原冠軍' if (mod_name=='RandomForest' and imb_name=='SMOTEENN' and fs_name=='Top30' and imp_name=='KNN_k1') else f'#{int(idx)+1}'
        print(f'{marker:12s} {imp_name:8s} | {mod_name:18s} | {imb_name:22s} | {fs_name:12s} | '
              f'Test recall={recall:.3f} AUC={auc:.3f} MCC={mcc:.3f} F1={f1:.3f} Spec={spec:.3f}')
    except Exception as e:
        print(f'  {imp_name} | {mod_name} | {imb_name} | {fs_name}: ERROR {e}')

# 存
df = pd.DataFrame(results)
out = os.path.join(CSV_DIR, 'top10_test_eval.csv')
df.to_csv(out, index=False)
print(f'\n已存: {out}')

# 比較 CV vs Test 的排名變化
print('\n' + '=' * 100)
print('排名變化(CV composite vs Test composite):')
print('=' * 100)
df['test_composite'] = (
    df['test_recall'].rank(pct=True) +
    df['test_auc'].rank(pct=True) +
    df['test_mcc'].rank(pct=True)
) / 3
df_sorted_test = df.sort_values('test_composite', ascending=False).reset_index(drop=True)
print(df_sorted_test[['cv_rank', 'imputer', 'model', 'imbalance', 'feature_subset',
                       'cv_composite', 'test_composite', 'test_recall', 'test_auc', 'test_mcc']].to_string(index=True))
