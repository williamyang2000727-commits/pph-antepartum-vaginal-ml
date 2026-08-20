"""
INTERNAL ONLY — 全 2,646 個 pipeline 在 test set 上的 evaluation
用對的算式: 0.4 × MCC + 0.3 × AUC + 0.3 × Recall

⚠️ 結果是否寫進論文由用戶決定
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
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE, '02_experiment_csv')

# 原 `from joint_search import ...` 依賴的 joint_search.py 已更名為 06_joint_search.py,
# 檔名以數字開頭無法直接 import,改用 importlib 由路徑載入(等價於原 import)。
import importlib.util
_spec = importlib.util.spec_from_file_location(
    'joint_search', os.path.join(SCRIPT_DIR, '06_joint_search.py'))
joint_search = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(joint_search)
get_imputer, get_model, get_imbalance = (joint_search.get_imputer,
                                         joint_search.get_model,
                                         joint_search.get_imbalance)
IMPUTERS, MODELS, IMBALANCES, FEATURE_SUBSETS = (joint_search.IMPUTERS,
                                                 joint_search.MODELS,
                                                 joint_search.IMBALANCES,
                                                 joint_search.FEATURE_SUBSETS)

# 載入 data
csv_path = os.path.join(CSV_DIR, 'all_features.csv')
all_features = pd.read_csv(csv_path)
feature_cols = [c for c in all_features.columns if c not in ['PID', 'pph']]
X = all_features[feature_cols]
y = all_features['pph']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
y_test_arr = y_test.values
y_train_arr = y_train.values

freq_path = os.path.join(CSV_DIR, 'bootstrap_selection_frequency_clean.csv')
freq_df = pd.read_csv(freq_path)

feature_subsets = {}
for name, kind, val in FEATURE_SUBSETS:
    if kind == 'top':
        feats = freq_df.head(val)['feature'].tolist()
    elif kind == 'threshold':
        feats = freq_df[freq_df['selection_frequency'] >= val]['feature'].tolist()
    feature_subsets[name] = feats

# 構造所有 2,646 個組合
combos = []
for imp in IMPUTERS:
    for mod in MODELS:
        for imb in IMBALANCES:
            for fs_name, _, _ in FEATURE_SUBSETS:
                combos.append((imp, mod, imb, fs_name))

print(f'總組合: {len(combos)}')
print(f'Test set: {len(y_test_arr)} 樣本, PPH+: {y_test_arr.sum()}')
print('=' * 80)

results = []
output_dir = CSV_DIR  # 產出 full_test_eval_results.csv / ranked.csv 存 02_experiment_csv/
csv_out = os.path.join(output_dir, 'full_test_eval_results.csv')
os.makedirs(os.path.join(BASE, '04_logs'), exist_ok=True)   # 2026-08-20：倉庫裡沒有這個目錄
log_out = os.path.join(BASE, '04_logs', 'full_test_eval_progress.log')  # log 統一進 04_logs/

# 清舊 log
if os.path.exists(log_out):
    os.remove(log_out)

start = time.time()
for i, (imp_name, mod_name, imb_name, fs_name) in enumerate(combos):
    try:
        feats = feature_subsets[fs_name]
        X_tr_sel = X_train[feats].values
        X_te_sel = X_test[feats].values

        # Imputation: 對 train 全體 fit_transform(同原實驗)
        imputer = get_imputer(imp_name)
        X_tr_imp = imputer.fit_transform(X_tr_sel)
        X_te_imp = imputer.transform(X_te_sel)

        # Scaler
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_imp)
        X_te_sc = scaler.transform(X_te_imp)

        # Imbalance(對 train 全體)
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

    # 進度
    if (i + 1) % 50 == 0 or i == len(combos) - 1:
        el = time.time() - start
        eta = (el / (i + 1)) * (len(combos) - i - 1) / 60
        msg = f'[{i+1}/{len(combos)}] {imp_name:8s} | {mod_name:18s} | {imb_name:22s} | {fs_name:12s} | ETA {eta:.1f} min'
        print(msg)
        with open(log_out, 'a') as f:
            f.write(msg + '\n')
        # 寫 csv
        pd.DataFrame(results).to_csv(csv_out, index=False)

# 收尾
df = pd.DataFrame(results)
df.to_csv(csv_out, index=False)

# 計算真 composite
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
print(f'✅ 完成!總耗時 {total_min:.1f} 分鐘')
print(f'  有效結果: {len(df_valid)} / {len(combos)}')
print('=' * 80)
print('\nTop 10 by test composite:')
print(df_valid.head(10)[['imputer','model','imbalance','feature_subset',
                          'test_recall','test_auc','test_mcc','test_composite']].to_string(index=True))

# 原冠軍排名
champ = df_valid[(df_valid['imputer']=='KNN_k1') & (df_valid['model']=='RandomForest') &
                  (df_valid['imbalance']=='SMOTEENN') & (df_valid['feature_subset']=='Top30')]
if len(champ) > 0:
    rank = champ.index[0] + 1
    print(f'\n🎯 原冠軍 RF+SMOTEENN+Top30+KNN k=1 在全 {len(df_valid)} 個 test eval 中排名: {rank}')
    print(champ[['test_recall','test_auc','test_mcc','test_composite']].to_string())
