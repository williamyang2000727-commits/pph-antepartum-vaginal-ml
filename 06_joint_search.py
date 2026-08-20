"""
Reduced Joint Search Experiment (2026-06-05)
=============================================
目的:驗證原大搜索冠軍 RF+SMOTEENN+Top30+KNN k=1 在「imputation × algorithm × imbalance × feature」
四維度聯合搜索下仍是最佳,擋 reviewer 對 sequential selection bias 的攻擊。

設計:1:1 對齊原 run_experiment.py 的所有細節
- StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
- train_test_split(test_size=0.2, random_state=42, stratify=y)
- Imputation: 在**每一個 CV training fold 內**各自 fit,再 transform 該 fold 的 validation set
- StandardScaler: 同上,每一個 training fold 內各自 fit
  ⚠️ 2026-08-19 修正:原本這兩步在切 fold 之前就對整個 training set fit_transform,
     造成 cross-validation 層級的 data leakage(見 run_one_pipeline() 內的說明)。
     此處檔頭原本仍寫著修正前的做法,2026-08-20 一併對齊。
     independent test set 的評估流程不受影響(本來就是 fit on train 615 → transform test 154)。
- Imbalance: 在 CV training fold 內 resample(同原實驗)
- Composite score: (recall.rank_pct + auc.rank_pct + mcc.rank_pct) / 3

代表性挑選:
- imputation: 7 種 (KNN k=1/3/5/7/10 + MICE + Median)
- algorithm:  9 種(各家族代表 + GaussianNB 弱基線)
- imbalance:  7 種(各家族代表 + None 對照)
- feature:    6 種(top_8/15/20/25/30 + threshold_45)
總:7 × 9 × 7 × 6 = 2,646 unique pipelines × 10-fold CV = 26,460 fits
"""

import sys
import os
import argparse

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
import time
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import (recall_score, precision_score, f1_score,
                             roc_auc_score, matthews_corrcoef, confusion_matrix)

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek


# =====================================================================
# 1. Imputation 字典(7 種)
# =====================================================================
def get_imputer(name):
    """回傳 fresh imputer instance(每次 fit 都用新的,避免狀態污染)"""
    if name == 'KNN_k1':
        return KNNImputer(n_neighbors=1)
    elif name == 'KNN_k3':
        return KNNImputer(n_neighbors=3)
    elif name == 'KNN_k5':
        return KNNImputer(n_neighbors=5)
    elif name == 'KNN_k7':
        return KNNImputer(n_neighbors=7)
    elif name == 'KNN_k10':
        return KNNImputer(n_neighbors=10)
    elif name == 'MICE':
        return IterativeImputer(random_state=42, max_iter=10)
    elif name == 'Median':
        return SimpleImputer(strategy='median')
    else:
        raise ValueError(f'Unknown imputer: {name}')

IMPUTERS = ['KNN_k1', 'KNN_k3', 'KNN_k5', 'KNN_k7', 'KNN_k10', 'MICE', 'Median']


# =====================================================================
# 2. Algorithm 字典(9 種,各家族代表 + GaussianNB 弱基線)
# =====================================================================
def get_model(name):
    """回傳 fresh model instance"""
    if name == 'LogisticRegression':
        return LogisticRegression(max_iter=2000, random_state=42)
    elif name == 'DecisionTree':
        return DecisionTreeClassifier(random_state=42)
    elif name == 'RandomForest':
        return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif name == 'ExtraTrees':
        return ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif name == 'XGBoost':
        return XGBClassifier(n_estimators=100, random_state=42,
                             eval_metric='logloss', verbosity=0)
    elif name == 'LightGBM':
        return LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    elif name == 'SVM_RBF':
        return SVC(kernel='rbf', probability=True, random_state=42)
    elif name == 'KNN':
        return KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    elif name == 'GaussianNB':
        return GaussianNB()
    else:
        raise ValueError(f'Unknown model: {name}')

MODELS = ['LogisticRegression', 'DecisionTree', 'RandomForest', 'ExtraTrees',
          'XGBoost', 'LightGBM', 'SVM_RBF', 'KNN', 'GaussianNB']


# =====================================================================
# 3. Imbalance 字典(7 種,各家族代表 + None 對照)
# =====================================================================
def get_imbalance(name):
    """回傳 imbalance method instance 或 None / 'balanced'"""
    if name == 'SMOTE':
        return SMOTE(random_state=42)
    elif name == 'ADASYN':
        return ADASYN(random_state=42)
    elif name == 'RandomUndersampling':
        return RandomUnderSampler(random_state=42)
    elif name == 'SMOTEENN':
        return SMOTEENN(random_state=42)
    elif name == 'SMOTETomek':
        return SMOTETomek(random_state=42)
    elif name == 'ClassWeight_Balanced':
        return 'balanced'  # 特殊處理
    elif name == 'None':
        return None
    else:
        raise ValueError(f'Unknown imbalance: {name}')

IMBALANCES = ['SMOTE', 'ADASYN', 'RandomUndersampling', 'SMOTEENN', 'SMOTETomek',
              'ClassWeight_Balanced', 'None']


# =====================================================================
# 4. Feature subsets(6 種:top_8/15/20/25/30 + threshold_45)
# =====================================================================
FEATURE_SUBSETS = [
    ('Top8', 'top', 8),
    ('Top15', 'top', 15),
    ('Top20', 'top', 20),
    ('Top25', 'top', 25),
    ('Top30', 'top', 30),
    ('Threshold45', 'threshold', 0.45),
]


# =====================================================================
# 主要實驗函數
# =====================================================================
def run_one_pipeline(X_train_raw, y_train, imputer_name, model_name, imbalance_name,
                     feature_list, kfold):
    """跑一個 (imputation, algorithm, imbalance, feature) 組合,回傳 10-fold CV 平均指標"""
    try:
        # 1. Subset features
        X_train_sel = X_train_raw[feature_list].values
        y_train_arr = y_train.values

        # 2026-08-19 修正 cross-validation 層級的 data leakage：
        #   原本 imputation 與 StandardScaler 在切 fold 之前就對整個 training set fit_transform，
        #   使每個 validation fold 的資料都參與了填補與標準化參數的估計（KNN k=1 尤其嚴重），
        #   導致 cross-validation 效能偏樂觀，也與論文所引 Kapoor & Narayanan 的 fit-on-train 原則相違。
        #   現改為在每一個 training fold 內獨立 fit，再 transform 該 fold 的 validation set。
        #   ⚠️ independent test set 的評估流程不受影響（本來就是 fit on train 615 → transform test 154）。
        cv_recalls, cv_precisions, cv_f1s, cv_aucs, cv_mccs = [], [], [], [], []

        for train_idx, val_idx in kfold.split(X_train_sel, y_train_arr):
            imputer = get_imputer(imputer_name)
            X_cv_train = imputer.fit_transform(X_train_sel[train_idx])
            X_cv_val = imputer.transform(X_train_sel[val_idx])
            scaler = StandardScaler()
            X_cv_train = scaler.fit_transform(X_cv_train)
            X_cv_val = scaler.transform(X_cv_val)
            y_cv_train = y_train_arr[train_idx]
            y_cv_val = y_train_arr[val_idx]

            # Imbalance
            imb = get_imbalance(imbalance_name)
            class_weight = None
            if imb is None:
                X_res, y_res = X_cv_train, y_cv_train
            elif imb == 'balanced':
                X_res, y_res = X_cv_train, y_cv_train
                class_weight = 'balanced'
            else:
                try:
                    X_res, y_res = imb.fit_resample(X_cv_train, y_cv_train)
                except Exception:
                    X_res, y_res = X_cv_train, y_cv_train

            # Model
            model = get_model(model_name)
            if class_weight is not None and hasattr(model, 'class_weight'):
                model.set_params(class_weight=class_weight)

            try:
                model.fit(X_res, y_res)
            except Exception:
                continue

            y_pred = model.predict(X_cv_val)

            cv_recalls.append(recall_score(y_cv_val, y_pred, zero_division=0))
            cv_precisions.append(precision_score(y_cv_val, y_pred, zero_division=0))
            cv_f1s.append(f1_score(y_cv_val, y_pred, zero_division=0))
            cv_mccs.append(matthews_corrcoef(y_cv_val, y_pred))

            if hasattr(model, 'predict_proba'):
                try:
                    y_prob = model.predict_proba(X_cv_val)[:, 1]
                    cv_aucs.append(roc_auc_score(y_cv_val, y_prob))
                except Exception:
                    cv_aucs.append(0.5)
            elif hasattr(model, 'decision_function'):
                try:
                    y_score = model.decision_function(X_cv_val)
                    cv_aucs.append(roc_auc_score(y_cv_val, y_score))
                except Exception:
                    cv_aucs.append(0.5)
            else:
                cv_aucs.append(0.5)

        if len(cv_recalls) == 0:
            return None

        return {
            'imputer': imputer_name,
            'model': model_name,
            'imbalance': imbalance_name,
            'feature_subset': '',  # 後填
            'n_features': len(feature_list),
            'cv_recall_mean': float(np.mean(cv_recalls)),
            'cv_recall_std': float(np.std(cv_recalls)),
            'cv_precision_mean': float(np.mean(cv_precisions)),
            'cv_f1_mean': float(np.mean(cv_f1s)),
            'cv_auc_mean': float(np.mean(cv_aucs)),
            'cv_mcc_mean': float(np.mean(cv_mccs)),
            'n_folds_success': len(cv_recalls),
        }
    except Exception as e:
        return {
            'imputer': imputer_name,
            'model': model_name,
            'imbalance': imbalance_name,
            'error': str(e),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry_run', action='store_true', help='只跑 8 組合校時')
    parser.add_argument('--output_dir', default='.')
    parser.add_argument('--start_idx', type=int, default=0, help='從第 N 個組合開始(續跑用)')
    args = parser.parse_args()

    print('=' * 80)
    print('  PPH Reduced Joint Search Experiment (2026-06-05)')
    print('=' * 80)

    # 1. 載入資料
    csv_path = os.path.join(CSV_DIR, 'all_features.csv')
    all_features = pd.read_csv(csv_path)
    feature_cols = [c for c in all_features.columns if c not in ['PID', 'pph']]
    X = all_features[feature_cols]
    y = all_features['pph']

    # 1:1 對齊原 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f'  Train: {len(y_train)} 樣本, PPH+: {y_train.sum()}')
    print(f'  Test : {len(y_test)} 樣本, PPH+: {y_test.sum()} (本實驗不用,只用 train 做 CV)')

    # 2. 載入 LASSO ranking
    freq_path = os.path.join(CSV_DIR, 'bootstrap_selection_frequency_clean.csv')
    freq_df = pd.read_csv(freq_path)

    # 3. 構造 feature subsets
    feature_subsets = {}
    for name, kind, val in FEATURE_SUBSETS:
        if kind == 'top':
            feats = freq_df.head(val)['feature'].tolist()
        elif kind == 'threshold':
            feats = freq_df[freq_df['selection_frequency'] >= val]['feature'].tolist()
        feature_subsets[name] = feats
        print(f'  Feature subset {name}: {len(feats)} features')

    # 4. KFold(1:1 對齊原實驗)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # 5. 構造所有組合
    combos = []
    for imp in IMPUTERS:
        for mod in MODELS:
            for imb in IMBALANCES:
                for fs_name, _, _ in FEATURE_SUBSETS:
                    combos.append((imp, mod, imb, fs_name))

    print(f'\n  總組合數: {len(combos)} ({len(IMPUTERS)}×{len(MODELS)}×{len(IMBALANCES)}×{len(FEATURE_SUBSETS)})')

    if args.dry_run:
        combos = combos[:8]
        print(f'  ⚠️ DRY RUN: 只跑前 8 個校時')

    # 6. 跑!
    results = []
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, 'progress.log')
    csv_file = os.path.join(args.output_dir, 'joint_search_results.csv')

    print(f'\n  輸出 CSV: {csv_file}')
    print(f'  進度 log: {log_file}')
    print('=' * 80)

    start_time = time.time()

    # 如果是續跑,先載入舊結果
    if args.start_idx > 0 and os.path.exists(csv_file):
        old_df = pd.read_csv(csv_file)
        results = old_df.to_dict('records')
        print(f'  從 idx {args.start_idx} 續跑,已有 {len(results)} 筆舊結果')

    for i, (imp, mod, imb, fs_name) in enumerate(combos):
        if i < args.start_idx:
            continue

        feats = feature_subsets[fs_name]
        t0 = time.time()
        r = run_one_pipeline(X_train, y_train, imp, mod, imb, feats, kfold)
        if r is not None:
            r['feature_subset'] = fs_name
            results.append(r)
        elapsed_combo = time.time() - t0

        # 每 10 個寫一次 csv(防中斷)
        if (i + 1) % 10 == 0 or i == len(combos) - 1:
            pd.DataFrame(results).to_csv(csv_file, index=False)

        # 進度
        total_elapsed = time.time() - start_time
        avg_per_combo = total_elapsed / (i + 1 - args.start_idx) if (i + 1 - args.start_idx) > 0 else 0
        remaining_combos = len(combos) - (i + 1)
        eta_min = (remaining_combos * avg_per_combo) / 60

        msg = f'[{i+1}/{len(combos)}] {imp:8s} | {mod:18s} | {imb:22s} | {fs_name:12s} | {elapsed_combo:5.1f}s | ETA {eta_min:.1f} min'
        print(msg)
        with open(log_file, 'a') as f:
            f.write(msg + '\n')

    # 7. 收尾
    df = pd.DataFrame(results)
    df.to_csv(csv_file, index=False)

    # 加 composite score(同原實驗)
    df_valid = df[df['cv_recall_mean'].notna()].copy()
    df_valid['composite_score'] = (
        df_valid['cv_recall_mean'].rank(pct=True) +
        df_valid['cv_auc_mean'].rank(pct=True) +
        df_valid['cv_mcc_mean'].rank(pct=True)
    ) / 3
    df_valid = df_valid.sort_values('composite_score', ascending=False).reset_index(drop=True)
    df_valid.to_csv(os.path.join(args.output_dir, 'joint_search_ranked.csv'), index=False)

    total_min = (time.time() - start_time) / 60
    print(f'\n  ✅ 完成!總耗時 {total_min:.1f} 分鐘')
    print(f'  Top 10 composite score:')
    print(df_valid.head(10)[['imputer', 'model', 'imbalance', 'feature_subset',
                              'cv_recall_mean', 'cv_auc_mean', 'cv_mcc_mean',
                              'composite_score']].to_string(index=False))

    # 找原冠軍排名
    champ = df_valid[(df_valid['imputer'] == 'KNN_k1') &
                     (df_valid['model'] == 'RandomForest') &
                     (df_valid['imbalance'] == 'SMOTEENN') &
                     (df_valid['feature_subset'] == 'Top30')]
    if len(champ) > 0:
        rank = champ.index[0] + 1
        print(f'\n  🎯 原冠軍 RF+SMOTEENN+Top30+KNN k=1 排名: {rank}/{len(df_valid)}')
        print(champ.iloc[0][['cv_recall_mean', 'cv_auc_mean', 'cv_mcc_mean', 'composite_score']])


if __name__ == '__main__':
    main()
