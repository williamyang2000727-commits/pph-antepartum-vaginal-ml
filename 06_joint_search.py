#!/usr/bin/env python3
"""
06_joint_search.py - Joint search across four pipeline dimensions.

Design:
  - StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
  - train_test_split(test_size=0.2, random_state=42, stratify=y)
  - Imputation: fit_transform on full training set (same as the
    original protocol, not inside CV)
  - Imbalance handling: resampling applied inside each CV training
    fold only (validation fold untouched)

Search grid:
  - Imputation:        7 methods  (KNN k=1/3/5/7/10, MICE, Median)
  - Algorithms:        9 families
                       (LogisticRegression, DecisionTree, RandomForest,
                        ExtraTrees, XGBoost, LightGBM, SVM_RBF, KNN,
                        GaussianNB)
  - Imbalance:         7 strategies
                       (SMOTE, ADASYN, RandomUndersampling, SMOTEENN,
                        SMOTETomek, ClassWeight_Balanced, None)
  - Feature subsets:   6 configurations
                       (Top8, Top15, Top20, Top25, Top30, Threshold45)

Total: 7 x 9 x 7 x 6 = 2,646 unique pipelines x 10 folds = 26,460 fits
"""



import sys
import os
import argparse
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
# 1. Imputation dict(7 種)
# =====================================================================
def get_imputer(name):
    """iter傳 fresh imputer instance(eachtimes fit 都usenew的,to avoid狀態污染)"""
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
# 2. Algorithm dict(9 種,family representatives + GaussianNB weak baseline)
# =====================================================================
def get_model(name):
    """iter傳 fresh model instance"""
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
# 3. Imbalance dict(7 種,family representatives + None control)
# =====================================================================
def get_imbalance(name):
    """iter傳 imbalance method instance or None / 'balanced'"""
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
        return 'balanced'  # 特殊process
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
# 主toexperimentfunction
# =====================================================================
def run_one_pipeline(X_train_raw, y_train, imputer_name, model_name, imbalance_name,
                     feature_list, kfold):
    """Runone (imputation, algorithm, imbalance, feature) group合,iter傳 10-fold CV average指mark"""
    try:
        # 1. Subset features
        X_train_sel = X_train_raw[feature_list].values
        y_train_arr = y_train.values

        # 2. Imputation(1:1 aligned with the originalexperiment:fit_transform on the full training set)
        imputer = get_imputer(imputer_name)
        X_train_imp = imputer.fit_transform(X_train_sel)

        # 3. Scaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)

        # 4. 10-fold CV(imbalance 在 fold inside resample)
        cv_recalls, cv_precisions, cv_f1s, cv_aucs, cv_mccs = [], [], [], [], []

        for train_idx, val_idx in kfold.split(X_train_scaled, y_train_arr):
            X_cv_train = X_train_scaled[train_idx]
            X_cv_val = X_train_scaled[val_idx]
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
            'feature_subset': '',  # after填
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
    parser.add_argument('--dry_run', action='store_true', help='onlyRun 8 group合校when')
    parser.add_argument('--output_dir', default='.')
    parser.add_argument('--start_idx', type=int, default=0, help='from N group合開始(續Runuse)')
    args = parser.parse_args()

    print('=' * 80)
    print('  PPH Reduced Joint Search experiment (2026-06-05)')
    print('=' * 80)

    # 1. Load data
    csv_path = '/Users/yangyongcheng/Desktop/PPH_Prediction_Model-main/PPH_v2_corrected/corrected/_ORIGINAL_EXPERIMENTS_DO_NOT_MODIFY/02_experiment_csv/all_features.csv'
    all_features = pd.read_csv(csv_path)
    feature_cols = [c for c in all_features.columns if c not in ['PID', 'pph']]
    X = all_features[feature_cols]
    y = all_features['pph']

    # 1:1 aligned with the original split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f'  Train: {len(y_train)} samples, PPH+: {y_train.sum()}')
    print(f'  Test : {len(y_test)} samples, PPH+: {y_test.sum()} (本experimentnotuse,onlyuse train do CV)')

    # 2. Load LASSO ranking
    freq_path = '/Users/yangyongcheng/Desktop/PPH_Prediction_Model-main/PPH_v2_corrected/corrected/_ORIGINAL_EXPERIMENTS_DO_NOT_MODIFY/02_experiment_csv/bootstrap_selection_frequency_clean.csv'
    freq_df = pd.read_csv(freq_path)

    # 3. Build feature subsets
    feature_subsets = {}
    for name, kind, val in FEATURE_SUBSETS:
        if kind == 'top':
            feats = freq_df.head(val)['feature'].tolist()
        elif kind == 'threshold':
            feats = freq_df[freq_df['selection_frequency'] >= val]['feature'].tolist()
        feature_subsets[name] = feats
        print(f'  Feature subset {name}: {len(feats)} features')

    # 4. KFold(1:1 aligned with the originalexperiment)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # 5. Build所hasgroup合
    combos = []
    for imp in IMPUTERS:
        for mod in MODELS:
            for imb in IMBALANCES:
                for fs_name, _, _ in FEATURE_SUBSETS:
                    combos.append((imp, mod, imb, fs_name))

    print(f'\n  Total combinations數: {len(combos)} ({len(IMPUTERS)}x{len(MODELS)}x{len(IMBALANCES)}x{len(FEATURE_SUBSETS)})')

    if args.dry_run:
        combos = combos[:8]
        print(f'  WARNING: DRY RUN: onlyRunbefore 8 校when')

    # 6. Run!
    results = []
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, 'progress.log')
    csv_file = os.path.join(args.output_dir, 'joint_search_results.csv')

    print(f'\n  Output CSV: {csv_file}')
    print(f'  Progress log: {log_file}')
    print('=' * 80)

    start_time = time.time()

    # e.g.果is續Run,firstLoadoldresult
    if args.start_idx > 0 and os.path.exists(csv_file):
        old_df = pd.read_csv(csv_file)
        results = old_df.to_dict('records')
        print(f'  from idx {args.start_idx} 續Run,has {len(results)} recordsoldresult')

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

        # each 10 writeonetimes csv(防middle斷)
        if (i + 1) % 10 == 0 or i == len(combos) - 1:
            pd.DataFrame(results).to_csv(csv_file, index=False)

        # Progress
        total_elapsed = time.time() - start_time
        avg_per_combo = total_elapsed / (i + 1 - args.start_idx) if (i + 1 - args.start_idx) > 0 else 0
        remaining_combos = len(combos) - (i + 1)
        eta_min = (remaining_combos * avg_per_combo) / 60

        msg = f'[{i+1}/{len(combos)}] {imp:8s} | {mod:18s} | {imb:22s} | {fs_name:12s} | {elapsed_combo:5.1f}s | ETA {eta_min:.1f} min'
        print(msg)
        with open(log_file, 'a') as f:
            f.write(msg + '\n')

    # 7. Cleanup
    df = pd.DataFrame(results)
    df.to_csv(csv_file, index=False)

    # add composite score(same as the original experiment)
    df_valid = df[df['cv_recall_mean'].notna()].copy()
    df_valid['composite_score'] = (
        df_valid['cv_recall_mean'].rank(pct=True) +
        df_valid['cv_auc_mean'].rank(pct=True) +
        df_valid['cv_mcc_mean'].rank(pct=True)
    ) / 3
    df_valid = df_valid.sort_values('composite_score', ascending=False).reset_index(drop=True)
    df_valid.to_csv(os.path.join(args.output_dir, 'joint_search_ranked.csv'), index=False)

    total_min = (time.time() - start_time) / 60
    print(f'\n  ✅ Done!Totalelapsed {total_min:.1f} min')
    print(f'  Top 10 composite score:')
    print(df_valid.head(10)[['imputer', 'model', 'imbalance', 'feature_subset',
                              'cv_recall_mean', 'cv_auc_mean', 'cv_mcc_mean',
                              'composite_score']].to_string(index=False))

    # 找original championranking
    champ = df_valid[(df_valid['imputer'] == 'KNN_k1') &
                     (df_valid['model'] == 'RandomForest') &
                     (df_valid['imbalance'] == 'SMOTEENN') &
                     (df_valid['feature_subset'] == 'Top30')]
    if len(champ) > 0:
        rank = champ.index[0] + 1
        print(f'\n  🎯 original champion RF+SMOTEENN+Top30+KNN k=1 ranking: {rank}/{len(df_valid)}')
        print(champ.iloc[0][['cv_recall_mean', 'cv_auc_mean', 'cv_mcc_mean', 'composite_score']])


if __name__ == '__main__':
    main()
