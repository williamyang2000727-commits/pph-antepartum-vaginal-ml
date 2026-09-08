"""投稿模型重現：特徵選擇與最終模型共用同一個實作。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault("SCIPY_ARRAY_API", "1")
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, matthews_corrcoef, brier_score_loss
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent

def select_features(all_features, config):
    feature_cols = [col for col in all_features.columns if col not in ['PID', 'pph']]

    X = all_features[feature_cols]
    y = all_features['pph']
    pids = all_features['PID']

    X_train, X_test, y_train, y_test, pid_train, pid_test = train_test_split(
        X, y, pids,
        test_size=config["test_size"],
        random_state=config["seed"],
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

    N_BOOTSTRAP = config["bootstrap_resamples"]
    SAMPLE_FRACTION = config["bootstrap_fraction"]
    C_VALUES = config["lasso_c"]
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
                random_state=config["seed"]
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

    return freq_df, X_train, X_test, y_train, y_test


def evaluate(y, prob, threshold):
    pred = prob >= threshold
    tn, fp, fn, tp = map(int, confusion_matrix(y, pred, labels=[0, 1]).ravel())
    recall = tp / (tp + fn)
    auc = float(roc_auc_score(y, prob))
    mcc = float(matthews_corrcoef(y, pred))
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, sensitivity=recall,
                specificity=tn/(tn+fp), precision=tp/(tp+fp) if tp+fp else 0.,
                npv=tn/(tn+fn) if tn+fn else 0., f1=2*tp/(2*tp+fp+fn),
                accuracy=(tp+tn)/len(y), auc=auc, mcc=mcc,
                brier=float(brier_score_loss(y, prob)),
                average_precision=float(average_precision_score(y, prob)),
                composite=.4*mcc+.3*auc+.3*recall)


def verify(actual, expected, tolerance):
    bad = {k: dict(actual=actual.get(k), expected=v) for k,v in expected.items()
           if k not in actual or not np.isfinite(actual[k]) or not np.isfinite(v)
           or abs(actual[k]-v) > tolerance}
    if bad:
        raise ValueError("結果與投稿參考值不符：" + json.dumps(bad, ensure_ascii=False))


def main(stage="evaluate"):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=ROOT/"02_experiment_csv")
    ap.add_argument("--output-dir", type=Path, default=ROOT/"results")
    ap.add_argument("--config", type=Path, default=ROOT/"submission_config.json")
    ap.add_argument("--verify", action="store_true", help="核對原研究資料的特徵與投稿指標；其他資料不應啟用")
    args = ap.parse_args()
    config = json.loads(args.config.read_text())
    data_path = args.data_dir/"all_features.csv"
    df = pd.read_csv(data_path)
    if df.PID.duplicated().any() or not set(df.pph.unique()) <= {0,1}:
        raise ValueError("資料必須每位一列，pph 必須為0或1")
    freq, Xtr, Xte, ytr, yte = select_features(df, config)
    if args.verify:
        ref = pd.DataFrame(config["selection_reference"]).set_index("feature")
        got = freq.set_index("feature")
        if set(got.index) != set(ref.index) or not np.array_equal(
                got.loc[ref.index,"selection_count"], ref.selection_count):
            raise ValueError("Bootstrap選擇次數與投稿參考不符")
        # 固定已發表的並列次序；驗證頻率後才使用，不掩蓋選擇結果差異。
        freq = got.loc[ref.index].reset_index()
    feats = freq.head(config["n_features"]).feature.tolist()
    report = dict(configuration={k:v for k,v in config.items() if k not in
                  ("selection_reference","expected_metrics")},
                  train_n=len(ytr), test_n=len(yte), train_events=int(ytr.sum()),
                  test_events=int(yte.sum()), features=feats,
                  input_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest())
    if stage != "select":
        imp = KNNImputer(n_neighbors=config["knn_k"])
        a, b = imp.fit_transform(Xtr[feats]), imp.transform(Xte[feats])
        sc = StandardScaler(); a, b = sc.fit_transform(a), sc.transform(b)
        a, yr = SMOTEENN(random_state=config["seed"]).fit_resample(a, ytr.to_numpy())
        model = XGBClassifier(**config["xgboost"])
        model.fit(a, yr)
        prob = model.predict_proba(b)[:,1]
        metrics = evaluate(yte, prob, config["threshold"])
        report.update(metrics=metrics, resampled_n=len(yr), resampled_events=int(sum(yr)))
        if args.verify:
            verify(metrics, config["expected_metrics"], config["tolerance"])
        args.output_dir.mkdir(parents=True, exist_ok=True)
        # 原生樹模型不含KNN的訓練參考資料；不輸出病人列或序列化填補器。
        model.save_model(args.output_dir/"final_model_xgboost_native.json")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    freq.to_csv(args.output_dir/"bootstrap_selection_frequency_clean.csv", index=False)
    report["verification"] = "passed" if args.verify else "not_requested"
    (args.output_dir/"reproduction_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k:v for k,v in report.items() if k not in ("features","configuration")}, ensure_ascii=False, indent=2))
    return 0
