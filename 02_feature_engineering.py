#!/usr/bin/env python3
"""
02_feature_engineering.py - Antepartum feature engineering.

Key principle:
  All features use data with measurement_date < delivery_day + 1
  (i.e. including delivery day itself, up to 23:59), because some
  antepartum diagnoses (e.g. preeclampsia) are sometimes recorded
  on the delivery day. This is still leakage-free: the PPH outcome
  itself is recorded post-delivery and is never included.

Literature support:
  - Preeclampsia is a known PPH risk factor (Yunas et al. 2025 Lancet)
  - Chronic hypertension is also associated with PPH risk

Output:
  - all_features.csv: 304 candidate features
  - feature_statistics.csv: per-feature stats and chi-square / t-test results
"""



import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("[Step 2] Feature engineering (including delivery day)")
print("=" * 70)

# =====================================================================
# Loading data
# =====================================================================
print("\nLoading data...")
labels = pd.read_csv('vaginal_delivery_labels.csv')
lab = pd.read_csv('lab_values.csv')
diag = pd.read_csv('diagnoses.csv')

# Convert date formats
labels['time_cutoff'] = pd.to_datetime(labels['time_cutoff'])
lab['measurement_date'] = pd.to_datetime(lab['measurement_date'])
diag['condition_start_date'] = pd.to_datetime(diag['condition_start_date'])

# cleannameswhitespace
diag['condition_clean'] = diag['condition_concept_name'].str.strip()

print(f"labeldata: {len(labels)} records")
print(f"lab testdata: {len(lab)} records")
print(f"diagnosisdata: {len(diag)} records")

# Keep only vaginal-delivery patients
valid_pids = set(labels['PID'])
lab = lab[lab['PID'].isin(valid_pids)].copy()
diag = diag[diag['PID'].isin(valid_pids)].copy()

print(f"Filtered lab data: {len(lab)} records")
print(f"Filtered diagnosis data: {len(diag)} records")

# =====================================================================
# Build cutoff time lookup table
# =====================================================================
cutoff_dict = labels.set_index('PID')['time_cutoff'].to_dict()

# =====================================================================
# Step 1: baseline features
# =====================================================================
print("\n" + "=" * 70)
print("【Step 1: baseline features】")
print("=" * 70)

features = labels[['PID', 'age', 'pph']].copy()
features['is_advanced_maternal_age'] = (features['age'] >= 35).astype(int)

print(f"age: average {features['age'].mean():.1f} yr")
print(f"is_advanced_maternal_age: {features['is_advanced_maternal_age'].sum()}  ({features['is_advanced_maternal_age'].mean()*100:.1f}%)")

# =====================================================================
# Step 2: lab-value features(including delivery day)
# =====================================================================
print("\n" + "=" * 70)
print("【Step 2: lab-value features】(including delivery day)")
print("=" * 70)

# For each lab record, merge the patient cutoff time
lab['time_cutoff'] = lab['PID'].map(cutoff_dict)

# Time filter: including delivery day的data(time_cutoff isdelivery day 00:00)
# Condition: measurement_date < time_cutoff + 1天 = including delivery day
lab_before = lab[lab['measurement_date'] < (lab['time_cutoff'] + pd.Timedelta(days=1))].copy()
print(f"Time filterbefore: {len(lab)} records")
print(f"Time filterafter: {len(lab_before)} records(including delivery day)")

# Coverage rate per lab test item
lab_items = lab_before.groupby('concept_name')['PID'].nunique()
total_patients = len(labels)
coverage = (lab_items / total_patients * 100).sort_values(ascending=False)

print(f"\nLab item coverage rate (>=30% required for inclusion):")
valid_lab_items = []
for item, cov in coverage.items():
    if cov >= 30:
        valid_lab_items.append(item)
        print(f"  {item}: {cov:.1f}%")

print(f"\n符合Condition的lab testitem目: {len(valid_lab_items)} ")

# buildlab valuesfeatures
lab_features = {}

for item in valid_lab_items:
    item_data = lab_before[lab_before['concept_name'] == item]

    # Compute per-patient statistics
    for pid in labels['PID']:
        if pid not in lab_features:
            lab_features[pid] = {}

        patient_data = item_data[item_data['PID'] == pid]['value_as_number']

        item_clean = item.replace(' ', '_').replace('/', '_').replace('-', '_')

        if len(patient_data) > 0:
            lab_features[pid][f'{item_clean}_last'] = patient_data.iloc[-1]
            lab_features[pid][f'{item_clean}_mean'] = patient_data.mean()
            lab_features[pid][f'{item_clean}_min'] = patient_data.min()
            lab_features[pid][f'{item_clean}_max'] = patient_data.max()
            lab_features[pid][f'{item_clean}_count'] = len(patient_data)
            lab_features[pid][f'{item_clean}_std'] = patient_data.std() if len(patient_data) >= 2 else 0
        else:
            lab_features[pid][f'{item_clean}_last'] = np.nan
            lab_features[pid][f'{item_clean}_mean'] = np.nan
            lab_features[pid][f'{item_clean}_min'] = np.nan
            lab_features[pid][f'{item_clean}_max'] = np.nan
            lab_features[pid][f'{item_clean}_count'] = 0
            lab_features[pid][f'{item_clean}_std'] = np.nan

lab_df = pd.DataFrame.from_dict(lab_features, orient='index')
lab_df.index.name = 'PID'
lab_df = lab_df.reset_index()

features = features.merge(lab_df, on='PID', how='left')
print(f"Number of lab-value features: {len(lab_df.columns) - 1}")

# =====================================================================
# Step 3: diagnosis features(including delivery day)
# =====================================================================
print("\n" + "=" * 70)
print("【Step 3: diagnosis features】(including delivery day)")
print("=" * 70)

# For each diagnosis record, merge the patient cutoff time
diag['time_cutoff'] = diag['PID'].map(cutoff_dict)

# Time filter: including delivery day的data
# Condition: condition_start_date < time_cutoff + 1天 = including delivery day
diag_before = diag[diag['condition_start_date'] < (diag['time_cutoff'] + pd.Timedelta(days=1))].copy()
print(f"Time filterbefore: {len(diag)} records")
print(f"Time filterafter: {len(diag_before)} records(including delivery day)")

# ----- 定義diagnosissplitgroup -----

# preeclampsia(PDF 定義的 11 )
preeclampsia_keywords = [
    'Unspecified pre-eclampsia, second trimester',
    'Unspecified pre-eclampsia, third trimester',
    'Unspecified pre-eclampsia, unspecified trimester',
    'Eclampsia complicating the puerperium',
    'HELLP syndrome (HELLP), third trimester',
    'HELLP syndrome (HELLP), unspecified trimester',
    'Mild to moderate pre-eclampsia, second trimester',
    'Mild to moderate pre-eclampsia, third trimester',
    'Pre-existing hypertension with pre-eclampsia, third trimester',
    'Severe pre-eclampsia, second trimester',
    'Severe pre-eclampsia, third trimester'
]

# its他Reference風險因子關鍵詞
diagnosis_groups = {
    'has_preeclampsia': preeclampsia_keywords,
    'has_anemia': ['anemia', 'anaemia'],
    'has_coagulation_disorder': ['coagulation', 'coagulopathy', 'thrombocytopenia', 'platelet'],
    'has_placenta_previa': ['placenta previa', 'placenta praevia'],
    'has_placental_abruption': ['placental abruption', 'abruptio placentae'],
    'has_multiple_gestation': ['twin', 'triplet', 'multiple gestation', 'multiple pregnancy'],
    'has_polyhydramnios': ['polyhydramnios'],
    'has_oligohydramnios': ['oligohydramnios'],
    'has_gestational_diabetes': ['gestational diabetes', 'gdm'],
    'has_gestational_hypertension': ['gestational hypertension', 'pregnancy-induced hypertension', 'pregnancy-induced] hypertension'],
    'has_chronic_hypertension': ['chronic hypertension', 'pre-existing hypertension', r'essential \(primary\) hypertension'],
    'has_uterine_fibroids': ['fibroid', 'leiomyoma', 'myoma'],
    'has_previous_cesarean': ['previous cesarean', 'previous c/s', 'previous caesarean'],
    'has_obesity': ['obesity', 'obese'],
    'has_diabetes_mellitus': ['diabetes mellitus', 'type 1 diabetes', 'type 2 diabetes', 'pre-existing diabetes'],
    'has_thyroid_disorder': ['thyroid', 'hypothyroid', 'hyperthyroid'],
    'has_heart_disease': ['heart disease', 'cardiac', 'cardiomyopathy'],
    'has_kidney_disease': ['kidney disease', 'renal disease', 'chronic kidney'],
    'has_liver_disease': ['liver disease', 'hepatic'],
    'has_autoimmune': ['lupus', 'autoimmune', 'rheumatoid'],
    'has_infection': ['chorioamnionitis', 'infection', 'sepsis'],
    'has_fetal_macrosomia': ['macrosomia', 'large for gestational'],
    'has_fetal_growth_restriction': ['growth restriction', 'iugr', 'small for gestational'],
    'has_preterm': ['preterm', 'premature'],
    'has_prolonged_labor': ['prolonged labor', 'prolonged labour'],
    'has_induced_labor': ['induction', 'induced labor', 'induced labour'],
}

# builddiagnosisfeatures
def check_diagnosis(pid, keywords, diag_data):
    """check病患is否has特定diagnosis(part匹match)"""
    patient_diag = diag_data[diag_data['PID'] == pid]['condition_clean'].str.lower()
    for kw in keywords:
        if isinstance(kw, str):
            kw_lower = kw.lower()
            if patient_diag.str.contains(kw_lower, na=False).any():
                return 1
    return 0

def check_diagnosis_exact(pid, keywords, diag_data):
    """check病患is否has特定diagnosis(completely匹match)"""
    patient_diag = diag_data[diag_data['PID'] == pid]['condition_clean']
    return 1 if patient_diag.isin(keywords).any() else 0

print("\nbuilddiagnosisfeatures...")
diag_features = {pid: {} for pid in labels['PID']}

# has_preeclampsia usecompletely匹match(PDF 定義)
for pid in labels['PID']:
    diag_features[pid]['has_preeclampsia'] = check_diagnosis_exact(pid, preeclampsia_keywords, diag_before)

# its他diagnosisusepart匹match
for feature_name, keywords in diagnosis_groups.items():
    if feature_name == 'has_preeclampsia':
        continue  # process
    for pid in labels['PID']:
        diag_features[pid][feature_name] = check_diagnosis(pid, keywords, diag_before)

    # displayProgress
    count = sum(diag_features[pid][feature_name] for pid in labels['PID'])
    print(f"  {feature_name}: {count}  ({count/len(labels)*100:.1f}%)")

diag_df = pd.DataFrame.from_dict(diag_features, orient='index')
diag_df.index.name = 'PID'
diag_df = diag_df.reset_index()

features = features.merge(diag_df, on='PID', how='left')

# =====================================================================
# four, 卡方檢定篩選
# =====================================================================
print("\n" + "=" * 70)
print("【four, 卡方檢定篩選】")
print("=" * 70)

# 對binaryfeaturesdo卡方檢定
binary_features = ['is_advanced_maternal_age'] + list(diagnosis_groups.keys())
chi2_results = []

for feat in binary_features:
    if feat not in features.columns:
        continue

    # buildcolumn聯Table
    contingency = pd.crosstab(features[feat], features['pph'])

    # 卡方檢定
    if contingency.shape == (2, 2):
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

        # Computeeachgroup比例
        pph_pos = features[features['pph'] == 1]
        pph_neg = features[features['pph'] == 0]

        rate_pos = pph_pos[feat].mean() * 100 if len(pph_pos) > 0 else 0
        rate_neg = pph_neg[feat].mean() * 100 if len(pph_neg) > 0 else 0

        chi2_results.append({
            'feature': feat,
            'chi2': chi2,
            'p_value': p_value,
            'pph_positive_rate': rate_pos,
            'pph_negative_rate': rate_neg,
            'rate_diff': rate_pos - rate_neg,
            'significant': 'Yes' if p_value < 0.05 else 'No'
        })

chi2_df = pd.DataFrame(chi2_results).sort_values('p_value')

print("\nbinaryfeatures卡方檢定result(p < 0.05 for顯著):")
print("-" * 80)
print(f"{'features':<30} {'p值':<12} {'PPH+比例':<12} {'PPH-比例':<12} {'顯著':<8}")
print("-" * 80)

significant_binary = []
for _, row in chi2_df.iterrows():
    sig_mark = "***" if row['p_value'] < 0.001 else ("**" if row['p_value'] < 0.01 else ("*" if row['p_value'] < 0.05 else ""))
    print(f"{row['feature']:<30} {row['p_value']:<12.4f} {row['pph_positive_rate']:<12.1f} {row['pph_negative_rate']:<12.1f} {sig_mark:<8}")
    if row['p_value'] < 0.05:
        significant_binary.append(row['feature'])

print(f"\n顯著的binaryfeatures: {len(significant_binary)} ")

# 對continuousfeaturesdo t-test
continuous_features = ['age'] + [col for col in features.columns if col.endswith(('_last', '_mean', '_min', '_max', '_count', '_std'))]
ttest_results = []

for feat in continuous_features:
    if feat not in features.columns:
        continue

    pph_pos = features[features['pph'] == 1][feat].dropna()
    pph_neg = features[features['pph'] == 0][feat].dropna()

    if len(pph_pos) > 1 and len(pph_neg) > 1:
        t_stat, p_value = stats.ttest_ind(pph_pos, pph_neg)

        ttest_results.append({
            'feature': feat,
            't_statistic': t_stat,
            'p_value': p_value,
            'pph_positive_mean': pph_pos.mean(),
            'pph_negative_mean': pph_neg.mean(),
            'significant': 'Yes' if p_value < 0.05 else 'No'
        })

ttest_df = pd.DataFrame(ttest_results).sort_values('p_value')

# firstcollect所has顯著features(p < 0.05)
significant_continuous = ttest_df[ttest_df['p_value'] < 0.05]['feature'].tolist()

print(f"\ncontinuousfeatures t-test result(共 {len(significant_continuous)} 顯著, displaybefore 20 ):")
print("-" * 90)
print(f"{'features':<35} {'p值':<12} {'PPH+average':<15} {'PPH-average':<15} {'顯著':<8}")
print("-" * 90)

for _, row in ttest_df.head(20).iterrows():
    sig_mark = "***" if row['p_value'] < 0.001 else ("**" if row['p_value'] < 0.01 else ("*" if row['p_value'] < 0.05 else ""))
    print(f"{row['feature']:<35} {row['p_value']:<12.4f} {row['pph_positive_mean']:<15.2f} {row['pph_negative_mean']:<15.2f} {sig_mark:<8}")

if len(significant_continuous) > 20:
    print(f"... andits他 {len(significant_continuous) - 20} 顯著features")

print(f"\n顯著的continuousfeatures: {len(significant_continuous)} ")

# =====================================================================
# five, Saveresult
# =====================================================================
print("\n" + "=" * 70)
print("【five, Saveresult】")
print("=" * 70)

# Saveall features
features.to_csv('all_features.csv', index=False)
print(f"Save: all_features.csv ({len(features)} records, {len(features.columns)} column)")

# Save統countresult
chi2_df.to_csv('chi2_test_results.csv', index=False)
ttest_df.to_csv('ttest_results.csv', index=False)
print(f"Save: chi2_test_results.csv, ttest_results.csv")

# Save顯著features list
significant_features = {
    'binary_features': significant_binary,
    'continuous_features': significant_continuous,
    'all_significant': significant_binary + significant_continuous
}

import json
with open('significant_features.json', 'w') as f:
    json.dump(significant_features, f, indent=2, ensure_ascii=False)
print(f"Save: significant_features.json")

# =====================================================================
# six, 摘to
# =====================================================================
print("\n" + "=" * 70)
print("【摘to】")
print("=" * 70)
print(f"Totalsamples數: {len(features)}")
print(f"PPH+: {features['pph'].sum()} ({features['pph'].mean()*100:.1f}%)")
print(f"PPH-: {len(features) - features['pph'].sum()} ({(1-features['pph'].mean())*100:.1f}%)")
print(f"\nTotalfeatures數: {len(features.columns) - 2} (扣除 PID, pph)")
print(f"顯著binaryfeatures: {len(significant_binary)} ")
print(f"顯著continuousfeatures: {len(significant_continuous)} ")
print(f"Total顯著features: {len(significant_binary) + len(significant_continuous)} ")

print("\nDone!")
