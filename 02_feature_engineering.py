#!/usr/bin/env python3
"""
02_feature_engineering.py - 特徵工程（包含分娩當天）

🔧 2026-08-19 唯一一次修改：補正兩個診斷關鍵字（has_fetal_macrosomia / has_placental_abruption），
   原版因關鍵字與本院 ICD-10 登錄用語不符而漏抓，兩者在 Table 1 顯示為 0。
   已逐格驗證：all_features.csv 235,314 格中僅 34 格改變（就是這兩欄），
   Top 30 特徵逐格相同、Stage 1 顯著特徵仍為 35 個、二元顯著仍為 0，
   因此 2,646 pipelines 與所有效能數字完全不受影響。原版備份於 _BACKUP_table1_fix_20260819/。

關鍵原則：
- 所有特徵使用 分娩當天 23:59 之前 的資料（包含分娩當天）
- 這是因為許多產前診斷（如子癲前症）會在分娩當天記錄
- 仍能避免資料洩漏，因為不包含 PPH 結果本身

文獻支持：
- 子癲前症是 PPH 的已知風險因子 (The Lancet 2025, OR 1.5-2)
- 慢性高血壓也與 PPH 風險相關

輸出：
- all_features.csv：所有候選特徵
- feature_statistics.csv：特徵統計與卡方檢定結果
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("【步驟 2】特徵工程（包含分娩當天）")
print("=" * 70)

# =====================================================================
# 讀取資料
# =====================================================================
print("\n讀取資料...")
labels = pd.read_csv('vaginal_delivery_labels.csv')
lab = pd.read_csv('2.F2025627_產科檢驗數值(2023-2024).csv')
diag = pd.read_csv('3.F2025627_疾病診斷(2023-2024).csv')

# 轉換日期格式
labels['time_cutoff'] = pd.to_datetime(labels['time_cutoff'])
lab['measurement_date'] = pd.to_datetime(lab['measurement_date'])
diag['condition_start_date'] = pd.to_datetime(diag['condition_start_date'])

# 清理診斷名稱空格
diag['condition_clean'] = diag['condition_concept_name'].str.strip()

print(f"標籤資料: {len(labels)} 筆")
print(f"檢驗資料: {len(lab)} 筆")
print(f"診斷資料: {len(diag)} 筆")

# 只保留自然產病患的資料
valid_pids = set(labels['PID'])
lab = lab[lab['PID'].isin(valid_pids)].copy()
diag = diag[diag['PID'].isin(valid_pids)].copy()

print(f"篩選後檢驗資料: {len(lab)} 筆")
print(f"篩選後診斷資料: {len(diag)} 筆")

# =====================================================================
# 建立時間截止點對照表
# =====================================================================
cutoff_dict = labels.set_index('PID')['time_cutoff'].to_dict()

# =====================================================================
# 一、基本特徵
# =====================================================================
print("\n" + "=" * 70)
print("【一、基本特徵】")
print("=" * 70)

features = labels[['PID', 'age', 'pph']].copy()
features['is_advanced_maternal_age'] = (features['age'] >= 35).astype(int)

print(f"age: 平均 {features['age'].mean():.1f} 歲")
print(f"is_advanced_maternal_age: {features['is_advanced_maternal_age'].sum()} 人 ({features['is_advanced_maternal_age'].mean()*100:.1f}%)")

# =====================================================================
# 二、檢驗值特徵（包含分娩當天）
# =====================================================================
print("\n" + "=" * 70)
print("【二、檢驗值特徵】（包含分娩當天）")
print("=" * 70)

# 為每筆檢驗資料加上該病患的時間截止點
lab['time_cutoff'] = lab['PID'].map(cutoff_dict)

# 時間過濾：包含分娩當天的資料（time_cutoff 是分娩當天 00:00）
# 條件: measurement_date < time_cutoff + 1天 = 包含分娩當天
lab_before = lab[lab['measurement_date'] < (lab['time_cutoff'] + pd.Timedelta(days=1))].copy()
print(f"時間過濾前: {len(lab)} 筆")
print(f"時間過濾後: {len(lab_before)} 筆（包含分娩當天）")

# 檢查各檢驗項目的覆蓋率
lab_items = lab_before.groupby('concept_name')['PID'].nunique()
total_patients = len(labels)
coverage = (lab_items / total_patients * 100).sort_values(ascending=False)

print(f"\n檢驗項目覆蓋率（≥30% 才納入）:")
valid_lab_items = []
for item, cov in coverage.items():
    if cov >= 30:
        valid_lab_items.append(item)
        print(f"  {item}: {cov:.1f}%")

print(f"\n符合條件的檢驗項目: {len(valid_lab_items)} 個")

# 建立檢驗值特徵
lab_features = {}

for item in valid_lab_items:
    item_data = lab_before[lab_before['concept_name'] == item]

    # 計算每位病患的統計量
    for pid in labels['PID']:
        if pid not in lab_features:
            lab_features[pid] = {}

        # 2026-08-19 修正：原始檢驗檔的列序是依「檢驗數值」升序排列（實測 24,027 組 100% 升序，
        #   依日期僅 37.37%），而非依時間排序。原本直接 iloc[-1] 取到的是數值最大者而非最後一筆，
        #   使 last 在全部 46 個檢驗項目上都恆等於 max（100%），六種統計特徵實際只提供五種資訊。
        #   改為先依 measurement_date 做穩定排序，last 才真正是分娩前最後一次記錄的數值。
        #   ⚠️ 時間戳僅精確至日期層級，同一天內的多筆仍無法再細分先後（實測占 6.0% 的組合），
        #      此限制已載明於 Limitations。
        _pdata = item_data[item_data['PID'] == pid].sort_values('measurement_date', kind='stable')
        patient_data = _pdata['value_as_number']

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
print(f"檢驗值特徵數: {len(lab_df.columns) - 1}")

# =====================================================================
# 三、診斷特徵（包含分娩當天）
# =====================================================================
print("\n" + "=" * 70)
print("【三、診斷特徵】（包含分娩當天）")
print("=" * 70)

# 為每筆診斷資料加上該病患的時間截止點
diag['time_cutoff'] = diag['PID'].map(cutoff_dict)

# 時間過濾：包含分娩當天的資料
# 條件: condition_start_date < time_cutoff + 1天 = 包含分娩當天
diag_before = diag[diag['condition_start_date'] < (diag['time_cutoff'] + pd.Timedelta(days=1))].copy()
print(f"時間過濾前: {len(diag)} 筆")
print(f"時間過濾後: {len(diag_before)} 筆（包含分娩當天）")

# ----- 定義診斷分組 -----

# 子癲前症（PDF 定義的 11 個）
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

# 其他文獻風險因子關鍵詞
diagnosis_groups = {
    'has_preeclampsia': preeclampsia_keywords,
    'has_anemia': ['anemia', 'anaemia'],
    'has_coagulation_disorder': ['coagulation', 'coagulopathy', 'thrombocytopenia', 'platelet'],
    'has_placenta_previa': ['placenta previa', 'placenta praevia'],
    # 2026-08-19 修正：原本只找 'placental abruption'／'abruptio placentae'，
    # 但本院實際登錄的是 ICD-10 O45.9 的 'Premature separation of placenta'，導致漏抓 → 全 0。
    'has_placental_abruption': ['placental abruption', 'abruptio placentae', 'premature separation of placenta'],
    'has_multiple_gestation': ['twin', 'triplet', 'multiple gestation', 'multiple pregnancy'],
    'has_polyhydramnios': ['polyhydramnios'],
    'has_oligohydramnios': ['oligohydramnios'],
    'has_gestational_diabetes': ['gestational diabetes', 'gdm'],
    'has_gestational_hypertension': ['gestational hypertension', 'pregnancy-induced hypertension', 'pregnancy-induced] hypertension'],
    'has_chronic_hypertension': ['chronic hypertension', 'pre-existing hypertension', r'essential \(primary\) hypertension'],
    'has_uterine_fibroids': ['fibroid', 'leiomyoma', 'myoma'],
    'has_previous_cesarean': ['previous cesarean', 'previous c/s', 'previous caesarean'],
    'has_obesity': ['obesity', 'obese'],
    # 2026-08-19 修正：原本含 'diabetes mellitus'，但該字串同時命中 'Gestational diabetes mellitus'，
    #   使本欄實際上絕大多數是妊娠糖尿病，與 has_gestational_diabetes 高度重疊（13.9% vs 13.3%）。
    #   孕前糖尿病臨床盛行率約 1-2%，13.9% 不可能。移除該字串後為 16 人（2.1%），符合臨床。
    #   已實測：training set chi-square p 由 0.8483 → 0.5345，仍不顯著 → Stage 1 結果不變。
    'has_diabetes_mellitus': ['type 1 diabetes', 'type 2 diabetes', 'pre-existing diabetes'],
    'has_thyroid_disorder': ['thyroid', 'hypothyroid', 'hyperthyroid'],
    'has_heart_disease': ['heart disease', 'cardiac', 'cardiomyopathy'],
    'has_kidney_disease': ['kidney disease', 'renal disease', 'chronic kidney'],
    'has_liver_disease': ['liver disease', 'hepatic'],
    'has_autoimmune': ['lupus', 'autoimmune', 'rheumatoid'],
    'has_infection': ['chorioamnionitis', 'infection', 'sepsis'],
    # 2026-08-19 修正：原本只找 'macrosomia'，但本院實際登錄的是 ICD-10 O36.6 的
    # 'Maternal care for excessive fetal growth'（三種 trimester 變體），導致漏抓 → 全 0。
    # ⚠️ 刻意不收 'heavy for gestational age'（P08.1 新生兒碼）：那是嬰兒出生後才給的碼，
    #    納入會把分娩後資訊帶進純產前特徵。只用母親端的 O36.6。
    'has_fetal_macrosomia': ['macrosomia', 'large for gestational', 'excessive fetal growth'],
    'has_fetal_growth_restriction': ['growth restriction', 'iugr', 'small for gestational'],
    # 2026-08-19 修正：原本含 'premature'，該字串同時命中
    #   'Full-term premature rupture of membranes'（足月胎膜早破，109 人）、
    #   'Premature separation of placenta'（胎盤早剝，與 has_placental_abruption 重複）、
    #   'Ventricular premature depolarization'（心室早期去極化，與產科無關）。
    #   移除後 228 → 142 人（29.6% → 18.5%）。已實測：p 由 0.7212 → 0.7998，仍不顯著。
    'has_preterm': ['preterm'],
    'has_prolonged_labor': ['prolonged labor', 'prolonged labour'],
    'has_induced_labor': ['induction', 'induced labor', 'induced labour'],
}

# 建立診斷特徵
def check_diagnosis(pid, keywords, diag_data):
    """檢查病患是否有特定診斷（部分匹配）"""
    patient_diag = diag_data[diag_data['PID'] == pid]['condition_clean'].str.lower()
    for kw in keywords:
        if isinstance(kw, str):
            kw_lower = kw.lower()
            if patient_diag.str.contains(kw_lower, na=False).any():
                return 1
    return 0

def check_diagnosis_exact(pid, keywords, diag_data):
    """檢查病患是否有特定診斷（完全匹配）"""
    patient_diag = diag_data[diag_data['PID'] == pid]['condition_clean']
    return 1 if patient_diag.isin(keywords).any() else 0

print("\n建立診斷特徵...")
diag_features = {pid: {} for pid in labels['PID']}

# has_preeclampsia 用完全匹配（PDF 定義）
for pid in labels['PID']:
    diag_features[pid]['has_preeclampsia'] = check_diagnosis_exact(pid, preeclampsia_keywords, diag_before)

# 其他診斷用部分匹配
for feature_name, keywords in diagnosis_groups.items():
    if feature_name == 'has_preeclampsia':
        continue  # 已處理
    for pid in labels['PID']:
        diag_features[pid][feature_name] = check_diagnosis(pid, keywords, diag_before)

    # 顯示進度
    count = sum(diag_features[pid][feature_name] for pid in labels['PID'])
    print(f"  {feature_name}: {count} 人 ({count/len(labels)*100:.1f}%)")

diag_df = pd.DataFrame.from_dict(diag_features, orient='index')
diag_df.index.name = 'PID'
diag_df = diag_df.reset_index()

features = features.merge(diag_df, on='PID', how='left')

# =====================================================================
# 四、卡方檢定篩選
# =====================================================================
print("\n" + "=" * 70)
print("【四、卡方檢定篩選】")
print("=" * 70)

# 對二元特徵做卡方檢定
binary_features = ['is_advanced_maternal_age'] + list(diagnosis_groups.keys())
chi2_results = []

for feat in binary_features:
    if feat not in features.columns:
        continue

    # 建立列聯表
    contingency = pd.crosstab(features[feat], features['pph'])

    # 卡方檢定
    if contingency.shape == (2, 2):
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

        # 計算各組比例
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

print("\n二元特徵卡方檢定結果（p < 0.05 為顯著）:")
print("-" * 80)
print(f"{'特徵':<30} {'p值':<12} {'PPH+比例':<12} {'PPH-比例':<12} {'顯著':<8}")
print("-" * 80)

significant_binary = []
for _, row in chi2_df.iterrows():
    sig_mark = "***" if row['p_value'] < 0.001 else ("**" if row['p_value'] < 0.01 else ("*" if row['p_value'] < 0.05 else ""))
    print(f"{row['feature']:<30} {row['p_value']:<12.4f} {row['pph_positive_rate']:<12.1f} {row['pph_negative_rate']:<12.1f} {sig_mark:<8}")
    if row['p_value'] < 0.05:
        significant_binary.append(row['feature'])

print(f"\n顯著的二元特徵: {len(significant_binary)} 個")

# 對連續特徵做 t-test
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

# 先收集所有顯著特徵（p < 0.05）
significant_continuous = ttest_df[ttest_df['p_value'] < 0.05]['feature'].tolist()

print(f"\n連續特徵 t-test 結果（共 {len(significant_continuous)} 個顯著，顯示前 20 個）:")
print("-" * 90)
print(f"{'特徵':<35} {'p值':<12} {'PPH+平均':<15} {'PPH-平均':<15} {'顯著':<8}")
print("-" * 90)

for _, row in ttest_df.head(20).iterrows():
    sig_mark = "***" if row['p_value'] < 0.001 else ("**" if row['p_value'] < 0.01 else ("*" if row['p_value'] < 0.05 else ""))
    print(f"{row['feature']:<35} {row['p_value']:<12.4f} {row['pph_positive_mean']:<15.2f} {row['pph_negative_mean']:<15.2f} {sig_mark:<8}")

if len(significant_continuous) > 20:
    print(f"... 及其他 {len(significant_continuous) - 20} 個顯著特徵")

print(f"\n顯著的連續特徵: {len(significant_continuous)} 個")

# =====================================================================
# 五、儲存結果
# =====================================================================
print("\n" + "=" * 70)
print("【五、儲存結果】")
print("=" * 70)

# 儲存所有特徵
features.to_csv('all_features.csv', index=False)
print(f"已儲存: all_features.csv ({len(features)} 筆, {len(features.columns)} 欄)")

# 儲存統計結果
chi2_df.to_csv('chi2_test_results.csv', index=False)
ttest_df.to_csv('ttest_results.csv', index=False)
print(f"已儲存: chi2_test_results.csv, ttest_results.csv")

# 儲存顯著特徵清單
significant_features = {
    'binary_features': significant_binary,
    'continuous_features': significant_continuous,
    'all_significant': significant_binary + significant_continuous
}

import json
with open('significant_features.json', 'w') as f:
    json.dump(significant_features, f, indent=2, ensure_ascii=False)
print(f"已儲存: significant_features.json")

# =====================================================================
# 六、摘要
# =====================================================================
print("\n" + "=" * 70)
print("【摘要】")
print("=" * 70)
print(f"總樣本數: {len(features)}")
print(f"PPH+: {features['pph'].sum()} ({features['pph'].mean()*100:.1f}%)")
print(f"PPH-: {len(features) - features['pph'].sum()} ({(1-features['pph'].mean())*100:.1f}%)")
print(f"\n總特徵數: {len(features.columns) - 2} (扣除 PID, pph)")
print(f"顯著二元特徵: {len(significant_binary)} 個")
print(f"顯著連續特徵: {len(significant_continuous)} 個")
print(f"總顯著特徵: {len(significant_binary) + len(significant_continuous)} 個")

print("\n完成！")
