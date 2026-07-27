# 🚨 2026-07-25 未完成同步警告(勿直接重跑覆蓋 S5_table.docx):
#    ① 本腳本產出的 Note 與現行實體檔有落差。
#    ② 更重要:Note 內「Positive differences (Test > CV) indicate no overfitting;
#       the model generalizes well to unseen data.」這句論證方向錯誤(Option B 已於
#       正文移除),實體 S5_table.docx 也應改為「CV 為保守下界、test 為樂觀上界」的說法,
#       且 Interpretation 欄的 "No overfitting" 需一併重擬。修好前勿重跑。
#!/usr/bin/env python3
"""
重新生成 S5_table.docx — 從原 SI_Table6 沿用 + 升格 + 換冠軍 CV 數據

1:1 沿用原版表格結構,只:
- 編號 S6 → S5
- CV Training 數字改為 joint search 中冠軍 (KNN_k1 + RF + SMOTEENN + Top30) 的 CV
- Test set 數字不變(同一個 final model)
"""

import pandas as pd
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ✅ 2026-07-25 已修復並可重跑:OUT 指向現行 07_tables/S5_table.docx;
#    test 值改由 full_test_eval_ranked.csv 讀全精度(原為四捨五入硬編碼,導致 Recall 顯示
#    +14.1% 與正文 +14.0% 對不上);移除 Interpretation 欄的 "No overfitting"
#    (正向落差主要來自「以 test set 選冠軍」的樂觀偏誤,正文已更正,表格同步)。
OUT = f'{BASE}/07_tables/S5_table.docx'
JOINT_RESULTS_CSV = f'{BASE}/02_experiment_csv/joint_search_results.csv'

# 載入 joint search CV results,抓 RF+SMOTEENN+Top30+KNN_k1 那行
df = pd.read_csv(JOINT_RESULTS_CSV)
champ = df[(df['imputer']=='KNN_k1') & (df['model']=='RandomForest') &
           (df['imbalance']=='SMOTEENN') & (df['feature_subset']=='Top30')]
print(f'冠軍 CV 結果:')
print(champ[['cv_recall_mean','cv_recall_std','cv_precision_mean','cv_f1_mean',
             'cv_auc_mean','cv_mcc_mean']].to_string(index=False))

cv_recall = champ['cv_recall_mean'].iloc[0]
cv_recall_std = champ['cv_recall_std'].iloc[0]
cv_precision = champ['cv_precision_mean'].iloc[0]
cv_f1 = champ['cv_f1_mean'].iloc[0]
cv_auc = champ['cv_auc_mean'].iloc[0]
cv_mcc = champ['cv_mcc_mean'].iloc[0]

# Test set 值:改由 full_test_eval_ranked.csv 讀全精度(勿硬編碼四捨五入值)
_te = pd.read_csv(f'{BASE}/02_experiment_csv/full_test_eval_ranked.csv')
_t = _te[(_te['imputer']=='KNN_k1') & (_te['model']=='RandomForest') &
         (_te['imbalance']=='SMOTEENN') & (_te['feature_subset']=='Top30')].iloc[0]
test_recall    = _t['test_recall']
test_spec      = _t['test_spec']
test_precision = _t['test_precision']
test_f1        = _t['test_f1']
test_auc       = _t['test_auc']
test_mcc       = _t['test_mcc']

# 95% CI(原版 bootstrap n=1000,我們沿用)
ci_recall = "0.579 – 0.869"
ci_spec = "0.562 – 0.739"
ci_precision = "0.281 – 0.515"
ci_f1 = "0.392 – 0.625"
ci_auc = "0.631 – 0.802"
ci_mcc = "0.174 – 0.478"

# Specificity 在我們的 joint search csv 沒有 cv_specificity,顯示 N/A 沿用原版
# 同樣 precision, f1 用 cv_precision_mean / cv_f1_mean(joint search csv 有)
# 沒 std 就標 N/A

def diff_str(test, cv):
    d = test - cv
    pct = d / cv * 100 if cv > 0 else 0
    return f'{d:+.3f} ({pct:+.1f}%)'

# 建 docx
doc = Document()

# Caption
p = doc.add_paragraph()
# 🚨 2026-07-27：此 caption 必須與正式稿要用的 SI caption 逐字相同（sentence case）。
#    舊版是 Title Case 的 'Cross-validation vs Independent Test Set Performance Consistency (Final Model).'，
#    2026-07-26 已在 docx 手動改成下面這句，但腳本沒同步 → 2026-07-27 重跑時被跑回舊版，已在此修正。
run = p.add_run('S5 Table. Cross-validation and independent test set performance of the final model.')
run.font.name = 'Times New Roman'
run.font.size = Pt(11)
run.bold = True

doc.add_paragraph()

# Note
headers = ['Metric', 'CV Training\n(Mean ± SD)', 'Test Set',
           '95% CI\n(Bootstrap, n=1000)', 'Difference\n(Test − CV)']
table = doc.add_table(rows=7, cols=len(headers))
table.style = 'Light Grid Accent 1'

# Header
hdr = table.rows[0].cells
for i, h in enumerate(headers):
    hdr[i].text = h
    for p in hdr[i].paragraphs:
        for run in p.runs:
            run.font.name = 'Times New Roman'
            run.font.size = Pt(10)
            run.bold = True
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER

# Data
rows_data = [
    ('Recall (Sensitivity)', f'{cv_recall:.3f} ± {cv_recall_std:.3f}', f'{test_recall:.3f}',
     ci_recall, diff_str(test_recall, cv_recall)),
    ('Specificity', 'N/A', f'{test_spec:.3f}',
     ci_spec, 'N/A'),
    ('Precision (PPV)', f'{cv_precision:.3f}', f'{test_precision:.3f}',
     ci_precision, diff_str(test_precision, cv_precision)),
    ('F1 Score', f'{cv_f1:.3f}', f'{test_f1:.3f}',
     ci_f1, diff_str(test_f1, cv_f1)),
    ('AUC (ROC)', f'{cv_auc:.3f}', f'{test_auc:.3f}',
     ci_auc, diff_str(test_auc, cv_auc)),
    ('MCC', f'{cv_mcc:.3f}', f'{test_mcc:.3f}',
     ci_mcc, diff_str(test_mcc, cv_mcc)),
]

for ri, row_data in enumerate(rows_data, 1):
    cells = table.rows[ri].cells
    for ci, val in enumerate(row_data):
        cells[ci].text = val
        for p in cells[ci].paragraphs:
            for run in p.runs:
                run.font.name = 'Times New Roman'
                run.font.size = Pt(10)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER

p = doc.add_paragraph()
run = p.add_run(
    # 'Notes: ' 標籤是 2026-07-26 為了與 S1-S4 體例一致而加的，腳本原本漏了，2026-07-27 補回
    'Notes: Test-set metrics computed on the independent test set (n = 154; 37 postpartum hemorrhage positive, '
    '117 postpartum hemorrhage negative) for the final model (RandomForest + SMOTEENN + Top 30 + KNN k=1), '
    'selected from joint search across 2,646 pipelines. '
    'CV training results from 10-fold stratified cross-validation (random_state = 42). '
    '95% confidence intervals estimated using percentile bootstrap method (n = 1,000 resamples). '
    '"Difference" calculated as Test Set value minus CV Training mean. '
    'All differences are positive, which reflects two factors: the final model was trained on the complete '
    'training set (n = 615) whereas each cross-validation fold used approximately 554 samples, and the final '
    'model was selected by composite score on this same independent test set, which biases the test-set '
    'metrics upward. Cross-validation values should therefore be read as a conservative lower bound and '
    'test-set values as an optimistic upper bound on generalization performance. '
    'N/A for specificity indicates that specificity was not among the metrics recorded during the '
    'cross-validation phase; standard deviations were recorded only for recall.'
)
run.font.name = 'Times New Roman'
run.font.size = Pt(11)

doc.add_paragraph()

p = doc.add_paragraph()
run = p.add_run(
    'Abbreviations: CV, cross-validation; MCC, Matthews correlation coefficient; '
    'PPV, positive predictive value; ROC, receiver operating characteristic; SD, standard deviation.'
)
run.font.name = 'Times New Roman'
run.font.size = Pt(11)

doc.add_paragraph()

# 表

doc.save(OUT)
print(f'\n✅ 已存: {OUT}')

# 驗證
d2 = Document(OUT)
print('\n===== 改後表格 =====')
for r in d2.tables[0].rows:
    print(' | '.join(c.text.strip()[:25] for c in r.cells))
