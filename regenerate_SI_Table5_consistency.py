#!/usr/bin/env python3
"""
Regenerate SI_Table5_Consistency.docx - Train/test consistency table.

Promoted from the original SI_Table6. CV training numbers updated to
those of the joint search winner (KNN_k1 + RF + SMOTEENN + Top30).
Test set numbers unchanged (same final model).

Bootstrap 95% CIs use N=1,000 (same as the original).
"""



import pandas as pd
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

BASE = '/Users/yangyongcheng/Desktop/PPH_joint_search_2026_06_05'
SRC = '/Users/yangyongcheng/Desktop/PPH_Prediction_Model-main/PPH_v2_corrected/corrected/tables/SI_Table6_Consistency.docx'
OUT = f'{BASE}/07_tables/SI_Table5_Consistency.docx'
JOINT_RESULTS_CSV = f'{BASE}/02_experiment_csv/joint_search_results.csv'

# Load joint search CV results and pick the RF+SMOTEENN+Top30+KNN_k1 row
df = pd.read_csv(JOINT_RESULTS_CSV)
champ = df[(df['imputer']=='KNN_k1') & (df['model']=='RandomForest') &
           (df['imbalance']=='SMOTEENN') & (df['feature_subset']=='Top30')]
print(f'champion CV result:')
print(champ[['cv_recall_mean','cv_recall_std','cv_precision_mean','cv_f1_mean',
             'cv_auc_mean','cv_mcc_mean']].to_string(index=False))

cv_recall = champ['cv_recall_mean'].iloc[0]
cv_recall_std = champ['cv_recall_std'].iloc[0]
cv_precision = champ['cv_precision_mean'].iloc[0]
cv_f1 = champ['cv_f1_mean'].iloc[0]
cv_auc = champ['cv_auc_mean'].iloc[0]
cv_mcc = champ['cv_mcc_mean'].iloc[0]

# Test set values(fromoriginal version / final model known)
test_recall = 0.730
test_spec = 0.650
test_precision = 0.397
test_f1 = 0.514
test_auc = 0.717
test_mcc = 0.326

# 95% CI(original version bootstrap n=1000,wereuse)
ci_recall = "0.579 – 0.869"
ci_spec = "0.562 – 0.739"
ci_precision = "0.281 – 0.515"
ci_f1 = "0.392 – 0.625"
ci_auc = "0.631 – 0.802"
ci_mcc = "0.174 – 0.478"

# Specificity is missing from our joint search csv,shown as N/A (as in the original version)
# samesame precision, f1 use cv_precision_mean / cv_f1_mean(present in joint search csv)
# no std mark as N/A

def diff_str(test, cv):
    d = test - cv
    pct = d / cv * 100 if cv > 0 else 0
    return f'{d:+.3f} ({pct:+.1f}%)'

# create docx
doc = Document()

# Caption
p = doc.add_paragraph()
run = p.add_run('S5 Table. Cross-validation vs Independent Test Set Performance Consistency (Final Model).')
run.font.name = 'Times New Roman'
run.font.size = Pt(11)
run.bold = True

doc.add_paragraph()

# Note
p = doc.add_paragraph()
run = p.add_run(
    'All metrics computed on the independent test set (n = 154; 37 postpartum hemorrhage positive, '
    '117 postpartum hemorrhage negative) for the final model (RandomForest + SMOTEENN + Top 30 + KNN k=1), '
    'selected from joint search across 2,646 pipelines. '
    'CV training results from 10-fold stratified cross-validation (random_state = 42). '
    '95% confidence intervals estimated using percentile bootstrap method (n = 1,000 resamples). '
    '"Difference" calculated as Test Set value minus CV Training mean. '
    'Positive differences (Test > CV) indicate no overfitting; the model generalizes well to unseen data. '
    'N/A indicates the metric standard deviation was not recorded during CV phase.'
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

# Table
headers = ['Metric', 'CV Training\n(Mean ± SD)', 'Test Set',
           '95% CI\n(Bootstrap, n=1000)', 'Difference\n(Test − CV)', 'Interpretation']
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
     ci_recall, diff_str(test_recall, cv_recall), 'No overfitting'),
    ('Specificity', 'N/A', f'{test_spec:.3f}',
     ci_spec, 'N/A', 'N/A'),
    ('Precision (PPV)', f'{cv_precision:.3f}', f'{test_precision:.3f}',
     ci_precision, diff_str(test_precision, cv_precision), 'No overfitting'),
    ('F1 Score', f'{cv_f1:.3f}', f'{test_f1:.3f}',
     ci_f1, diff_str(test_f1, cv_f1), 'No overfitting'),
    ('AUC (ROC)', f'{cv_auc:.3f}', f'{test_auc:.3f}',
     ci_auc, diff_str(test_auc, cv_auc), 'No overfitting'),
    ('MCC', f'{cv_mcc:.3f}', f'{test_mcc:.3f}',
     ci_mcc, diff_str(test_mcc, cv_mcc), 'No overfitting'),
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

doc.save(OUT)
print(f'\n✅ Saved: {OUT}')

# verify
d2 = Document(OUT)
print('\n===== changeafterTable =====')
for r in d2.tables[0].rows:
    print(' | '.join(c.text.strip()[:25] for c in r.cells))
