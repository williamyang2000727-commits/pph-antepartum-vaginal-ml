#!/usr/bin/env python3
"""
Regenerate Main_Table4_Top10.docx - Joint Search Top 10 pipelines.

Based on the original Main_Table4 layout, with:
  - One extra column added: Imputation
  - Content updated to the top 10 pipelines from the joint search,
    ranked by test composite score
  - Caption updated to the new main-experiment description
  - Note updated to reference the 2,646 joint search pipelines

Format: Times New Roman 11pt (matches the original docx).
"""



import pandas as pd
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL

# Path
BASE = '/Users/yangyongcheng/Desktop/PPH_joint_search_2026_06_05'
RANKED_CSV = f'{BASE}/02_experiment_csv/full_test_eval_ranked.csv'
OUT = f'{BASE}/07_tables/Main_Table4_Top10.docx'

# Load joint search test eval ranked(by test composite)
df = pd.read_csv(RANKED_CSV)
top10 = df.head(10).copy()
print(f'Top 10 (by test composite):')
print(top10[['imputer','model','imbalance','feature_subset',
              'test_recall','test_spec','test_f1','test_auc','test_mcc','test_composite']].to_string())

# Standardize column display
imputer_display = {
    'KNN_k1': 'KNN k=1', 'KNN_k3': 'KNN k=3', 'KNN_k5': 'KNN k=5',
    'KNN_k7': 'KNN k=7', 'KNN_k10': 'KNN k=10',
    'MICE': 'MICE', 'Median': 'Median',
}
feature_display = {
    'Top8': 'Top 8', 'Top15': 'Top 15', 'Top20': 'Top 20',
    'Top25': 'Top 25', 'Top30': 'Top 30', 'Threshold45': 'Threshold >= 0.45',
}

# create docx
doc = Document()

# Caption
p = doc.add_paragraph()
run = p.add_run('Table 4. Top 10 Pipelines Ranked by Composite Score from Joint Search (2,646 Pipelines)')
run.font.name = 'Times New Roman'
run.font.size = Pt(11)
run.bold = True

# emptysection
doc.add_paragraph()

# Note
p = doc.add_paragraph()
run = p.add_run(
    'Note: Composite Score = 0.4 x MCC + 0.3 x AUC + 0.3 x Recall. '
    'Pipelines were selected from a joint search across 2,646 unique configurations '
    '(7 imputation methods x 9 algorithms x 7 imbalance methods x 6 feature subsets), '
    'each evaluated via 10-fold stratified cross-validation. '
    'All metrics displayn are from the independent test set '
    '(n = 154; 37 PPH+, 117 PPH−).'
)
run.font.name = 'Times New Roman'
run.font.size = Pt(11)

# emptysection
doc.add_paragraph()

# Abbreviations
p = doc.add_paragraph()
run = p.add_run(
    'Abbreviations: AUC, area under the receiver operating characteristic curve; '
    'F1, harmonic mean of precision and recall; '
    'KNN, K-nearest neighbors imputation; '
    'MCC, Matthews correlation coefficient; '
    'MICE, Multivariate Imputation by Chained Equations; '
    'PPH, postpartum hemorrhage; '
    'RandomForest, Random Forest classifier; '
    'SMOTEENN, Synthetic Minority Oversampling Technique with Edited Nearest Neighbors; '
    'SMOTETomek, SMOTE combined with Tomek links removal; '
    'SVM_RBF, support vector machine with radial basis function kernel.'
)
run.font.name = 'Times New Roman'
run.font.size = Pt(11)

# emptysection
doc.add_paragraph()

# createTable 11 rows x 11 cols (header + 10 data)
headers = ['Rank', 'Imputation', 'Algorithm', 'Imbalance\nMethod',
           'Feature\nSubset', 'Recall', 'Specificity', 'F1', 'AUC', 'MCC',
           'Composite\nScore']
table = doc.add_table(rows=11, cols=len(headers))
table.style = 'Light Grid Accent 1'
table.autofit = False

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
for idx, row in enumerate(top10.itertuples(), 1):
    cells = table.rows[idx].cells
    data = [
        str(idx),
        imputer_display.get(row.imputer, row.imputer),
        row.model,
        row.imbalance,
        feature_display.get(row.feature_subset, row.feature_subset),
        f'{row.test_recall:.3f}',
        f'{row.test_spec:.3f}',
        f'{row.test_f1:.3f}',
        f'{row.test_auc:.3f}',
        f'{row.test_mcc:.3f}',
        f'{row.test_composite:.4f}',
    ]
    for i, val in enumerate(data):
        cells[i].text = val
        for p in cells[i].paragraphs:
            for run in p.runs:
                run.font.name = 'Times New Roman'
                run.font.size = Pt(10)
                # Rank 1 addthick
                if idx == 1:
                    run.bold = True
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER

# Save
doc.save(OUT)
print(f'\n✅ Saved: {OUT}')

# verify
d2 = Document(OUT)
print(f'verify:{len(d2.tables[0].rows)} rows x {len(d2.tables[0].columns)} cols')
print(f'  Rank 1 = {d2.tables[0].rows[1].cells[2].text} + {d2.tables[0].rows[1].cells[3].text}')
