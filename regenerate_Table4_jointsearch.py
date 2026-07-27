#!/usr/bin/env python3
"""
重新生成 Table3.docx — Joint Search Top 10(by Test composite)

1:1 沿用原 Main_Table4 結構,只:
- 加 1 欄「Imputation」
- 內容改成 joint search 2,646 中 Top 10(by Test composite)
- Caption 改成新主實驗描述
- Note 更新為 joint search 2,646 pipelines

格式:Times New Roman 11pt (跟原 docx 一致)
"""

import pandas as pd
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL

# 路徑（由腳本自身位置推導，整包搬到哪都能跑）
import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANKED_CSV = f'{BASE}/02_experiment_csv/full_test_eval_ranked.csv'
OUT = f'{BASE}/07_tables/Table3.docx'

# 載入 joint search test eval ranked(by test composite)
df = pd.read_csv(RANKED_CSV)
top10 = df.head(10).copy()
print(f'Top 10 (by Test composite):')
print(top10[['imputer','model','imbalance','feature_subset',
              'test_recall','test_spec','test_f1','test_auc','test_mcc','test_composite']].to_string())

# 標準化欄名顯示
imputer_display = {
    'KNN_k1': 'KNN k=1', 'KNN_k3': 'KNN k=3', 'KNN_k5': 'KNN k=5',
    'KNN_k7': 'KNN k=7', 'KNN_k10': 'KNN k=10',
    'MICE': 'MICE', 'Median': 'Median',
}
feature_display = {
    'Top8': 'Top 8', 'Top15': 'Top 15', 'Top20': 'Top 20',
    'Top25': 'Top 25', 'Top30': 'Top 30', 'Threshold45': 'Threshold ≥ 0.45',
}

# 建 docx
doc = Document()

# Caption
p = doc.add_paragraph()
run = p.add_run('Table 3. Top 10 pipelines from joint search ranked by composite score on the independent test set.')
run.font.name = 'Times New Roman'
run.font.size = Pt(11)
run.bold = True

# 空段
doc.add_paragraph()
doc.add_paragraph()

# Note
p = doc.add_paragraph()
run = p.add_run(
    'Note: Composite score = 0.4 × Matthews correlation coefficient + 0.3 × area under the receiver operating characteristic curve + 0.3 × sensitivity. '
    'Pipelines were selected from a joint search across 2,646 unique configurations '
    '(7 imputation methods × 9 algorithm family representatives × 7 class imbalance handling methods × 6 feature subsets), '
    'each evaluated via 10-fold stratified cross-validation and subsequently re-trained on the full training set (n = 615) and evaluated on the independent test set (n = 154). '
    'Specificity and F1 score are omitted from this table to maintain readability within the page margins; '
    'the full set of classification metrics for the final selected model is reported in Table 4.'
)
run.font.name = 'Times New Roman'
run.font.size = Pt(11)

# Abbreviations
p = doc.add_paragraph()
run = p.add_run(
    'Abbreviations: AUC, area under the receiver operating characteristic curve; '
    'KNN, K-nearest neighbors imputation; '
    'MCC, Matthews correlation coefficient; '
    'MICE, Multivariate Imputation by Chained Equations; '
    'PPH, postpartum hemorrhage; '
    'SMOTEENN, Synthetic Minority Oversampling Technique with Edited Nearest Neighbors; '
    'SVM_RBF, support vector machine with radial basis function kernel.'
)
run.font.name = 'Times New Roman'
run.font.size = Pt(11)

# 空段

# 建表 11 rows × 11 cols (header + 10 data)
headers = ['Rank', 'Imputation', 'Algorithm', 'Imbalance',
           'Subset', 'Recall', 'AUC', 'MCC',
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
        imputer_display.get(row.imputer, row.imputer).replace(' ', '\xa0'),
        row.model,
        row.imbalance,
        feature_display.get(row.feature_subset, row.feature_subset).replace(' ', '\xa0'),
        f'{row.test_recall:.3f}',
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
                # Rank 1 加粗
                if idx == 1:
                    run.bold = True
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER

# 存
doc.save(OUT)
print(f'\n✅ 已存: {OUT}')

# 驗證
d2 = Document(OUT)
print(f'驗證:{len(d2.tables[0].rows)} rows × {len(d2.tables[0].columns)} cols')
print(f'  Rank 1 = {d2.tables[0].rows[1].cells[2].text} + {d2.tables[0].rows[1].cells[3].text}')
