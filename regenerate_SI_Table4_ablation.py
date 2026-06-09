#!/usr/bin/env python3
"""
Regenerate SI_Table4_Ablation.docx - Ablation study of the pipeline
                                       components.

Promoted from the original SI_Table5 with label updates:
  - 'w/o Stage 2' -> 'w/o Bootstrap LASSO'
  - 'w/o Stage 1' -> 'w/o Statistical Filtering'

Numbers are unchanged (the final model RF+SMOTEENN+Top30+KNN k=1
is the same as before).
"""



from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

SRC = '/Users/yangyongcheng/Desktop/PPH_Prediction_Model-main/PPH_v2_corrected/corrected/tables/SI_Table5_Ablation.docx'
OUT = '/Users/yangyongcheng/Desktop/PPH_joint_search_2026_06_05/07_tables/SI_Table4_Ablation.docx'

# directreadoriginal version,one by onesectionchange label
doc = Document(SRC)

# Replacefunction
REPLACEMENTS = {
    'S5 Table': 'S4 Table',  # ID lowered
    'w/o SMOTEENN': 'w/o SMOTEENN',  # unchanged
    'w/o Bootstrap LASSO': 'w/o Bootstrap LASSO',  # change label
    'w/o Statistical Filtering': 'w/o Statistical Filtering',  # change label
}

# paragraph
print('paragraphchanges:')
for p in doc.paragraphs:
    for run in p.runs:
        for old, new in REPLACEMENTS.items():
            if old in run.text:
                old_text = run.text
                run.text = run.text.replace(old, new)
                print(f'  paragraph: "{old}" -> "{new}"')

# Table
print('\nTablechanges:')
for t in doc.tables:
    for row in t.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                for run in p.runs:
                    for old, new in REPLACEMENTS.items():
                        if old in run.text:
                            run.text = run.text.replace(old, new)
                            print(f'  cell: "{old}" -> "{new}"')

doc.save(OUT)

# verify
d2 = Document(OUT)
print('\n===== changeafterTableContent =====')
for t in d2.tables:
    for r in t.rows:
        print(' | '.join(c.text.strip()[:25] for c in r.cells))

print(f'\n✅ Saved: {OUT}')
