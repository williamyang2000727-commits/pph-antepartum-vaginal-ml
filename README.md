# Antepartum PPH Prediction in Vaginal Deliveries

Source code accompanying the manuscript:

> **"An algorithmic risk prediction model for postpartum hemorrhage prior to labor onset using antepartum features in vaginal deliveries."**
> Yung-Cheng Yang, Lan-Ying Huang, Yen-Wei Chu.
> *Submitted to PLOS Digital Health, 2026.*

This repository contains the full pipeline used to develop and evaluate a machine-learning model that predicts postpartum hemorrhage (PPH) risk **before labor onset**, using **antepartum features only** and restricted to **vaginal deliveries**. The model was trained and evaluated on de-identified electronic health records from Taichung Veterans General Hospital, Taiwan (IRB approval CE25412A).

## Final model

| Configuration | Performance on independent test set (n = 154) |
|---|---|
| **RandomForest + SMOTEENN + Top 30 features + KNN k=1 imputation** | Recall **0.7297** / AUC **0.7169** / MCC **0.3263** / Brier **0.219** |
| Composite score | **0.5645** — ranked **1 / 2,646** unique pipelines |

Composite formula: `0.4 × MCC + 0.3 × AUC + 0.3 × Recall`.

## Data availability

The de-identified electronic health records data are **not publicly shared** due to patient privacy regulations and restrictions imposed by the Institutional Review Board of Taichung Veterans General Hospital. De-identified data may be made available to qualified researchers upon reasonable request to the corresponding author (william.yang2000727@gmail.com) and subject to additional institutional review board approval.

## Repository contents

### Core pipeline (run in order)

| Script | Purpose |
|---|---|
| `01_create_labels.py` | Build PPH labels from ICD-10 diagnoses; restrict cohort to vaginal deliveries |
| `02_feature_engineering.py` | Six-dimension statistical summaries (mean / min / max / std / last / count) on 46 lab items plus 27 binary diagnoses = 304 candidate features |
| `03_feature_selection.py` | Bootstrap LASSO stability selection (100 iterations × 4 C values = 400 fits) |
| `04_train_RF_SMOTEENN_Top30.py` | Train the final RandomForest + SMOTEENN model on Top 30 features with KNN k=1 imputation |
| `05_final_model_evaluation.py` | Evaluate the final model on the independent test set: bootstrap 95% CIs, calibration, decision curve analysis |
| `06_joint_search.py` | Joint search across 7 × 9 × 7 × 6 = **2,646 unique pipelines** via 10-fold stratified cross-validation (26,460 fits) |
| `07_full_test_eval.py` | Re-evaluate every joint-search pipeline on the held-out test set |
| `08_test_eval_top10.py` | Sanity check across the top 10 pipelines |

### Figure / table regeneration scripts

| Script | Output |
|---|---|
| `regenerate_Fig2_pipeline_jointsearch.py` | Main_Fig2_Pipeline.png — pipeline overview |
| `regenerate_Fig3_ROC_jointsearch.py` | Main_Fig3_ROC.png — top 5 pipelines by test composite |
| `regenerate_Table4_jointsearch.py` | Main_Table4_Top10.docx — top 10 joint-search pipelines |
| `regenerate_SI_Table4_ablation.py` | SI_Table4_Ablation.docx — ablation study |
| `regenerate_SI_Table5_consistency.py` | SI_Table5_Consistency.docx — CV vs test consistency |

## Reproducibility

All random states are fixed to `42`. The data partitioning, cross-validation, bootstrap resampling, and class-imbalance routines all use the same seed; results should reproduce exactly under the package versions in `requirements.txt`.

## Installation

```bash
git clone https://github.com/<USERNAME>/pph-antepartum-vaginal-ml.git
cd pph-antepartum-vaginal-ml
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Note: the analysis scripts expect the raw hospital CSVs (not included in this repository) to be placed alongside `01_create_labels.py`. See "Data availability" above.

## Comments

A small number of inline comments contain Chinese-language fragments from the original development. The English README above and the docstrings at the top of each script provide the authoritative description of each component.

## License

MIT — see [LICENSE](LICENSE).

## Citation

```bibtex
@article{Yang2026PPH,
  author  = {Yang, Yung-Cheng and Huang, Lan-Ying and Chu, Yen-Wei},
  title   = {An algorithmic risk prediction model for postpartum hemorrhage prior to labor onset using antepartum features in vaginal deliveries},
  journal = {Submitted to PLOS Digital Health},
  year    = {2026}
}
```

## Contact

Yung-Cheng Yang — william.yang2000727@gmail.com
Corresponding author: Yen-Wei Chu — ywchu@nchu.edu.tw
National Chung Hsing University, Graduate Institute of Genomics and Bioinformatics, Taiwan
