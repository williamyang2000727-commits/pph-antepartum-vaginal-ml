# Antepartum PPH Prediction in Vaginal Deliveries

Source code accompanying the manuscript:

> **"An algorithmic risk prediction model for postpartum hemorrhage prior to
> delivery using antepartum features in vaginal deliveries."**
> Yung-Cheng Yang, Lan-Ying Huang, Yen-Wei Chu.
> *Submitted to PLOS Digital Health, 2026.*

This repository contains the pipeline used to develop and evaluate a
machine-learning model that predicts postpartum hemorrhage (PPH) risk **before
delivery**, using **antepartum features only** and restricted to **vaginal
deliveries**. The model was developed on de-identified electronic health records
from Taichung Veterans General Hospital, Taiwan (institutional review board
approval CE25412A).

The scripts here are the ones that produced the numbers reported in the
manuscript; they are provided for methodological transparency and inspection.
See **Data availability** below — the underlying records cannot be shared, so
the pipeline cannot be executed end-to-end from this repository alone.

## Final model

| Configuration | Performance on the independent test set (n = 154) |
|---|---|
| **RandomForest + SMOTEENN + Top 30 features + KNN k=1 imputation** | Sensitivity **0.7297** / AUC **0.7169** / MCC **0.3263** / Brier **0.2192** |
| Confusion matrix | TP 27 / FP 41 / FN 10 / TN 76 |
| Composite score | **0.5645** — ranked **1 / 2,646** unique pipelines |

Composite score = `0.4 × MCC + 0.3 × AUC + 0.3 × sensitivity`, computed on the
independent test set. The same pipeline ranked **92 / 2,646** on the
cross-validation composite, which played no part in its selection.

Because the final model was selected by its test-set composite score, the
test-set metrics are an **optimistic bound**, not an unbiased estimate of
generalization. The manuscript reports this explicitly and quantifies it: the
winning pipeline's test-minus-cross-validation composite gap is +0.1149 against
a mean of +0.0069 across all 2,646 pipelines (16.6×, the 95th percentile), and
cross-validation AUC correlates only weakly with test AUC (Spearman
ρ = 0.170). The conservative lower bound is the cross-validation AUC of 0.6210.

## Data availability

The de-identified electronic health records are **not publicly shared**, because
of patient privacy regulations and restrictions imposed by the institutional
review board of Taichung Veterans General Hospital. De-identified data may be
made available to qualified researchers on reasonable request to the
corresponding author (ywchu@nchu.edu.tw), subject to additional institutional
review board approval.

Consequently no data file is included here — neither the four raw hospital
exports, nor the derived `all_features.csv`, which holds individual-level
clinical values for all 769 patients.

## Expected directory layout

Scripts `04` through `08` resolve their input and output directories from the
**parent** of the directory they live in:

```python
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
```

So place them one level below the project root:

```
project_root/
├── scripts/                  <- the .py files in this repository
├── 01_KEY_RESULTS/           <- model artefacts written by 04 and 05
└── 02_experiment_csv/        <- all_features.csv and joint-search outputs
```

`01_create_labels.py`, `02_feature_engineering.py` and `03_feature_selection.py`
instead read and write plain filenames in the **current working directory**, so
run them from the directory holding the raw exports.

### Input files expected by 01 and 02

The two cohort-building scripts refer to the hospital's export filenames
verbatim. Rename your own files to match, or edit the `read_csv` calls near the
top of each script:

| Script | Expects |
|---|---|
| `01_create_labels.py` | `1.F2025627_2024產婦基本資料N1239.csv` (maternal baseline), `3.F2025627_疾病診斷(2023-2024).csv` (diagnoses), `16.F2025627_產婦手術術式(2024).csv` (delivery procedures) |
| `02_feature_engineering.py` | `vaginal_delivery_labels.csv` (written by `01`), `2.F2025627_產科檢驗數值(2023-2024).csv` (laboratory and vital-sign values), `3.F2025627_疾病診斷(2023-2024).csv` |

## Repository contents

### Cohort construction and feature engineering

| Script | Purpose |
|---|---|
| `01_create_labels.py` | Derive PPH labels from ICD-10 diagnoses and restrict the cohort to vaginal deliveries (1,239 deliveries, minus 445 cesarean and 25 fetal-demise cases, giving **769**). Writes `vaginal_delivery_labels.csv` |
| `02_feature_engineering.py` | Six statistical summaries (mean / minimum / maximum / standard deviation / last recorded value / measurement count) for each of 46 laboratory and vital-sign items, plus maternal age, giving 277 continuous features; plus 27 binary features (26 ICD-10 diagnosis flags and one age-derived flag) = **304 candidate features**. Writes `all_features.csv` |

### Canonical model pipeline

| Script | Purpose |
|---|---|
| `04_train_RF_SMOTEENN_Top30.py` | **The canonical feature-selection and training script.** Stratified 80/20 split, then train-only chi-square and t-test filtering, train-only Bootstrap LASSO stability selection (100 resamples × 4 regularization parameters = 400 fits), KNN k=1 imputation, standardization, SMOTEENN, RandomForest. Checks its own output against the reported metrics |
| `06_joint_search.py` | Joint search over 7 × 9 × 7 × 6 = **2,646 unique pipelines** by 10-fold stratified cross-validation (**26,460 fits**) |
| `07_full_test_eval.py` | Retrain all 2,646 pipelines on the full training set and evaluate each on the independent test set |
| `05_final_model_evaluation.py` | Retrain the final model and write the confusion matrix, receiver operating characteristic curve, mean-decrease-in-impurity importances and a four-panel summary figure |
| `08_test_eval_top10.py` | Cross-check the ten best cross-validation pipelines on the test set |

Imputation and standardization are fitted once on the full training set
(n = 615), **outside** the cross-validation loop; class-imbalance resampling is
the step confined to each training fold, with the validation fold left
untouched. Feature ranking is likewise computed on the training set only and
loaded outside the loop. The code is the authority on this: in
`06_joint_search.py` the imputer and scaler are fitted before the fold loop
begins.

### Not used for the published results

| Script | Why it is kept |
|---|---|
| `03_feature_selection.py` | An earlier stability-selection variant. It applies a fixed 0.6 selection-frequency threshold, stratified bootstrap resampling, `class_weight='balanced'`, `C = [0.01, 0.1, 0.5, 1.0]`, and the whole cohort with no train/test split. **It does not produce the published 30 features.** Those come from `04_train_RF_SMOTEENN_Top30.py`, which uses unstratified resampling, no class weighting, `C = [0.01, 0.1, 1.0, 10.0]`, and the 615 training cases only, then ranks features by selection frequency and takes the top N. Re-running both variants reproduces the published selection counts exactly (35 of 35) for the `04` procedure and not at all (0 of 35) for this one. Retained for provenance; do not use it to reproduce the published feature set |

### Figure and table generation

| Script | Output |
|---|---|
| `regenerate_Fig2_pipeline_jointsearch.py` | `06_figures/Fig2.png` — pipeline overview |
| `regenerate_Fig3_ROC_jointsearch.py` | `06_figures/Fig3.png` — receiver operating characteristic curves of the five best pipelines |
| `regenerate_Table4_jointsearch.py` | `07_tables/Table3.docx` — the ten best joint-search pipelines (the script name retains an earlier table number) |
| `regenerate_SI_Table5_consistency.py` | `07_tables/S5_table.docx` — cross-validation versus independent test set |

Bootstrap confidence intervals, the calibration curve, the decision curve
analysis, the Shapley additive explanations plots and the remaining supporting
tables were produced by separate scripts that are not included here; they are
available from the authors on request.

## Reproducibility

Every random state is fixed to `42` — data partitioning, cross-validation,
bootstrap resampling and class-imbalance routines all use the same seed. Given
`all_features.csv`, `04_train_RF_SMOTEENN_Top30.py` reproduces the reported
confusion matrix (TP 27 / FP 41 / FN 10 / TN 76) and the 400-fit selection
counts exactly; the script asserts this itself, to a tolerance of 0.001.

Two caveats worth knowing before re-deriving anything:

1. **Tied selection counts.** Nine pairs of features share an identical
   selection count, and the default pandas sort is not stable, so regenerating
   the ranking table from scratch can swap tied features. The Top 30 boundary is
   unaffected (30th = 156 selections, 31st = 143), as are the Top 8, Top 20 and
   Top 25 boundaries — but the **Top 15 boundary is a tie at 212 selections**, so
   that subset can differ by one feature between regenerations.

2. **The stored ranking table.** Scripts `05` through `08` read a stored
   selection-frequency table, `bootstrap_selection_frequency_clean.csv`. No
   script in this repository writes that filename: `04` computes the ranking in
   memory and uses it without saving, and `03` writes a differently named file
   from a different variant of the procedure. To regenerate the table, run the
   Stage 1 and Stage 2 sections of `04_train_RF_SMOTEENN_Top30.py` and export
   its frequency DataFrame.

## Installation

```bash
git clone https://github.com/williamyang2000727-commits/pph-antepartum-vaginal-ml.git
cd pph-antepartum-vaginal-ml
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The experiments were run on **Python 3.13.5**. `requirements.txt` pins the
versions reported in the manuscript for the packages that affect numerical
results; the plotting and document packages are left unpinned, because they do
not.

## Comments

Some inline comments and console messages are in Traditional Chinese, from the
original development. This README and the docstring at the top of each script
give the authoritative English description of every component.

## License

MIT — see [LICENSE](LICENSE).

## Citation

```bibtex
@article{Yang2026PPH,
  author  = {Yang, Yung-Cheng and Huang, Lan-Ying and Chu, Yen-Wei},
  title   = {An algorithmic risk prediction model for postpartum hemorrhage prior to delivery using antepartum features in vaginal deliveries},
  journal = {Submitted to PLOS Digital Health},
  year    = {2026}
}
```

## Contact

Yung-Cheng Yang — william.yang2000727@gmail.com
Corresponding author: Yen-Wei Chu — ywchu@nchu.edu.tw
National Chung Hsing University, Graduate Institute of Genomics and
Bioinformatics, Taichung, Taiwan
