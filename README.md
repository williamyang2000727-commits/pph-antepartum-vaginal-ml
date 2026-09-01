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
| **XGBoost + SMOTEENN + Top 30 features + KNN k=7 imputation** | Sensitivity **0.7838** / AUC **0.7404** / MCC **0.3876** / Brier **0.2418** |
| Confusion matrix | TP 29 / FP 39 / FN 8 / TN 78 |
| Composite score | **0.6123** — ranked **1 / 2,646** unique pipelines |

Composite score = `0.4 × MCC + 0.3 × AUC + 0.3 × sensitivity`, computed on the
independent test set. The same pipeline ranked **522 / 2,646** (top 19.7%) on the
cross-validation composite, which played no part in its selection.

Because the final model was selected by its test-set composite score, the
test-set metrics are an **optimistic bound**, not an unbiased estimate of
generalization. The manuscript reports this explicitly and quantifies it: the
winning pipeline's test-minus-cross-validation composite gap is +0.2066 against
a mean of +0.0033 across all 2,646 pipelines (62.9×, the highest percentile), and
cross-validation AUC correlates only weakly with test AUC (Spearman
ρ = 0.276). The conservative lower bound is the cross-validation AUC of 0.6266.

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
| `04_train_RF_SMOTEENN_Top30.py` | **The canonical feature-selection and training script.** Stratified 80/20 split, then train-only chi-square and t-test filtering, train-only Bootstrap LASSO stability selection (100 resamples × 4 regularization parameters = 400 fits), KNN k=1 imputation, standardization, SMOTEENN, RandomForest. Retained because it is the canonical source of the Top 30 feature set and of `bootstrap_selection_frequency_clean.csv`; its own self-check still targets the original RandomForest baseline, which is **not** the final model reported in the manuscript |
| `06_joint_search.py` | Joint search over 7 × 9 × 7 × 6 = **2,646 unique pipelines** by 10-fold stratified cross-validation (**26,460 fits**) |
| `07_full_test_eval.py` | Retrain all 2,646 pipelines on the full training set and evaluate each on the independent test set |
| `05_final_model_evaluation.py` | Retrain the **RandomForest baseline** (Top 30 + SMOTEENN, the configuration this script was written for) and write its confusion matrix, receiver operating characteristic curve, mean-decrease-in-impurity importances and a four-panel summary figure. Kept for provenance; the model reported in the manuscript is extreme gradient boosting, produced by `06`/`07` |
| `08_test_eval_top10.py` | Cross-check the ten best cross-validation pipelines on the test set |

Inside the cross-validation loop, imputation, standardization and
class-imbalance resampling are **all fitted separately within each training
fold** and then applied to that fold's validation data, which is never used to
estimate any of them. The code is the authority on this: in `06_joint_search.py`
the imputer and the scaler are constructed inside the `for train_idx, val_idx`
loop.

> **Corrected on 2026-08-19.** Earlier versions fitted the imputer and the
> scaler **once on the full training set before the fold split**, so every
> validation fold contributed to the imputation and scaling parameters. That is
> cross-validation-level data leakage, it made the cross-validation figures
> optimistic, and it contradicted the fit-on-train principle this study cites in
> its own Discussion. The experiment was rerun after the fix and every number in
> the manuscript comes from the rerun. This README described the pre-fix
> behaviour until 2026-08-20.

Two things remain outside the loop, by design and stated here for transparency:
the feature ranking, which is computed on the training set only and loaded once
(so folds do not re-select features), and the independent test-set evaluation,
which fits on the full training set of 615 and transforms the 154 test cases.

### The trained model

`final_model_xgboost_top30.json` is the model reported in the manuscript,
exported in a form that can be evaluated without this codebase: 100 gradient
boosted trees, the 30 feature names in the order the model expects them, and the
preprocessing constants (the median used for imputation and the mean and scale
used for standardisation, one value per feature). A prediction is the logistic
transform of the summed leaf values plus the base margin, as recorded in the
file's `note` field. Node comparisons must be done in float32; in float64 a
handful of values sit on the wrong side of a split threshold.

**What this file does and does not reproduce.** The trees, the base margin and
the standardisation constants are exact: supply the held-out test set imputed
the way the manuscript describes — K-nearest neighbors with k = 7, fitted on the
training set — and traversing the trees in this file reproduces the published
confusion matrix cell for cell (TP 29 / FP 39 / FN 8 / TN 78, sensitivity 0.7838,
area under the curve 0.7404). The imputation step is the exception. K-nearest
neighbors imputation needs the training set to compute, and that set cannot be
released, so what ships here instead is the vector of training-set medians. Those
medians let the model score a new individual with no reference data at all, which
is the point of releasing it, but they are a substitute rather than the procedure
the reported metrics came from: scoring the same test set with them gives a
sensitivity of 0.5946 and an area under the curve of 0.7277. Anyone reproducing
the manuscript numbers from this file should therefore impute with K-nearest
neighbors (k = 7) rather than with the shipped constants. The `imputation` field
inside `preprocessing` records which method the model was trained under, not the
method the accompanying constants implement.

The file contains model parameters and per-feature aggregates only. It carries no
patient records, no identifiers and no row-level values, so releasing it does not
conflict with the restrictions on the underlying data. TRIPOD+AI item 22 asks for
the model in a form that allows predictions in new individuals and third party
evaluation; this file is that form, and it is covered by the MIT licence of this
repository.

### Not used for the published results

| Script | Why it is kept |
|---|---|
| `03_feature_selection.py` | An earlier stability-selection variant. It applies a fixed 0.6 selection-frequency threshold, stratified bootstrap resampling, `class_weight='balanced'`, `C = [0.01, 0.1, 0.5, 1.0]`, and the whole cohort with no train/test split. **It does not produce the published 30 features.** Those come from `04_train_RF_SMOTEENN_Top30.py`, which uses unstratified resampling, no class weighting, `C = [0.01, 0.1, 1.0, 10.0]`, and the 615 training cases only, then ranks features by selection frequency and takes the top N. Re-running both variants reproduces the published selection counts exactly (35 of 35) for the `04` procedure and not at all (0 of 35) for this one. Retained for provenance; do not use it to reproduce the published feature set |

### Not included

This repository holds the cohort-construction, feature-engineering, model-search
and evaluation code — that is, everything that produces a reported number. The
figure and table scripts live in a separate analysis directory and are not
included here: some only draw schematics or format results into Word tables, and
the rest cover the bootstrap confidence intervals, the calibration curve, the
decision curve analysis and the Shapley additive explanations plots. All of them
are available from the authors on request.

## Reproducibility

Every random state is fixed to `42` — data partitioning, cross-validation,
bootstrap resampling and class-imbalance routines all use the same seed. Given
`all_features.csv`, `04_train_RF_SMOTEENN_Top30.py` reproduces the 400-fit
selection counts exactly. Note that the confusion matrix this script asserts
against (TP 27 / FP 41 / FN 10 / TN 76, to a tolerance of 0.001) belongs to the
original RandomForest baseline it was written to reproduce, **not** to the final
model reported in the manuscript, whose confusion matrix is TP 29 / FP 39 /
FN 8 / TN 78.

Two caveats worth knowing before re-deriving anything:

1. **Tied selection counts.** Five groups of features share an identical
   selection count — ranks 3–4 (275), 18–19 (187), 21–22 (180), 30–31 (142) and
   33–35 (115) — and the default pandas sort is not stable, so regenerating the
   ranking table from scratch can swap tied features. The Top 8, Top 15, Top 20
   and Top 25 boundaries fall clear of every tie. The **Top 30 boundary does sit
   on a tie**: the 30th and 31st features (measurement count of estimated
   glomerular filtration rate and measurement count of creatinine) are both
   selected 142 times. This changes nothing, for a structural reason: estimated
   glomerular filtration rate is derived from creatinine by the MDRD formula, so
   the two are always ordered together and their measurement counts are identical
   for all 769 deliveries (Pearson r = 1.000000, maximum absolute difference 0).
   Substituting one for the other and retraining reproduces the reported metrics
   and confusion matrix exactly.

2. **The stored ranking table.** Scripts `05` through `08` read a stored
   selection-frequency table, `bootstrap_selection_frequency_clean.csv`. This is
   written by `04_train_RF_SMOTEENN_Top30.py`, which computes the ranking and
   exports it directly, so the file can never drift from the procedure that
   produced it. Script `03` writes a differently named file from a different
   variant of the procedure and is not used by the final model.

## Installation

```bash
git clone https://github.com/williamyang2000727-commits/pph-antepartum-vaginal-ml.git
cd pph-antepartum-vaginal-ml
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running the pipeline

The raw hospital records cannot be shared (see **Data availability**), so the
pipeline cannot be executed end-to-end from this repository alone. If you supply
your own data in the same shape, note that the scripts use **two different path
conventions**, a legacy of the directory they were developed in:

| Stage | Working directory | Reads | Writes |
|---|---|---|---|
| `01` `02` `03` | the directory holding the data files | plain filenames in the current directory | plain filenames in the current directory |
| `04` `05` `06` `07` `08` | anywhere | `<base>/02_experiment_csv/` | `<base>/02_experiment_csv/`, `<base>/01_KEY_RESULTS/`, `<base>/04_logs/` |

`<base>` is resolved by `_resolve_base()`: it walks up from the script looking
for a directory that contains `02_experiment_csv`, and falls back to the
script's own directory. So placing your data in
`pph-antepartum-vaginal-ml/02_experiment_csv/` and running the scripts from the
repository root works; the output directories are created automatically.

Because of the split convention, `02` writes `all_features.csv` into the working
directory while `04` reads it from `02_experiment_csv/`. Move or symlink the file
between those two stages.

> **Corrected on 2026-08-20.** `04` through `08` previously derived `<base>` as
> "two directories above the script", which is correct only in the original
> project layout, where the scripts sit in a subdirectory. In this repository the
> scripts sit at the root, so that expression resolved to the directory
> *containing* the clone and every one of those five scripts failed on a fresh
> checkout.

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
