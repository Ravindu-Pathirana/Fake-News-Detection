# Reproducing the paper's results

This file explains how to reproduce every table and figure in the ICAC 2026 paper
from the code in `Updated Improvements/`. The `artifacts/` folder holds the original
draft with its label bug. Do not rerun or edit anything in it.

## Environment

```
python3 -m pip install -r requirements.txt
python3 -c "import nltk; nltk.download('stopwords')"
```

The results were produced with Python 3.9.6, scikit-learn 1.6.1, XGBoost 2.1.4, and
SHAP 0.49.1. Every script uses `RANDOM_STATE = 42` from `liar_utils.py`.
GridSearchCV uses unshuffled stratified 5-fold splits, so results reproduce exactly.
The exception is Random Forest and XGBoost, which can vary slightly with library
version. The `torch`/`transformers` block in `requirements.txt` is only needed for
DistilBERT.

## Run order

Run each notebook from inside `Updated Improvements/`, because the notebooks load the
CSVs by relative path:

```
cd "Updated Improvements"
python3 -m nbconvert --to notebook --execute --inplace <notebook>.ipynb
```

1. **`baseline_pipeline_v2.ipynb`** — TF-IDF baselines and the two Dummy classifiers.
   Writes `baseline_results_v2.csv`.
2. **`proposed_improvements_v2.ipynb`** — text-only models with Chi-square / Mutual
   Information feature selection, chosen by cross-validation.
   - Reads: `baseline_results_v2.csv`
   - Writes: `model_comparison_results_v2.csv` and the test predictions
3. **`proposed_with_metadata_v1.ipynb`** — text + metadata models.
   - Reads: `model_comparison_results_v2.csv`
   - Writes: `model_comparison_results_v3_metadata.csv` and the test predictions
4. **`ablation_v1.ipynb`** — single-model Logistic Regression ablation in four
   cumulative stages, fit on the train split only.
   - Writes: `ablation_results_v1.csv` and the test predictions
   - Independent of steps 1–3
5. **`metadata_ablation_v1.ipynb`** — text-only and metadata-only baselines, the
   job-title / context ablation, the job-title speaker statistics, and the
   speaker-disjoint cross-validation (`StratifiedGroupKFold` on Speaker).
   - Reads: `model_comparison_results_v3_metadata.csv`
   - Writes: `metadata_ablation_results_v1.csv`, `metadata_ablation_readout_v1.csv`,
     `jobtitle_speaker_stats_v1.csv`, `speaker_disjoint_folds_v1.csv`,
     `speaker_disjoint_summary_v1.csv`, and the test predictions
6. **`distilbert_fair_v2_kaggle.ipynb`** (plain-script copy:
   `distilbert_fair_v2_kaggle.py`) — the DistilBERT reference.
   - **Training:** fit on the same 11,524-row pool. Learning rate and epochs are
     chosen on a 10% hold-out, then the model is refit with seeds 42–44.
   - **Where it runs:** on Kaggle with a GPU. Setup steps are in the notebook's first
     cell.
   - **Data check:** the loader confirms with row fingerprints that it reads the same
     rows as the project CSVs.
   - **Outputs:** copy `distilbert_v2_*` back into `Updated Improvements/`.
   - **Smoke test:** run locally with `SMOKE=1`.
7. **`significance_v3.ipynb`** — the fixed family of 12 paired McNemar tests with
   Holm correction, plus paired bootstrap 95% confidence intervals.
   - Reads: the test predictions from steps 3–6
   - Writes: `pvalue_family_camera_ready.csv`
8. **`shap_explainability_v1.ipynb`** — SHAP on Naive Bayes (text + metadata, alpha
   0.5). It uses `shap.LinearExplainer` with the closed-form log-odds coefficients,
   so the Shapley values are exact.
   - Writes: `shap_top_features_v1.csv` and the figures in `figures/`
   - Independent of the other steps

Dependencies: run step 2 after step 1, step 3 after step 2, and step 5 after step 3.
Step 7 needs the outputs of steps 3–6. Steps 4 and 8 can run at any time.

## Where each paper result comes from

| Paper item | Source |
|---|---|
| Table I (split sizes, class balance) | `liar_utils.py` label mapping on `train/valid/test.csv` |
| Table II (model comparison) | `model_comparison_results_v2.csv`, `model_comparison_results_v3_metadata.csv`, `metadata_ablation_results_v1.csv` (NB/LR text-only rows), `distilbert_v2_summary.csv` |
| Table III (ablation) | `ablation_results_v1.csv` |
| Table IV (12 paired tests) | `pvalue_family_camera_ready.csv` |
| Table V (feature sources, speaker-disjoint CV) | `metadata_ablation_results_v1.csv`, `speaker_disjoint_summary_v1.csv` |
| Table VI ("Ours" rows) | `metadata_ablation_results_v1.csv`, `distilbert_v2_summary.csv` |
| Fig. 1 (pipeline) | drawn in the paper's LaTeX source |
| Fig. 2 (per-class SHAP) | `figures/shap_per_class.png` |
| Fig. 3 (local SHAP) | `figures/shap_local_examples.png` |

Not used by the camera-ready paper, kept for the record:
- `distilbert_baseline_v1.ipynb` (DistilBERT on the official split, 3 fixed epochs)
- `make_feature_selection_figures.py` (Chi-square / MI top-feature figures)
- `figures/shap_bar.png`, `shap_beeswarm.png`, `shap_force_*`, `ablation_cv_vs_test.png`

## Data note

The CSVs read LIAR's tab-separated release with standard CSV quote handling.
Quotation marks inside eight statements absorb the 45 statements that follow them,
which gives 10,240 / 1,284 / 1,267 rows instead of the official 10,269 / 1,284 /
1,283. All results use these rows, and the paper says so. Any new code must load the
same rows.

## Leakage guardrails

- **Credit-history counts:** the five `*_counts` columns are never used as features.
  They are derived from the labels and would leak the target. `METADATA_COLUMNS` in
  `metadata_features.py` excludes them.
- **Speaker:** excluded by default. With `include_speaker=True` it goes through a
  64-dimensional `FeatureHasher` (`alternate_sign=False`). It does not improve
  results, so the default is reported.
- **Per-fold refitting:** every fitted step (category clean-up, TF-IDF vocabulary and
  IDF, one-hot encoders, `SelectKBest`) is wrapped with the classifier in one
  scikit-learn `Pipeline` and refit inside each cross-validation fold.
- **Model selection:** configurations are selected by cross-validation only. Test
  scores are reported, never used for selection.
