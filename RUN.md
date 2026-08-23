# Reproducing the corrected results

This documents how to reproduce the label-fix, metadata, ablation, transformer
reference, and explainability results (`claude-workspace/ISSUE_PLAN.md`
Phases 1-5). See `CLAUDE.md` for repo layout.

## Environment

```
python3 -m pip install -r requirements.txt
python3 -c "import nltk; nltk.download('stopwords')"
```

All notebooks pin `RANDOM_STATE = 42` via `artifacts/liar_utils.py`. Deterministic
components (TF-IDF, one-hot encoders, GridSearchCV with `shuffle=True` on this seed)
reproduce exactly; RandomForest/XGBoost are also seeded but may vary slightly by
library version.

## Run order

From `artifacts/`, in order (each is `python3 -m nbconvert --to notebook --execute --inplace <file>`):

1. `baseline_pipeline_v2.ipynb` -- label-corrected baseline (TF-IDF + NB/LR/SVM/RF +
   Dummy baselines). Writes `baseline_results_v2.csv`.
2. `proposed_improvements_v2.ipynb` -- label-corrected proposed pipeline (Chi2/MI
   feature selection + GridSearchCV, macro-F1 objective). Reads
   `baseline_results_v2.csv`, writes `model_comparison_results_v2.csv`.
3. `proposed_with_metadata_v1.ipynb` -- adds the text+metadata feature space (Phase
   3, I-3). Reads `model_comparison_results_v2.csv`, writes
   `model_comparison_results_v3_metadata.csv`.
4. `ablation_v1.ipynb` -- Phase 2 ablation (text -> +metadata -> +feature selection
   -> +tuning) on a single fixed model (Logistic Regression), with 5-fold CV mean +/-
   std per stage, a bootstrap 95% CI on the final test macro-F1, and McNemar's test
   on the headline comparisons. Writes `ablation_results_v1.csv`. Self-contained --
   only depends on `train.csv`/`test.csv`, not on steps 1-3's output.
5. `distilbert_baseline_v1.ipynb` -- Phase 4 (I-8) transformer reference point.
   Fine-tunes `distilbert-base-uncased` for 3 epochs on raw `Statement` text.
   Requires the extra `torch`/`transformers`/`datasets`/`accelerate` dependencies
   in `requirements.txt` and a network connection (downloads the pretrained model
   on first run, ~260MB). Uses MPS on Apple Silicon automatically if available;
   falls back to CPU otherwise (slower). Writes `transformer_baseline_results_v1.csv`.
   Self-contained -- only depends on `train.csv`/`test.csv`.
6. `shap_explainability_v1.ipynb` -- Phase 5 (I-10, I-12) explainability, rerun on
   the corrected labels with un-stemmed display labels and a per-class breakdown.
   Writes `shap_top_features_v1.csv` and five figures to `artifacts/figures/`.
   Self-contained -- only depends on `train.csv`/`test.csv`.
7. `make_feature_selection_figures.py` (plain script, not a notebook) --
   regenerates the paper's Chi2/MI top-feature figures on corrected labels with
   readable un-stemmed labels: `python3 make_feature_selection_figures.py`.

Steps 1-2 must run before step 3. Steps 4-7 are each independent and can run any
time after step 1's label fix exists in `liar_utils.py`.

`claude-working-files/paper_v2.tex` is the full corrected paper draft (IEEEtran,
single file) incorporating every result above; `claude-working-files/figures/` is
a copy of the figures the paper references. No LaTeX toolchain was available to
compile-verify it in this environment -- verify with Overleaf or a local texlive
install before submission, and see the TODO comments at the top of the .tex file.

The original `baseline_pipeline_notebook.ipynb`, `proposed_improvements.ipynb`, and
`model_comparison_results.csv` are left untouched as the as-submitted (buggy) record
for before/after comparison -- do not rerun or overwrite them.

## Leakage guardrails

- The five `*_counts` credit-history columns (`Barely true counts`, `False counts`,
  `Half true counts`, `Mostly true counts`, `Pants on fire counts`) are never used as
  features anywhere in this pipeline -- they are derived from the label history and
  leak the target. `metadata_features.py`'s `METADATA_COLUMNS` deliberately excludes
  them.
- `Speaker` is excluded from the default metadata feature set (high-cardinality
  identity leakage risk); when included via `include_speaker=True` it goes through a
  64-dim `FeatureHasher`, not one-hot encoding. Phase 3 found this doesn't improve
  results, so the default (no speaker) is what should be reported.
- `ablation_v1.ipynb` wraps TF-IDF/one-hot/`SelectKBest` inside a single
  `sklearn.pipeline.Pipeline`, so vocabulary/category fitting happens inside each CV
  fold, not once on the full train set before cross-validation.

## Reading the results

`model_comparison_results_v3_metadata.csv` and `ablation_results_v1.csv` are the
current source of truth for the corrected numbers. See
`claude-workspace/ISSUE_PLAN.md` Phase 1/2/3 STATUS notes for the narrative summary
and what each number means for the paper.
