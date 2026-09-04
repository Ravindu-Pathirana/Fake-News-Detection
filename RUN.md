# Reproducing the corrected results

This documents how to reproduce the label-fix, metadata, ablation, transformer
reference, explainability, and revision-cycle (generalization audit, corrected
significance) results (`claude-workspace/ISSUE_PLAN.md` Phases 1-5 and
`claude-workspace/CLAUDE_ACCEPTANCE_PLAN.md` Phases R0-R8). See `CLAUDE.md` for
repo layout.

## Environment

```
python3 -m pip install -r requirements.txt
python3 -c "import nltk; nltk.download('stopwords')"
```

All notebooks pin `RANDOM_STATE = 42` via `liar_utils.py`. Deterministic
components (TF-IDF, one-hot encoders, GridSearchCV with `shuffle=True` on this
seed) reproduce exactly; RandomForest/XGBoost are also seeded but may vary
slightly by library version.

## Run order

From `Updated Improvements/` (not `artifacts/` -- that folder holds only the
original, deliberately-unfixed, as-submitted record; see `CLAUDE.md`), in
order (each is `python3 -m nbconvert --to notebook --execute --inplace <file>`):

1. `baseline_pipeline_v2.ipynb` -- label-corrected baseline (TF-IDF + NB/LR/SVM/RF +
   Dummy baselines; TF-IDF config matches every other v2+ notebook -- 10k
   features, `min_df=2`, `max_df=0.9` -- since 2026-09-04's R5.1 fix). Writes
   `baseline_results_v2.csv`.
2. `proposed_improvements_v2.ipynb` -- label-corrected proposed pipeline (Chi2/MI
   feature selection + GridSearchCV, macro-F1 objective). Reads
   `baseline_results_v2.csv`, writes `model_comparison_results_v2.csv`.
3. `proposed_with_metadata_v1.ipynb` -- adds the text+metadata feature space (Phase
   3, I-3). Reads `model_comparison_results_v2.csv`, writes
   `model_comparison_results_v3_metadata.csv` -- the source of truth for the
   paper's Table II best-model numbers.
4. `ablation_v1.ipynb` -- Phase 2 ablation (text -> +metadata -> +feature selection
   -> +tuning) on a single fixed model (Logistic Regression), with 5-fold CV mean +/-
   std per stage, a bootstrap 95% CI on the final test macro-F1, and McNemar's test
   on the headline comparisons. Writes `ablation_results_v1.csv`. Self-contained --
   only depends on `train.csv`/`test.csv`, fit on `train.csv` only (by design, to
   isolate pipeline components; see `CLAUDE.md`), not on steps 1-3's output.
5. `distilbert_baseline_v1.ipynb` -- Phase 4 (I-8) transformer reference point.
   Fine-tunes `distilbert-base-uncased` for 3 epochs on raw `Statement` text.
   Requires the extra `torch`/`transformers`/`datasets`/`accelerate` dependencies
   in `requirements.txt` and a network connection (downloads the pretrained model
   on first run, ~260MB). Uses MPS on Apple Silicon automatically if available;
   falls back to CPU otherwise (slower). Writes `transformer_baseline_results_v1.csv`.
   Self-contained -- only depends on `train.csv`/`test.csv`.
6. `shap_explainability_v1.ipynb` -- Phase 5 (I-10, I-12) explainability, rerun on
   the corrected labels with un-stemmed display labels and a per-class breakdown.
   Writes `shap_top_features_v1.csv` and five figures to `figures/`.
   Self-contained -- only depends on `train.csv`/`test.csv`.
7. `make_feature_selection_figures.py` (plain script, not a notebook) --
   regenerates the paper's Chi2/MI top-feature figures on corrected labels with
   readable un-stemmed labels: `python3 make_feature_selection_figures.py`.
8. `significance_v2.ipynb` (added 2026-09-04, Phase R1) -- Holm-Bonferroni
   correction across the paper's full McNemar comparison family, including the
   actual best model (Naive Bayes, text+metadata), not just the LR ablation.
   Writes `test_predictions_v2.csv`, `pvalue_family_r1.csv`,
   `bootstrap_diff_ci_v1.csv`, `seed_variance_v1.csv`. Depends only on
   `train.csv`/`valid.csv`/`test.csv`.
9. `generalization_audit_v1.ipynb` (added 2026-09-04, Phase R2, flagship) --
   speaker-disjoint / party-shift / subject-disjoint generalization audit
   against a matched random-split control, plus a joint Holm correction that
   folds in step 8's family. Writes `speaker_overlap_stats_v1.csv`,
   `generalization_results_v1.csv`, `generalization_gap_summary_v1.csv`,
   `pvalue_family_r2.csv`, `pvalue_family_final.csv` (the authoritative,
   fully-corrected p-value table for the manuscript),
   `shap_disjoint_rank_comparison_v1.csv`. **Run this after step 8** --
   it reads `pvalue_family_r1.csv` to build the joint family. Depends only on
   `train.csv`/`valid.csv`/`test.csv`.

Steps 1-2 must run before step 3. Steps 4-7 are each independent of 1-3 and can
run any time after step 1's label fix exists in `liar_utils.py`. Step 9 depends
on step 8's output file; both are independent of steps 1-7.

`claude-working-files/paper_icac2026.tex` is the active manuscript (IEEE
double-column, 6-page ICAC 2026 cap) incorporating every result above,
including the Phase R1/R2 revision; `claude-working-files/paper_icac2026_blind.tex`
is its anonymized counterpart (blank author block, no live repo link, no
acknowledgment). `claude-working-files/paper_v2.tex` is a retired,
uncondensed Phase 0-5 draft -- do not treat it as current (see its header
comment). Compile with the local TinyTeX toolchain per `CLAUDE.md`.

The original `baseline_pipeline_notebook.ipynb`, `proposed_improvements.ipynb`, and
`model_comparison_results.csv` (in `artifacts/`) are left untouched as the as-submitted (buggy) record
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
- `ablation_v1.ipynb`, `significance_v2.ipynb`, and `generalization_audit_v1.ipynb`
  each wrap TF-IDF/one-hot/`SelectKBest` inside a single `sklearn.pipeline.Pipeline`,
  so vocabulary/category fitting happens inside each CV fold, not once on the full
  training set before cross-validation -- including the pooled, cross-protocol
  fitting in step 9.

## Reading the results

`model_comparison_results_v3_metadata.csv` and `ablation_results_v1.csv` are the
source of truth for the pre-revision corrected numbers. `pvalue_family_final.csv`
and `generalization_gap_summary_v1.csv` are the source of truth for the revision's
corrected-significance and generalization-audit numbers. See
`claude-workspace/RESULTS_LEDGER.md` for a full number-to-source mapping used in
the manuscript, and `claude-workspace/ISSUE_PLAN.md` / `CLAUDE_ACCEPTANCE_PLAN.md`
STATUS notes for the narrative summary of what each number means for the paper.
