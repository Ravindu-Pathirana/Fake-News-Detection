# Project Memory — Fake News Detection

_Re-read this at the start of each session. Append dated entries as work progresses._

## Project
- Paper: "An Explainable Fake News Detection Using Classical Machine Learning"
  (Ravindu Pathirana, University of Moratuwa). Target: a conference submission.
- Goal: get the paper to acceptance quality. User wants an honest reviewer critique
  plus concrete fixes.
- Dataset: LIAR (train 10240 / valid 1284 / test 1267), 6-class collapsed to binary
  (fake = pants-fire/false/barely-true ; real = half-true/mostly-true/true).
- Pipeline: TF-IDF -> Chi2 / Mutual Information feature selection -> LR, SVM, NB, RF,
  XGBoost with GridSearchCV -> SHAP explainability.

## Working location
- All Claude files for this project go in `claude-workspace/` (this folder), inside
  the project on "My Disk 2". Do NOT use any My Disk 1 / external default memory.

## KEY FINDING (2026-08-12) — CRITICAL
- **Label-mapping bug** in both `baseline_pipeline_notebook.ipynb` and
  `proposed_improvements.ipynb`. The CSVs store labels as UPPERCASE `FALSE`/`TRUE`,
  but `convert_label()` tests lowercase `'false'`. String match is case-sensitive, so
  ~1,995 plain-`FALSE` (largest fake category) fall through to `else` and are
  mislabeled **real**.
- Effect: artificial 76/24 class imbalance. Reported "F1 up to 0.87" is
  positive-class-only F1 of a near-majority classifier (RF/XGB show recall = 1.000).
- Verified rerun (LR, TF-IDF unigram, sklearn stopwords):
  - Buggy: acc 0.761, F1pos 0.862, **macro-F1 0.474**
  - Fixed: acc 0.606, F1pos 0.680, **macro-F1 0.584** (true LIAR-binary range)
- So the headline result is invalid; honest performance is ~0.60 acc / ~0.58 macro-F1.

## Other confirmed issues
- Code is NOT leaky otherwise: TF-IDF and SelectKBest are fit on train only; credit-count
  columns (Barely true counts, etc.) are NOT used. Good.
- BUT paper claims a "text + meta-data" combined feature space (one-hot speaker/party/etc.).
  The notebooks never build this — **paper/code mismatch**.
- Baseline reported in accuracy, proposed in F1 -> inconsistent metric, not comparable.
- Related work stops at 2020; no transformer baseline.
- SHAP top features are topical/entity words (obama, obamacar, socialist) -> evidence of
  topical bias, not deception cues. Currently framed positively; should be a limitation.

## Progress (2026-08-23) — Phase 0 + Phase 1 done
- Phase 0: branch `fix/label-and-metrics`; `requirements.txt` pinned from local env;
  `artifacts/liar_utils.py` created (RANDOM_STATE=42, shared `load_and_label`/`evaluate_full`).
- Phase 1 (I-1, I-2, I-4, I-5) fixed in `artifacts/baseline_pipeline_v2.ipynb` and
  `artifacts/proposed_improvements_v2.ipynb` (both executed clean, outputs saved).
  Corrected results in `artifacts/model_comparison_results_v2.csv`.
  **Corrected headline: LR + Mutual Information, test macro-F1 = 0.619** (acc 0.622,
  fake-class F1 0.587) — matches the Appendix's predicted honest range. All real models
  beat both Dummy baselines. Original notebooks/CSV left untouched as the as-submitted
  record. Full detail in `claude-workspace/ISSUE_PLAN.md` Phase 1 STATUS note.

## Progress (2026-08-23, cont'd) — Phase 3 done (Option A)
- `artifacts/metadata_features.py`: builds Subject(s)/Party/State/job/Context features
  (multi-label + one-hot, min_frequency=5 on high-cardinality ones), excludes the leaky
  `*_counts` columns, and hashes Speaker (64-dim FeatureHasher, alternate_sign=False)
  instead of one-hotting it, gated behind `include_speaker`.
- `artifacts/proposed_with_metadata_v1.ipynb`: reruns LR/SVM/NB/RF/XGBoost with
  GridSearchCV(f1_macro) on text+metadata and text+metadata+speaker. Results merged
  into `artifacts/model_comparison_results_v3_metadata.csv`.
- **New best result: Naive Bayes + text+metadata (no speaker), test macro-F1 = 0.649**
  (up from 0.619 text-only). Every model improved with metadata added.
- **Speaker (hashed) does NOT help** — NB drops to 0.643 with it added, others flat/mixed.
  This is a real empirical finding (not just a theoretical leakage worry): supports
  dropping speaker from the paper's main pipeline and reporting the null result as a
  robustness/limitations point.
- Debug note for future runs: FeatureHasher's default `alternate_sign=True` produces
  negative values that hard-fail MultinomialNB (`ValueError: Negative values`); use
  `alternate_sign=False` whenever hashed features feed into a Naive Bayes model.

## Open threads / next steps
1. ~~Fix label mapping (case-insensitive) and rerun full pipeline; report macro-F1 + per-class.~~ DONE
2. ~~Add majority-class + class-weighted baselines; report confusion matrices.~~ DONE
3. ~~Reconcile paper claims (metadata) with code, or actually implement metadata features.~~ DONE (Option A)
4. Full ablation: text -> +metadata -> +feature selection -> +tuning, one metric throughout,
   with seeds/variance (Phase 2). Phase 3 already gives the text vs. text+metadata step;
   still need +feature selection (chi2/MI) applied ON TOP of text+metadata, and +tuning
   variance across seeds.
5. Refresh related work; add a modern (BERT) reference point. (Phase 4)
6. Consider revision-checklist document for the author.
7. Statistical rigor (Phase 2, I-6): repeat over >=5 seeds, mean +/- std, significance test.
8. Reproducibility harness (Phase 2, I-13): wrap TF-IDF -> SelectKBest -> model in a single
   sklearn Pipeline per model so preprocessing is fit inside each CV fold structurally.
