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

## Progress (2026-08-23, cont'd) — Phase 2 done
- `artifacts/metadata_features.build_pipeline_transformer()` + new
  `artifacts/ablation_v1.ipynb`: full `sklearn.Pipeline` (TF-IDF/one-hot/SelectKBest
  fit inside each CV fold, not before it — I-13) running the ablation
  text -> +metadata -> +feature selection (chi2 k=3000) -> +tuning, fixed model = LR.
- `RUN.md` (repo root) documents the run order / reproduction steps for Phases 1-3.
- **Important finding: the ablation's CV ranking and test ranking disagree.** CV says
  stage 4 (+tuning) is best (0.621 mean macro-F1); on held-out test, stage 3
  (+feature selection, untuned) is actually best (0.628) and stage 4 is worse
  (0.623) — GridSearchCV picked `class_weight="balanced"` for a CV gain that didn't
  generalize. McNemar's test: stage1-vs-stage3 IS significant (p=0.034); stage1-vs-
  stage4 is NOT (p=0.58). **Recommend the paper cite stage 3, not stage 4, as the
  headline result**, with the p-value stated explicitly — reporting stage 4 as "our
  best model" would repeat the overclaiming pattern this whole review exists to fix.
- Bootstrap 95% CI on stage 4 test macro-F1: [0.597, 0.649] (point estimate 0.623).

## Progress (2026-08-23, cont'd) — Phase 4 (I-8) and Phase 5 done; full LaTeX draft written
- `artifacts/distilbert_baseline_v1.ipynb`: fine-tuned distilbert-base-uncased, 3
  epochs, MPS backend, seed 42. Test macro-F1 = 0.636 (acc 0.650, fake F1 0.565).
  **Does NOT beat the best classical result** (NB+metadata, 0.649) — a genuinely
  interesting finding, not a placeholder number. No significance test run between
  them yet (flagged as future work in the paper draft).
- `artifacts/shap_explainability_v1.ipynb`: SHAP on XGBoost (text+metadata, chi2
  k=1000, corrected labels, test macro-F1 0.605). Un-stemmed display labels,
  genuine per-class breakdown (top fake-pushing vs. real-pushing features), 5
  figures saved to `artifacts/figures/`. **Party affiliation is directly visible as
  a top bias feature** (Party:democrat pushes fake, Party:republican pushes real) —
  strong, concrete evidence for the "topical/dataset bias, not deception cues"
  reframing.
- `artifacts/make_feature_selection_figures.py`: regenerated the original paper's
  Fig.1/Fig.2 (top Chi2/MI features) on corrected labels with readable un-stemmed
  labels.
- **Full corrected LaTeX paper written**: `claude-working-files/paper_v2.tex`
  (IEEEtran, single self-contained file, thebibliography inline). Same title/
  author as the original PDF (this supersedes it, not a new paper). Incorporates
  every number from Phases 1-5 across 6 tables + 8 figures (all real, copied into
  `claude-working-files/figures/` — no placeholder images). Two TODOs left
  in-file for the author: (1) verify/add 2-3 more 2021-2025 LIAR-specific
  transformer citations beyond BERT/DistilBERT/FakeBERT (deliberately not
  fabricated); (2) double check affiliation/email are current. Not compiled
  locally (no LaTeX toolchain on this machine) — recommend Overleaf or local
  texlive to verify before submission.

## Open threads / next steps
1-4. ~~Phases 1-3 (label fix, baselines, metadata, ablation+significance).~~ DONE
5. ~~Refresh related work; add a modern (BERT) reference point.~~ DONE (Phase 4,
   DistilBERT — see finding above)
6. Consider revision-checklist document for the author. (Phase 7 covers this)
7-8. ~~Statistical rigor + reproducibility Pipeline harness.~~ DONE (Phase 2)
9. Remaining: Phase 6 polish (the LaTeX draft already writes IEEE prose and cites
   the significance caveat, but hasn't been proofread/compiled), and Phase 7's
   final pre-submission checklist pass once the author reviews `paper_v2.tex`.
