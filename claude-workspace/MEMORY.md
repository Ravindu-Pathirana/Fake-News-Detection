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

## Open threads / next steps
1. Fix label mapping (case-insensitive) and rerun full pipeline; report macro-F1 + per-class.
2. Add majority-class + class-weighted baselines; report confusion matrices.
3. Reconcile paper claims (metadata) with code, or actually implement metadata features.
4. Ablation: text -> +metadata -> +feature selection -> +tuning, one metric throughout.
5. Refresh related work; add a modern (BERT) reference point.
6. Consider revision-checklist document for the author.
