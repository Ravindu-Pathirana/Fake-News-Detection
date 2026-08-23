# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This repo is the codebase behind the paper *"An Explainable Fake News Detection Using
Classical Machine Learning"* (Ravindu Pathirana, University of Moratuwa). It trains
classical ML models (Logistic Regression, SVM, Naive Bayes, Random Forest, XGBoost) on
the LIAR benchmark dataset for binary fake-news classification, using TF-IDF features,
Chi-Square/Mutual-Information feature selection, GridSearchCV tuning, and SHAP for
explainability.

There is no application code — the project is entirely a set of Jupyter notebooks plus
data/result CSVs.

## Repository layout

- `artifacts/identify_datasets_notebook.ipynb` — initial dataset exploration (loads the
  LIAR train/valid/test CSVs, basic EDA).
- `artifacts/baseline_pipeline_notebook.ipynb` — baseline reproduction: text preprocessing
  → TF-IDF → Naive Bayes / Logistic Regression / SVM / Random Forest → accuracy-based eval.
- `artifacts/proposed_improvements.ipynb` — the main experimental pipeline: label
  encoding → text preprocessing (NLTK stopwords + Porter stemming) → TF-IDF → Chi²/MI
  feature selection → GridSearchCV over LR/SVM/NB/RF/XGBoost → results table → SHAP
  explainability on the best model.
- `artifacts/train.csv`, `valid.csv`, `test.csv` — LIAR dataset splits (TSV-derived CSV,
  see `artifacts/README` for column definitions: label, statement, subject, speaker,
  job title, state, party, 5 credit-history count columns, context).
- `artifacts/model_comparison_results.csv` — recorded output of the proposed pipeline
  (per-method, per-model Accuracy/Precision/Recall/F1 on valid and test splits).
- `An_Explainable_Fake_News_Detection_Using_Classical_Machine_Learning.pdf` — the paper
  itself.
- `claude-workspace/` — Claude's own working notes for this project (not application
  code). `claude-workspace/MEMORY.md` has running project context/decisions;
  `claude-workspace/ISSUE_PLAN.md` has a phased plan for getting the paper to
  acceptance quality; `claude-workspace/paper_review_findings.md` has detailed
  peer-review notes. Re-read these at the start of a session before touching the
  notebooks or paper.

## Environment

No `requirements.txt`/`environment.yml` exists yet. The notebooks depend on: `pandas`,
`numpy`, `scikit-learn`, `nltk` (with the `stopwords` corpus downloaded), `xgboost`,
`shap`, `matplotlib`, `seaborn`. Run notebooks with Jupyter (`jupyter notebook` /
`jupyter lab`, or `jupyter nbconvert --to notebook --execute <file>` for a headless
run) from the `artifacts/` directory, since the notebooks load the CSVs with relative
paths (`pd.read_csv("train.csv")`).

## Critical known bug — read before trusting any reported numbers

Both `baseline_pipeline_notebook.ipynb` and `proposed_improvements.ipynb` contain a
**label-mapping bug**: the CSVs store labels as uppercase (`FALSE`/`TRUE`/etc.), but
`convert_label()` compares against lowercase strings. Because the match is
case-sensitive, ~1,995 plain-`FALSE` rows (the largest fake-class category) fall
through to the `else` branch and get mislabeled as **real**. This creates an
artificial 76/24 class imbalance, and the paper's reported "F1 up to 0.87" is a
positive-class-only F1 of a near-majority classifier (RF/XGBoost show test recall =
1.000). The honest, label-corrected performance is ~0.60 accuracy / ~0.58 macro-F1.

When working on this codebase (fixing notebooks, rerunning experiments, or revising
the paper), assume this bug is still present unless you've verified it's fixed, and
see `claude-workspace/ISSUE_PLAN.md` (Phase 1, issue I-1/I-2) for the exact fix and
verification steps. Do not report or reuse existing numbers in
`model_comparison_results.csv` as valid without noting they were produced under this
bug.

Other things confirmed clean: TF-IDF/SelectKBest are fit on train only (no leakage);
the five `*_counts` credit-history columns are correctly excluded from features (they
leak the label) — keep it that way in any refactor.
