# Fake-News-Detection

Code and data for the paper **"A Controlled and Explainable Evaluation of Classical
Machine Learning for Fake News Detection"** by Ravindu Pathirana (University of
Moratuwa), accepted at ICAC 2026.

The study evaluates five classical machine learning models (Logistic Regression,
SVM, Naive Bayes, Random Forest, XGBoost) on the binary LIAR benchmark. It uses
TF-IDF text features and structured metadata. The evaluation protocol is the main
subject: leakage-safe pipelines, cross-validation-only model selection, Holm-corrected
significance tests, a speaker-disjoint check, and SHAP explanations read as a bias
audit. A fine-tuned DistilBERT model is included as a reference point.

## Main results

All numbers are macro-F1 on the 1,267-statement test split (see the data note below).

| Configuration | Test macro-F1 |
|---|---|
| Naive Bayes, text + metadata (best classical model) | 0.655 |
| Naive Bayes, metadata only | 0.618 |
| Logistic Regression, text only | 0.615 |
| Naive Bayes, text only | 0.604 |
| DistilBERT, text + metadata (mean of 3 seeds) | 0.651 |
| DistilBERT, text only (mean of 3 seeds) | 0.641 |

- **Text + metadata beats either source alone.** Naive Bayes on text + metadata beats
  text alone and metadata alone, and beats Logistic Regression and SVM on the same
  features. These gaps stay significant after Holm correction over a fixed family of
  12 paired McNemar tests. The gaps to Random Forest and XGBoost do not.
- **Feature selection and tuning do not hold up.** In a single-model Logistic
  Regression ablation, the gains of about 0.03 from Chi-square feature selection and
  from tuning are not significant after correction. Configurations are selected by
  cross-validation only, never by test score.
- **Job title acts as a speaker identifier.** 90.3% of distinct job titles belong to
  a single speaker. Under speaker-disjoint cross-validation, the job-title benefit
  disappears, while subject, party, state, and context keep a smaller gain.
- **DistilBERT is stronger on text alone and level once metadata is added.** Trained
  under the same conditions, it beats the text-only classical models, but its
  text + metadata score is not significantly different from Naive Bayes.
- **The strongest features are metadata, not wording.** SHAP ranks party, state, job
  title, and subject as most influential, and no text token appears in the top
  twenty. This is reported as dataset bias, not as evidence that the model detects
  deception.

Macro-F1 of about 0.60–0.65 is the normal range for binary LIAR with classical
models.

## Data

- **Dataset:** LIAR (Wang, 2017), 12,836 short political statements from PolitiFact
  with six truthfulness labels.
- **Binary mapping:** pants-fire, false, and barely-true are mapped to *fake*;
  half-true, mostly-true, and true are mapped to *real*.
- **Rows used:** the CSVs in this repository read the release's tab-separated files
  with standard CSV quote handling. Quotation marks inside eight statements absorb the
  45 statements that follow them, so the splits have 10,240 / 1,284 / 1,267 rows
  instead of the official 10,269 / 1,284 / 1,283. Every result uses these rows.
- **Fitting pool:** classical models are fit on train + validation merged (11,524
  rows) with 5-fold cross-validation. The test split is scored once per model.
- **Excluded columns:** the five credit-history count columns are never used as
  features, because they are derived from the labels. Speaker identity is excluded
  by default.

## Repository layout

- **`Updated Improvements/`** — the corrected pipeline behind the paper: notebooks,
  the two helper modules (`liar_utils.py`, `metadata_features.py`), result CSVs, and
  figures. See [`RUN.md`](RUN.md) for the run order and which file produces each
  table and figure.
- **`artifacts/`** — the original first draft, kept unchanged as a before/after
  record. Its notebooks have a label-mapping bug: `FALSE`/`TRUE` labels are stored in
  uppercase but compared in lowercase. As a result, about 2,000 false statements were
  labelled *real*, and the reported F1 of 0.87 was an artifact of that bug. Do not
  use `artifacts/` to reproduce the paper.
- **`requirements.txt`** — pinned library versions.

## Tech stack

Python 3.9, scikit-learn 1.6.1, XGBoost 2.1.4, SHAP 0.49.1, pandas, NumPy, NLTK,
statsmodels. DistilBERT uses PyTorch and Hugging Face Transformers.

## Citation

R. Pathirana, "A Controlled and Explainable Evaluation of Classical Machine Learning
for Fake News Detection," in *Proc. ICAC 2026* (to appear).

## Author

Ravindu Pathirana, Department of Computer Science & Engineering, University of
Moratuwa, Sri Lanka.
