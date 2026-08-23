# Paper Review Findings — Notebook Inspection

Date: 2026-08-12
Paper: "An Explainable Fake News Detection Using Classical Machine Learning"
Artifacts inspected: `baseline_pipeline_notebook.ipynb`, `proposed_improvements.ipynb`,
`identify_datasets_notebook.ipynb`, `train/valid/test.csv`, `model_comparison_results.csv`.

## 1. Critical: label-mapping bug invalidates the headline result

**Root cause.** The LIAR CSVs store two labels in uppercase — `FALSE` and `TRUE` —
(likely from a spreadsheet auto-casting). The mapping function is:

```python
def convert_label(label):
    if label in ['false', 'pants-fire', 'barely-true']:
        return 0  # fake
    else:
        return 1  # real
```

Python comparison is case-sensitive, so `'FALSE' != 'false'`. Every plain-`FALSE`
statement — the single largest "fake" category — falls to `else` and is labeled **real**.

**Impact on label counts (train):**
- Mislabeled: `FALSE` = 1,995 rows (19.5% of training data) flipped fake -> real.
- Resulting balance: 76% real / 24% fake (should be ~44% fake / 56% real).

**Impact on metrics.** F1 is computed with the default `pos_label=1` (the "real",
now-majority class). A model that predicts "real" for everything therefore scores
F1 ≈ 2·0.76·1 / (0.76+1) ≈ 0.863. The saved `model_comparison_results.csv` shows
Random Forest and XGBoost with **test recall = exactly 1.000** and precision = 0.760 —
i.e. literal all-positive classifiers. That is the paper's "best model, F1 0.87".

**Verification rerun** (LR, TF-IDF 5k unigrams, sklearn English stopwords):

| Version | Test balance | Accuracy | F1 (pos-only) | Macro-F1 |
|---|---|---|---|---|
| Buggy (as submitted) | 76 / 24 | 0.761 | 0.862 | 0.474 |
| Fixed (case-insensitive) | 56 / 44 | 0.606 | 0.680 | 0.584 |

The reported 0.86 corresponds to a **macro-F1 of 0.47** — failure on the minority
(fake) class hidden by positive-class-only scoring. Fixed, honest performance is
~0.60 accuracy / ~0.58 macro-F1, consistent with published text-only LIAR-binary work.

**Fix.** `0 if str(label).strip().lower() in ['false','pants-fire','barely-true'] else 1`,
then rerun everything and report macro-F1 + per-class precision/recall + confusion matrix.

## 2. No data leakage elsewhere (good)
- TF-IDF `fit_transform` on train only; `transform` on valid/test. Correct.
- `SelectKBest` (chi2 / mutual_info) fit on train only. Correct.
- Credit-history count columns (`Barely true counts`, ... `Pants on fire counts`) are
  present in the CSV but NOT used as features. Correct (these are the classic LIAR leak).

## 3. Paper/code mismatch: metadata features
- Paper (Sec IV-B "Meta-data Features", Sec IV-D) states the proposed approach builds a
  combined feature space of TF-IDF text + one-hot encoded speaker, subject, party, context,
  job, state.
- Neither notebook constructs this. Both use ONLY `clean_text`. The described
  "text + meta-data" contribution does not exist in the code as submitted.

## 4. Metric / comparison problems
- Baseline reported in accuracy (Table I); proposed in F1 (Table II) -> not comparable,
  which undercuts the "outperforms baseline" claim.
- Single run, no std/CI, no significance test.
- "Feature selection matters more than model choice" is asserted from ~0.02 gaps that are
  actually all models collapsing toward the same majority predictor.

## 5. Interpretability caveat
- Top Chi2 / MI / SHAP features are topical/entity terms (obama, obamacar, rep, socialist,
  say, state). These indicate the model keys on *who/what topic* is being fact-checked,
  i.e. dataset topical bias — not deception signals. Should be framed as a limitation,
  not as validation.

## 6. Recommended fix order
1. Fix labels -> rerun -> report macro-F1, per-class metrics, confusion matrices,
   plus majority-class and class-weighted baselines.
2. Reconcile the metadata claim (either implement metadata features or remove the claim).
3. Add an ablation (text -> +metadata -> +feature selection -> +tuning), one metric.
4. Repeated runs (seeds) with mean ± std; significance test on the headline comparison.
5. Refresh related work (2021-2025) and add one modern (BERT) reference point.
6. Convert bullet-heavy Results section to prose; fix truncated-stem figure labels.
