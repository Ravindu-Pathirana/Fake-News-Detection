# Issue Plan — Road to Acceptance
### Paper: "An Explainable Fake News Detection Using Classical Machine Learning"

Author: Ravindu Pathirana · Dataset: LIAR (binary) · Goal: **conference acceptance**
Plan created: 2026-08-12 · Owner: Ravindu · Status: draft v1

---

## 0. How to read this plan

Every issue has a **severity**, the **reason it triggers rejection**, the **safest fix
steps**, and a **recommended method/tool**. Work top to bottom — phases are ordered so
that later work is not wasted on numbers that a later fix would change.

| Severity | Meaning |
|---|---|
| 🔴 Blocker | Paper is not submittable until fixed; a reviewer will reject on this alone. |
| 🟠 Major | Strong reason for rejection / major revision. |
| 🟡 Minor | Polish; won't sink the paper but reviewers notice. |

**Reality check on the goal.** Once the label bug is fixed, honest text-only LIAR-binary
performance is ~0.60 accuracy / ~0.58 macro-F1. That is *normal and publishable* — LIAR is
hard. Acceptance will come from **correctness, honest analysis, a clean ablation, and the
explainability angle**, not from a big accuracy number. Aim the paper at an applied/student/
workshop venue where a rigorous, well-explained classical benchmark is a good fit (see §8).

---

## Issue register (quick view)

| ID | Sev | Issue | Fix in phase |
|----|-----|-------|--------------|
| I-1 | ✅🔴 | Label-mapping bug (`FALSE`/`TRUE` uppercase) mislabels ~20% of data | P1 — fixed |
| I-2 | ✅🔴 | Headline F1 is positive-class-only on an inflated majority → invalid | P1 — fixed |
| I-3 | 🔴 | Paper claims text+metadata feature space that the code never builds | P3 |
| I-4 | ✅🟠 | Baseline (accuracy) vs proposed (F1) not comparable | P1 — fixed |
| I-5 | ✅🟠 | No confusion matrix / per-class metrics / trivial baselines | P1 — fixed |
| I-6 | 🟠 | Single run; no variance, CIs, or significance test | P2 |
| I-7 | 🟠 | Feature-selection claim not isolated (no ablation) | P2 |
| I-8 | 🟠 | Related work stops at 2020; no modern (transformer) reference point | P4 |
| I-9 | 🟠 | Overclaims ("outperforms baseline", "feature selection > model choice") | P4 |
| I-10 | 🟡 | SHAP shows topical bias but is framed as validation | P5 |
| I-11 | 🟡 | Results section is bullet-heavy; not IEEE prose | P6 |
| I-12 | 🟡 | Figures: truncated stems, unreadable force plot | P6 |
| I-13 | 🟡 | No reproducibility artifacts (seeds, requirements, run script) | P2 |

---

## Phase 0 — Safety setup (do this first, ~30 min)

Non-destructive scaffolding so nothing you already have is lost and every result is
traceable.

1. **Branch, don't overwrite.** `git checkout -b fix/label-and-metrics`. Keep the original
   notebooks untouched as the "as-submitted" record; do corrected work in new files
   (`*_v2.ipynb`). This lets you show before/after honestly.
2. **Freeze the current results** by committing `model_comparison_results.csv` as-is so you
   can cite the buggy vs fixed contrast if useful.
3. **Pin the environment.** Create `requirements.txt` (pandas, scikit-learn, xgboost, shap,
   matplotlib, nltk) with versions. *Method:* `pip freeze > requirements.txt`.
4. **Set one global seed** and reuse it everywhere (`RANDOM_STATE = 42`).

---

## Phase 1 — 🔴 Correctness blockers (must fix before any other numbers)

### I-1 — Label-mapping bug
**Problem.** CSV labels include uppercase `FALSE`/`TRUE`; `convert_label` tests lowercase
`'false'`, so all plain-`FALSE` rows (largest fake class, ~20% of data) become "real".
**Why it rejects.** The dependent variable is wrong; every downstream number is invalid.
**Safest steps.**
1. Replace the mapping with a case/whitespace-robust version:
   ```python
   FAKE = {"false", "pants-fire", "barely-true"}
   def convert_label(l): return 0 if str(l).strip().lower() in FAKE else 1
   ```
2. **Add an assertion guard** so this can never silently recur:
   ```python
   raw = set(df["Label"].str.strip().str.lower().unique())
   assert raw <= {"false","pants-fire","barely-true","half-true","mostly-true","true"}, raw
   ```
3. Print the class balance after mapping and confirm ≈ 44% fake / 56% real.
**Recommended method.** Normalize once in a single `load_and_label()` helper reused by every
notebook, so baseline and proposed can never diverge.

### I-2 — Invalid headline metric
**Problem.** F1 uses default `pos_label=1` (the inflated majority). Trivial "predict real"
scores ~0.86.
**Why it rejects.** Reported metric rewards majority-class collapse (RF/XGB recall = 1.000).
**Safest steps.**
1. After the label fix, **report macro-F1 as the primary metric**, plus accuracy, and
   **per-class precision/recall/F1** and a **confusion matrix** for every model.
2. Also report **PR-AUC / ROC-AUC** for the fake class (the class you actually care about).
**Recommended method.** `sklearn.metrics.classification_report(..., digits=3)` +
`confusion_matrix`; for AUC use `average_precision_score` / `roc_auc_score` on
`predict_proba` (LinearSVC has none → use `CalibratedClassifierCV` or `decision_function`).

### I-4 — Non-comparable baseline vs proposed
**Problem.** Table I = accuracy, Table II = F1; declared "not comparable."
**Safest steps.** Report the **same metric set** (accuracy, macro-F1, fake-class P/R/F1) for
baseline *and* proposed, on the **same corrected binary labels**, in one merged table.
**Recommended method.** One evaluation function returning a dict; build both tables from it.

### I-5 — Missing trivial baselines & diagnostics
**Problem.** No reference point proving the models beat "predict majority".
**Safest steps.** Add two baselines to every table:
`DummyClassifier(strategy="most_frequent")` and `strategy="stratified"`, plus a
`class_weight="balanced"` variant of LR/SVM.
**Recommended method.** Show that macro-F1 of your models clearly exceeds the Dummy macro-F1;
that single comparison is what makes the honest ~0.58 result meaningful.

**Exit criterion for Phase 1:** a corrected results table (baseline + proposed + dummies),
macro-F1 primary, with confusion matrices. This is the new factual core of the paper.

**STATUS (2026-08-23): Phase 1 complete.** Implemented in `artifacts/liar_utils.py`
(shared `load_and_label`, `evaluate_full`, `print_report`, `RANDOM_STATE=42`),
`artifacts/baseline_pipeline_v2.ipynb`, and `artifacts/proposed_improvements_v2.ipynb`.
Both notebooks executed cleanly end-to-end (`python3 -m nbconvert --execute`); merged
table saved to `artifacts/model_comparison_results_v2.csv`. Corrected train class
balance: 43.8% fake / 56.2% real (assertion guard in both notebooks enforces this).
Best result: Logistic Regression + Mutual Information, test macro-F1 = 0.619 (test
acc 0.622, fake-class P/R/F1 = 0.561/0.615/0.587) — in the honest ~0.58-0.62 range
predicted in the Appendix, not the invalid buggy 0.87. Every real model clearly beats
both Dummy baselines (most-frequent macro-F1 0.360, stratified 0.485), satisfying I-5.
Baseline and proposed are now reported on the identical metric set (I-4). Original
`baseline_pipeline_notebook.ipynb` / `proposed_improvements_notebook.ipynb` and
`model_comparison_results.csv` were left untouched as the "as-submitted" record.
Next: I-3 (Phase 3, metadata claim reconciliation) gates the rest per the critical
path in §8.

---

## Phase 2 — 🟠 Methodological integrity

### I-13 — Reproducibility harness
**Steps.** Convert the ad-hoc notebook flow into a single, seeded pipeline; save all outputs
to versioned CSVs; write a short `RUN.md`. **Method:** wrap TF-IDF → SelectKBest → model in a
`sklearn.pipeline.Pipeline` so preprocessing is fit inside each CV fold automatically (this
also makes leakage structurally impossible — see below).

### I-7 — Isolate the feature-selection claim (ablation)
**Problem.** Gains conflate metadata + feature selection + extra models + tuning.
**Steps.** Run a controlled ablation, one metric (macro-F1), same splits:
`text-only → +feature selection → +tuning (→ +metadata if kept)`.
**Recommended method.** Configure variants via `Pipeline` + `GridSearchCV`; present as one
ablation table. Only claim what the table shows.

### I-6 — Statistical rigor
**Problem.** Single run; 0.02 gaps reported as findings.
**Steps.**
1. Repeat each config over ≥5 seeds; report **mean ± std**.
2. For the headline model-vs-model / feature-selection comparison, run a significance test.
**Recommended method.** **McNemar's test** (`statsmodels.stats.contingency_tables.mcnemar`)
for two classifiers on the same test set; **bootstrap 95% CIs** on macro-F1 for the reported
best model. Don't claim a difference that isn't significant.

### Leakage double-check (currently clean — keep it that way)
The code is not leaky today (TF-IDF/SelectKBest fit on train only; credit-count columns
excluded). Putting everything in a single `Pipeline` guarantees it stays clean when you
refactor. **Do not** add the `*_counts` columns as features — they leak the label.

---

## Phase 3 — 🔴/🟠 Reconcile paper claims with code (I-3)

**Problem.** The paper describes a combined TF-IDF + one-hot metadata feature space
(speaker, subject, party, context, job, state). The notebooks use text only. As written this
is an unsupported claim — a serious integrity issue.
**Decision point — pick one:**
- **Option A (stronger paper): actually implement metadata features.** One-hot encode
  `party`, `subject`, `state`, `job`, `context`; combine with TF-IDF via
  `sklearn.compose.ColumnTransformer` + `scipy.sparse.hstack`. **Cautions:** (1) exclude the
  five credit-count columns (leakage); (2) treat `speaker` carefully — high-cardinality
  one-hot of speaker identity lets the model memorize *who* rather than *what*, which is a
  softer form of leakage; prefer dropping speaker or hashing it, and report the effect
  explicitly. Then metadata becomes a legitimate ablation row.
- **Option B (safest, fastest): remove the metadata claim** and present an honest text-only
  study. Rewrite Sec IV-B/IV-D accordingly.

**Recommendation.** Option A if you have time (it gives you a real second contribution and a
believable lift over text-only); otherwise Option B — never ship a claim the code doesn't do.

---

## Phase 4 — 🟠 Positioning & claims

### I-8 — Related work / modern reference point
**Steps.** Add 2021–2025 references and at least one **transformer baseline** for context
(even a single fine-tuned **DistilBERT/BERT** number, or a cited one). Frame classical ML as
the *interpretable, low-resource* alternative, not the state of the art.
**Recommended method.** If compute is limited, fine-tune `distilbert-base-uncased` for 2–3
epochs on the binary task via HuggingFace `Trainer`; report its macro-F1 as an upper-reference
row. If not feasible, cite published LIAR transformer results and position accordingly.

### I-9 — Rewrite overclaims
**Steps.** Replace "F1 up to 0.87 outperforming baseline" and "feature selection matters more
than model choice" with claims the corrected ablation + significance tests actually support
(e.g., "modest, statistically-significant gains from MI feature selection; model choice has
limited effect at this feature quality"). State the honest macro-F1 up front.

---

## Phase 5 — 🟡 Explainability, reframed (I-10)

**Problem.** Top Chi²/MI/SHAP features are topical/entity words (obama, obamacar, socialist,
say, state) — evidence the model keys on *topic/speaker*, i.e. LIAR's topical bias, not
deception cues. Currently framed as validation.
**Steps.**
1. Reframe as an **honest limitation / dataset-bias finding** — this is genuinely
   interesting and reviewers respect it.
2. Add **per-class SHAP** (what pushes toward *fake* specifically) and, if possible, a couple
   of qualitative examples.
**Recommended method.** `shap.TreeExplainer` on the corrected best model; global bar +
beeswarm; discuss that interpretability reveals bias, motivating future contextual features.

---

## Phase 6 — 🟡 Writing & figures

- **I-11:** Convert the bullet-list Results/Analysis sections to IEEE prose paragraphs.
  Keep tables; drop slide-style bullets.
- **I-12:** Fix figures — show readable feature labels (avoid over-stemmed fragments like
  "obamacar", "feder reserv"; display un-stemmed forms for the plots), enlarge/replace the
  SHAP force plot with a legible example, add axis labels and captions that state the metric.
- Tighten structure: Sections IV/V/VI repeat the same model lists — merge.
- Update the abstract to lead with the corrected, honest headline.

---

## Phase 7 — Pre-submission verification checklist

- [ ] `assert` on label set passes; class balance ≈ 44/56 printed.
- [ ] Every table uses the same corrected labels and the same metric set.
- [ ] Macro-F1 is primary; per-class P/R/F1 + confusion matrices included.
- [ ] Dummy + class-weighted baselines present and beaten.
- [ ] Ablation table isolates each component's contribution.
- [ ] Mean ± std over seeds; significance test on the headline comparison.
- [ ] No credit-count columns used; speaker handling documented.
- [ ] Every claim in abstract/conclusion traces to a table.
- [ ] Related work includes 2021–2025 + one transformer reference point.
- [ ] SHAP framed as bias/limitation; per-class explanation added.
- [ ] Prose (not bullets) in Results; figures legible.
- [ ] `requirements.txt`, seed, and a one-command run script committed.
- [ ] **Independent re-run reproduces every reported number** (recommend a fresh-kernel run,
      or a second pair of eyes / verification pass).

---

## 8. Suggested order & venue note

**Critical path:** P0 → P1 (label + metrics) → P3 decision (metadata in or out) →
P2 (ablation + stats) → P4 (positioning) → P5 → P6 → P7.

P1 and P3 gate everything; do them before writing prose. P4–P6 are writing-heavy and safe to
do once the numbers are frozen.

**Venue:** with an honest ~0.60 macro-F1, target an applied ML / NLP / student / workshop
venue that values rigor + interpretability over leaderboard numbers, rather than a top-tier
NLP conference. A correct, well-analyzed, reproducible study with an honest limitations
discussion is very acceptable there — an inflated 0.87 at a competitive venue is not.

---

## Appendix — evidence for the blocker (I-1/I-2)

Verified rerun (LR, TF-IDF 5k unigrams, sklearn English stopwords, LIAR test set):

| Version | Test balance (real/fake) | Accuracy | F1 (pos-only) | Macro-F1 |
|---|---|---|---|---|
| Buggy (as submitted) | 76 / 24 | 0.761 | 0.862 | 0.474 |
| Fixed (case-insensitive) | 56 / 44 | 0.606 | 0.680 | 0.584 |

Results CSV corroboration: RF and XGBoost show test **recall = 1.000**, precision = 0.760 —
the literal all-positive classifier that the buggy F1 rewards.
