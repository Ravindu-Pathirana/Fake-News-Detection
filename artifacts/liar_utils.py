"""Shared config/helpers for the LIAR fake-news-detection notebooks.

Import this from every v2 notebook so the random seed, label mapping, and
evaluation metrics can never diverge between the baseline and proposed
pipelines. See claude-workspace/ISSUE_PLAN.md Phase 0/1 (issues I-1, I-2,
I-4, I-5).
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

RANDOM_STATE = 42

# LIAR labels, collapsed to binary: 0 = fake, 1 = real.
FAKE_LABELS = {"false", "pants-fire", "barely-true"}
ALL_LABELS = {"false", "pants-fire", "barely-true", "half-true", "mostly-true", "true"}


def convert_label(label):
    """0 = fake, 1 = real.

    Case/whitespace-robust: the raw CSVs store `FALSE`/`TRUE` in uppercase
    but the other four labels in lowercase. A naive `label in [...]` check
    against lowercase strings silently mislabels every `FALSE` row as real
    (~1,995 rows in train alone) -- see ISSUE_PLAN.md I-1.
    """
    return 0 if str(label).strip().lower() in FAKE_LABELS else 1


def load_and_label(path):
    """Load a LIAR split CSV and add a binary Label column, with a guard
    against silently accepting an unexpected raw label value."""
    df = pd.read_csv(path)
    raw = set(df["Label"].str.strip().str.lower().unique())
    unexpected = raw - ALL_LABELS
    assert not unexpected, f"Unexpected label values in {path}: {unexpected}"
    df["Label"] = df["Label"].apply(convert_label)
    return df


def evaluate_full(y_true, y_pred):
    """One evaluation dict, reused by every model/notebook so baseline and
    proposed results are always reported on the same metric set (I-4).

    Macro-F1 is the primary metric (I-2); per-class P/R/F1 for the fake
    class (label 0) is reported separately since that's the class the
    naive positive-class-only F1 was inflating.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "fake_precision": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "fake_recall": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "fake_f1": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "real_precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "real_recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "real_f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def print_report(name, y_true, y_pred):
    print(f"\n{name}")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=["fake", "real"], digits=3))
