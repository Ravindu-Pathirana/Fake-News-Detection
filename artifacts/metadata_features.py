"""Metadata feature engineering for the LIAR dataset (Phase 3, issue I-3).

The paper claims a combined TF-IDF + one-hot-metadata feature space
(speaker, subject, party, context, job, state) that the original notebooks
never build. This module implements it for real, so the claim in the paper
matches the code (Option A from claude-workspace/ISSUE_PLAN.md Phase 3).

Design notes (see ISSUE_PLAN.md Phase 3 for the full rationale):
  - The five `*_counts` credit-history columns are NEVER used here -- they
    are computed from the label history and leak the target (see the
    Phase 2 "Leakage double-check" note).
  - `Subject(s)` is multi-label (comma-separated, e.g. "energy,history"),
    so it is binarized per subject token, not one-hot'd as a single string.
  - `Party`, `State`, `Speaker's job title`, and `Context` are one-hot
    encoded with `handle_unknown="ignore"` (so unseen valid/test categories
    degrade gracefully to an all-zero row) and `min_frequency` to collapse
    rare categories -- this matters most for `Context`, which is almost as
    high-cardinality as the number of rows and would otherwise memorize
    near-unique venue strings.
  - `Speaker` (2,910 unique values in train) is deliberately NOT included
    by default: a high-cardinality one-hot of speaker identity lets a
    model memorize *who* said something rather than learn *what* was said,
    which is a softer form of leakage. When `include_speaker=True`, it is
    hashed into a small fixed-width space (`FeatureHasher`) rather than
    one-hot encoded, so its effect can still be measured and reported
    explicitly without giving the model an identity lookup table.
"""

from scipy.sparse import csr_matrix, hstack
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder

METADATA_COLUMNS = ["Subject(s)", "Speaker's job title", "State", "Party", "Context"]
SPEAKER_HASH_DIM = 64


def _fillna_str(df, columns):
    df = df.copy()
    for c in columns:
        df[c] = df[c].fillna("unknown").astype(str)
    return df


def _split_subjects(text):
    return [s.strip() for s in text.split(",") if s.strip()]


def build_metadata_transformer():
    """A ColumnTransformer over the non-leaking, non-identity metadata
    columns. Fit on train only, like the TF-IDF vectorizer."""
    subject_vectorizer = CountVectorizer(tokenizer=_split_subjects, lowercase=False, binary=True)
    low_card_ohe = OneHotEncoder(handle_unknown="ignore")
    high_card_ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=5)

    return ColumnTransformer(
        transformers=[
            ("subject", subject_vectorizer, "Subject(s)"),
            ("party", low_card_ohe, ["Party"]),
            ("state", high_card_ohe, ["State"]),
            ("job", high_card_ohe, ["Speaker's job title"]),
            ("context", high_card_ohe, ["Context"]),
        ]
    )


def _hash_speaker(df):
    # alternate_sign=False: MultinomialNB requires non-negative input, and the
    # default signed hashing would otherwise silently break it on every fold.
    hasher = FeatureHasher(n_features=SPEAKER_HASH_DIM, input_type="string", alternate_sign=False)
    return hasher.transform([[s] for s in df["Speaker"].fillna("unknown").astype(str)])


def build_metadata_features(train_df, valid_df, test_df, include_speaker=False):
    """Fit metadata encoders on train, transform all three splits, and
    return (X_train_meta, X_valid_meta, X_test_meta, transformer)."""
    cols_needed = METADATA_COLUMNS + (["Speaker"] if include_speaker else [])
    train_f = _fillna_str(train_df, [c for c in cols_needed if c != "Speaker"])
    valid_f = _fillna_str(valid_df, [c for c in cols_needed if c != "Speaker"])
    test_f = _fillna_str(test_df, [c for c in cols_needed if c != "Speaker"])

    transformer = build_metadata_transformer()
    X_train_meta = transformer.fit_transform(train_f)
    X_valid_meta = transformer.transform(valid_f)
    X_test_meta = transformer.transform(test_f)

    if include_speaker:
        X_train_meta = hstack([X_train_meta, _hash_speaker(train_df)]).tocsr()
        X_valid_meta = hstack([X_valid_meta, _hash_speaker(valid_df)]).tocsr()
        X_test_meta = hstack([X_test_meta, _hash_speaker(test_df)]).tocsr()

    return csr_matrix(X_train_meta), csr_matrix(X_valid_meta), csr_matrix(X_test_meta), transformer


def combine_text_and_metadata(X_text, X_meta):
    return hstack([X_text, X_meta]).tocsr()
