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
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

METADATA_COLUMNS = ["Subject(s)", "Speaker's job title", "State", "Party", "Context"]
SPEAKER_HASH_DIM = 64


def _build_canonical_categories(df, columns):
    """Fit, per column, a mapping from a whitespace/case-insensitive key to
    one canonical surface form (the most frequent stripped spelling in this
    data). Computed on train only and reused to transform test, exactly like
    the TF-IDF vocabulary and one-hot categories, so both splits agree on
    which one-hot column a given raw value maps to.

    Without this, raw metadata values that differ only by trailing
    whitespace or case (e.g. 'Georgia' vs 'Georgia ', 'ohio' vs 'Ohio') were
    one-hot encoded as separate columns, silently splitting one category's
    signal in two. Genuine spelling/wording variants (e.g. 'a chain e-mail'
    vs 'a chain email') are a different, free-text problem and are not
    touched here -- merging those would require a hand-built alias table
    rather than a mechanical, principled rule."""
    canonical = {}
    for c in columns:
        stripped = df[c].fillna("unknown").astype(str).str.strip()
        counts = stripped.value_counts()
        col_map = {}
        for val, cnt in counts.items():
            key = val.casefold()
            if key not in col_map or cnt > counts[col_map[key]]:
                col_map[key] = val
        canonical[c] = col_map
    return canonical


def _fillna_str(df, columns, canonical=None):
    """Fill missing metadata values and strip whitespace. If `canonical`
    (from `_build_canonical_categories`, fit on train) is given, also remap
    whitespace/case-only duplicate categories to their shared canonical
    spelling. A test-only value with no match in `canonical` (a category
    unseen in train) is left as its own stripped form, which is exactly
    what `OneHotEncoder(handle_unknown="ignore")` already expects."""
    df = df.copy()
    for c in columns:
        s = df[c].fillna("unknown").astype(str).str.strip()
        if canonical is not None:
            col_map = canonical[c]
            s = s.map(lambda v: col_map.get(v.casefold(), v))
        df[c] = s
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


def build_metadata_features(train_df, test_df, include_speaker=False):
    """Fit metadata encoders on train, transform test, and return
    (X_train_meta, X_test_meta, transformer).

    2026-08-24: dropped the separate `valid_df` argument -- callers now merge
    train+valid into `train_df` themselves before calling this (see
    claude-workspace/MEMORY.md), since valid was never used as a distinct
    held-out split for any decision here."""
    cols_needed = METADATA_COLUMNS + (["Speaker"] if include_speaker else [])
    non_speaker_cols = [c for c in cols_needed if c != "Speaker"]
    canonical = _build_canonical_categories(train_df, non_speaker_cols)
    train_f = _fillna_str(train_df, non_speaker_cols, canonical)
    test_f = _fillna_str(test_df, non_speaker_cols, canonical)

    transformer = build_metadata_transformer()
    X_train_meta = transformer.fit_transform(train_f)
    X_test_meta = transformer.transform(test_f)

    if include_speaker:
        X_train_meta = hstack([X_train_meta, _hash_speaker(train_df)]).tocsr()
        X_test_meta = hstack([X_test_meta, _hash_speaker(test_df)]).tocsr()

    return csr_matrix(X_train_meta), csr_matrix(X_test_meta), transformer


def combine_text_and_metadata(X_text, X_meta):
    return hstack([X_text, X_meta]).tocsr()


def build_pipeline_transformer(use_metadata=True):
    """A single flat ColumnTransformer combining TF-IDF text features with
    optional metadata features, for use as the first step of a full
    sklearn Pipeline (Phase 2, I-13). Unlike `build_metadata_features`
    (which fits encoders once, outside of cross-validation), a Pipeline
    built on top of this refits every encoder -- TF-IDF vocabulary
    included -- inside each CV fold, so hyperparameter search can no
    longer see fold-held-out vocabulary/categories.

    Expects a DataFrame with a precomputed `clean_text` column (text
    cleaning is deterministic and label-independent, so doing it once
    outside the CV loop is not a leakage risk) plus the raw metadata
    columns already filled via `_fillna_str`. Speaker is never included
    here -- see the module docstring.
    """
    transformers = [
        ("text", TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2, max_df=0.9), "clean_text")
    ]
    if use_metadata:
        transformers += [
            ("subject", CountVectorizer(tokenizer=_split_subjects, lowercase=False, binary=True), "Subject(s)"),
            ("party", OneHotEncoder(handle_unknown="ignore"), ["Party"]),
            ("state", OneHotEncoder(handle_unknown="ignore", min_frequency=5), ["State"]),
            ("job", OneHotEncoder(handle_unknown="ignore", min_frequency=5), ["Speaker's job title"]),
            ("context", OneHotEncoder(handle_unknown="ignore", min_frequency=5), ["Context"]),
        ]
    return ColumnTransformer(transformers=transformers)
