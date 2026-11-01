# %% [markdown]
# # DistilBERT fair reference (v2) — camera-ready issue CR-3
#
# Fine-tunes `distilbert-base-uncased` under the **same data conditions as the
# classical models**, answering all three reviewers (CR-3 in
# `claude-workspace/CLAUDE_CAMERA_READY_PLAN.md`):
#
# - **Same fitting pool:** LIAR train + valid merged (11,524 rows), the pool every
#   classical model is fit on. The 1,267-row test split is touched **once**, after
#   all selection is finished.
# - **Tuned, not fixed:** learning rate x epochs is chosen on a stratified 10%
#   inner hold-out carved from the pool (the transformer analogue of the classical
#   models' internal CV). The model is then refit on the full pool with the chosen
#   setting.
# - **Like-for-like inputs:** `text` (statement only, compare with text-only
#   classical rows) and `text_meta` (statement + the same five metadata columns
#   NB/LR use, serialized as text; compare with text+metadata classical rows).
#   `Speaker` and the five `*_counts` credit-history columns are never used.
# - **Seed variance:** the chosen setting is refit with 3 seeds. Seed 42 (the
#   project-wide `RANDOM_STATE`) is the pre-declared primary run for the paired
#   McNemar tests, which are run locally in the significance notebook (CR-4), not
#   here. This notebook deliberately computes **no p-values**, so no uncorrected
#   test can leak into the paper.
#
# **Kaggle setup.** Add the input dataset `uom230467r/liar-dataset` (expected at
# `/kaggle/input/datasets/uom230467r/liar-dataset/`; set `LIAR_DATA_DIR` to override).
# It may hold either the raw LIAR release (`train/valid/test.tsv`) or the project's
# `train/valid/test.csv`; both are found, including inside subfolders. The loader parses the TSVs so that they
# reproduce the project's CSVs row for row, and checks this against fingerprints,
# stopping if they don't match (see `PARSE_MODE` in the data cell). Then:
# Settings → Accelerator **GPU T4 x2 or P100** (only one GPU is used) → Internet
# **On** (downloads `distilbert-base-uncased`, ~260 MB) → Run All. Expected
# runtime is roughly 1–2 h for both variants on a T4. Results are written
# incrementally to `/kaggle/working/`, so re-running after a session timeout skips
# finished runs. Download everything in `/kaggle/working/` and copy it into
# `Updated Improvements/`.
#
# Local smoke test (tiny subsample, 1 epoch, no GPU needed): `SMOKE=1`.

# %%
import os

# Must be set before torch is imported.
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")  # one GPU: no DataParallel
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # deterministic cuBLAS
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import csv
import glob
import hashlib
import json
import platform
import random
import sys
import time

import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

SMOKE = os.environ.get("SMOKE") == "1"
RANDOM_STATE = 42
MODEL_NAME = "distilbert-base-uncased"

# Hyperparameter grid and protocol, fixed before any run (do not edit after seeing results).
LRS = [2e-5, 3e-5, 5e-5]
EPOCHS = [2, 3, 4]
FINAL_SEEDS = [42, 43, 44]
PRIMARY_SEED = 42
INNER_HOLDOUT = 0.10
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 64
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.0
MAX_GRAD_NORM = 1.0
MAX_LENGTH = 128
VARIANTS = ["text", "text_meta"]
# Optional (CR-5 cross-check): add "text_meta_nojob" to VARIANTS to drop Speaker's job title.

if SMOKE:
    LRS, EPOCHS, FINAL_SEEDS = [5e-5], [1], [42]

ON_KAGGLE = os.path.exists("/kaggle/working")
OUT_DIR = "/kaggle/working" if ON_KAGGLE else ("distilbert_v2_smoke" if SMOKE else ".")
if os.environ.get("PARSE_MODE", "project") == "official":  # never mix the two parses' outputs
    OUT_DIR = os.path.join(OUT_DIR, "official_parse")
os.makedirs(OUT_DIR, exist_ok=True)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
USE_AMP = DEVICE.type == "cuda"
AMP_DEVICE = "cuda" if USE_AMP else "cpu"  # autocast is disabled off-CUDA anyway
print("Device:", DEVICE, "| AMP fp16:", USE_AMP, "| smoke:", SMOKE, "| out:", OUT_DIR)

# %% [markdown]
# ## Data and labels
# Label mapping is an exact copy of `liar_utils.convert_label` / `load_and_label`
# (0 = fake, 1 = real), including the guard against unexpected raw labels. The
# uppercase `FALSE`/`TRUE` handling is the reason this must not be simplified.

# %%
FAKE_LABELS = {"false", "pants-fire", "barely-true"}
ALL_LABELS = {"false", "pants-fire", "barely-true", "half-true", "mostly-true", "true"}
METADATA_COLUMNS = ["Subject(s)", "Speaker's job title", "State", "Party", "Context"]  # = metadata_features.METADATA_COLUMNS


def convert_label(label):
    return 0 if str(label).strip().lower() in FAKE_LABELS else 1


def load_and_label(df, name):
    raw = set(df["Label"].str.strip().str.lower().unique())
    unexpected = raw - ALL_LABELS
    assert not unexpected, f"Unexpected label values in {name}: {unexpected}"
    df["Label"] = df["Label"].apply(convert_label)
    return df


# Raw LIAR release (train/valid/test.tsv: no header, 14 tab-separated columns).
LIAR_COLUMNS = [
    "ID", "Label", "Statement", "Subject(s)", "Speaker", "Speaker's job title", "State", "Party",
    "Barely true counts", "False counts", "Half true counts", "Mostly true counts", "Pants on fire counts", "Context",
]
DATA_DIR = os.environ.get("LIAR_DATA_DIR", "/kaggle/input/datasets/uom230467r/liar-dataset")

# PARSE_MODE decides which rows exist, and must match the classical pipeline:
#   "project"  -- pandas' default quote handling. Reproduces the project's train/valid/test.csv
#                 exactly (10,240 / 1,284 / 1,267 rows), i.e. the data every classical result in
#                 the paper uses. Under this parse, 8 rows (6 train, 2 test) whose statement contains
#                 a stray '"' absorb the next 45 rows into one long Statement string.
#   "official" -- csv.QUOTE_NONE, the release as published (10,269 / 1,284 / 1,283 rows, the
#                 sizes in the original LIAR paper). Only use it if the classical pipeline is
#                 re-run on the same parse (plan gate D6), otherwise the comparison is not like-for-like.
PARSE_MODE = os.environ.get("PARSE_MODE", "project")

# Fingerprints (row count, md5 of the ID column in file order, raw label counts), computed
# 2026-11-01 from the project CSVs and the official UCSB release. A mismatch means this is a
# different LIAR copy, and the results would not be comparable.
EXPECTED = {
    "project": {
        "train": (10240, "fa7ac543aeb0e26efc14613e5c11deed"),
        "valid": (1284, "4455ce8961181696f46b7a964ed748a6"),
        "test": (1267, "511c770a7847e5a656aa57cf6d4bea88"),
    },
    "official": {
        "train": (10269, "73ca7b4a18f0eb512b75e93d49c7b178"),
        "valid": (1284, "4455ce8961181696f46b7a964ed748a6"),
        "test": (1283, "9cd9e9a2651e279cbd532153b507ae0c"),
    },
}
assert PARSE_MODE in EXPECTED, f"PARSE_MODE must be one of {list(EXPECTED)}"


def find_file(split):
    """First {split}.tsv or {split}.csv (a file, not a folder) under DATA_DIR, then anywhere
    under /kaggle/input, then the working directory. Subfolders are searched too, because
    Kaggle sometimes nests uploaded files one level down."""
    for root in (DATA_DIR, "/kaggle/input", "."):
        if not os.path.isdir(root):
            continue
        for ext in ("tsv", "csv"):
            hits = sorted(h for h in glob.glob(os.path.join(root, "**", f"{split}.{ext}"), recursive=True)
                          if os.path.isfile(h) and "distilbert_v2" not in h)
            if hits:
                return hits[0]
    found = sorted(glob.glob(os.path.join(DATA_DIR, "**", "*"), recursive=True))[:30] if os.path.isdir(DATA_DIR) else []
    raise FileNotFoundError(
        f"No {split}.tsv or {split}.csv found. DATA_DIR={DATA_DIR!r} "
        f"({'exists' if os.path.isdir(DATA_DIR) else 'does not exist'}); its contents: {found}")


def read_split(split):
    path = find_file(split)
    if path.endswith(".csv"):  # the project CSVs are already the "project" parse
        assert PARSE_MODE == "project", "Project CSVs only support PARSE_MODE='project'"
        df = pd.read_csv(path)
    else:
        quoting = csv.QUOTE_NONE if PARSE_MODE == "official" else csv.QUOTE_MINIMAL
        df = pd.read_csv(path, sep="\t", header=None, names=LIAR_COLUMNS, quoting=quoting)
    n, md5 = EXPECTED[PARSE_MODE][split]
    got = hashlib.md5("\n".join(df["ID"].astype(str)).encode()).hexdigest()
    assert len(df) == n and got == md5, (
        f"{split}: {len(df)} rows / ID md5 {got}, expected {n} / {md5} for PARSE_MODE={PARSE_MODE!r} ({path})")
    print(f"{split}: {path} | {len(df)} rows | fingerprint OK ({PARSE_MODE})")
    return load_and_label(df, path)


train_df = read_split("train")
valid_df = read_split("valid")
test_df = read_split("test")

pool_df = pd.concat([train_df, valid_df], ignore_index=True)  # same 11,524-row pool as the classical models
fake_share = (pool_df["Label"] == 0).mean()
assert 0.35 < fake_share < 0.5, f"Fake share {fake_share:.3f} outside the expected ~44% -- check label mapping"
print(f"Pool: {len(pool_df)} rows, fake share {fake_share:.3f} | Test: {len(test_df)} rows")

if SMOKE:
    pool_df = pool_df.sample(400, random_state=RANDOM_STATE).reset_index(drop=True)
    test_df = test_df.sample(120, random_state=RANDOM_STATE).reset_index(drop=True)


def clean(value):
    s = "unknown" if pd.isna(value) else str(value).strip()
    return s if s else "unknown"


def build_inputs(df, variant):
    """Returns (first_segment, second_segment_or_None). Metadata goes in the first
    segment, the statement in the second, giving [CLS] meta [SEP] statement [SEP]."""
    statements = df["Statement"].map(clean).tolist()
    if variant == "text":
        return statements, None
    cols = METADATA_COLUMNS if variant == "text_meta" else [c for c in METADATA_COLUMNS if c != "Speaker's job title"]
    names = {"Subject(s)": "subject", "Speaker's job title": "job", "State": "state", "Party": "party", "Context": "context"}
    meta = [" | ".join(f"{names[c]}: {clean(row[c])}" for c in cols) for _, row in df[cols].iterrows()]
    return meta, statements


inner_train_idx, inner_val_idx = train_test_split(
    np.arange(len(pool_df)), test_size=INNER_HOLDOUT, stratify=pool_df["Label"], random_state=RANDOM_STATE
)
print(f"Inner split: {len(inner_train_idx)} train / {len(inner_val_idx)} hold-out (stratified, seed {RANDOM_STATE})")

# %% [markdown]
# ## Training utilities
# A plain PyTorch loop (no `Trainer`), so behaviour doesn't drift with
# `transformers` API changes. It mirrors the v1 notebook's `Trainer` defaults:
# AdamW, weight decay 0.01 (not on bias/LayerNorm), linear decay, no warmup,
# grad-norm clip 1.0.

# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
collator = DataCollatorWithPadding(tokenizer)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def encode(df, variant):
    first, second = build_inputs(df, variant)
    enc = tokenizer(first, second, truncation=True, max_length=MAX_LENGTH)
    untruncated = tokenizer(first, second, truncation=False)["input_ids"]
    truncated = float(np.mean([len(ids) > MAX_LENGTH for ids in untruncated]))
    items = [{"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i]} for i in range(len(df))]
    return items, truncated


def make_loader(items, labels=None, shuffle=False, seed=RANDOM_STATE, batch_size=BATCH_SIZE):
    rows = [dict(it, labels=int(y)) for it, y in zip(items, labels)] if labels is not None else list(items)
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(rows, batch_size=batch_size, shuffle=shuffle, collate_fn=collator, generator=gen)


def train_model(items, labels, lr, epochs, seed):
    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)
    loader = make_loader(items, labels, shuffle=True, seed=seed)
    no_decay = ("bias", "LayerNorm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)], "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=lr)
    total = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP_RATIO * total), total)
    scaler = torch.amp.GradScaler(AMP_DEVICE, enabled=USE_AMP)
    model.train()
    for _ in range(epochs):
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with torch.autocast(device_type=AMP_DEVICE, dtype=torch.float16, enabled=USE_AMP):
                loss = model(**batch).loss
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
    return model


@torch.no_grad()
def predict_proba(model, items):
    model.eval()
    probs = []
    for batch in make_loader(items, batch_size=EVAL_BATCH_SIZE):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.autocast(device_type=AMP_DEVICE, dtype=torch.float16, enabled=USE_AMP):
            logits = model(**batch).logits
        probs.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(probs)  # P(real)


def evaluate_full(y_true, y_pred):  # = liar_utils.evaluate_full
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "fake_precision": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "fake_recall": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "fake_f1": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "real_precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "real_recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "real_f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def load_done(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def append_row(path, row):
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)

# %% [markdown]
# ## Step 1 — select (lr, epochs) on the inner hold-out (test split untouched)

# %%
GRID_CSV = os.path.join(OUT_DIR, "distilbert_v2_grid.csv")
y_pool = pool_df["Label"].to_numpy()
encoded_pool, truncation = {}, {}
for variant in VARIANTS:
    encoded_pool[variant], truncation[variant] = encode(pool_df, variant)
    print(f"{variant}: {truncation[variant]:.2%} of pool inputs exceed {MAX_LENGTH} tokens (truncated)")

for variant in VARIANTS:
    items = encoded_pool[variant]
    tr_items = [items[i] for i in inner_train_idx]
    va_items = [items[i] for i in inner_val_idx]
    for lr in LRS:
        for ep in EPOCHS:
            done = load_done(GRID_CSV)
            if len(done) and ((done["variant"] == variant) & np.isclose(done["lr"], lr) & (done["epochs"] == ep)).any():
                continue
            t0 = time.time()
            model = train_model(tr_items, y_pool[inner_train_idx], lr, ep, RANDOM_STATE)
            pred = (predict_proba(model, va_items) >= 0.5).astype(int)
            m = evaluate_full(y_pool[inner_val_idx], pred)
            append_row(GRID_CSV, {"variant": variant, "lr": lr, "epochs": ep, "seed": RANDOM_STATE,
                                  "inner_val_macro_f1": m["macro_f1"], "inner_val_accuracy": m["accuracy"],
                                  "minutes": round((time.time() - t0) / 60, 2)})
            print(f"[grid] {variant} lr={lr:g} ep={ep}: inner macro-F1 {m['macro_f1']:.4f} ({(time.time() - t0) / 60:.1f} min)")
            del model
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

grid = load_done(GRID_CSV)
# Highest inner macro-F1; ties -> fewer epochs, then lower lr (cheaper, less overfit-prone).
best = {}
for variant in VARIANTS:
    g = grid[grid["variant"] == variant].sort_values(["inner_val_macro_f1", "epochs", "lr"], ascending=[False, True, True])
    best[variant] = g.iloc[0]
    print(f"Selected for {variant}: lr={best[variant]['lr']:g}, epochs={int(best[variant]['epochs'])} "
          f"(inner macro-F1 {best[variant]['inner_val_macro_f1']:.4f})")
grid.pivot_table(index=["variant", "lr"], columns="epochs", values="inner_val_macro_f1").round(4)

# %% [markdown]
# ## Step 2 — refit on the full 11,524-row pool with the selected setting, 3 seeds; score test once

# %%
RESULTS_CSV = os.path.join(OUT_DIR, "distilbert_v2_results.csv")
PRED_CSV = os.path.join(OUT_DIR, "distilbert_v2_test_predictions.csv")
y_test = test_df["Label"].to_numpy()
preds = pd.read_csv(PRED_CSV) if os.path.exists(PRED_CSV) else pd.DataFrame({"ID": test_df["ID"], "y_true": y_test})
assert preds["ID"].tolist() == test_df["ID"].tolist(), "Saved predictions are not in test.csv order"

for variant in VARIANTS:
    test_items, test_trunc = encode(test_df, variant)
    lr, ep = float(best[variant]["lr"]), int(best[variant]["epochs"])
    for seed in FINAL_SEEDS:
        col = f"pred__{variant}__seed{seed}"
        if col in preds.columns:
            continue
        t0 = time.time()
        model = train_model(encoded_pool[variant], y_pool, lr, ep, seed)
        p_real = predict_proba(model, test_items)
        pred = (p_real >= 0.5).astype(int)
        m = evaluate_full(y_test, pred)
        preds[col] = pred
        preds[f"p_real__{variant}__seed{seed}"] = p_real.round(6)
        preds.to_csv(PRED_CSV, index=False)
        append_row(RESULTS_CSV, {
            "Pipeline": "Transformer", "Model": MODEL_NAME, "Variant": variant, "Seed": seed,
            "Primary": seed == PRIMARY_SEED, "LR": lr, "Epochs": ep, "Fit pool rows": len(pool_df),
            "Test Accuracy": m["accuracy"], "Test Macro-F1": m["macro_f1"],
            "Test Fake Precision": m["fake_precision"], "Test Fake Recall": m["fake_recall"],
            "Test Fake F1": m["fake_f1"], "Test Real F1": m["real_f1"],
            "Test Confusion Matrix": json.dumps(m["confusion_matrix"]),
            "Test truncated share": test_trunc, "Minutes": round((time.time() - t0) / 60, 2),
        })
        print(f"[final] {variant} seed={seed}: test macro-F1 {m['macro_f1']:.4f} ({(time.time() - t0) / 60:.1f} min)")
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

# %% [markdown]
# ## Step 3 — summary and environment record

# %%
results = load_done(RESULTS_CSV)
summary = results.groupby("Variant").agg(
    LR=("LR", "first"), Epochs=("Epochs", "first"), Seeds=("Seed", "count"),
    macro_f1_mean=("Test Macro-F1", "mean"), macro_f1_std=("Test Macro-F1", "std"),
    accuracy_mean=("Test Accuracy", "mean"),
).reset_index()
primary = results[results["Primary"]].set_index("Variant")["Test Macro-F1"].rename("macro_f1_primary_seed42")
summary = summary.merge(primary, left_on="Variant", right_index=True)
summary.to_csv(os.path.join(OUT_DIR, "distilbert_v2_summary.csv"), index=False)

env = {
    "python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
    "transformers": transformers.__version__, "sklearn": sklearn.__version__, "device": str(DEVICE),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "cuda": torch.version.cuda, "amp_fp16": USE_AMP, "smoke": SMOKE, "parse_mode": PARSE_MODE,
    "data_files": {s: find_file(s) for s in ("train", "valid", "test")},
    "protocol": {"lrs": LRS, "epochs": EPOCHS, "final_seeds": FINAL_SEEDS, "primary_seed": PRIMARY_SEED,
                 "inner_holdout": INNER_HOLDOUT, "batch_size": BATCH_SIZE, "weight_decay": WEIGHT_DECAY,
                 "warmup_ratio": WARMUP_RATIO, "max_grad_norm": MAX_GRAD_NORM, "max_length": MAX_LENGTH,
                 "variants": VARIANTS, "metadata_columns": METADATA_COLUMNS, "truncated_share_pool": truncation},
}
with open(os.path.join(OUT_DIR, "distilbert_v2_environment.json"), "w") as f:
    json.dump(env, f, indent=2)
print(json.dumps(env, indent=2))
summary
