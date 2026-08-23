"""Regenerate the paper's Fig. 1 / Fig. 2 (top Chi2 / MI features) on the
label-corrected data, with readable (un-stemmed) feature labels (I-12).
Not a notebook -- a one-off figure-generation script, run once from artifacts/.
"""
import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif

from liar_utils import RANDOM_STATE, load_and_label

FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

train = load_and_label("train.csv")

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
surface_forms = Counter()


def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    out = []
    for w in words:
        if w in stop_words:
            continue
        stem = stemmer.stem(w)
        surface_forms[(stem, w)] += 1
        out.append(stem)
    return " ".join(out)


train["clean_text"] = train["Statement"].apply(preprocess)

stem_to_surface = {}
best_count = {}
for (stem, surface), count in surface_forms.items():
    if stem not in best_count or count > best_count[stem]:
        best_count[stem] = count
        stem_to_surface[stem] = surface


def display_name(feature):
    words = feature.split(" ")
    return " ".join(stem_to_surface.get(w, w) for w in words)


tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2, max_df=0.9)
X_train = tfidf.fit_transform(train["clean_text"])
y_train = train["Label"]
feature_names = tfidf.get_feature_names_out()

chi2_scores, _ = chi2(X_train, y_train)
mi_scores = mutual_info_classif(X_train, y_train, random_state=RANDOM_STATE)

chi2_df = pd.DataFrame({"feature": feature_names, "score": chi2_scores}).sort_values("score", ascending=False)
mi_df = pd.DataFrame({"feature": feature_names, "score": mi_scores}).sort_values("score", ascending=False)

for name, df, title, fname in [
    ("chi2", chi2_df, "Top Features by Chi-Square Selection (corrected labels)", "chi2_top_features.png"),
    ("mi", mi_df, "Top Features by Mutual Information (corrected labels)", "mi_top_features.png"),
]:
    top = df.head(15).copy()
    top["display"] = top["feature"].apply(display_name)
    plt.figure(figsize=(7, 5))
    plt.barh(top["display"][::-1], top["score"][::-1], color="#1f77b4")
    plt.xlabel(f"{name.upper()} score")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{fname}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {FIGURES_DIR}/{fname}")

print(chi2_df.head(15)[["feature", "score"]].assign(display=chi2_df.head(15)["feature"].apply(display_name)))
