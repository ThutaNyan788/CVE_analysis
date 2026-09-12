"""
Fits the TF-IDF + Logistic Regression severity classifier used by the
Streamlit "Severity Predictor" page, and saves it to
models/severity_predictor.joblib.

This reproduces sections 12.1-12.3 of the project notebook (temporal split,
TF-IDF on desc_lemma, Logistic Regression) — Logistic Regression rather than
the higher-macro-F1 LinearSVC because the live demo needs predict_proba,
exactly the reasoning the notebook itself gives in section 12.7.

Usage (run once, and again whenever the underlying dataset changes):

    cd cve_dashboard
    python train_severity_model.py
    python train_severity_model.py --input data/nvd_cve_with_joint_clusters.csv

Expects a CSV with: description, base_severity, cvss_version, published_year,
cve_id — i.e. either of the notebook's saved
nvd_cve_with_description_clusters.csv / nvd_cve_with_joint_clusters.csv both
qualify, since they're the same underlying dataframe at different pipeline
stages. If the CSV already has a `desc_lemma` column (it will, if it came out
of the notebook), that's used directly and NLTK is never touched. Only if
`desc_lemma` is missing does this script fall back to recomputing it, which
needs `nltk` installed with internet access for `nltk.download(...)`.
"""
import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from utils.text_preprocessing import build_stopwords, clean_text, lemmatize_series

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

DEFAULT_INPUT = DATA_DIR / "nvd_cve_with_description_clusters.csv"
DEFAULT_OUTPUT = MODEL_DIR / "severity_predictor.joblib"

SEV_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
SPLIT_YEAR = 2024
REQUIRED_COLUMNS = {"description", "base_severity", "cvss_version", "published_year", "cve_id"}


def load_and_prepare(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, low_memory=False)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        sys.exit(f"{input_path.name} is missing required column(s): {sorted(missing)}")

    if "desc_lemma" in df.columns:
        print("Found desc_lemma in the CSV — using it directly (no NLTK needed).")
    else:
        print("No desc_lemma column found — recomputing clean -> POS-tag -> lemmatize.")
        print("(This needs `nltk` installed with internet access for nltk.download.)")
        stopwords = build_stopwords(df)
        df["desc_clean"] = df["description"].map(clean_text)
        df["desc_lemma"] = lemmatize_series(df["desc_clean"], stopwords)

    # Normalize to real strings BEFORE filtering. astype(str) on a column that
    # still contains actual NaN turns each one into the literal string "nan",
    # which is non-empty and would slip past the empty-text filter below,
    # leaving real NaN values in `sub` that later crash TfidfVectorizer.
    n_missing = df["desc_lemma"].isna().sum()
    if n_missing:
        print(f"Found {n_missing:,} missing desc_lemma value(s) — treating as empty text.")
    df["desc_lemma"] = df["desc_lemma"].fillna("").astype(str)

    return df


def temporal_split(df: pd.DataFrame):
    sub = df[
        df["cvss_version"].isin([3.0, 3.1])
        & df["base_severity"].isin(SEV_ORDER)
        & df["desc_lemma"].str.strip().ne("")
    ].copy()

    train = sub[sub["published_year"] < SPLIT_YEAR]
    test = sub[sub["published_year"] >= SPLIT_YEAR]
    print(f"Usable rows: {len(sub):,} of {len(df):,} (CVSS v3.x, non-empty text)")
    print(f"Train (< {SPLIT_YEAR}): {len(train):,}   Test (>= {SPLIT_YEAR}): {len(test):,}")
    return train, test


def fit_model(train: pd.DataFrame):
    vec = TfidfVectorizer(
        min_df=3, max_df=0.6, ngram_range=(1, 2), sublinear_tf=True, max_features=100_000
    )
    X_train = vec.fit_transform(train["desc_lemma"])
    print(
        f"TF-IDF matrix: {X_train.shape[0]:,} x {X_train.shape[1]:,} "
        f"({100 * (1 - X_train.nnz / (X_train.shape[0] * X_train.shape[1])):.2f}% zeros)"
    )

    model = LogisticRegression(max_iter=1000, C=2.0, class_weight="balanced")
    model.fit(X_train, train["base_severity"].values)
    return vec, model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help=f"Source CSV (default: {DEFAULT_INPUT.relative_to(BASE_DIR)})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Where to save the model bundle (default: {DEFAULT_OUTPUT.relative_to(BASE_DIR)})")
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(
            f"Can't find {args.input}. Point --input at one of the notebook's saved "
            "*_clusters.csv files, e.g.:\n"
            f"  python train_severity_model.py --input data/nvd_cve_with_joint_clusters.csv"
        )

    df = load_and_prepare(args.input)
    train, test = temporal_split(df)
    if len(train) < 50:
        sys.exit("Fewer than 50 usable training rows — check the input file's columns/values.")

    vec, model = fit_model(train)

    if len(test):
        acc = model.score(vec.transform(test["desc_lemma"]), test["base_severity"].values)
        print(f"Held-out accuracy on {len(test):,} test rows (>= {SPLIT_YEAR}): {acc:.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "vectorizer": vec,
            "model": model,
            "sev_order": SEV_ORDER,
            "split_year": SPLIT_YEAR,
            "trained_rows": len(train),
            "source_file": args.input.name,
        },
        args.output,
    )
    print(f"\nSaved model bundle -> {args.output}")
    print("Restart (or rerun) the Streamlit app to pick it up.")


if __name__ == "__main__":
    main()