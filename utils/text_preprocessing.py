"""
Text-cleaning helpers, copied from the project notebook's section 7.1-7.2
(final-01.ipynb) so a new, never-seen CVE description gets exactly the same
treatment the notebook demonstrates in section 12.7 ("Predicting on unseen
text") before it reaches the TF-IDF vectorizer.

`clean_text` is the only step applied to live user input at prediction time —
that is a deliberate match to the notebook's own predict_severity() (cell
176), which also skips POS-lemmatization for new text and relies on TF-IDF
simply ignoring out-of-vocabulary tokens. `build_stopwords` / `lemmatize_text`
are only used by train_severity_model.py, and only as a fallback for the rare
case where the input CSV doesn't already contain a `desc_lemma` column.
"""
import re

import numpy as np

# ---------------------------------------------------------------------------
# Section 7.1 — normalization
# ---------------------------------------------------------------------------
URL_RE = re.compile(r"https?://\S+|www\.\S+")
CVE_RE = re.compile(r"cve-\d{4}-\d+", re.I)
VERSION_RE = re.compile(r"\b\d+(?:\.\d+)+[a-z0-9\-_]*\b")
HEX_RE = re.compile(r"\b[0-9a-f]{7,}\b")
NUMBER_RE = re.compile(r"\b\d+\b")
NONALPHA_RE = re.compile(r"[^a-z\s]")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(s: str) -> str:
    """Lowercase and strip identifiers, versions, digits and punctuation."""
    s = str(s).lower()
    s = URL_RE.sub(" ", s)
    s = CVE_RE.sub(" ", s)
    s = VERSION_RE.sub(" ", s)
    s = HEX_RE.sub(" ", s)
    s = NUMBER_RE.sub(" ", s)
    s = NONALPHA_RE.sub(" ", s)
    return WHITESPACE_RE.sub(" ", s).strip()


# ---------------------------------------------------------------------------
# Section 7.2 — stopwords (only used by the training script's fallback path)
# ---------------------------------------------------------------------------
CVE_BOILERPLATE = {
    "vulnerability", "vulnerabilities", "vulnerable", "attacker", "attackers", "attack",
    "allow", "allows", "allowing", "allowed", "issue", "issues", "affect", "affects",
    "affected", "version", "versions", "flaw", "found", "via", "could", "may", "also",
    "use", "used", "using", "prior", "due", "aka", "discovered", "identified", "exists",
    "unspecified", "unknown", "related", "note", "reported", "possible", "possibly",
}

KEEP_TOKENS = {
    "kernel", "server", "web", "code", "file", "user", "network", "service", "system",
    "core", "security", "access", "client", "manager", "agent", "engine", "library",
    "browser", "database", "framework", "plugin", "module", "driver", "protocol",
}


def build_vendor_token_set(df) -> set:
    """Alphabetic tokens from the vendor/product columns (section 7.2)."""
    toks = set()
    for col in ("vendor", "product"):
        if col not in df.columns:
            continue
        s = (
            df[col].fillna("").astype(str).str.lower()
            .str.split(r"[|\s/,\-_]+").explode()
        )
        toks.update(t for t in s.dropna().unique() if len(t) >= 3 and t.isalpha())
    return toks


def build_stopwords(df=None) -> set:
    """English + CVE boilerplate + vendor/product tokens, matching section 7.2."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    try:
        import nltk
        from nltk.corpus import stopwords as nltk_stopwords

        nltk.download("stopwords", quiet=True)
        base_stop = set(nltk_stopwords.words("english"))
    except Exception:
        base_stop = set()

    vendor_tokens = (build_vendor_token_set(df) - KEEP_TOKENS) if df is not None else set()
    return base_stop | set(ENGLISH_STOP_WORDS) | CVE_BOILERPLATE | vendor_tokens


_NLTK_READY = False


def ensure_nltk_data() -> None:
    global _NLTK_READY
    if _NLTK_READY:
        return
    import nltk

    for pkg in (
        "punkt", "punkt_tab", "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng", "wordnet", "omw-1.4",
    ):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    _NLTK_READY = True


def _penn_to_wordnet(tag: str):
    from nltk.corpus import wordnet

    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def lemmatize_series(cleaned_series, stopwords, chunk_size=5000, min_len=3):
    """POS-aware lemmatization over a pandas Series, matching section 7.3.

    Only used by train_severity_model.py when the input CSV has no
    `desc_lemma` column already (the common case is that it does, since the
    notebook saves it, so this path rarely runs).
    """
    import nltk
    from nltk.stem import WordNetLemmatizer

    ensure_nltk_data()
    lemmatizer = WordNetLemmatizer()
    cache = {}
    texts = cleaned_series.tolist()
    out = []
    for start in range(0, len(texts), chunk_size):
        chunk = texts[start:start + chunk_size]
        token_lists = [t.split() for t in chunk]
        tagged_lists = nltk.pos_tag_sents(token_lists)
        for tagged in tagged_lists:
            kept = []
            for tok, tag in tagged:
                if len(tok) < min_len:
                    continue
                key = (tok, _penn_to_wordnet(tag))
                lemma = cache.get(key)
                if lemma is None:
                    lemma = lemmatizer.lemmatize(tok, key[1])
                    cache[key] = lemma
                if len(lemma) < min_len or lemma in stopwords:
                    continue
                kept.append(lemma)
            out.append(" ".join(kept))
    return out


# ---------------------------------------------------------------------------
# Live inference (Severity Predictor page) — mirrors notebook cell 176
# ---------------------------------------------------------------------------
def predict_severity(text: str, vectorizer, model, top_terms: int = 6) -> dict:
    """Clean -> vectorize -> predict, exactly like the notebook's section 12.7 demo.

    `model` must expose predict_proba (Logistic Regression in the notebook,
    chosen there specifically because LinearSVC has no predict_proba).
    """
    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "Model has no predict_proba — the live demo needs a probabilistic "
            "classifier (the notebook uses Logistic Regression for this reason)."
        )

    processed = clean_text(text)
    Xn = vectorizer.transform([processed])
    vocab = vectorizer.get_feature_names_out()

    probs = model.predict_proba(Xn)[0]
    classes = model.classes_
    order = np.argsort(probs)[::-1]
    predicted = classes[order[0]]
    runner_up = classes[order[1]] if len(order) > 1 else None

    k = list(classes).index(predicted)
    contrib = Xn.toarray().ravel() * model.coef_[k]
    top_idx = np.argsort(contrib)[::-1][:top_terms]
    drivers = [vocab[j] for j in top_idx if contrib[j] > 0]

    return {
        "processed_text": processed,
        "matched_terms": int((Xn.toarray().ravel() > 0).sum()),
        "predicted": predicted,
        "confidence": float(probs[order[0]]),
        "runner_up": runner_up,
        "probabilities": {c: float(p) for c, p in zip(classes, probs)},
        "top_terms": drivers,
    }
