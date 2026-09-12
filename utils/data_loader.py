"""
Cached loaders for every CSV the notebook (final-01.ipynb) writes with
`.to_csv(...)`. The app never re-runs the notebook's pipeline — it only reads
these artifacts, so it opens instantly regardless of dataset size.

Drop the notebook's output files into cve_dashboard/data/ using their default
names (below) and every page will pick them up automatically.
"""
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# key -> filename the notebook actually writes (see cells 85, 112, 159, 179, 186)
EXPECTED_FILES = {
    "desc_clusters": "nvd_cve_with_description_clusters.csv",
    "desc_cluster_summary": "description_cluster_summary.csv",
    "joint_clusters": "nvd_cve_with_joint_clusters.csv",
    "joint_cluster_summary": "joint_cluster_summary.csv",
    "assoc_rules": "cve_association_rules.csv",
    "frequent_itemsets": "cve_frequent_itemsets.csv",
    "severity_comparison": "severity_model_comparison.csv",
    "severity_test_predictions": "severity_test_predictions.csv",
    "severity_comparison_all_reps": "severity_model_comparison_all_representations.csv",
}

MODEL_FILENAME = "severity_predictor.joblib"


def data_path(key: str) -> Path:
    return DATA_DIR / EXPECTED_FILES[key]


def model_path() -> Path:
    return MODEL_DIR / MODEL_FILENAME


def missing_files(*keys: str) -> list:
    return [EXPECTED_FILES[k] for k in keys if not data_path(k).exists()]


def require_data(*keys: str, note: str = "") -> None:
    """Halt the current page with setup instructions if any required CSV is absent."""
    missing = missing_files(*keys)
    if not missing:
        return
    st.error("This page needs files that aren't in `data/` yet:")
    for name in missing:
        st.code(name, language=None)
    st.info(
        "Run the project notebook end-to-end (it writes these with `.to_csv(...)`), "
        f"then copy them into `{DATA_DIR}`."
        + (f" {note}" if note else "")
    )
    st.stop()


@st.cache_data(show_spinner="Loading data…")
def load_csv(key: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(data_path(key), low_memory=False, **kwargs)


@st.cache_resource(show_spinner="Loading severity model…")
def load_severity_model() -> dict:
    import joblib

    return joblib.load(model_path())


def any_data_present() -> bool:
    return any(data_path(k).exists() for k in EXPECTED_FILES)


def status_table() -> pd.DataFrame:
    """Setup-status rows for the home page: which artifacts exist, which don't."""
    rows = [
        {"file": name, "found": (DATA_DIR / name).exists()}
        for name in EXPECTED_FILES.values()
    ]
    rows.append({"file": f"models/{MODEL_FILENAME}", "found": model_path().exists()})
    return pd.DataFrame(rows)
