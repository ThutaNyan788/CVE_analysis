"""
Severity Predictor — live text -> severity inference (needs
models/severity_predictor.joblib, produced by train_severity_model.py), plus
the static model-comparison results from notebook section 12.
"""
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import confusion_matrix

from utils.data_loader import data_path, load_csv, load_severity_model, model_path
from utils.style import PLOTLY_TEMPLATE, SEV_ORDER, SEV_PALETTE
from utils.text_preprocessing import predict_severity

st.set_page_config(page_title="Severity Predictor", page_icon="🔮", layout="wide")
st.title("🔮 Severity Predictor")

# ---------------------------------------------------------------------------
st.subheader("Try it")
st.caption(
    "Matches the notebook's own unseen-text demo (section 12.7): the text is only "
    "lowercased and stripped of URLs/CVE-IDs/version numbers/punctuation before "
    "vectorizing — not re-lemmatized — because that's exactly what predict_severity() "
    "does there."
)

EXAMPLES = [
    "An unauthenticated remote attacker can send a crafted HTTP request to the management "
    "interface, resulting in arbitrary code execution with root privileges on the appliance.",
    "Stored cross-site scripting in the comment field allows an authenticated low-privileged "
    "user to inject arbitrary web script that executes in the browser of another user.",
    "A local user with administrative access can read the contents of a temporary log file "
    "that may contain non-sensitive diagnostic information.",
]

if not model_path().exists():
    st.warning(
        f"No trained model at `{model_path()}` yet. From the `cve_dashboard/` folder, run:\n\n"
        "```\npython train_severity_model.py\n```\n\n"
        "It needs `nvd_cve_with_description_clusters.csv` (or the joint-clusters version) "
        "in `data/`, since both already contain the `desc_lemma` column the model trains on."
    )
else:
    bundle = load_severity_model()
    vec, model = bundle["vectorizer"], bundle["model"]
    st.caption(
        f"Model trained on {bundle.get('trained_rows', '?'):,} CVEs published before "
        f"{bundle.get('split_year', '?')}, from `{bundle.get('source_file', '?')}`."
    )

    text = st.text_area("CVE-style description", value="", height=110,
                         placeholder="Paste or write a vulnerability description…")
    example_pick = st.selectbox("…or try one of the notebook's demo examples",
                                 ["(none)"] + EXAMPLES, format_func=lambda s: s if s == "(none)" else s[:80] + "…")
    if example_pick != "(none)":
        text = example_pick

    if st.button("Predict severity", type="primary", disabled=not text.strip()):
        result = predict_severity(text, vec, model, top_terms=6)
        pred = result["predicted"]

        c1, c2 = st.columns([1, 2])
        c1.markdown(
            f"<div style='padding:1rem;border-radius:0.5rem;background:{SEV_PALETTE.get(pred, '#4C72B0')}22;"
            f"border:2px solid {SEV_PALETTE.get(pred, '#4C72B0')}'>"
            f"<div style='font-size:0.85rem;color:gray'>Predicted severity</div>"
            f"<div style='font-size:1.8rem;font-weight:700;color:{SEV_PALETTE.get(pred, '#333')}'>{pred}</div>"
            f"<div style='font-size:0.85rem'>confidence {result['confidence']:.1%} · runner-up {result['runner_up']}</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        prob_series = pd.Series(result["probabilities"]).reindex(SEV_ORDER).fillna(0)
        fig = px.bar(x=prob_series.index, y=prob_series.values, color=prob_series.index,
                     color_discrete_map=SEV_PALETTE, labels={"x": "", "y": "Probability"},
                     template=PLOTLY_TEMPLATE, title=None)
        fig.update_layout(showlegend=False, height=280)
        c2.plotly_chart(fig, use_container_width=True)

        if result["top_terms"]:
            st.markdown("**Terms driving this prediction:** " +
                        " ".join(f"`{t}`" for t in result["top_terms"]))
        if result["matched_terms"] == 0:
            st.info(
                "None of this text's words matched the model's vocabulary — the prediction "
                "above is effectively the class prior, not a real read on this text."
            )

st.divider()

# ---------------------------------------------------------------------------
st.subheader("Model comparison (section 12.3-12.4)")
if data_path("severity_comparison").exists():
    comparison = load_csv("severity_comparison")
    best_row = comparison.sort_values("macro_F1", ascending=False).iloc[0]
    st.caption(
        f"Best by macro-F1: **{best_row['model']}** ({best_row['macro_F1']:.4f}). The live "
        "predictor above uses Logistic Regression regardless, because it's the one with "
        "`predict_proba` — same reasoning as the notebook's section 12.7."
    )

    fig = px.bar(comparison.sort_values("macro_F1"), x="macro_F1", y="model", orientation="h",
                 template=PLOTLY_TEMPLATE, title="Macro-F1 by model")
    dummy_row = comparison[comparison["model"].str.contains("Dummy", case=False, na=False)]
    if len(dummy_row):
        fig.add_vline(x=float(dummy_row.iloc[0]["macro_F1"]), line_dash="dash", line_color="gray",
                      annotation_text="Dummy baseline")
    st.plotly_chart(fig, use_container_width=True)

    show_cols = [c for c in comparison.columns if c != "fit_seconds"]
    st.dataframe(comparison[show_cols].round(4), hide_index=True, use_container_width=True)
else:
    st.info(f"Copy `{data_path('severity_comparison').name}` into `data/` to see this section.")

# ---------------------------------------------------------------------------
st.subheader("Best model on held-out data (section 12.5-12.6)")
if data_path("severity_test_predictions").exists():
    samples = load_csv("severity_test_predictions")

    if {"actual", "predicted"} <= set(samples.columns):
        labels = [s for s in SEV_ORDER if s in set(samples["actual"]) | set(samples["predicted"])]
        cm = confusion_matrix(samples["actual"], samples["predicted"], labels=labels)
        col_cm, col_pct = st.columns(2)
        fig = px.imshow(cm, x=labels, y=labels, text_auto="d", color_continuous_scale="Blues",
                        labels={"x": "predicted", "y": "actual"}, template=PLOTLY_TEMPLATE,
                        title="Confusion matrix (counts)")
        col_cm.plotly_chart(fig, use_container_width=True)

        cm_pct = (cm / cm.sum(axis=1, keepdims=True) * 100).round(1)
        fig = px.imshow(cm_pct, x=labels, y=labels, text_auto=".1f", color_continuous_scale="Blues",
                        labels={"x": "predicted", "y": "actual"}, template=PLOTLY_TEMPLATE,
                        title="Row-normalized (% of each true class)", zmin=0, zmax=100)
        col_pct.plotly_chart(fig, use_container_width=True)

    if "margin" in samples.columns and "correct" in samples.columns:
        st.markdown("**Sample predictions, by confidence margin**")
        which = st.radio("Show", ["Confident & correct", "Confident but wrong", "Least confident"],
                          horizontal=True)
        if which == "Confident & correct":
            shown = samples[samples["correct"]].nlargest(8, "margin")
        elif which == "Confident but wrong":
            shown = samples[~samples["correct"]].nlargest(8, "margin")
        else:
            shown = samples.nsmallest(8, "margin")
        cols = [c for c in ["cve_id", "description", "actual", "predicted", "margin"] if c in shown.columns]
        st.dataframe(shown[cols], hide_index=True, use_container_width=True)
else:
    st.info(f"Copy `{data_path('severity_test_predictions').name}` into `data/` to see this section.")

# ---------------------------------------------------------------------------
if data_path("severity_comparison_all_reps").exists():
    st.divider()
    st.subheader("TF-IDF vs. sentence embeddings (section 13)")
    combined = load_csv("severity_comparison_all_reps")
    label_col = "representation" if "representation" in combined.columns else None
    combined = combined.sort_values("macro_F1")
    y_label = combined["model"] + (" (" + combined[label_col] + ")" if label_col else "")
    fig = px.bar(x=combined["macro_F1"], y=y_label, orientation="h",
                 template=PLOTLY_TEMPLATE, title="Every model × representation, ranked by macro-F1")
    st.plotly_chart(fig, use_container_width=True)
