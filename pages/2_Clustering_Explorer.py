"""
Clustering Explorer — browse the text-only clusters (section 7) and the
joint text+structured clusters (section 9), and check how well each lines up
with labels the clustering never saw (section 7.10 / 9.6 external validation).
"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from utils.data_loader import data_path, load_csv, require_data
from utils.style import PLOTLY_TEMPLATE, SEV_ORDER, SEV_PALETTE

st.set_page_config(page_title="Clustering Explorer", page_icon="🧩", layout="wide")
st.title("🧩 Clustering Explorer")

MODE = st.radio(
    "Clustering model", ["Text-only (section 7)", "Joint: text + structured (section 9)"],
    horizontal=True,
)
IS_JOINT = MODE.startswith("Joint")

if IS_JOINT:
    require_data("joint_clusters", "joint_cluster_summary")
    df = load_csv("joint_clusters")
    summary = load_csv("joint_cluster_summary").set_index("joint_cluster")
    cluster_col = "joint_cluster"
else:
    require_data("desc_clusters", "desc_cluster_summary")
    df = load_csv("desc_clusters")
    summary = load_csv("desc_cluster_summary").set_index("desc_cluster")
    cluster_col = "desc_cluster"

n_no_text = int((df[cluster_col] == -1).sum()) if cluster_col in df.columns else 0
valid = df[df[cluster_col] != -1] if cluster_col in df.columns else df

st.caption(
    f"{len(valid):,} clustered CVEs across {summary.shape[0]} clusters"
    + (f"  ·  {n_no_text:,} excluded (empty description after preprocessing)" if n_no_text else "")
)

# ---------------------------------------------------------------------------
st.subheader("Cluster sizes")
size_order = summary["n_cves"].sort_values(ascending=False)
fig = px.bar(x=size_order.index.astype(str), y=size_order.values,
             labels={"x": "Cluster", "y": "Number of CVEs"}, template=PLOTLY_TEMPLATE,
             title=None)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
st.subheader("Inspect a cluster")
cluster_ids = sorted(summary.index.tolist())
picked = st.selectbox("Cluster", cluster_ids, format_func=lambda c: f"Cluster {c}")
row = summary.loc[picked]
sub = valid[valid[cluster_col] == picked]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Size", f"{int(row['n_cves']):,}")
c2.metric("Share of dataset", f"{100 * row['n_cves'] / max(len(valid), 1):.1f}%")
if "mean_cvss" in row:
    c3.metric("Mean CVSS", f"{row['mean_cvss']:.2f}")
elif "mean_impact" in row:
    c3.metric("Mean impact", f"{row['mean_impact']:.2f}")
c4.metric("Known-exploited", f"{row['known_exploited_pct']:.2f}%")

st.markdown("**Top terms**")
terms = [t.strip() for t in str(row.get("top_terms", "")).split(",") if t.strip()]
st.write(" ".join(f"`{t}`" for t in terms) or "—")

col_sev, col_kev = st.columns(2)
if "base_severity" in sub.columns:
    sev_mix = sub["base_severity"].value_counts().reindex(SEV_ORDER).fillna(0)
    fig = px.bar(x=sev_mix.index, y=sev_mix.values, color=sev_mix.index,
                 color_discrete_map=SEV_PALETTE, labels={"x": "Severity", "y": "Count"},
                 title=f"Severity mix — cluster {picked}", template=PLOTLY_TEMPLATE)
    col_sev.plotly_chart(fig, use_container_width=True)

if "is_known_exploited" in valid.columns:
    kev_by_cluster = valid.groupby(cluster_col)["is_known_exploited"].mean() * 100
    overall_rate = valid["is_known_exploited"].mean() * 100
    fig = px.bar(x=kev_by_cluster.index.astype(str), y=kev_by_cluster.values,
                 labels={"x": "Cluster", "y": "% known exploited"},
                 title="Known-exploited rate by cluster", template=PLOTLY_TEMPLATE)
    fig.add_hline(y=overall_rate, line_dash="dash", line_color="gray",
                  annotation_text="overall rate")
    fig.update_traces(marker_color=["#EB5757" if c == picked else "#B0BEC5"
                                     for c in kev_by_cluster.index])
    col_kev.plotly_chart(fig, use_container_width=True)

if "description" in sub.columns and len(sub):
    st.markdown("**Sample descriptions from this cluster** _(random sample, not centroid-nearest — that needs the notebook's fitted vectorizer)_")
    n_show = min(5, len(sub))
    for _, r in sub.sample(n_show, random_state=42).iterrows():
        cve_id = r.get("cve_id", "")
        st.markdown(f"- **{cve_id}** — {str(r['description'])[:220]}{'…' if len(str(r['description'])) > 220 else ''}")

# ---------------------------------------------------------------------------
st.divider()
st.subheader("External validation")
st.caption(
    "NMI/ARI against labels the clustering never saw. Computed live from the loaded CSV, "
    "so these numbers may differ slightly from the notebook's printed values if you're "
    "looking at a re-run with different data."
)

val_cols = st.columns(3)
if "cwe_id" in valid.columns:
    nmi_cwe = normalized_mutual_info_score(valid["cwe_id"].astype(str), valid[cluster_col])
    val_cols[0].metric("NMI vs. CWE", f"{nmi_cwe:.4f}")
if "vulnerability_types" in valid.columns:
    nmi_vt = normalized_mutual_info_score(valid["vulnerability_types"].astype(str), valid[cluster_col])
    val_cols[1].metric("NMI vs. vulnerability type", f"{nmi_vt:.4f}")
if "base_severity" in valid.columns:
    nmi_sev = normalized_mutual_info_score(valid["base_severity"].astype(str), valid[cluster_col])
    val_cols[2].metric("NMI vs. severity", f"{nmi_sev:.4f}")

both_present = data_path("desc_clusters").exists() and data_path("joint_clusters").exists()
if both_present:
    st.markdown("**Text-only vs. joint agreement**")
    text_df = load_csv("desc_clusters")[["cve_id", "desc_cluster"]]
    joint_df = load_csv("joint_clusters")[["cve_id", "joint_cluster"]]
    merged = text_df.merge(joint_df, on="cve_id", how="inner")
    merged = merged[(merged["desc_cluster"] != -1) & (merged["joint_cluster"] != -1)]
    if len(merged):
        ari = adjusted_rand_score(merged["desc_cluster"], merged["joint_cluster"])
        st.metric("Adjusted Rand Index (text-only vs. joint)", f"{ari:.4f}",
                  help="Near 0 means the two solutions largely disagree; near 1 means they "
                       "assign CVEs to (relabeled) clusters almost the same way.")
