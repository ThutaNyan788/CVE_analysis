"""
EDA Dashboard — interactive versions of the project book's Chapter 2 figures
(sections 2.3.1-2.3.12 / notebook cells 19-42).
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from utils.data_loader import load_csv, require_data
from utils.style import PLOTLY_TEMPLATE, SEV_ORDER, SEV_PALETTE, sev_colors

st.set_page_config(page_title="EDA Dashboard", page_icon="📊", layout="wide")
st.title("📊 EDA Dashboard")

require_data("desc_clusters")
df = load_csv("desc_clusters")

NUMERIC_COLS = [
    "cvss_base_score", "exploitability_score", "impact_score", "affected_product_count",
    "affected_version_count", "reference_count", "description_length",
    "description_word_count", "vulnerability_age_days", "days_to_last_update",
    "cia_impact_count",
]
VULN_FLAGS = [
    "is_remote_code_execution", "is_privilege_escalation", "is_denial_of_service",
    "is_information_disclosure", "is_authentication_bypass", "is_sql_injection",
    "is_cross_site_scripting", "is_path_traversal", "is_buffer_overflow",
    "is_memory_corruption", "is_command_injection", "is_deserialization",
]

# --- Sidebar filters (the notebook's charts were static; this is the interactive add-on) --
st.sidebar.header("Filters")
if "published_year" in df.columns:
    yr_min, yr_max = int(df["published_year"].min()), int(df["published_year"].max())
    yr_range = st.sidebar.slider("Publication year", yr_min, yr_max, (yr_min, yr_max))
else:
    yr_range = None

sev_present = [s for s in SEV_ORDER if s in df.get("base_severity", pd.Series(dtype=str)).unique()]
sev_pick = st.sidebar.multiselect("Severity", sev_present, default=sev_present)

dff = df.copy()
if yr_range and "published_year" in dff.columns:
    dff = dff[dff["published_year"].between(*yr_range)]
if sev_pick and "base_severity" in dff.columns:
    dff = dff[dff["base_severity"].isin(sev_pick)]

st.caption(f"{len(dff):,} of {len(df):,} CVEs match the current filters.")

tab_vol, tab_cvss, tab_ecosystem, tab_kev, tab_corr = st.tabs(
    ["Volume & Severity", "CVSS & Attack Profile", "Vendors / Products / Weaknesses",
     "Known-Exploited (KEV)", "Correlations"]
)

# ---------------------------------------------------------------------------
with tab_vol:
    col1, col2 = st.columns([1, 1])

    if "published_year" in dff.columns:
        vol = dff["published_year"].value_counts().sort_index()
        fig = px.bar(x=vol.index, y=vol.values, labels={"x": "Year", "y": "Number of CVEs"},
                     title="CVEs Published per Year", template=PLOTLY_TEMPLATE, text=vol.values)
        fig.update_traces(marker_color="#4C72B0", textposition="outside")
        col1.plotly_chart(fig, use_container_width=True)

    if "base_severity" in dff.columns:
        sev_counts = dff["base_severity"].value_counts().reindex(SEV_ORDER).fillna(0)
        fig = px.pie(values=sev_counts.values, names=sev_counts.index,
                     color=sev_counts.index, color_discrete_map=SEV_PALETTE,
                     title="Severity Share", template=PLOTLY_TEMPLATE, hole=0.35)
        col2.plotly_chart(fig, use_container_width=True)

        fig = px.bar(x=sev_counts.index, y=sev_counts.values,
                     color=sev_counts.index, color_discrete_map=SEV_PALETTE,
                     labels={"x": "Severity", "y": "Count"}, title="CVE Count by Severity",
                     template=PLOTLY_TEMPLATE, text=sev_counts.values)
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
with tab_cvss:
    if {"cvss_base_score", "base_severity"} <= set(dff.columns):
        fig = px.histogram(
            dff, x="cvss_base_score", color="base_severity", category_orders={"base_severity": SEV_ORDER},
            color_discrete_map=SEV_PALETTE, nbins=40, barmode="stack",
            title="CVSS Base Score Distribution (stacked by severity)", template=PLOTLY_TEMPLATE,
        )
        st.plotly_chart(fig, use_container_width=True)

    attack_cols = ["attack_vector", "attack_complexity", "privileges_required", "user_interaction"]
    present = [c for c in attack_cols if c in dff.columns]
    if present:
        fig = make_subplots(rows=1, cols=len(present),
                             subplot_titles=[c.replace("_", " ").title() for c in present])
        for i, col in enumerate(present, start=1):
            vc = dff[col].value_counts()
            fig.add_trace(go.Bar(x=vc.values, y=vc.index, orientation="h",
                                  marker_color="#2E5266", showlegend=False), row=1, col=i)
        fig.update_layout(template=PLOTLY_TEMPLATE, height=380,
                           title_text="Attack Vector, Complexity, Privileges & User Interaction")
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    if {"attack_vector", "cvss_base_score"} <= set(dff.columns):
        fig = px.box(dff, x="attack_vector", y="cvss_base_score", color="attack_vector",
                     title="CVSS Base Score by Attack Vector", template=PLOTLY_TEMPLATE)
        fig.update_layout(showlegend=False)
        col1.plotly_chart(fig, use_container_width=True)

    if "published_month" in dff.columns:
        monthly = dff["published_month"].value_counts().sort_index()
        fig = px.bar(x=monthly.index, y=monthly.values,
                     labels={"x": "Month", "y": "Count"},
                     title="CVEs Published by Month (seasonality, all years combined)",
                     template=PLOTLY_TEMPLATE)
        fig.update_traces(marker_color="#67A9CF")
        fig.update_xaxes(dtick=1)
        col2.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
with tab_ecosystem:
    def top_entities(series: pd.Series, n: int = 15) -> pd.Series:
        exploded = series.fillna("unknown").str.lower().str.split("|").explode().str.strip()
        exploded = exploded[~exploded.isin(["n/a", "unknown", ""])]
        return exploded.value_counts().head(n)

    col1, col2 = st.columns(2)
    if "vendor" in dff.columns:
        top_vendors = top_entities(dff["vendor"]).sort_values()
        fig = px.bar(x=top_vendors.values, y=top_vendors.index, orientation="h",
                     labels={"x": "Number of CVEs", "y": ""}, title="Top 15 Vendors by CVE Count",
                     template=PLOTLY_TEMPLATE)
        fig.update_traces(marker_color="#2A9D8F")
        col1.plotly_chart(fig, use_container_width=True)

    if "product" in dff.columns:
        top_products = top_entities(dff["product"]).sort_values()
        fig = px.bar(x=top_products.values, y=top_products.index, orientation="h",
                     labels={"x": "Number of CVEs", "y": ""}, title="Top 15 Products by CVE Count",
                     template=PLOTLY_TEMPLATE)
        fig.update_traces(marker_color="#2A9D8F")
        col2.plotly_chart(fig, use_container_width=True)

    if "cwe_id" in dff.columns:
        top_cwe = dff["cwe_id"].value_counts().head(15).sort_values()
        fig = px.bar(x=top_cwe.values, y=top_cwe.index, orientation="h",
                     labels={"x": "Number of CVEs", "y": ""},
                     title="Top 15 CWE (Weakness) Categories", template=PLOTLY_TEMPLATE)
        fig.update_traces(marker_color="#E76F51")
        st.plotly_chart(fig, use_container_width=True)

    present_flags = [c for c in VULN_FLAGS if c in dff.columns]
    if present_flags:
        flag_sums = dff[present_flags].sum().sort_values()
        labels = [c.replace("is_", "").replace("_", " ").title() for c in flag_sums.index]
        fig = px.bar(x=flag_sums.values, y=labels, orientation="h",
                     labels={"x": "Number of CVEs", "y": ""},
                     title="CVEs Flagged by Vulnerability Type", template=PLOTLY_TEMPLATE)
        fig.update_traces(marker_color="#6A4C93")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
with tab_kev:
    if "is_known_exploited" in dff.columns:
        kev_rate = dff["is_known_exploited"].mean() * 100
        col1, col2 = st.columns(2)

        ke_counts = dff["is_known_exploited"].value_counts().sort_index()
        fig = px.pie(values=ke_counts.values,
                     names=["Not Known-Exploited", "Known Exploited"][:len(ke_counts)],
                     color_discrete_sequence=["#B0BEC5", "#EB5757"],
                     title=f"Known-Exploited Share ({kev_rate:.2f}% of filtered CVEs)",
                     template=PLOTLY_TEMPLATE, hole=0.35)
        col1.plotly_chart(fig, use_container_width=True)

        if "base_severity" in dff.columns:
            kev_sev = (dff[dff["is_known_exploited"] == 1]["base_severity"]
                       .value_counts().reindex(SEV_ORDER).fillna(0))
            fig = px.bar(x=kev_sev.index, y=kev_sev.values, color=kev_sev.index,
                         color_discrete_map=SEV_PALETTE, text=kev_sev.values,
                         labels={"x": "Severity", "y": "Count"},
                         title="Severity of Known-Exploited CVEs", template=PLOTLY_TEMPLATE)
            fig.update_traces(textposition="outside")
            col2.plotly_chart(fig, use_container_width=True)

        if "published_year" in dff.columns:
            kev_by_year = (
                dff.groupby("published_year")["is_known_exploited"]
                .agg(known_exploited_count="sum", total_cves="count")
            )
            kev_by_year["rate_pct"] = (
                kev_by_year["known_exploited_count"] / kev_by_year["total_cves"] * 100
            ).round(2)

            col3, col4 = st.columns(2)
            fig = px.bar(x=kev_by_year.index.astype(str), y=kev_by_year["known_exploited_count"],
                         labels={"x": "Year", "y": "Count"}, title="Known-Exploited CVEs per Year",
                         template=PLOTLY_TEMPLATE, text=kev_by_year["known_exploited_count"])
            fig.update_traces(marker_color="#EB5757", textposition="outside")
            col3.plotly_chart(fig, use_container_width=True)

            fig = px.line(x=kev_by_year.index, y=kev_by_year["rate_pct"], markers=True,
                          labels={"x": "Year", "y": "% of CVEs Known Exploited"},
                          title="Known-Exploited Rate per Year", template=PLOTLY_TEMPLATE)
            fig.update_traces(line_color="#EB5757")
            col4.plotly_chart(fig, use_container_width=True)
    else:
        st.info("`is_known_exploited` column not found in the loaded dataset.")

# ---------------------------------------------------------------------------
with tab_corr:
    present_numeric = [c for c in NUMERIC_COLS if c in dff.columns]
    if len(present_numeric) >= 2:
        corr = dff[present_numeric].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        title="Correlation Heatmap — Numeric Features", template=PLOTLY_TEMPLATE,
                        aspect="auto")
        fig.update_layout(height=650)
        st.plotly_chart(fig, use_container_width=True)

    if {"exploitability_score", "impact_score", "base_severity"} <= set(dff.columns):
        sample = dff[dff["base_severity"].isin(SEV_ORDER)]
        sample = sample.sample(min(6000, len(sample)), random_state=42) if len(sample) else sample
        fig = px.scatter(sample, x="exploitability_score", y="impact_score", color="base_severity",
                         category_orders={"base_severity": SEV_ORDER}, color_discrete_map=SEV_PALETTE,
                         opacity=0.45, title="Exploitability vs. Impact Score (sampled, colored by severity)",
                         template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)
