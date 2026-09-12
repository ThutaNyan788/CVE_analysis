"""
Association Rules Explorer — filter/sort the Apriori rules mined in
section 11 (cve_association_rules.csv) and browse the underlying frequent
itemsets (cve_frequent_itemsets.csv).
"""
import plotly.express as px
import streamlit as st

from utils.data_loader import data_path, load_csv, require_data
from utils.style import PLOTLY_TEMPLATE

st.set_page_config(page_title="Association Rules", page_icon="🔗", layout="wide")
st.title("🔗 Association Rules")

require_data("assoc_rules")
rules = load_csv("assoc_rules")

tab_rules, tab_itemsets = st.tabs(["Rules", "Frequent Itemsets"])

with tab_rules:
    st.caption(
        f"{len(rules):,} strong rules (support ≥ min_sup, confidence ≥ 60%, same-column and "
        "reverse-duplicate rules removed — see notebook section 11.4)."
    )

    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
    min_support = col1.slider("Min support", 0.0, float(rules["support"].max()), 0.0, step=0.01)
    min_confidence = col2.slider("Min confidence", 0.0, 1.0, 0.6, step=0.05)
    min_lift = col3.slider("Min lift", 0.0, float(max(rules["lift"].max(), 1.0)), 1.0, step=0.1)
    search = col4.text_input("Search rule text (e.g. attack_vector, cluster=C0, base_severity)")

    filtered = rules[
        (rules["support"] >= min_support)
        & (rules["confidence"] >= min_confidence)
        & (rules["lift"] >= min_lift)
    ]
    if search:
        filtered = filtered[filtered["rule"].str.contains(search, case=False, na=False)]

    sort_by = st.radio("Sort by", ["lift", "confidence", "support"], horizontal=True)
    filtered = filtered.sort_values(sort_by, ascending=False)

    st.caption(f"Showing {len(filtered):,} of {len(rules):,} rules.")

    col_scatter, col_top = st.columns(2)
    if len(filtered):
        fig = px.scatter(
            filtered, x="support", y="confidence", color="lift", size="lift",
            color_continuous_scale="RdYlBu_r", template=PLOTLY_TEMPLATE,
            title="Support vs. confidence (color/size = lift)",
            hover_data={"rule": True, "support": ":.3f", "confidence": ":.3f", "lift": ":.2f"},
        )
        col_scatter.plotly_chart(fig, use_container_width=True)

        top_lift = filtered.nlargest(12, "lift").sort_values("lift")
        fig = px.bar(top_lift, x="lift", y="rule", orientation="h",
                     title="Top rules by lift", template=PLOTLY_TEMPLATE)
        fig.update_layout(yaxis={"tickfont": {"size": 10}})
        fig.add_vline(x=1.0, line_dash="dash", line_color="red")
        col_top.plotly_chart(fig, use_container_width=True)

    display_cols = ["rule", "support", "confidence", "lift", "all_conf", "max_conf", "kulc", "cosine"]
    display_cols = [c for c in display_cols if c in filtered.columns]
    st.dataframe(
        filtered[display_cols].round(4),
        hide_index=True,
        use_container_width=True,
        height=420,
    )

with tab_itemsets:
    if data_path("frequent_itemsets").exists():
        itemsets = load_csv("frequent_itemsets")
        k_present = sorted(itemsets["k"].unique().tolist())
        k_pick = st.multiselect("Itemset size (k)", k_present, default=k_present)
        filtered_is = itemsets[itemsets["k"].isin(k_pick)].sort_values("sup_count", ascending=False)
        st.caption(f"{len(filtered_is):,} of {len(itemsets):,} frequent itemsets.")
        st.dataframe(filtered_is.round(4), hide_index=True, use_container_width=True, height=500)
    else:
        st.info(
            f"`{data_path('frequent_itemsets').name}` isn't in `data/` — only the rules table "
            "is shown. Copy that file over too if you want to browse raw frequent itemsets."
        )
