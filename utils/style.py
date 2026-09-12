"""
Shared visual constants, kept identical to the notebook's SEV_ORDER / SEV_PALETTE
(cell 2 of final-01.ipynb) so every chart in the app uses the same severity
ordering and colors as the project book's figures.
"""

SEV_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

SEV_PALETTE = {
    "LOW": "#6FCF97",
    "MEDIUM": "#F2C94C",
    "HIGH": "#F2994A",
    "CRITICAL": "#EB5757",
}

PLOTLY_TEMPLATE = "plotly_white"


def sev_colors(order=None):
    """List of hex colors in severity order, for use as a Plotly discrete map."""
    order = order or SEV_ORDER
    return [SEV_PALETTE.get(s, "#4C72B0") for s in order]


def kpi_number(value, suffix="") -> str:
    """Compact display formatting for KPI cards (1234567 -> '1.23M')."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    for threshold, label in ((1_000_000, "M"), (1_000, "K")):
        if abs(value) >= threshold:
            return f"{value / threshold:.2f}{label}{suffix}"
    if value == int(value):
        return f"{int(value):,}{suffix}"
    return f"{value:,.2f}{suffix}"
