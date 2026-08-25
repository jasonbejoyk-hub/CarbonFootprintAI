import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline import (
    VALUE_COL,
    build_projection,
    load_clean,
    modern_period,
)

st.set_page_config(
    page_title="India CO₂ per capita explorer",
    page_icon="🌿",
    layout="wide",
)


@st.cache_data
def get_data():
    clean = load_clean()
    return clean, modern_period(clean)


clean_df, modern = get_data()
last_year = int(modern["Year"].max())
last_value = float(modern.loc[modern["Year"] == last_year, VALUE_COL].iloc[0])

st.title("India CO₂ emissions per capita")
st.caption(
    "Historical Global Carbon Budget data, a linear trend for 1990–"
    f"{last_year}, and a simple policy scenario you can adjust."
)

with st.sidebar:
    st.header("Scenario controls")
    horizon = st.slider(
        "Forecast through year",
        min_value=last_year + 1,
        max_value=2100,
        value=2050,
        step=1,
        help="Last year to project. History ends in the data; everything after that is a model.",
    )
    reduction_pct = st.slider(
        "Extra annual cut vs business-as-usual (%)",
        min_value=0.0,
        max_value=8.0,
        value=0.0,
        step=0.1,
        help="0% follows the fitted trend. 2% means each future year is 2% below that trend path.",
    )
    show_full_history = st.checkbox("Show full history (from 1858)", value=False)
    st.markdown(
        f"**Last observed year:** {last_year}  \n"
        f"**Last observed value:** {last_value:.2f} t/person"
    )

annual_reduction = reduction_pct / 100.0
proj = build_projection(modern, annual_reduction=annual_reduction, horizon_year=horizon)
holdout = proj["holdout"]
scenario = proj["scenario"]
fitted = proj["fitted"]

test_m = holdout["test_metrics"]
train_m = holdout["train_metrics"]
bau_2050 = None
pol_end = None
if not scenario.empty:
    pol_end = float(scenario.loc[scenario["Year"] == horizon, "policy"].iloc[0])
    bau_end = float(scenario.loc[scenario["Year"] == horizon, "bau"].iloc[0])
else:
    bau_end = None

m1, m2, m3, m4 = st.columns(4)
m1.metric("Test MAE (t/person)", f"{test_m['mae']:.3f}")
m2.metric("Test RMSE (t/person)", f"{test_m['rmse']:.3f}")
m3.metric(
    "Test R²",
    f"{test_m['r2']:.3f}",
    help="Negative means a 1990–2015 line missed the faster rise after 2015.",
)
m4.metric(
    f"{horizon} under this scenario",
    f"{pol_end:.2f} t" if pol_end is not None else "—",
    delta=f"{pol_end - bau_end:.2f} vs BAU" if pol_end is not None else None,
)

hist = clean_df if show_full_history else modern

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=hist["Year"],
        y=hist[VALUE_COL],
        mode="lines",
        name="Observed",
        line=dict(color="#1f77b4", width=2),
    )
)
fig.add_trace(
    go.Scatter(
        x=fitted["Year"],
        y=fitted["fitted"],
        mode="lines",
        name="Linear fit (1990–2024)",
        line=dict(color="#ff7f0e", width=2, dash="dot"),
    )
)
if not scenario.empty:
    fig.add_trace(
        go.Scatter(
            x=scenario["Year"],
            y=scenario["bau"],
            mode="lines",
            name="Business-as-usual",
            line=dict(color="#d62728", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=scenario["Year"],
            y=scenario["policy"],
            mode="lines",
            name=f"Policy ({reduction_pct:.1f}% extra cut/year)",
            line=dict(color="#2ca02c", width=2),
        )
    )

fig.update_layout(
    title="Observed emissions, fitted trend, and scenario projection",
    xaxis_title="Year",
    yaxis_title="CO₂ emissions per capita (tonnes per person)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    margin=dict(l=40, r=20, t=60, b=40),
    height=520,
)
fig.update_xaxes(rangeslider_visible=True)
st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("What the model is doing")
    st.markdown(
        f"""
The target is **CO₂ emissions per capita** (tonnes per person). The feature is **year**.

A straight line is fit on **1990–{last_year}**, when India's energy use is in a
modern growth period. Using 1858 onward would pull the slope down because early
values are near zero.

Evaluation uses a **time split**, not a random split:
- train: 1990–2015
- test: 2016–{last_year}

Train MAE = **{train_m['mae']:.3f}**, test MAE = **{test_m['mae']:.3f}**.
Test **R² is {test_m['r2']:.2f}** (negative): emissions rose faster after 2015
than a 1990–2015 line predicted. The app therefore **refits on all 1990–{last_year}**
for the displayed forecast, and you should treat long-run BAU as a trend, not a
confident prediction.

The full-sample slope is **{proj['slope']:.4f}** tonnes per person per year.
"""
    )
with right:
    st.subheader("Policy assumption")
    st.markdown(
        f"""
**Business-as-usual (BAU)** continues the fitted line after {last_year}.

**Policy** does not invent new drivers (GDP, coal, EVs). It asks: *if each year
after {last_year} per-capita emissions were an extra {reduction_pct:.1f}% below
that BAU path, where would we be by {horizon}?*

Formula: `policy_t = max(0, BAU_t × (1 − r)^(t − {last_year}))`

This is a **scenario**, not a forecast of real Indian climate policy.
COVID-year 2020 is a dip in the history; a straight line cannot capture shocks.
"""
    )

if not scenario.empty:
    table = scenario.rename(
        columns={"bau": "BAU (t/person)", "policy": "Policy scenario (t/person)"}
    )
    table["Year"] = table["Year"].astype(int)
    st.subheader("Projected values")
    st.dataframe(table, use_container_width=True, hide_index=True)

    csv = table.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download projection CSV",
        data=csv,
        file_name=f"india_co2_projection_{horizon}.csv",
        mime="text/csv",
    )

st.expander("Data source and limits").markdown(
    """
Source: Our World in Data chart *CO₂ emissions per capita*, India filter.
Original series: Global Carbon Budget (2025); population from various sources (2024).

Figures are **territorial fossil and industry CO₂**, not land-use change, and not
emissions embodied in imports. International aviation and shipping are excluded
from the country series.

To reproduce locally: `pip install -r requirements.txt` then
`streamlit run src/app.py` from the repository root.
"""
)
