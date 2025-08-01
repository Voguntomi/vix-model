import streamlit as st
import pandas as pd
from VIX_clean import df_run

st.set_page_config(page_title="VIX Model Viewer", layout="wide")

st.sidebar.title("📊 Display Options")
view_option = st.sidebar.radio("Select view mode:", ["Formatted Table", "Raw Data Table"])

min_date = df_run.index.min().date()
max_date = df_run.index.max().date()

date_range = st.sidebar.date_input(
    "Select Date Range:",
    value=[min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    st.error("⚠️ Please select a valid start and end date range.")
    st.stop()

mask = (df_run.index.date >= start_date) & (df_run.index.date <= end_date)
df_filtered = df_run.loc[mask].copy()
df_filtered.sort_index(ascending=False, inplace=True)


display_df = df_filtered[["VIX", "pre_5", "post_5", "fitted", "posit", "ret"]].copy()
display_df.rename(columns={
    "pre_5": "Pre 5d",
    "post_5": "Post 5d",
    "fitted": "Signal",
    "posit": "Position",
    "ret": "Return"
}, inplace=True)

pos_map = {1: "Long", 0: "Flat", -1: "Short"}
display_df["Position"] = display_df["Position"].map(pos_map)

if isinstance(display_df.index, pd.DatetimeIndex):
    display_df.insert(0, "Date", display_df.index.strftime("%Y-%m-%d"))

def style_position(val):
    if val == "Long":
        return "background-color:#e8f5e9; color:#1b5e20; font-weight:600"
    elif val == "Short":
        return "background-color:#ffebee; color:#b71c1c; font-weight:600"
    elif val == "Flat":
        return "background-color:#fff3e0; color:#e65100; font-weight:600"
    return ""

styled = (
    display_df
    .style
    .format({
        "VIX": "{:.2f}",
        "Pre 5d": "{:.2%}",
        "Post 5d": "{:.2%}",
        "Signal": "{:.3f}",
        "Return": "{:.2%}"
    })
    .applymap(style_position, subset=["Position"])
    .background_gradient(subset=["Signal", "Return"], cmap="RdYlGn")
)

if view_option == "Formatted Table":
    st.subheader("📈 VIX Model - Formatted Output")
    st.caption("All values are display-formatted only; underlying calculations are untouched.")
    st.dataframe(styled, use_container_width=True, height=900)

elif view_option == "Raw Data Table":
    st.subheader("🔍 VIX Model - Raw Data Output")
    st.caption("Unformatted output of the df_run DataFrame.")
    st.dataframe(df_filtered, use_container_width=True, height=900)


