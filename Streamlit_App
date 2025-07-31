import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

st.set_page_config(page_title="VIX Model", layout="wide")

@st.cache_data(show_spinner=True)
def build_df_run():
    raw_params = {
        'neg_corr': {
            'Coefficient': [-0.3, 0.1],
            'P-value': [0.001, 0.002],
            'Variable': ['pre_5', 'VIX'],
            'high': [20, 20],
            'low': [10, 10],
        },
        'pos_corr': {
            'Coefficient': [-0.1, 0.05],
            'P-value': [0.05, 0.03],
            'Variable': ['pre_5', 'VIX'],
            'high': [20, 20],
            'low': [10, 10],
        }
    }

    d_params = {k: pd.DataFrame(v) for k, v in raw_params.items()}

    df_vix = yf.download('^VIX', start='2010-01-01', interval='1d', auto_adjust=False, progress=False)
    if df_vix.empty:
        raise ValueError("Yahoo Finance returned no data for ^VIX.")

    df = pd.DataFrame()
    df['VIX'] = df_vix['Close']
    df.dropna(inplace=True)

    for i in range(1, 10):
        df[f'pre_{i}'] = df['VIX'].pct_change(i)
        df[f'post_{i}'] = df['VIX'].shift(-i) / df['VIX'] - 1

    df_shifted = df.shift(1)
    df['rollcorr_post1_pre8'] = df_shifted['post_1'].rolling(25).corr(df_shifted['pre_8'])

    df_run = df[['VIX', 'pre_5', 'post_5', 'rollcorr_post1_pre8']].copy()
    df_run.dropna(subset=['rollcorr_post1_pre8'], inplace=True)
    df_run['original_index'] = df_run.index

    df_run['corr_type'] = np.select(
        [df_run['rollcorr_post1_pre8'] > 0, df_run['rollcorr_post1_pre8'] < 0],
        ['pos_corr', 'neg_corr'], default=None)

    def prepare_model_df(df_model, label):
        df_model = df_model[df_model['Variable'].isin(['pre_5', 'VIX'])].copy()
        df_model['corr_type'] = label
        return df_model

    model_df = pd.concat([
        prepare_model_df(d_params['pos_corr'], 'pos_corr'),
        prepare_model_df(d_params['neg_corr'], 'neg_corr')], ignore_index=True)

    for var in ['pre_5', 'VIX']:
        coeffs = model_df[model_df['Variable'] == var].copy()

        def lookup_coeff(row):
            match = coeffs[
                (coeffs['corr_type'] == row['corr_type']) &
                (coeffs['low'] <= row['VIX']) &
                (row['VIX'] < coeffs['high'])
            ]
            return match['Coefficient'].values[0] if not match.empty else np.nan

        df_run[f'{var}_coeff'] = df_run.apply(lookup_coeff, axis=1)

    df_run.set_index('original_index', inplace=True)
    df_run.sort_index(inplace=True)

    threshold_long = 0.02
    threshold_short = 0.0
    stop_loss_short = 0.2

    df_run['contr_pre_5'] = df_run['pre_5'] * df_run['pre_5_coeff']
    df_run['contr_vix'] = df_run['VIX'] * df_run['VIX_coeff']
    df_run['fitted'] = df_run['contr_pre_5'] + df_run['contr_vix']

    df_run['posit'] = np.where(df_run['fitted'] > threshold_long, 1,
                               np.where(df_run['fitted'] < threshold_short, -1, 0))

    df_run['orig/fitted'] = df_run['post_5'] * abs(df_run['posit']) / df_run['fitted']

    df_run['ret'] = np.where(
        df_run['posit'] == -1,
        np.maximum(-stop_loss_short, df_run['posit'] * df_run['post_5']),
        df_run['posit'] * df_run['post_5']
    )

    return df_run

def main():
    st.title("📈 VIX Model Results")
    with st.spinner("Running model and preparing table..."):
        df_run = build_df_run()

    required = {'VIX', 'pre_5', 'post_5', 'fitted', 'posit', 'ret'}
    if not required.issubset(df_run.columns):
        st.error("Missing required columns in df_run.")
        return

    display_df = df_run[['VIX', 'pre_5', 'post_5', 'fitted', 'posit', 'ret']].copy()
    display_df.rename(columns={
        'pre_5': 'Pre 5d',
        'post_5': 'Post 5d',
        'fitted': 'Signal',
        'posit': 'Position',
        'ret': 'Return'
    }, inplace=True)

    pos_map = {1: 'Long', 0: 'Flat', -1: 'Short'}
    display_df['Position'] = display_df['Position'].map(pos_map)

    if isinstance(display_df.index, pd.DatetimeIndex):
        display_df.insert(0, 'Date', display_df.index.strftime('%Y-%m-%d'))

    # 🔹 Add date filter for formatted display
    st.subheader("📅 Filter by Date Range")
    min_date = display_df.index.min().date()
    max_date = display_df.index.max().date()
    start_date, end_date = st.date_input(
        "Select Date Range:",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )

    mask = (display_df.index.date >= start_date) & (display_df.index.date <= end_date)
    display_df = display_df.loc[mask]

    def style_position(val):
        if val == 'Long':
            return 'background-color:#e8f5e9; color:#1b5e20; font-weight:600'
        elif val == 'Short':
            return 'background-color:#ffebee; color:#b71c1c; font-weight:600'
        elif val == 'Flat':
            return 'background-color:#fff3e0; color:#e65100; font-weight:600'
        return ''

    styled = (
        display_df
        .style
        .format({
            'VIX': '{:.2f}',
            'Pre 5d': '{:.2%}',
            'Post 5d': '{:.2%}',
            'Signal': '{:.3f}',
            'Return': '{:.2%}'
        })
        .applymap(style_position, subset=['Position'])
        .background_gradient(subset=['Signal', 'Return'], cmap='RdYlGn')
    )

    st.subheader("📊 Model Output Table")
    st.caption("All values are display-formatted only; underlying calculations are untouched.")
    st.dataframe(styled, use_container_width=True, height=600)

    with st.expander("🔍 Raw df_run (no formatting)"):
        st.dataframe(df_run.tail(50), use_container_width=True)

if __name__ == "__main__":
    main()
