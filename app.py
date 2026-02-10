import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import numpy as np

# 1. Create a custom session to 'pretend' to be a browser
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
})

# 2. When you create your ticker object, use this session
# Example:
info = ticker_data.info

# Set page configuration
st.set_page_config(
    page_title="Justin's Tiered Growth Theme Stock Analyzer", 
    layout="wide"
)

st.title("Tiered High-Growth Theme Stock Analyzer")
st.markdown("Metrics sorted by **Income Statement Flow**. Featuring **Justin's Multi-Tier Criteria** and **2026 Fair Value Benchmarks**.")

# -----------------------------------------------------------------------------
# Fair P/E Configuration (2026 Benchmarks)
# -----------------------------------------------------------------------------
FAIR_PE_LOOKUP = {
    "Semiconductors": 42,
    "Software—Application": 32,
    "Software—Infrastructure": 28,
    "Aerospace & Defense": 40,
    "Data Processing": 25,
    "Information Technology Services": 25,
    "Financial Data & Stock Exchanges": 25,
    "Technology": 32
}
DEFAULT_PE = 20

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def get_metric_safe(info_dict, key):
    val = info_dict.get(key)
    if val is None or val == "" or val == "N/A":
        return np.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        return val 

def calculate_metrics(ticker_symbol):
    stock = yf.Ticker(ticker_symbol, session=session)

    try:
        info = stock.info
        if not info or len(info) < 5: return None
    except Exception: return None 

    q_fin = stock.quarterly_financials
    q_cf = stock.quarterly_cashflow
    
    # --- Identity ---
    name = info.get('longName', ticker_symbol)
    sector = info.get('sector', 'N/A')
    industry = info.get('industry', 'N/A')
    market_cap = get_metric_safe(info, 'marketCap')
    current_price = get_metric_safe(info, 'currentPrice') or get_metric_safe(info, 'regularMarketPrice')
    summary_text = info.get('longBusinessSummary', "No business summary available.")
    
    # --- Income Statement Flow Items ---
    total_rev = get_metric_safe(info, 'totalRevenue')
    ebitda = get_metric_safe(info, 'ebitda')
    fcf_raw = get_metric_safe(info, 'freeCashflow')
    
    rev_m = total_rev / 1e6 if not np.isnan(total_rev) else np.nan
    ebitda_m = ebitda / 1e6 if not np.isnan(ebitda) else np.nan
    fcf_m = fcf_raw / 1e6 if not np.isnan(fcf_raw) else np.nan

    # --- Growth Logic ---
    rev_growth_yoy = np.nan
    try:
        if not q_fin.empty and 'Total Revenue' in q_fin.index:
            revs = q_fin.loc['Total Revenue']
            if len(revs) >= 5: 
                rev_growth_yoy = (revs.iloc[0] - revs.iloc[4]) / revs.iloc[4]
            elif len(revs) >= 2:
                qoq = (revs.iloc[0] - revs.iloc[1]) / revs.iloc[1]
                rev_growth_yoy = ((1 + qoq) ** 4) - 1 
    except: pass

    # --- Rule of 40 (Strict: Rev Growth + EBITDA Margin) ---
    rule_of_40 = np.nan
    try:
        if not np.isnan(rev_growth_yoy) and not np.isnan(total_rev) and not np.isnan(ebitda):
            ebitda_margin = ebitda / total_rev
            rule_of_40 = rev_growth_yoy + ebitda_margin
    except: pass

    # --- Fair Value & Upside Logic ---
    fwd_eps = get_metric_safe(info, 'forwardEps')
    
    # Lookup Fair PE
    fair_pe = FAIR_PE_LOOKUP.get(industry, DEFAULT_PE)
    
    # Calculate Fair Value & Upside
    intrinsic_fair_value = np.nan
    upside_pct = np.nan
    if not np.isnan(fwd_eps) and fwd_eps > 0:
        intrinsic_fair_value = fwd_eps * fair_pe
        if not np.isnan(current_price) and current_price > 0:
            upside_pct = ((intrinsic_fair_value / current_price) - 1)

    return {
        "Ticker": ticker_symbol, "Company Name": name, "Sector": sector, "Industry": industry,
        "Market Size": market_cap, "Rev Growth (YoY)": rev_growth_yoy, "Revenue (M)": rev_m,
        "Gross Margin": get_metric_safe(info, 'grossMargins'), "EBITDA (M)": ebitda_m, "FCF (M)": fcf_m, 
        "CapEx Intensity": get_metric_safe(info, 'capExIntensity'), # Some APIs provide this directly
        "Price": current_price, "Fair P/E": fair_pe, "Fair Value": intrinsic_fair_value, 
        "Upside %": upside_pct, "Rule of 40": rule_of_40, "Summary": summary_text
    }

@st.cache_data(ttl=3600)
def load_data(tickers):
    data = []
    progress_bar = st.progress(0)
    for i, t in enumerate(tickers):
        progress_bar.progress((i + 1) / len(tickers))
        metrics = calculate_metrics(t.strip().upper())
        if metrics: data.append(metrics)
    progress_bar.empty()
    return pd.DataFrame(data)

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.title("📁 Portfolio Upload")
uploaded_file = st.sidebar.file_uploader("Upload Excel/CSV", type=["xlsx", "csv"])
if uploaded_file:
    user_data = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
    target_col = [col for col in user_data.columns if any(x in col.lower() for x in ['ticker', 'symbol'])]
    ticker_list = user_data[target_col[0]].dropna().unique().tolist() if target_col else user_data.iloc[:, 0].dropna().unique().tolist()
else:
    default_list = "NVDA, PLTR, ASML, CRWD, DDOG, SNOW, TSLA, MSFT, AVGO"
    ticker_list = [x.strip() for x in st.sidebar.text_area("Ticker List", value=default_list).split(',')]

if st.sidebar.button("Refresh All Data"):
    st.cache_data.clear()
    st.rerun()

# -----------------------------------------------------------------------------
# Main Dashboard
# -----------------------------------------------------------------------------
if ticker_list:
    raw_df = load_data(ticker_list)
    if not raw_df.empty:
        # Numeric cleanup
        num_cols = ["Market Size", "Rev Growth (YoY)", "Revenue (M)", "Gross Margin", "EBITDA (M)", "FCF (M)", 
                    "Price", "Fair Value", "Upside %", "Rule of 40"]
        for col in num_cols: raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce')

        # Justin's Tier Logic
        def get_tier(row):
            g, m, fcf = row.get('Rev Growth (YoY)', 0), row.get('Gross Margin', 0), row.get('FCF (M)', 0)
            if g > 0.40 and m > 0.70 and fcf > 0: return 1
            if g > 0.30 and m > 0.60 and fcf <= 0: return 3
            if g > 0.30 and m > 0.60: return 2
            return 0
        raw_df['Tier'] = raw_df.apply(get_tier, axis=1)

        # Prepare Display
        df_display = raw_df.copy()
        pct_cols = ['Rev Growth (YoY)', 'Gross Margin', 'Rule of 40', 'Upside %']
        for col in pct_cols: df_display[col] = df_display[col] * 100
        
        cols_order = ["Ticker", "Company Name", "Sector", "Price", "Rev Growth (YoY)", "Revenue (M)", 
                      "Gross Margin", "EBITDA (M)", "FCF (M)", "Rule of 40", 
                      "Fair P/E", "Fair Value", "Upside %"]

        # Visible slice for styling
        visible_df = df_display[[c for c in cols_order if c in df_display.columns]]

        def tiered_style(row):
            styles = [''] * len(row)
            tier = raw_df.loc[row.name, 'Tier']
            curr_price = row.get("Price")
            
            # 1. Tier Backgrounds
            bg = ''
            if tier == 1: bg = 'background-color: #2ECC71; color: white;' # Green
            elif tier == 2: bg = 'background-color: #F1C40F; color: black;' # Yellow
            elif tier == 3: bg = 'background-color: #E67E22; color: white;' # Orange
            styles = [bg] * len(row)

            # 2. Undervalued Highlight (Cyan)
            idx_fv = visible_df.columns.get_loc("Fair Value")
            if pd.notnull(row[idx_fv]) and row[idx_fv] > curr_price:
                styles[idx_fv] = 'background-color: #00FFFF; color: black; font-weight: bold;'
            
            idx_up = visible_df.columns.get_loc("Upside %")
            if pd.notnull(row[idx_up]) and row[idx_up] > 0:
                styles[idx_up] = 'background-color: #00FFFF; color: black; font-weight: bold;'
                
            return styles

        st.dataframe(
            visible_df.style.apply(tiered_style, axis=1).format({
                "Price": "${:.2f}", "Rev Growth (YoY)": "{:.1f}%", "Revenue (M)": "${:,.0f}M",
                "Gross Margin": "{:.1f}%", "EBITDA (M)": "${:,.0f}M", "FCF (M)": "${:,.1f}M",
                "Rule of 40": "{:.1f}", "Fair P/E": "{:.0f}x", "Fair Value": "${:.2f}", 
                "Upside %": "{:.1f}%"
            }, na_rep="N/A"),
            use_container_width=True, hide_index=True
        )

        # Summaries
        st.divider()
        st.header("🎯 Justin's Conviction Summaries")
        for t_num, label in [(1, "Tier 1: Elite"), (2, "Tier 2: Solid"), (3, "Tier 3: High Burn")]:
            tier_df = raw_df[raw_df['Tier'] == t_num]
            if not tier_df.empty:
                st.subheader(label)
                for _, row in tier_df.iterrows():
                    with st.expander(f"💡 {row['Ticker']} - {row['Company Name']}"):
                        c1, c2 = st.columns([1, 2])
                        with c1:
                            st.metric("Upside", f"{row['Upside %']*100:.1f}%" if pd.notnull(row['Upside %']) else "N/A")
                            st.metric("Fair Value", f"${row['Fair Value']:.2f}" if pd.notnull(row['Fair Value']) else "N/A")
                            st.write(f"**Industry:** {row['Industry']}")
                        with c2:
                            st.write("**Business Summary:**")
                            st.write(row.get('Summary', 'N/A'))
    else: st.warning("No data found for the input tickers.")

else: st.info("Upload file or enter tickers in sidebar.")

