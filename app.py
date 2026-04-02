import streamlit as st
import math
from scipy.stats import norm

# --- Core Logic: Black-Scholes Pricing ---
def calculate_black_scholes(S, K, days, r_pct, iv_pct):
    T = days / 365.0
    r = r_pct / 100.0
    sigma = iv_pct / 100.0
    
    if T <= 0:
        return max(0, S - K), max(0, K - S)
        
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    put_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return call_price, put_price

# --- Core Logic: Newton-Raphson for Implied Volatility ---
def calculate_iv(S, K, target_price, days, r_pct, option_type='put'):
    T = days / 365.0
    r = r_pct / 100.0
    sigma = 0.5  # Initial guess
    
    for i in range(100):
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
        diff = price - target_price
        if abs(diff) < 1e-5:
            return sigma * 100.0
            
        vega = S * norm.pdf(d1) * math.sqrt(T)
        if vega == 0:
            return None
        sigma = sigma - diff / vega
    return sigma * 100.0

# --- Streamlit UI Design ---
st.set_page_config(page_title="Options Pricing & IV Calculator", layout="centered")

st.title("📈 Options Pricing & IV Calculator")
st.markdown("Calculate Black-Scholes theoretical prices or reverse-engineer Implied Volatility (IV).")
st.markdown("This tool uses the Black-Scholes model and Newton-Raphson iteration for IV calculation.")

# Create Tabs for the two main functions
tab1, tab2 = st.tabs(["💰 Pricing (權利金計算)", "📊 Implied Volatility (IV 隱含波動率 反推)"])

# --- TAB 1: PRICING ---
with tab1:
    st.subheader("Input Market Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        S = st.number_input("Current Price (標的價)", value=343.25, step=1.0, key="s_p")
        r1 = st.number_input("Risk-free Rate % (利率)", value=3.7, step=0.1, key="r_p")
    with col2:
        K = st.number_input("Strike Price (履約價)", value=320.0, step=1.0, key="k_p")
        iv1 = st.number_input("Implied Volatility % (IV 隱含波動率)", value=57.0, step=1.0, key="iv_p")
    with col3:
        days1 = st.number_input("Days to Expiry (天數)", value=3.0, step=1.0, key="d_p")

    if st.button("Calculate Premium", type="primary", key="btn_p"):
        call, put = calculate_black_scholes(S, K, days1, r1, iv1)
        st.divider()
        st.success("Calculation Complete!")
        c1, c2 = st.columns(2)
        c1.metric(label="📈 Call Price (買權)", value=f"${call:.4f}")
        c2.metric(label="📉 Put Price (賣權)", value=f"${put:.4f}")

# --- TAB 2: IV CALCULATION ---
with tab2:
    st.subheader("Reverse IV from Market Premium")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        S2 = st.number_input("Current Price (標的價)", value=343.25, step=1.0, key="s_iv")
        days2 = st.number_input("Days to Expiry (天數)", value=3.0, step=1.0, key="d_iv")
    with col2:
        K2 = st.number_input("Strike Price (履約價)", value=320.0, step=1.0, key="k_iv")
        r2 = st.number_input("Risk-free Rate % (利率)", value=3.7, step=0.1, key="r_iv")
    with col3:
        target_price = st.number_input("Market Premium (Bid/Ask MID 權利金)", value=0.70, step=0.1, key="target_iv")
        opt_type = st.selectbox("Option Type (類型)", ["put", "call"], key="type_iv")

    if st.button("Calculate Implied Volatility", type="primary", key="btn_iv"):
        iv_result = calculate_iv(S2, K2, target_price, days2, r2, opt_type)
        st.divider()
        if iv_result is not None and iv_result > 0:
            st.success("Calculation Complete!")
            st.metric(label=f"📊 Implied Volatility (IV)", value=f"{iv_result:.2f} %")
        else:
            st.error("Could not converge. Please check if the premium is lower than the intrinsic value.")
