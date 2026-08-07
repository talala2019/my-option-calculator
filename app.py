import streamlit as st
import math
import datetime
import pandas as pd
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

# --- Core Logic: Delta ---
def calculate_delta(S, K, days, r_pct, iv_pct, option_type='call'):
    T = days / 365.0
    r = r_pct / 100.0
    sigma = iv_pct / 100.0

    if T <= 0:
        # At expiry, Delta is binary: in-the-money -> 1/-1, out-of-the-money -> 0.
        if option_type == 'call':
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1

# --- Core Logic: Days until the next Friday ---
def days_until_next_friday(today=None):
    """Days from `today` to the next upcoming Friday. Always strictly in the
    future -- if `today` is itself a Friday, returns 7 (next week's Friday),
    not 0."""
    if today is None:
        today = datetime.date.today()
    days_ahead = (4 - today.weekday()) % 7  # Monday=0 ... Friday=4
    if days_ahead == 0:
        days_ahead = 7
    return days_ahead

def next_friday_label(today=None):
    """Next Friday's date formatted like '2026 AUG 14', for display on the
    quick-fill button."""
    if today is None:
        today = datetime.date.today()
    target = today + datetime.timedelta(days=days_until_next_friday(today))
    return target.strftime("%Y %b %d").upper()

def days_until_this_friday(today=None):
    """Days from `today` to this week's Friday. 0 if today is Friday itself.
    Clamped to 0 if today is Sat/Sun (that Friday has already passed)."""
    if today is None:
        today = datetime.date.today()
    return max(4 - today.weekday(), 0)  # Monday=0 ... Friday=4

def this_friday_label(today=None):
    """This week's Friday date formatted like '2026 AUG 07'."""
    if today is None:
        today = datetime.date.today()
    target = today + datetime.timedelta(days=days_until_this_friday(today))
    return target.strftime("%Y %b %d").upper()

# --- UI Helper: colored Call/Put result panel ---
def render_result_panel(container, header, metric_label, metric_value, delta_value, color):
    """Render a colored card showing one option side's result + Delta.
    color: 'red' (買權/Call, TW market convention) or 'green' (賣權/Put)."""
    if color == "red":
        bg, border = "#fdecea", "#e57373"
    else:
        bg, border = "#e6f4ea", "#66bb6a"
    container.markdown(
        f"""
<div style="background-color:{bg};border:1px solid {border};border-radius:10px;
            padding:14px 16px;">
  <div style="font-size:0.9rem;color:#555;margin-bottom:4px;">{header}</div>
  <div style="font-size:1.4rem;font-weight:700;color:#222;">{metric_label}: {metric_value}</div>
  <div style="font-size:0.95rem;color:#333;margin-top:6px;">Δ Delta: {delta_value:.4f}</div>
</div>
""",
        unsafe_allow_html=True,
    )

# --- UI Section: one ticker's Pricing panel (Tab 1) ---
# Each ticker gets fully independent widgets (own keys), so there is no
# "which ticker is active" state to track and nothing to keep in sync --
# what's on screen for TSM and what's on screen for MU are just two
# ordinary sets of Streamlit widgets. That also means this data lives only
# in st.session_state, which Streamlit already keeps private per browser
# connection -- no shared file, so nothing for multiple visitors to leak
# into each other. The trade-off: a page reload starts a fresh session, so
# values reset to the defaults below, same as any other Streamlit widget.
def render_pricing_section(ticker):
    def k(name):
        return f"{name}_{ticker}"

    d = TICKER_DEFAULTS[ticker]

    if f"history_{ticker}" not in st.session_state:
        st.session_state[f"history_{ticker}"] = []

    # Loading a saved history row has the same timing constraint as the
    # Friday buttons below: it can only write s_p/k_p/etc. because this runs
    # before those widgets are (re-)created further down.
    pending_load = st.session_state.pop(f"_load_row_{ticker}", None)
    if pending_load is not None:
        row = st.session_state[f"history_{ticker}"][pending_load]
        st.session_state[k("s_p")] = row["S"]
        st.session_state[k("k_p")] = row["K"]
        st.session_state[k("r_p")] = row["r"]
        st.session_state[k("iv_p")] = row["IV"]
        st.session_state[k("d_p")] = row["days"]

    if st.session_state.pop(f"_apply_this_friday_p_{ticker}", False):
        st.session_state[k("d_p")] = float(days_until_this_friday())
    if st.session_state.pop(f"_apply_next_friday_p_{ticker}", False):
        st.session_state[k("d_p")] = float(days_until_next_friday())

    col1, col2, col3 = st.columns(3)
    with col1:
        S = st.number_input("Current Price (標的價)", value=d["S"], step=1.0, key=k("s_p"))
        r1 = st.number_input("Risk-free Rate % (利率)", value=d["r"], step=0.1, key=k("r_p"))
    with col2:
        K = st.number_input("Strike Price (履約價)", value=d["K"], step=1.0, key=k("k_p"))
        iv1 = st.number_input("Implied Volatility % (IV 隱含波動率)", value=d["iv"], step=1.0, key=k("iv_p"))
    with col3:
        days1 = st.number_input("Days to Expiry (天數)", value=d["days"], step=1.0, key=k("d_p"))
        if st.button(f"📅 算至本週五 ({this_friday_label()})", key=k("this_friday_p")):
            st.session_state[f"_apply_this_friday_p_{ticker}"] = True
            st.rerun()
        if st.button(f"📅 算至下週五 ({next_friday_label()})", key=k("next_friday_p")):
            st.session_state[f"_apply_next_friday_p_{ticker}"] = True
            st.rerun()

    if st.button("Calculate Premium", type="primary", key=k("btn_p")):
        call, put = calculate_black_scholes(S, K, days1, r1, iv1)
        call_delta = calculate_delta(S, K, days1, r1, iv1, 'call')
        put_delta = calculate_delta(S, K, days1, r1, iv1, 'put')
        st.session_state[f"history_{ticker}"].append({
            "S": S, "K": K, "r": r1, "IV": iv1, "days": days1,
            "Call": round(call, 4), "Put": round(put, 4),
            "CallΔ": round(call_delta, 4), "PutΔ": round(put_delta, 4),
        })
        st.divider()
        st.success(f"Calculation Complete! (for {ticker})")
        p1, p2 = st.columns(2)
        render_result_panel(p1, "📈 買權 Call", "Price", f"${call:.4f}", call_delta, "red")
        render_result_panel(p2, "📉 賣權 Put", "Price", f"${put:.4f}", put_delta, "green")

    # Calculation history: click a row to select it, then load it back into
    # the inputs above or delete it. Session-only (see the module docstring
    # comment above render_pricing_section) -- resets on reload, same as
    # everything else here, and never written to disk.
    history = st.session_state[f"history_{ticker}"]
    if history:
        st.divider()
        st.subheader(f"📜 計算紀錄（{ticker}）")
        st.caption("本次連線期間有效，reload 頁面會清空")
        hist_df = pd.DataFrame(history).rename(columns={
            "S": "股價", "K": "履約價", "r": "利率%", "IV": "IV%", "days": "天數",
            "Call": "Call 價", "Put": "Put 價", "CallΔ": "Call Δ", "PutΔ": "Put Δ",
        })
        event = st.dataframe(
            hist_df, on_select="rerun", selection_mode="single-row",
            key=k("history_table"), hide_index=True,
            column_config={
                "股價": st.column_config.NumberColumn(format="%.1f"),
                "履約價": st.column_config.NumberColumn(format="%.1f"),
                "利率%": st.column_config.NumberColumn(format="%.1f"),
                "IV%": st.column_config.NumberColumn(format="%.1f"),
                "天數": st.column_config.NumberColumn(format="%.0f"),
                "Call 價": st.column_config.NumberColumn(format="%.2f"),
                "Put 價": st.column_config.NumberColumn(format="%.2f"),
                "Call Δ": st.column_config.NumberColumn(format="%.3f"),
                "Put Δ": st.column_config.NumberColumn(format="%.3f"),
            },
        )
        selected_rows = event["selection"]["rows"]
        if selected_rows:
            idx = selected_rows[0]
            lc, dc = st.columns(2)
            if lc.button("📥 載入此列", key=k("load_row"), use_container_width=True):
                st.session_state[f"_load_row_{ticker}"] = idx
                st.rerun()
            if dc.button("🗑️ 刪除此列", key=k("delete_row"), use_container_width=True):
                st.session_state[f"history_{ticker}"].pop(idx)
                st.rerun()

# --- UI Section: one ticker's Implied Volatility panel (Tab 2) ---
def render_iv_section(ticker):
    def k(name):
        return f"{name}_{ticker}"

    d = TICKER_DEFAULTS[ticker]

    if st.session_state.pop(f"_apply_this_friday_iv_{ticker}", False):
        st.session_state[k("d_iv")] = float(days_until_this_friday())
    if st.session_state.pop(f"_apply_next_friday_iv_{ticker}", False):
        st.session_state[k("d_iv")] = float(days_until_next_friday())

    col1, col2, col3 = st.columns(3)
    with col1:
        S2 = st.number_input("Current Price (標的價)", value=d["S"], step=1.0, key=k("s_iv"))
        days2 = st.number_input("Days to Expiry (天數)", value=d["days"], step=1.0, key=k("d_iv"))
        if st.button(f"📅 算至本週五 ({this_friday_label()})", key=k("this_friday_iv")):
            st.session_state[f"_apply_this_friday_iv_{ticker}"] = True
            st.rerun()
        if st.button(f"📅 算至下週五 ({next_friday_label()})", key=k("next_friday_iv")):
            st.session_state[f"_apply_next_friday_iv_{ticker}"] = True
            st.rerun()
    with col2:
        K2 = st.number_input("Strike Price (履約價)", value=d["K"], step=1.0, key=k("k_iv"))
        r2 = st.number_input("Risk-free Rate % (利率)", value=d["r"], step=0.1, key=k("r_iv"))
    with col3:
        target_price = st.number_input("Market Premium (Bid/Ask MID 權利金)", value=d["premium"], step=0.1, key=k("target_iv"))
        opt_type = st.selectbox("Option Type (類型)", ["put", "call"], key=k("type_iv"))

    if st.button("Calculate Implied Volatility", type="primary", key=k("btn_iv")):
        iv_result = calculate_iv(S2, K2, target_price, days2, r2, opt_type)
        st.divider()
        if iv_result is not None and iv_result > 0:
            st.success(f"Calculation Complete! (for {ticker})")
            delta_result = calculate_delta(S2, K2, days2, r2, iv_result, opt_type)
            if opt_type == "call":
                header, color = "📈 買權 Call", "red"
            else:
                header, color = "📉 賣權 Put", "green"
            render_result_panel(
                st, header, "Implied Volatility", f"{iv_result:.2f} %", delta_result, color
            )
        else:
            st.error("Could not converge. Please check if the premium is lower than the intrinsic value.")

# --- Streamlit UI Design ---
st.set_page_config(page_title="Options Pricing & IV Calculator", layout="centered")

st.title("📈 Options Pricing & IV Calculator")

# Color the two expiry quick-fill buttons differently so they're easy to
# tell apart at a glance: this-Friday blue, next-Friday orange. Matches any
# widget whose Streamlit-assigned `st-key-<key>` class contains these
# prefixes, so it covers both tickers' buttons in both tabs without listing
# every key explicitly.
st.markdown(
    """
<style>
[class*="st-key-this_friday"] button { color: #1976d2 !important; }
[class*="st-key-next_friday"] button { color: #f57c00 !important; }
</style>
""",
    unsafe_allow_html=True,
)

TICKERS = ["TSM", "MU"]
# Starting numbers shown before you've typed anything -- just a sane
# ballpark per ticker's actual price scale (TSM ~$418, MU ~$881 as of
# Aug 2026), not live data. Edit freely; these aren't fetched or refreshed.
TICKER_DEFAULTS = {
    "TSM": {"S": 418.0, "K": 390.0, "r": 3.8, "iv": 45.0, "days": 8.0, "premium": 1.98},
    "MU": {"S": 880.0, "K": 750.0, "r": 3.8, "iv": 78.0, "days": 8.0, "premium": 4.5},
}

# Create Tabs for the two main functions
tab1, tab2 = st.tabs(["💰 Pricing (權利金計算)", "📊 Implied Volatility (IV 隱含波動率 反推)"])

# --- TAB 1: PRICING ---
with tab1:
    st.subheader("Input Market Parameters")
    ticker_subtabs_1 = st.tabs(TICKERS)
    for ticker, subtab in zip(TICKERS, ticker_subtabs_1):
        with subtab:
            render_pricing_section(ticker)

# --- TAB 2: IV CALCULATION ---
with tab2:
    st.subheader("Reverse IV from Market Premium")
    ticker_subtabs_2 = st.tabs(TICKERS)
    for ticker, subtab in zip(TICKERS, ticker_subtabs_2):
        with subtab:
            render_iv_section(ticker)
