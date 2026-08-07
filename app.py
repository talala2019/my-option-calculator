import streamlit as st
import math
import datetime
import json
from pathlib import Path
from scipy.stats import norm

# --- Persistence: remember TSM/MU presets across page reloads ---
# st.session_state only lives for one browser connection -- reloading the
# page starts a brand new session and wipes it. A small JSON file next to
# app.py survives reloads (it doesn't survive a fresh redeploy, which is
# fine: this is "don't lose today's numbers on refresh", not a database).
PRESETS_FILE = Path(__file__).resolve().parent / ".option_calc_presets.json"

def load_presets_from_disk():
    if not PRESETS_FILE.exists():
        return None
    try:
        return json.loads(PRESETS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

def save_presets_to_disk(presets, active_ticker_tab1, active_ticker_tab2):
    data = {
        "presets": presets,
        "active_ticker_tab1": active_ticker_tab1,
        "active_ticker_tab2": active_ticker_tab2,
    }
    try:
        PRESETS_FILE.write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        pass

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

# --- Streamlit UI Design ---
st.set_page_config(page_title="Options Pricing & IV Calculator", layout="centered")

st.title("📈 Options Pricing & IV Calculator")

# Color the two expiry quick-fill buttons differently so they're easy to
# tell apart at a glance: this-Friday blue, next-Friday orange. Targets the
# `st-key-<key>` class Streamlit adds to each widget's wrapper.
st.markdown(
    """
<style>
.st-key-this_friday_p button, .st-key-this_friday_iv button { color: #1976d2 !important; }
.st-key-next_friday_p button, .st-key-next_friday_iv button { color: #f57c00 !important; }
</style>
""",
    unsafe_allow_html=True,
)

# Per-tab memory of each ticker's last-used inputs, so switching between
# TSM/MU doesn't lose what you typed for the other one -- and so reloading
# the page doesn't lose it either.
if "presets" not in st.session_state:
    disk_data = load_presets_from_disk()
    if disk_data:
        st.session_state["presets"] = disk_data.get("presets", {"tab1": {}, "tab2": {}})
        st.session_state["active_ticker_tab1"] = disk_data.get("active_ticker_tab1")
        st.session_state["active_ticker_tab2"] = disk_data.get("active_ticker_tab2")
    else:
        st.session_state["presets"] = {"tab1": {}, "tab2": {}}
        st.session_state["active_ticker_tab1"] = None
        st.session_state["active_ticker_tab2"] = None

    # This is the first run of a brand new session (e.g. right after a page
    # reload) -- no widget has been created yet, so it's safe to seed their
    # session_state values here, same as the ticker-button click handlers do.
    active1 = st.session_state["active_ticker_tab1"]
    preset1 = st.session_state["presets"]["tab1"].get(active1) if active1 else None
    if preset1:
        st.session_state["s_p"] = preset1["S"]
        st.session_state["k_p"] = preset1["K"]
        st.session_state["r_p"] = preset1["r1"]
        st.session_state["iv_p"] = preset1["iv1"]
        st.session_state["d_p"] = preset1["days1"]

    active2 = st.session_state["active_ticker_tab2"]
    preset2 = st.session_state["presets"]["tab2"].get(active2) if active2 else None
    if preset2:
        st.session_state["s_iv"] = preset2["S2"]
        st.session_state["k_iv"] = preset2["K2"]
        st.session_state["r_iv"] = preset2["r2"]
        st.session_state["target_iv"] = preset2["target_price"]
        st.session_state["d_iv"] = preset2["days2"]
        st.session_state["type_iv"] = preset2["opt_type"]

TICKERS = ["TSM", "MU"]
# Fallback shown the first time a ticker is selected and has no saved preset
# yet, so switching to an unseen ticker starts from a clean baseline instead
# of carrying over whatever the previous ticker's numbers happened to be.
DEFAULT_TAB1 = {"S": 418.0, "K": 390.0, "r1": 3.8, "iv1": 45.0, "days1": 8.0}
DEFAULT_TAB2 = {
    "S2": 418.0, "K2": 390.0, "r2": 3.8, "target_price": 1.98,
    "days2": 8.0, "opt_type": "put",
}

# Create Tabs for the two main functions
tab1, tab2 = st.tabs(["💰 Pricing (權利金計算)", "📊 Implied Volatility (IV 隱含波動率 反推)"])

# --- TAB 1: PRICING ---
with tab1:
    st.subheader("Input Market Parameters")

    # Ticker quick-switch: restores the last values you used for this ticker.
    # Must run BEFORE the number_inputs below so the session_state values it
    # sets are picked up as those widgets' initial values this run.
    ticker_cols = st.columns(len(TICKERS))
    for i, ticker in enumerate(TICKERS):
        is_active = ticker == st.session_state["active_ticker_tab1"]
        clicked = ticker_cols[i].button(
            ticker, key=f"ticker_{ticker}_p", use_container_width=True,
            type="primary" if is_active else "secondary",
        )
        if clicked:
            st.session_state["active_ticker_tab1"] = ticker
            preset = st.session_state["presets"]["tab1"].get(ticker, DEFAULT_TAB1)
            st.session_state["s_p"] = preset["S"]
            st.session_state["k_p"] = preset["K"]
            st.session_state["r_p"] = preset["r1"]
            st.session_state["iv_p"] = preset["iv1"]
            st.session_state["d_p"] = preset["days1"]
            # Rerun so the buttons above re-render with the new highlight --
            # without this, the just-clicked button stays unhighlighted until
            # the next unrelated interaction (see Tab 1's Friday-button
            # comment below for why the same-run render can't reflect it).
            st.rerun()
    active_tab1 = st.session_state["active_ticker_tab1"]
    if active_tab1:
        st.caption(f"目前追蹤: **{active_tab1}**（輸入變更會自動記住）")

    # A button placed after its target widget can't write that widget's
    # session_state key in the same run (Streamlit forbids mutating an
    # already-instantiated widget's key). So the click just sets a pending
    # flag + reruns; this check -- which runs before the widget below -- is
    # what actually applies the new value.
    if st.session_state.pop("_apply_this_friday_p", False):
        st.session_state["d_p"] = float(days_until_this_friday())
    if st.session_state.pop("_apply_next_friday_p", False):
        st.session_state["d_p"] = float(days_until_next_friday())

    col1, col2, col3 = st.columns(3)

    with col1:
        S = st.number_input("Current Price (標的價)", value=418.0, step=1.0, key="s_p")
        r1 = st.number_input("Risk-free Rate % (利率)", value=3.8, step=0.1, key="r_p")
    with col2:
        K = st.number_input("Strike Price (履約價)", value=390.0, step=1.0, key="k_p")
        iv1 = st.number_input("Implied Volatility % (IV 隱含波動率)", value=45.0, step=1.0, key="iv_p")
    with col3:
        days1 = st.number_input("Days to Expiry (天數)", value=8.0, step=1.0, key="d_p")
        if st.button(f"📅 算至本週五 ({this_friday_label()})", key="this_friday_p"):
            st.session_state["_apply_this_friday_p"] = True
            st.rerun()
        if st.button(f"📅 算至下週五 ({next_friday_label()})", key="next_friday_p"):
            st.session_state["_apply_next_friday_p"] = True
            st.rerun()

    # Remember this ticker's latest inputs for next time it's selected --
    # in session_state (same-session tab switches) and on disk (page reloads).
    if active_tab1:
        st.session_state["presets"]["tab1"][active_tab1] = {
            "S": S, "K": K, "r1": r1, "iv1": iv1, "days1": days1,
        }
        save_presets_to_disk(
            st.session_state["presets"],
            st.session_state["active_ticker_tab1"],
            st.session_state["active_ticker_tab2"],
        )

    if st.button("Calculate Premium", type="primary", key="btn_p"):
        call, put = calculate_black_scholes(S, K, days1, r1, iv1)
        call_delta = calculate_delta(S, K, days1, r1, iv1, 'call')
        put_delta = calculate_delta(S, K, days1, r1, iv1, 'put')
        st.divider()
        st.success(f"Calculation Complete! (for {active_tab1})" if active_tab1 else "Calculation Complete!")
        p1, p2 = st.columns(2)
        render_result_panel(p1, "📈 買權 Call", "Price", f"${call:.4f}", call_delta, "red")
        render_result_panel(p2, "📉 賣權 Put", "Price", f"${put:.4f}", put_delta, "green")

# --- TAB 2: IV CALCULATION ---
with tab2:
    st.subheader("Reverse IV from Market Premium")

    # Ticker quick-switch (same pattern as Tab 1, own memory).
    ticker_cols2 = st.columns(len(TICKERS))
    for i, ticker in enumerate(TICKERS):
        is_active2 = ticker == st.session_state["active_ticker_tab2"]
        clicked2 = ticker_cols2[i].button(
            ticker, key=f"ticker_{ticker}_iv", use_container_width=True,
            type="primary" if is_active2 else "secondary",
        )
        if clicked2:
            st.session_state["active_ticker_tab2"] = ticker
            preset = st.session_state["presets"]["tab2"].get(ticker, DEFAULT_TAB2)
            st.session_state["s_iv"] = preset["S2"]
            st.session_state["k_iv"] = preset["K2"]
            st.session_state["r_iv"] = preset["r2"]
            st.session_state["target_iv"] = preset["target_price"]
            st.session_state["d_iv"] = preset["days2"]
            st.session_state["type_iv"] = preset["opt_type"]
            st.rerun()  # see Tab 1's comment on why this is needed for the highlight
    active_tab2 = st.session_state["active_ticker_tab2"]
    if active_tab2:
        st.caption(f"目前追蹤: **{active_tab2}**（輸入變更會自動記住）")

    # See Tab 1's comment: apply the pending value before the widget renders.
    if st.session_state.pop("_apply_this_friday_iv", False):
        st.session_state["d_iv"] = float(days_until_this_friday())
    if st.session_state.pop("_apply_next_friday_iv", False):
        st.session_state["d_iv"] = float(days_until_next_friday())

    col1, col2, col3 = st.columns(3)

    with col1:
        S2 = st.number_input("Current Price (標的價)", value=418.0, step=1.0, key="s_iv")
        days2 = st.number_input("Days to Expiry (天數)", value=8.0, step=1.0, key="d_iv")
        if st.button(f"📅 算至本週五 ({this_friday_label()})", key="this_friday_iv"):
            st.session_state["_apply_this_friday_iv"] = True
            st.rerun()
        if st.button(f"📅 算至下週五 ({next_friday_label()})", key="next_friday_iv"):
            st.session_state["_apply_next_friday_iv"] = True
            st.rerun()
    with col2:
        K2 = st.number_input("Strike Price (履約價)", value=390.0, step=1.0, key="k_iv")
        r2 = st.number_input("Risk-free Rate % (利率)", value=3.8, step=0.1, key="r_iv")
    with col3:
        target_price = st.number_input("Market Premium (Bid/Ask MID 權利金)", value=1.98, step=0.1, key="target_iv")
        opt_type = st.selectbox("Option Type (類型)", ["put", "call"], key="type_iv")

    # Remember this ticker's latest inputs for next time it's selected --
    # in session_state (same-session tab switches) and on disk (page reloads).
    if active_tab2:
        st.session_state["presets"]["tab2"][active_tab2] = {
            "S2": S2, "K2": K2, "r2": r2, "target_price": target_price,
            "days2": days2, "opt_type": opt_type,
        }
        save_presets_to_disk(
            st.session_state["presets"],
            st.session_state["active_ticker_tab1"],
            st.session_state["active_ticker_tab2"],
        )

    if st.button("Calculate Implied Volatility", type="primary", key="btn_iv"):
        iv_result = calculate_iv(S2, K2, target_price, days2, r2, opt_type)
        st.divider()
        if iv_result is not None and iv_result > 0:
            st.success(f"Calculation Complete! (for {active_tab2})" if active_tab2 else "Calculation Complete!")
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
