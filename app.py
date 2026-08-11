import streamlit as st
import math
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
from scipy.signal import argrelextrema

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
    """Days from `today` to the Friday of the FOLLOWING week -- always
    distinct from days_until_this_friday(). On Mon-Fri that means skipping
    past this week's Friday entirely (even midweek, not just when today IS
    Friday); on Sat/Sun, this week's Friday is already gone, so there's no
    week to skip and this is just the nearest upcoming Friday."""
    if today is None:
        today = datetime.date.today()
    raw = 4 - today.weekday()  # Monday=4 ... Friday=0, Saturday=-1, Sunday=-2
    return raw + 7 if raw >= 0 else raw % 7

def next_friday_label(today=None):
    """Next Friday's date formatted like '08/14', for display on the
    quick-fill button. Short on purpose -- the fuller '2026 AUG 14' format
    was wrapping onto two lines inside the button in narrower layouts."""
    if today is None:
        today = datetime.date.today()
    target = today + datetime.timedelta(days=days_until_next_friday(today))
    return target.strftime("%m/%d")

def days_until_this_friday(today=None):
    """Days from `today` to this week's Friday. 0 if today is Friday itself.
    None if today is Sat/Sun -- that Friday has already passed, and there's
    no non-negative day count that still means "this Friday" at that point
    (0 would silently mean "today", mislabeling a Saturday as a Friday)."""
    if today is None:
        today = datetime.date.today()
    days = 4 - today.weekday()  # Monday=0 ... Friday=4
    return days if days >= 0 else None

def this_friday_label(today=None):
    """This week's Friday date formatted like '08/07'. Only call this when
    days_until_this_friday() is not None."""
    if today is None:
        today = datetime.date.today()
    target = today + datetime.timedelta(days=days_until_this_friday(today))
    return target.strftime("%m/%d")

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

def strike_pct_of_price_caption(K, S):
    """'= 92.3% of 標的價' style live readout of how the current Strike
    Price input compares to the current Current Price input -- recomputed
    on every rerun from whatever's in those two widgets right now, so it
    tracks manual edits as well as the discount/round-number buttons."""
    if not S:
        return "= — % of 標的價"
    return f"= {K / S * 100:.1f}% of 標的價"

# Column formatters for the printable history table, matching the
# column_config formats used on the interactive st.dataframe version above.
_HISTORY_PRINT_FORMATTERS = {
    "股價": "{:.1f}".format, "履約價": "{:.1f}".format,
    "折數": "{:.1f}%".format, "利率%": "{:.1f}".format, "IV%": "{:.1f}".format,
    "天數": "{:.0f}".format, "Call 價": "${:.2f}".format, "Put 價": "${:.2f}".format,
    "Call Δ": "{:.3f}".format, "Put Δ": "{:.3f}".format,
}

HISTORY_STATUS_OPTIONS = ["一般", "已成交", "收盤", "盤前", "盤中", "估價"]
_STATUS_ROW_COLOR = {
    "已成交": "#bbdefb",  # light blue
    "盤前": "#fff9c4",    # light yellow
    "估價": "#e1bee7",    # light purple (was light green -- clashed with the Put price column's green)
    # 一般 / 收盤 / 盤中: no color
}

def _status_row_background(row):
    """Styler .apply(axis=1) callback: tint the whole row by its 標記
    (status) column, so a manually-noted trade is visually flagged the same
    way whether you're looking at the on-screen table or the printed one."""
    color = _STATUS_ROW_COLOR.get(row.get("標記"))
    if not color:
        return [""] * len(row)
    return [f"background-color: {color}"] * len(row)

def style_history_table(df):
    """Shared styling for both the interactive st.dataframe table and the
    print/PDF export -- row tint by status first, then Call/Put price
    columns' own fixed red/green on top (those always win over the row
    tint, since they're applied after in the chain). Either price column
    may be absent (the Call/Put visibility checkboxes can drop them), so
    each tint is only applied if that column actually survived."""
    styled = df.style.apply(_status_row_background, axis=1)
    if "Call 價" in df.columns:
        styled = styled.set_properties(subset=["Call 價"], **{"background-color": "#f5c6c6", "color": "#222"})
    if "Put 價" in df.columns:
        styled = styled.set_properties(subset=["Put 價"], **{"background-color": "#c3e6cb", "color": "#222"})
    return styled

_PRINT_TABLE_STYLE = """
table.print-history {
    border-collapse: collapse; width: auto; min-width: 100%; font-size: 0.9rem;
    print-color-adjust: exact; -webkit-print-color-adjust: exact;
}
table.print-history th, table.print-history td {
    border: 1px solid #ccc; padding: 6px 10px; text-align: center;
}
/* Headers are short labels (股價/利率%/標記/...) that only wrapped because
   the table was squeezed into a narrow column -- keep them on one line and
   let the table itself grow wider (scrolling sideways if needed) instead. */
table.print-history th { white-space: nowrap; }
table.print-history thead th { background-color: #37474f; color: #fff; font-weight: 600; }
table.print-history tbody tr:nth-child(even) td { background-color: #f7f7f7; }
@media print {
    @page { size: landscape; }
}
"""

def render_printable_history(hist_df, ticker):
    """A plain-HTML copy of the history table for printing/Save-as-PDF.
    st.dataframe renders as an interactive canvas grid, which browsers
    often print blank or truncated -- a real <table> prints reliably.

    Offered as a downloadable standalone .html file rather than an
    in-page "print only this part" CSS trick: isolating one element from
    the rest of a live Streamlit page for printing turned out fragile in
    practice (it either printed the whole page, or -- once the other
    content was hidden with enough force to fix that -- clipped the
    target itself to nothing, since it sits nested inside the very
    containers being collapsed). A standalone page has nothing else on it
    to fight with, so printing/saving it as PDF just works."""
    print_df = hist_df.copy()
    for col, fmt in _HISTORY_PRINT_FORMATTERS.items():
        if col in print_df.columns:
            print_df[col] = print_df[col].map(fmt)
    # Reuse the same row-status + Call/Put tinting as the on-screen table
    # (via pandas Styler), so a manually-marked "已成交"/"觀察中" row and
    # the price colors carry over into the exported page automatically.
    table_html = style_history_table(print_df).set_table_attributes(
        'class="print-history"'
    ).hide(axis="index").to_html()
    standalone_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{ticker} 計算紀錄</title>
<style>
body {{ font-family: -apple-system, "Segoe UI", "PingFang TC", "Microsoft JhengHei", sans-serif; margin: 24px; }}
{_PRINT_TABLE_STYLE}
.print-scroll {{ overflow-x: auto; }}
</style>
</head><body>
<h2>{ticker} 計算紀錄</h2>
<div class="print-scroll">{table_html}</div>
</body></html>"""

    with st.expander("🖨️ 列印 / 另存 PDF 用表格"):
        st.caption("下載成獨立網頁，用瀏覽器打開後 Ctrl+P / Cmd+P 列印或另存 PDF——不受這個頁面其他內容影響")
        st.download_button(
            "📥 下載可列印網頁 (.html)",
            data=standalone_html,
            file_name=f"{ticker}_計算紀錄.html",
            mime="text/html",
            key=f"download_print_{ticker}",
        )
        st.caption("下方是預覽（欄位較多時可左右滑動）：")
        st.markdown(
            f'<style>{_PRINT_TABLE_STYLE}</style><div class="print-scroll" style="overflow-x:auto;">{table_html}</div>',
            unsafe_allow_html=True,
        )

# --- Live price + per-ticker default computation ---
# Fixed fallbacks if the live fetch fails (offline, API blocked/rate-limited,
# etc.) -- the app should never crash or block on this, just fall back to a
# reasonable ballpark, same spirit as the pre-live-price defaults.
FALLBACK_PRICE = {"TSM": 418.0, "MU": 880.0, "NVDA": 224.0, "AMD": 483.0, "GOOG": 353.0}
# Strike = this % below current price. Also reused as the discount buttons'
# preset list below (5%/8%/10%/15% match TSM's/NVDA's/-/MU's own defaults).
STRIKE_DISCOUNT_PRESETS = [5.0, 8.0, 10.0, 15.0]
TICKER_STRIKE_DISCOUNT_PCT = {"TSM": 8.0, "MU": 15.0, "NVDA": 8.0, "AMD": 10.0, "GOOG": 5.0}
TICKER_IV_DEFAULT = {"TSM": 43.0, "MU": 76.0, "NVDA": 40.0, "AMD": 64.0, "GOOG": 30.0}
TICKER_PREMIUM_DEFAULT = {"TSM": 1.98, "MU": 4.5, "NVDA": 1.2, "AMD": 2.5, "GOOG": 1.5}

@st.cache_data(ttl=300)  # 5 minutes -- avoid re-fetching on every rerun/click
def fetch_live_price(symbol):
    try:
        price = yf.Ticker(symbol).fast_info["lastPrice"]
        if price and price > 0:
            return float(price)
    except Exception:
        pass
    return None

def get_ticker_defaults(ticker):
    """Starting values for `ticker`'s inputs: live current price (falls back
    to a fixed ballpark if the fetch fails), strike computed as a % below
    that price, this week's Friday for days-to-expiry, and fixed IV/rate/
    premium starting points. Recomputed on every render, but the live-price
    fetch itself is cached, so this is cheap."""
    S = fetch_live_price(ticker) or FALLBACK_PRICE[ticker]
    discount = TICKER_STRIKE_DISCOUNT_PCT[ticker]
    K = round(S * (1 - discount / 100), 1)
    # This week's Friday if it hasn't passed yet, otherwise next week's --
    # there's no sensible "days to expiry" default once this week's Friday
    # is already gone (Sat/Sun).
    days_this_fri = days_until_this_friday()
    days = float(days_this_fri) if days_this_fri is not None else float(days_until_next_friday())
    return {
        "S": round(S, 1), "K": K, "r": 3.8, "iv": TICKER_IV_DEFAULT[ticker],
        "days": days, "premium": TICKER_PREMIUM_DEFAULT[ticker],
    }

# --- Support/Resistance analysis (adapted from talala2019/tw-stock-levels) ---
# Institutional-flow signals from that project (法人成本防線, 法人大買/大賣K棒)
# are dropped here: they come from FinMind's Taiwan-only institutional
# investor data, which doesn't cover TSM/MU. Everything else (moving
# averages, volume-based levels, swing highs/lows, trendline projections,
# round-number levels) only needs price/volume history, so it works the
# same for any yfinance-supported ticker.
@st.cache_data(ttl=1800)  # 30 min -- levels don't need to be as fresh as live price
def fetch_price_history(symbol, days=380):
    try:
        end = datetime.date.today()
        start = end - datetime.timedelta(days=days)
        df = yf.download(symbol, start=start.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        return df
    except Exception:
        return None

def cluster_levels(candidates, cur_price, dynamic_threshold=0.015):
    """Merge candidate levels within `dynamic_threshold` (relative) of each
    other into clusters, each carrying its member count and the distinct
    signal types that landed there. `candidates` is a list of
    {"Price": float, "Type": str} dicts."""
    if not candidates or cur_price <= 0:
        return []
    ordered = sorted(candidates, key=lambda c: abs(c["Price"] - cur_price))
    clusters = []
    for cand in ordered:
        price = cand["Price"]
        merged = False
        for cl in clusters:
            if cl["mean_price"] > 0 and abs(price - cl["mean_price"]) / cl["mean_price"] <= dynamic_threshold:
                cl["prices"].append(price)
                cl["types"].append(cand["Type"])
                cl["mean_price"] = sum(cl["prices"]) / len(cl["prices"])
                merged = True
                break
        if not merged:
            clusters.append({"prices": [price], "mean_price": price, "types": [cand["Type"]]})
    result = []
    for cl in clusters:
        mean_p = cl["mean_price"]
        pct = (mean_p - cur_price) / cur_price * 100
        result.append({
            "Price": round(mean_p, 2), "Pct": round(pct, 1),
            "Count": len(cl["prices"]), "Types": list(dict.fromkeys(cl["types"])),
        })
    return result

def compute_support_resistance(df, cur_price):
    """Returns (resistances, supports), each a list of {"Price", "Type"}
    candidate levels (not yet clustered) above/below `cur_price`."""
    supports, resistances = [], []

    for ma in [5, 10, 20, 60, 120, 240]:
        df[f"MA{ma}"] = df["Close"].rolling(ma).mean()
        val = df[f"MA{ma}"].iloc[-1]
        if pd.notna(val):
            (supports if val < cur_price else resistances).append(
                {"Price": round(float(val), 2), "Type": f"MA{ma}"}
            )

    df["Vol_MA20"] = df["Volume"].rolling(20).mean()
    if len(df) >= 2:
        y = df.iloc[-2]
        if pd.notna(y["Vol_MA20"]) and y["Volume"] > y["Vol_MA20"]:
            if y["Low"] < cur_price and y["Close"] > y["Open"]:
                supports.append({"Price": round(float(y["Low"]), 2), "Type": "前日量增紅K低點"})
            if y["High"] > cur_price and y["Close"] < y["Open"]:
                resistances.append({"Price": round(float(y["High"]), 2), "Type": "前日量增黑K高點"})

    top_vol = df.tail(120).nlargest(3, "Volume")
    for _, row in top_vol.iterrows():
        if row["Close"] > cur_price:
            resistances.append({"Price": round(float(row["Close"]), 2), "Type": "大量套牢區"})
        else:
            supports.append({"Price": round(float(row["Close"]), 2), "Type": "大量換手支撐區"})

    n_swing = 10
    max_idx = argrelextrema(df["High"].values, np.greater_equal, order=n_swing)[0]
    min_idx = argrelextrema(df["Low"].values, np.less_equal, order=n_swing)[0]
    for i in max_idx[-3:]:
        resistances.append({"Price": round(float(df["High"].iloc[i]), 2), "Type": "波段壓力"})
    for i in min_idx[-3:]:
        supports.append({"Price": round(float(df["Low"].iloc[i]), 2), "Type": "波段支撐"})

    current_idx = len(df) - 1
    if len(max_idx) >= 2:
        i1, i2 = max_idx[-2], max_idx[-1]
        if i1 != i2 and i2 < current_idx:
            slope = (df["High"].iloc[i2] - df["High"].iloc[i1]) / (i2 - i1)
            proj = df["High"].iloc[i2] + slope * (current_idx - i2)
            if proj > cur_price:
                resistances.append({"Price": round(float(proj), 2), "Type": "近期高點連線"})
    if len(min_idx) >= 2:
        i1, i2 = min_idx[-2], min_idx[-1]
        if i1 != i2 and i2 < current_idx:
            slope = (df["Low"].iloc[i2] - df["Low"].iloc[i1]) / (i2 - i1)
            proj = df["Low"].iloc[i2] + slope * (current_idx - i2)
            if 0 < proj < cur_price:
                supports.append({"Price": round(float(proj), 2), "Type": "近期低點連線"})

    # Sweep every round-number level within +-30% of price (a bit wider than
    # the +-25% display filter downstream, so a level sitting right near
    # that edge isn't lost to rounding), not just the single nearest one
    # above/below -- otherwise a level like $1000 never becomes a candidate
    # in the first place when price is $880 and the nearest round number
    # ($900) is close enough to get filtered out by the +5% minimum anyway.
    step = 10 if cur_price < 100 else (50 if cur_price < 500 else 100)
    lo = int((cur_price * 0.70) // step) * step
    hi = int((cur_price * 1.30) // step) * step + step
    level = lo
    while level <= hi:
        if level > 0:
            (supports if level < cur_price else resistances).append(
                {"Price": float(level), "Type": "整數心理關卡"}
            )
        level += step

    return resistances, supports

def render_support_resistance(ticker, cur_price):
    """Compact +5%~+20% resistance / -5%~-20% support reference, embedded
    in the ticker's existing Pricing sub-tab rather than a separate tab."""
    df = fetch_price_history(ticker)
    if df is None or len(df) < 30:
        st.caption("⚠️ 支撐壓力參考：歷史資料抓取失敗或不足，暫不顯示")
        return

    resistances, supports = compute_support_resistance(df, cur_price)
    r_clusters = cluster_levels(resistances, cur_price)
    s_clusters = cluster_levels(supports, cur_price)
    r_filtered = sorted((c for c in r_clusters if 5 <= c["Pct"] <= 25), key=lambda c: c["Pct"])[:5]
    s_filtered = sorted((c for c in s_clusters if -25 <= c["Pct"] <= -5), key=lambda c: -c["Pct"])[:5]

    # A major historical high/low further than +-20% away doesn't just
    # vanish -- it's still real resistance/support, just outside this
    # window's intended "near-term" range. Surface it separately so it
    # doesn't look like it was never considered.
    major_high_idx = df["High"].idxmax()
    major_high_price = float(df["High"].loc[major_high_idx])
    major_high_date = df["Date"].loc[major_high_idx]
    major_high_pct = (major_high_price - cur_price) / cur_price * 100 if cur_price else 0.0

    major_low_idx = df["Low"].idxmin()
    major_low_price = float(df["Low"].loc[major_low_idx])
    major_low_date = df["Date"].loc[major_low_idx]
    major_low_pct = (major_low_price - cur_price) / cur_price * 100 if cur_price else 0.0

    with st.expander(f"📊 {ticker} 支撐壓力參考（壓力 +5%~+25% ／支撐 -5%~-25%）"):
        st.caption("均線、量能、波段高低點等技術訊號綜合判斷，僅供參考，非投資建議")
        rc, sc = st.columns(2)
        with rc:
            st.markdown("**🔴 壓力區**")
            if r_filtered:
                for lv in r_filtered:
                    st.markdown(f"- ${lv['Price']:.2f}（+{lv['Pct']:.1f}%）· {'/'.join(lv['Types'])}")
            else:
                st.caption("此區間無明顯壓力訊號")
        with sc:
            st.markdown("**🟢 支撐區**")
            if s_filtered:
                for lv in s_filtered:
                    st.markdown(f"- ${lv['Price']:.2f}（{lv['Pct']:.1f}%）· {'/'.join(lv['Types'])}")
            else:
                st.caption("此區間無明顯支撐訊號")

        if major_high_pct > 25:
            st.caption(
                f"⚠️ 範圍外重大高點：${major_high_price:.2f}（+{major_high_pct:.1f}%，"
                f"{major_high_date.strftime('%Y-%m-%d')}）"
            )
        if major_low_pct < -25:
            st.caption(
                f"⚠️ 範圍外重大低點：${major_low_price:.2f}（{major_low_pct:.1f}%，"
                f"{major_low_date.strftime('%Y-%m-%d')}）"
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

    d = get_ticker_defaults(ticker)

    if f"history_{ticker}" not in st.session_state:
        st.session_state[f"history_{ticker}"] = []

    # Loading a saved history row has the same timing constraint as the
    # Friday buttons below: it can only write s_p/k_p/etc. because this runs
    # before those widgets are (re-)created further down.
    pending_load = st.session_state.pop(f"_load_row_{ticker}", None)
    if pending_load is not None:
        hist_for_load = st.session_state[f"history_{ticker}"]
        # Bounds-check: a delete elsewhere can shift/shrink the list between
        # when this index was captured (at click time) and when it's
        # applied (next run) -- an out-of-range index is stale, not a bug
        # to crash on, so just skip it.
        if 0 <= pending_load < len(hist_for_load):
            row = hist_for_load[pending_load]
            st.session_state[k("s_p")] = row["S"]
            st.session_state[k("k_p")] = row["K"]
            st.session_state[k("r_p")] = row["r"]
            st.session_state[k("iv_p")] = row["IV"]
            st.session_state[k("d_p")] = row["days"]

    # The dataframe below remembers its last row selection across reruns by
    # key. After a load or delete we want that selection cleared -- an old
    # position can point at the wrong row (or nothing) once the list has
    # changed -- but the widget can't be touched after it's rendered this
    # run, so (like the pending-load flag above) this is applied here, one
    # run before the dataframe widget itself is created below.
    if st.session_state.pop(f"_clear_selection_{ticker}", False):
        st.session_state.pop(k("history_table"), None)

    if st.session_state.pop(f"_apply_this_friday_p_{ticker}", False):
        days_this_fri = days_until_this_friday()
        if days_this_fri is not None:  # button is hidden once it's None, but be defensive
            st.session_state[k("d_p")] = float(days_this_fri)
    if st.session_state.pop(f"_apply_next_friday_p_{ticker}", False):
        st.session_state[k("d_p")] = float(days_until_next_friday())

    # A strike-quick-select button (below) can't write k_p after that widget
    # is already instantiated this run either -- same pending-flag pattern.
    # The stored value is a SIGNED percent: positive = strike above price,
    # negative = strike below price -- so applying it is just one formula.
    pending_pct = st.session_state.pop(f"_apply_strike_pct_p_{ticker}", None)
    if pending_pct is not None:
        current_S = st.session_state.get(k("s_p"), d["S"])
        st.session_state[k("k_p")] = round(current_S * (1 + pending_pct / 100), 1)

    col1, col2, col3 = st.columns(3)
    with col1:
        S = st.number_input("Current Price (標的價)", value=d["S"], step=1.0, key=k("s_p"))
        r1 = st.number_input("Risk-free Rate % (利率)", value=d["r"], step=0.1, key=k("r_p"))
    with col2:
        K = st.number_input("Strike Price (履約價)", value=d["K"], step=1.0, key=k("k_p"))
        st.caption(strike_pct_of_price_caption(K, S))
        iv1 = st.number_input("Implied Volatility % (IV 隱含波動率)", value=d["iv"], step=1.0, key=k("iv_p"))
    with col3:
        days1 = st.number_input("Days to Expiry (天數)", value=d["days"], step=1.0, key=k("d_p"))
        if days_until_this_friday() is not None:  # hidden once this week's Friday has passed
            if st.button(f"📅 算至本週五 ({this_friday_label()})", key=k("this_friday_p")):
                st.session_state[f"_apply_this_friday_p_{ticker}"] = True
                st.rerun()
        if st.button(f"📅 算至下週五 ({next_friday_label()})", key=k("next_friday_p")):
            st.session_state[f"_apply_next_friday_p_{ticker}"] = True
            st.rerun()

    # Full-width row (not squeezed into col2) so 8 buttons have room --
    # ordered low-to-high price, like reading the strikes off a chain.
    st.caption("履約價快速選取（標的價 × ％）")
    # +15% leftmost (red, Call side) down to -15% rightmost (green, Put side).
    signed_pcts = list(reversed(STRIKE_DISCOUNT_PRESETS)) + [-p for p in STRIKE_DISCOUNT_PRESETS]
    pct_cols = st.columns(len(signed_pcts))
    for i, pct in enumerate(signed_pcts):
        label = f"{pct:+.0f}%"
        safe_key = f"discount_p_{'m' if pct < 0 else 'p'}{abs(pct):.0f}"
        if pct_cols[i].button(
            label, key=k(safe_key), use_container_width=True,
            help=f"履約價 = 標的價 × {100 + pct:.0f}%",
        ):
            st.session_state[f"_apply_strike_pct_p_{ticker}"] = pct
            st.rerun()

    render_support_resistance(ticker, S)

    if st.button("Calculate Premium", type="primary", key=k("btn_p")):
        call, put = calculate_black_scholes(S, K, days1, r1, iv1)
        call_delta = calculate_delta(S, K, days1, r1, iv1, 'call')
        put_delta = calculate_delta(S, K, days1, r1, iv1, 'put')
        st.session_state[f"history_{ticker}"].append({
            "S": S, "K": K, "Discount": round(K / S * 100, 1) if S else 0.0,
            "r": r1, "IV": iv1, "days": days1,
            "Call": round(call, 4), "CallΔ": round(call_delta, 4),
            "Put": round(put, 4), "PutΔ": round(put_delta, 4),
            "Note": "", "Status": "一般",
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
        # Streamlit's dataframe has its own column-hide menu, but that's a
        # frontend-only toggle -- it doesn't tell this code anything, so the
        # print/PDF export (a completely separate render) can't honor it.
        # These checkboxes are a real, server-side choice instead: hiding a
        # side here actually drops those columns before either table is
        # built, so the interactive view and the export always agree, and a
        # single-side trade (e.g. a Put-only entry) doesn't need 4 unused
        # Call columns cluttering (or wrapping) either one.
        cc, pc = st.columns(2)
        show_call = cc.checkbox("顯示 Call 欄位", value=True, key=k("show_call_cols"))
        show_put = pc.checkbox("顯示 Put 欄位", value=True, key=k("show_put_cols"))

        hist_df = pd.DataFrame(history).rename(columns={
            "S": "股價", "K": "履約價", "Discount": "折數", "r": "利率%", "IV": "IV%", "days": "天數",
            "Call": "Call 價", "Put": "Put 價", "CallΔ": "Call Δ", "PutΔ": "Put Δ",
            "Note": "備註", "Status": "標記",
        })
        if not show_call:
            hist_df = hist_df.drop(columns=["Call 價", "Call Δ"])
        if not show_put:
            hist_df = hist_df.drop(columns=["Put 價", "Put Δ"])

        # Row tinted by 標記 (status), Call/Put price columns keep their own
        # fixed red/green regardless -- see style_history_table's docstring.
        styled_hist = style_history_table(hist_df)
        # height="auto" (the default) is supposed to size to the row count,
        # but its own calculation runs a bit short in practice and clips
        # the last row's bottom edge. Size it explicitly instead: header +
        # one row per entry, capped at 12 rows tall before it scrolls.
        visible_rows = min(len(hist_df), 12)
        table_height = 38 + 35 * visible_rows + 6
        event = st.dataframe(
            styled_hist, on_select="rerun", selection_mode="single-row",
            key=k("history_table"), hide_index=True, height=table_height,
            column_config={
                "股價": st.column_config.NumberColumn(format="%.1f", alignment="center"),
                "履約價": st.column_config.NumberColumn(format="%.1f", alignment="center"),
                "折數": st.column_config.NumberColumn(format="%.1f%%", alignment="center", help="履約價 ÷ 股價 × 100%"),
                "利率%": st.column_config.NumberColumn(format="%.1f", alignment="center"),
                "IV%": st.column_config.NumberColumn(format="%.1f", alignment="center"),
                "天數": st.column_config.NumberColumn(format="%.0f", alignment="center"),
                "Call 價": st.column_config.NumberColumn(format="$%.2f", alignment="center"),
                "Put 價": st.column_config.NumberColumn(format="$%.2f", alignment="center"),
                "Call Δ": st.column_config.NumberColumn(format="%.3f", alignment="center"),
                "Put Δ": st.column_config.NumberColumn(format="%.3f", alignment="center"),
                "備註": st.column_config.TextColumn(width="medium"),
                "標記": st.column_config.TextColumn(width="small"),
            },
        )
        render_printable_history(hist_df, ticker)

        selected_rows = event["selection"]["rows"]
        if selected_rows:
            idx = selected_rows[0]
            lc, dc, ec = st.columns(3)
            if lc.button("📥 載入此列", key=k("load_row"), use_container_width=True):
                st.session_state[f"_load_row_{ticker}"] = idx
                st.session_state[f"_clear_selection_{ticker}"] = True
                st.rerun()
            if dc.button("🗑️ 刪除此列", key=k("delete_row"), use_container_width=True):
                st.session_state[f"history_{ticker}"].pop(idx)
                st.session_state[f"_clear_selection_{ticker}"] = True
                st.rerun()
            if ec.button("✏️ 編輯備註", key=k("edit_row"), use_container_width=True):
                st.session_state[f"_editing_note_{ticker}"] = idx
                st.rerun()

        editing_idx = st.session_state.get(f"_editing_note_{ticker}")
        if editing_idx is not None and 0 <= editing_idx < len(history):
            row_data = history[editing_idx]
            with st.form(key=k("edit_note_form")):
                st.caption(f"編輯第 {editing_idx + 1} 列備註（{ticker}）")
                note_val = st.text_input("備註", value=row_data.get("Note", ""), key=k("edit_note_input"))
                status_val = st.selectbox(
                    "標記", HISTORY_STATUS_OPTIONS,
                    index=HISTORY_STATUS_OPTIONS.index(row_data.get("Status", "一般")),
                    key=k("edit_status_input"),
                )
                save_col, cancel_col = st.columns(2)
                saved = save_col.form_submit_button("💾 儲存", use_container_width=True, key=k("edit_note_save"))
                cancelled = cancel_col.form_submit_button("取消", use_container_width=True, key=k("edit_note_cancel"))
            if saved:
                history[editing_idx]["Note"] = note_val
                history[editing_idx]["Status"] = status_val
                st.session_state.pop(f"_editing_note_{ticker}", None)
                st.rerun()
            if cancelled:
                st.session_state.pop(f"_editing_note_{ticker}", None)
                st.rerun()

# --- UI Section: one ticker's Implied Volatility panel (Tab 2) ---
def render_iv_section(ticker):
    def k(name):
        return f"{name}_{ticker}"

    d = get_ticker_defaults(ticker)

    if st.session_state.pop(f"_apply_this_friday_iv_{ticker}", False):
        days_this_fri_iv = days_until_this_friday()
        if days_this_fri_iv is not None:
            st.session_state[k("d_iv")] = float(days_this_fri_iv)
    if st.session_state.pop(f"_apply_next_friday_iv_{ticker}", False):
        st.session_state[k("d_iv")] = float(days_until_next_friday())

    pending_pct_iv = st.session_state.pop(f"_apply_strike_pct_iv_{ticker}", None)
    if pending_pct_iv is not None:
        current_S2 = st.session_state.get(k("s_iv"), d["S"])
        st.session_state[k("k_iv")] = round(current_S2 * (1 + pending_pct_iv / 100), 1)

    col1, col2, col3 = st.columns(3)
    with col1:
        S2 = st.number_input("Current Price (標的價)", value=d["S"], step=1.0, key=k("s_iv"))
        days2 = st.number_input("Days to Expiry (天數)", value=d["days"], step=1.0, key=k("d_iv"))
        if days_until_this_friday() is not None:
            if st.button(f"📅 算至本週五 ({this_friday_label()})", key=k("this_friday_iv")):
                st.session_state[f"_apply_this_friday_iv_{ticker}"] = True
                st.rerun()
        if st.button(f"📅 算至下週五 ({next_friday_label()})", key=k("next_friday_iv")):
            st.session_state[f"_apply_next_friday_iv_{ticker}"] = True
            st.rerun()
    with col2:
        K2 = st.number_input("Strike Price (履約價)", value=d["K"], step=1.0, key=k("k_iv"))
        st.caption(strike_pct_of_price_caption(K2, S2))
        r2 = st.number_input("Risk-free Rate % (利率)", value=d["r"], step=0.1, key=k("r_iv"))
    with col3:
        target_price = st.number_input("Market Premium (Bid/Ask MID 權利金)", value=d["premium"], step=0.1, key=k("target_iv"))
        opt_type = st.selectbox("Option Type (類型)", ["put", "call"], key=k("type_iv"))

    st.caption("履約價快速選取（標的價 × ％）")
    signed_pcts_iv = list(reversed(STRIKE_DISCOUNT_PRESETS)) + [-p for p in STRIKE_DISCOUNT_PRESETS]
    pct_cols_iv = st.columns(len(signed_pcts_iv))
    for i, pct in enumerate(signed_pcts_iv):
        label = f"{pct:+.0f}%"
        safe_key = f"discount_iv_{'m' if pct < 0 else 'p'}{abs(pct):.0f}"
        if pct_cols_iv[i].button(
            label, key=k(safe_key), use_container_width=True,
            help=f"履約價 = 標的價 × {100 + pct:.0f}%",
        ):
            st.session_state[f"_apply_strike_pct_iv_{ticker}"] = pct
            st.rerun()

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
[class*="st-key-this_friday"] button { color: #1976d2 !important; white-space: nowrap; }
[class*="st-key-next_friday"] button { color: #f57c00 !important; white-space: nowrap; }
[class*="st-key-discount_p_"] button,
[class*="st-key-discount_iv_"] button { white-space: nowrap; padding-left: 0.3rem; padding-right: 0.3rem; }
/* Positive %% (strike above price, Call side) red; negative %% (Put side)
   green -- matches the buy/sell-side convention used everywhere else.
   "discount_p_p"/"discount_iv_p" only matches the '+' buttons because the
   '-' ones are "discount_p_m"/"discount_iv_m" -- no overlap with the
   "discount_p_"/"discount_iv_" tab-scope prefix itself. */
[class*="st-key-discount_p_p"] button,
[class*="st-key-discount_iv_p"] button { color: #d32f2f !important; }
[class*="st-key-discount_p_m"] button,
[class*="st-key-discount_iv_m"] button { color: #2e7d32 !important; }
.st-key-ticker_subtabs_p [data-baseweb="tab"] p,
.st-key-ticker_subtabs_iv [data-baseweb="tab"] p { font-size: 1.15rem !important; font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)

TICKERS = ["TSM", "MU", "NVDA", "AMD", "GOOG"]

# Create Tabs for the two main functions
tab1, tab2 = st.tabs(["💰 Pricing (權利金計算)", "📊 Implied Volatility (IV 隱含波動率 反推)"])

# --- TAB 1: PRICING ---
with tab1:
    st.subheader("Input Market Parameters")
    ticker_subtabs_1 = st.tabs(TICKERS, key="ticker_subtabs_p")
    for ticker, subtab in zip(TICKERS, ticker_subtabs_1):
        with subtab:
            render_pricing_section(ticker)

# --- TAB 2: IV CALCULATION ---
with tab2:
    st.subheader("Reverse IV from Market Premium")
    ticker_subtabs_2 = st.tabs(TICKERS, key="ticker_subtabs_iv")
    for ticker, subtab in zip(TICKERS, ticker_subtabs_2):
        with subtab:
            render_iv_section(ticker)
