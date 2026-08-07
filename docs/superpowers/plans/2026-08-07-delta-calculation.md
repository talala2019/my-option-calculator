# Delta Calculation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Black-Scholes Delta calculation to the options app, displayed on both the Pricing tab and the Implied Volatility tab.

**Architecture:** Add one new pure function `calculate_delta()` to `app.py`, next to the existing `calculate_black_scholes()` / `calculate_iv()` functions. Wire its output into the existing `st.metric` blocks in Tab 1 and Tab 2. No changes to existing pricing/IV logic.

**Tech Stack:** Python, Streamlit 1.56, SciPy (`scipy.stats.norm`), pytest (dev-only, not added to `requirements.txt`).

## Global Constraints

- Delta only — no Gamma/Theta/Vega (per approved spec, deferred to a future feature).
- `calculate_black_scholes` and `calculate_iv` must not change behavior.
- At expiry (`days <= 0`), Delta is binary: call = 1.0 if `S > K` else 0.0; put = -1.0 if `S < K` else 0.0.
- Tab 2 only shows Delta when `calculate_iv` succeeds (`iv_result is not None and iv_result > 0`) — matches existing success condition, no new error paths.
- Reference spec: `docs/superpowers/specs/2026-08-07-delta-calculation-design.md`

---

### Task 1: `calculate_delta()` function with unit tests

**Files:**
- Modify: `app.py:44-46` (insert new function between `calculate_iv` and the `# --- Streamlit UI Design ---` comment)
- Create: `tests/test_delta.py`

**Interfaces:**
- Produces: `calculate_delta(S: float, K: float, days: float, r_pct: float, iv_pct: float, option_type: str = 'call') -> float`, importable as `from app import calculate_delta`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_delta.py`:

```python
import sys
from pathlib import Path

# app.py lives at the project root; tests/ is not a package, so add the
# root explicitly rather than relying on pytest's rootdir insertion.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import calculate_delta


def test_call_delta_is_between_zero_and_one():
    delta = calculate_delta(420.0, 390.0, 8, 4.0, 52.0, 'call')
    assert 0.0 <= delta <= 1.0


def test_put_delta_is_between_minus_one_and_zero():
    delta = calculate_delta(420.0, 390.0, 8, 4.0, 52.0, 'put')
    assert -1.0 <= delta <= 0.0


def test_put_call_delta_parity():
    # Call Delta - Put Delta == 1 for the same S, K, T, r, sigma.
    call_delta = calculate_delta(420.0, 390.0, 8, 4.0, 52.0, 'call')
    put_delta = calculate_delta(420.0, 390.0, 8, 4.0, 52.0, 'put')
    assert abs((call_delta - put_delta) - 1.0) < 1e-9


def test_expiry_call_in_the_money_delta_is_one():
    delta = calculate_delta(420.0, 390.0, 0, 4.0, 52.0, 'call')
    assert delta == 1.0


def test_expiry_call_out_of_the_money_delta_is_zero():
    delta = calculate_delta(380.0, 390.0, 0, 4.0, 52.0, 'call')
    assert delta == 0.0


def test_expiry_put_in_the_money_delta_is_minus_one():
    delta = calculate_delta(380.0, 390.0, 0, 4.0, 52.0, 'put')
    assert delta == -1.0


def test_expiry_put_out_of_the_money_delta_is_zero():
    delta = calculate_delta(420.0, 390.0, 0, 4.0, 52.0, 'put')
    assert delta == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_delta.py -v` (from the project root: `C:\Users\tully\Python\Test_Env\Option\Streamlit Community Cloud`)

Expected: `ImportError: cannot import name 'calculate_delta' from 'app'` (all 7 tests error out).

- [ ] **Step 3: Implement `calculate_delta()`**

In `app.py`, insert this new function directly after `calculate_iv()` (currently ending at line 44) and before the `# --- Streamlit UI Design ---` comment (currently line 46):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_delta.py -v`

Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_delta.py
git commit -m "Add calculate_delta function with unit tests"
```

---

### Task 2: Wire Delta into Tab 1 (Pricing)

**Files:**
- Modify: `app.py` (the `with tab1:` button block — currently lines 70-76, shifted by however many lines Task 1 added above it)

**Interfaces:**
- Consumes: `calculate_delta(S, K, days, r_pct, iv_pct, option_type)` from Task 1.

- [ ] **Step 1: Update the Tab 1 button block**

Find this block (inside `with tab1:`):

```python
    if st.button("Calculate Premium", type="primary", key="btn_p"):
        call, put = calculate_black_scholes(S, K, days1, r1, iv1)
        st.divider()
        st.success("Calculation Complete!")
        c1, c2 = st.columns(2)
        c1.metric(label="📈 Call Price (買權)", value=f"${call:.4f}")
        c2.metric(label="📉 Put Price (賣權)", value=f"${put:.4f}")
```

Replace it with:

```python
    if st.button("Calculate Premium", type="primary", key="btn_p"):
        call, put = calculate_black_scholes(S, K, days1, r1, iv1)
        call_delta = calculate_delta(S, K, days1, r1, iv1, 'call')
        put_delta = calculate_delta(S, K, days1, r1, iv1, 'put')
        st.divider()
        st.success("Calculation Complete!")
        c1, c2 = st.columns(2)
        c1.metric(label="📈 Call Price (買權)", value=f"${call:.4f}")
        c2.metric(label="📉 Put Price (賣權)", value=f"${put:.4f}")
        d1, d2 = st.columns(2)
        d1.metric(label="Call Δ (Delta)", value=f"{call_delta:.4f}")
        d2.metric(label="Put Δ (Delta)", value=f"{put_delta:.4f}")
```

- [ ] **Step 2: Manually verify in the running app**

Run: `streamlit run app.py` (from the project root)

In the browser tab that opens:
1. Go to the "💰 Pricing" tab, keep default inputs, click "Calculate Premium".
2. Confirm a new row appears below the price metrics showing "Call Δ (Delta)" and "Put Δ (Delta)".
3. Confirm Call Δ is between 0 and 1, Put Δ is between -1 and 0, and `Call Δ - Put Δ ≈ 1.0000`.
4. Set "Days to Expiry" to `0` and recalculate — confirm Call Δ / Put Δ show `1.0000`/`0.0000` or `0.0000`/`-1.0000` depending on whether Current Price is above or below Strike Price.
5. Stop the app (Ctrl+C in the terminal running `streamlit run`).

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "Show Call/Put Delta on Pricing tab"
```

---

### Task 3: Wire Delta into Tab 2 (Implied Volatility)

**Files:**
- Modify: `app.py` (the `with tab2:` button block — currently lines 93-100, shifted by Tasks 1-2 above it)

**Interfaces:**
- Consumes: `calculate_delta(S, K, days, r_pct, iv_pct, option_type)` from Task 1.

- [ ] **Step 1: Update the Tab 2 button block**

Find this block (inside `with tab2:`):

```python
    if st.button("Calculate Implied Volatility", type="primary", key="btn_iv"):
        iv_result = calculate_iv(S2, K2, target_price, days2, r2, opt_type)
        st.divider()
        if iv_result is not None and iv_result > 0:
            st.success("Calculation Complete!")
            st.metric(label=f"📊 Implied Volatility (IV)", value=f"{iv_result:.2f} %")
        else:
            st.error("Could not converge. Please check if the premium is lower than the intrinsic value.")
```

Replace it with:

```python
    if st.button("Calculate Implied Volatility", type="primary", key="btn_iv"):
        iv_result = calculate_iv(S2, K2, target_price, days2, r2, opt_type)
        st.divider()
        if iv_result is not None and iv_result > 0:
            st.success("Calculation Complete!")
            delta_result = calculate_delta(S2, K2, days2, r2, iv_result, opt_type)
            e1, e2 = st.columns(2)
            e1.metric(label=f"📊 Implied Volatility (IV)", value=f"{iv_result:.2f} %")
            e2.metric(label="Δ (Delta)", value=f"{delta_result:.4f}")
        else:
            st.error("Could not converge. Please check if the premium is lower than the intrinsic value.")
```

- [ ] **Step 2: Manually verify in the running app**

Run: `streamlit run app.py` (from the project root)

In the browser tab that opens:
1. Go to the "📊 Implied Volatility" tab, keep default inputs (Option Type = "put"), click "Calculate Implied Volatility".
2. Confirm the IV metric now has a "Δ (Delta)" metric next to it, and its value is between -1 and 0 (since default type is "put").
3. Switch "Option Type (類型)" to "call", recalculate, confirm Δ is between 0 and 1.
4. Set "Market Premium" to an unreasonably high value (e.g. `500`) so Newton-Raphson fails to converge, recalculate, and confirm the existing error message still shows and no Δ metric appears.
5. Stop the app (Ctrl+C in the terminal running `streamlit run`).

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "Show Delta on Implied Volatility tab"
```

---

## After All Tasks

Do **not** push to GitHub automatically. After Task 3 is verified, stop and ask the user for explicit confirmation before running `git push origin main` — pushing triggers a live redeploy on Streamlit Community Cloud.

Note: `app.py` currently has an unrelated uncommitted change (updated default input values: Current Price 418, IV 45%, Market Premium 1.98) from the user's own editing, predating this plan. Leave those changes as part of the working tree — they will ride along in Task 1's commit unless the user asks to handle them separately. Flag this to the user rather than silently dropping or splitting it out.
