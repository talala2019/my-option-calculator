# Delta Calculation — Design

## Purpose
Add option Delta (∂price/∂S) to the existing Black-Scholes pricing app so users can see
directional exposure alongside the price/IV outputs they already compute.

## Scope
- Delta only (no Gamma/Theta/Vega in this pass — can be added later as separate features).
- Add to both existing tabs:
  - **Tab 1 (Pricing)**: show Call Delta and Put Delta alongside Call/Put Price.
  - **Tab 2 (Implied Volatility)**: after solving IV, show the Delta of the selected
    option type (`opt_type`) at that solved IV.

## Design

### New function
```python
def calculate_delta(S, K, days, r_pct, iv_pct, option_type='call'):
    T = days / 365.0
    r = r_pct / 100.0
    sigma = iv_pct / 100.0

    if T <= 0:
        # At expiry, Delta is binary: 1/-1 if in-the-money, 0 otherwise.
        if option_type == 'call':
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1
```
This mirrors the existing d1 formula in `calculate_black_scholes` — no shared state,
just the same math, kept as a small standalone function (consistent with the existing
`calculate_black_scholes` / `calculate_iv` style — no shared refactor needed for two
formulas).

### Tab 1 (Pricing) — UI change
After the existing `st.button("Calculate Premium")` block computes `call, put`, also
compute:
```python
call_delta = calculate_delta(S, K, days1, r1, iv1, 'call')
put_delta = calculate_delta(S, K, days1, r1, iv1, 'put')
```
Display in a second row of two `st.metric` columns below the existing Call/Put Price
row: "Call Δ" and "Put Δ", formatted to 4 decimal places (e.g. `0.5234`).

### Tab 2 (IV) — UI change
After `iv_result = calculate_iv(...)` succeeds (`iv_result is not None and iv_result > 0`,
matching the existing success condition), also compute:
```python
delta_result = calculate_delta(S2, K2, days2, r2, iv_result, opt_type)
```
Display alongside the existing IV `st.metric`, as a second column: "Delta", formatted
to 4 decimal places. If IV calculation fails (`iv_result is None`), Delta is not shown —
existing error path (`st.error(...)`) is unchanged.

## Out of scope
- Other Greeks (Gamma, Theta, Vega) — explicitly deferred per user decision.
- No changes to `calculate_black_scholes` or `calculate_iv` logic.
- No changes to input fields/layout beyond adding the new metric row(s).

## Testing
Manual verification in the running Streamlit app:
1. Tab 1: default values → Call Δ should be in [0,1], Put Δ in [-1,0], and
   `call_delta - put_delta ≈ 1` (put-call delta parity).
2. Tab 1: set Days to Expiry = 0 → Delta should show binary 1/0 or -1/0 per moneyness.
3. Tab 2: default values → Delta shown matches sign/range expected for `opt_type`.
4. Tab 2: force non-convergence (e.g. absurd target_price) → confirm no Delta shown,
   existing error message still appears.
