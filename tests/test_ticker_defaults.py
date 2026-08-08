import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import get_ticker_defaults, days_until_this_friday, days_until_next_friday


# These call the live get_ticker_defaults(), which tries a real network
# fetch (yfinance) and falls back to a fixed price if it fails. The
# assertions below hold either way -- K is always derived from whatever S
# ends up being, live or fallback -- so the test doesn't need network
# access to be meaningful, and isn't flaky if the fetch is unavailable.

def _expected_default_days():
    # Mirrors get_ticker_defaults' own fallback: this week's Friday if it
    # hasn't passed, otherwise next week's -- whichever day the test runs on.
    days_this_fri = days_until_this_friday()
    return float(days_this_fri) if days_this_fri is not None else float(days_until_next_friday())


def test_tsm_defaults_structure():
    d = get_ticker_defaults("TSM")
    assert d["iv"] == 43.0
    assert d["r"] == 3.8
    assert d["days"] == _expected_default_days()
    assert d["S"] > 0
    assert abs(d["K"] / d["S"] - 0.92) < 0.01  # strike = 92% of price (8% discount)


def test_mu_defaults_structure():
    d = get_ticker_defaults("MU")
    assert d["iv"] == 76.0
    assert d["r"] == 3.8
    assert d["days"] == _expected_default_days()
    assert d["S"] > 0
    assert abs(d["K"] / d["S"] - 0.85) < 0.01  # strike = 85% of price (15% discount)


def test_fetch_live_price_bad_symbol_returns_none():
    from app import fetch_live_price
    # An invalid ticker symbol should fail gracefully, not raise.
    result = fetch_live_price("THIS_IS_NOT_A_REAL_TICKER_XYZ")
    assert result is None
