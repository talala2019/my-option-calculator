import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import strike_pct_of_price_caption


def test_strike_below_price():
    assert strike_pct_of_price_caption(92.0, 100.0) == "= 92.0% of 標的價"


def test_strike_above_price():
    assert strike_pct_of_price_caption(115.0, 100.0) == "= 115.0% of 標的價"


def test_strike_equals_price():
    assert strike_pct_of_price_caption(100.0, 100.0) == "= 100.0% of 標的價"


def test_zero_price_does_not_divide_by_zero():
    assert strike_pct_of_price_caption(100.0, 0.0) == "= — % of 標的價"
