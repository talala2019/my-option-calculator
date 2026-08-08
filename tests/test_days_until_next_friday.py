import sys
import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import days_until_next_friday, days_until_this_friday


def test_monday_is_four_days_out():
    assert days_until_next_friday(datetime.date(2026, 8, 3)) == 4


def test_tuesday_is_three_days_out():
    assert days_until_next_friday(datetime.date(2026, 8, 4)) == 3


def test_wednesday_is_two_days_out():
    assert days_until_next_friday(datetime.date(2026, 8, 5)) == 2


def test_thursday_is_one_day_out():
    assert days_until_next_friday(datetime.date(2026, 8, 6)) == 1


def test_friday_skips_to_next_week():
    # On Friday itself, "next Friday" means next week's, not today (0).
    assert days_until_next_friday(datetime.date(2026, 8, 7)) == 7


def test_saturday_is_six_days_out():
    assert days_until_next_friday(datetime.date(2026, 8, 1)) == 6


def test_sunday_is_five_days_out():
    assert days_until_next_friday(datetime.date(2026, 8, 2)) == 5


def test_this_friday_monday_is_four_days_out():
    assert days_until_this_friday(datetime.date(2026, 8, 3)) == 4


def test_this_friday_on_friday_itself_is_zero():
    assert days_until_this_friday(datetime.date(2026, 8, 7)) == 0


def test_this_friday_on_saturday_is_none():
    # This week's Friday already passed. Returning 0 here would silently
    # mean "today" (a Saturday) and mislabel it as Friday -- so this signals
    # "no valid this-Friday" instead of a wrong non-negative number.
    assert days_until_this_friday(datetime.date(2026, 8, 8)) is None


def test_this_friday_on_sunday_is_none():
    assert days_until_this_friday(datetime.date(2026, 8, 9)) is None
