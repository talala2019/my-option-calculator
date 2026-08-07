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
