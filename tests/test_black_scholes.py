import pytest
from src.black_scholes import (
    bs_call,
    bs_put,
    bs_vega,
    implied_vol_newton,
    implied_vol_brent,
)


def test_bs_call_known_value():
    """ATM call, S=K=100, T=1, r=0.05, sigma=0.2 -> price ~ 10.45"""
    price = bs_call(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
    assert price == pytest.approx(10.4506, abs=1e-3)


def test_put_call_parity():
    """C - P = S - K * e^{-rT} must hold for any valid inputs."""
    import numpy as np

    S, K, T, r, sigma = 100, 95, 0.5, 0.03, 0.25
    call = bs_call(S, K, T, r, sigma)
    put = bs_put(S, K, T, r, sigma)

    lhs = call - put
    rhs = S - K * np.exp(-r * T)

    assert lhs == pytest.approx(rhs, abs=1e-8)


def test_implied_vol_roundtrip():
    """Generate a price from a known sigma, recover sigma to 6dp."""
    S, K, T, r = 100, 105, 0.75, 0.02
    true_sigma = 0.3

    price = bs_call(S, K, T, r, true_sigma)
    recovered = implied_vol_newton(
        market_price=price, S=S, K=K, T=T, r=r,
        option_type="call", initial_guess=0.2,
    )

    assert recovered is not None
    assert recovered == pytest.approx(true_sigma, abs=1e-6)


def test_deep_otm_returns_none():
    """
    Deep OTM call with tiny T -> vega collapses to ~0 -> Newton
    should bail out and return None.
    """
    result = implied_vol_newton(
        market_price=0.0001,
        S=100, K=300, T=0.01, r=0.05,
        option_type="call", initial_guess=0.2,
    )
    assert result is None


def test_negative_price_raises():
    """Non-positive S, K, T, or sigma should raise ValueError."""
    with pytest.raises(ValueError):
        bs_call(S=-100, K=100, T=1.0, r=0.05, sigma=0.2)

    with pytest.raises(ValueError):
        bs_call(S=100, K=100, T=1.0, r=0.05, sigma=-0.2)

    with pytest.raises(ValueError):
        bs_call(S=100, K=100, T=-1.0, r=0.05, sigma=0.2)


def test_implied_vol_roundtrip_put():
    """Same roundtrip check as calls, but for a put."""
    S, K, T, r = 100, 95, 0.75, 0.02
    true_sigma = 0.3

    price = bs_put(S, K, T, r, true_sigma)
    recovered = implied_vol_newton(
        market_price=price, S=S, K=K, T=T, r=r,
        option_type="put", initial_guess=0.2,
    )

    assert recovered is not None
    assert recovered == pytest.approx(true_sigma, abs=1e-6)


def test_implied_vol_brent_roundtrip():
    """Brent should also recover a known sigma from a generated price."""
    S, K, T, r = 100, 100, 1.0, 0.05
    true_sigma = 0.25

    price = bs_call(S, K, T, r, true_sigma)
    recovered = implied_vol_brent(
        market_price=price, S=S, K=K, T=T, r=r, option_type="call",
    )

    assert recovered is not None
    assert recovered == pytest.approx(true_sigma, abs=1e-6)