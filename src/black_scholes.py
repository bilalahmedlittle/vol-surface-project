import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def _validate_inputs(S: float, K: float, T: float, r: float, sigma: float) -> None:
    """Raise ValueError if any Black-Scholes input is non-physical."""
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive")
    if T <= 0:
        raise ValueError("T must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")


def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate d1 component of the Black-Scholes formula.
    ...
    """
    _validate_inputs(S, K, T, r, sigma)
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate d2 component of the Black-Scholes formula.
    ...
    """
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Price a European call option under Black-Scholes.

    C = S * N(d1) - K * e^{-rT} * N(d2)
    """
    _d1 = d1(S, K, T, r, sigma)
    _d2 = d2(S, K, T, r, sigma)
    return S * norm.cdf(_d1) - K * np.exp(-r * T) * norm.cdf(_d2)


def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Price a European put option under Black-Scholes.

    P = K * e^{-rT} * N(-d2) - S * N(-d1)
    """
    _d1 = d1(S, K, T, r, sigma)
    _d2 = d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-_d2) - S * norm.cdf(-_d1)


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes vega: sensitivity of price to a unit change in sigma.

    Vega = S * sqrt(T) * N'(d1)

    Same for calls and puts. Always positive, maximised ATM,
    approaches zero deep ITM/OTM.
    """
    _d1 = d1(S, K, T, r, sigma)
    return S * np.sqrt(T) * norm.pdf(_d1)


def implied_vol_newton(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    initial_guess: float = 0.2,
    max_iter: int = 100,
    tolerance: float = 1e-6,
) -> float | None:
    """
    Solve for implied volatility via Newton-Raphson.

    sigma_{n+1} = sigma_n - (BS(sigma_n) - market_price) / Vega(sigma_n)

    Returns None if the method fails to converge or vega becomes
    too small to safely divide by (caller should fall back to
    implied_vol_brent in that case).
    """
    if market_price <= 0:
        raise ValueError("market_price must be positive")
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")

    sigma = initial_guess
    price_fn = bs_call if option_type == "call" else bs_put

    for _ in range(max_iter):
        price = price_fn(S, K, T, r, sigma)
        diff = price - market_price

        if abs(diff) < tolerance:
            return sigma

        vega = bs_vega(S, K, T, r, sigma)
        if vega < 1e-8:
            return None  # near-zero vega, Newton unsafe

        sigma -= diff / vega

        if sigma <= 0:
            return None  # stepped into non-physical territory

    return None  # did not converge


def implied_vol_brent(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
) -> float | None:
    """
    Solve for implied volatility via Brent's method.

    Bracketing fallback for cases where Newton-Raphson fails
    (e.g. deep OTM options with near-zero vega). Searches
    sigma in (1e-6, 5.0) for a root of BS(sigma) - market_price.

    Returns None if no root exists in the bracket (e.g. market_price
    is outside the no-arbitrage bounds for this option).
    """
    if market_price <= 0:
        raise ValueError("market_price must be positive")
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")

    price_fn = bs_call if option_type == "call" else bs_put

    def objective(sigma: float) -> float:
        return price_fn(S, K, T, r, sigma) - market_price

    try:
        return brentq(objective, 1e-6, 5.0)
    except ValueError:
        return None  # objective doesn't change sign in bracket 