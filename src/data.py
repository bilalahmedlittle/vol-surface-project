import logging
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DEFAULT_TICKER = "SPY"
RATE_TICKER = "^IRX"  # 13-week T-bill, quoted as annualised percentage


def fetch_spot_price(ticker: str = DEFAULT_TICKER) -> float:
    """
    Fetch the most recent closing price for a ticker.

    Args:
        ticker: Yahoo Finance ticker symbol (default: SPY)

    Returns:
        Most recent close price as a float.

    Raises:
        RuntimeError: if no price data is returned.
    """
    history = yf.Ticker(ticker).history(period="1d")

    if history.empty:
        raise RuntimeError(f"No price data returned for {ticker}")

    price = history["Close"].iloc[-1]
    logger.info("Fetched spot price for %s: %.2f", ticker, price)
    return float(price)


def fetch_risk_free_rate(ticker: str = RATE_TICKER) -> float:
    """
    Fetch the current risk-free rate proxy from the 13-week T-bill (^IRX).

    ^IRX is quoted as an annualised percentage (e.g. 5.25 means 5.25%),
    so this divides by 100 to return a decimal suitable for use as `r`
    in the Black-Scholes functions.

    Args:
        ticker: Yahoo Finance ticker symbol (default: ^IRX)

    Returns:
        Risk-free rate as a decimal (e.g. 0.0525 for 5.25%).

    Raises:
        RuntimeError: if no rate data is returned.
    """
    history = yf.Ticker(ticker).history(period="1d")

    if history.empty:
        raise RuntimeError(f"No rate data returned for {ticker}")

    rate_pct = history["Close"].iloc[-1]
    logger.info("Fetched risk-free rate from %s: %.4f%%", ticker, rate_pct)
    return float(rate_pct) / 100

# Add near the top, with your other constants
MONEYNESS_LOWER = 0.5  # filter out strikes below 0.5x spot
MONEYNESS_UPPER = 2.0  # filter out strikes above 2x spot


def fetch_options_chain(
    ticker: str,
    expiry: str,
    spot_price: float | None = None,
) -> pd.DataFrame:
    """
    Fetch and clean the options chain (calls + puts) for one expiry.

    Args:
        ticker: Yahoo Finance ticker symbol (e.g. "SPY")
        expiry: Expiry date string in "YYYY-MM-DD" format, must be one
            of the dates returned by yf.Ticker(ticker).options
        spot_price: Current spot price, used for moneyness filtering.
            If None, it will be fetched (costs an extra API call —
            pass it explicitly when looping over many expiries).

    Returns:
        DataFrame with one row per cleaned option, columns:
        strike, option_type, mid_price, bid, ask, volume,
        open_interest, expiry, T (time to expiry in years).

    Raises:
        ValueError: if expiry is not available for this ticker.
    """
    yf_ticker = yf.Ticker(ticker)

    if expiry not in yf_ticker.options:
        raise ValueError(f"Expiry {expiry} not available for {ticker}")

    if spot_price is None:
        spot_price = fetch_spot_price(ticker)

    chain = yf_ticker.option_chain(expiry)

    calls = chain.calls.copy()
    calls["option_type"] = "call"

    puts = chain.puts.copy()
    puts["option_type"] = "put"

    combined = pd.concat([calls, puts], ignore_index=True)

    # --- Data cleaning ---
    n_before = len(combined)

    # 1. Drop quotes with no real market (zero bid/volume/OI)
    combined = combined[
        (combined["bid"] > 0)
        & (combined["volume"] > 0)
        & (combined["openInterest"] > 0)
    ]

    # 2. Filter strikes too far from spot (illiquid, unreliable IV)
    combined = combined[
        (combined["strike"] >= MONEYNESS_LOWER * spot_price)
        & (combined["strike"] <= MONEYNESS_UPPER * spot_price)
    ]

    n_after = len(combined)
    logger.info(
        "Cleaned options chain for %s %s: kept %d/%d rows",
        ticker, expiry, n_after, n_before,
    )

    # --- Derived columns ---
    combined["mid_price"] = (combined["bid"] + combined["ask"]) / 2

    expiry_date = pd.Timestamp(expiry)
    today = pd.Timestamp.now().normalize()
    T = (expiry_date - today).days / 365

    combined["expiry"] = expiry
    combined["T"] = T

    return combined[
        ["strike", "option_type", "mid_price", "bid", "ask",
         "volume", "openInterest", "expiry", "T"]
    ].rename(columns={"openInterest": "open_interest"}).reset_index(drop=True)

