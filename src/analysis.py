"""
analysis.py

VRP decomposition: structural risk premium vs. forecast error.

At each horizon and date t:
    Total gap         = IV(t) — Forward Realised Vol(t, t+N)
    Structural premium = IV(t) — GARCH forecast(t, t+N)
    Forecast error    = GARCH forecast(t, t+N) — Forward Realised Vol(t, t+N)

IV proxies: CBOE VIX term structure (VIX, VIX3M, VIX6M).
Realised vol: log returns on SPY (no lookahead).
GARCH forecasts: walk_forward_garch_term_structure from realised_vol.py.
"""

import logging

import numpy as np
import pandas as pd
import yfinance as yf

from src.realised_vol import (
    fetch_price_history,
    compute_log_returns,
    walk_forward_garch_term_structure,
    DEFAULT_TERM_HORIZONS,
    DEFAULT_TRAIN_WINDOW,
    TRADING_DAYS_PER_YEAR,
)

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
SPY_TICKER  = "SPY"
SPY_PERIOD  = "15y"                         # covers full VIX3M/VIX6M history back to ~2006

VIX_TICKERS = ["^VIX", "^VIX3M", "^VIX6M"] # VIX9D excluded — see notebook 02
GARCH_HORIZONS = [21, 63, 126]              # trading-day horizons matching VIX, VIX3M, VIX6M

VIX_HORIZON_MAP: dict[str, int] = {
    "^VIX":   21,
    "^VIX3M": 63,
    "^VIX6M": 126,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tz_naive(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a tz-naive DatetimeIndex regardless of input tz."""
    df = df.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    return df


# ── Core pipeline functions ───────────────────────────────────────────────────

def fetch_vix_term_structure(tickers: list[str] = VIX_TICKERS) -> pd.DataFrame:
    """
    Fetch historical daily closes for VIX term structure tickers.

    Returns a tz-naive DataFrame with one column per ticker, outer-joined on
    date (so shorter series have NaN before their start). Drops rows where all
    values are NaN.
    """
    series = {}
    for ticker in tickers:
        raw = yf.download(ticker, period="max", auto_adjust=False, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        close = raw["Close"].dropna().sort_index()
        series[ticker] = close
        logger.info("Fetched %s: %d rows (%s to %s)",
                    ticker, len(close),
                    close.index[0].date(), close.index[-1].date())

    df = pd.DataFrame(series)
    df = df[~df.isna().all(axis=1)]
    return _tz_naive(df)


def compute_forward_realised_vol(
    returns: pd.Series,
    horizons: list[int] = GARCH_HORIZONS,
) -> pd.DataFrame:
    """
    At each date t, compute annualised realised vol of the N trading days
    immediately following t (log-returns r_{t+1} … r_{t+N}).

    NaN in the last N rows of each column is correct — the forward window
    is incomplete at the tail of the sample.
    """
    squared = returns ** 2
    result = {}
    for N in horizons:
        forward_var_sum = squared.rolling(N).sum().shift(-N)
        result[f"forward_rv_{N}d"] = np.sqrt(252 / N * forward_var_sum)
    return _tz_naive(pd.DataFrame(result, index=returns.index))


def decompose_vrp(
    vix_df: pd.DataFrame,
    garch_df: pd.DataFrame,
    forward_rv_df: pd.DataFrame,
    horizon_map: dict[str, int] = VIX_HORIZON_MAP,
) -> pd.DataFrame:
    """
    Long-form VRP decomposition across horizons.

    Inputs must share a common tz-naive DatetimeIndex. Rows where any of
    IV / GARCH forecast / forward RV is NaN are dropped — removes the GARCH
    warmup prefix and the tail where the forward window is incomplete.

    All vol columns in output are in decimal annualised form.

    Identity (enforced by construction):
        total_gap = structural_premium + forecast_error
    """
    frames = []
    for ticker, h in horizon_map.items():
        iv     = vix_df[ticker] / 100
        garch  = garch_df[f"garch_forecast_{h}d"]
        fwd_rv = forward_rv_df[f"forward_rv_{h}d"]

        block = pd.DataFrame({
            "iv":             iv,
            "garch_forecast": garch,
            "forward_rv":     fwd_rv,
        }).dropna()

        block = block.assign(
            horizon            = h,
            vix_ticker         = ticker,
            total_gap          = block["iv"] - block["forward_rv"],
            structural_premium = block["iv"] - block["garch_forecast"],
            forecast_error     = block["garch_forecast"] - block["forward_rv"],
        )
        frames.append(block)

    out = (
        pd.concat(frames)
        .rename_axis("date")
        .reset_index()
    )
    col_order = [
        "date", "horizon", "vix_ticker",
        "iv", "garch_forecast", "forward_rv",
        "total_gap", "structural_premium", "forecast_error",
    ]
    return out[col_order].sort_values(["date", "horizon"]).reset_index(drop=True)


def build_vrp_dataframe(
    spy_period: str = SPY_PERIOD,
    train_window: int = DEFAULT_TRAIN_WINDOW,
    horizons: list[int] = GARCH_HORIZONS,
    vix_tickers: list[str] = VIX_TICKERS,
) -> pd.DataFrame:
    """
    Full VRP decomposition pipeline. Fetches all data, runs GARCH walk-forward,
    computes forward realised vol, and returns a clean long-form DataFrame.

    Columns: date, horizon, vix_ticker, iv, garch_forecast, forward_rv,
             total_gap, structural_premium, forecast_error.

    All vol values are decimal annualised (e.g. 0.20 = 20%).
    """
    logger.info("Building VRP dataframe — SPY period=%s, horizons=%s", spy_period, horizons)

    prices  = fetch_price_history(SPY_TICKER, period=spy_period)
    returns = compute_log_returns(prices)
    logger.info("SPY returns: %d rows (%s → %s)",
                len(returns), returns.index[0].date(), returns.index[-1].date())

    vix_df = fetch_vix_term_structure(vix_tickers)
    logger.info("VIX data: %d rows (%s → %s)",
                len(vix_df), vix_df.index[0].date(), vix_df.index[-1].date())

    garch_df = walk_forward_garch_term_structure(
        returns,
        horizons=horizons,
        train_window=train_window,
    )
    garch_df = _tz_naive(garch_df.dropna())
    logger.info("GARCH forecasts: %d rows (%s → %s)",
                len(garch_df), garch_df.index[0].date(), garch_df.index[-1].date())

    forward_rv_df = compute_forward_realised_vol(returns, horizons=horizons)
    logger.info("Forward RV computed: %d rows", len(forward_rv_df))

    vrp_df = decompose_vrp(vix_df, garch_df, forward_rv_df)
    logger.info(
        "VRP dataframe built: %d rows across %d horizons (%s → %s)",
        len(vrp_df),
        vrp_df["horizon"].nunique(),
        vrp_df["date"].min().date(),
        vrp_df["date"].max().date(),
    )
    return vrp_df
