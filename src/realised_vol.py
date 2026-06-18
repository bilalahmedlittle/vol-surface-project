"""
realised_vol.py

Fetch historical price data, compute log returns, rolling volatility,
and GARCH(1,1) forecasts for use in VRP decomposition.
"""

import logging
import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
TRADING_DAYS_PER_YEAR = 252
DEFAULT_ROLLING_WINDOWS = [10, 21, 63]   # ~2-week, 1-month, 1-quarter
DEFAULT_TRAIN_WINDOW = 252               # rolling GARCH train window (1 year)
DEFAULT_PERIOD = "5y"                    # yfinance period string
DEFAULT_TERM_HORIZONS = [21, 63, 126]    # ~1mo, ~3mo, ~6mo trading days


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_price_history(ticker: str, period: str = DEFAULT_PERIOD) -> pd.Series:
    """
    Fetch daily adjusted closing prices for `ticker` over `period`.

    Parameters
    ----------
    ticker : str
        e.g. "SPY"
    period : str
        yfinance period string: "1y", "2y", "5y", "10y", "max"

    Returns
    -------
    pd.Series
        Daily adjusted close prices, DatetimeIndex, sorted ascending.
        Raises ValueError if no data returned.
    """
    if not ticker or not isinstance(ticker, str):
        raise ValueError(f"ticker must be a non-empty string, got: {ticker!r}")

    valid_periods = {"1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}
    if period not in valid_periods:
        raise ValueError(f"period must be one of {valid_periods}, got: {period!r}")

    logger.info(f"Fetching price history for {ticker}, period={period}")
    raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError(f"yfinance returned no data for ticker={ticker!r}, period={period!r}")

    # yfinance returns MultiIndex columns when auto_adjust=True — flatten if needed
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    prices = raw["Close"].dropna().sort_index()

    logger.info(f"Fetched {len(prices)} daily closes for {ticker} "
                f"({prices.index[0].date()} to {prices.index[-1].date()})")
    return prices


# ── Pure maths ────────────────────────────────────────────────────────────────

def compute_log_returns(prices: pd.Series) -> pd.Series:
    """
    Compute daily log returns from a price series.

    Parameters
    ----------
    prices : pd.Series
        Daily prices with DatetimeIndex. Must have at least 2 observations.

    Returns
    -------
    pd.Series
        Log returns: r_t = ln(P_t / P_{t-1}).
        First observation is dropped (NaN from diff).
        Same index as input minus first row.
    """
    if not isinstance(prices, pd.Series):
        raise TypeError(f"prices must be a pd.Series, got {type(prices)}")
    if len(prices) < 2:
        raise ValueError(f"prices must have at least 2 observations, got {len(prices)}")
    if prices.isnull().any():
        raise ValueError("prices contains NaN values — clean before passing in")
    if (prices <= 0).any():
        raise ValueError("prices contains non-positive values — log returns undefined")

    returns = np.log(prices / prices.shift(1)).dropna()
    logger.info(f"Computed {len(returns)} log returns "
                f"({returns.index[0].date()} to {returns.index[-1].date()})")
    return returns

# ── Rolling volatility ────────────────────────────────────────────────────────

def compute_rolling_vol(
    returns: pd.Series,
    windows: list[int] = DEFAULT_ROLLING_WINDOWS
) -> pd.DataFrame:
    """
    Compute annualised rolling volatility for multiple windows.

    Parameters
    ----------
    returns : pd.Series
        Daily log returns with DatetimeIndex.
    windows : list[int]
        Rolling window sizes in trading days. Default [10, 21, 63].

    Returns
    -------
    pd.DataFrame
        Columns: f"rolling_vol_{w}d" for each w in windows.
        Annualised: std of log returns * sqrt(252).
        First (w-1) rows per column are NaN by construction.
    """
    if not isinstance(returns, pd.Series):
        raise TypeError(f"returns must be a pd.Series, got {type(returns)}")
    if len(returns) < max(windows):
        raise ValueError(
            f"returns length {len(returns)} is shorter than longest window {max(windows)}"
        )
    if any(w < 2 for w in windows):
        raise ValueError("all windows must be >= 2")

    result = pd.DataFrame(index=returns.index)
    for w in windows:
        col = f"rolling_vol_{w}d"
        result[col] = (
            returns.rolling(window=w).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        )
        logger.info(f"Computed {col}: {result[col].notna().sum()} non-NaN observations")

    return result


# ── GARCH(1,1) — full sample fit (diagnostics) ───────────────────────────────

def fit_garch(returns: pd.Series) -> dict:
    """
    Fit GARCH(1,1) on the full returns sample. For diagnostics and parameter
    inspection only — do NOT use forecasts from this for VRP decomposition
    (lookahead bias). Use walk_forward_garch_forecast for that.

    Parameters
    ----------
    returns : pd.Series
        Daily log returns.

    Returns
    -------
    dict with keys:
        "model"   : fitted arch ARCHModelResult object
        "omega"   : float
        "alpha"   : float (ARCH term)
        "beta"    : float (GARCH term)
        "persistence" : float (alpha + beta)
        "long_run_ann_vol" : float (annualised long-run vol)
    """
    try:
        from arch import arch_model
    except ImportError:
        raise ImportError("arch package required: pip install arch")

    if not isinstance(returns, pd.Series):
        raise TypeError(f"returns must be a pd.Series, got {type(returns)}")
    if len(returns) < 100:
        raise ValueError(f"need at least 100 observations to fit GARCH, got {len(returns)}")

    # arch expects returns in percentage terms for numerical stability
    returns_pct = returns * 100

    logger.info(f"Fitting GARCH(1,1) on {len(returns)} observations")
    am = arch_model(returns_pct, vol="Garch", p=1, q=1, dist="normal", rescale=False)
    result = am.fit(disp="off")

    omega = result.params["omega"]
    alpha = result.params["alpha[1]"]
    beta  = result.params["beta[1]"]
    persistence = alpha + beta
    long_run_daily_var = omega / (1 - persistence)
    long_run_ann_vol = np.sqrt(long_run_daily_var * TRADING_DAYS_PER_YEAR) / 100  # back to decimal

    logger.info(
        f"GARCH(1,1) fit — omega={omega:.6f}, alpha={alpha:.4f}, "
        f"beta={beta:.4f}, persistence={persistence:.4f}, "
        f"long_run_ann_vol={long_run_ann_vol:.2%}"
    )

    if persistence >= 1:
        logger.warning(f"GARCH persistence={persistence:.4f} >= 1 — non-stationary fit, interpret with caution")

    return {
        "model": result,
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "persistence": persistence,
        "long_run_ann_vol": long_run_ann_vol,
    }


# ── Walk-forward GARCH forecast (pipeline input) ─────────────────────────────

def walk_forward_garch_term_structure(
    returns: pd.Series,
    horizons: list[int] = DEFAULT_TERM_HORIZONS,
    train_window: int = DEFAULT_TRAIN_WINDOW,
) -> pd.DataFrame:
    """
    Produce multi-horizon annualised GARCH(1,1) vol forecasts using a
    rolling train window. At each date t, fits once on the trailing
    train_window and calls res.forecast(horizon=max_horizon) once, then
    takes cumulative sums of the daily variance path to derive each
    horizon's N-day annualised vol forecast.

    No lookahead: forecast for date t uses only data up to and including t-1.

    Parameters
    ----------
    returns : pd.Series
        Daily log returns with DatetimeIndex.
    horizons : list[int]
        Forecast horizons in trading days. Default [21, 63, 126].
    train_window : int
        Rolling training window size. Default 252.

    Returns
    -------
    pd.DataFrame
        Columns: f"garch_forecast_{N}d" for each N in horizons.
        Index aligned to returns. First train_window rows are NaN.
    """
    try:
        from arch import arch_model
    except ImportError:
        raise ImportError("arch package required: pip install arch")

    if not isinstance(returns, pd.Series):
        raise TypeError(f"returns must be a pd.Series, got {type(returns)}")
    if not horizons:
        raise ValueError("horizons must be a non-empty list")
    if any(h < 1 for h in horizons):
        raise ValueError("all horizons must be >= 1")
    if len(returns) < train_window + 1:
        raise ValueError(
            f"returns length {len(returns)} must exceed train_window {train_window}"
        )

    max_horizon = max(horizons)
    returns_pct = returns * 100
    cols = [f"garch_forecast_{h}d" for h in horizons]
    result = pd.DataFrame(np.nan, index=returns.index, columns=cols, dtype=float)
    n = len(returns)
    total_steps = n - train_window
    log_every = max(1, total_steps // 10)

    logger.info(
        f"Starting walk-forward GARCH term structure: {total_steps} steps, "
        f"train_window={train_window}, horizons={horizons}"
    )

    for i in range(train_window, n):
        train_slice = returns_pct.iloc[i - train_window: i]
        forecast_date = returns.index[i]

        try:
            am = arch_model(train_slice, vol="Garch", p=1, q=1, dist="normal", rescale=False)
            res = am.fit(disp="off", show_warning=False)

            # one call covers all horizons up to max_horizon
            fc = res.forecast(horizon=max_horizon, reindex=False)
            daily_vars = fc.variance.iloc[-1].values  # shape (max_horizon,), pct² units

            for h in horizons:
                # cumulative variance over h days → annualise → convert from pct to decimal
                cum_var_pct2 = daily_vars[:h].sum()
                result.loc[forecast_date, f"garch_forecast_{h}d"] = (
                    np.sqrt(cum_var_pct2 * TRADING_DAYS_PER_YEAR / h) / 100
                )

        except Exception as e:
            logger.warning(f"GARCH fit failed at {forecast_date.date()}: {e} — skipping")

        if (i - train_window + 1) % log_every == 0:
            pct_done = (i - train_window + 1) / total_steps * 100
            logger.info(f"Walk-forward term structure progress: {pct_done:.0f}%")

    valid = result.notna().all(axis=1).sum()
    logger.info(f"Walk-forward term structure complete: {valid}/{total_steps} fully valid rows")
    return result


def walk_forward_garch_forecast(
    returns: pd.Series,
    train_window: int = DEFAULT_TRAIN_WINDOW
) -> pd.Series:
    """
    Produce one-step-ahead annualised GARCH(1,1) vol forecasts using a
    rolling train window. No lookahead: forecast for date t is made using
    only data up to and including t-1.

    Parameters
    ----------
    returns : pd.Series
        Daily log returns with DatetimeIndex.
    train_window : int
        Number of trading days in each rolling training window. Default 252.

    Returns
    -------
    pd.Series
        Annualised vol forecasts, DatetimeIndex aligned to returns.
        First (train_window) entries are NaN (no forecast until first
        full window is available).
    """
    try:
        from arch import arch_model
    except ImportError:
        raise ImportError("arch package required: pip install arch")

    if not isinstance(returns, pd.Series):
        raise TypeError(f"returns must be a pd.Series, got {type(returns)}")
    if len(returns) < train_window + 1:
        raise ValueError(
            f"returns length {len(returns)} must exceed train_window {train_window}"
        )

    returns_pct = returns * 100
    forecasts = pd.Series(index=returns.index, dtype=float)
    n = len(returns)
    total_steps = n - train_window
    log_every = max(1, total_steps // 10)

    logger.info(
        f"Starting walk-forward GARCH forecast: {total_steps} steps, "
        f"train_window={train_window}"
    )

    for i in range(train_window, n):
        train_slice = returns_pct.iloc[i - train_window: i]
        forecast_date = returns.index[i]

        try:
            am = arch_model(train_slice, vol="Garch", p=1, q=1, dist="normal", rescale=False)
            res = am.fit(disp="off", show_warning=False)

            # one-step-ahead variance forecast (still in pct² units)
            fc = res.forecast(horizon=1, reindex=False)
            daily_var_pct2 = fc.variance.iloc[-1, 0]

            # annualise and convert back to decimal vol
            ann_vol = np.sqrt(daily_var_pct2 * TRADING_DAYS_PER_YEAR) / 100
            forecasts.iloc[i] = ann_vol

        except Exception as e:
            logger.warning(f"GARCH fit failed at {forecast_date.date()}: {e} — skipping")
            # leave as NaN, analysis layer handles gaps

        if (i - train_window + 1) % log_every == 0:
            pct_done = (i - train_window + 1) / total_steps * 100
            logger.info(f"Walk-forward progress: {pct_done:.0f}%")

    valid = forecasts.notna().sum()
    logger.info(f"Walk-forward complete: {valid}/{total_steps} valid forecasts")
    return forecasts


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_vol_comparison(
    rolling_vols: pd.DataFrame,
    garch_forecasts: pd.Series,
    iv_series: pd.Series | None = None,
    title: str = "Volatility Comparison: Rolling vs GARCH vs IV"
) -> None:
    """
    Plot rolling vol estimates, GARCH walk-forward forecasts, and
    optionally an IV time series on the same axis.

    Parameters
    ----------
    rolling_vols : pd.DataFrame
        Output of compute_rolling_vol — columns rolling_vol_Xd.
    garch_forecasts : pd.Series
        Output of walk_forward_garch_forecast — annualised, same index.
    iv_series : pd.Series or None
        ATM IV time series (annualised decimal). Optional.
    title : str
        Plot title.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly required: pip install plotly")

    fig = go.Figure()

    colors = {"rolling_vol_10d": "#636EFA", "rolling_vol_21d": "#EF553B", "rolling_vol_63d": "#00CC96"}
    labels = {"rolling_vol_10d": "Rolling 10d", "rolling_vol_21d": "Rolling 21d", "rolling_vol_63d": "Rolling 63d"}

    for col in rolling_vols.columns:
        fig.add_trace(go.Scatter(
            x=rolling_vols.index,
            y=rolling_vols[col] * 100,   # plot as percentage
            mode="lines",
            name=labels.get(col, col),
            line=dict(color=colors.get(col, None), width=1.2),
            opacity=0.75,
        ))

    fig.add_trace(go.Scatter(
        x=garch_forecasts.index,
        y=garch_forecasts * 100,
        mode="lines",
        name="GARCH(1,1) forecast",
        line=dict(color="#FFA15A", width=1.8, dash="dash"),
    ))

    if iv_series is not None:
        fig.add_trace(go.Scatter(
            x=iv_series.index,
            y=iv_series * 100,
            mode="lines",
            name="ATM IV",
            line=dict(color="#FF6692", width=1.8),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Annualised Volatility (%)",
        hovermode="x unified",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.show()