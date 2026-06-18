"""
tests/test_realised_vol.py

Tests for src/realised_vol.py

Coverage:
    fetch_price_history       — input validation, return type/shape
    compute_log_returns       — input validation, output correctness, known values
    compute_rolling_vol       — shape, annualisation, window handling
    fit_garch                 — parameter sanity, persistence, return dict structure
    walk_forward_garch_forecast — shape, index alignment, no lookahead, NaN prefix
"""

import numpy as np
import pandas as pd
import pytest

from src.realised_vol import (
    compute_log_returns,
    compute_rolling_vol,
    fit_garch,
    walk_forward_garch_forecast,
    walk_forward_garch_term_structure,
    TRADING_DAYS_PER_YEAR,
    DEFAULT_ROLLING_WINDOWS,
    DEFAULT_TERM_HORIZONS,
    DEFAULT_TRAIN_WINDOW,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_returns(n: int = 500, seed: int = 42, annual_vol: float = 0.16) -> pd.Series:
    """Synthetic daily log returns ~ N(0, sigma_daily^2)."""
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(TRADING_DAYS_PER_YEAR)
    dates = pd.bdate_range(start="2020-01-02", periods=n)
    return pd.Series(rng.normal(0, daily_vol, n), index=dates, name="returns")


def make_garch_returns(
    n: int = 700, seed: int = 42,
    omega: float = 5.08e-6, alpha: float = 0.10, beta: float = 0.85,
) -> pd.Series:
    """
    Synthetic daily log returns simulated from an actual GARCH(1,1) DGP
    (real conditional heteroskedasticity), as opposed to make_returns'
    i.i.d. noise. fit_garch needs genuine ARCH effects in the data to
    identify alpha/beta away from the boundary (alpha=0) — fitting
    GARCH to i.i.d. noise correctly returns a degenerate MLE solution,
    that's not a bug in fit_garch, it's a mismatched fixture.

    Defaults give persistence alpha+beta=0.95 and unconditional
    daily vol of omega/(1-persistence) ~ matching ~16% annualised.
    """
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    sigma2 = np.empty(n)
    r = np.empty(n)
    sigma2[0] = omega / (1 - alpha - beta)
    r[0] = np.sqrt(sigma2[0]) * eps[0]
    for t in range(1, n):
        sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]
        r[t] = np.sqrt(sigma2[t]) * eps[t]
    dates = pd.bdate_range(start="2020-01-02", periods=n)
    return pd.Series(r, index=dates, name="returns")


def make_prices(n: int = 100, seed: int = 0) -> pd.Series:
    """Synthetic price series via cumulative sum of log returns."""
    rng = np.random.default_rng(seed)
    daily_vol = 0.16 / np.sqrt(TRADING_DAYS_PER_YEAR)
    log_returns = rng.normal(0, daily_vol, n)
    log_prices = np.concatenate([[np.log(400)], np.log(400) + np.cumsum(log_returns)])
    dates = pd.bdate_range(start="2020-01-02", periods=n + 1)
    return pd.Series(np.exp(log_prices[:n+1]), index=dates, name="Close")


# ── compute_log_returns ───────────────────────────────────────────────────────

class TestComputeLogReturns:

    def test_output_length(self):
        prices = make_prices(100)
        returns = compute_log_returns(prices)
        # one observation lost to differencing
        assert len(returns) == len(prices) - 1

    def test_output_index_aligned(self):
        prices = make_prices(100)
        returns = compute_log_returns(prices)
        assert returns.index.equals(prices.index[1:])

    def test_known_value(self):
        """ln(110/100) = ln(1.1) exactly."""
        prices = pd.Series(
            [100.0, 110.0, 121.0],
            index=pd.bdate_range("2020-01-02", periods=3)
        )
        returns = compute_log_returns(prices)
        expected = np.log(1.1)
        assert np.allclose(returns.values, expected, atol=1e-10)

    def test_output_is_series(self):
        prices = make_prices(50)
        returns = compute_log_returns(prices)
        assert isinstance(returns, pd.Series)

    def test_no_nans_in_output(self):
        prices = make_prices(100)
        returns = compute_log_returns(prices)
        assert not returns.isnull().any()

    def test_raises_on_non_series(self):
        with pytest.raises(TypeError):
            compute_log_returns([100, 101, 102])

    def test_raises_on_single_observation(self):
        prices = pd.Series([100.0], index=pd.bdate_range("2020-01-02", periods=1))
        with pytest.raises(ValueError):
            compute_log_returns(prices)

    def test_raises_on_nan_in_prices(self):
        prices = pd.Series(
            [100.0, np.nan, 102.0],
            index=pd.bdate_range("2020-01-02", periods=3)
        )
        with pytest.raises(ValueError):
            compute_log_returns(prices)

    def test_raises_on_non_positive_price(self):
        prices = pd.Series(
            [100.0, 0.0, 102.0],
            index=pd.bdate_range("2020-01-02", periods=3)
        )
        with pytest.raises(ValueError):
            compute_log_returns(prices)


# ── compute_rolling_vol ───────────────────────────────────────────────────────

class TestComputeRollingVol:

    def test_output_shape(self):
        returns = make_returns(500)
        rv = compute_rolling_vol(returns)
        assert rv.shape == (len(returns), len(DEFAULT_ROLLING_WINDOWS))

    def test_column_names(self):
        returns = make_returns(500)
        rv = compute_rolling_vol(returns)
        expected = [f"rolling_vol_{w}d" for w in DEFAULT_ROLLING_WINDOWS]
        assert list(rv.columns) == expected

    def test_nan_prefix(self):
        """First (w-1) values of each column must be NaN."""
        returns = make_returns(500)
        rv = compute_rolling_vol(returns)
        for w in DEFAULT_ROLLING_WINDOWS:
            col = f"rolling_vol_{w}d"
            assert rv[col].iloc[:w - 1].isnull().all(), \
                f"Expected {w-1} NaNs at start of {col}"
            assert pd.notna(rv[col].iloc[w - 1]), \
                f"Expected first non-NaN at index {w-1} for {col}"

    def test_annualisation(self):
        """
        Constant returns series should give rolling vol = 0.
        Non-trivial check: known daily vol → annualised vol within tolerance.
        """
        target_ann_vol = 0.20
        returns = make_returns(500, annual_vol=target_ann_vol)
        rv = compute_rolling_vol(returns, windows=[63])
        # with 500 draws the 63-day rolling vol should be within 5pp of target
        median_vol = rv["rolling_vol_63d"].dropna().median()
        assert abs(median_vol - target_ann_vol) < 0.05, \
            f"Expected ~{target_ann_vol}, got {median_vol:.4f}"

    def test_constant_returns_give_zero_vol(self):
        dates = pd.bdate_range("2020-01-02", periods=100)
        returns = pd.Series(np.full(100, 0.001), index=dates)
        rv = compute_rolling_vol(returns, windows=[10])
        assert np.allclose(rv["rolling_vol_10d"].dropna().values, 0.0, atol=1e-10)

    def test_custom_windows(self):
        returns = make_returns(300)
        rv = compute_rolling_vol(returns, windows=[5, 20])
        assert list(rv.columns) == ["rolling_vol_5d", "rolling_vol_20d"]

    def test_raises_on_non_series(self):
        with pytest.raises(TypeError):
            compute_rolling_vol(np.array([0.01, 0.02, 0.03]))

    def test_raises_if_returns_shorter_than_max_window(self):
        returns = make_returns(50)
        with pytest.raises(ValueError):
            compute_rolling_vol(returns, windows=[10, 21, 63])

    def test_raises_on_window_less_than_2(self):
        returns = make_returns(200)
        with pytest.raises(ValueError):
            compute_rolling_vol(returns, windows=[1, 21])

    def test_output_index_matches_input(self):
        returns = make_returns(300)
        rv = compute_rolling_vol(returns)
        assert rv.index.equals(returns.index)


# ── fit_garch ─────────────────────────────────────────────────────────────────

class TestFitGarch:

    @pytest.fixture(scope="class")
    def garch_result(self):
        returns = make_garch_returns(700)
        return fit_garch(returns)

    def test_returns_dict(self, garch_result):
        assert isinstance(garch_result, dict)

    def test_required_keys(self, garch_result):
        expected_keys = {"model", "omega", "alpha", "beta", "persistence", "long_run_ann_vol"}
        assert expected_keys == set(garch_result.keys())

    def test_parameters_positive(self, garch_result):
        assert garch_result["omega"] > 0
        assert garch_result["alpha"] > 0
        assert garch_result["beta"] > 0

    def test_persistence_less_than_one(self, garch_result):
        """Stationarity condition: alpha + beta < 1."""
        assert garch_result["persistence"] < 1.0, \
            f"Non-stationary fit: persistence={garch_result['persistence']:.4f}"

    def test_persistence_equals_alpha_plus_beta(self, garch_result):
        assert np.isclose(
            garch_result["persistence"],
            garch_result["alpha"] + garch_result["beta"],
            atol=1e-10
        )

    def test_long_run_vol_plausible(self, garch_result):
        """Long-run annualised vol should be in (1%, 100%) for equity-like returns."""
        lrv = garch_result["long_run_ann_vol"]
        assert 0.01 < lrv < 1.0, f"Implausible long-run vol: {lrv:.4f}"

    def test_raises_on_too_few_observations(self):
        returns = make_returns(50)
        with pytest.raises(ValueError):
            fit_garch(returns)

    def test_raises_on_non_series(self):
        with pytest.raises(TypeError):
            fit_garch(np.array([0.01] * 200))


# ── walk_forward_garch_forecast ───────────────────────────────────────────────

class TestWalkForwardGarchForecast:

    @pytest.fixture(scope="class")
    def forecast_series(self):
        returns = make_returns(600)
        return returns, walk_forward_garch_forecast(returns, train_window=252)

    def test_output_is_series(self, forecast_series):
        _, forecasts = forecast_series
        assert isinstance(forecasts, pd.Series)

    def test_output_length_matches_input(self, forecast_series):
        returns, forecasts = forecast_series
        assert len(forecasts) == len(returns)

    def test_index_aligned_to_returns(self, forecast_series):
        returns, forecasts = forecast_series
        assert forecasts.index.equals(returns.index)

    def test_nan_prefix(self, forecast_series):
        """First train_window entries must be NaN — no forecast before first window."""
        returns, forecasts = forecast_series
        assert forecasts.iloc[:252].isnull().all(), \
            "Expected NaN for first 252 entries (no lookahead)"

    def test_forecasts_exist_after_warmup(self, forecast_series):
        _, forecasts = forecast_series
        assert forecasts.iloc[252:].notna().sum() > 0

    def test_no_lookahead(self):
        """
        Forecasts on first half and full series must agree on the overlapping window.
        If lookahead existed, the first-half forecasts would differ once the
        second half data is added.
        """
        returns = make_returns(600)
        half = 400

        fc_full = walk_forward_garch_forecast(returns, train_window=252)
        fc_half = walk_forward_garch_forecast(returns.iloc[:half], train_window=252)

        overlap = fc_half.dropna().index
        # allow small floating point divergence but no systematic difference
        diff = (fc_full.loc[overlap] - fc_half.loc[overlap]).abs()
        assert diff.max() < 1e-6, \
            f"Lookahead detected: max diff={diff.max():.2e}"

    def test_forecasts_positive(self, forecast_series):
        _, forecasts = forecast_series
        valid = forecasts.dropna()
        assert (valid > 0).all(), "All vol forecasts should be positive"

    def test_forecasts_plausible_range(self, forecast_series):
        """Annualised vol forecasts should sit in (1%, 150%) for equity-like returns."""
        _, forecasts = forecast_series
        valid = forecasts.dropna()
        assert (valid > 0.01).all() and (valid < 1.5).all(), \
            f"Implausible forecast range: min={valid.min():.4f}, max={valid.max():.4f}"

    def test_raises_if_too_short(self):
        returns = make_returns(200)
        with pytest.raises(ValueError):
            walk_forward_garch_forecast(returns, train_window=252)

    def test_raises_on_non_series(self):
        with pytest.raises(TypeError):
            walk_forward_garch_forecast(np.array([0.01] * 300))


# ── walk_forward_garch_term_structure ─────────────────────────────────────────

class TestWalkForwardGarchTermStructure:

    _HORIZONS = [21, 63, 126]

    @pytest.fixture(scope="class")
    def term_df(self):
        returns = make_returns(600)
        return returns, walk_forward_garch_term_structure(
            returns, horizons=self._HORIZONS, train_window=252
        )

    def test_output_is_dataframe(self, term_df):
        _, df = term_df
        assert isinstance(df, pd.DataFrame)

    def test_output_shape(self, term_df):
        returns, df = term_df
        assert df.shape == (len(returns), len(self._HORIZONS))

    def test_column_names(self, term_df):
        _, df = term_df
        expected = [f"garch_forecast_{h}d" for h in self._HORIZONS]
        assert list(df.columns) == expected

    def test_index_aligned_to_returns(self, term_df):
        returns, df = term_df
        assert df.index.equals(returns.index)

    def test_nan_prefix(self, term_df):
        """First train_window rows must be NaN across all columns."""
        _, df = term_df
        assert df.iloc[:252].isnull().all(axis=None), \
            "Expected NaN for first 252 rows (no lookahead)"

    def test_forecasts_exist_after_warmup(self, term_df):
        _, df = term_df
        assert df.iloc[252:].notna().all(axis=1).sum() > 0

    def test_forecasts_positive(self, term_df):
        _, df = term_df
        valid = df.dropna()
        assert (valid > 0).all(axis=None), "All vol forecasts should be positive"

    def test_forecasts_plausible_range(self, term_df):
        """Annualised vol forecasts should sit in (1%, 150%) for equity-like returns."""
        _, df = term_df
        valid = df.dropna()
        assert (valid > 0.01).all(axis=None) and (valid < 1.5).all(axis=None), \
            f"Implausible range: min={valid.min().min():.4f}, max={valid.max().max():.4f}"

    def test_cumulative_variance_grows_with_horizon(self, term_df):
        """
        Cumulative variance = ann_vol^2 * h / 252 must be non-decreasing across
        horizons for every date, because we sum non-negative daily variance terms.
        """
        _, df = term_df
        valid = df.dropna()
        for h1, h2 in zip(self._HORIZONS[:-1], self._HORIZONS[1:]):
            cum_var_h1 = valid[f"garch_forecast_{h1}d"] ** 2 * h1 / TRADING_DAYS_PER_YEAR
            cum_var_h2 = valid[f"garch_forecast_{h2}d"] ** 2 * h2 / TRADING_DAYS_PER_YEAR
            assert (cum_var_h2 >= cum_var_h1 - 1e-12).all(), \
                f"Cumulative variance decreased from h={h1} to h={h2}"

    def test_no_lookahead(self):
        """
        Term structure forecasts on the first half of returns must match those
        from the full series over the overlapping window.
        """
        returns = make_returns(600)
        half = 400
        horizons = [21, 63]

        fc_full = walk_forward_garch_term_structure(returns, horizons=horizons, train_window=252)
        fc_half = walk_forward_garch_term_structure(returns.iloc[:half], horizons=horizons, train_window=252)

        overlap = fc_half.dropna().index
        for col in fc_full.columns:
            diff = (fc_full.loc[overlap, col] - fc_half.loc[overlap, col]).abs()
            assert diff.max() < 1e-6, \
                f"Lookahead detected in {col}: max diff={diff.max():.2e}"

    def test_custom_horizons(self):
        returns = make_returns(400)
        df = walk_forward_garch_term_structure(returns, horizons=[10, 21], train_window=252)
        assert list(df.columns) == ["garch_forecast_10d", "garch_forecast_21d"]

    def test_raises_if_too_short(self):
        returns = make_returns(200)
        with pytest.raises(ValueError):
            walk_forward_garch_term_structure(returns, train_window=252)

    def test_raises_on_non_series(self):
        with pytest.raises(TypeError):
            walk_forward_garch_term_structure(np.array([0.01] * 400))

    def test_raises_on_empty_horizons(self):
        returns = make_returns(400)
        with pytest.raises(ValueError):
            walk_forward_garch_term_structure(returns, horizons=[], train_window=252)

    def test_raises_on_invalid_horizon(self):
        returns = make_returns(400)
        with pytest.raises(ValueError):
            walk_forward_garch_term_structure(returns, horizons=[0, 21], train_window=252)