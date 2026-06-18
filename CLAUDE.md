# Volatility Surface Analysis & Risk Premium Decomposition

## Project Goal
Extract the implied volatility surface from SPY (S&P 500 ETF) options, compare to realised
volatility via GARCH(1,1), and decompose the variance risk premium (VRP) into structural risk
premium vs. forecast error. Motivated by Bakshi & Kapadia (2003) and Guo & Loeper (2020).

## Repository Structure
src/
  black_scholes.py     # BS pricing, greeks, Newton-Raphson + Brent IV solvers
  data.py              # yfinance data fetching, options chain cleaning
  vol_surface.py       # IV surface construction and visualisation
  realised_vol.py      # returns, rolling vol, GARCH(1,1), multi-horizon walk-forward forecasts
  analysis.py          # VRP decomposition pipeline — in progress
tests/                 # pytest unit tests, one file per module
notebooks/             # one notebook per module

## Current State

### Complete and tested
- **Module 1** — black_scholes.py: BS pricing, vega, IV solvers (Newton-Raphson + Brent fallback). Full test suite passing.
- **Module 2** — data.py + vol_surface.py: spot, risk-free rate, options chain cleaning (OTM only, liquidity filters), IV surface construction, smile/term structure/3D plots. Validated against VIX.
- **Module 3** — realised_vol.py: log returns, rolling vol (10/21/63-day), GARCH(1,1) fit, multi-horizon walk-forward forecast (21/63/126-day). 37/37 tests passing.

### In progress
- **Module 4** — analysis.py: VRP decomposition. Pipeline design complete, implementation starting now.

## Module 4 — VRP Decomposition (src/analysis.py)

### Data
- SPY price history: 15 years via yfinance (captures multiple vol regimes — 2008, 2011, 2018, 2020, 2022)
- IV proxy: CBOE VIX term structure — VIX (30-day), VIX3M (63-day), VIX6M (126-day)
- VIX9D dropped — only ~6 trading days, no clean GARCH match, dropping it extends common history to 2006
- VIX1Y dropped — series too short, thin usable sample after forward-vol requirement
- All horizons in trading days: 21, 63, 126

### Decomposition
At each horizon N and date t:
- **Total gap**           = IV(t) — Forward Realised Vol(t, t+N)
- **Structural premium**  = IV(t) — GARCH forecast(t, t+N)
- **Forecast error**      = GARCH forecast(t, t+N) — Forward Realised Vol(t, t+N)
- Identity: Total gap = Structural premium + Forecast error (enforce in tests)

Forward realised vol computed from t to t+N using actual SPY returns — strictly no lookahead.
GARCH multi-horizon forecasts from updated walk_forward_garch_forecast — single fit per t,
cumulative variance sums across horizons, no refitting per horizon.

### Output
Single long-form DataFrame: date × horizon × {iv, garch_forecast, forward_realised_vol,
total_gap, structural_premium, forecast_error}. Built once, passed to all downstream functions.

### Known caveats (acknowledge in writeup, not bugs)
- Overlapping forward windows → autocorrelated series. Use Newey-West if running inference.
- VIX built from SPX options, project uses SPY — small dividend/basis difference, immaterial.
- GARCH forecast accuracy degrades at longer horizons (half-life ~23 days for SPY). 
  Decomposition still valid as benchmark separation, less so as precise forecast.

## Planned Visualisations (src/analysis.py)
- **Heatmap** — date (x) × horizon (y) × structural premium (colour). Shows full term structure
  evolution over time. Primary deliverable.
- **Stacked area chart** — structural premium vs forecast error over time per horizon
- **Term structure snapshot** — premium across horizons on a given date
- **Forecast error distribution** — by horizon, testing mean-zero hypothesis visually
- **Scatter** — structural premium vs VIX level, coloured by date (regime analysis)

## Planned Extensions (after core scope complete)
- Vol selling backtest — use structural premium as signal, compute simple P&L
- VRP as equity return predictor — regress forward SPY returns on lagged structural premium
- Regime analysis — does premium behaviour differ in high vs low vol environments

## Conventions — Enforce Strictly
- Pure maths functions and I/O functions in separate files
- Raise loudly on bad input in core functions
- Return None or skip gracefully at data edges
- logging not print
- Constants defined once at top of file, never repeated inline
- Fetch shared/expensive data once outside loops, pass down
- Functional flat-file style — no OOP, do not suggest it
- Always state exact file path before giving any code
- Commit after each working milestone, confirm via git log and GitHub directly

## Code Style
Python 3.12. Dependencies: yfinance, arch, plotly, pandas, numpy, scipy, pytest.
No classes. No print statements. Type hints welcome but not required.
Descriptive variable names. Single responsibility per function.

## Background
Built by Bilal — final-year Manchester maths undergrad, incoming Warwick MSc Mathematical
Finance. Targeting sell-side quant strats internships 2026. Deadline: 5 August 2026.