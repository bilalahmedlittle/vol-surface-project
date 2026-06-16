# Volatility Surface Analysis & Risk Premium Decomposition

## Overview
This project builds a volatility analysis toolkit for SPY (S&P 500 ETF) options,
implementing core quantitative finance concepts from first principles. The toolkit:

- Extracts implied volatility from live options chain data via Black-Scholes 
  inversion (Newton-Raphson)
- Constructs and visualises the full volatility surface across strikes and expiries
- Forecasts realised volatility using GARCH(1,1) with walk-forward validation
- Analyses the volatility risk premium — the persistent spread between implied 
  and realised volatility

## Motivation
Implied volatility tends to exceed subsequently realised volatility on average —
a phenomenon known as the volatility risk premium (VRP). This premium represents
compensation option sellers receive for bearing variance risk. Guo and Loeper (2020)
document a persistent positive VRP for S&P 500 options over 2010-2018. This project
measures, decomposes, and analyses that premium using more recent SPY options data
(2018-2024), covering COVID and the 2022 rate shock.

## Project Structure

## Methodology
### Module 1 — Black-Scholes Implied Vol Solver
Implements the BS formula from scratch and solves for implied vol via 
Newton-Raphson with Brent's method as fallback.

### Module 2 — Volatility Surface
Pulls SPY options chain via yfinance, applies the implied vol solver across 
all strikes and expiries, constructs and visualises the 3D surface.

### Module 3 — Realised Vol and GARCH
Computes rolling realised vol at multiple windows and fits GARCH(1,1) via MLE 
using walk-forward validation to generate out-of-sample forecasts.

### Module 4 — VRP Decomposition
Decomposes the implied-realised spread into:
- Risk premium: IV minus GARCH forecast (structural seller's premium)
- Forecast error: GARCH forecast minus realised vol

## Key Results
*To be completed*

## How to Run
```bash
pip install -r requirements.txt
python src/analysis.py
```

## Dependencies
See requirements.txt

## References
- Guo, I. and Loeper, G. (2020). The Volatility Risk Premium: An Empirical 
  Study on the S&P 500 Index. SSRN: 3739933.
- Bakshi, G. and Kapadia, N. (2003). Delta-Hedged Gains and the Negative 
  Market Volatility Risk Premium. Review of Financial Studies, 16(2), 527-566.