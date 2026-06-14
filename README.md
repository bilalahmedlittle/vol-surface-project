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