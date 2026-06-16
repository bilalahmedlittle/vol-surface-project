import pandas as pd

from src.black_scholes import implied_vol_newton, implied_vol_brent
from src.data import fetch_options_chain, fetch_spot_price, fetch_risk_free_rate


def compute_surface(ticker: str, expiries: list[str]) -> pd.DataFrame:
    """
    Build the implied volatility surface across multiple expiries.

    Uses only OTM options for IV extraction (OTM calls for
    strikes >= spot, OTM puts for strikes <= spot), since ITM
    options are dominated by intrinsic value and produce
    numerically unstable/meaningless implied vols.

    Args:
        ticker: Yahoo Finance ticker symbol (e.g. "SPY")
        expiries: list of expiry date strings ("YYYY-MM-DD") to include

    Returns:
        DataFrame with columns: strike, option_type, expiry, T,
        mid_price, implied_vol
    """
    spot = fetch_spot_price(ticker)
    r = fetch_risk_free_rate()

    rows = []
    n_skipped = 0

    for expiry in expiries:
        chain = fetch_options_chain(ticker, expiry, spot_price=spot)

        # Keep only OTM options — see docstring
        otm_calls = chain[(chain["option_type"] == "call") & (chain["strike"] >= spot)]
        otm_puts = chain[(chain["option_type"] == "put") & (chain["strike"] <= spot)]
        chain = pd.concat([otm_calls, otm_puts], ignore_index=True)

        for _, opt in chain.iterrows():
            if opt["T"] <= 0:
                n_skipped += 1
                continue

            iv = implied_vol_newton(
                market_price=opt["mid_price"],
                S=spot, K=opt["strike"], T=opt["T"], r=r,
                option_type=opt["option_type"],
            )

            if iv is None:
                iv = implied_vol_brent(
                    market_price=opt["mid_price"],
                    S=spot, K=opt["strike"], T=opt["T"], r=r,
                    option_type=opt["option_type"],
                )

            if iv is None:
                n_skipped += 1
                continue

            rows.append({
                "strike": opt["strike"],
                "option_type": opt["option_type"],
                "expiry": expiry,
                "T": opt["T"],
                "mid_price": opt["mid_price"],
                "implied_vol": iv,
            })

    surface = pd.DataFrame(rows)
    print(f"Surface built: {len(surface)} options, {n_skipped} skipped (no IV solution)")
    return surface