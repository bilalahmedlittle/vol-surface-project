import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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

def plot_vol_smile(surface_df: pd.DataFrame, expiry: str) -> None:
    """
    Plot the volatility smile (IV vs strike) for a single expiry.

    Args:
        surface_df: output of compute_surface
        expiry: expiry date string to plot (must exist in surface_df)
    """
    data = surface_df[surface_df["expiry"] == expiry].sort_values("strike")

    fig, ax = plt.subplots()
    for option_type, group in data.groupby("option_type"):
        ax.plot(group["strike"], group["implied_vol"], marker='o', label=option_type)

    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied Vol")
    ax.set_title(f"SPY Vol Smile — {expiry}")
    ax.legend()
    plt.show()


def plot_term_structure(surface_df: pd.DataFrame, spot_price: float) -> None:
    """
    Plot ATM implied vol against time to expiry (term structure).

    For each expiry, finds the option whose strike is closest to
    spot_price and uses its implied vol as the ATM proxy.

    Args:
        surface_df: output of compute_surface
        spot_price: current underlying spot price
    """
    atm_rows = []
    for expiry, group in surface_df.groupby("expiry"):
        idx = (group["strike"] - spot_price).abs().idxmin()
        atm_rows.append(group.loc[idx])

    atm_df = pd.DataFrame(atm_rows).sort_values("T")

    fig, ax = plt.subplots()
    ax.plot(atm_df["T"], atm_df["implied_vol"], marker='o')
    ax.set_xlabel("Time to Expiry (years)")
    ax.set_ylabel("ATM Implied Vol")
    ax.set_title("SPY ATM Vol Term Structure")
    plt.show()


def plot_vol_surface(surface_df: pd.DataFrame) -> None:
    """
    Plot the full implied volatility surface (strike x expiry x IV) in 3D.

    Args:
        surface_df: output of compute_surface, ideally across multiple expiries
    """
    fig = go.Figure(data=[go.Scatter3d(
        x=surface_df["strike"],
        y=surface_df["T"],
        z=surface_df["implied_vol"],
        mode='markers',
        marker=dict(size=3, color=surface_df["implied_vol"], colorscale='Viridis'),
    )])

    fig.update_layout(
        scene=dict(
            xaxis_title='Strike',
            yaxis_title='Time to Expiry (years)',
            zaxis_title='Implied Vol',
        ),
        title='SPY Implied Volatility Surface',
    )
    fig.show()