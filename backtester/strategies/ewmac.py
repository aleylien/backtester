import pandas as pd
import numpy as np


def ewmac(
    df: pd.DataFrame,
    span_short: int = 16,
    span_long: int = 64,
    vol_window: int = 21,
    forecast_scale: float = 4.1,
    cap: float = 20.0,
    capital: float = 100_000,
    idm: float = 1.0,
    weight: float = 0.2,
    tau: float = 1.0,
    multiplier: float = 50.0,
    fx: float = 1.0,
) -> pd.DataFrame:
    """
    Computes EWMA crossover forecasts, sizes positions directly to rounded N,
    simulates P&L, and returns a DataFrame with:
      price, price_diff, sigma_price, ewma_short, ewma_long,
      raw_forecast, scaled_forecast, capped_forecast,
      N_unrounded, position, equity, drawdown
    """
    price      = df['close']
    price_diff = price.diff().fillna(0)

    # sigma_price = rolling std of price changes
    sigma_price = price_diff.rolling(vol_window).std().bfill().ffill().fillna(1e-8)

    # Exponential moving averages
    ewma_s = price.ewm(span=span_short, adjust=False).mean()
    ewma_l = price.ewm(span=span_long,  adjust=False).mean()

    # raw / scaled / capped forecasts
    raw    = (ewma_s - ewma_l) / sigma_price
    scaled = raw * forecast_scale
    capped = scaled.clip(-cap, cap)

    # stdev
    pct_change = price.pct_change().fillna(0)
    ann_std = pct_change.rolling(vol_window).std().bfill().ffill().fillna(1e-8) * 16

    # ideal (unrounded) position
    N_unrounded = (
        capped
        * capital * idm * weight * tau
        / (10 * multiplier * price * fx * ann_std)
    )
    # round to integer contracts
    position = np.round(N_unrounded).astype(int)

    return pd.DataFrame({
        'price':           price,
        'price_diff':      price_diff,
        'pct_change':      pct_change,
        'sigma_price':     sigma_price,
        'ewma_short':      ewma_s,
        'ewma_long':       ewma_l,
        'raw_forecast':    raw,
        'scaled_forecast': scaled,
        'capped_forecast': capped,
        'N_unrounded':     N_unrounded,
        'position':        position,
    })
