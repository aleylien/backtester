import pandas as pd
import numpy as np
from strategies.B_buffering import compute_buffer_contracts, apply_buffer_to_contracts


def ewmac(
        df: pd.DataFrame,
        capital: int,
        vol_window: int = 21,
        cap: float = 20.0,
        idm: float = 1.0,
        tau: float = 0.2,
        multiplier: float = 50.0,
        fx: float = 1.0,
        span_short: int = 16,
        span_long: int = 64,
        buffer_F: float = 0.10,
        use_buffer: bool = True,
):
    """
    Computes EWMA crossover forecasts, sizes positions to N contracts,
    simulates position (but not P&L here), and returns a DataFrame with:
      price, price_diff, sigma_price, ewma_short, ewma_long,
      raw_forecast, scaled_forecast, capped_forecast,
      N_unrounded, position
    """
    price = df['close']
    price_diff = price.diff().fillna(0.0)  # first diff = 0 (no prior price)
    # Rolling volatility of price changes
    sigma_price = price_diff.rolling(vol_window).std()
    # Fill forward to maintain last known volatility for trailing bars (no backward fill for start)
    sigma_price = sigma_price.ffill()
    # If still NaN at the very beginning (before vol_window data), leave as NaN for now.
    sigma_price.replace({0.0: 1e-8}, inplace=True)  # guard against zero std to avoid division by zero

    # Exponential moving averages for momentum
    ewma_s = price.ewm(span=span_short, adjust=False).mean()
    ewma_l = price.ewm(span=span_long, adjust=False).mean()

    # Compute raw forecast
    raw = (ewma_s - ewma_l) / sigma_price
    # # --- Automatically determine forecast_scale if requested ---
    # if forecast_scale is None or forecast_scale == 0:
    #     # Avoid including NaNs in mean (use dropna); use absolute mean as per formula
    #     avg_abs_forecast = raw.abs().mean(skipna=True)
    #     # Set scale so that long-run average |scaled_forecast| ≈ 10
    #     # If avg_abs_forecast is 0 (degenerate case), fallback to 1 to avoid division by zero
    #     forecast_scale = 10.0 / (avg_abs_forecast if avg_abs_forecast != 0 else 1.0)
    # Avoid including NaNs in mean (use dropna); use absolute mean as per formula
    avg_abs_forecast = raw.abs().mean(skipna=True)
    # Set scale so that long-run average |scaled_forecast| ≈ 10
    # If avg_abs_forecast is 0 (degenerate case), fallback to 1 to avoid division by zero
    forecast_scale = 10.0 / (avg_abs_forecast if avg_abs_forecast != 0 else 1.0)
    scaled = raw * forecast_scale
    capped = scaled.clip(-cap, cap)

    # Annualized volatility (for position sizing), using rolling std of pct changes
    pct_change = price.pct_change().fillna(0.0)
    ann_std = pct_change.rolling(vol_window).std()
    ann_std = ann_std.ffill() * 16  # 252^0.5 ≈ 15.87, rounded ~16 for daily -> annual multiplier
    ann_std.replace({0.0: 1e-8}, inplace=True)

    # Ideal position (unrounded) based on forecast and risk allocation
    N_unrounded = (
            capped * capital * idm * tau
            / (10 * multiplier * price * fx * ann_std)
    )

    initial_pos: int = 0

    # --- Buffering (Carver) ---
    if use_buffer:
        B = compute_buffer_contracts(
            price=price,
            ann_std=ann_std,
            capital=capital,
            idm=idm,
            tau=tau,
            multiplier=multiplier,
            fx=fx,
            F=buffer_F,
        )
        position = apply_buffer_to_contracts(
            N_unrounded=N_unrounded,
            buffer_contracts=B,
            initial_pos=initial_pos,
        ).astype(int)
    else:
        position = np.round(N_unrounded).astype(int)

    return (pd.DataFrame({
        'price': price,
        'price_diff': price_diff,
        'sigma_price': sigma_price,
        'ewma_short': ewma_s,
        'ewma_long': ewma_l,
        'raw_forecast': raw,
        'scaled_forecast': scaled,
        'capped_forecast': capped,
        'N_unrounded': N_unrounded,
        'position': position,
        'forecast_scale': forecast_scale,
        'buffer_contracts': B if use_buffer else pd.Series(0.0, index=N_unrounded.index),
    }))
