import pandas as pd
import numpy as np


def trend_breakout(
        df: pd.DataFrame,
        capital,
        horizon: int,
        vol_window: int = 21,
        cap: float = 20.0,
        idm: float = 1.0,
        tau: float = 0.2,
        multiplier: float = 50.0,
        fx: float = 1.0,
) -> pd.DataFrame:
    # INPUTS
    horizon = int(horizon)
    price = df['close']
    price_diff = price.diff().fillna(0.0)  # first diff = 0 (no prior price)
    sigma_price = price_diff.rolling(vol_window).std()
    sigma_price = sigma_price.ffill()
    sigma_price.replace({0.0: 1e-8}, inplace=True)  # guard against zero std to avoid division by zero



    # GETTING THE RAW FORECAST
    hi = price.rolling(window=horizon+1).max()
    lo = price.rolling(window=horizon+1).min()
    mean = (hi + lo) / 2.0
    unsmoothed = 40 * (price - mean) / (hi - lo)
    smoothed = unsmoothed.ewm(span=horizon+4).mean()
    raw = smoothed

    # SCALING FORECASTS
    avg_abs_forecast = raw.abs().mean(skipna=True)
    forecast_scale = 10.0 / (avg_abs_forecast if avg_abs_forecast != 0 else 1.0)
    scaled = raw * forecast_scale
    capped = scaled.clip(-cap, cap)

    # ANN STD FOR POSITION
    pct_change = price.pct_change().fillna(0.0)
    ann_std = pct_change.rolling(vol_window).std()
    ann_std = ann_std.ffill() * 16  # 252^0.5 ≈ 15.87, rounded ~16 for daily -> annual multiplier
    ann_std.replace({0.0: 1e-8}, inplace=True)

    # IDEAL POSITION
    N_unrounded = (
            capped * capital * idm * tau
            / (10 * multiplier * price * fx * ann_std)
    )

    N_unrounded = N_unrounded.fillna(0.0)


    # BUFFERING
    # skip for now

    position = np.round(N_unrounded).astype(int)

    return pd.DataFrame({
        'price': price,
        'price_diff': price_diff,
        'sigma_price': sigma_price,
        'ann_std': ann_std,
        'raw_forecast': raw,
        'scaled_forecast': scaled,
        'capped_forecast': capped,
        'N_unrounded': N_unrounded,
        'position': position
    })
