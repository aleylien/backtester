import pandas as pd
import numpy as np


def trend_breakout1(
        df: pd.DataFrame,
        horizons: list = [10, 20, 40, 80, 160, 320],
        ewma_offset: int = 4,
        forecast_scalars: dict = None,
        forecast_scale: float = None,
        capital: float = 1.0,
        cap: float = 20.0,
        idm: float = 1.0,
        vol_window: int = 21,
        # weight: float = 1.0,
        tau: float = 0.2,
        multiplier: float = 1.0,
        fx: float = 1.0,
        buffer_fraction: float = 0.10,
) -> pd.DataFrame:
    """
    Trend-breakout strategy with multiple horizons and EWMA smoothing.

    Parameters (must match strategy_config.yaml):
    - df: pd.DataFrame with df['close'] and a DateTimeIndex.
    - horizons: list of lookback periods (e.g., [10,20,40,80,160,320]).
    - ewma_offset: additional span added to each horizon for EWMA.
    - forecast_scalars: optional dict mapping horizon -> scalar to adjust each raw forecast variation.
    - forecast_scale: if provided and non-zero, use this; otherwise auto-scale to target avg abs forecast = 10.
    - capital, idm, weight, tau, ann_std: sizing parameters.
    - multiplier: contract multiplier.
    - fx: FX conversion rate.
    - buffer_fraction: fraction for position rounding buffer.

    Returns:
    - DataFrame (same index) with columns:
        'raw_forecast', 'scaled_forecast', 'capped_forecast', 'position'
    """
    price = df['close'].copy()

    if not hasattr(horizons, '__iter__') or isinstance(horizons, (str, bytes)):
        horizons = [horizons]

    # Annualized volatility (for position sizing), using rolling std of pct changes
    pct_change = price.pct_change().fillna(0.0)
    ann_std = pct_change.rolling(vol_window).std()
    ann_std = ann_std.ffill() * 16  # 252^0.5 ≈ 15.87, rounded ~16 for daily -> annual multiplier
    ann_std.replace({0.0: 1e-8}, inplace=True)


    # 1. Compute smoothed raw forecasts for each horizon
    raw_variations = []
    # Ensure horizons is a list, even if a single int is provided
    if isinstance(horizons, int):
        horizons = [horizons]
    for h in horizons:
        # rolling high, low, mid
        hi = price.rolling(window=h).max()
        lo = price.rolling(window=h).min()
        mid = (hi + lo) / 2.0

        # raw signal
        denom = hi - lo
        raw = 40.0 * (price - mid) / denom
        raw = raw.replace([np.inf, -np.inf], np.nan)

        # EWMA smoothing
        span = h + ewma_offset
        smoothed = raw.ewm(span=span, adjust=False).mean()

        # per-variation forecast scalar
        if forecast_scalars and (h in forecast_scalars):
            smoothed = smoothed * forecast_scalars[h]

        raw_variations.append(smoothed.rename(f"f_{h}"))

    # combine variations (equal weight)
    raw_df = pd.concat(raw_variations, axis=1)
    raw_combined = raw_df.mean(axis=1).rename("raw_forecast")

    # 2. Scale forecast to target avg abs = 10 (unless forecast_scale provided)
    avg_abs = raw_combined.abs().mean()
    auto_scale = 10.0 / (avg_abs if avg_abs != 0 else 1.0)
    scale = forecast_scale if (forecast_scale and forecast_scale != 0.0) else auto_scale
    scaled = (raw_combined * scale).rename("scaled_forecast")

    # 3. Cap forecasts
    capped = scaled.clip(lower=-cap, upper=cap).rename("capped_forecast")

    # 4. Compute unrounded N
    N_unrounded = (
            capped
            * capital
            * idm
            * tau
            / (10.0 * multiplier * price * fx * ann_std)
    )
    N_unrounded = N_unrounded.fillna(0.0)

    # 5. Simple rounding with buffer
    N_round = N_unrounded.round()
    # buffer B (series)
    B = (
            buffer_fraction
            * capital
            * idm
            * tau
            / (multiplier * price * fx * ann_std)
    )

    # 6. Apply buffering rules iteratively
    position = pd.Series(index=df.index, dtype=float)
    prev_pos = 0.0
    for t in df.index:
        N_opt = N_unrounded.loc[t]
        N_r = N_round.loc[t]
        b = B.loc[t]
        lower = N_r - b
        upper = N_r + b

        if prev_pos < lower:
            pos = lower
        elif prev_pos > upper:
            pos = upper
        else:
            pos = prev_pos

        position.loc[t] = pos
        prev_pos = pos

    position = position.fillna(0.0).astype(int).rename("position")

    # # 7. Assemble output
    # out = pd.concat([raw_combined, scaled, capped, position], axis=1)
    # return out

    # return pd.DataFrame({
    #     'price': price,
    #     'price_diff': price_diff,
    #     'sigma_price': sigma_price,
    #     'ewma_short': ewma_s,
    #     'ewma_long': ewma_l,
    #     'raw_forecast': raw,
    #     'scaled_forecast': scaled,
    #     'capped_forecast': capped,
    #     'N_unrounded': N_unrounded,
    #     'position': position
    # })


def trend_breakout(
        df: pd.DataFrame,
        capital,
        horizon: int,
        vol_window: int = 21,
        # forecast_scale: float = 4.1,
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
        'raw_forecast': raw,
        'scaled_forecast': scaled,
        'capped_forecast': capped,
        'N_unrounded': N_unrounded,
        'position': position
    })
