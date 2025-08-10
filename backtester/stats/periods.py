# backtester/stats/periods.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class PeriodStats:
    period_label: str
    start: pd.Timestamp
    end: pd.Timestamp
    bars: int
    cum_return: float
    vol: float
    max_drawdown: float
    ret_over_vol: float


def _dd(series: pd.Series) -> float:
    """Max drawdown from a cumulative index series."""
    if series.empty:
        return np.nan
    cummax = series.cummax()
    dd = (series / cummax) - 1.0
    return dd.min()


def _periodize(returns: pd.Series, resolution: str) -> pd.core.groupby.SeriesGroupBy:
    """
    resolution: 'monthly' | 'quarterly'
    returns: daily (or finer) arithmetic returns indexed by datetime
    """
    if resolution == "monthly":
        rule = "M"
    elif resolution == "quarterly":
        rule = "Q"
    else:
        raise ValueError("resolution must be 'monthly' or 'quarterly'")
    return returns.resample(rule)


def compute_period_stats(returns: pd.Series, resolution: str = "monthly") -> pd.DataFrame:
    """
    Returns a DataFrame with one row per period with:
      cum_return, start, end, bars, vol, max_drawdown, ret_over_vol
    """
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("returns index must be DatetimeIndex")

    out = []
    for period_end, x in _periodize(returns, resolution):
        if x.empty:
            continue
        start = x.index[0]
        end = x.index[-1]
        bars = x.size
        cum_ret = (1.0 + x).prod() - 1.0
        vol = x.std(ddof=1)
        # Build a within-period equity index for DD
        idx = (1.0 + x).cumprod()
        mdd = _dd(idx)
        rov = np.nan if vol == 0 or np.isnan(vol) else cum_ret / vol
        out.append(PeriodStats(
            period_label=period_end.strftime("%Y-%m") if resolution == "monthly" else f"{period_end.quarter}Q{period_end.year}",
            start=start, end=end, bars=bars,
            cum_return=cum_ret, vol=vol, max_drawdown=mdd, ret_over_vol=rov
        ))

    df = pd.DataFrame([s.__dict__ for s in out])
    if df.empty:
        return df
    # Nice ordering
    return df[["period_label","start","end","bars","cum_return","vol","max_drawdown","ret_over_vol"]]\
             .sort_values("start").reset_index(drop=True)


def top_and_bottom_periods(returns: pd.Series, resolution: str = "monthly", n: int = 3) -> dict:
    """
    Convenience: return top/bottom n periods by cumulative return (desc/asc)
    plus the full table.
    """
    table = compute_period_stats(returns, resolution)
    if table.empty:
        return {"table": table, "top": table, "bottom": table}
    top = table.sort_values("cum_return", ascending=False).head(n).reset_index(drop=True)
    bottom = table.sort_values("cum_return", ascending=True).head(n).reset_index(drop=True)
    return {"table": table, "top": top, "bottom": bottom}
