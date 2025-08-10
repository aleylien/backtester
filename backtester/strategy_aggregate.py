from __future__ import annotations
import pandas as pd
from typing import Dict, Sequence
from strategies.A_weights import get_portfolio_weights
from backtester.utils import compute_statistics


def _safe_stats(pnl: pd.Series, equity: pd.Series) -> pd.DataFrame:
    if pnl.empty or equity.empty:
        return pd.DataFrame([{"cagr": None, "ann_vol": None, "sharpe": None, "max_dd": None}])
    rets = (pnl / equity.shift(1)).dropna()
    ann = (1 + rets.mean())**252 - 1 if len(rets) else None
    vol = (rets.std() * (252 ** 0.5)) if len(rets) else None
    sharpe = (ann / vol) if (ann is not None and vol and vol != 0) else None
    rollmax = equity.cummax()
    dd = (equity - rollmax) / rollmax
    max_dd = dd.min() if len(dd) else None
    return pd.DataFrame([{"cagr": ann, "ann_vol": vol, "sharpe": sharpe, "max_dd": max_dd}])


def aggregate_by_strategy(
    oos_returns_by_symbol: Dict[str, pd.Series],
    strategy_name: str,
    instruments: Sequence[str],
    run_out: str,
    config: dict,
    initial_capital: float = 1_000_000.0,
) -> dict:
    """
    Build a synthetic portfolio from all instruments using `strategy_name`
    and compute stats via the central compute_statistics(...).
    Returns stats dict and attaches 'series' with 'returns' and 'equity'.
    """
    if not instruments:
        return {}

    # 1) Align OOS returns for the instruments in this strategy
    df = pd.DataFrame({sym: oos_returns_by_symbol[sym] for sym in instruments}).sort_index()
    df = df.dropna(how="all")
    if df.empty:
        return {}

    # 2) Weights
    w = pd.Series(get_portfolio_weights(config, instruments))
    if w.sum() == 0:
        w[:] = 1.0 / len(w)
    else:
        w = w / w.sum()

    # 3) Synthetic portfolio returns & equity
    port_ret = (df.mul(w, axis=1)).sum(axis=1).dropna()
    equity = (1.0 + port_ret).cumprod() * float(initial_capital)

    # 4) Build the 'combined' DataFrame for compute_statistics
    combined = pd.DataFrame({
        "date": port_ret.index,
        "returns": port_ret.values,
        "equity": equity.values,
        "sample": "OOS",         # mark these rows as OOS
        "strategy": strategy_name,
    })

    # 5) Call your central stats function
    stats = compute_statistics(combined=combined, run_out=run_out, config=config)

    # 6) Attach the series for other consumers (best/worst; plots)
    stats["series"] = {"returns": port_ret, "equity": equity}
    stats["combined"] = combined
    return stats
