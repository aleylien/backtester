from __future__ import annotations

import os

import pandas as pd
from typing import Dict, Sequence
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter, MultipleLocator
import matplotlib.dates as mdates

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
    and compute stats via compute_statistics(...).
    """
    if not instruments:
        return {}

    # 1) Align OOS arithmetic returns across instruments
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

    # 4) Build full 'combined' expected by compute_statistics
    #    - position: set to 1 (always “live”) so trade stats logic can run
    #    - pnl: equity difference bar-to-bar
    #    - drawdown: from equity
    #    - bundle: set to 1 (single synthetic bundle)
    eq_ser = equity.copy()
    pnl_ser = eq_ser.diff().fillna(0.0)
    cummax = eq_ser.cummax()
    dd_ser = (cummax - eq_ser) / cummax
    pos_ser = pd.Series(1, index=eq_ser.index, dtype=int)

    combined = pd.DataFrame({
        "date": eq_ser.index,
        "equity": eq_ser.values,
        "pnl": pnl_ser.values,
        "drawdown": dd_ser.values,
        "position": pos_ser.values,
        "returns": port_ret.reindex(eq_ser.index).fillna(0.0).values,
        "sample": "OOS",
        "bundle": 1,
        "strategy": strategy_name,
    })

    # 5A) Call your central stats function
    stats = compute_statistics(combined=combined, run_out=run_out, config=config)

    # 5B) Create an equity curve
    eq = pd.Series(eq_ser.values).dropna().sort_index()
    eq.index = pd.to_datetime(eq_ser.index)
    eq_norm = eq / float(eq.iloc[0])
    eq_rel = eq_norm - 1.0
    rolling_max = eq_norm.cummax()
    dd = (eq_norm / rolling_max) - 1.0

    fig, (ax_top, ax_dd) = plt.subplots(
        2, 1, figsize=(12, 8), dpi=170, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax_top.plot(eq_rel.index, eq_rel.values, linewidth=1.4, label="Portfolio")
    ax_top.set_title(f"{strategy_name} Equity (Rebased to 1.0)")
    ax_top.set_ylabel("Return vs Start")
    ax_top.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax_top.yaxis.set_major_locator(MultipleLocator(0.10))
    ax_top.grid(axis="y", which="major", alpha=0.35, linestyle="--")
    ax_top.xaxis.set_major_locator(mdates.YearLocator())
    ax_top.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax_top.grid(axis="x", which="major", alpha=0.25)
    ax_top.legend(loc="best")
    ax_dd.fill_between(dd.index, dd.values, 0.0, step=None, alpha=0.5)
    ax_dd.set_ylabel("Drawdown")
    ax_dd.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax_dd.grid(axis="y", which="major", alpha=0.35, linestyle="--")
    ax_dd.xaxis.set_major_locator(mdates.YearLocator())
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax_dd.grid(axis="x", which="major", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(run_out, "portfolio_equity_rebased.png"), bbox_inches="tight")
    plt.close(fig)

    # 6) Attach series for downstream consumers (best/worst; plots)
    stats["series"] = {"returns": port_ret, "equity": equity}
    stats["combined"] = combined
    return stats
