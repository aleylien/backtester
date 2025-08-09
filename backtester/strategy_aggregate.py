import os
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt


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


def aggregate_by_strategy(instrument_runs, out_root: str, make_plots: bool = True):
    """
    instrument_runs: list of dicts with:
      {
        'symbol': str,
        'strategy': str,          # e.g., 'ewmac'
        'alloc_capital': float,   # allocation used for this instrument
        'df': DataFrame           # must include 'date','pnl' and (optionally) 'sample'
      }
    """
    groups = defaultdict(list)
    for r in instrument_runs:
        df = r["df"].copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
        if "sample" in df.columns:
            df = df[df["sample"] == "OOS"]
        if "pnl" not in df.columns:
            continue
        pnl = df["pnl"].fillna(0.0).rename(r["symbol"])
        groups[r["strategy"]].append({"pnl": pnl, "alloc_capital": float(r["alloc_capital"])})

    results = {}
    base_dir = os.path.join(out_root, "strategies")
    os.makedirs(base_dir, exist_ok=True)

    for strat_key, items in groups.items():
        if not items:
            continue
        combined = pd.concat([it["pnl"] for it in items], axis=1).fillna(0.0)
        strat_pnl = combined.sum(axis=1)
        init_cap = sum(it["alloc_capital"] for it in items)
        strat_equity = strat_pnl.cumsum() + init_cap

        strat_dir = os.path.join(base_dir, strat_key)
        os.makedirs(strat_dir, exist_ok=True)

        out = pd.DataFrame({
            "date": strat_equity.index,
            "pnl": strat_pnl.values,
            "equity": strat_equity.values
        })
        out.to_csv(os.path.join(strat_dir, "details_all_bundles.csv"), index=False)

        _safe_stats(strat_pnl, strat_equity).to_csv(os.path.join(strat_dir, "stats.csv"), index=False)

        if make_plots:
            plt.figure(figsize=(7,4))
            plt.plot(strat_equity.index, strat_equity.values, label=strat_key)
            plt.title(f"{strat_key} Equity")
            plt.xlabel("Date"); plt.ylabel("Equity"); plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(strat_dir, "equity.png"))
            plt.close()

        results[strat_key] = {"pnl": strat_pnl, "equity": strat_equity, "init_capital": init_cap}

    return results
