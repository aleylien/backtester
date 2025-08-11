import pandas as pd
from numpy import inf
from itertools import product
import numpy as np
from backtester.utils import run_strategy
import logging


def generate_splits(df: pd.DataFrame, n_splits: int):
    """
    Sliding window with
      IS_length  = 4*w
      OOS_length = 1*w
      Step       = w

    where w = floor(N / (n_splits + 4)).  That way the last OOS ends at N.
    """
    N = len(df)
    w = N // (n_splits + 4)
    if w < 1:
        raise ValueError(f"Not enough data for {n_splits} bundles")

    for i in range(n_splits):
        start_train = i * w
        end_train   = start_train + 4 * w
        start_test  = end_train
        # last bundle: include any extra bars so we don't drop tail data
        if i == n_splits - 1:
            end_test = N
        else:
            end_test = start_test + w

        is_df  = df.iloc[start_train:end_train]
        oos_df = df.iloc[start_test:end_test]
        yield is_df, oos_df


def optimize_strategy(df, param_grid, target, config):

    best_score = -np.inf
    best_params = None

    # Cartesian product of all grid values
    for vals in product(*(param_grid[k] for k in param_grid)):
        params = dict(zip(param_grid.keys(), vals))
        metrics = run_strategy(df, config, **params)

        # ---- Robust scoring ---------------------------------------------------
        score = None
        try:
            if target == "sharpe":
                score = float(metrics.get("sharpe", -np.inf))

            elif target == "pnl":
                # 1) scalar pnl
                if "pnl" in metrics and metrics["pnl"] is not None:
                    score = float(metrics["pnl"])
                # 2) equity delta
                if score is None and "equity" in metrics:
                    e = np.asarray(metrics["equity"], dtype=float)
                    e = e[np.isfinite(e)]
                    if e.size >= 2:
                        score = float(e[-1] - e[0])
                # 3) compounded return from returns
                if score is None and "returns" in metrics:
                    r = pd.to_numeric(pd.Series(metrics["returns"]), errors="coerce").dropna()
                    if not r.empty:
                        score = float((1.0 + r).prod() - 1.0)
                # 4) sum of pnl_series
                if score is None and "pnl_series" in metrics:
                    p = pd.to_numeric(pd.Series(metrics["pnl_series"]), errors="coerce").dropna()
                    if not p.empty:
                        score = float(p.sum())
                if score is None:
                    logging.warning("Optimization: no pnl/equity/returns in metrics for params %s; skipping", params)
                    continue

            elif target == "profit_factor":
                score = float(metrics.get("profit_factor", -np.inf))

            elif target == "max_drawdown":
                md = metrics.get("max_drawdown", None)
                score = -float(md) if md is not None and np.isfinite(md) else -np.inf

            else:
                raise ValueError(f"Unknown optimization target: {target!r}")
        except Exception as e:
            logging.warning("Optimization scoring failed for params %s: %s", params, e)
            continue
        # ----------------------------------------------------------------------

        if score > best_score:
            best_score, best_params = score, params

    if best_params is None:
        raise RuntimeError("No valid parameter set found for target %r" % target)
    return best_params


def run_backtest(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Walk‐forward backtest:
     1) Split into n bundles
     2) For each bundle:
        a) Optimize on the IS slice
        b) Evaluate chosen params on the OOS slice
        c) Collect OOS metrics (sharpe, pnl, profit_factor, max_drawdown, etc.)
     3) Return one row per bundle with best‐params + oos_<metric>.
    """
    df = df.copy()

    n_bundles = config['optimization']['bundles']
    target    = config['optimization']['target']
    grid      = config['optimization']['param_grid']

    results = []

    for i, (train_df, test_df) in enumerate(generate_splits(df, n_bundles), start=1):
        # Debug
        # print(f"Bundle {i}/{n_bundles}: train={len(train_df):,} bars, test={len(test_df):,} bars")

        # a) find best params on IS
        best_params = optimize_strategy(train_df, grid, target, config)
        # Debug
        # print(f"  → Best params: {best_params}")

        # b) evaluate on OOS
        metrics = run_strategy(test_df, config, **best_params)
        # print(f"  → OOS {target.upper()}: {metrics[target]:.4f}\n")

        # c) record row
        row = {'bundle': i}
        row.update(best_params)
        for k, v in metrics.items():
            row[f"oos_{k}"] = v
        results.append(row)

    return pd.DataFrame(results)
