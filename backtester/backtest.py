import pandas as pd
from numpy import inf
from itertools import product
from backtester.utils import run_strategy


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


def optimize_strategy(
    df: pd.DataFrame,
    param_grid: dict,
    target: str,
    config: dict
) -> dict:
    """
    Brute‐force search over param_grid.  For each params dict:
      metrics = run_strategy(df, config, **params)
    then picks the params that maximize/minimize the chosen target.
    """
    best_score  = -inf
    best_params = None

    # Cartesian product of all grid values
    for vals in product(*(param_grid[k] for k in param_grid)):
        params = dict(zip(param_grid.keys(), vals))
        metrics = run_strategy(df, config, **params)

        if target == 'sharpe':
            score = metrics['sharpe']
        elif target == 'pnl':
            score = metrics.get('pnl',
                       metrics['equity'][-1] - metrics['equity'][0])
        elif target == 'profit_factor':
            score = metrics['profit_factor']
        elif target == 'max_drawdown':
            # minimize drawdown ⇒ maximize negative
            score = -metrics['max_drawdown']
        else:
            raise ValueError(f"Unknown optimization target: {target!r}")

        if score > best_score:
            best_score, best_params = score, params

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
    n_bundles = config['optimization']['bundles']
    target    = config['optimization']['target']
    grid      = config['optimization']['param_grid']

    results = []

    for i, (train_df, test_df) in enumerate(generate_splits(df, n_bundles), start=1):
        # print(f"Bundle {i}/{n_bundles}: train={len(train_df):,} bars, test={len(test_df):,} bars")

        # a) find best params on IS
        best_params = optimize_strategy(train_df, grid, target, config)
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
