import pandas as pd
from sklearn.model_selection import ParameterGrid
from backtester.utils import run_strategy


def generate_splits(df, n_splits):
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


def optimize_strategy(train_df: pd.DataFrame, param_grid: dict, target: str):
    """
    Loop through param_grid, run your strategy on train_df,
    and return the params that best optimize `target` (e.g. min drawdown).
    Stub: replace `run_strategy` with your actual strategy call.
    """
    best = None
    best_score = float('inf') if target == 'max_drawdown' else float('-inf')
    for params in ParameterGrid(param_grid):
        # TODO: implement your strategy logic returning a dict of metrics
        metrics = run_strategy(train_df, **params)
        score = metrics[target]
        if (target == 'max_drawdown' and score < best_score) or \
           (target != 'max_drawdown' and score > best_score):
            best_score = score
            best = params
    return best


def run_backtest(df: pd.DataFrame, config: dict):
    n_bundles = config['optimization']['bundles']
    target    = config['optimization']['target']

    results = []

    for i, (train_df, test_df) in enumerate(generate_splits(df, n_bundles), start=1):
        print(f"Bundle {i}/{n_bundles}: "
              f"train={len(train_df):,} bars, "
              f"test={len(test_df):,} bars")

        best_params = optimize_strategy(train_df, config['optimization']['param_grid'], target)
        print(f"  → Best params: {best_params}")

        oos = run_strategy(test_df, **best_params)
        print(f"  → OOS {target.upper()}: {oos[target]:.4f}\n")

        row = {'bundle': i, **best_params, **{f"oos_{k}": v for k,v in oos.items()}}
        results.append(row)

    return pd.DataFrame(results)

