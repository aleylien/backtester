from importlib import import_module
import numpy as np
import pandas as pd
from typing import Dict, Callable
from backtester.data_loader import DataLoader
from backtester.backtest import run_backtest


np.random.seed(42)


def test1_permutation_oos(inst_cfg: dict, inst_folder: str, w, B: int = 1000) -> float:
    """
    Permutation Test 1 on OOS price series:
    1) Read inst_folder/results.csv to get optimal params for each bundle.
    2) Load inst_folder/details_all_bundles.csv to get OOS price series by bundle.
    3) Compute total actual OOS PnL (sum over all bundles).
    4) For B permutations: for each bundle, shuffle the OOS price sequence, run strategy with fixed optimal params on this permuted price, and sum the PnL across bundles.
       Count how many permuted total PnL >= actual total PnL.
    5) Return p-value = (count+1)/(B+1).
    """
    # 1) Load results.csv and extract best params per bundle
    res = pd.read_csv(f"{inst_folder}/results.csv")
    pnl_cols = [c for c in res.columns if c.lower().endswith('_pnl')]
    if not pnl_cols:
        raise KeyError(f"No '*_pnl' column in results.csv; columns={res.columns.tolist()}")
    pnl_col = pnl_cols[0]  # e.g. 'oos_pnl'
    # We assume each row in results corresponds to one bundle (already optimal for that bundle)
    best_params_list = []
    for _, row in res.iterrows():
        params = {k: row[k] for k in res.columns if k not in ('bundle',) and not k.startswith('oos_')}
        # Cast vol_window to int if present (avoid float issues)
        if 'vol_window' in params:
            params['vol_window'] = int(params['vol_window'])
        best_params_list.append(params)
    # 2) Load OOS price series per bundle from details_all_bundles.csv
    det = pd.read_csv(f"{inst_folder}/details_all_bundles.csv")
    if 'sample' in det.columns:
        det = det[det['sample'] == 'OOS']
    det.sort_values(['bundle', 'date'], inplace=True)
    # Group by bundle
    oos_groups = {bundle: grp.copy() for bundle, grp in det.groupby('bundle')}
    # Identify price column
    price_col = 'price' if 'price' in det.columns else 'close'
    # 3) Calculate original total OOS PnL
    orig_total_pnl = 0.0
    for bundle, grp in oos_groups.items():
        # Sum of 'pnl' column for that bundle
        if 'pnl' in grp.columns:
            orig_total_pnl += grp['pnl'].sum()
        else:
            # If details do not have PnL (e.g., find-signal mode was used without per-asset PnL),
            # we can compute it via equity or position changes.
            if 'equity' in grp.columns:
                pnl_series = grp['equity'].diff().fillna(0.0)
                orig_total_pnl += pnl_series.sum()
            else:
                raise KeyError("No PnL or Equity data to compute original PnL for Test 1.")
    # 4) Permutation trials
    strat_mod = import_module(f"backtester.strategies.{inst_cfg['strategy']['module']}")
    strat_fn = getattr(strat_mod, inst_cfg['strategy']['function'])
    count_ge = 0
    for _ in range(B):
        perm_total_pnl = 0.0
        for i, params in enumerate(best_params_list, start=1):
            if i not in oos_groups:
                continue  # skip if no such bundle
            price_series = oos_groups[i][price_col].values
            perm_prices = np.random.permutation(price_series)
            # Build a permuted series under the name 'close' so strategy can find it
            df_perm = pd.DataFrame({'close': perm_prices})
            # Run strategy on permuted price

            params['capital'] = inst_cfg['portfolio']['capital'] * w

            pos_df = strat_fn(df_perm, **params)
            # Simulate PnL for this OOS segment
            if 'position' in pos_df:
                # Compute pnl as price diff * prev position * multiplier * fx (similar to simulate_pnl logic, but simpler for test)
                price_diff = np.diff(perm_prices, prepend=perm_prices[0])
                prev_pos = pos_df['position'].shift(1).fillna(0.0).values
                pnl_arr = price_diff * prev_pos * params.get('multiplier', 1.0) * params.get('fx', 1.0)
                perm_total_pnl += pnl_arr.sum()
            else:
                # If strategy returns PnL or equity directly in pos_df (less likely), use that
                if 'pnl' in pos_df:
                    perm_total_pnl += pos_df['pnl'].sum()
                elif 'equity' in pos_df:
                    perm_total_pnl += (pos_df['equity'].iloc[-1] - pos_df['equity'].iloc[0])
        if perm_total_pnl >= orig_total_pnl:
            count_ge += 1
    p_value = (count_ge + 1) / (B + 1)
    return p_value


def test2_permutation_training(
    inst_cfg: dict,
    inst_folder: str,
    B: int = 1000
) -> float:
    """
    Book Test 2 on in‐sample price:
    1. Read <inst_folder>/results.csv, detect the OOS‐PnL column (e.g. 'oos_pnl')
       and find the best bundle → best_params.
    2. Load raw price series via DataLoader using inst_cfg['data'].
    3. Chop off the last bundle’s worth of bars to get the IS slice.
    4. Run your optimizer (run_backtest) once on that IS to get orig_best.
    5. For each of B perms: shuffle IS 'close', rerun run_backtest(IS_perm, inst_cfg),
       record its best bundle’s OOS‐PnL. Count how often perm_best ≥ orig_best.
    6. Return (count+1)/(B+1).
    """
    # 1) Load results.csv & find the PnL column
    res = pd.read_csv(f"{inst_folder}/results.csv")
    pnl_cols = [c for c in res.columns if c.lower().endswith('_pnl')]
    if not pnl_cols:
        raise KeyError(f"No '*_pnl' column in {inst_folder}/results.csv; cols={res.columns.tolist()}")
    pnl_col = pnl_cols[0]  # e.g. 'oos_pnl'

    # pick best bundle and freeze its params
    best_idx  = res[pnl_col].idxmax()
    best_row  = res.loc[best_idx]
    best_params = {
        k: best_row[k]
        for k in res.columns
        if k not in ('bundle',) and not k.startswith('oos_')
    }
    if 'vol_window' in best_params:
        best_params['vol_window'] = int(best_params['vol_window'])

    # 2) Load raw price DataFrame
    data_cfg = inst_cfg['data']
    dl = DataLoader(
        data_dir       = data_cfg['path'],
        symbol         = data_cfg['symbol'],
        timeframe      = data_cfg['timeframe'],
        base_timeframe = data_cfg.get('base_timeframe')
    )
    df = dl.load().reset_index()  # keep 'date' column

    # 3) Compute IS slice (all but final bundle)
    nb      = inst_cfg['optimization']['bundles']
    n       = len(df)
    size    = n // nb
    is_end  = size * (nb - 1)
    is_df   = df.iloc[:is_end].copy()

    # 4) Original best OOS‐PnL on IS
    orig_res   = run_backtest(is_df.set_index('date'), inst_cfg)
    orig_best  = orig_res[pnl_col].max()

    # 5) Permutation trials
    count = 0
    for _ in range(B):
        perm = is_df['close'].sample(frac=1, replace=False).reset_index(drop=True)
        isp  = is_df.copy()
        isp['close'] = perm

        perm_res  = run_backtest(isp.set_index('date'), inst_cfg)
        perm_best = perm_res[pnl_col].max()
        if perm_best >= orig_best:
            count += 1

    # 6) p-value
    return (count + 1) / (B + 1)


def permutation_test_multiple(
    inst_folders: Dict[str, str],
    B: int = 1000
) -> pd.DataFrame:
    """
    Book Test 5: multiple‐system selection bias.
    inst_folders: dict mapping system name -> folder containing details_all_bundles.csv
    Returns a DataFrame indexed by system with columns [solo_p, unbiased_p].
    """

    # 1) Compute each system's observed OOS return (sum of pnl over all bundles)
    orig = {}
    bundle_pnls = {}
    for name, folder in inst_folders.items():
        det = pd.read_csv(f"{folder}/details_all_bundles.csv")
        # group by bundle, extract each bundle's PnL series
        grouped = det.groupby('bundle')['pnl'].apply(lambda s: s.values)
        bundle_pnls[name] = grouped.tolist()
        orig[name] = det['pnl'].sum()

    solo_counts = {n: 0 for n in orig}
    best_counts = {n: 0 for n in orig}

    # 2) Permutation trials
    for _ in range(B):
        perm_returns = {}
        # a) for each system, permute each bundle internally and re-sum
        for name, bundles in bundle_pnls.items():
            total = 0.0
            for arr in bundles:
                total += np.random.permutation(arr).sum()
            perm_returns[name] = total
            # solo test
            if total >= orig[name]:
                solo_counts[name] += 1

        # b) unbiased: pick the system with highest permuted return
        winner = max(perm_returns, key=perm_returns.get)
        if perm_returns[winner] >= orig[winner]:
            best_counts[winner] += 1

    # 3) Build result DataFrame (explicit 'System' column; no index)
    rows = []
    for name in orig:
        rows.append({
            'System':     name,
            'solo_p':     (solo_counts[name] + 1) / (B + 1),
            'unbiased_p': (best_counts[name] + 1) / (B + 1)
        })
    return pd.DataFrame(rows)


def partition_return(
    backtest_fn: Callable[[pd.DataFrame], float],
    price_data: pd.DataFrame,
    drift_rate: float,
    oos_start: int,
    B: int = 1000
) -> Dict[str, float]:
    """
    Book Test 6: partition total return into trend, bias, skill.

    - trend = (#long bars – #short bars) * drift_rate
    - bias = average over B perms of (r_perm – trend)
    - skill = r_orig – trend – bias
    """
    # 1) original total return
    r_orig = backtest_fn(price_data)

    # 2) compute trend from positions
    long_count  = (price_data['position'] > 0).sum()
    short_count = (price_data['position'] < 0).sum()
    trend = (long_count - short_count) * drift_rate

    # 3) permute only the OOS price changes
    changes = price_data['price_change'].values[oos_start:]
    bias_acc = 0.0
    for _ in range(B):
        perm = np.random.permutation(changes)
        dfp = price_data.copy()
        dfp.loc[price_data.index[oos_start:], 'price_change'] = perm
        r_perm = backtest_fn(dfp)
        bias_acc += (r_perm - trend)

    mean_bias = bias_acc / B
    skill     = r_orig - trend - mean_bias

    return {
        'orig_return': r_orig,
        'trend':       trend,
        'mean_bias':   mean_bias,
        'skill':       skill
    }

