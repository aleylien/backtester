import numpy as np
import pandas as pd
from typing import Dict, Callable
from backtester.data_loader import DataLoader
from backtester.backtest import run_backtest


np.random.seed(42)


def test1_permutation_oos(
    inst_folder: str,
    B: int = 1000
) -> float:
    """
    Permutation Test 1 on bundle-level OOS:
    1) Read results.csv, detect 'oos_pnl' (or any '*_pnl') as the OOS metric.
    2) Load details_all_bundles.csv, group PnL by bundle.
    3) orig_total = sum of all PnL.
    4) For B reps: permute each bundle’s PnL array internally, sum → perm_total.
       Count how often perm_total >= orig_total.
    5) Return (count+1)/(B+1).
    """
    # 1) load results and find the PnL column
    res = pd.read_csv(f"{inst_folder}/results.csv")
    pnl_cols = [c for c in res.columns if c.lower().endswith('_pnl')]
    if not pnl_cols:
        raise KeyError(f"No '*_pnl' column in {inst_folder}/results.csv; cols={res.columns.tolist()}")
    ret_col = pnl_cols[0]  # e.g. 'oos_pnl'

    # (we don't actually need the best bundle's params here, since we're permuting PnL itself)

    # 2) load the detailed PnL by bundle
    det = pd.read_csv(f"{inst_folder}/details_all_bundles.csv")
    det = det.sort_values(['bundle','date']).reset_index(drop=True)

    # collect each bundle's raw PnL array
    bundle_pnls = [grp['pnl'].values for _, grp in det.groupby('bundle')]

    # 3) original total PnL
    orig_total = det['pnl'].sum()

    # 4) permutation trials
    count = 0
    for _ in range(B):
        perm_sum = 0.0
        for arr in bundle_pnls:
            perm_sum += np.random.permutation(arr).sum()
        if perm_sum >= orig_total:
            count += 1

    return (count + 1) / (B + 1)


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

    # 3) Build result DataFrame
    rows = []
    for name in orig:
        rows.append({
            'system':    name,
            'solo_p':    (solo_counts[name] + 1) / (B + 1),
            'unbiased_p':(best_counts[name] + 1) / (B + 1)
        })
    return pd.DataFrame(rows).set_index('system')


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

