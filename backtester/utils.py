import pandas as pd
from importlib import import_module
from backtester.pnl_engine import simulate_pnl
import os
import numpy as np
import matplotlib.pyplot as plt


def run_strategy(df: pd.DataFrame, config: dict, **params) -> dict:
    # load strategy fn
    mod_name  = config['strategy']['module']
    fn_name   = config['strategy']['function']
    strat_mod = import_module(f"backtester.strategies.{mod_name}")
    strat_fn  = getattr(strat_mod, fn_name)

    # drop internal keys
    strat_params = {k: v for k, v in params.items() if not k.startswith('_')}

    # 1) positions
    pos_df = strat_fn(df, **strat_params)

    # 2) simulate pnl/equity/drawdown
    pnl_df = simulate_pnl(
        positions=pos_df['position'],
        price=df['close'],
        multiplier=strat_params.get('multiplier', 1.0),
        fx=strat_params.get('fx', 1.0),
        capital=strat_params.get('capital', 100_000),
        commission_usd=strat_params.get('commission_usd', 0.0),
        commission_pct=strat_params.get('commission_pct', 0.0),
        slippage_pct=strat_params.get('slippage_pct', 0.0),
    )

    # 3) join only the PnL columns that exist
    wanted = [
        'pos_value',
        'delta_pos',
        'cost_usd',
        'cost_pct',
        'slip_cost',
        'pnl',
        'equity',
        'drawdown',
    ]
    avail = pnl_df.columns.intersection(wanted)
    diag = pos_df.join(pnl_df[avail], how='left')

    # 4) compute summary metrics
    pnl_series    = diag['pnl']
    equity_series = diag['equity']
    dd_series     = diag['drawdown']

    win    = pnl_series[pnl_series>0].sum()
    loss   = -pnl_series[pnl_series<0].sum()

    return {
        'pnl':            pnl_series.sum(),
        'sharpe':         (pnl_series.mean()/pnl_series.std()) * (252**0.5)
                          if pnl_series.std()>0 else 0.0,
        'max_drawdown':   dd_series.min(),
        'profit_factor':  (win/loss) if loss>0 else float('inf'),
        'start_equity':   equity_series.iloc[0],
        'end_equity':     equity_series.iloc[-1],
    }


def compute_statistics(combined: pd.DataFrame, run_out: str) -> dict:
    """
    From the combined diagnostics DataFrame:
     - prints per-bundle & aggregate stats,
     - saves three PNGs,
     - writes aggregate metrics to strategy_statistics.csv,
     - returns the agg_stats dict.
    """
    # 1) Prepare DataFrame with datetime index
    df = combined.copy()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', drop=True, inplace=True)
    df.index.name = 'date'

    # 2) Split out-of-sample bars
    oos_full = df[df['sample'] == 'OOS']

    # 2b) And only those bars with a live position
    if 'position' in oos_full.columns:
        oos_pos = oos_full[oos_full['position'] != 0]
    else:
        oos_pos = oos_full

    # --- Per-bundle printout (unchanged, but now using oos_full) ---
    print("\n=== Per-Bundle Performance ===")
    for b, grp in oos_full.groupby('bundle'):
        eq        = grp['equity']
        ret       = eq.pct_change().dropna()
        cagr      = (eq.iloc[-1]/eq.iloc[0]) ** (252/len(ret)) - 1
        ann_vol   = ret.std() * np.sqrt(252)
        sharpe    = (ret.mean()/ret.std()*np.sqrt(252)) if ret.std()>0 else np.nan
        neg       = ret[ret<0]
        sortino   = (ret.mean()/neg.std()*np.sqrt(252)) if len(neg)>0 else np.nan

        dd_vals   = grp['drawdown'].dropna()
        max_dd    = dd_vals.max()
        avg_dd    = dd_vals[dd_vals>0].mean()

        # drawdown durations
        durs = []
        curr = 0
        for x in grp['drawdown']:
            if x>0:
                curr += 1
            else:
                if curr>0:
                    durs.append(curr)
                    curr = 0
        if curr>0:
            durs.append(curr)
        avg_dd_dur = np.mean(durs) if durs else 0

        # trade-based metrics only on bars with position != 0
        trades    = grp['delta_pos'].abs().gt(0).sum()
        pnl_sum   = grp['pnl'].sum()
        wins      = grp['pnl'][grp['pnl']>0].sum()
        loss      = -grp['pnl'][grp['pnl']<0].sum()
        pf        = (wins/loss) if loss>0 else np.nan
        expectancy= pnl_sum/trades if trades>0 else np.nan
        win_rate  = grp['pnl'].gt(0).sum()/trades if trades>0 else np.nan

        std_dev   = ret.std()
        tail_5, tail_95 = np.percentile(ret, [5, 95])

        print(
            f"Bundle {b}: CAGR={cagr:.2%}, Vol={ann_vol:.2%}, Sharpe={sharpe:.2f}, "
            f"Sortino={sortino:.2f}, MaxDD={max_dd:.2%}, AvgDD={avg_dd:.2%}, "
            f"AvgDDdur={avg_dd_dur:.1f}, PF={pf:.2f}, Exp={expectancy:.2f}, "
            f"WR={win_rate:.1%}, Std={std_dev:.4f}, "
            f"5%={tail_5:.2%}, 95%={tail_95:.2%}"
        )

    # --- Aggregate Performance ---
    print("\n=== Aggregate Performance ===")

    # equity‐return series from all bundles
    all_rets = pd.concat([
        g['equity'].pct_change().dropna()
        for _, g in oos_full.groupby('bundle')
    ])

    # equity‐based metrics on oos_full / all_rets
    cagr_a       = (all_rets.add(1).prod()) ** (252/len(all_rets)) - 1
    ann_vol_a    = all_rets.std() * np.sqrt(252)
    sharpe_a     = (all_rets.mean()/all_rets.std()*np.sqrt(252)) if all_rets.std()>0 else np.nan
    neg_a        = all_rets[all_rets<0]
    sortino_a    = (all_rets.mean()/neg_a.std()*np.sqrt(252)) if len(neg_a)>0 else np.nan
    max_dd_a     = oos_full['drawdown'].max()
    avg_dd_a     = oos_full['drawdown'][oos_full['drawdown']>0].mean()

    # aggregate drawdown durations
    dur_list = []
    for _, grp in oos_full.groupby('bundle'):
        curr = 0
        for x in grp['drawdown']:
            if x>0:
                curr += 1
            else:
                if curr>0:
                    dur_list.append(curr)
                    curr = 0
        if curr>0:
            dur_list.append(curr)
    avg_dd_dur_a = np.mean(dur_list) if dur_list else 0

    # return‐percentiles
    tail_5_a, tail_95_a = np.percentile(all_rets, [5, 95])

    # trade‐based metrics on oos_pos only
    trades_a     = oos_pos['delta_pos'].abs().gt(0).sum()
    wins_a       = oos_pos['pnl'][oos_pos['pnl']>0].sum()
    loss_a       = -oos_pos['pnl'][oos_pos['pnl']<0].sum()
    pf_a         = (wins_a/loss_a) if loss_a>0 else np.nan
    expectancy_a = oos_pos['pnl'].sum()/trades_a if trades_a>0 else np.nan
    win_rate_a   = oos_pos['pnl'].gt(0).sum()/trades_a if trades_a>0 else np.nan

    std_dev_a    = all_rets.std()

    print(
        f"Aggregate: CAGR={cagr_a:.2%}, Vol={ann_vol_a:.2%}, Sharpe={sharpe_a:.2f}, "
        f"Sortino={sortino_a:.2f}, MaxDD={max_dd_a:.2%}, AvgDD={avg_dd_a:.2%}, "
        f"AvgDDdur={avg_dd_dur_a:.1f}, PF={pf_a:.2f}, Exp={expectancy_a:.2f}, "
        f"WR={win_rate_a:.1%}, Std={std_dev_a:.4f}, "
        f"5%={tail_5_a:.2%}, 95%={tail_95_a:.2%}"
    )

    # --- Save aggregate stats to CSV (all metrics!) ---
    agg_stats = {
        'cagr':           cagr_a,
        'annual_vol':     ann_vol_a,
        'sharpe':         sharpe_a,
        'sortino':        sortino_a,
        'max_drawdown':   max_dd_a,
        'avg_drawdown':   avg_dd_a,
        'avg_dd_duration':avg_dd_dur_a,
        'pf':             pf_a,
        'expectancy':     expectancy_a,
        'win_rate':       win_rate_a,
        'std_daily':      std_dev_a,
        'ret_5pct':       tail_5_a,
        'ret_95pct':      tail_95_a,
    }
    pd.DataFrame([agg_stats]).to_csv(
        os.path.join(run_out, "strategy_statistics.csv"), index=False
    )
    print(f"Strategy statistics saved to {os.path.join(run_out, 'strategy_statistics.csv')}")

    # --- Charts (unchanged) ---
    # drawdown distribution
    dd_vals = oos_full['drawdown'].dropna()
    fig, ax = plt.subplots()
    ax.hist(dd_vals, bins=50, edgecolor='black')
    ax.set_title("Distribution of OOS Drawdowns")
    ax.set_xlabel("Drawdown")
    ax.set_ylabel("Frequency")
    fp2 = os.path.join(run_out, "drawdown_distribution.png")
    fig.savefig(fp2, bbox_inches='tight')
    plt.close(fig)

    # drawdown duration vs magnitude
    dur_mag = []
    for _, grp in oos_full.groupby('bundle'):
        curr, mx = 0, 0
        for x in grp['drawdown']:
            if x > 0:
                curr += 1
                mx = max(mx, x)
            else:
                if curr > 0:
                    dur_mag.append((curr, mx))
                curr, mx = 0, 0
        if curr > 0:
            dur_mag.append((curr, mx))
    durs, mags = zip(*dur_mag) if dur_mag else ([], [])
    fig, ax = plt.subplots()
    ax.scatter(durs, mags)
    ax.set_title("Drawdown Duration vs Magnitude")
    ax.set_xlabel("Duration (bars)")
    ax.set_ylabel("Drawdown")
    fp3 = os.path.join(run_out, "dd_duration_vs_magnitude.png")
    fig.savefig(fp3, bbox_inches='tight')
    plt.close(fig)

    return agg_stats


def statistical_tests(
    combined: pd.DataFrame,
    run_out: str,
    bootstrap_reps: int = 1000,
    permutation_reps: int = 1000
) -> None:
    import numpy as np, pandas as pd, os

    # 1) Prep OOS P&L returns
    df = combined.copy()
    if 'date' in df:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', drop=True, inplace=True)
    oos = df[df['sample']=='OOS']
    cap = oos['equity'].iloc[0]
    if 'pnl' in oos:
        ret_series = oos['pnl'] / cap
    else:
        ret_series = oos['equity'].pct_change().dropna()
    # drop zero-PnL bars if any
    ret_series = ret_series[ret_series != 0]
    rets = ret_series.values
    # print(f"Statistical tests: {len(rets)} non-zero return bars")

    # 2) Build equity & drawdown
    eq = (1 + ret_series).cumprod() * cap
    drawdown = (eq.cummax() - eq) / eq.cummax()
    actual_dd = drawdown.max()

    # 3) Actual metrics
    actual_mean   = rets.mean()
    wins          = rets[rets>0].sum()
    loss          = -rets[rets<0].sum()
    actual_pf     = wins/loss if loss>0 else np.nan
    actual_log_pf = np.log(actual_pf) if actual_pf>0 else np.nan

    N = len(rets)

    # 4) Bootstrap distributions
    bs_means, bs_log_pfs, bs_dds = [], [], []
    for _ in range(bootstrap_reps):
        samp = np.random.choice(rets, N, replace=True)
        bs_means.append(samp.mean())
        w = samp[samp>0].sum(); l = -samp[samp<0].sum()
        pf = w/l if l>0 else np.nan
        bs_log_pfs.append(np.log(pf) if pf>0 else np.nan)
        # build a proper pandas Series before cummax
        eq_s = pd.Series(samp).add(1).cumprod().mul(cap)
        dd_samp = (eq_s.cummax() - eq_s) / eq_s.cummax()
        bs_dds.append(dd_samp.max())

    # 5) Permutation distributions
    perm_means, perm_log_pfs, perm_dds = [], [], []
    for _ in range(permutation_reps):
        perm = np.random.permutation(rets)
        perm_means.append(perm.mean())
        w = perm[perm>0].sum(); l = -perm[perm<0].sum()
        pf = w/l if l>0 else np.nan
        perm_log_pfs.append(np.log(pf) if pf>0 else np.nan)
        # same here: ensure a pandas Series for cummax
        eq_p = pd.Series(perm).add(1).cumprod().mul(cap)
        dd_perm = (eq_p.cummax() - eq_p) / eq_p.cummax()
        perm_dds.append(dd_perm.max())

    # — after collecting bs_means, bs_log_pfs, bs_dds and perm_dds —

    # (A) Ensure enough data
    if len(rets) < 5:
        print(f"Only {len(rets)} non-zero returns; skipping statistical tests.")
        return

    # (B) Quantiles to report (both tails)
    quantiles = [0.001, 0.01, 0.05, 0.10, 0.90, 0.95, 0.99]
    labels    = [f"{q*100:.1f}%" for q in quantiles]

    # (C) Build summary dict
    summary = {
        'actual_mean':       actual_mean,
        'actual_log_pf':     actual_log_pf,
        'actual_drawdown':   actual_dd,
        'num_nonzero_rets':  len(rets),
    }

    # Bootstrap quantiles
    for q, lbl in zip(quantiles, labels):
        summary[f"bs_mean_{lbl}"]    = np.quantile(bs_means,   q)
        summary[f"bs_log_pf_{lbl}"]  = np.quantile(bs_log_pfs, q)
        summary[f"bs_dd_{lbl}"]      = np.quantile(bs_dds,     q)

    # Permutation quantiles (only for drawdown)
    for q, lbl in zip(quantiles, labels):
        summary[f"perm_dd_{lbl}"]    = np.quantile(perm_dds,   q)

    # Averages
    summary['avg_bs_dd']    = np.mean(bs_dds)
    summary['avg_perm_dd']  = np.mean(perm_dds)

    # P-values
    summary['p_two_sided_mean']       = np.mean(np.abs(bs_means)   >= abs(actual_mean))
    summary['p_two_sided_log_pf']     = np.mean(np.abs(bs_log_pfs) >= abs(actual_log_pf))
    summary['p_one_sided_drawdown']   = np.mean(np.array(perm_dds) <= actual_dd)

    # (D) Write out
    pd.DataFrame([summary]).to_csv(
        os.path.join(run_out, "statistical_tests.csv"),
        index=False
    )
    print(f"Statistical tests saved to {run_out}/statistical_tests.csv")
