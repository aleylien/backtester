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



def statistical_tests(combined: pd.DataFrame, run_out: str,
                      bootstrap_reps: int = 1000,
                      permutation_reps: int = 1000) -> None:
    """
    1) Bootstrap the OOS returns to bound:
       - mean return
       - log(profit factor)
       - max drawdown
       and compute average drawdown.
    2) Permutation test on the same metrics to estimate p-values.
    """
    # prepare OOS returns
    df = combined.copy()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', drop=True, inplace=True)
    oos = df[df['sample']=='OOS']
    rets = oos['equity'].pct_change().dropna().values
    N    = len(rets)

    # actual metrics
    actual_mean = rets.mean()
    wins = rets[rets>0].sum()
    loss = -rets[rets<0].sum()
    actual_pf   = wins/loss if loss>0 else np.nan
    actual_log_pf = np.log(actual_pf) if actual_pf>0 else np.nan
    cum = np.cumsum(rets)
    dd_actual = np.max(np.maximum.accumulate(cum) - cum)

    # bootstrap distributions
    bs_mean, bs_log_pf, bs_dd = [], [], []
    for _ in range(bootstrap_reps):
        samp = np.random.choice(rets, size=N, replace=True)
        bs_mean.append(samp.mean())
        w = samp[samp>0].sum()
        l = -samp[samp<0].sum()
        pf = w/l if l>0 else np.nan
        bs_log_pf.append(np.log(pf) if pf>0 else np.nan)
        cs = np.cumsum(samp)
        bs_dd.append(np.max(np.maximum.accumulate(cs) - cs))
    bs_mean = np.array(bs_mean)
    bs_log_pf = np.array(bs_log_pf)
    bs_dd = np.array(bs_dd)

    # quantile bounds
    def q(a, p): return np.nanpercentile(a, p*100)
    for name, arr in (('Mean return', bs_mean),
                      ('Log PF',    bs_log_pf),
                      ('Drawdown',  bs_dd)):
        print(f"\n{name} bootstrap quantiles:")
        for p in (0.001, 0.01, 0.05, 0.10):
            print(f"  {p:.3%}: {q(arr, p):.6f}")
    print(f"\nAverage bootstrapped drawdown: {bs_dd.mean():.6f}")

    # permutation p-values
    perm_mean, perm_log_pf, perm_dd = [], [], []
    for _ in range(permutation_reps):
        perm = np.random.permutation(rets)
        perm_mean.append(perm.mean())
        w = perm[perm>0].sum()
        l = -perm[perm<0].sum()
        pf = w/l if l>0 else np.nan
        perm_log_pf.append(np.log(pf) if pf>0 else np.nan)
        cs = np.cumsum(perm)
        perm_dd.append(np.max(np.maximum.accumulate(cs) - cs))
    perm_mean = np.array(perm_mean)
    perm_log_pf = np.array(perm_log_pf)
    perm_dd = np.array(perm_dd)

    p_mean = np.mean(perm_mean >= actual_mean)
    p_log_pf = np.mean(perm_log_pf >= actual_log_pf)
    p_dd = np.mean(perm_dd >= dd_actual)

    print("\nPermutation test p-values:")
    print(f"  Mean return p-value:           {p_mean:.4f}")
    print(f"  Log profit-factor p-value:     {p_log_pf:.4f}")
    print(f"  Drawdown p-value (≥ actual):   {p_dd:.4f}")

    # 3) Save bootstrap & permutation test results to CSV helper to find quantiles
    def q(arr, p): return np.nanpercentile(arr, p*100)

    results = {
        'bootstrap_mean_0.1%':    q(bs_mean,   0.001),
        'bootstrap_mean_1%':      q(bs_mean,   0.01),
        'bootstrap_mean_5%':      q(bs_mean,   0.05),
        'bootstrap_mean_10%':     q(bs_mean,   0.10),
        'bootstrap_log_pf_0.1%':  q(bs_log_pf, 0.001),
        'bootstrap_log_pf_1%':    q(bs_log_pf, 0.01),
        'bootstrap_log_pf_5%':    q(bs_log_pf, 0.05),
        'bootstrap_log_pf_10%':   q(bs_log_pf, 0.10),
        'bootstrap_dd_0.1%':      q(bs_dd,     0.001),
        'bootstrap_dd_1%':        q(bs_dd,     0.01),
        'bootstrap_dd_5%':        q(bs_dd,     0.05),
        'bootstrap_dd_10%':       q(bs_dd,     0.10),
        'avg_bootstrap_dd':       bs_dd.mean(),
        'p_value_mean':           p_mean,
        'p_value_log_pf':         p_log_pf,
        'p_value_dd':             p_dd
    }
    pd.DataFrame([results]).to_csv(
        os.path.join(run_out, 'statistical_tests.csv'),
        index=False
    )
    print(f"Statistical test results saved to {os.path.join(run_out,'statistical_tests.csv')}")

