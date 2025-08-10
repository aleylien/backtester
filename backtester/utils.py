import pandas as pd
from importlib import import_module
from backtester.pnl_engine import simulate_pnl
import os
import numpy as np
import matplotlib.pyplot as plt
from inspect import signature


def filter_params_for_callable(func, params: dict) -> dict:
    """Keep only kwargs the function can accept (unless it has **kwargs)."""
    try:
        sig = signature(func)
    except (ValueError, TypeError):
        return dict(params)
    # If function has **kwargs, no need to filter
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(params)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in params.items() if k in allowed}


def run_strategy(df: pd.DataFrame, config: dict, **params) -> dict:
    """
    Load and execute a single strategy, then simulate PnL and return key metrics.
    """
    # 1) Dynamically import the strategy module and function
    mod_name  = config['strategy']['module']
    fn_name   = config['strategy']['function']
    strat_mod = import_module(f"backtester.strategies.{mod_name}")
    strat_fn  = getattr(strat_mod, fn_name)

    # 2) Prepare parameters for the strategy function, dropping any internal keys
    strat_params = {k: v for k, v in params.items() if not k.startswith('_')}

    # 3) Inject the per-instrument capital from config into strat_params
    strat_params['capital'] = config.get('capital', 100_000)

    # 4) Generate positions from the strategy
    pos_df = strat_fn(df, **strat_params)

    # 5) Simulate PnL, equity, and drawdown using the generated positions
    pnl_df = simulate_pnl(
        positions=pos_df['position'],
        price=df['close'],
        multiplier=strat_params.get('multiplier', 1.0),
        fx=strat_params.get('fx', 1.0),
        capital=strat_params['capital'],
        commission_usd=strat_params.get('commission_usd', 0.0),
        commission_pct=strat_params.get('commission_pct', 0.0),
        slippage_pct=strat_params.get('slippage_pct', 0.0),
    )

    # 6) Join position diagnostics with PnL results
    wanted = ['pos_value', 'delta_pos', 'cost_usd', 'cost_pct',
              'slip_cost', 'pnl', 'equity', 'drawdown']
    avail = pnl_df.columns.intersection(wanted)
    diag = pos_df.join(pnl_df[avail], how='left')

    # 7) Compute summary metrics
    pnl_series    = diag['pnl']
    equity_series = diag['equity']
    dd_series     = diag['drawdown']

    win  = pnl_series[pnl_series > 0].sum()
    loss = -pnl_series[pnl_series < 0].sum()

    return {
        'pnl':            pnl_series.sum(),
        'sharpe':         (pnl_series.mean()/pnl_series.std()) * (252**0.5)
                          if pnl_series.std() > 0 else 0.0,
        'max_drawdown':   dd_series.min(),
        'profit_factor':  (win / loss) if loss > 0 else float('inf'),
        'start_equity':   equity_series.iloc[0],
        'end_equity':     equity_series.iloc[-1],
    }


def get_periods_per_year(tf: str) -> float:
    """Estimate number of periods per year based on timeframe code (e.g. '1d','1h','5m')."""
    tf = tf.lower()
    if tf.endswith('d'):
        # Assume 252 trading days per year
        num = int(''.join(filter(str.isdigit, tf)) or 1)
        return 252 / num
    if tf.endswith('h'):
        # Assume 24*365 hours (8760) per year for hourly data
        num = int(''.join(filter(str.isdigit, tf)) or 1)
        return 8760 / num
    if tf.endswith('m'):
        # Minutes per year (525600 for 365 days)
        num = int(''.join(filter(str.isdigit, tf)) or 1)
        return 525600 / num
    return 252  # default fallback


def hac_se_for_mean(x: pd.Series, L: int) -> float:
    """
    Newey-West (HAC) standard error of the mean using Bartlett kernel.
    L is the max lag; for a window of W bars use L = W-1 (e.g., 29 for 30-bar).
    """
    x = pd.to_numeric(x, errors="coerce").dropna()
    n = x.size
    if n == 0:
        return np.nan
    xc = x - x.mean()
    gamma0 = (xc * xc).mean()
    var_hat = gamma0
    for k in range(1, min(L, n - 1) + 1):
        gk = (xc.iloc[:-k] * xc.iloc[k:]).mean()
        w = 1.0 - k / (L + 1.0)
        var_hat += 2.0 * w * gk
    return np.sqrt(var_hat / n)


def thirty_day_stats_from_returns(oos_full: pd.DataFrame, lookback: int = 30):
    """
    Compute overlapping 30-bar compounded returns from *daily returns* derived from equity.
    Input df must contain at least 'equity'; optional: 'sample', 'bundle'.
    - If 'sample' present, we filter to OOS.
    - If 'bundle' present, 30-bar windows are computed within each bundle (no crossing).
    Returns dict with mean, mean ± 2σ, and a naive 95% CI (fractions; multiply by 100 when saving as %).
    """
    if oos_full is None or len(oos_full) == 0:
        return {k: np.nan for k in (
            "avg_30d_ret","avg_30d_ret_plus_2std","avg_30d_ret_minus_2std",
            "avg_30d_ret_ci_low","avg_30d_ret_ci_high"
        )}

    df = oos_full.copy()

    # Ensure we have an equity column
    if 'equity' not in df.columns:
        # if df itself is a Series of equity, convert it
        if isinstance(df, pd.Series):
            df = df.to_frame(name='equity')
        else:
            return {k: np.nan for k in (
                "avg_30d_ret","avg_30d_ret_plus_2std","avg_30d_ret_minus_2std",
                "avg_30d_ret_ci_low","avg_30d_ret_ci_high"
            )}

    # Normalize a datetime index (handle 'date' as column or as index name)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.set_index('date')
    else:
        # use existing index
        df.index = pd.to_datetime(df.index, errors='coerce')
    df = df.sort_index()
    df.index.name = 'date'

    # OOS only (if provided)
    if 'sample' in df.columns:
        df = df[df['sample'] == 'OOS']

    # Clean equity to numeric
    eq = pd.to_numeric(df['equity'], errors='coerce').replace([np.inf, -np.inf], np.nan)

    # Need enough points
    if eq.dropna().size < lookback + 1:
        return {k: np.nan for k in (
            "avg_30d_ret","avg_30d_ret_plus_2std","avg_30d_ret_minus_2std",
            "avg_30d_ret_ci_low","avg_30d_ret_ci_high"
        )}

    # Daily simple returns from equity (per bundle if available)
    if 'bundle' in df.columns:
        # align bundle labels with eq index
        bundles = df.loc[eq.index, 'bundle']
        r1 = eq.groupby(bundles).pct_change()
        # 30-bar compounded returns within bundle
        r30 = (
            (1.0 + r1)
            .groupby(bundles)
            .rolling(lookback, min_periods=lookback)
            .apply(np.prod, raw=True)
            .reset_index(level=0, drop=True) - 1.0
        )
    else:
        r1 = eq.pct_change()
        r30 = (1.0 + r1).rolling(lookback, min_periods=lookback).apply(np.prod, raw=True) - 1.0

    r30 = pd.to_numeric(r30, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    n = int(r30.size)
    if n == 0:
        return {k: np.nan for k in (
            "avg_30d_ret","avg_30d_ret_plus_2std","avg_30d_ret_minus_2std",
            "avg_30d_ret_ci_low","avg_30d_ret_ci_high"
        )}

    mean_pm = float(r30.mean())
    std_pm  = float(r30.std(ddof=1))
    # ci_half = 1.96 * std_pm / np.sqrt(n)

    window = 30
    se = hac_se_for_mean(r30, L=window - 1)
    mean_ = r30.mean()
    z = 1.96
    ci_low = mean_ - z * se
    ci_high = mean_ + z * se

    return {
        "avg_30d_ret": mean_pm,
        "avg_30d_ret_plus_2std":  mean_pm + 2.0 * std_pm,
        "avg_30d_ret_minus_2std": mean_pm - 2.0 * std_pm,
        "avg_30d_ret_ci_low":     ci_low,
        "avg_30d_ret_ci_high":    ci_high,
    }


def compute_statistics(combined: pd.DataFrame, run_out: str, config: dict = None) -> dict:
    """
    From the combined diagnostics DataFrame:
     - prints per-bundle & aggregate stats,
     - saves PNGs,
     - writes aggregate metrics to strategy_statistics.csv,
     - returns the agg_stats dict.

    Now supports `config` for timeframe-based annualization via get_periods_per_year.
    """
    # Prepare DataFrame with datetime index
    df = combined.copy()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', drop=True, inplace=True)
    df.index.name = 'date'

    # Determine periods per year from timeframe in config
    timeframe = config.get('data', {}).get('timeframe', '1d') if config else '1d'
    periods = get_periods_per_year(timeframe)

    # Split out-of-sample bars
    oos_full = df[df['sample'] == 'OOS']

    # Only bars with a live position
    oos_pos = oos_full[oos_full['position'] != 0] if 'position' in oos_full.columns else oos_full

    print("\n=== Per-Bundle Performance ===")
    all_trade_pnls = []  # collect PnL of each trade across all bundles for aggregate stats
    for b, grp in oos_full.groupby('bundle'):
        eq = grp['equity']
        ret = eq.pct_change().dropna()
        # Fix CAGR to handle complete loss:
        if len(ret) == 0 or eq.iloc[-1] <= 0:
            cagr = -1.0  # -100% CAGR if no data or equity <= 0 (total loss or worse)
        else:
            cagr = (eq.iloc[-1] / eq.iloc[0]) ** (periods / len(ret)) - 1

        ann_vol = ret.std() * np.sqrt(periods) if ret.std() > 0 else np.nan
        sharpe = (ret.mean() / ret.std() * np.sqrt(periods)) if ret.std() > 0 else np.nan
        neg = ret[ret < 0]
        sortino = (ret.mean() / neg.std() * np.sqrt(periods)) if len(neg) > 0 else np.nan

        # 30-day return
        # robust overlapping 30-bar returns + CI
        eq_ser = pd.Series(eq['equity'] if isinstance(eq, pd.DataFrame) else eq)
        eq_ser.index = pd.to_datetime(eq_ser.index)
        eq_ser = pd.to_numeric(eq_ser, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        eq_ser = eq_ser.sort_index()

        # need at least lookback+1 points for any return
        lookback = 30
        port_monthly = eq_ser.pct_change(lookback)
        port_monthly = pd.to_numeric(port_monthly, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

        n = int(port_monthly.size)
        if n > 0:
            mean_pm = float(port_monthly.mean())
            std_pm = float(port_monthly.std(ddof=1))
            ci_half = 1.96 * std_pm / np.sqrt(n)
            ci_low = mean_pm - ci_half
            ci_high = mean_pm + ci_half
        else:
            mean_pm = std_pm = ci_low = ci_high = np.nan  # or 0.0 if you prefer

        # Drawdown metrics (unchanged)
        dd_vals = grp['drawdown'].dropna()
        max_dd = dd_vals.max()
        avg_dd = dd_vals[dd_vals > 0].mean()
        # Average drawdown duration in bars
        durs = []
        curr = 0
        for x in grp['drawdown']:
            if x > 0:
                curr += 1
            else:
                if curr > 0:
                    durs.append(curr)
                    curr = 0
        if curr > 0:
            durs.append(curr)
        avg_dd_dur = np.mean(durs) if durs else 0

        # **Trade-level performance stats**
        positions = grp['position']
        pnl_series = grp['pnl']
        trade_pnls = []
        current_pnl = 0.0
        last_sign = 0  # track the sign of the last open position (1 for long, -1 for short, 0 for flat)

        for pos, pnl in zip(positions, pnl_series):
            sign = 0 if pos == 0 else (1 if pos > 0 else -1)
            if last_sign == 0 and sign != 0:
                # Trade opens
                last_sign = sign
                current_pnl = 0.0
                current_pnl += pnl  # include any immediate cost on entry
            elif last_sign != 0:
                if sign == 0:
                    # Trade closes
                    current_pnl += pnl  # include PnL on the closing bar
                    trade_pnls.append(current_pnl)
                    last_sign = 0
                    current_pnl = 0.0
                elif sign != last_sign:
                    # Position flipped direction (close old trade and start new back-to-back)
                    # Split the current bar's PnL between the closing trade and the new trade
                    half_pnl = pnl * 0.5
                    current_pnl += half_pnl
                    trade_pnls.append(current_pnl)  # close old trade with half the bar's PnL
                    # Start a new trade with remaining half PnL
                    last_sign = sign
                    current_pnl = half_pnl
                    # (Note: We continue in a trade with the new direction)
                else:
                    # Still in an open trade
                    current_pnl += pnl

        # If a trade is still open at end of bundle, close it at final equity (assume zero PnL beyond last bar)
        if last_sign != 0:
            trade_pnls.append(current_pnl)
        # Add this bundle's trades to the aggregate list
        all_trade_pnls.extend(trade_pnls)

        # Calculate trade-level metrics for this bundle
        total_trades = len(trade_pnls)
        wins_count = sum(1 for p in trade_pnls if p > 0)
        loss_count = sum(1 for p in trade_pnls if p < 0)
        # Win rate = percentage of trades with net profit (ignore breakeven trades)
        win_rate = (wins_count / (wins_count + loss_count)) if (wins_count + loss_count) > 0 else np.nan
        # Profit factor = ratio of total profit to total loss (trade-level)
        total_win = sum(p for p in trade_pnls if p > 0)
        total_loss = sum(p for p in trade_pnls if p < 0)
        profit_factor = (total_win / -total_loss) if loss_count > 0 else (np.inf if wins_count > 0 else np.nan)
        # Expectancy = average PnL per trade
        expectancy = (sum(trade_pnls) / total_trades) if total_trades > 0 else np.nan

        std_dev = ret.std()
        tail_5, tail_95 = np.percentile(ret, [5, 95]) if len(ret) > 0 else (np.nan, np.nan)

        print(
            f"Bundle {b}: CAGR={cagr:.2%}, Vol={ann_vol:.2%}, Sharpe={sharpe:.2f}, "
            f" Mean 30-d return={mean_pm:.2%}, +2std={ci_high:.2%}, -2std={ci_low:.2%}, "
            f"Sortino={sortino:.2f}, MaxDD={max_dd:.2%}, AvgDD={avg_dd:.2%}, "
            f"AvgDDdur={avg_dd_dur:.1f}, PF={profit_factor:.2f}, Exp={expectancy:.2f}, "
            f"WR={win_rate:.1%}, Std={std_dev:.4f}, 5%={tail_5:.2%}, 95%={tail_95:.2%}"
        )

    # --- Aggregate Performance ---
    print("\n=== Aggregate Performance ===")
    # Aggregate daily returns across all bundles (for aggregate Sharpe, Sortino, etc.)
    all_rets = pd.concat([g['equity'].pct_change().dropna() for _, g in oos_full.groupby('bundle')])
    if len(all_rets) == 0 or (oos_full['equity'].iloc[-1] <= 0):
        cagr_a = -1.0
    else:
        cagr_a = (all_rets.add(1).prod()) ** (periods / len(all_rets)) - 1
    ann_vol_a = all_rets.std() * np.sqrt(periods) if all_rets.std() > 0 else np.nan
    sharpe_a = (all_rets.mean() / all_rets.std() * np.sqrt(periods)) if all_rets.std() > 0 else np.nan
    neg_a = all_rets[all_rets < 0]
    sortino_a = (all_rets.mean() / neg_a.std() * np.sqrt(periods)) if len(neg_a) > 0 else np.nan

    max_dd_a = oos_full['drawdown'].max()
    avg_dd_a = oos_full['drawdown'][oos_full['drawdown'] > 0].mean()

    # Aggregate avg drawdown duration (same logic as per-bundle)
    dur_list = []
    for _, grp in oos_full.groupby('bundle'):
        curr = 0
        for x in grp['drawdown']:
            if x > 0:
                curr += 1
            else:
                if curr > 0:
                    dur_list.append(curr)
                    curr = 0
        if curr > 0:
            dur_list.append(curr)
    avg_dd_dur_a = np.mean(dur_list) if dur_list else 0

    tail_5_a, tail_95_a = np.percentile(all_rets, [5, 95]) if len(all_rets) > 0 else (np.nan, np.nan)

    # **Aggregate trade-level stats**
    total_trades_a = len(all_trade_pnls)
    wins_count_a = sum(1 for p in all_trade_pnls if p > 0)
    loss_count_a = sum(1 for p in all_trade_pnls if p < 0)
    win_rate_a = (wins_count_a / (wins_count_a + loss_count_a)) if (wins_count_a + loss_count_a) > 0 else np.nan
    total_win_a = sum(p for p in all_trade_pnls if p > 0)
    total_loss_a = sum(p for p in all_trade_pnls if p < 0)
    pf_a = (total_win_a / -total_loss_a) if loss_count_a > 0 else (np.inf if wins_count_a > 0 else np.nan)
    expectancy_a = (sum(all_trade_pnls) / total_trades_a) if total_trades_a > 0 else np.nan

    std_dev_a = all_rets.std()
    avg_win_a = all_rets[all_rets > 0].mean() if len(all_rets[all_rets > 0]) > 0 else 0.0
    avg_loss_a = all_rets[all_rets < 0].mean() if len(all_rets[all_rets < 0]) > 0 else 0.0
    max_loss_bar_a = all_rets.min() if len(all_rets) > 0 else 0.0

    # 30 day returns
    stats_30 = thirty_day_stats_from_returns(oos_full, lookback=30)
    avg_30d_ret = stats_30["avg_30d_ret"]
    avg_30d_ret_plus_2std = stats_30["avg_30d_ret_plus_2std"]
    avg_30d_ret_minus_2std = stats_30["avg_30d_ret_minus_2std"]
    avg_30d_ret_ci_low = stats_30["avg_30d_ret_ci_low"]
    avg_30d_ret_ci_high = stats_30["avg_30d_ret_ci_high"]
    print(f"30d stats: {avg_30d_ret, avg_30d_ret_minus_2std, avg_30d_ret_plus_2std, avg_30d_ret_ci_low, avg_30d_ret_ci_high}")

    agg_stats = {
        'cagr': cagr_a,
        'annual_vol': ann_vol_a,
        'sharpe': sharpe_a,
        'sortino': sortino_a,
        "avg_30d_ret": avg_30d_ret,
        "avg_30d_ret_plus_2std": avg_30d_ret_plus_2std,
        "avg_30d_ret_minus_2std": avg_30d_ret_minus_2std,
        "avg_30d_ret_ci_low": avg_30d_ret_ci_low,
        "avg_30d_ret_ci_high": avg_30d_ret_ci_high,
        'max_drawdown': max_dd_a,
        'avg_drawdown': avg_dd_a,
        'avg_dd_duration': avg_dd_dur_a,
        'profit_factor': pf_a,
        'expectancy': expectancy_a,
        'win_rate': win_rate_a,
        'std_daily': std_dev_a,
        'ret_5pct': tail_5_a,
        'ret_95pct': tail_95_a,
        'avg_win': avg_win_a if not np.isnan(avg_win_a) else 0.0,
        'avg_loss': avg_loss_a if not np.isnan(avg_loss_a) else 0.0,
        'max_loss_pct': max_loss_bar_a if not np.isnan(max_loss_bar_a) else 0.0
    }
    # Save aggregate CSV
    pd.DataFrame([agg_stats]).to_csv(
        os.path.join(run_out, "strategy_statistics.csv"), index=False
    )

    # Transaction cost metrics
    avg_cost_frac = np.nan
    sharpe_no_cost = np.nan
    if {'cost_usd','cost_pct','slip_cost'}.issubset(oos_full.columns):
        notional = oos_full['price'] * oos_full['delta_pos'].abs()
        total_notional = notional.sum()
        total_cost = oos_full['cost_usd'].sum() + oos_full['cost_pct'].sum() + oos_full['slip_cost'].sum()
        if total_notional>0:
            avg_cost_frac = total_cost/total_notional
        gross_list=[]
        for _, grp in oos_full.groupby('bundle'):
            eq_gross = grp['equity'] + (grp['cost_usd']+grp['cost_pct']+grp['slip_cost']).cumsum()
            gross_list.append(eq_gross.pct_change().dropna())
        all_gross = pd.concat(gross_list)
        if all_gross.std()>0:
            sharpe_no_cost = (all_gross.mean()/all_gross.std()*np.sqrt(periods))
    if not np.isnan(avg_cost_frac):   agg_stats['avg_cost_pct']=avg_cost_frac
    if not np.isnan(sharpe_no_cost): agg_stats['sharpe_no_cost']=sharpe_no_cost

    # --- Charts ---
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
