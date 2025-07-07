import argparse
from copy import deepcopy
import json
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from importlib import import_module

from backtester.config import load_config
from backtester.data_loader import DataLoader
from backtester.backtest import run_backtest, generate_splits
from backtester.pnl_engine import simulate_pnl
from backtester.utils import compute_statistics, statistical_tests


def get_portfolio_weights(instruments):
    """
    Returns
      - weights: list of floats summing to 1.0 (one per instrument)
      - total_capital: sum of each instrument's 'capital' field (or default 100k)
    """
    n = len(instruments)
    if n == 0:
        raise ValueError("No instruments provided to portfolio_weights")

    # equal‐weight for now
    weights = [1.0 / n for _ in instruments]

    # sum up each instrument's capital (default to 100k if absent)
    total_capital = sum(inst.get('capital', 100_000) for inst in instruments)

    return weights, total_capital


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run walk-forward backtests with diagnostics"
    )
    parser.add_argument(
        '-c', '--config', required=True,
        help='Path to config YAML or JSON file'
    )
    parser.add_argument(
        '--log', default='INFO',
        help='Logging level (DEBUG, INFO, WARNING, ERROR)'
    )
    return parser.parse_args()


def run_single_backtest(cfg, comm_usd, comm_pct, slip_pct):
    # load & filter data
    dl = DataLoader(
        data_dir=cfg['data']['path'],
        symbol=cfg['data']['symbol'],
        timeframe=cfg['data']['timeframe'],
        base_timeframe=cfg['data'].get('base_timeframe')
    )
    df = dl.load()
    start, end = cfg['data'].get('start'), cfg['data'].get('end')
    if start or end:
        df = df.loc[start or df.index[0] : end or df.index[-1]]

    # IS/OOS + optimization
    results = run_backtest(df, cfg)
    splits = list(generate_splits(df, cfg['optimization']['bundles']))

    # collect per‐bar diagnostics
    all_diags = []
    strat_mod = import_module(f"backtester.strategies.{cfg['strategy']['module']}")
    strat_fn  = getattr(strat_mod, cfg['strategy']['function'])

    for (_, row), (train_df, test_df) in zip(results.iterrows(), splits):
        bundle = int(row['bundle'])
        params = {
            k: row[k]
            for k in results.columns
            if k not in ('bundle',) and not k.startswith('oos_') and not k.startswith('_')
        }
        if 'vol_window' in params:
            params['vol_window'] = int(params['vol_window'])

        for sample, df_slice in (('IS', train_df), ('OOS', test_df)):
            pos_df = strat_fn(df_slice, **params)
            pnl_df = simulate_pnl(
                positions=pos_df['position'],
                price=df_slice['close'],
                multiplier=params['multiplier'],
                fx=params['fx'],
                capital=params.get('capital', 100_000),
                commission_usd=comm_usd,
                commission_pct=comm_pct,
                slippage_pct=slip_pct,
            )
            diag = pos_df.join(
                pnl_df[['pnl','equity','drawdown','delta_pos','cost_usd','cost_pct','slip_cost']],
                how='left'
            )
            diag['bundle'] = bundle
            diag['sample'] = sample
            all_diags.append(diag)

    combined = pd.concat(all_diags, ignore_index=False)
    combined.index.name = 'date'
    return combined.reset_index()


def generate_summary_md(run_out, cfg):
    def df_to_markdown(df: pd.DataFrame) -> str:
        cols = df.columns.tolist()
        header = "| " + " | ".join(cols) + " |"
        sep    = "| " + " | ".join("---" for _ in cols) + " |"
        rows   = ["| " + " | ".join(str(x) for x in r) + " |" for r in df.values]
        return "\n".join([header, sep] + rows)

    md_lines = [
        f"# Backtest Summary: {os.path.basename(run_out)}",
        f"**Strategy:** `{cfg['strategy']['module']}.{cfg['strategy']['function']}`",
        f"**Run date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ""
    ]

    # Per‐bundle summary
    summary_csv = os.path.join(run_out, 'bundle_summary.csv')
    if os.path.exists(summary_csv):
        df = pd.read_csv(summary_csv)
        md_lines += ["## Per-Bundle Summary", df_to_markdown(df), ""]

    # Strategy‐level stats
    stats_csv = os.path.join(run_out, 'strategy_statistics.csv')
    if os.path.exists(stats_csv):
        df = pd.read_csv(stats_csv)
        md_lines += ["## Aggregate Strategy Statistics", df_to_markdown(df), ""]

    # Embed charts
    for title, fname in [
        ("OOS Equity Curves", "equity_all_bundles.png"),
        ("OOS Drawdowns", "drawdown_all_bundles.png"),
        ("Monthly Return Distribution", "monthly_return_distribution.png"),
        ("Drawdown Distribution", "drawdown_distribution.png"),
        ("Drawdown Duration vs Magnitude", "dd_duration_vs_magnitude.png"),
    ]:
        if os.path.exists(os.path.join(run_out, fname)):
            md_lines += [f"## {title}", f"![{title}]({fname})", ""]

    # Portfolio equity
    pe = os.path.join(run_out, 'portfolio', 'portfolio_equity.png')
    if os.path.exists(pe):
        md_lines += ["## Equal-Weight Portfolio Equity", f"![Portfolio Equity](portfolio/portfolio_equity.png)", ""]

    # write the markdown
    with open(os.path.join(run_out, 'summary.md'), 'w') as f:
        f.write("\n".join(md_lines))
    print(f"Generated summary.md → {run_out}/summary.md")


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log))
    cfg = load_config(args.config)

    # Load fees & slippage mapping
    with open(cfg['fees']['mapping']) as f:
        fees_map = json.load(f)

    # Create run folder
    base_out = cfg['output']['root']
    now      = datetime.now()
    run_name = now.strftime("%H:%M %d.%m.%Y") + f" ({cfg['strategy']['module']})"
    run_out  = os.path.join(base_out, run_name)
    os.makedirs(run_out, exist_ok=True)
    with open(os.path.join(run_out, "config_used.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Running into → {run_out}")

    # Determine whether to create per-asset subfolders
    save_per_asset = bool(cfg['output'].get('save_per_asset', 0))

    # Build list of instruments
    instruments = cfg.get('portfolio', {}).get('instruments', [])
    if not instruments:
        instruments = [{
            'symbol':       cfg['data']['symbol'],
            'fees_mapping': cfg['fees']['mapping'],
            'strategy':     cfg['strategy'],
            'param_grid':   cfg['optimization']['param_grid'],
            'capital':      cfg['optimization'].get('capital', 100_000)
        }]

    # Compute portfolio weights and total capital
    weights, total_capital = get_portfolio_weights(instruments)

    inst_stats     = {}
    portfolio_rets = {}

    # 1) Per‐instrument backtests
    for inst, w in zip(instruments, weights):
        print(f"DEBUG → loop entry for symbol={inst['symbol']}, weight={w}")
        symbol = inst['symbol']
        print(f"\n=== Running backtest for {symbol} ===")
        fee_row = fees_map[symbol]
        comm_usd, comm_pct, slip_pct = (
            fee_row['commission_usd'],
            fee_row['commission_pct'],
            fee_row['slippage_pct'],
        )

        # Instrument‐specific config
        inst_cfg = deepcopy(cfg)
        inst_cfg['data']['symbol']             = symbol
        inst_cfg['fees']['mapping']            = inst['fees_mapping']
        inst_cfg['strategy']                   = inst['strategy']
        inst_cfg['optimization']['param_grid'] = inst['param_grid']

        # Allocate capital slice
        inst_capital = total_capital * w
        inst_cfg['optimization']['param_grid']['capital'] = [inst_capital]

        # Create per‐asset folder
        out_sub = os.path.join(run_out, symbol) if save_per_asset else run_out
        os.makedirs(out_sub, exist_ok=True)

        # Load data
        dl_inst = DataLoader(
            data_dir=inst_cfg['data']['path'],
            symbol=symbol,
            timeframe=inst_cfg['data']['timeframe'],
            base_timeframe=inst_cfg['data'].get('base_timeframe')
        )
        df_inst = dl_inst.load()
        start, end = inst_cfg['data'].get('start'), inst_cfg['data'].get('end')
        if start or end:
            df_inst = df_inst.loc[start or df_inst.index[0] : end or df_inst.index[-1]]
        logging.info(f"{symbol}: {len(df_inst):,} bars loaded")

        # Run walk‐forward backtest
        results_inst = run_backtest(df_inst, inst_cfg)
        results_inst.to_csv(os.path.join(out_sub, 'results.csv'), index=False)

        # Collect OOS diagnostics
        splits_inst = list(generate_splits(df_inst, inst_cfg['optimization']['bundles']))
        diags = []
        strat_mod = import_module(f"backtester.strategies.{inst['strategy']['module']}")
        strat_fn  = getattr(strat_mod, inst['strategy']['function'])
        for (_, row), (_, test_df) in zip(results_inst.iterrows(), splits_inst):
            params = {
                k: row[k]
                for k in results_inst.columns
                if k != 'bundle' and not k.startswith('oos_') and not k.startswith('_')
            }
            if 'vol_window' in params:
                params['vol_window'] = int(params['vol_window'])
            pos_df = strat_fn(test_df, **params)
            pnl_df = simulate_pnl(
                positions     = pos_df['position'],
                price         = test_df['close'],
                multiplier    = params['multiplier'],
                fx            = params['fx'],
                capital       = inst_capital,
                commission_usd= comm_usd,
                commission_pct= comm_pct,
                slippage_pct  = slip_pct,
            )
            diags.append(
                pos_df.join(pnl_df, how='left', rsuffix='_pnl')
                      .assign(bundle=row['bundle'], sample='OOS')
            )

        combined_inst = pd.concat(diags).reset_index()
        combined_inst.to_csv(os.path.join(out_sub, 'details_all_bundles.csv'), index=False)
        print(f"{symbol}: details_all_bundles.csv saved")

        # Plot OOS equity & drawdown
        oos = combined_inst[combined_inst['sample'] == 'OOS']
        for col, title, fname in [
            ('equity',   f"{symbol} OOS Equity Curves", 'equity_all_bundles.png'),
            ('drawdown', f"{symbol} OOS Drawdowns",      'drawdown_all_bundles.png')
        ]:
            fig, ax = plt.subplots()
            for b, grp in oos.groupby('bundle'):
                series = grp[col]
                if col == 'equity':
                    series = series / series.iloc[0]
                ax.plot(np.arange(len(series)), series.values, label=f"B{b}")
            ax.set_title(title)
            ax.set_xlabel("Bar # since OOS start")
            ax.set_ylabel(col.capitalize())
            ax.legend(fontsize='small', ncol=2)
            fig.savefig(os.path.join(out_sub, fname), bbox_inches='tight')
            plt.close(fig)

        # Compute performance stats
        stats = compute_statistics(combined_inst, out_sub)
        inst_stats[symbol] = stats

        # Collect OOS returns
        portfolio_rets[symbol] = oos['pnl'] / inst_capital

        # Prompt for advanced statistical tests
        resp = input(f"Run permutation & bootstrap tests for {symbol}? [y/N]: ").strip().lower()
        if resp.startswith('y'):
            statistical_tests(
                combined_inst, out_sub,
                bootstrap_reps = cfg.get('tests', {}).get('bootstrap_reps', 1000),
                permutation_reps= cfg.get('tests', {}).get('permutation_reps', 1000)
            )
        else:
            print(f"Skipping tests for {symbol}.")

    # 2) Portfolio aggregation
    rets_df     = pd.DataFrame(portfolio_rets).fillna(0.0)
    port_ret    = rets_df.sum(axis=1) / total_capital
    port_equity = (1 + port_ret).cumprod() * total_capital

    out_port = os.path.join(run_out, 'portfolio')
    os.makedirs(out_port, exist_ok=True)

    pd.DataFrame({'equity': port_equity}, index=port_equity.index) \
      .to_csv(os.path.join(out_port, 'portfolio_equity.csv'), index_label='date')
    fig, ax = plt.subplots()
    ax.plot(port_equity.index, port_equity.values)
    ax.set_title("Portfolio Equity")
    ax.set_ylabel("Equity")
    ax.set_xlabel("Date")
    fig.savefig(os.path.join(out_port, 'portfolio_equity.png'), bbox_inches='tight')
    plt.close(fig)
    print("Portfolio equity saved")

    # --- Portfolio 30-bar “month” return distribution with 95% CI ---
    # Build a DataFrame so we can group easily
    df_eq = pd.DataFrame({'equity': port_equity.values}, index=port_equity.index)

    # Assign each row to a 30-bar group
    df_eq['grp'] = np.arange(len(df_eq)) // 30

    # Compute first & last equity per group
    grp_first = df_eq.groupby('grp')['equity'].first()
    grp_last = df_eq.groupby('grp')['equity'].last()

    # Compute group returns
    port_monthly = (grp_last / grp_first - 1).dropna()

    # Stats & CI
    mean_pm = port_monthly.mean()
    std_pm = port_monthly.std()
    ci_low_p = mean_pm - 1.96 * std_pm / np.sqrt(len(port_monthly))
    ci_high_p = mean_pm + 1.96 * std_pm / np.sqrt(len(port_monthly))

    # Plot histogram
    fig, ax = plt.subplots()
    ax.hist(port_monthly, bins=30, density=True, alpha=0.6)
    ax.axvline(ci_low_p, color='red', linestyle='--', label='95% CI')
    ax.axvline(ci_high_p, color='red', linestyle='--')
    ax.set_title("Portfolio 30-Bar Return Distribution with 95% CI")
    ax.set_xlabel("Return")
    ax.set_ylabel("Frequency")
    ax.legend()

    fig.savefig(
        os.path.join(out_port, 'portfolio_30bar_return_distribution.png'),
        bbox_inches='tight'
    )
    plt.close(fig)
    print(
        f"Portfolio 30-bar monthly return distribution chart saved to {out_port}/portfolio_30bar_return_distribution.png")

    # Portfolio diagnostics DataFrame
    df_port = pd.DataFrame({
        'date':      port_equity.index,
        'pnl':       port_ret * total_capital,
        'equity':    port_equity,
        'drawdown':  (port_equity.cummax() - port_equity) / port_equity.cummax(),
        'delta_pos': 0,
        'cost_usd':  0.0,
        'cost_pct':  0.0,
        'slip_cost': 0.0,
        'sample':    'OOS',
        'bundle':    1
    })
    stats = compute_statistics(df_port, out_port)
    inst_stats['Portfolio'] = stats

    resp = input("Run permutation & bootstrap tests for Portfolio? [y/N]: ").strip().lower()
    if resp.startswith('y'):
        statistical_tests(
            df_port, out_port,
            bootstrap_reps = cfg.get('tests', {}).get('bootstrap_reps', 1000),
            permutation_reps= cfg.get('tests', {}).get('permutation_reps', 1000)
        )
    else:
        print("Skipping tests for Portfolio.")

    # 3) Save combined statistics
    df_comb = pd.DataFrame.from_dict(inst_stats, orient='index')
    df_comb.to_csv(os.path.join(run_out, 'combined_stats.csv'))
    print(f"combined_stats.csv saved → {run_out}/combined_stats.csv")

    # 4) Generate summary
    generate_summary_md(run_out, cfg)


if __name__ == '__main__':
    main()
