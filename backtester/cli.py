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
    # equal‐weight for now
    weights = [1.0/n] * n
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
    # load and filter data
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

    # run optimization
    results = run_backtest(df, cfg)

    # gather diagnostics
    splits = list(generate_splits(df, cfg['optimization']['bundles']))
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
                capital=params.get('capital',100_000),
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
        # header
        header = "| " + " | ".join(cols) + " |"
        # separator
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        # rows
        rows = []
        for _, row in df.iterrows():
            rows.append("| " + " | ".join(str(x) for x in row.tolist()) + " |")
        return "\n".join([header, sep] + rows)

    md_lines = [
        f"# Backtest Summary: {os.path.basename(run_out)}",
        f"**Strategy:** `{cfg['strategy']['module']}.{cfg['strategy']['function']}`",
        f"**Run date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ""
    ]

    # Per-Bundle Summary
    summary_csv = os.path.join(run_out, 'bundle_summary.csv')
    if os.path.exists(summary_csv):
        df = pd.read_csv(summary_csv)
        md_lines += ["## Per-Bundle Summary", df_to_markdown(df), ""]

    # Aggregate Strategy Statistics
    stats_csv = os.path.join(run_out, 'strategy_statistics.csv')
    if os.path.exists(stats_csv):
        df = pd.read_csv(stats_csv)
        md_lines += ["## Aggregate Strategy Statistics", df_to_markdown(df), ""]

    # Images
    for title, fname in [
        ("OOS Equity Curves",         "equity_all_bundles.png"),
        ("OOS Drawdowns",             "drawdown_all_bundles.png"),
        ("Monthly Return Distribution","monthly_return_distribution.png"),
        ("Drawdown Distribution",     "drawdown_distribution.png"),
        ("Drawdown Duration vs Magnitude","dd_duration_vs_magnitude.png")
    ]:
        if os.path.exists(os.path.join(run_out, fname)):
            md_lines += [f"## {title}", f"![{title}]({fname})", ""]

    # Portfolio
    if 'portfolio' in cfg:
        pe = os.path.join(run_out, 'portfolio_equity.png')
        if os.path.exists(pe):
            md_lines += ["## Equal-Weight Portfolio Equity", f"![Portfolio Equity](portfolio_equity.png)", ""]

    # write out
    with open(os.path.join(run_out, 'summary.md'), 'w') as f:
        f.write("\n".join(md_lines))

    print(f"Generated summary.md → {os.path.join(run_out,'summary.md')}")


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log))
    cfg = load_config(args.config)

    # --- load fees mapping once ---
    with open(cfg['fees']['mapping']) as f:
        fees_map = json.load(f)

    # --- create run folder ---
    base_out = cfg['output']['root']
    now      = datetime.now()
    run_name = now.strftime("%H:%M %d.%m.%Y") + f" ({cfg['strategy']['module']})"
    run_out  = os.path.join(base_out, run_name)
    os.makedirs(run_out, exist_ok=True)
    with open(os.path.join(run_out,"config_used.json"),"w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Running into → {run_out}")

    save_per_asset = bool(cfg['output'].get('save_per_asset', 0))

    # --- determine instruments & weights ---
    instruments = cfg.get('portfolio',{}).get('instruments')
    if instruments:
        weights, total_capital = get_portfolio_weights(instruments)
    else:
        instruments = [{
            'symbol':       cfg['data']['symbol'],
            'fees_mapping': cfg['fees']['mapping'],
            'strategy':     cfg['strategy'],
            'param_grid':   cfg['optimization']['param_grid'],
            'capital':      cfg['optimization'].get('capital',100_000)
        }]
        weights, total_capital = [1.0], instruments[0]['capital']

    # collect stats per instrument
    inst_stats = {}
    rets       = {}

    # --- 1) Per‐asset backtests ---
    for idx, inst in enumerate(instruments):
        symbol       = inst['symbol']
        fee_row      = fees_map[symbol]
        comm_usd     = fee_row['commission_usd']
        comm_pct     = fee_row['commission_pct']
        slip_pct     = fee_row['slippage_pct']

        # prepare instrument config
        inst_cfg = deepcopy(cfg)
        inst_cfg['data']['symbol']            = symbol
        inst_cfg['fees']['mapping']           = inst['fees_mapping']
        inst_cfg['strategy']                  = inst['strategy']
        inst_cfg['optimization']['param_grid']= inst['param_grid']

        # split total capital
        inst_capital = total_capital * weights[idx]
        # force capital into param grid for optimization
        inst_cfg['optimization']['param_grid']['capital'] = [inst_capital]

        # create output subfolder
        subfolder = symbol
        out_sub   = os.path.join(run_out, subfolder)
        os.makedirs(out_sub, exist_ok=True)

        # load data
        dl = DataLoader(
            data_dir=inst_cfg['data']['path'],
            symbol=inst_cfg['data']['symbol'],
            timeframe=inst_cfg['data']['timeframe'],
            base_timeframe=inst_cfg['data'].get('base_timeframe')
        )
        df = dl.load()
        # date filtering
        start, end = inst_cfg['data'].get('start'), inst_cfg['data'].get('end')
        if start or end:
            df = df.loc[start or df.index[0] : end or df.index[-1]]
        logging.info(f"{symbol}: {len(df):,} bars loaded")

        # run IS/OOS optimization & backtest
        results = run_backtest(df, inst_cfg)

        # bundle summary CSV
        df_sum = results.set_index('bundle')
        cols   = [c for c in ['train_bars','test_bars'] if c in df_sum.columns]
        cols  += [c for c in df_sum.columns if not c.startswith('oos_')]
        cols  += [c for c in df_sum.columns if c.startswith('oos_')]
        df_sum = df_sum[cols]
        df_sum.to_csv(os.path.join(out_sub,'bundle_summary.csv'))
        print(f"{symbol}: bundle_summary.csv saved")

        # diagnostics + charts via run_single_backtest
        combined = run_single_backtest(inst_cfg, comm_usd, comm_pct, slip_pct)
        combined.to_csv(os.path.join(out_sub,'details_all_bundles.csv'), index=False)
        print(f"{symbol}: details_all_bundles.csv saved")

        # plots
        oos = combined[combined['sample']=='OOS']
        # equity
        fig,ax = plt.subplots()
        for b,grp in oos.groupby('bundle'):
            eq = grp['equity']/grp['equity'].iloc[0]
            ax.plot(np.arange(len(eq)), eq.values, label=f"B{b}")
        ax.set_title(f"{symbol} OOS Equity Curves")
        ax.set_ylabel("Equity (start=1.0)")
        ax.set_xlabel("Bar # since OOS start")
        ax.legend(fontsize='small', ncol=2)
        fig.savefig(os.path.join(out_sub,'equity_all_bundles.png'),bbox_inches='tight')
        plt.close(fig)

        # drawdown
        fig,ax = plt.subplots()
        for b,grp in oos.groupby('bundle'):
            dd = grp['drawdown']
            ax.plot(np.arange(len(dd)), dd.values, label=f"B{b}")
        ax.set_title(f"{symbol} OOS Drawdowns")
        ax.set_ylabel("Drawdown")
        ax.set_xlabel("Bar # since OOS start")
        ax.legend(fontsize='small', ncol=2)
        fig.savefig(os.path.join(out_sub,'drawdown_all_bundles.png'),bbox_inches='tight')
        plt.close(fig)

        # compute & save stats
        stats = compute_statistics(combined, out_sub)
        inst_stats[symbol] = stats

        # collect returns for portfolio
        rets[symbol] = oos['pnl'] / inst_capital

        # optional per‐asset tests
        if cfg.get('tests'):
            resp = input(f"Run permutation & bootstrap tests for {symbol}? [y/N]: ").strip().lower()
            if resp.startswith('y'):
                statistical_tests(
                    combined, out_sub,
                    bootstrap_reps=cfg['tests']['bootstrap_reps'],
                    permutation_reps=cfg['tests']['permutation_reps']
                )
            else:
                print(f"Skipping tests for {symbol}.")

    # --- 2) Portfolio aggregation ---
    # build portfolio equity
    rets_df     = pd.DataFrame(rets).fillna(0.0)
    port_ret    = (rets_df * weights).sum(axis=1)
    port_equity = (1 + port_ret).cumprod() * total_capital

    # portfolio subfolder
    out_port = os.path.join(run_out,'portfolio')
    os.makedirs(out_port, exist_ok=True)

    # save equity CSV & chart
    pd.DataFrame({'equity':port_equity}, index=port_equity.index) \
      .to_csv(os.path.join(out_port,'portfolio_equity.csv'))
    fig,ax = plt.subplots()
    ax.plot(port_equity.index, port_equity.values)
    ax.set_title("Portfolio Equity")
    ax.set_ylabel("Equity")
    ax.set_xlabel("Date")
    fig.savefig(os.path.join(out_port,'portfolio_equity.png'),bbox_inches='tight')
    plt.close(fig)
    print("Portfolio equity saved")

    # portfolio diagnostics DataFrame
    df_port = pd.DataFrame({
        'date':     port_equity.index,
        'pnl':      port_ret * total_capital,
        'equity':   port_equity,
        'drawdown': (port_equity.cummax()-port_equity)/port_equity.cummax(),
        'delta_pos': 0,   # no per-bar trades here
        'sample':   'OOS',
        'bundle':   1
    })
    stats = compute_statistics(df_port, out_port)
    inst_stats['Portfolio'] = stats

    # optional portfolio tests
    if cfg.get('tests'):
        resp = input("Run permutation & bootstrap tests for Portfolio? [y/N]: ").strip().lower()
        if resp.startswith('y'):
            statistical_tests(
                df_port, out_port,
                bootstrap_reps=cfg['tests']['bootstrap_reps'],
                permutation_reps=cfg['tests']['permutation_reps']
            )
        else:
            print("Skipping tests for Portfolio.")

    # --- 3) write combined_stats.csv ---
    df_comb = pd.DataFrame.from_dict(inst_stats, orient='index')
    df_comb.to_csv(os.path.join(run_out,'combined_stats.csv'))
    print(f"combined_stats.csv saved → {run_out}/combined_stats.csv")


if __name__ == '__main__':
    main()
