import argparse
import copy
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


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log))
    cfg = load_config(args.config)

    # load fees & slippage mapping
    mapping_file = cfg['fees']['mapping']
    ext = os.path.splitext(mapping_file)[1].lower()
    if ext == '.json':
        with open(mapping_file) as f:
            fees_map = json.load(f)
    else:
        raise ValueError(f"Unsupported fees mapping type: {ext}")
    fee_row = fees_map[cfg['data']['symbol']]
    comm_usd = fee_row['commission_usd']
    comm_pct = fee_row['commission_pct']
    slip_pct = fee_row['slippage_pct']

    # create run folder
    base_out = cfg['output']['root']
    now = datetime.now()
    run_name = now.strftime("%H:%M %d.%m.%Y") + f" ({cfg['strategy']['module']})"
    run_out = os.path.join(base_out, run_name)
    os.makedirs(run_out, exist_ok=True)
    with open(os.path.join(run_out, "config_used.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Running into → {run_out}")

    # single-asset backtest
    combined = run_single_backtest(cfg, comm_usd, comm_pct, slip_pct)

    # summary of optimization results
    results = run_backtest(
        DataLoader(
            data_dir=cfg['data']['path'],
            symbol=cfg['data']['symbol'],
            timeframe=cfg['data']['timeframe'],
            base_timeframe=cfg['data'].get('base_timeframe')
        ).load(), cfg
    )
    df_summary = results.set_index('bundle')
    cols = ['train_bars','test_bars'] + \
           [c for c in df_summary.columns if not c.startswith('oos_')] + \
           [c for c in df_summary.columns if c.startswith('oos_')]
    cols = [c for c in cols if c in df_summary.columns]
    df_summary = df_summary[cols]
    print("\nBundle summary:")
    print(df_summary.to_string(float_format="%.4f"))
    df_summary.to_csv(os.path.join(run_out, 'bundle_summary.csv'), index=False)
    print(f"Bundle summary saved to {run_out}/bundle_summary.csv")

    # save raw results and diagnostics
    results.to_csv(os.path.join(run_out, 'results.csv'), index=False)
    combined.to_csv(os.path.join(run_out,'details_all_bundles.csv'), index=False)
    print(f"CSV outputs saved to {run_out}")

    # plots for single asset
    oos = combined[combined['sample']=='OOS']
    fig, ax = plt.subplots()
    for b, grp in oos.groupby('bundle'):
        eq = grp['equity']/grp['equity'].iloc[0]
        x = np.arange(len(eq))
        ax.plot(x, eq.values, label=f'Bundle {b}')
    ax.set_title("OOS Equity Curves")
    ax.legend()
    fig.savefig(os.path.join(run_out,'equity_all_bundles.png'), bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots()
    for b, grp in oos.groupby('bundle'):
        dd = grp['drawdown']
        x = np.arange(len(dd))
        ax.plot(x, dd.values, label=f'Bundle {b}')
    ax.set_title("OOS Drawdowns")
    ax.legend()
    fig.savefig(os.path.join(run_out,'drawdown_all_bundles.png'), bbox_inches='tight')
    plt.close(fig)

    # statistics and tests
    compute_statistics(combined, run_out)
    resp = input("Proceed with permutation & bootstrap tests? [y/N]: ").strip().lower()
    if resp.startswith('y'):
        statistical_tests(
            combined, run_out,
            bootstrap_reps=cfg['tests'].get('bootstrap_reps',1000),
            permutation_reps=cfg['tests'].get('permutation_reps',1000)
        )
    else:
        print("Skipping statistical tests.")

    # equal-weight portfolio
    if 'portfolio' in cfg:
        rets = {}
        symbols_order = []
        # run each instrument
        for inst in cfg['portfolio']['instruments']:
            symbols_order.append(inst['symbol'])
            # load that instrument's fees
            with open(inst['fees_mapping']) as f:
                inst_fees = json.load(f)
            fr = inst_fees[inst['symbol']]
            cu, cp, sp = fr['commission_usd'], fr['commission_pct'], fr['slippage_pct']

            # build a per‐instrument cfg copy
            inst_cfg = copy.deepcopy(cfg)
            inst_cfg['data']['symbol']             = inst['symbol']
            inst_cfg['fees']['mapping']            = inst['fees_mapping']
            inst_cfg['strategy']                   = inst['strategy']
            inst_cfg['optimization']['param_grid'] = inst['param_grid']

            combined_inst = run_single_backtest(inst_cfg, cu, cp, sp)
            oos_inst = combined_inst[combined_inst['sample']=='OOS']
            # per‐instrument daily returns
            rets[inst['symbol']] = oos_inst['pnl'] / inst.get('capital', 100_000)

        # turn into DataFrame, preserving original instrument order
        rets_df = pd.DataFrame({s: rets[s] for s in symbols_order}).fillna(0.0)

        # get weights & total capital
        weights, total_cap = get_portfolio_weights(cfg['portfolio']['instruments'])

        # portfolio return & equity
        port_ret    = (rets_df * weights).sum(axis=1)
        port_equity = (1 + port_ret).cumprod() * total_cap

        # save & plot
        port_equity.to_csv(os.path.join(run_out, 'portfolio_equity.csv'),
                           header=['equity'])
        fig, ax = plt.subplots()
        port_equity.plot(ax=ax)
        ax.set_title("Equal-Weight Portfolio Equity")
        fig.savefig(os.path.join(run_out, 'portfolio_equity.png'),
                    bbox_inches='tight')
        plt.close(fig)
        print(f"Portfolio equity chart saved to {run_out}/portfolio_equity.png")

        # compute portfolio stats
        df_port = pd.DataFrame({
            'date': port_equity.index,
            'pnl': port_ret * total_cap,
            'equity': port_equity,
            'drawdown': (port_equity.cummax() - port_equity) / port_equity.cummax(),
            'delta_pos': 0,  # ← add this
            'sample': 'OOS',
            'bundle': 1
        })
        compute_statistics(df_port, run_out)


if __name__ == '__main__':
    main()
