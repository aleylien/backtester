import argparse
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
from backtester.utils import compute_statistics
from backtester.utils import statistical_tests


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


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log))
    cfg = load_config(args.config)

    # --- load fees & slippage mapping ---
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

    # --- create unique, human‐readable run folder and save config ---
    base_out = cfg['output']['root']
    now = datetime.now()
    run_name = now.strftime("%H:%M %d.%m.%Y") + f" ({cfg['strategy']['module']})"
    run_out = os.path.join(base_out, run_name)
    os.makedirs(run_out, exist_ok=True)
    with open(os.path.join(run_out, "config_used.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Running into → {run_out}")

    # --- load data ---
    dl = DataLoader(
        data_dir=cfg['data']['path'],
        symbol=cfg['data']['symbol'],
        timeframe=cfg['data']['timeframe'],
        base_timeframe=cfg['data'].get('base_timeframe')
    )
    df = dl.load()
    start = cfg['data'].get('start')
    end = cfg['data'].get('end')
    if start or end:
        df = df.loc[start or df.index[0]: end or df.index[-1]]
        logging.info(f"Filtered data to {df.index[0]} → {df.index[-1]} ({len(df):,} bars)")
    logging.info(f"Loaded {len(df):,} bars for {cfg['data']['symbol']} at {cfg['data']['timeframe']}")

    # --- run backtest & optimization ---
    results = run_backtest(df, cfg)

    # --- display bundle summary table ---
    df_summary = results.set_index('bundle')
    # choose column order: train/test, params, then oos metrics
    cols = ['train_bars', 'test_bars'] + \
           [c for c in df_summary.columns if not c.startswith('oos_')] + \
           [c for c in df_summary.columns if c.startswith('oos_')]
    # if train_bars/test_bars not in results, drop them from cols
    cols = [c for c in cols if c in df_summary.columns]
    df_summary = df_summary[cols]
    print("\nBundle summary:")
    print(df_summary.to_string(float_format="%.4f"))

    # save summary CSV
    df_summary.to_csv(os.path.join(run_out, 'bundle_summary.csv'))
    print(f"Bundle summary saved to {run_out}/bundle_summary.csv")

    # --- save raw results ---
    results.to_csv(os.path.join(run_out, 'results.csv'), index=False)
    print(f"Results saved to {run_out}/results.csv")

    # --- bundle date ranges ---
    n_bundles = cfg['optimization']['bundles']
    splits = list(generate_splits(df, n_bundles))
    print("\nBundle date ranges:")
    for i, (train_df, test_df) in enumerate(splits, 1):
        print(f" Bundle {i}: IS {train_df.index[0].date()}→{train_df.index[-1].date()}, "
              f"OOS {test_df.index[0].date()}→{test_df.index[-1].date()}")
    print()

    # --- collect IS/OOS diagnostics ---
    all_diags = []
    strat_mod = import_module(f"backtester.strategies.{cfg['strategy']['module']}")
    strat_fn  = getattr(strat_mod, cfg['strategy']['function'])
    for (_, row), (train_df, test_df) in zip(results.iterrows(), splits):
        bundle = int(row['bundle'])
        # extract only real strategy params
        params = {
            k: row[k]
            for k in results.columns
            if k not in ('bundle',) and not k.startswith('oos_') and not k.startswith('_')
        }
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
                pnl_df[[
                    'price_diff',
                    'prev_pos',
                    'pnl_price',
                    'delta_pos',
                    'cost_usd',
                    'cost_pct',
                    'slip_cost',
                    'pnl',
                    'equity',
                    'drawdown',
                ]],
                how='left',
                rsuffix='_pnl'
            )
            diag['bundle'] = bundle
            diag['sample'] = sample
            all_diags.append(diag)

    # --- combine & save full diagnostics ---
    combined = pd.concat(all_diags, ignore_index=False)
    combined.index.name = 'date'
    combined = combined.reset_index()
    combined.to_csv(os.path.join(run_out, 'details_all_bundles.csv'), index=False)
    print(f"Combined diagnostics saved to {run_out}/details_all_bundles.csv")

    # --- plot OOS equity curves (bar-index x-axis) ---
    oos = combined[combined['sample'] == 'OOS']
    fig, ax = plt.subplots()
    for b, grp in oos.groupby('bundle'):
        eq = grp['equity'] / grp['equity'].iloc[0]
        x  = np.arange(len(eq))
        ax.plot(x, eq.values, label=f'Bundle {b}')
    ax.set_title("OOS Equity Curves by Bundle")
    ax.set_xlabel("Bar number since OOS start")
    ax.set_ylabel("Equity (start = 1.0)")
    ax.legend()
    fig.savefig(os.path.join(run_out, 'equity_all_bundles.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"Equity overlay chart saved to {run_out}/equity_all_bundles.png")

    # --- plot OOS drawdowns (bar-index x-axis) ---
    fig, ax = plt.subplots()
    for b, grp in oos.groupby('bundle'):
        dd = grp['drawdown']
        x  = np.arange(len(dd))
        ax.plot(x, dd.values, label=f'Bundle {b}')
    ax.set_title("OOS Drawdowns by Bundle")
    ax.set_xlabel("Bar number since OOS start")
    ax.set_ylabel("Drawdown")
    ax.legend()
    fig.savefig(os.path.join(run_out, 'drawdown_all_bundles.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"Drawdown overlay chart saved to {run_out}/drawdown_all_bundles.png")

    # --- compute & save performance statistics ---
    compute_statistics(combined, run_out)

    # --- optional statistical tests ---
    resp = input("Proceed with permutation & bootstrap tests? [y/N]: ").strip().lower()
    if resp.startswith('y'):
        statistical_tests(
            combined, run_out,
            bootstrap_reps=cfg['tests'].get('bootstrap_reps', 1000),
            permutation_reps=cfg['tests'].get('permutation_reps', 1000),
        )
    else:
        print("Skipping statistical tests.")


if __name__ == '__main__':
    main()
