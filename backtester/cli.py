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


    # — create unique run folder and save config —
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    base_out = cfg['output']['root']
    run_out = os.path.join(base_out, run_id)
    os.makedirs(run_out, exist_ok=True)
    with open(os.path.join(run_out, "config_used.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # --- load data ---
    dl = DataLoader(
        data_dir=cfg['data']['path'],
        symbol=cfg['data']['symbol'],
        timeframe=cfg['data']['timeframe'],
        base_timeframe=cfg['data'].get('base_timeframe')
    )
    df = dl.load()

    # --- apply date filters if set ---
    start = cfg['data'].get('start')
    end   = cfg['data'].get('end')
    if start or end:
        df = df.loc[start or df.index[0]: end or df.index[-1]]
        logging.info(f"Filtered data to {df.index[0]} → {df.index[-1]} ({len(df):,} bars)")

    logging.info(f"Loaded {len(df):,} bars for {cfg['data']['symbol']} at {cfg['data']['timeframe']}")

    # --- run backtest & optimization ---
    results = run_backtest(df, cfg)

    # 1) save summary results
    results.to_csv(os.path.join(run_out, 'results.csv'), index=False)
    print(f"Results saved to {run_out}/results.csv")

    # 2) prepare walk-forward splits
    n_bundles = cfg['optimization']['bundles']
    splits    = list(generate_splits(df, n_bundles))

    print("\nBundle date ranges:")
    for i, (train_df, test_df) in enumerate(splits, 1):
        print(f" Bundle {i}: "
              f"IS {train_df.index[0].date()}→{train_df.index[-1].date()}, "
              f"OOS {test_df.index[0].date()}→{test_df.index[-1].date()}")
    print()

    # 3) collect diagnostics for IS & OOS
    all_diags = []
    strat_mod = import_module(f"backtester.strategies.{cfg['strategy']['module']}")
    strat_fn  = getattr(strat_mod, cfg['strategy']['function'])

    for (idx, row), (train_df, test_df) in zip(results.iterrows(), splits):
        bundle = int(row['bundle'])
        params = {
            k: row[k]
            for k in results.columns
            if k not in ('bundle',)
               and not k.startswith('oos_')
               and not k.startswith('_')
        }

        for sample, df_slice in (('IS', train_df), ('OOS', test_df)):
            # 3.1) get positions + any extra outputs from strategy
            pos_df = strat_fn(df_slice, **params)

            # 3.2) simulate PnL/equity/drawdown
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

            # 3.3) join the new PnL columns
            # 3.3) join the new PnL columns (avoid name collisions via rsuffix)
            diag = pos_df.join(
                pnl_df[[
                    'price_diff',  # Δprice[t]
                    'prev_pos',  # position[t-1]
                    'pnl_price',  # market PnL
                    'delta_pos',  # size of trades
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

    # 4) combine & save full diagnostics
    combined = pd.concat(all_diags, ignore_index=False)
    combined.index.name = 'date'
    combined = combined.reset_index()
    combined.to_csv(os.path.join(run_out, 'details_all_bundles.csv'), index=False)
    print(f"Combined diagnostics saved to {run_out}/details_all_bundles.csv")

    # 5) plot OOS equity curves (bar-count x-axis)
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

    # 6) plot OOS drawdowns (bar-count x-axis)
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


if __name__ == '__main__':
    main()
