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

    # --- create unique, human-readable run folder and save config ---
    base_out = cfg['output']['root']
    now      = datetime.now()
    run_name = now.strftime("%H:%M %d.%m.%Y") + f" ({cfg['strategy']['module']})"
    run_out  = os.path.join(base_out, run_name)
    os.makedirs(run_out, exist_ok=True)
    with open(os.path.join(run_out, "config_used.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Running into → {run_out}")

    # toggle: per-asset subfolders?
    save_per_asset = bool(cfg['output'].get('save_per_asset', 0))

    # load portfolio instruments (if any)
    instruments = cfg.get('portfolio', {}).get('instruments', None)
    if instruments:
        # compute weights & total capital
        weights, total_capital = get_portfolio_weights(instruments)
    else:
        # single asset => one “instrument”
        instruments = [{
            'symbol': cfg['data']['symbol'],
            'fees_mapping': cfg['fees']['mapping'],
            'strategy':     cfg['strategy'],
            'param_grid':   cfg['optimization']['param_grid'],
            'capital':      cfg.get('optimization',{}).get('capital', 100_000)
        }]
        weights, total_capital = [1.0], instruments[0]['capital']

    # collect per-instrument stats for the final combined_stats.csv
    all_stats = {}

    # for each instrument (or just one, if no portfolio)
    for idx, inst in enumerate(instruments):
        # -------------------------------
        # 1) build instrument-specific config
        # -------------------------------
        inst_cfg = deepcopy(cfg)
        inst_cfg['data']['symbol'] = inst['symbol']
        inst_cfg['fees']['mapping'] = inst['fees_mapping']
        inst_cfg['strategy'] = inst['strategy']
        inst_cfg['optimization']['param_grid'] = inst['param_grid']

        # split the total capital by weight
        inst_capital = total_capital * weights[idx]
        # **ensure** that your strategy uses this capital by inserting it into the param grid
        inst_cfg['optimization']['param_grid']['capital'] = [inst_capital]

        # -------------------------------
        # 2) create sub-folder
        # -------------------------------
        folder_name = 'portfolio' if (idx == len(instruments)-1 and instruments!=[inst]) else inst['symbol']
        out_sub = os.path.join(run_out, folder_name)
        os.makedirs(out_sub, exist_ok=True)

        # -------------------------------
        # 3) load data & backtest
        # -------------------------------
        dl = DataLoader(
            data_dir=inst_cfg['data']['path'],
            symbol=inst_cfg['data']['symbol'],
            timeframe=inst_cfg['data']['timeframe'],
            base_timeframe=inst_cfg['data'].get('base_timeframe')
        )
        df = dl.load()
        # date filter
        start, end = inst_cfg['data'].get('start'), inst_cfg['data'].get('end')
        if start or end:
            df = df.loc[start or df.index[0] : end or df.index[-1]]
            logging.info(f"Filtered {inst['symbol']} to {df.index[0]}→{df.index[-1]} ({len(df):,} bars)")
        logging.info(f"Loaded {len(df):,} bars for {inst['symbol']}")

        # run the IS/OOS walk-forward
        results = run_backtest(df, inst_cfg)

        # --- bundle summary table & CSV ---
        df_sum = results.set_index('bundle')
        cols   = []
        for c in ['train_bars','test_bars']:
            if c in df_sum.columns: cols.append(c)
        cols += [c for c in df_sum.columns if not c.startswith('oos_')]
        cols += [c for c in df_sum.columns if c.startswith('oos_')]
        df_sum = df_sum[cols]
        print(f"\nBundle summary for {folder_name}:")
        print(df_sum.to_string(float_format="%.4f"))
        df_sum.to_csv(os.path.join(out_sub,'bundle_summary.csv'))

        # --- raw results.csv & diagnostics.csv + charts ---
        results.to_csv(os.path.join(out_sub,'results.csv'), index=False)
        print(f"Results → {folder_name}/results.csv")

        # splits
        splits = list(generate_splits(df, cfg['optimization']['bundles']))
        print(f"\nDate ranges for {folder_name}:")
        for i,(tr,te) in enumerate(splits,1):
            print(f"  Bundle {i}: IS {tr.index[0].date()}→{tr.index[-1].date()}, "
                  f"OOS {te.index[0].date()}→{te.index[-1].date()}")
        print()

        # collect per-bundle diagnostics & plot equity / drawdown
        all_diags = []
        strat_mod = import_module(f"backtester.strategies.{inst_cfg['strategy']['module']}")
        strat_fn  = getattr(strat_mod, inst_cfg['strategy']['function'])
        for (_, row), (tr,te) in zip(results.iterrows(), splits):
            b = int(row['bundle'])
            params = {
                k: row[k] for k in results.columns
                if k not in ('bundle',)
                   and not k.startswith('oos_')
                   and not k.startswith('_')
            }
            # ensure int window
            if 'vol_window' in params:
                params['vol_window'] = int(params['vol_window'])

            for sample, subset in (('IS',tr),('OOS',te)):
                pos_df = strat_fn(subset, **params)
                pnl_df = simulate_pnl(
                    positions    = pos_df['position'],
                    price        = subset['close'],
                    multiplier   = params['multiplier'],
                    fx           = params['fx'],
                    capital      = params.get('capital',100_000),
                    commission_usd = comm_usd,
                    commission_pct = comm_pct,
                    slippage_pct   = slip_pct,
                )
                diag = pos_df.join(
                    pnl_df[[
                        'price_diff','prev_pos','pnl_price',
                        'delta_pos','cost_usd','cost_pct','slip_cost',
                        'pnl','equity','drawdown'
                    ]],
                    how='left', rsuffix='_pnl'
                )
                diag['bundle'] = b
                diag['sample'] = sample
                all_diags.append(diag)

        combined = pd.concat(all_diags, ignore_index=False)
        combined.index.name = 'date'
        combined = combined.reset_index()
        combined.to_csv(os.path.join(out_sub,'details_all_bundles.csv'),index=False)
        print(f"Diagnostics → {folder_name}/details_all_bundles.csv")

        # plot OOS equity
        oos = combined[combined['sample']=='OOS']
        fig,ax = plt.subplots()
        for b,grp in oos.groupby('bundle'):
            eq = grp['equity']/grp['equity'].iloc[0]
            x  = np.arange(len(eq))
            ax.plot(x, eq.values, label=f'B{b}')
        ax.set_title(f"OOS Equity Curves ({folder_name})")
        ax.set_xlabel("Bar # since OOS start")
        ax.set_ylabel("Equity (start=1.0)")
        ax.legend(ncol=2, fontsize='small')
        fig.savefig(os.path.join(out_sub,'equity_all_bundles.png'),bbox_inches='tight')
        plt.close(fig)

        # plot OOS drawdown
        fig,ax = plt.subplots()
        for b,grp in oos.groupby('bundle'):
            dd = grp['drawdown']
            x  = np.arange(len(dd))
            ax.plot(x, dd.values, label=f'B{b}')
        ax.set_title(f"OOS Drawdowns ({folder_name})")
        ax.set_xlabel("Bar # since OOS start")
        ax.set_ylabel("Drawdown")
        ax.legend(ncol=2, fontsize='small')
        fig.savefig(os.path.join(out_sub,'drawdown_all_bundles.png'),bbox_inches='tight')
        plt.close(fig)

        # compute & save full statistics (includes monthly histogram)
        stats = compute_statistics(combined, out_sub)
        all_stats[folder_name] = stats

    # end per-instrument loop

    # --- combined_stats.csv across all instruments + portfolio ---
    df_comb = pd.DataFrame.from_dict(all_stats, orient='index')
    df_comb.to_csv(os.path.join(run_out,'combined_stats.csv'))
    print(f"\nCombined stats → {run_out}/combined_stats.csv")

    # --- optional tests on portfolio only ---
    if cfg.get('tests'):
        resp = input("\nProceed with permutation & bootstrap tests on portfolio? [y/N]: ").strip().lower()
        if resp.startswith('y'):
            statistical_tests(
                None,  # inside the util we’ll pick up only the “portfolio” group in combined
                run_out,
                bootstrap_reps=cfg['tests'].get('bootstrap_reps',1000),
                permutation_reps=cfg['tests'].get('permutation_reps',1000),
            )
        else:
            print("Skipping portfolio statistical tests.")


if __name__ == '__main__':
    main()
