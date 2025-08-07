import argparse
import logging, os, json
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from importlib import import_module
from copy import deepcopy
from datetime import datetime

from backtester.config import load_config
from backtester.data_loader import DataLoader
from backtester.backtest import run_backtest, generate_splits
from backtester.pnl_engine import simulate_pnl
from backtester.utils import compute_statistics, statistical_tests
from backtester.stat_tests import (
    test1_permutation_oos,
    test2_permutation_training,
    permutation_test_multiple,
    partition_return
)


def get_portfolio_weights(instruments, portfolio_cfg):
    """
    Returns:
      - weights: list of floats summing to 1.0 (one per instrument)
      - total_capital: float from portfolio_cfg['capital']
    """
    # 1) Read TOTAL capital from portfolio.capital; fallback to old per-inst sum
    total_capital = portfolio_cfg.get(
        'capital',
        sum(inst.get('capital', 100_000) for inst in instruments)
    )

    # 2) Read portfolio.weights (dict or list), else equal-weight
    wcfg = portfolio_cfg.get('weights')
    if isinstance(wcfg, dict):
        raw     = [wcfg.get(inst['symbol'], 0.0) for inst in instruments]
        total_w = sum(raw) or 1.0
        weights = [r / total_w for r in raw]

    elif isinstance(wcfg, (list, tuple)):
        if len(wcfg) != len(instruments):
            raise ValueError("portfolio.weights list must match instruments")
        total_w = sum(wcfg) or 1.0
        weights = [w / total_w for w in wcfg]

    else:
        n = len(instruments)
        if n == 0:
            raise ValueError("No instruments provided")
        weights = [1.0 / n] * n

    return weights, total_capital


def parse_args():
    parser = argparse.ArgumentParser(
        description="Backtester CLI: run backtests and statistical tests"
    )
    parser.add_argument(
        "-c", "--config", required=True,
        help="Path to strategy_config.yaml"
    )
    parser.add_argument(
        "-l", "--log", default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    parser.add_argument(
        "--run-tests",
        type=str,
        default="",
        help="Comma-separated list of tests to run (e.g. '1,2,5,6'). "
             "Omit to skip all statistical tests."
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


def df_to_markdown(df: pd.DataFrame, decimals: int = 2) -> str:
    df = df.copy()
    # Round all numeric columns
    for c in df.select_dtypes(include=['float', 'int']).columns:
        df[c] = df[c].round(decimals)
    # Replace NaN/embed N/A
    df = df.where(pd.notnull(df), "N/A").astype(str)
    # Build Markdown
    cols   = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep    = "| " + " | ".join("---" for _ in cols) + " |"
    rows   = ["| " + " | ".join(r) + " |" for r in df.values]
    return "\n".join([header, sep] + rows)


def generate_summary_md(run_out: str, cfg: dict, portfolio_cfg, save_per_asset, instruments):
    md = []
    md.append(f"# Backtest Summary: `{os.path.basename(run_out)}`")
    md.append(f"**Run date:** {datetime.now():%Y-%m-%d %H:%M}")
    # md.append(f"**Strategy:** `{cfg['strategy']['module']}.{cfg['strategy']['function']}`")
    md.append("")

    md.append("**Contents:**")
    md.append("- [1. Combined Statistics](#1-combined-statistics)")
    md.append("- [2. Per-Asset Permutation Tests](#2-per-asset-permutation-tests)")
    md.append("- [3. Multiple-System Selection Bias](#3-multiple-system-selection-bias)")
    md.append("- [4. Key Charts](#4-key-charts)")
    md.append("- [5. Correlation Analysis](#5-correlation-analysis)")
    md.append("")

    # --- 1. Combined Statistics ---
    combined_csv = os.path.join(run_out, "combined_stats.csv")
    if os.path.exists(combined_csv):
        df = (pd.read_csv(combined_csv, index_col=0)
                .rename_axis("Instrument")
                .reset_index())
        # Reorder so Portfolio is last
        df = pd.concat([
            df[df["Instrument"] != "Portfolio"],
            df[df["Instrument"] == "Portfolio"]
        ], ignore_index=True)
        # Rename percentiles
        df = df.rename(columns={"ret_5pct": "5th pctile", "ret_95pct": "95th pctile",
                                "avg_cost_pct": "Cost %/Trade", "sharpe_no_cost": "Sharpe (no cost)"})
        # Format columns
        pct_cols = ["cagr", "annual_vol", "max_drawdown", "avg_drawdown", "win_rate",
                    "5th pctile", "95th pctile", "avg_win", "avg_loss", "max_loss_pct", "Cost %/Trade"]
        for c in pct_cols:
            if c in df:
                df[c] = df[c].apply(lambda x: f"{100*x:.1f}%" if pd.notnull(x) else "N/A")
        num_cols = ["sharpe", "sortino", "profit_factor", "expectancy", "std_daily", "Sharpe (no cost)"]
        for c in num_cols:
            if c in df:
                df[c] = df[c].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
        # Bold best Sharpe
        if "sharpe" in df:
            best = df[df.Instrument!="Portfolio"]["sharpe"].astype(float).idxmax()
            df.loc[best,"Instrument"] = f"**{df.loc[best,'Instrument']}**"
        md.append("## 1. Combined Statistics")
        md.append(df_to_markdown(df, decimals=2))
        md.append("")

    # --- 2. Per-Asset Permutation Tests (consolidated table) -------------
    rows = []
    for inst in instruments:
        sym    = inst["symbol"]
        strategy = inst['strategy']['module']
        ins_strat_folder = f"{sym}_{strategy}"
        folder = save_per_asset and os.path.join(run_out, ins_strat_folder) or run_out

        # load each test result (or N/A if missing)
        p1 = pd.read_csv(os.path.join(folder, "permutation_test_oos.csv")).iloc[0,0] \
             if os.path.exists(os.path.join(folder, "permutation_test_oos.csv")) else None
        p2 = pd.read_csv(os.path.join(folder, "permutation_test_training.csv")).iloc[0,0] \
             if os.path.exists(os.path.join(folder, "permutation_test_training.csv")) else None
        pr = pd.read_csv(os.path.join(folder, "partition_return.csv")) if os.path.exists(os.path.join(folder, "partition_return.csv")) else None

        trend     = pr["trend"].iloc[0]    if pr is not None else None
        bias      = pr["mean_bias"].iloc[0] if pr is not None else None
        skill     = pr["skill"].iloc[0]    if pr is not None else None

        rows.append({
            "Instrument":   f"{sym}-{strategy}",
            "Test 1 p":     f"{p1:.3f}" if p1 is not None else "N/A",
            "Test 2 p":     f"{p2:.3f}" if p2 is not None else "N/A",
            "Trend":        f"{trend:.1f}" if trend is not None else "N/A",
            "Bias":         f"{100*bias:.2f}%" if bias is not None else "N/A",
            "Skill":        f"{skill:.2f}"   if skill is not None else "N/A",
        })

    df_tests = pd.DataFrame(rows).set_index("Instrument")
    md.append("## 2. Per-Asset Permutation Tests")
    md.append(df_to_markdown(df_tests.reset_index(), decimals=0))
    md.append("")

    # --- 3. Multiple-System Selection Bias ---
    p5 = os.path.join(run_out, "permutation_test_multiple.csv")
    if os.path.exists(p5):
        df5 = pd.read_csv(p5)  # 'System' is a column

        # Split System -> Instrument/Strategy
        df5[['Instrument', 'Strategy']] = df5['System'].str.split('_', n=1, expand=True)

        # Format numerics
        for col in df5.select_dtypes(include=['number']).columns:
            df5[col] = df5[col].apply(lambda x: f"{x:.3f}")

        # Reorder & render
        cols_first = ['Instrument', 'Strategy', 'System']
        cols_rest = [c for c in df5.columns if c not in cols_first]
        df5 = df5[cols_first + cols_rest]
        md.append("## 3. Multiple-System Selection Bias")
        md.append(df_to_markdown(df5))
        md.append("")

    # --- 4. Key Charts ---
    md.append("## 4. Key Charts")
    for title, fname in [
        ("Equity Curves",           "equity_all_bundles.png"),
        ("Drawdowns",               "drawdown_all_bundles.png"),
        ("Portfolio Equity",        "portfolio/portfolio_equity.png"),
        ("30-Bar Return Dist.",     "portfolio/portfolio_30bar_return_distribution.png"),
        ("Drawdown Distribution",   "portfolio/drawdown_distribution.png"),
        ("DD Duration vs Magnitude","portfolio/dd_duration_vs_magnitude.png"),
    ]:
        path = os.path.join(run_out, fname)
        if os.path.exists(path):
            md.append(f"### {title}")
            md.append(f"![{title}]({fname})")
            md.append("")

    # --- 5. Correlation Analysis ---
    corr_asset_csv = os.path.join(run_out, "asset_correlation.csv")
    if os.path.exists(corr_asset_csv):
        md.append("## 5. Correlation Analysis")
        strat_corr_csv = os.path.join(run_out, "strategy_correlation.csv")
        if os.path.exists(strat_corr_csv):
            strat_corr = pd.read_csv(strat_corr_csv, index_col=0)
            md.append("### Strategy Return Correlation")
            md.append(df_to_markdown(strat_corr.reset_index(), decimals=2))
            md.append("")
        asset_corr = pd.read_csv(corr_asset_csv, index_col=0)
        md.append("### Asset Return Correlation")
        md.append(df_to_markdown(asset_corr.reset_index(), decimals=2))
        md.append("")

    # write file
    out_md = os.path.join(run_out, "summary.md")
    with open(out_md, "w") as f:
        f.write("\n\n".join(md))
    print(f"Generated summary.md → {out_md}")


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log))
    cfg = load_config(args.config)
    # Load fees mapping
    with open(cfg['fees']['mapping']) as f:
        fees_map = json.load(f)

    # Create run folder
    base_out = cfg['output']['root']
    now      = datetime.now()
    run_name = now.strftime("%H:%M %d.%m.%Y")    # + f" ({cfg['strategy']['module']})"
    run_out  = os.path.join(base_out, run_name)
    os.makedirs(run_out, exist_ok=True)
    with open(os.path.join(run_out, "config_used.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Running into → {run_out}")

    save_per_asset = bool(cfg['output'].get('save_per_asset', 0))

    # 1) Build instrument list (same as before)
    portfolio_cfg = cfg['portfolio']
    assets      = portfolio_cfg['assets']
    strat_defs  = portfolio_cfg['strategies']
    assign_map  = portfolio_cfg.get('assignments', {})
    default_strs= assign_map.get('default', list(strat_defs.keys()))

    instruments = []
    for symbol in assets:
        strat_keys = assign_map.get(symbol, default_strs)
        for sk in strat_keys:
            sdef = strat_defs[sk]
            instruments.append({
                'symbol':       symbol,
                'strategy':     {'module': sdef['module'], 'function': sdef['function']},
                'param_grid':   sdef.get('param_grid', cfg['optimization']['param_grid']),
                'fees_mapping': cfg['fees']['mapping']
            })

    # 2) Determine portfolio weights & total capital
    weights, total_capital = get_portfolio_weights(instruments, portfolio_cfg)
    symbol_counts = Counter(inst['symbol'] for inst in instruments)

    # 3) Per-instrument backtests
    inst_stats = {}
    per_inst_results = []
    instrument_names = []  # unique names "Symbol (Strategy)" if needed
    for inst, w in zip(instruments, weights):
        symbol = inst['symbol']
        strat_mod_name = inst['strategy']['module']
        # Construct unique instrument name for outputs
        inst_name = symbol if symbol_counts[symbol] == 1 else f"{symbol} ({strat_mod_name})"
        instrument_names.append(inst_name)

        logging.info(f"\n=== Running backtest for {inst_name} (weight {w:.2%}) ===")
        # Prepare per-instrument config and data
        inst_cfg = deepcopy(cfg)
        inst_cfg['data']['symbol']             = symbol
        inst_cfg['fees']['mapping']            = inst['fees_mapping']
        inst_cfg['strategy']                   = inst['strategy']
        inst_cfg['optimization']['param_grid'] = inst['param_grid']
        dl = DataLoader(data_dir=inst_cfg['data']['path'], symbol=symbol,
                        timeframe=inst_cfg['data']['timeframe'],
                        base_timeframe=inst_cfg['data'].get('base_timeframe'))
        df_inst = dl.load().copy()
        start, end = inst_cfg['data'].get('start'), inst_cfg['data'].get('end')
        if start or end:
            # Trim the DataFrame to the specified date range
            df_inst = df_inst.loc[start or df_inst.index[0]: end or df_inst.index[-1]]
        # Run walk-forward optimization on the filtered data
        results_inst = run_backtest(df_inst, inst_cfg)
        out_sub = os.path.join(run_out,
                   f"{symbol}_{strat_mod_name}" if save_per_asset and symbol_counts[symbol] > 1
                   else (symbol if save_per_asset else ""))
        os.makedirs(out_sub, exist_ok=True)
        results_inst.to_csv(os.path.join(out_sub, 'results.csv'), index=False)

        # Collect OOS diagnostics
        splits_inst = list(generate_splits(df_inst, inst_cfg['optimization']['bundles']))
        diags = []
        strat_mod = import_module(f"backtester.strategies.{inst_cfg['strategy']['module']}")
        strat_fn = getattr(strat_mod, inst_cfg['strategy']['function'])

        for (idx, row), (_, test_df) in zip(results_inst.iterrows(), splits_inst):
            # Pull out the best OOS parameters for this bundle
            params = {
                k: row[k]
                for k in results_inst.columns
                if k not in ('bundle',) and not k.startswith(('oos_', '_'))
            }
            if 'vol_window' in params:
                params['vol_window'] = int(params['vol_window'])

            # Generate positions
            params['capital'] = portfolio_cfg.get('capital')
            pos_df = strat_fn(test_df, **params)

            if portfolio_cfg.get('max_open_trades'):
                # Find-signal mode: record positions only
                diag = pos_df.copy()
            else:
                # Standard mode: simulate PnL
                fee_row = fees_map[symbol]
                pnl_df = simulate_pnl(
                    positions=pos_df['position'],
                    price=test_df['close'],
                    multiplier=params.get('multiplier', 1.0),
                    fx=params.get('fx', 1.0),
                    capital=w * total_capital,
                    commission_usd=fee_row['commission_usd'],
                    commission_pct=fee_row['commission_pct'],
                    slippage_pct=fee_row['slippage_pct'],
                )

                # —— NEW: drop any overlapping columns to avoid the join conflict ——
                overlap = pos_df.columns.intersection(pnl_df.columns)
                if not overlap.empty:
                    pnl_df = pnl_df.drop(columns=overlap)

                # Now safe to join
                diag = pos_df.join(pnl_df, how='left')

            # Tag bundle & sample, add date index back
            diag['bundle'] = int(row['bundle'])
            diag['sample'] = 'OOS'
            diags.append(diag.reset_index())

        combined_inst = pd.concat(diags, ignore_index=True)
        combined_inst.to_csv(
            os.path.join(out_sub, 'details_all_bundles.csv'),
            index=False
        )
        print(f"{symbol}: details_all_bundles.csv saved")
        if not portfolio_cfg.get('max_open_trades'):
            # Compute and store stats for this instrument, passing its config for timeframe info
            stats = compute_statistics(combined_inst, out_sub, config=inst_cfg)
            inst_stats[inst_name] = stats
        per_inst_results.append(combined_inst)

    # --- 4) Portfolio aggregation ---
    if portfolio_cfg.get('max_open_trades'):
        # Find-Signal Mode: aggregate signals from all instruments
        max_trades = portfolio_cfg['max_open_trades']
        all_dates  = sorted({d for df in per_inst_results for d in df['date']})
        symbols    = [
            f"{inst['symbol']}_{inst['strategy']['module']}"
            for inst in instruments
        ]
        port_positions = pd.DataFrame(0.0,
                                      index=pd.to_datetime(all_dates),
                                      columns=symbols)
        port_prices    = {}
        port_forecasts = {}

        # Prepare forecast & price series
        for sym, df_inst in zip(symbols, per_inst_results):
            df_inst['date'] = pd.to_datetime(df_inst['date'])
            df_inst.set_index('date', inplace=True)
            if 'capped_forecast' in df_inst:
                forecast = df_inst['capped_forecast']
            elif 'raw_forecast' in df_inst:
                forecast = df_inst['raw_forecast']
            else:
                forecast = df_inst['position'].astype(float)
            price = df_inst['price']

            port_forecasts[sym] = (
                forecast.reindex(port_positions.index)
                        .ffill()
                        .fillna(0.0)
            )
            port_prices[sym]    = price.reindex(port_positions.index).ffill()

        # 1) Cache multipliers & fx once
        param_map = {
            sym: {
                'multiplier': inst['strategy'].get('multiplier', 1.0),
                'fx':         inst['strategy'].get('fx', 1.0)
            }
            for inst in instruments
            for sym in [f"{inst['symbol']}_{inst['strategy']['module']}"]
        }

        open_trades   = set()
        pending_entry = {}
        total_cap     = portfolio_cfg.get('capital', total_capital)

        # 2) Simulate trade logic (entries/exits)
        for current_date in port_positions.index:
            # Exits
            to_exit = []
            for sym in list(open_trades):
                prev_s = port_forecasts[sym].shift(1).loc[current_date]
                curr_s = port_forecasts[sym].loc[current_date]
                if curr_s == 0 or (np.sign(prev_s)*np.sign(curr_s) == -1):
                    to_exit.append(sym)
                    if curr_s != 0:
                        pending_entry[sym] = np.sign(curr_s)
            for sym in to_exit:
                open_trades.discard(sym)
                pending_entry.pop(sym, None)
                port_positions.loc[current_date:, sym] = 0.0

            # Entries
            signals = []
            for sym in symbols:
                if sym in open_trades:
                    continue
                prev_s = np.sign(port_forecasts[sym].shift(1).loc[current_date])
                curr_s = np.sign(port_forecasts[sym].loc[current_date])
                if prev_s == 0 and curr_s != 0:
                    signals.append((sym, abs(port_forecasts[sym].loc[current_date]), curr_s))
                if sym in pending_entry and np.sign(port_forecasts[sym].loc[current_date]) == pending_entry[sym]:
                    signals.append((sym, abs(port_forecasts[sym].loc[current_date]), pending_entry[sym]))
                    pending_entry.pop(sym, None)

            slots = max_trades - len(open_trades)
            if slots > 0 and signals:
                signals.sort(key=lambda x: x[1], reverse=True)
                chosen = signals[:slots]
            else:
                chosen = []

            for sym, strength, direction in chosen:
                open_trades.add(sym)
                frac = 1.0 / max_trades
                price = port_prices[sym].loc[current_date]
                if pd.isna(price) or price == 0:
                    continue
                mult = param_map[sym]['multiplier']
                fx   = param_map[sym]['fx']
                size = int(round(frac * total_cap / (price * mult * fx))) * direction
                port_positions.loc[current_date:, sym] = size

        # 3) Vectorized PnL & equity
        price_changes = (
            pd.DataFrame({sym: port_prices[sym].diff().fillna(0)
                         for sym in symbols},
                         index=port_positions.index)
        )
        pos_shift = port_positions.shift(1).fillna(0)
        mult_df   = pd.DataFrame(
            {sym: param_map[sym]['multiplier'] for sym in symbols},
            index=port_positions.index
        )
        fx_df     = pd.DataFrame(
            {sym: param_map[sym]['fx'] for sym in symbols},
            index=port_positions.index
        )

        pnl_matrix  = price_changes * pos_shift * mult_df * fx_df
        daily_pnl   = pnl_matrix.sum(axis=1)
        port_equity = daily_pnl.cumsum() + total_cap
        portfolio_pnl = daily_pnl.tolist()

        # Save & plot portfolio equity
        out_port = os.path.join(run_out, 'portfolio')
        os.makedirs(out_port, exist_ok=True)
        pd.DataFrame({'equity': port_equity}, index=port_equity.index).to_csv(
            os.path.join(out_port, 'portfolio_equity.csv'),
            index_label='date'
        )
        fig, ax = plt.subplots()
        ax.plot(port_equity.index, port_equity.values, label="Portfolio")
        ax.set_title("Portfolio Equity")
        ax.set_ylabel("Equity")
        ax.set_xlabel("Date")
        ax.legend()
        fig.savefig(os.path.join(out_port, 'portfolio_equity.png'), bbox_inches='tight')
        plt.close(fig)
        print("Portfolio equity saved (find-signal mode)")

        # Stats & summary
        cummax = port_equity.cummax()
        raw_dd = (cummax - port_equity) / cummax
        df_port = pd.DataFrame({
            'date':     port_equity.index,
            'equity':   port_equity.values,
            'drawdown': raw_dd.values,
            'delta_pos': (port_positions.diff().fillna(0)!=0).any(axis=1).astype(int),
            'pnl':       pd.Series([0.0]+portfolio_pnl, index=port_equity.index),
            'sample':    'OOS',
            'bundle':    1
        })
        agg_stats = compute_statistics(df_port, out_port)
        inst_stats["Portfolio"] = agg_stats
        port_rets = port_equity.pct_change().fillna(0.0)

    if not portfolio_cfg.get('max_open_trades'):
        # Sum dollar PnL across all instruments to build portfolio equity
        symbols = instrument_names
        # Align per-instrument PnL series by date, fill missing with 0
        pnl_series_list = []
        for name, df_inst in zip(symbols, per_inst_results):
            df_i = df_inst.copy()
            df_i['date'] = pd.to_datetime(df_i['date'])
            df_i.set_index('date', inplace=True)
            pnl_ser = df_i['pnl'].fillna(0.0)
            pnl_ser.name = name
            pnl_series_list.append(pnl_ser)
        pnl_df = pd.concat(pnl_series_list, axis=1).fillna(0.0)
        port_pnl = pnl_df.sum(axis=1)
        port_equity = (port_pnl.cumsum() + total_capital)
        port_rets = port_equity.pct_change().fillna(0.0)
        # Save portfolio equity series and chart
        out_port = os.path.join(run_out, 'portfolio')
        os.makedirs(out_port, exist_ok=True)
        pd.DataFrame({'equity': port_equity}, index=port_equity.index)\
            .to_csv(os.path.join(out_port, 'portfolio_equity.csv'), index_label='date')
        fig, ax = plt.subplots()
        ax.plot(port_equity.index, port_equity.values, label="Portfolio")
        ax.set_title("Portfolio Equity")
        ax.set_ylabel("Equity")
        ax.set_xlabel("Date")
        ax.legend()
        fig.savefig(os.path.join(out_port, 'portfolio_equity.png'), bbox_inches='tight')
        plt.close(fig)
        print("Portfolio equity saved")
        # Prepare portfolio diagnostics DataFrame for stats
        cummax = port_equity.cummax()
        raw_dd = (cummax - port_equity) / cummax
        # Determine bars where any instrument is in position
        pos_series_list = []
        for name, df_inst in zip(symbols, per_inst_results):
            df_i = df_inst.copy()
            df_i['date'] = pd.to_datetime(df_i['date'])
            df_i.set_index('date', inplace=True)
            pos_ser = df_i['position'].fillna(0.0)
            pos_ser.name = name
            pos_series_list.append(pos_ser)
        pos_df = pd.concat(pos_series_list, axis=1).fillna(0.0)
        open_pos = pos_df.abs().sum(axis=1) > 0
        df_port = pd.DataFrame({
            'date': port_equity.index,
            'position': open_pos.values,
            'equity': port_equity.values,
            'drawdown': raw_dd.values,
            'delta_pos': open_pos.astype(float).values,
            'pnl': port_pnl.values,
            'sample': 'OOS',
            'bundle': 1
        })
        df_port.to_csv(os.path.join(out_port, 'details_all_bundles.csv'), index=False)
        # Compute portfolio stats and aggregate all stats
        stats = compute_statistics(df_port, out_port, config=cfg)
        inst_stats["Portfolio"] = stats
        inst_stats["Portfolio"] = stats
        combined_stats = pd.DataFrame.from_dict(inst_stats, orient='index')
        combined_stats.to_csv(os.path.join(run_out, 'combined_stats.csv'))
        print(f"✅ combined_stats.csv created → {os.path.join(run_out, 'combined_stats.csv')}")

        # 5) Generate key charts for equity and drawdowns across all instruments
        if len(per_inst_results) > 0:
            # Equity curves for all instruments (normalized per bundle)
            fig, ax = plt.subplots()
            for name, df_inst in zip(instrument_names, per_inst_results):
                df_i = df_inst.copy()
                df_i['date'] = pd.to_datetime(df_i['date'])
                df_i.set_index('date', inplace=True)
                # Plot each OOS bundle as separate line
                for b, grp in df_i[df_i['sample'] == 'OOS'].groupby('bundle'):
                    eq_norm = grp['equity'] / grp['equity'].iloc[0]  # normalize equity to 1 at segment start
                    ax.plot(eq_norm.index, eq_norm.values, label=f"{name} (Bundle {int(b)})")
            ax.set_title("Equity Curves (OOS bundles normalized to 1)")
            ax.set_ylabel("Normalized Equity")
            ax.set_xlabel("Date")
            # For readability, show legend only if few series; else omit bundle-level labels
            if len(instrument_names) <= 3:
                ax.legend(fontsize='x-small')
            fig.savefig(os.path.join(run_out, "equity_all_bundles.png"), bbox_inches='tight')
            plt.close(fig)
            print("equity_all_bundles.png saved")

            # Drawdown curves for all instruments
            fig, ax = plt.subplots()
            for name, df_inst in zip(instrument_names, per_inst_results):
                df_i = df_inst.copy()
                df_i['date'] = pd.to_datetime(df_i['date'])
                df_i.set_index('date', inplace=True)
                dd = df_i['drawdown'].fillna(0.0)
                ax.plot(dd.index, dd.values, label=name)
            ax.set_title("Drawdowns (OOS)")
            ax.set_ylabel("Drawdown")
            ax.set_xlabel("Date")
            if len(instrument_names) <= 5:
                ax.legend(fontsize='x-small')
            fig.savefig(os.path.join(run_out, "drawdown_all_bundles.png"), bbox_inches='tight')
            plt.close(fig)
            print("drawdown_all_bundles.png saved")

            # Save per-asset equity chart in each subdirectory
            if save_per_asset:
                for name, df_inst in zip(instrument_names, per_inst_results):
                    sym_folder = os.path.join(run_out, name if symbol_counts[name.split()[0]] > 1 else name.split()[0])
                    if os.path.isdir(sym_folder):
                        fig, ax = plt.subplots()
                        df_i = df_inst.copy()
                        df_i['date'] = pd.to_datetime(df_i['date'])
                        df_i.set_index('date', inplace=True)
                        for b, grp in df_i[df_i['sample'] == 'OOS'].groupby('bundle'):
                            eq_norm = grp['equity'] / grp['equity'].iloc[0]
                            ax.plot(eq_norm.index, eq_norm.values, label=f"Bundle {int(b)}")
                        ax.set_title(f"{name} Equity (OOS bundles)")
                        ax.set_ylabel("Normalized Equity")
                        ax.set_xlabel("Date")
                        ax.legend(fontsize='x-small')
                        fig.savefig(os.path.join(sym_folder, "equity_all_bundles.png"), bbox_inches='tight')
                        plt.close(fig)

    # --- 5) Portfolio 30-bar return dist & CI (unchanged) ---
    df_eq = pd.DataFrame({'equity': port_equity.values}, index=port_equity.index)
    df_eq['grp'] = np.arange(len(df_eq)) // 30
    first = df_eq.groupby('grp')['equity'].first()
    last  = df_eq.groupby('grp')['equity'].last()
    port_monthly = (last / first - 1).dropna()
    mean_pm = port_monthly.mean()
    std_pm  = port_monthly.std()
    ci_low  = mean_pm - 1.96 * std_pm / np.sqrt(len(port_monthly))
    ci_high = mean_pm + 1.96 * std_pm / np.sqrt(len(port_monthly))
    fig, ax = plt.subplots()
    ax.hist(port_monthly, bins=30, density=True, edgecolor='black', alpha=0.6)
    ax.axvspan(ci_low, ci_high, color='orange', alpha=0.3, label='95% CI')
    ax.axvline(mean_pm, color='black', linestyle='--', label='Mean')
    ax.set_title("Portfolio 30-Bar Return Distribution (95% CI)")
    ax.set_xlabel("30-bar Return")
    ax.set_ylabel("Density")
    ax.legend()
    fig.savefig(os.path.join(out_port, 'portfolio_30bar_return_distribution.png'), bbox_inches='tight')
    plt.close(fig)
    print("30-bar return distribution chart saved")

    # 6) Correlation analysis tables
    if len(instrument_names) > 1:
        # Compute daily returns for each instrument (skip bundle reset days)
        returns_dict = {}
        for name, df_inst in zip(instrument_names, per_inst_results):
            df_i = df_inst.copy()
            df_i['date'] = pd.to_datetime(df_i['date'])
            df_i.set_index('date', inplace=True)
            rets = []
            for _, grp in df_i[df_i['sample']=='OOS'].groupby('bundle'):
                r = grp['equity'].pct_change().dropna()
                rets.append(r)
            if rets:
                returns_dict[name] = pd.concat(rets)
        returns_df = pd.DataFrame(returns_dict)
        asset_corr = returns_df.corr().round(4)
        asset_corr.to_csv(os.path.join(run_out, "asset_correlation.csv"))
        # Strategy-level correlation if more than one strategy present
        strat_set = {inst['strategy']['module'] for inst in instruments}
        if len(strat_set) > 1:
            strat_returns = {}
            # Map instrument to strategy
            inst_to_strat = {name: inst['strategy']['module'] for name, inst in zip(instrument_names, instruments)}
            for strat in strat_set:
                strat_cols = [col for col in returns_df.columns if inst_to_strat[col] == strat]
                if not strat_cols:
                    continue
                # Treat missing asset returns as 0 (idle capital) for averaging
                strat_series = returns_df[strat_cols].fillna(0.0).mean(axis=1)
                strat_series.name = strat
                strat_returns[strat] = strat_series
            strat_df = pd.DataFrame(strat_returns)
            strat_corr = strat_df.corr().round(4)
            strat_corr.to_csv(os.path.join(run_out, "strategy_correlation.csv"))

    # --- 6) Permutation tests using --run-tests flags ---
    inst_folders = {
        # key = "SP500_ewmac" or "SP500_trend breakout"
        f"{inst['symbol']}_{inst['strategy']['module']}":
            # if saving per asset, the folder is run_out/<symbol>_<module>,
            # otherwise it’s just run_out
            (save_per_asset
                and os.path.join(run_out, f"{inst['symbol']}_{inst['strategy']['module']}")
                or run_out)
        for inst in instruments
    }
    perms = cfg.get('tests', {}).get('permutations', 1000)

    # Test 1: OOS bundle permutation (per-asset)
    if input("Run Test 1 (OOS bundle permutation) for each asset? [y/N]: ").strip().lower().startswith('y'):
        for inst, w in zip(instruments, weights):
            inst_cfg = deepcopy(cfg)
            inst_cfg['data']['symbol'] = inst['symbol']
            inst_cfg['fees']['mapping'] = inst['fees_mapping']
            inst_cfg['strategy'] = inst['strategy']
            inst_cfg['optimization']['param_grid'] = inst['param_grid']
            folder = save_per_asset \
                and os.path.join(run_out, f"{inst['symbol']}_{inst['strategy']['module']}") \
                or run_out
            p1 = test1_permutation_oos(inst_cfg, folder, w, B=perms)
            pd.DataFrame({'oos_bundle_p': [p1]}).to_csv(os.path.join(folder, 'permutation_test_oos.csv'), index=False)
            print(f"  ✔ {inst['symbol']} - {inst['strategy']['module']}: Test 1 complete (p={p1:.3f})")

    # Test 2: Training-process overfit (per-asset)
    if input("Run Test 2 (training-process overfit) for each asset? [y/N]: ").strip().lower().startswith('y'):
        for inst in instruments:
            inst_cfg = deepcopy(cfg)
            inst_cfg['data']['symbol'] = inst['symbol']
            inst_cfg['fees']['mapping'] = inst['fees_mapping']
            inst_cfg['strategy'] = inst['strategy']
            inst_cfg['optimization']['param_grid'] = inst['param_grid']
            folder = save_per_asset \
                and os.path.join(run_out, f"{inst['symbol']}_{inst['strategy']['module']}") \
                or run_out
            p2 = test2_permutation_training(inst_cfg, folder, B=perms)
            pd.DataFrame({'training_overfit_p': [p2]}).to_csv(os.path.join(folder, 'permutation_test_training.csv'), index=False)
            print(f"  ✔ {inst['symbol']} - {inst['strategy']['module']}: Test 2 complete (p={p2:.3f})")

    # Test 5: Multiple-system selection bias (portfolio-wide)
    if input("Run Test 5 (multiple-system selection bias)? [y/N]: ").strip().lower().startswith('y'):
        df5 = permutation_test_multiple(inst_folders, B=perms)
        out_csv = os.path.join(run_out, 'permutation_test_multiple.csv')
        df5.to_csv(out_csv, index=False)  # 'System' is a normal column now
        print(f"  ✔ Test 5 complete (saved {out_csv})")

    # Test 6: Partition return (per-asset)
    if input("Run Test 6 (partition return) for each asset? [y/N]: ").strip().lower().startswith('y'):
        drift = cfg.get('tests', {}).get('drift_rate', 0.0)
        for inst, w in zip(instruments, weights):
            folder = save_per_asset \
                and os.path.join(run_out, f"{inst['symbol']}_{inst['strategy']['module']}") \
                or run_out

            # Load price series for this instrument
            dl_inst = DataLoader(
                data_dir=cfg['data']['path'],
                symbol=inst['symbol'],
                timeframe=cfg['data']['timeframe'],
                base_timeframe=cfg['data'].get('base_timeframe')
            )
            df_price = dl_inst.load()
            df_price['price_change'] = df_price['close'].pct_change().fillna(0)

            # Load results and find the best-performing parameters
            res = pd.read_csv(os.path.join(folder, 'results.csv'))
            pnl_cols = [c for c in res.columns if c.lower().endswith('_pnl')]
            pnl_col = pnl_cols[0]
            best_row = res.loc[res[pnl_col].idxmax()]
            params = {k: best_row[k] for k in res.columns if k not in ('bundle',) and not k.startswith('oos_')}
            if 'vol_window' in params:
                params['vol_window'] = int(params['vol_window'])

            # Re-run strategy on full data with best params to get positions
            strat_mod = import_module(f"backtester.strategies.{inst['strategy']['module']}")
            strat_fn = getattr(strat_mod, inst['strategy']['function'])

            params['capital'] = inst_cfg['portfolio']['capital'] * w

            df_price['position'] = strat_fn(df_price, **params)['position'].values

            # Define a backtest function that computes total return on any given subset (for partition test)
            inst_cap = total_capital * w
            fee_row = fees_map[inst['symbol']]
            def inst_backtest(df):
                pnl = simulate_pnl(
                    positions=df['position'],
                    price=df['close'],
                    multiplier=params.get('multiplier', 1.0),
                    fx=params.get('fx', 1.0),
                    capital=inst_cap,
                    commission_usd=fee_row['commission_usd'],
                    commission_pct=fee_row['commission_pct'],
                    slippage_pct=fee_row['slippage_pct'],
                )
                eq = (1 + pnl['pnl'] / inst_cap).cumprod() * inst_cap
                return eq.iloc[-1] / eq.iloc[0] - 1

            # Run partition return test
            pr = partition_return(inst_backtest, df_price, drift_rate=drift, oos_start=0, B=perms)
            pd.DataFrame([pr]).to_csv(os.path.join(folder, 'partition_return.csv'), index=False)
            print(f"  ✔ {inst['symbol']}: Test 6 complete (trend={pr['trend']:.1f}, bias={100*pr['mean_bias']:.2f}%, skill={pr['skill']:.2f})")

    # 6) Append Portfolio row to combined_stats.csv with proper stats
    cummax = port_equity.cummax()
    raw_dd = (cummax - port_equity) / cummax
    durations = []
    in_dd = False
    d = 0
    for dd in raw_dd:
        if dd > 0:
            in_dd, d = True, d + 1
        else:
            if in_dd and d > 0:
                durations.append(d)
            in_dd, d = False, 0
    if in_dd and d > 0:
        durations.append(d)

    # Aggregate portfolio performance metrics
    rets     = port_rets
    cagr     = (1 + rets).prod() ** (252/len(rets)) - 1
    ann_vol  = rets.std() * np.sqrt(252)
    sharpe   = (rets.mean()/rets.std()*np.sqrt(252)) if rets.std() > 0 else np.nan
    sortino  = (rets.mean()/rets[rets<0].std()*np.sqrt(252)) if len(rets[rets<0]) > 0 else np.nan
    max_dd   = raw_dd.max()
    avg_dd   = raw_dd[raw_dd > 0].mean()
    avg_dd_dur = np.mean(durations) if durations else 0

    # Trade-level aggregate metrics across instruments
    pos_df   = pd.concat([df.set_index('date')['position'].rename(sym)
                          for sym, df in zip(symbols, per_inst_results)], axis=1)
    open_pos = pos_df.abs().sum(axis=1) > 0
    pnl_df   = pd.concat([df.set_index('date')['pnl'].rename(sym)
                          for sym, df in zip(symbols, per_inst_results)], axis=1)
    total_pnl = pnl_df.sum(axis=1).loc[open_pos]
    trades    = total_pnl.diff().abs().gt(0).sum()
    wins      = total_pnl[total_pnl > 0].sum()
    loss      = -total_pnl[total_pnl < 0].sum()
    pf        = wins / loss if loss > 0 else np.nan
    expectancy = total_pnl.sum() / trades if trades > 0 else np.nan
    win_rate  = total_pnl.gt(0).sum() / trades if trades > 0 else np.nan

    # Append the computed Portfolio stats to combined_stats.csv
    combined_stats = pd.DataFrame.from_dict(inst_stats, orient='index')
    combined_stats.loc['Portfolio'] = {
        'cagr': cagr,
        'annual_vol': ann_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_dd,
        'avg_drawdown': avg_dd,
        'avg_dd_duration': avg_dd_dur,
        'profit_factor': pf,
        'expectancy': expectancy,
        'win_rate': win_rate,
        'std_daily': rets.std()
    }
    combined_stats.to_csv(os.path.join(run_out, 'combined_stats.csv'))
    print(f"✅ combined_stats.csv created → {run_out}/combined_stats.csv")

    # 7) Generate final summary.md
    generate_summary_md(run_out, cfg, portfolio_cfg, save_per_asset, instruments)



if __name__ == '__main__':
    main()
