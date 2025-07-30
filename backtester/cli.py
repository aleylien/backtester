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
    md.append(f"**Strategy:** `{cfg['strategy']['module']}.{cfg['strategy']['function']}`")
    md.append("")

    md.append("**Contents:**")
    md.append("- [1. Combined Statistics](#1-combined-statistics)")
    md.append("- [2. Per-Asset Permutation Tests](#2-per-asset-permutation-tests)")
    md.append("- [3. Multiple-System Selection Bias](#3-multiple-system-selection-bias)")
    md.append("- [4. Key Charts](#4-key-charts)")
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
        df = df.rename(columns={"ret_5pct":"5th pctile","ret_95pct":"95th pctile"})
        # Format columns
        pct_cols = ["cagr", "annual_vol", "max_drawdown", "avg_drawdown", "win_rate", "5th pctile", "95th pctile",
                    "avg_win", "avg_loss", "max_loss_pct"]
        for c in pct_cols:
            if c in df:
                df[c] = df[c].apply(lambda x: f"{100*x:.1f}%" if pd.notnull(x) else "N/A")
        num_cols = ["sharpe","sortino","profit_factor","expectancy","std_daily"]
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
        folder = save_per_asset and os.path.join(run_out, sym) or run_out

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
            "Instrument":   sym,
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
        df5 = pd.read_csv(p5, index_col=0)
        df5.index.name = "System"
        df5 = df5.rename(columns={"solo_p":"Solo p","unbiased_p":"Unbiased p"})
        # Format
        for c in ["Solo p","Unbiased p"]:
            df5[c] = df5[c].apply(lambda x: f"{x:.3f}")
        md.append("## 3. Multiple-System Selection Bias")
        md.append(df_to_markdown(df5.reset_index(), decimals=3))
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

    # write file
    out_md = os.path.join(run_out, "summary.md")
    with open(out_md, "w") as f:
        f.write("\n\n".join(md))
    print(f"Generated summary.md → {out_md}")


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

    save_per_asset = bool(cfg['output'].get('save_per_asset', 0))

    # --- 1) Build the full instrument list dynamically -----------------------
    portfolio_cfg = cfg['portfolio']
    assets        = portfolio_cfg['assets']
    strat_defs    = portfolio_cfg['strategies']
    assign_map    = portfolio_cfg.get('assignments', {})
    default_strs  = assign_map.get('default', list(strat_defs.keys()))

    instruments = []
    for symbol in assets:
        strat_keys = assign_map.get(symbol, default_strs)
        for sk in strat_keys:
            sdef = strat_defs[sk]
            instruments.append({
                'symbol':       symbol,
                'strategy':     {'module':   sdef['module'],
                                 'function': sdef['function']},
                'param_grid':   sdef.get('param_grid',
                                         cfg['optimization']['param_grid']),
                'fees_mapping': cfg['fees']['mapping']
            })

    # --- 2) Compute portfolio weights & total capital ------------------------
    weights, total_capital = get_portfolio_weights(instruments, portfolio_cfg)

    # --- 3) Per-instrument backtests ---
    inst_stats = {}
    per_inst_results = []
    for inst, w in zip(instruments, weights):

        # САМ ДОБАВИЛ ЭТО:
        fee_row = fees_map[symbol]
        comm_usd, comm_pct, slip_pct = (
            fee_row['commission_usd'],
            fee_row['commission_pct'],
            fee_row['slippage_pct'],
        )
        dl_inst = DataLoader(
            data_dir       = inst_cfg['data']['path'],
            symbol         = symbol,
            timeframe      = inst_cfg['data']['timeframe'],
            base_timeframe = inst_cfg['data'].get('base_timeframe')
        )
        df_inst = dl_inst.load()

        inst_cfg = deepcopy(cfg)
        inst_cfg['data']['symbol']             = symbol
        inst_cfg['fees']['mapping']            = inst['fees_mapping']
        inst_cfg['strategy']                   = inst['strategy']
        inst_cfg['optimization']['param_grid'] = inst['param_grid']

        out_sub = os.path.join(run_out, symbol) if save_per_asset else run_out
        os.makedirs(out_sub, exist_ok=True)


        symbol = inst['symbol']
        logging.info(f"\n=== Running backtest for {symbol} (weight {w:.2%}) ===")
        # ... (loading data and optimizing per bundle) ...
        results_inst = run_backtest(df_inst, inst_cfg)
        results_inst.to_csv(os.path.join(out_sub, 'results.csv'), index=False)
        # Collect OOS diagnostics (positions and forecasts)
        splits_inst = list(generate_splits(df_inst, inst_cfg['optimization']['bundles']))
        diags = []
        strat_mod = import_module(f"backtester.strategies.{inst_cfg['strategy']['module']}")
        strat_fn = getattr(strat_mod, inst_cfg['strategy']['function'])
        for (idx, row), (_, test_df) in zip(results_inst.iterrows(), splits_inst):
            params = {k: row[k] for k in results_inst.columns
                      if k not in ('bundle',) and not k.startswith('oos_') and not k.startswith('_')}
            # Ensure any float params that should be int (like vol_window) are cast
            if 'vol_window' in params:
                params['vol_window'] = int(params['vol_window'])
            # Run strategy on OOS data with optimal params
            pos_df = strat_fn(test_df, **params)
            # We do NOT simulate PnL here yet if using find-signal mode; just collect position/forecast
            # But for standard mode, simulate PnL as before
            if portfolio_cfg.get('max_open_trades'):
                # For find-signal mode, just mark sample and bundle
                diag = pos_df.copy()
                diag['bundle'] = row['bundle']
                diag['sample'] = 'OOS'
            else:
                pnl_df = simulate_pnl(
                    positions=pos_df['position'],
                    price=test_df['close'],
                    multiplier=params.get('multiplier', 1.0),
                    fx=params.get('fx', 1.0),
                    capital=params.get('capital', 100_000),
                    commission_usd=comm_usd,
                    commission_pct=comm_pct,
                    slippage_pct=slip_pct,
                )
                diag = pos_df.join(pnl_df, how='left')
                diag['bundle'] = row['bundle']; diag['sample'] = 'OOS'
            diags.append(diag.reset_index())
        combined_inst = pd.concat(diags, ignore_index=True)
        combined_inst.to_csv(os.path.join(out_sub, 'details_all_bundles.csv'), index=False)
        print(f"{symbol}: details_all_bundles.csv saved")
        # If not using find-signal mode, compute stats for this instrument
        if not portfolio_cfg.get('max_open_trades'):
            stats = compute_statistics(combined_inst, out_sub)
            inst_stats[symbol] = stats
        per_inst_results.append(combined_inst)

    # --- 4) Portfolio aggregation ---
    if portfolio_cfg.get('max_open_trades'):
        # Find-Signal Mode: aggregate signals from all instruments
        max_trades = portfolio_cfg['max_open_trades']
        # Prepare a unified date index (all dates where any instrument had OOS data)
        all_dates = sorted({d for df in per_inst_results for d in df['date']})
        # Create a DataFrame for combined positions, initially zeros
        symbols = [inst['symbol'] + "_" + inst['strategy']['module'] for inst in instruments]
        port_positions = pd.DataFrame(0.0, index=pd.to_datetime(all_dates), columns=symbols)
        port_prices = {}
        port_forecasts = {}
        # Build a dictionary of forecast series for ranking signals
        for sym, df_inst in zip(symbols, per_inst_results):
            df_inst['date'] = pd.to_datetime(df_inst['date'])
            df_inst.set_index('date', inplace=True)
            # Ensure the DataFrame has forecast column; if not present, derive from position sign
            if 'capped_forecast' in df_inst:
                forecast_series = df_inst['capped_forecast'].copy()
            elif 'raw_forecast' in df_inst:
                forecast_series = df_inst['raw_forecast'].copy()
            else:
                # If strategy doesn't output forecast, use position as proxy (entry/exit signals)
                forecast_series = df_inst['position'].astype(float).copy()
            # Price series for PnL calculations
            price_series = df_inst['price']
            # Reindex forecast and price to all_dates, forward-fill to carry last value through closed days, and fill NaN with 0 for forecast
            forecast_series = forecast_series.reindex(port_positions.index).ffill().fillna(0.0)
            price_series = price_series.reindex(port_positions.index).ffill()
            port_forecasts[sym] = forecast_series
            port_prices[sym] = price_series
        # Now iterate over each day to simulate trades
        open_trades = set()  # currently open trades (by symbol)
        pending_entry = {}  # pending flip entries to open next day
        # Determine contract size for a trade (function of capital, price on entry)
        total_capital = portfolio_cfg.get('capital', total_capital)  # ensure we have total_capital
        for current_date in port_positions.index:
            # 1. Process exits
            exits_today = []
            for sym in list(open_trades):
                # If forecast sign became 0 or flipped, we close the trade
                if np.sign(port_forecasts[sym].loc[current_date]) == 0:
                    exits_today.append(sym)
                elif np.sign(port_forecasts[sym].loc[current_date]) * np.sign(
                        port_forecasts[sym].shift(1).loc[current_date]) == -1:
                    # forecast sign flip: close now, and mark for re-entry
                    exits_today.append(sym)
                    pending_entry[sym] = np.sign(port_forecasts[sym].loc[current_date])
            for sym in exits_today:
                open_trades.discard(sym)
                pending_entry.pop(sym,
                                  None)  # if a pending entry was set for a flip, removing it because we'll handle fresh
                # Set position to 0 from this day onward (closing position)
                port_positions.loc[current_date:, sym] = 0.0
            # 2. Process new entries
            # Gather entry signals for symbols that are not currently open
            entry_signals = []
            for sym in symbols:
                if sym in open_trades:
                    continue  # skip already open
                # Determine if a new entry should occur:
                prev_sign = np.sign(port_forecasts[sym].shift(1).loc[current_date])
                curr_sign = np.sign(port_forecasts[sym].loc[current_date])
                if prev_sign == 0 and curr_sign != 0:
                    entry_signals.append((sym, abs(port_forecasts[sym].loc[current_date]), int(np.sign(curr_sign))))
                # Handle pending entries from previous flip (sign didn't go through zero)
                if sym in pending_entry:
                    # If the current forecast still supports the pending direction
                    pend_dir = pending_entry[sym]
                    if np.sign(port_forecasts[sym].loc[current_date]) == pend_dir:
                        entry_signals.append((sym, abs(port_forecasts[sym].loc[current_date]), int(pend_dir)))
                    # Remove pending flag regardless (we either use it now or drop it if signal changed)
                    pending_entry.pop(sym, None)
            # If there are more signals than available slots, sort by strength and cut off
            available_slots = max_trades - len(open_trades)
            if available_slots > 0 and entry_signals:
                entry_signals.sort(key=lambda x: x[1], reverse=True)
                chosen_signals = entry_signals[:available_slots]
            else:
                chosen_signals = []
            for sym, strength, direction in chosen_signals:
                # Open trade for sym
                open_trades.add(sym)
                # Determine position size (contracts) based on fraction of capital
                # fraction = 1/max_trades (equal allocation per allowed trade)
                trade_fraction = 1.0 / max_trades
                entry_price = port_prices[sym].loc[current_date]
                if np.isnan(entry_price) or entry_price == 0:
                    # If price data missing or zero (should not happen), skip
                    continue
                # Position contracts = (fraction * total_capital) / (price * multiplier * fx)
                # Retrieve multiplier and fx for this instrument from its config or results (assuming constant across bundles)
                fx = 1.0;
                mult = 1.0
                # We stored param grid in inst_cfg; get those if available (fallback to 1.0)
                for inst in instruments:
                    sym_key = inst['symbol'] + "_" + inst['strategy']['module']
                    if sym_key == sym:
                        mult = inst['strategy'].get('multiplier', 1.0) if 'multiplier' in inst[
                            'strategy'] else params.get('multiplier', 1.0)
                        fx = inst['strategy'].get('fx', 1.0) if 'fx' in inst['strategy'] else params.get('fx', 1.0)
                        break
                position_size = trade_fraction * total_capital / (entry_price * mult * fx)
                # Use sign for direction, and round to nearest whole contract
                position_size = int(round(position_size)) * direction
                # Assign this position from today onward until closed
                port_positions.loc[current_date:, sym] = position_size
        # After iterating all days, ensure any open trades are closed at the end for final equity calc
        if open_trades:
            last_day = port_positions.index[-1]
            for sym in list(open_trades):
                open_trades.remove(sym)
                port_positions.loc[last_day:,
                sym] = 0.0  # close at last day (effectively same as valuing at last price)
        # Calculate portfolio PnL and equity by summing PnL of all positions
        port_equity = pd.Series(0.0, index=port_positions.index)
        port_equity.iloc[0] = total_capital
        # We will accumulate equity manually: equity[t] = equity[t-1] + sum_i(Δprice_i[t] * prev_position_i * multiplier_i * fx_i)
        prev_prices = {sym: port_prices[sym].iloc[0] for sym in symbols}
        prev_equity = total_capital
        portfolio_pnl = []  # to store daily pnl for distribution
        for t in range(1, len(port_positions.index)):
            date = port_positions.index[t]
            daily_pnl = 0.0
            for sym in symbols:
                prev_pos = port_positions.iloc[t - 1][sym]
                if prev_pos != 0:
                    # Price change from prev day to current day
                    price_change = port_prices[sym].iloc[t] - prev_prices[sym]
                    # Use multiplier and fx for this asset
                    # (We fetch these similar to above; ideally store in a dict for efficiency)
                    fx = 1.0;
                    mult = 1.0
                    for inst in instruments:
                        if sym == inst['symbol'] + "_" + inst['strategy']['module']:
                            fx = inst['strategy'].get('fx', 1.0) if 'fx' in inst['strategy'] else 1.0
                            mult = inst['strategy'].get('multiplier', 1.0) if 'multiplier' in inst['strategy'] else 1.0
                            break
                    daily_pnl += price_change * prev_pos * mult * fx
                # update stored prev price for next iteration
                prev_prices[sym] = port_prices[sym].iloc[t]
            prev_equity = prev_equity + daily_pnl
            portfolio_pnl.append(daily_pnl)
            port_equity.iloc[t] = prev_equity
        # Save portfolio equity curve and stats
        out_port = os.path.join(run_out, 'portfolio')
        os.makedirs(out_port, exist_ok=True)
        pd.DataFrame({'equity': port_equity}, index=port_equity.index).to_csv(
            os.path.join(out_port, 'portfolio_equity.csv'), index_label='date')
        fig, ax = plt.subplots()
        ax.plot(port_equity.index, port_equity.values, label="Portfolio")
        ax.set_title("Portfolio Equity")
        ax.set_ylabel("Equity")
        ax.set_xlabel("Date")
        ax.legend()
        fig.savefig(os.path.join(out_port, 'portfolio_equity.png'), bbox_inches='tight')
        plt.close(fig)
        print("Portfolio equity saved (find-signal mode)")
        # Compute drawdowns and additional portfolio stats for summary
        cummax = port_equity.cummax()
        raw_dd = (cummax - port_equity) / cummax
        # Compose a portfolio diagnostics DataFrame similar to per-asset details
        df_port = pd.DataFrame({
            'date': port_equity.index,
            'equity': port_equity.values,
            'drawdown': raw_dd.values,
        })
        df_port['delta_pos'] = (port_positions.diff().fillna(0) != 0).any(axis=1).astype(
            int)  # 1 if any position changed (entry/exit) on that day
        df_port['pnl'] = pd.Series([0.0] + portfolio_pnl, index=port_equity.index)  # align PnL series (0 for first day)
        df_port['sample'] = 'OOS'
        df_port['bundle'] = 1  # treat as single bundle covering entire OOS period
        # Now compute statistics on the portfolio level
        agg_stats = compute_statistics(df_port, out_port)
        inst_stats["Portfolio"] = agg_stats
    else:
        # Standard equal-weight portfolio aggregation (existing logic, simplified for clarity)
        symbols = [inst['symbol'] for inst in instruments]
        rets_list = []
        for sym, df_inst in zip(symbols, per_inst_results):
            df_inst = df_inst.copy()
            df_inst['date'] = pd.to_datetime(df_inst['date'])
            df_inst.set_index('date', inplace=True)
            r = df_inst['equity'].pct_change().fillna(0.0).rename(sym)
            rets_list.append(r)
        rets_df = pd.concat(rets_list, axis=1).fillna(0.0)
        # use either equal weights or user-specified weights from portfolio_cfg
        w_map = portfolio_cfg.get('weights', {})
        w_vec = np.array([w_map.get(sym, 1.0 / len(symbols)) for sym in symbols], dtype=float)
        w_vec /= w_vec.sum()
        port_rets = rets_df.dot(w_vec)
        port_equity = (1 + port_rets).cumprod() * total_capital
        # Save portfolio outputs similar to above...
        out_port = os.path.join(run_out, 'portfolio')
        os.makedirs(out_port, exist_ok=True)
        pd.DataFrame({'equity': port_equity}, index=port_equity.index).to_csv(
            os.path.join(out_port, 'portfolio_equity.csv'), index_label='date')
        fig, ax = plt.subplots()
        ax.plot(port_equity.index, port_equity.values, label="Portfolio")
        ax.set_title("Portfolio Equity")
        ax.set_ylabel("Equity")
        ax.set_xlabel("Date")
        ax.legend()
        fig.savefig(os.path.join(out_port, 'portfolio_equity.png'), bbox_inches='tight')
        plt.close(fig)
        print("Portfolio equity saved")
        # Prepare portfolio diagnostics DataFrame for stats (similar to existing code)
        cummax = port_equity.cummax()
        raw_dd = (cummax - port_equity) / cummax
        # Determine bars where any instrument is in market (for delta_pos)
        pos_df = pd.concat([
            df_inst.set_index('date')['position'].rename(sym)
            for sym, df_inst in zip(symbols, per_inst_results)
        ], axis=1).fillna(0)
        open_pos = pos_df.abs().sum(axis=1) > 0
        df_port = pd.DataFrame({
            'date': port_equity.index,
            'equity': port_equity.values,
            'drawdown': raw_dd.values,
            'delta_pos': open_pos.astype(int).astype(float).values,
            'pnl': port_rets * total_capital,
            'sample': 'OOS',
            'bundle': 1
        })
        df_port.to_csv(os.path.join(out_port, 'details_all_bundles.csv'), index=False)
        compute_statistics(df_port, out_port)
        # Combine inst_stats and portfolio stats
        combined_stats = pd.DataFrame.from_dict(inst_stats, orient='index')
        combined_stats.loc['Portfolio'] = {
            'cagr': np.nan,  # will populate below
            # ... we will fill the portfolio stats after computing them ...
        }
        # Compute portfolio stats similar to how compute_statistics does aggregate, or simply read strategy_statistics.csv in out_port
        port_stats = pd.read_csv(os.path.join(out_port, 'strategy_statistics.csv')).iloc[0].to_dict()
        combined_stats.loc['Portfolio'] = port_stats
        combined_stats.to_csv(os.path.join(run_out, 'combined_stats.csv'))
        print(f"✅ combined_stats.csv created → {os.path.join(run_out, 'combined_stats.csv')}")

    # 5) Portfolio 30-bar return distribution with 95% CI ---------------------
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
    ax.hist(port_monthly, bins=30, density=True, alpha=0.6)
    ax.axvline(ci_low, linestyle='--', label='95% CI')
    ax.axvline(ci_high, linestyle='--')
    ax.set_title("Portfolio 30-Bar Return Distribution with 95% CI")
    ax.set_xlabel("Return")
    ax.set_ylabel("Density")
    ax.legend()
    fig.savefig(
        os.path.join(out_port, 'portfolio_30bar_return_distribution.png'),
        bbox_inches='tight'
    )
    plt.close(fig)
    print("30-bar return distribution chart saved")


    # --- 4e) Book-Style Permutation Tests with confirmations ----------------

    inst_folders = {}
    for inst in instruments:
        name = f"{inst['symbol']}_{inst['strategy']['module']}"
        folder = save_per_asset and os.path.join(run_out, inst['symbol']) or run_out
        inst_folders[name] = folder
    perms = cfg.get('tests', {}).get('permutations', 1000)

    # Test 1: OOS bundle permutation (per-asset)
    if input("Run Test 1 (OOS bundle permutation) for each asset? [y/N]: ").strip().lower().startswith('y'):
        for inst, w in zip(instruments, weights):
            # prepare inst_cfg similarly to above
            inst_cfg = deepcopy(cfg)
            inst_cfg['data']['symbol'] = inst['symbol']
            inst_cfg['fees']['mapping'] = inst['fees_mapping']
            inst_cfg['strategy'] = inst['strategy']
            inst_cfg['optimization']['param_grid'] = inst['param_grid']
            # Now run the test
            folder = save_per_asset and os.path.join(run_out, inst['symbol']) or run_out
            p1 = test1_permutation_oos(inst_cfg, folder, B=perms)
            pd.DataFrame({'oos_bundle_p': [p1]}).to_csv(os.path.join(folder, 'permutation_test_oos.csv'), index=False)
            print(f"  ✔ {inst['symbol']}: Test 1 complete (p={p1:.3f})")

    # Test 2: Training-process overfit (per-asset)
    if input("Run Test 2 (training-process overfit) for each asset? [y/N]: ").strip().lower().startswith('y'):
        for inst in instruments:
            folder = save_per_asset and os.path.join(run_out, inst['symbol']) or run_out
            p2 = test2_permutation_training(inst_cfg, folder, B=perms)
            pd.DataFrame({'training_overfit_p': [p2]}) \
                .to_csv(os.path.join(folder, 'permutation_test_training.csv'), index=False)
            print(f"  ✔ {inst['symbol']}: Test 2 complete (p={p2:.3f})")

    # Test 5: Multiple-system selection bias (portfolio-wide)
    if input("Run Test 5 (multiple-system selection bias)? [y/N]: ").strip().lower().startswith('y'):
        df5 = permutation_test_multiple(inst_folders, B=perms)
        df5.to_csv(os.path.join(run_out, 'permutation_test_multiple.csv'))
        print(f"  ✔ Test 5 complete (saved permutation_test_multiple.csv)")

    # Test 6: Partition return (per-asset)
    if input("Run Test 6 (partition return) for each asset? [y/N]: ").strip().lower().startswith('y'):
        drift = cfg.get('tests', {}).get('drift_rate', 0.0)
        for inst, w in zip(instruments, weights):
            folder = save_per_asset and os.path.join(run_out, inst['symbol']) or run_out

            # load price series
            dl_inst = DataLoader(
                data_dir=inst_cfg['data']['path'],
                symbol=inst['symbol'],
                timeframe=inst_cfg['data']['timeframe'],
                base_timeframe=inst_cfg['data'].get('base_timeframe')
            )
            df_price = dl_inst.load()
            df_price['price_change'] = df_price['close'].pct_change().fillna(0)

            # load results and detect pnl column
            res = pd.read_csv(os.path.join(folder, 'results.csv'))
            pnl_cols = [c for c in res.columns if c.lower().endswith('_pnl')]
            pnl_col = pnl_cols[0]
            best_row = res.loc[res[pnl_col].idxmax()]
            params = {k: best_row[k] for k in res.columns if k not in ('bundle',) and not k.startswith('oos_')}
            if 'vol_window' in params: params['vol_window'] = int(params['vol_window'])

            # get positions
            strat_mod = import_module(f"backtester.strategies.{inst['strategy']['module']}")
            strat_fn = getattr(strat_mod, inst['strategy']['function'])
            df_price['position'] = strat_fn(df_price, **params)['position'].values

            # build instance backtest fn
            inst_cap = total_capital * w
            fee_row = fees_map[inst['symbol']]

            def inst_backtest(df):
                pnl = simulate_pnl(
                    positions=df['position'],
                    price=df['close'],
                    multiplier=params['multiplier'],
                    fx=params['fx'],
                    capital=inst_cap,
                    commission_usd=fee_row['commission_usd'],
                    commission_pct=fee_row['commission_pct'],
                    slippage_pct=fee_row['slippage_pct'],
                )
                eq = (1 + pnl['pnl'] / inst_cap).cumprod() * inst_cap
                return eq.iloc[-1] / eq.iloc[0] - 1

            # run partition_return
            pr = partition_return(inst_backtest, df_price, drift_rate=drift, oos_start=0, B=perms)
            pd.DataFrame([pr]).to_csv(os.path.join(folder, 'partition_return.csv'), index=False)
            print(
                f"  ✔ {inst['symbol']}: Test 6 complete (trend={pr['trend']:.1f}, bias={100 * pr['mean_bias']:.2f}%, skill={pr['skill']:.2f})")

    # 6) Append Portfolio row to combined_stats.csv with proper stats ----------
    # compute drawdowns & durations
    cummax       = port_equity.cummax()
    raw_dd       = (cummax - port_equity) / cummax
    durations, in_dd, d = [], False, 0
    for dd in raw_dd:
        if dd > 0:
            in_dd, d = True, d + 1
        else:
            if in_dd and d > 0:
                durations.append(d)
            in_dd, d = False, 0
    if in_dd and d > 0:
        durations.append(d)

    # portfolio‐level stats
    rets         = port_rets
    cagr         = (1 + rets).prod() ** (252/len(rets)) - 1
    ann_vol      = rets.std() * np.sqrt(252)
    sharpe       = (rets.mean()/rets.std()*np.sqrt(252)) if rets.std()>0 else np.nan
    sortino      = (rets.mean()/rets[rets<0].std()*np.sqrt(252)) if len(rets[rets<0])>0 else np.nan
    max_dd       = raw_dd.max()
    avg_dd       = raw_dd[raw_dd>0].mean()
    avg_dd_dur   = np.mean(durations) if durations else 0

    # trade‐based metrics across all instruments when any position != 0
    pos_df    = pd.concat([
        df_inst.set_index('date')['position'].rename(sym)
        for sym, df_inst in zip(symbols, per_inst_results)
    ], axis=1)
    open_pos  = pos_df.abs().sum(axis=1) > 0
    pnl_df    = pd.concat([
        df_inst.set_index('date')['pnl'].rename(sym)
        for sym, df_inst in zip(symbols, per_inst_results)
    ], axis=1)
    total_pnl = pnl_df.sum(axis=1).loc[open_pos]
    trades     = total_pnl.diff().abs().gt(0).sum()
    wins       = total_pnl[total_pnl>0].sum()
    loss       = -total_pnl[total_pnl<0].sum()
    pf         = wins / loss if loss>0 else np.nan
    expectancy = total_pnl.sum() / trades if trades>0 else np.nan
    win_rate   = total_pnl.gt(0).sum() / trades if trades>0 else np.nan

    # read existing combined_stats and append Portfolio
    combined_stats = pd.DataFrame.from_dict(inst_stats, orient='index')

    # Append the Portfolio row we just computed
    combined_stats.loc['Portfolio'] = {
        'cagr':            cagr,
        'annual_vol':      ann_vol,
        'sharpe':          sharpe,
        'sortino':         sortino,
        'max_drawdown':    max_dd,
        'avg_drawdown':    avg_dd,
        'avg_dd_duration': avg_dd_dur,
        'profit_factor':   pf,
        'expectancy':      expectancy,
        'win_rate':        win_rate,
        'std_daily':       rets.std()
    }

    combined_stats.to_csv(os.path.join(run_out, 'combined_stats.csv'))
    print(f"✅ combined_stats.csv created → {run_out}/combined_stats.csv")

    # 7) Generate final summary.md
    generate_summary_md(run_out, cfg, portfolio_cfg, save_per_asset, instruments)


if __name__ == '__main__':
    main()
