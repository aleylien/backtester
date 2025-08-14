import argparse
import logging, os, json
import shutil
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter, MultipleLocator
from importlib import import_module
from copy import deepcopy
from datetime import datetime

from backtester.config import load_config
from backtester.data_loader import DataLoader
from backtester.backtest import run_backtest, generate_splits
from backtester.pnl_engine import simulate_pnl
from backtester.utils import compute_statistics, filter_params_for_callable, _safe_cagr_from_returns, _clean_returns_series
from backtester.stat_tests import (
    test1_permutation_oos,
    test2_permutation_training,
    permutation_test_multiple,
    partition_return
)
from backtester.fx import load_symbol_currency_map, get_fx_series
from backtester.reporting_excel import export_summary_xlsx
from backtester.strategy_aggregate import aggregate_by_strategy
from backtester.stats.periods import top_and_bottom_periods
from backtester.idm import compute_idm_map
from backtester.compounding import scale_pnl_with_policy
from strategies.A_weights import get_portfolio_weights
from backtester.normalised_price import compute_normalised_price


# === NEW: integrity checks & diagnostics ===

def _max_drawdown_window(equity: pd.Series):
    """
    Return (start_date, end_date, max_drawdown_float) for the equity series.
    Drawdown is (cummax - equity)/cummax.
    """
    if equity is None or len(equity) == 0:
        return None
    eq = pd.Series(pd.to_numeric(equity, errors="coerce")).dropna()
    if eq.empty:
        return None
    cummax = eq.cummax()
    dd = (cummax - eq) / cummax
    i_trough = int(dd.values.argmax())
    if i_trough <= 0 or not np.isfinite(dd.iloc[i_trough]):
        return None
    peak_val = float(cummax.iloc[i_trough])
    # Peak is first time cummax hit that value up to the trough
    i_peak = int(np.where(cummax.values[:i_trough + 1] == peak_val)[0][0])
    return eq.index[i_peak], eq.index[i_trough], float(dd.iloc[i_trough])


def _save_components_at_maxdd(out_port: str,
                              port_equity: pd.Series,
                              per_inst_results: list,
                              instrument_names: list) -> None:
    """
    For the portfolio's worst DD window, compute each component's DD within that window.
    Saves: portfolio/components_at_portfolio_maxDD.csv
    """
    res = _max_drawdown_window(port_equity)
    if res is None:
        return
    t0, t1, port_dd = res
    rows = []
    for name, df_inst in zip(instrument_names, per_inst_results):
        df = df_inst.copy()
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        eq_leg = pd.to_numeric(df.get('equity', pd.Series(dtype=float)), errors="coerce").dropna()
        eq_win = eq_leg.loc[(eq_leg.index >= t0) & (eq_leg.index <= t1)]
        if eq_win.empty:
            dd_leg = float('nan')
        else:
            cm = eq_win.cummax()
            dd_win = (cm - eq_win) / cm
            dd_leg = float(dd_win.max())
        rows.append({
            'component': name,
            'dd_in_portfolio_window': dd_leg,
            'portfolio_window_start': pd.Timestamp(t0),
            'portfolio_window_end': pd.Timestamp(t1),
            'portfolio_max_dd': port_dd
        })
    pd.DataFrame(rows).to_csv(os.path.join(out_port, 'components_at_portfolio_maxDD.csv'), index=False)


def _save_portfolio_integrity_checks(out_port: str,
                                     df_port: pd.DataFrame,
                                     pnl_by_symbol: dict,
                                     port_equity: pd.Series,
                                     tol: float = 1e-9) -> None:
    """
    Save JSON with:
      - max |Δportfolio_equity - portfolio_pnl|
      - max |sum(leg_pnl) - portfolio_pnl|   (aligned)
    File: portfolio/checks.json
    """
    dfp = df_port.copy()
    dfp['date'] = pd.to_datetime(dfp['date'])
    dfp.set_index('date', inplace=True)

    dEQ = pd.to_numeric(dfp['equity'], errors="coerce").diff().fillna(0.0)
    port_pnl = pd.to_numeric(dfp['pnl'], errors="coerce").fillna(0.0)
    max_abs_diff_dEq_pnl = float((dEQ - port_pnl).abs().max())

    leg = pd.DataFrame(pnl_by_symbol).copy()
    leg.index = pd.to_datetime(leg.index)
    leg = leg.sort_index()
    sum_legs = leg.sum(axis=1)

    # Align to portfolio pnl index; fill missing with 0
    sum_legs_aligned, port_pnl_aligned = sum_legs.align(port_pnl, join='outer', fill_value=0.0)
    max_abs_diff_sumleg = float((sum_legs_aligned - port_pnl_aligned).abs().max())

    checks = {
        'max_abs_diff_delta_equity_minus_portfolio_pnl': max_abs_diff_dEq_pnl,
        'max_abs_diff_sum_of_legs_minus_portfolio_pnl': max_abs_diff_sumleg,
        'tolerance': tol,
        'pass_delta_equity_equals_pnl': bool(max_abs_diff_dEq_pnl <= tol),
        'pass_sum_of_legs_equals_portfolio_pnl': bool(max_abs_diff_sumleg <= tol)
    }
    with open(os.path.join(out_port, 'checks.json'), 'w') as f:
        json.dump(checks, f, indent=2, default=str)


def parse_args():
    parser = argparse.ArgumentParser(description="Backtester CLI: run backtests and statistical tests")
    parser.add_argument("-c", "--config", required=True, help="Path to strategy_config.yaml")
    parser.add_argument("-l", "--log", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--run-tests", type=str, default="",
                        help="Comma-separated list of tests to run (e.g. '1,2,5,6'). Omit to skip all statistical tests.")
    return parser.parse_args()


def make_instrument_config(base_cfg: dict, inst: dict) -> dict:
    """
    Create a per-instrument configuration by combining the base config with instrument-specific settings.
    """
    inst_cfg = deepcopy(base_cfg)
    inst_cfg['data']['symbol'] = inst['symbol']
    inst_cfg['fees']['mapping'] = inst.get('fees_mapping', base_cfg['fees']['mapping'])
    inst_cfg['strategy'] = inst['strategy']
    inst_cfg['optimization']['param_grid'] = inst.get('param_grid', base_cfg['optimization']['param_grid'])
    return inst_cfg


def run_backtest_for_instrument(inst, base_cfg, portfolio_cfg, fees_map, weight, total_capital,
                                save_per_asset, run_out, symbol_counts,
                                base_ccy, symbol_ccy_map, fx_dir, timeframe, multipliers, idm_map,
                                save_per_instrument=True, strategy_inputs=None):
    """
    Run the walk-forward backtest for a single instrument and return its detailed results and stats.
    """
    symbol = inst['symbol']
    strat_mod_name = inst['strategy']['module']
    # Unique name for this instrument (include strategy name if symbol has multiple strategies)
    inst_name = symbol if symbol_counts[symbol] == 1 else f"{symbol} ({strat_mod_name})"
    logging.info(f"\n=== Running backtest for {inst_name} (weight {weight:.2%}) ===")
    # Prepare instrument-specific config and data
    inst_cfg = make_instrument_config(base_cfg, inst)
    dl = DataLoader(data_dir=inst_cfg['data']['path'],
                    symbol=symbol,
                    timeframe=inst_cfg['data']['timeframe'],
                    base_timeframe=inst_cfg['data'].get('base_timeframe'))
    df = dl.load().copy()
    # Trim to specified date range if provided
    start, end = inst_cfg['data'].get('start'), inst_cfg['data'].get('end')
    if start or end:
        df = df.loc[start or df.index[0] : end or df.index[-1]]
    # 1) Run walk-forward optimization to get best params for each bundle
    results = run_backtest(df, inst_cfg)
    # Prepare output directory for this instrument
    subdir = (f"{symbol}_{strat_mod_name}" if save_per_asset and symbol_counts[symbol] > 1
              else (symbol if save_per_asset else ""))
    out_dir = os.path.join(run_out, subdir)
    os.makedirs(out_dir, exist_ok=True)
    results.to_csv(os.path.join(out_dir, 'results.csv'), index=False)

    # 2) Re-run strategy on each OOS segment with best params to collect detailed diagnostics
    splits = list(generate_splits(df, inst_cfg['optimization']['bundles']))
    all_diags = []
    strat_module = import_module(f"strategies.{inst_cfg['strategy']['module']}")
    strat_fn = getattr(strat_module, inst_cfg['strategy']['function'])
    find_signal_mode = bool(portfolio_cfg.get('max_open_trades'))  # True if in "find signal" mode
    for (_, row), (_, oos_df) in zip(results.iterrows(), splits):
        # Best parameters for this bundle (exclude 'bundle' and any 'oos_' or internal columns)
        params = {k: row[k] for k in results.columns
                  if k not in ('bundle',) and not k.startswith(('oos_', '_'))}
        if 'vol_window' in params:
            params['vol_window'] = int(params['vol_window'])

        # Strip only what we truly externalized (keep multiplier)
        for k in ('fx', 'forecast_scale'):
            params.pop(k, None)

        # Decide the instrument allocation once (used for sizing AND PnL/equity math)
        alloc_capital = weight * total_capital

        # Multiplier from JSON (must reach the strategy BEFORE calling it)
        mult = float(multipliers.get(symbol, 1.0))

        # Ensure tau reaches the strategy if not in the grid/best row
        tau_global = None
        try:
            tau_global = base_cfg['portfolio']['strategy_defaults']['param_overrides']['tau']
        except Exception:
            pass
        if tau_global is None:
            tau_global = base_cfg['portfolio'].get('tau')
        if tau_global is not None and 'tau' not in params:
            params['tau'] = float(tau_global[0] if isinstance(tau_global, (list, tuple)) else tau_global)

        # FX series for this OOS bundle aligned to its index (use in sizing + PnL; do NOT modify price series)
        fx_series = get_fx_series(symbol, base_ccy, oos_df.index, symbol_ccy_map, fx_dir, timeframe)

        # Pass sizing knobs to the strategy BEFORE calling it
        params['capital'] = alloc_capital
        params['multiplier'] = mult
        params['fx'] = fx_series  # strategy may use it only in sizing denominator

        # ---- IDM injection ----
        # Priority: computed map per strategy -> YAML scalar -> default=1.0
        idm_yaml = float((base_cfg.get('strategy') or {}).get('idm', 1.0))
        try:
            idm_for_this_strategy = float(idm_map.get(strat_mod_name, idm_yaml))
        except Exception:
            idm_for_this_strategy = idm_yaml
        params['idm'] = idm_for_this_strategy

        # ---- Buffering configuration: merge YAML defaults -> instrument overrides -> existing params ----
        # Global defaults from YAML
        _sd = (base_cfg.get("strategy_defaults") or {})
        _glob_use_buffer = _sd.get("use_buffer", True)
        _glob_buffer_F  = _sd.get("buffer_F", 0.10)

        # Per-instrument overrides from YAML (strategy.params)
        _inst_params = (inst_cfg.get("strategy", {}).get("params") or {})

        # Precedence: params (already set) > instrument override > global default
        if "use_buffer" not in params:
            params["use_buffer"] = bool(_inst_params.get("use_buffer", _glob_use_buffer))
        else:
            params["use_buffer"] = bool(params["use_buffer"])

        if "buffer_F" not in params:
            try:
                params["buffer_F"] = float(_inst_params.get("buffer_F", _glob_buffer_F))
            except Exception:
                params["buffer_F"] = float(_glob_buffer_F)
        else:
            try:
                params["buffer_F"] = float(params["buffer_F"])
            except Exception:
                params["buffer_F"] = float(_glob_buffer_F)

        # Optional: an initial position for buffering to start from (default 0 contracts)
        params.setdefault("initial_pos", 0)

        # ---- Choose which price the strategy should see: RAW vs NORMALISED ----
        # start from 'close' and always ensure a 'price' column exists
        oos_df = oos_df.copy()
        if "price" not in oos_df.columns:
            oos_df["price"] = oos_df["close"]

        price_cfg = (base_cfg.get("data") or {})
        price_type = (price_cfg.get("price_type", "raw") or "raw").lower()
        if price_type in {"normalised", "normalized", "norm"}:
            norm_cfg = (price_cfg.get("normalized_price") or price_cfg.get("normalised_price") or {})
            method = (norm_cfg.get("method", "rolling") or "rolling").lower()
            window = int(norm_cfg.get("window", 25))
            span = norm_cfg.get("span", None)
            minp = norm_cfg.get("min_periods", None)
            scale = float(norm_cfg.get("scale", 100.0))
            start_value = float(norm_cfg.get("start_value", 0.0))

            oos_df["price"] = compute_normalised_price(
                oos_df["close"],
                method=method,
                window=window,
                span=span,
                min_periods=minp,
                scale=scale,
                start_value=start_value,
            )
        # (PnL still uses oos_df['close'] later — do NOT change that.)

        safe_params = filter_params_for_callable(strat_fn, params)
        res = strat_fn(oos_df, **safe_params)

        # Support return types: DataFrame OR (DataFrame, meta)
        meta = {}
        if isinstance(res, tuple) and len(res) >= 1:
            pos_df = res[0]
            if len(res) >= 2 and isinstance(res[1], dict):
                meta = res[1]
        else:
            pos_df = res

        # ---- Forecast scale: try meta → DataFrame.attrs → column (last non-NaN) ----
        forecast_scale_val = None
        try:
            if "forecast_scale" in meta:
                forecast_scale_val = float(meta["forecast_scale"])
            elif hasattr(pos_df, "attrs") and "forecast_scale" in pos_df.attrs:
                forecast_scale_val = float(pos_df.attrs["forecast_scale"])
            elif "forecast_scale" in pos_df.columns:
                s = pd.to_numeric(pos_df["forecast_scale"], errors="coerce").dropna()
                if not s.empty:
                    forecast_scale_val = float(s.iloc[-1])
        except Exception:
            forecast_scale_val = None

        # Save per-instrument forecast scale for later aggregation
        try:
            os.makedirs(out_dir, exist_ok=True)
            if forecast_scale_val is not None and not (
                    isinstance(forecast_scale_val, float) and np.isnan(forecast_scale_val)):
                pd.DataFrame([{
                    "instrument": f"{inst['symbol']}|{inst['strategy']['module']}",
                    "symbol": inst["symbol"],
                    "strategy": inst["strategy"]["module"],
                    "forecast_scale": float(forecast_scale_val),
                }]).to_csv(os.path.join(out_dir, "forecast_scale.csv"), index=False)
            else:
                logging.info("No forecast_scale returned for %s (%s).", inst.get("symbol"),
                             inst["strategy"]["module"])
        except Exception as e:
            logging.warning("Could not save forecast_scale.csv for %s: %s", inst.get("symbol"), e)

        if 'position' not in pos_df.columns:
            raise ValueError(f"{symbol}: strategy output must contain a 'position' column (contracts).")

        if find_signal_mode:
            # In find-signal mode, we don't price PnL here—just carry positions/forecasts forward
            diag = pos_df.copy()
        else:
            # Simulate dollar PnL for the OOS segment using LOCAL price + FX series
            fee = fees_map[symbol]
            pnl_df = simulate_pnl(
                positions=pos_df['position'],
                price=oos_df['close'],  # local price (do NOT pre-multiply by FX)
                multiplier=mult,
                fx=fx_series,  # per-date series
                capital=alloc_capital,
                commission_usd=fee.get('commission_usd', 0.0),
                commission_pct=fee.get('commission_pct', 0.0),
                slippage_pct=fee.get('slippage_pct', 0.0),
            )

            # Avoid duplicate columns if the strategy already produced any with same names
            overlap_cols = pos_df.columns.intersection(pnl_df.columns)
            if len(overlap_cols) > 0:
                pnl_df = pnl_df.drop(columns=list(overlap_cols))

            diag = pos_df.join(pnl_df, how='left')

        diag['bundle'] = int(row['bundle'])
        diag['sample'] = 'OOS'
        all_diags.append(diag.reset_index())  # reset_index to ensure 'date' is a column

        # Record for strategy aggregation (only OOS rows are used later)
        try:
            strat_key = inst['strategy']['module']  # adapt if your key lives elsewhere
        except Exception:
            strat_key = str(inst.get('strategy') or 'unknown')

        if strategy_inputs is not None:
            # Keep only the columns aggregator needs
            df_min = diag[['date', 'pnl', 'equity', 'sample']].copy() if all(
                c in diag.columns for c in ['date', 'pnl', 'equity', 'sample']) else diag.copy()
            strategy_inputs.append({
                "symbol": inst['symbol'],
                "strategy": strat_key,
                "alloc_capital": float(weight * total_capital),
                "df": df_min
            })

    combined = pd.concat(all_diags, ignore_index=True)

    if save_per_instrument:
        combined.to_csv(os.path.join(out_dir, 'details_all_bundles.csv'), index=False)
        print(f"{symbol}: details_all_bundles.csv saved")

    # 3) Compute performance statistics for this instrument (if not in find-signal mode)
    stats = None
    if save_per_instrument and not find_signal_mode:
        stats = compute_statistics(combined, out_dir, config=inst_cfg)
    return inst_name, combined, stats


def aggregate_portfolio(per_inst_results: list, instruments: list, instrument_names: list,
                        symbol_counts: Counter, total_capital: float, run_out: str,
                        portfolio_cfg: dict, base_cfg: dict):
    """
    Aggregate individual instrument results into overall portfolio results.
    Returns (port_equity_series, port_returns_series, portfolio_stats_dict).
    """
    find_signal_mode = bool(portfolio_cfg.get('max_open_trades'))
    out_port = os.path.join(run_out, 'portfolio')
    os.makedirs(out_port, exist_ok=True)
    if find_signal_mode:
        # --- Find-Signal Mode: construct portfolio by selecting up to max_open_trades simultaneous positions ---
        max_trades = portfolio_cfg['max_open_trades']
        # All unique timestamps across all instrument results
        all_dates = sorted({d for df in per_inst_results for d in df['date']})
        # Create identifier for each instrument: "Symbol_Module"
        symbols = [f"{inst['symbol']}_{inst['strategy']['module']}" for inst in instruments]
        # Initialize portfolio positions (rows indexed by date, one column per instrument, all zeros)
        port_positions = pd.DataFrame(0.0, index=pd.to_datetime(all_dates), columns=symbols)
        port_prices = {}    # to hold price series per instrument aligned to all_dates
        port_signals = {}   # to hold forecast/signal series per instrument aligned to all_dates
        # Prepare forecast & price series for each instrument
        for sym, df_inst in zip(symbols, per_inst_results):
            df_temp = df_inst.copy()
            df_temp['date'] = pd.to_datetime(df_temp['date'])
            df_temp.set_index('date', inplace=True)
            # Determine which series to use as the trading signal/forecast
            if 'capped_forecast' in df_temp:
                signal = df_temp['capped_forecast']
            elif 'raw_forecast' in df_temp:
                signal = df_temp['raw_forecast']
            else:
                # If no forecast output, fall back to position as a signal (convert position to float for sign)
                signal = df_temp['position'].astype(float)
            price = df_temp['price'] if 'price' in df_temp.columns else df_temp['close']
            # Align to the complete date index and forward-fill so that signal/price persist until next update
            port_signals[sym] = signal.reindex(port_positions.index).ffill().fillna(0.0)
            port_prices[sym] = price.reindex(port_positions.index).ffill()
        # Prepare multiplier and FX factors for each instrument (for position sizing)
        param_map = {sym: {
                        'multiplier': inst['strategy'].get('multiplier', 1.0),
                        'fx': inst['strategy'].get('fx', 1.0)
                     }
                     for inst, sym in zip(instruments, symbols)}
        open_trades = set()
        pending_entry = {}
        total_cap = portfolio_cfg.get('capital', total_capital)
        # Iterate through each time step to simulate portfolio trading logic
        for current_date in port_positions.index:
            # 1) Check exit conditions for currently open trades
            to_exit = []
            for sym in list(open_trades):
                prev_sig = port_signals[sym].shift(1).loc[current_date]
                curr_sig = port_signals[sym].loc[current_date]
                # Exit trade if signal goes to zero or flips direction
                if curr_sig == 0 or (np.sign(prev_sig) * np.sign(curr_sig) == -1):
                    to_exit.append(sym)
                    # If flipped direction (exit and immediately enter opposite), mark for re-entry
                    if curr_sig != 0:
                        pending_entry[sym] = np.sign(curr_sig)
            for sym in to_exit:
                open_trades.remove(sym)
                pending_entry.pop(sym, None)
                # Close position from this date onward
                port_positions.loc[current_date:, sym] = 0.0
            # 2) Check entry conditions for new trades
            signals = []
            for sym in symbols:
                if sym in open_trades:
                    continue  # skip signals for instruments already in a trade
                prev_sig = np.sign(port_signals[sym].shift(1).loc[current_date])
                curr_sig = np.sign(port_signals[sym].loc[current_date])
                # New entry signal if previously flat and now have a non-zero signal
                if prev_sig == 0 and curr_sig != 0:
                    signals.append((sym, abs(port_signals[sym].loc[current_date]), int(curr_sig)))
                # Delayed entry signal (if an exit occurred and the signal remains in same direction as pending)
                if sym in pending_entry and np.sign(port_signals[sym].loc[current_date]) == pending_entry[sym]:
                    signals.append((sym, abs(port_signals[sym].loc[current_date]), int(pending_entry[sym])))
                    pending_entry.pop(sym, None)
            # If there are free slots, sort signals by strength and take top signals up to max_trades
            slots = max_trades - len(open_trades)
            chosen_signals = []
            if slots > 0 and signals:
                signals.sort(key=lambda x: x[1], reverse=True)
                chosen_signals = signals[:slots]
            # Enter new trades for selected signals
            for sym, strength, direction in chosen_signals:
                open_trades.add(sym)
                # Determine position size allocating equal fraction of capital to each open trade
                price = port_prices[sym].loc[current_date]
                if pd.isna(price) or price == 0:
                    continue  # skip if missing price data
                mult = param_map[sym]['multiplier']
                fx   = param_map[sym]['fx']
                # Calculate how many contracts to trade with equal capital split (rounded to nearest integer)
                trade_size = int(round((total_cap / max_trades) / (price * mult * fx))) * direction
                port_positions.loc[current_date:, sym] = trade_size
        # 3) Once positions over time are determined, compute portfolio P&L and equity curve
        price_change = pd.DataFrame({sym: port_prices[sym].diff().fillna(0.0) for sym in symbols},
                                    index=port_positions.index)
        pos_shifted = port_positions.shift(1).fillna(0.0)  # positions held from previous period
        mult_df = pd.DataFrame({sym: param_map[sym]['multiplier'] for sym in symbols}, index=port_positions.index)
        fx_df   = pd.DataFrame({sym: param_map[sym]['fx'] for sym in symbols}, index=port_positions.index)
        pnl_matrix = price_change * pos_shifted * mult_df * fx_df
        daily_pnl = pnl_matrix.sum(axis=1)
        port_equity = daily_pnl.cumsum() + total_cap

        # Save portfolio equity series and chart
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
        print("Portfolio equity saved (find-signal mode)")

        # Compute basic portfolio stats and returns series
        cummax = port_equity.cummax()
        raw_dd = (cummax - port_equity) / cummax
        df_port = pd.DataFrame({
            'date': port_equity.index,
            'equity': port_equity.values,
            'drawdown': raw_dd.values,
            'delta_pos': (port_positions.diff().fillna(0.0) != 0).any(axis=1).astype(int),
            'pnl': pd.Series([0.0] + daily_pnl.tolist(), index=port_equity.index),
            'sample': 'OOS',
            'bundle': 1
        })
        portfolio_stats = compute_statistics(df_port, out_port, config=base_cfg)
        port_rets = port_equity.pct_change().fillna(0.0)

        period_res = top_and_bottom_periods(
            port_rets,
            resolution=base_cfg.get("statistics", {}).get("period_resolution", "monthly"),
            n=5
        )

        # Keep this somewhere to export later (step 7):
        bestworst_portfolio = period_res

        return port_equity, port_rets, portfolio_stats
    else:
        # --- Standard Mode with optional compounding / periodic rebalance ---
        symbols = instrument_names  # display names
        # Build per-symbol PnL series
        pnl_by_symbol = {}
        for name, df_inst in zip(symbols, per_inst_results):
            df_temp = df_inst.copy()
            df_temp['date'] = pd.to_datetime(df_temp['date'])
            df_temp.set_index('date', inplace=True)
            pnl_by_symbol[name] = df_temp['pnl'].fillna(0.0)

        # Target weights (by symbols) using your helper (equal if not specified)
        weights = get_portfolio_weights(base_cfg, symbols)

        # Capital policy (safe defaults)
        policy = (base_cfg.get("capital_policy") or {})
        mode = str(policy.get("mode", "none")).lower()
        reinv_freq = str(policy.get("reinvesting_frequency", "monthly")).lower()
        rebal_freq = str(policy.get("rebalance_frequency", "quarterly")).lower()

        # Scale pnl according to policy → portfolio equity & scaled per-symbol pnl
        equity_portfolio, scaled_by_symbol = scale_pnl_with_policy(
            pnl_by_symbol=pnl_by_symbol,
            weights=weights,
            initial_total_capital=float(total_capital),
            mode=mode,
            reinvesting_frequency=reinv_freq,
            rebalance_frequency=rebal_freq,
        )
        portfolio_pnl = sum(scaled_by_symbol.values())
        port_equity = equity_portfolio

        # === Save Portfolio equity & chart (kept as in your original code) ===
        # EQUITY CHART (unchanged code below)
        eq = pd.Series(port_equity).dropna().sort_index()
        eq.index = pd.to_datetime(eq.index)
        eq_norm = eq / float(eq.iloc[0])
        eq_rel = eq_norm - 1.0
        rolling_max = eq_norm.cummax()
        dd = (eq_norm / rolling_max) - 1.0

        fig, (ax_top, ax_dd) = plt.subplots(
            2, 1, figsize=(12, 8), dpi=170, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
        ax_top.plot(eq_rel.index, eq_rel.values, linewidth=1.4, label="Portfolio")
        ax_top.set_title("Portfolio Equity (Rebased to 1.0)")
        ax_top.set_ylabel("Return vs Start")
        ax_top.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax_top.yaxis.set_major_locator(MultipleLocator(0.10))
        ax_top.grid(axis="y", which="major", alpha=0.35, linestyle="--")
        ax_top.xaxis.set_major_locator(mdates.YearLocator())
        ax_top.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax_top.grid(axis="x", which="major", alpha=0.25)
        ax_top.legend(loc="best")
        ax_dd.fill_between(dd.index, dd.values, 0.0, step=None, alpha=0.5)
        ax_dd.set_ylabel("Drawdown")
        ax_dd.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax_dd.grid(axis="y", which="major", alpha=0.35, linestyle="--")
        ax_dd.xaxis.set_major_locator(mdates.YearLocator())
        ax_dd.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax_dd.grid(axis="x", which="major", alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(out_port, "portfolio_equity_rebased.png"), bbox_inches="tight")
        plt.close(fig)

        # Build df_port for compute_statistics and saving
        cummax = port_equity.cummax()
        raw_dd = (cummax - port_equity) / cummax
        df_port = pd.DataFrame({
            'date': port_equity.index,
            'position': (pd.DataFrame(scaled_by_symbol).abs().sum(axis=1) > 0).astype(int).reindex(
                port_equity.index).fillna(0).values,
            'equity': port_equity.values,
            'drawdown': raw_dd.values,
            'delta_pos': (pd.DataFrame(scaled_by_symbol).diff().abs().sum(axis=1)).reindex(port_equity.index).fillna(
                0.0).values,
            'pnl': portfolio_pnl.reindex(port_equity.index).fillna(0.0).values,
            'sample': 'OOS',
            'bundle': 1
        })
        df_port.to_csv(os.path.join(out_port, 'details_all_bundles.csv'), index=False)
        # NEW: save integrity checks + components-at-maxDD diagnostics
        _save_portfolio_integrity_checks(out_port, df_port, pnl_by_symbol, port_equity)
        _save_components_at_maxdd(out_port, port_equity, per_inst_results, symbols)

        # Stats via your central function
        portfolio_stats = compute_statistics(df_port, out_port, config=base_cfg)
        port_rets = port_equity.pct_change().fillna(0.0)

        # === Portfolio Best/Worst saving (used by Excel BestWorst sheet) ===
        period_res = top_and_bottom_periods(
            port_rets,
            resolution=base_cfg.get("statistics", {}).get("period_resolution", "monthly"),
            n=5
        )
        pd.DataFrame(period_res["table"]).to_csv(os.path.join(out_port, "period_stats.csv"), index=False)
        pd.DataFrame(period_res["top"]).to_csv(os.path.join(out_port, "top_periods.csv"), index=False)
        pd.DataFrame(period_res["bottom"]).to_csv(os.path.join(out_port, "bottom_periods.csv"), index=False)

        return port_equity, port_rets, portfolio_stats


def plot_equity_drawdown_curves(per_inst_results: list, instrument_names: list, instruments: list,
                                run_out: str, save_per_asset: bool, symbol_counts: Counter):
    """
    Plot and save equity and drawdown curves for all instruments (OOS segments) and for the portfolio.
    """
    if len(per_inst_results) == 0:
        return
    # 1. Equity curves for all instruments (each OOS bundle normalized to start at 1.0)
    fig, ax = plt.subplots()
    for name, df_inst in zip(instrument_names, per_inst_results):
        df_temp = df_inst.copy()
        df_temp['date'] = pd.to_datetime(df_temp['date'])
        df_temp.set_index('date', inplace=True)
        # Plot each OOS bundle equity curve normalized to 1 at the beginning of that bundle
        for b, grp in df_temp[df_temp['sample'] == 'OOS'].groupby('bundle'):
            if grp['equity'].iloc[0] == 0:
                continue  # avoid division by zero if equity starts at 0
            norm_equity = grp['equity'] / grp['equity'].iloc[0]
            ax.plot(norm_equity.index, norm_equity.values, label=f"{name} (Bundle {int(b)})")
    ax.set_title("Equity Curves (OOS bundles normalized to 1)")
    ax.set_ylabel("Normalized Equity")
    ax.set_xlabel("Date")
    if len(instrument_names) <= 3:
        ax.legend(fontsize='x-small')
    fig.savefig(os.path.join(run_out, "equity_all_bundles.png"), bbox_inches='tight')
    plt.close(fig)
    print("equity_all_bundles.png saved")
    # 2. Drawdown curves for all instruments
    fig, ax = plt.subplots()
    for name, df_inst in zip(instrument_names, per_inst_results):
        df_temp = df_inst.copy()
        df_temp['date'] = pd.to_datetime(df_temp['date'])
        df_temp.set_index('date', inplace=True)
        dd = df_temp['drawdown'].fillna(0.0)
        ax.plot(dd.index, dd.values, label=name)
    ax.set_title("Drawdowns (OOS)")
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")
    if len(instrument_names) <= 5:
        ax.legend(fontsize='x-small')
    fig.savefig(os.path.join(run_out, "drawdown_all_bundles.png"), bbox_inches='tight')
    plt.close(fig)
    print("drawdown_all_bundles.png saved")
    # 3. (Optional) Per-asset equity curves in each asset's subdirectory
    if save_per_asset:
        for inst, df_inst in zip(instruments, per_inst_results):
            sym = inst['symbol']
            mod = inst['strategy']['module']
            subfolder = os.path.join(run_out, f"{sym}_{mod}" if symbol_counts[sym] > 1 else sym)
            if os.path.isdir(subfolder):
                fig, ax = plt.subplots()
                df_temp = df_inst.copy()
                df_temp['date'] = pd.to_datetime(df_temp['date'])
                df_temp.set_index('date', inplace=True)
                for b, grp in df_temp[df_temp['sample'] == 'OOS'].groupby('bundle'):
                    if grp['equity'].iloc[0] == 0:
                        continue
                    norm_equity = grp['equity'] / grp['equity'].iloc[0]
                    ax.plot(norm_equity.index, norm_equity.values, label=f"Bundle {int(b)}")
                ax.set_title(f"{sym} Equity (OOS bundles)")
                ax.set_ylabel("Normalized Equity")
                ax.set_xlabel("Date")
                ax.legend(fontsize='x-small')
                fig.savefig(os.path.join(subfolder, "equity_all_bundles.png"), bbox_inches='tight')
                plt.close(fig)


def compute_correlation_analysis(per_inst_results: list, instrument_names: list, instruments: list, run_out: str):
    """
    Compute and save correlation matrices for asset-level and strategy-level OOS returns.
    """
    if len(instrument_names) <= 1:
        return  # need at least 2 instruments to compute correlation
    # Compute daily OOS returns for each instrument (concatenate all OOS bundles)
    returns_dict = {}
    for name, df_inst in zip(instrument_names, per_inst_results):
        df_temp = df_inst.copy()
        df_temp['date'] = pd.to_datetime(df_temp['date'])
        df_temp.set_index('date', inplace=True)
        oos_groups = df_temp[df_temp['sample'] == 'OOS'].groupby('bundle')
        # Concatenate returns from all OOS periods for this instrument
        oos_rets = [grp['equity'].pct_change().dropna() for _, grp in oos_groups]
        if oos_rets:
            returns_dict[name] = pd.concat(oos_rets)
    if not returns_dict:
        return
    returns_df = pd.DataFrame(returns_dict)
    # Asset return correlation
    asset_corr = returns_df.corr().round(4)
    asset_corr.to_csv(os.path.join(run_out, "asset_correlation.csv"))
    # Strategy-level return correlation (if multiple strategies are present)
    strat_set = {inst['strategy']['module'] for inst in instruments}
    if len(strat_set) > 1:
        inst_to_strat = {name: inst['strategy']['module'] for name, inst in zip(instrument_names, instruments)}
        strat_returns = {}
        for strat in strat_set:
            strat_cols = [col for col in returns_df.columns if inst_to_strat[col] == strat]
            if not strat_cols:
                continue
            # Average the returns of all instruments for each strategy (treat missing as 0)
            strat_series = returns_df[strat_cols].fillna(0.0).mean(axis=1)
            strat_series.name = strat
            strat_returns[strat] = strat_series
        strat_df = pd.DataFrame(strat_returns)
        strat_corr = strat_df.corr().round(4)
        strat_corr.to_csv(os.path.join(run_out, "strategy_correlation.csv"))


def run_statistical_tests(instruments: list, weights: list, symbol_counts: Counter, total_capital: float,
                          run_out: str, base_cfg: dict, save_per_asset: bool,
                          fees_map: dict, tests_to_run: set = None):
    """
    Run post-backtest statistical tests (permutation tests, partition return tests) as requested.
    Writes results to CSV files in the appropriate output folders.
    """
    perms = base_cfg.get('tests', {}).get('permutations', 1000)  # number of permutations/bootstraps
    # Map instrument key to its output folder (for tests that need to read per-asset files)
    inst_folders = {}
    for inst in instruments:
        sym = inst['symbol']; mod = inst['strategy']['module']
        folder = run_out
        if save_per_asset:
            # Use subfolder if per-asset outputs were saved
            folder_name = f"{sym}_{mod}" if symbol_counts[sym] > 1 else sym
            folder = os.path.join(run_out, folder_name)
        inst_folders[f"{sym}_{mod}"] = folder
    # Determine which tests to run (either from provided set or via interactive prompts)
    if tests_to_run is None or len(tests_to_run) == 0:
        # Interactive mode: ask user for each test
        def prompt(test_num, description):
            resp = input(f"Run Test {test_num} ({description})? [y/N]: ")
            return resp.strip().lower().startswith('y')
        run_test1 = prompt(1, "OOS bundle permutation for each asset")
        run_test2 = prompt(2, "training-process overfit for each asset")
        run_test5 = prompt(5, "multiple-system selection bias (portfolio-wide)")
        run_test6 = prompt(6, "partition return test for each asset")
    else:
        # Non-interactive: based on --run-tests flags
        run_test1 = 1 in tests_to_run
        run_test2 = 2 in tests_to_run
        run_test5 = 5 in tests_to_run
        run_test6 = 6 in tests_to_run
    # Test 1: Permutation test on OOS returns (per asset)
    if run_test1:
        for inst, w in zip(instruments, weights):
            inst_cfg = make_instrument_config(base_cfg, inst)
            sym = inst['symbol']; mod = inst['strategy']['module']
            folder = inst_folders[f"{sym}_{mod}"]
            p_val = test1_permutation_oos(inst_cfg, folder, w, B=perms)
            pd.DataFrame({'oos_bundle_p': [p_val]}).to_csv(os.path.join(folder, 'permutation_test_oos.csv'), index=False)
            print(f"  ✔ {sym} - {mod}: Test 1 complete (p={p_val:.3f})")
    # Test 2: Permutation test for training process overfit (per asset)
    if run_test2:
        for inst in instruments:
            inst_cfg = make_instrument_config(base_cfg, inst)
            sym = inst['symbol']; mod = inst['strategy']['module']
            folder = inst_folders[f"{sym}_{mod}"]
            p_val = test2_permutation_training(inst_cfg, folder, B=perms)
            pd.DataFrame({'training_overfit_p': [p_val]}).to_csv(os.path.join(folder, 'permutation_test_training.csv'), index=False)
            print(f"  ✔ {sym} - {mod}: Test 2 complete (p={p_val:.3f})")
    # Test 5: Multiple-system selection bias test (portfolio-wide)
    if run_test5:
        df5 = permutation_test_multiple(inst_folders, B=perms)
        out_csv = os.path.join(run_out, 'permutation_test_multiple.csv')
        df5.to_csv(out_csv, index=False)
        print(f"  ✔ Test 5 complete (results saved to {out_csv})")
    # Test 6: Partition return test (per asset)
    if run_test6:
        drift_rate = base_cfg.get('tests', {}).get('drift_rate', 0.0)
        for inst, w in zip(instruments, weights):
            sym = inst['symbol']; mod = inst['strategy']['module']
            folder = inst_folders[f"{sym}_{mod}"]
            # 6a. Load full price data for the asset
            dl = DataLoader(data_dir=base_cfg['data']['path'],
                            symbol=sym,
                            timeframe=base_cfg['data']['timeframe'],
                            base_timeframe=base_cfg['data'].get('base_timeframe'))
            df_price = dl.load()
            df_price['price_change'] = df_price['close'].pct_change().fillna(0.0)
            # 6b. Identify best overall parameters from the full backtest results
            res = pd.read_csv(os.path.join(folder, 'results.csv'))
            pnl_cols = [c for c in res.columns if c.lower().endswith('_pnl')]
            if not pnl_cols:
                continue  # skip if no PnL column found
            best_idx = res[pnl_cols[0]].idxmax()
            best_params = {k: res.loc[best_idx, k] for k in res.columns
                           if k not in ('bundle',) and not k.startswith('oos_')}
            if 'vol_window' in best_params:
                best_params['vol_window'] = int(best_params['vol_window'])
            # 6c. Generate positions on full data using best params
            strat_mod = import_module(f"strategies.{inst['strategy']['module']}")
            strat_fn = getattr(strat_mod, inst['strategy']['function'])
            best_params['capital'] = total_capital * w  # allocate capital fraction to this instrument

            safe_best_params = filter_params_for_callable(strat_fn, best_params)

            pos = strat_fn(df_price, **safe_best_params)['position']
            df_price['position'] = pos.values if hasattr(pos, 'values') else pos
            # 6d. Define a helper to compute total return for any given subset of price data
            inst_cap = total_capital * w
            fee = fees_map[sym]
            def inst_backtest(sub_df):
                pnl = simulate_pnl(
                    positions=sub_df['position'],
                    price=sub_df['close'],
                    multiplier=best_params.get('multiplier', 1.0),
                    fx=best_params.get('fx', 1.0),
                    capital=inst_cap,
                    commission_usd=fee['commission_usd'],
                    commission_pct=fee['commission_pct'],
                    slippage_pct=fee['slippage_pct'],
                )
                equity = (1 + pnl['pnl'] / inst_cap).cumprod() * inst_cap
                return equity.iloc[-1] / equity.iloc[0] - 1  # total return over the period
            # 6e. Run partition return analysis
            pr = partition_return(inst_backtest, df_price, drift_rate=drift_rate, oos_start=0, B=perms)
            pd.DataFrame([pr]).to_csv(os.path.join(folder, 'partition_return.csv'), index=False)
            trend = pr.get('trend', 0.0); bias = pr.get('mean_bias', 0.0); skill = pr.get('skill', 0.0)
            print(f"  ✔ {sym}: Test 6 complete (trend={trend:.1f}, bias={100*bias:.2f}%, skill={skill:.2f})")


def generate_summary_md(run_out: str, cfg: dict, portfolio_cfg: dict, save_per_asset: bool, instruments: list):
    """
    Generate a Markdown summary file (summary.md) compiling key results, statistics, and charts.
    """
    md = []
    md.append(f"# Backtest Summary: `{os.path.basename(run_out)}`")
    md.append(f"**Run date:** {datetime.now():%Y-%m-%d %H:%M}")
    md.append("")  # blank line
    md.append("**Contents:**")
    md.append("- [1. Combined Statistics](#1-combined-statistics)")
    md.append("- [2. Per-Asset Permutation Tests](#2-per-asset-permutation-tests)")
    md.append("- [3. Multiple-System Selection Bias](#3-multiple-system-selection-bias)")
    md.append("- [4. Key Charts](#4-key-charts)")
    md.append("- [5. Correlation Analysis](#5-correlation-analysis)")
    md.append("")
    # 1. Combined Statistics (table from combined_stats.csv)
    combined_csv = os.path.join(run_out, "combined_stats.csv")
    if os.path.exists(combined_csv):
        df = pd.read_csv(combined_csv, index_col=0).rename_axis("Instrument").reset_index()
        # Place Portfolio last
        df = pd.concat([df[df["Instrument"] != "Portfolio"], df[df["Instrument"] == "Portfolio"]], ignore_index=True)
        # Rename some columns for readability
        df = df.rename(columns={
            "ret_5pct": "5th pctile", "ret_95pct": "95th pctile",
            "avg_cost_pct": "Cost %/Trade", "sharpe_no_cost": "Sharpe (no cost)"
        })
        # Format percentage and float columns for markdown
        pct_cols = ["cagr", "annual_vol", "max_drawdown", "avg_drawdown", "win_rate",
                    "5th pctile", "95th pctile", "avg_win", "avg_loss", "max_loss_pct", "Cost %/Trade"]
        for c in pct_cols:
            if c in df:
                df[c] = df[c].apply(lambda x: f"{100*x:.1f}%" if pd.notnull(x) else "N/A")
        num_cols = ["sharpe", "sortino", "profit_factor", "expectancy", "std_daily", "Sharpe (no cost)"]
        for c in num_cols:
            if c in df:
                df[c] = df[c].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
        # Bold the best Sharpe ratio (excluding Portfolio row)
        if "sharpe" in df.columns:
            non_port = df[df.Instrument != "Portfolio"]["sharpe"].astype(float)
            if not non_port.empty:
                best_idx = non_port.idxmax()
                df.loc[best_idx, "Instrument"] = f"**{df.loc[best_idx, 'Instrument']}**"
        # Convert DataFrame to markdown table
        def df_to_markdown(df_table):
            cols = [str(c) for c in df_table.columns]
            header = "| " + " | ".join(cols) + " |"
            sep = "| " + " | ".join("---" for _ in cols) + " |"
            rows = ["| " + " | ".join(str(x) for x in row) + " |" for row in df_table.values]
            return "\n".join([header, sep] + rows)
        md.append("## 1. Combined Statistics")
        md.append(df_to_markdown(df))
        md.append("")
    # 2. Per-Asset Permutation Tests (Test 1, 2, 6 results per asset)
    rows = []
    for inst in instruments:
        sym = inst["symbol"]; mod = inst["strategy"]["module"]
        folder = save_per_asset and os.path.join(run_out, f"{sym}_{mod}") or run_out
        # Load p-values or metrics from each test's output file if present
        p1 = None; p2 = None; part = None
        p1_path = os.path.join(folder, "permutation_test_oos.csv")
        p2_path = os.path.join(folder, "permutation_test_training.csv")
        pr_path = os.path.join(folder, "partition_return.csv")
        if os.path.exists(p1_path):
            p1 = pd.read_csv(p1_path).iloc[0, 0]
        if os.path.exists(p2_path):
            p2 = pd.read_csv(p2_path).iloc[0, 0]
        if os.path.exists(pr_path):
            part = pd.read_csv(pr_path).iloc[0]
        # Extract partition return metrics if available
        trend = float(part["trend"]) if part is not None else None
        bias = float(part["mean_bias"]) if part is not None else None
        skill = float(part["skill"]) if part is not None else None
        rows.append({
            "Instrument": f"{sym}-{mod}",
            "Test 1 p": f"{p1:.3f}" if p1 is not None else "N/A",
            "Test 2 p": f"{p2:.3f}" if p2 is not None else "N/A",
            "Trend": f"{trend:.1f}" if trend is not None else "N/A",
            "Bias": f"{(100*bias):.2f}%" if bias is not None else "N/A",
            "Skill": f"{skill:.2f}" if skill is not None else "N/A"
        })
    if rows:
        df_tests = pd.DataFrame(rows)
        md.append("## 2. Per-Asset Permutation Tests")
        md.append(df_tests.to_markdown(index=False))  # DataFrame to markdown (using pandas' to_markdown for brevity)
        md.append("")
    # 3. Multiple-System Selection Bias (Test 5 results)
    multi_path = os.path.join(run_out, "permutation_test_multiple.csv")
    if os.path.exists(multi_path):
        df5 = pd.read_csv(multi_path)
        # Split "System" into Instrument and Strategy for clarity
        if "System" in df5.columns:
            df5[["Instrument", "Strategy"]] = df5["System"].str.split('_', n=1, expand=True)
        # Format numeric columns
        for col in df5.select_dtypes(include='number').columns:
            df5[col] = df5[col].map(lambda x: f"{x:.3f}")
        # Reorder columns if applicable
        if "Instrument" in df5.columns and "Strategy" in df5.columns:
            cols = ["Instrument", "Strategy"] + [c for c in df5.columns if c not in ("Instrument", "Strategy", "System")]
            df5 = df5[cols]
        md.append("## 3. Multiple-System Selection Bias")
        md.append(df5.to_markdown(index=False))
        md.append("")
    # 4. Key Charts (embed image links for generated charts)
    md.append("## 4. Key Charts")
    charts = [
        ("Equity Curves",           "equity_all_bundles.png"),
        ("Drawdowns",               "drawdown_all_bundles.png"),
        ("Portfolio Equity",        "portfolio/portfolio_equity.png"),
        ("30-Bar Return Dist.",     "portfolio_30bar_return_distribution.png"),
        ("Asset Return Correlation","asset_correlation.csv"),
    ]
    # Note: correlation charts are not directly images, but we include correlation in section 5 below.
    for title, fname in charts:
        path = os.path.join(run_out, fname)
        if os.path.exists(path):
            if fname.endswith(".png"):
                md.append(f"### {title}")
                md.append(f"![{title}]({fname})")
                md.append("")
    # 5. Correlation Analysis (embed correlation tables if present)
    asset_corr_path = os.path.join(run_out, "asset_correlation.csv")
    if os.path.exists(asset_corr_path):
        md.append("## 5. Correlation Analysis")
        # Strategy correlation (if file exists)
        strat_corr_path = os.path.join(run_out, "strategy_correlation.csv")
        if os.path.exists(strat_corr_path):
            strat_corr = pd.read_csv(strat_corr_path, index_col=0)
            md.append("### Strategy Return Correlation")
            md.append(strat_corr.reset_index().to_markdown(index=False, floatfmt=".2f"))
            md.append("")
        asset_corr = pd.read_csv(asset_corr_path, index_col=0)
        md.append("### Asset Return Correlation")
        md.append(asset_corr.reset_index().to_markdown(index=False, floatfmt=".2f"))
        md.append("")
    # Write the summary.md file
    with open(os.path.join(run_out, "summary.md"), "w") as f:
        f.write("\n\n".join(md))
    print(f"Generated summary.md → {os.path.join(run_out, 'summary.md')}")


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log))
    cfg = load_config(args.config)

    output_cfg = cfg.get('output', {}) or {}
    save_per_instrument = bool(output_cfg.get('save_per_instrument', True))
    save_per_strategy = bool(output_cfg.get('save_per_strategy', True))
    plot_per_strategy = bool((output_cfg.get('plots', {}) or {}).get('per_strategy', True))
    strategy_inputs = []  # feed to aggregate_by_strategy at the end

    TIMEFRAME_ALIASES = {
        "d": "1d", "1d": "1d", "daily": "1d",
        "h": "1h", "1h": "1h", "hour": "1h",
        "w": "1w", "1w": "1w", "week": "1w",
    }

    def _norm_tf(s: str) -> str:
        return TIMEFRAME_ALIASES.get(str(s).lower(), str(s))

    cfg['data']['timeframe'] = _norm_tf(cfg['data'].get('timeframe', '1d'))

    # Load fee mappings (e.g., commission/slippage per asset)
    with open(cfg['fees']['mapping']) as f:
        fees_map = json.load(f)

    fx_cfg = cfg.get('fx', {})
    base_ccy = fx_cfg.get('base_currency', 'USD')
    symbol_ccy_map = load_symbol_currency_map(fx_cfg.get('symbol_currency_map_file', ''))
    fx_dir = fx_cfg.get('pairs_csv_dir', 'fx_data')

    multipliers = {}
    mm = cfg.get('market_meta', {}).get('multipliers_file')
    if mm and os.path.exists(mm):
        with open(mm, 'r') as f:
            multipliers = json.load(f)

    # Set up output directory for this run
    base_out = cfg['output']['root']
    run_name = datetime.now().strftime("%H:%M %d.%m.%Y")
    run_out = os.path.join(base_out, run_name)
    os.makedirs(run_out, exist_ok=True)

    # Save the used configuration for reference
    with open(os.path.join(run_out, "config_used.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Running backtest. Output directory: {run_out}")
    save_per_asset = bool(cfg['output'].get('save_per_asset', 0))

    # Build instrument list from config (each instrument: symbol + strategy definition)
    portfolio_cfg = cfg['portfolio']
    assets = portfolio_cfg['assets']
    strat_defs = portfolio_cfg['strategies']
    assign_map = portfolio_cfg.get('assignments', {})
    default_strategies = assign_map.get('default', list(strat_defs.keys()))
    instruments = []
    for symbol in assets:
        strat_keys = assign_map.get(symbol, default_strategies)
        for sk in strat_keys:
            sdef = strat_defs[sk]
            inst = {
                'symbol': symbol,
                'strategy': {'module': sdef['module'], 'function': sdef['function']},
                'param_grid': sdef.get('param_grid', cfg['optimization']['param_grid']),
                'fees_mapping': cfg['fees']['mapping']
            }
            # Include optional multiplier/fx parameters if specified for the strategy
            if 'multiplier' in sdef:
                inst['strategy']['multiplier'] = sdef['multiplier']
            if 'fx' in sdef:
                inst['strategy']['fx'] = sdef['fx']
            instruments.append(inst)

    # ---- IDM: compute once per strategy (static) + save per-instrument CSV ----
    try:
        idm_map = compute_idm_map(cfg, instruments)  # returns {strategy_module: idm}
    except Exception as e:
        logging.warning("IDM computation failed; falling back to YAML idm scalars. Error: %s", e)
        idm_map = {}

    # Safety cap (even though compute_idm_map now caps internally)
    idm_map = {k: float(min(float(v), 2.5)) for k, v in (idm_map or {}).items()}

    # We'll save an instrument-level CSV so you can audit assignments.
    # Fallback IDM (if strategy missing from map)
    idm_fallback = float((cfg.get('strategy') or {}).get('idm', 1.0))

    rows = []
    for inst in instruments:
        sym = inst["symbol"]
        strat = inst["strategy"]["module"]
        idm_val = float(min(idm_map.get(strat, idm_fallback), 2.5))
        # Use same "instrument" label style as weights_resolved.csv for consistency
        instrument_label = f"{sym}|{strat}"
        rows.append({
            "instrument": instrument_label,
            "symbol": sym,
            "strategy": strat,
            "idm": idm_val,
        })

    try:
        idm_df = pd.DataFrame(rows)
        os.makedirs(run_out, exist_ok=True)
        idm_df.to_csv(os.path.join(run_out, "idm_resolved.csv"), index=False)
    except Exception as e:
        logging.warning("Could not save idm_resolved.csv: %s", e)


    # 1) Total capital from portfolio section
    total_capital = cfg['portfolio']['capital']

    # 2) Resolve weights using get_portfolio_weights
    weights_dict = get_portfolio_weights(cfg, [inst['symbol'] for inst in instruments])
    weights = [weights_dict.get(inst['symbol'], 0.0) for inst in instruments]

    # 3) Save resolved weights so you can audit later (OPTIONAL but recommended)
    resolved = pd.DataFrame(
        [{
            "instrument": f"{i['symbol']}|{i['strategy']['module']}",
            "symbol": i['symbol'],
            "strategy": i['strategy']['module'],
            "weight": w
        } for i, w in zip(instruments, weights)]
    )
    resolved.to_csv(os.path.join(run_out, "weights_resolved.csv"), index=False)

    symbol_counts = Counter(inst['symbol'] for inst in instruments)

    # Run backtest for each instrument and collect results
    inst_stats = {}
    per_inst_results = []
    instrument_names = []
    for inst, w in zip(instruments, weights):
        name, combined, stats = run_backtest_for_instrument(
            inst, cfg, portfolio_cfg, fees_map, w, total_capital, save_per_asset,
            run_out, symbol_counts,
            base_ccy, symbol_ccy_map, fx_dir, cfg['data']['timeframe'], multipliers, idm_map=idm_map,
            save_per_instrument=save_per_instrument,
            strategy_inputs=strategy_inputs,
        )
        instrument_names.append(name)
        per_inst_results.append(combined)
        if stats is not None:
            inst_stats[name] = stats

    if save_per_strategy and strategy_inputs:
        # Group results by strategy to compute aggregated stats per strategy
        inst_to_strat = {name: inst['strategy']['module'] for name, inst in zip(instrument_names, instruments)}
        strat_dir_base = os.path.join(run_out, "strategies")
        os.makedirs(strat_dir_base, exist_ok=True)
        stats_by_strategy = {}
        bestworst_by_strategy = {}
        # Compute daily OOS returns series for each instrument  ✅ FIXED: ensure DatetimeIndex
        returns_by_instrument = {}
        for name, df_inst in zip(instrument_names, per_inst_results):
            df_tmp = df_inst.copy()
            # ensure datetime index + OOS only
            df_tmp['date'] = pd.to_datetime(df_tmp['date'])
            oos = df_tmp.loc[df_tmp['sample'] == 'OOS'].sort_values('date')
            if oos.empty:
                continue
            oos_rets = []
            for _, grp in oos.groupby('bundle'):
                # set index to date for each bundle so pct_change has a DatetimeIndex
                s = grp.set_index('date')['equity'].pct_change().dropna()
                oos_rets.append(s)
            if oos_rets:
                # concat bundles on their datetime index
                returns_by_instrument[name] = pd.concat(oos_rets).sort_index()
        # Aggregate performance for each strategy
        for strat in sorted({inst['strategy']['module'] for inst in instruments}):
            # Collect returns of all instruments using this strategy
            strat_symbols = {name: ret for name, ret in returns_by_instrument.items() if inst_to_strat[name] == strat}
            if not strat_symbols:
                continue
            sdir = os.path.join(strat_dir_base, strat)
            os.makedirs(sdir, exist_ok=True)
            # Compute aggregated stats (treating all returns of this strategy as a portfolio)
            stats = aggregate_by_strategy(strat_symbols, strat, list(strat_symbols.keys()), sdir, cfg)
            # Rename output file for Excel integration (strategy_statistics.csv -> stats.csv)
            strat_stats_path = os.path.join(sdir, "strategy_statistics.csv")
            if os.path.exists(strat_stats_path):
                os.replace(strat_stats_path, os.path.join(sdir, "stats.csv"))
            stats_by_strategy[strat] = stats
            # Compute best/worst periods for this strategy’s returns
            period_res = top_and_bottom_periods(
                stats["series"]["returns"],
                resolution=cfg.get("statistics", {}).get("period_resolution", "monthly"),
                n=5
            )
            bestworst_by_strategy[strat] = period_res
            if period_res:
                period_res["table"].to_csv(os.path.join(sdir, "period_stats.csv"), index=False)
                period_res["top"].to_csv(os.path.join(sdir, "top_periods.csv"), index=False)
                period_res["bottom"].to_csv(os.path.join(sdir, "bottom_periods.csv"), index=False)
            print(f"✅ Saved aggregated stats for strategy '{strat}' in {sdir}")

    # Aggregate all instrument results into portfolio results
    port_equity, port_rets, portfolio_stats = aggregate_portfolio(per_inst_results, instruments, instrument_names, symbol_counts, total_capital, run_out, portfolio_cfg, cfg)
    inst_stats["Portfolio"] = portfolio_stats

    # --- Build combined_stats.csv robustly (preserve all fields, incl. 30d metrics) ---
    rows = []
    for name, stats in inst_stats.items():
        # normalize to plain dict
        stats = stats.to_dict() if hasattr(stats, "to_dict") else dict(stats)
        row = {"name": name}
        row.update(stats)
        rows.append(row)

    combined_stats_df = pd.DataFrame(rows).set_index("name")

    # Optional: keep a preferred order, then append any extras automatically
    BASE_COLS = [
        "cagr",
        'cagr_geom',
        'cagr_equity'
        'mean_annual_return',
        'annualised_return_log',
        'total_return',
        "annual_vol",
        "sharpe",
        "sortino",
        "skew",
        "max_drawdown",
        "avg_drawdown",
        "avg_dd_duration",
        "profit_factor",
        "expectancy",
        'expectancy_usd'
        'expectancy_pct'
        'expectancy'
        "win_rate",
        "std_daily",
        "ret_5pct",
        "ret_95pct",
        "avg_win",
        "avg_loss",
        "max_loss_pct",
    ]
    EXTRA_30D = [
        "avg_30d_ret",
        "avg_30d_ret_plus_2std",
        "avg_30d_ret_minus_2std",
        "avg_30d_ret_ci_low",
        "avg_30d_ret_ci_high",
    ]
    ordered = [c for c in BASE_COLS if c in combined_stats_df.columns] \
              + [c for c in EXTRA_30D if c in combined_stats_df.columns] \
              + [c for c in combined_stats_df.columns if c not in set(BASE_COLS + EXTRA_30D)]
    combined_stats_df = combined_stats_df[ordered]

    # --- insert this block here ---
    for col in EXTRA_30D:
        if col not in combined_stats_df.columns:
            combined_stats_df[col] = np.nan
    if "Portfolio" in combined_stats_df.index:
        combined_stats_df.loc["Portfolio", EXTRA_30D] = [
            portfolio_stats.get(k, np.nan) for k in EXTRA_30D
        ]
    # ------------------------------

    # logging.info("combined_stats cols: %s", combined_stats_df.columns.tolist())
    # logging.info(
    #     "Portfolio row 30d: %s",
    #     combined_stats_df.loc["Portfolio", [c for c in combined_stats_df.columns if "30d" in c]].to_dict()
    # )

    combined_stats_path = os.path.join(run_out, "combined_stats.csv")
    combined_stats_df.to_csv(combined_stats_path)
    print(f"✅ combined_stats.csv created → {combined_stats_path}")

    # Generate equity and drawdown charts
    plot_equity_drawdown_curves(per_inst_results, instrument_names, instruments, run_out, save_per_asset, symbol_counts)

    # Plot portfolio 30-bar return distribution and save
    df_eq = pd.DataFrame({'equity': port_equity.values}, index=port_equity.index)
    df_eq['grp'] = np.arange(len(df_eq)) // 30  # group index for each 30-bar period
    first = df_eq.groupby('grp')['equity'].first()
    last = df_eq.groupby('grp')['equity'].last()
    port_monthly = (last / first - 1).dropna()
    mean_pm = port_monthly.mean()
    std_pm = port_monthly.std()
    ci_low = mean_pm - 1.96 * std_pm / np.sqrt(len(port_monthly)) if len(port_monthly) > 0 else 0.0
    ci_high = mean_pm + 1.96 * std_pm / np.sqrt(len(port_monthly)) if len(port_monthly) > 0 else 0.0
    fig, ax = plt.subplots()
    ax.hist(port_monthly, bins=30, density=True, edgecolor='black', alpha=0.6)
    ax.axvspan(ci_low, ci_high, color='orange', alpha=0.3, label='95% CI')
    ax.axvline(mean_pm, color='black', linestyle='--', label='Mean')
    ax.set_title("Portfolio 30-Bar Return Distribution (95% CI)")
    ax.set_xlabel("30-bar Return")
    ax.set_ylabel("Density")
    ax.legend()
    fig.savefig(os.path.join(run_out, 'portfolio_30bar_return_distribution.png'), bbox_inches='tight')
    plt.close(fig)
    print("30-bar return distribution chart saved")

    # Compute and save correlation analysis results
    compute_correlation_analysis(per_inst_results, instrument_names, instruments, run_out)

    # Run statistical tests if specified by --run-tests (or via interactive prompt)
    tests_to_run = {int(x) for x in args.run_tests.split(',') if x}
    run_statistical_tests(instruments, weights, symbol_counts, total_capital, run_out, cfg, save_per_asset, fees_map,
                          tests_to_run if tests_to_run else None)

    # Recompute comprehensive Portfolio stats (CAGR, Sharpe, etc.) and update combined_stats.csv
    cummax = port_equity.cummax()
    raw_dd = (cummax - port_equity) / cummax

    # Calculate drawdown durations
    durations = []
    in_draw = False
    current_dur = 0
    for dd in raw_dd:
        if dd > 0:
            in_draw = True
            current_dur += 1
        else:
            if in_draw and current_dur > 0:
                durations.append(current_dur)
            in_draw = False
            current_dur = 0
    if in_draw and current_dur > 0:
        durations.append(current_dur)
    rets = port_rets  # daily returns series

    rets = _clean_returns_series(rets)
    cagr = _safe_cagr_from_returns(rets)
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else np.nan
    sortino = (rets.mean() / rets[rets < 0].std() * np.sqrt(252)) if len(rets[rets < 0]) > 0 else np.nan
    max_dd = raw_dd.max() if len(raw_dd) > 0 else 0.0
    avg_dd = raw_dd[raw_dd > 0].mean() if (raw_dd > 0).any() else 0.0
    avg_dd_dur = np.mean(durations) if durations else 0
    # Trade-level statistics across the entire portfolio history
    pos_df = pd.concat([df.set_index('date')['position'].rename(name)
                        for name, df in zip(instrument_names, per_inst_results)], axis=1).fillna(0.0)
    open_pos = pos_df.abs().sum(axis=1) > 0
    pnl_df = pd.concat([df.set_index('date')['pnl'].rename(name)
                        for name, df in zip(instrument_names, per_inst_results)], axis=1).fillna(0.0)
    total_pnl = pnl_df.sum(axis=1).loc[open_pos]
    trades = total_pnl.diff().abs().gt(0).sum()
    wins = total_pnl[total_pnl > 0].sum()
    loss = -total_pnl[total_pnl < 0].sum()
    profit_factor = wins / loss if loss > 0 else np.nan
    expectancy = total_pnl.sum() / trades if trades > 0 else np.nan
    win_rate = total_pnl.gt(0).sum() / trades if trades > 0 else np.nan

    # Only overwrite the fields you recompute here; preserve any others already present
    _update = {
        'cagr': cagr,
        'annual_vol': ann_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_dd,
        'avg_drawdown': avg_dd,
        'avg_dd_duration': avg_dd_dur,
        'profit_factor': profit_factor,
        'expectancy': expectancy,
        'win_rate': win_rate,
        'std_daily': rets.std(),
    }

    if 'Portfolio' in combined_stats_df.index:
        for k, v in _update.items():
            combined_stats_df.loc['Portfolio', k] = v
    else:
        # If somehow missing, create the row with NaNs then set the known fields
        combined_stats_df.loc['Portfolio'] = np.nan
        for k, v in _update.items():
            combined_stats_df.loc['Portfolio', k] = v

    combined_stats_df.to_csv(combined_stats_path)
    print("✅ combined_stats.csv updated with final Portfolio stats (fields merged, 30d metrics preserved)")

    # ---- Aggregate per-instrument forecast scales into run_out/forecast_scale_resolved.csv ----
    try:
        rows = []
        for d in sorted(os.listdir(run_out)):
            sub = os.path.join(run_out, d)
            if not os.path.isdir(sub) or d in ("portfolio", "strategies"):
                continue
            f = os.path.join(sub, "forecast_scale.csv")
            if os.path.exists(f):
                df_fs = pd.read_csv(f)
                if df_fs is not None and not df_fs.empty:
                    rows.append(df_fs)
        if rows:
            fs_all = pd.concat(rows, ignore_index=True)
        else:
            # create an empty file with headers so downstream joins don't fail
            fs_all = pd.DataFrame(columns=["instrument", "symbol", "strategy", "forecast_scale"])
        fs_all.to_csv(os.path.join(run_out, "forecast_scale_resolved.csv"), index=False)
    except Exception as e:
        logging.warning("Could not aggregate forecast_scale.csv files: %s", e)

    # Generate Markdown summary report
    generate_summary_md(run_out, cfg, portfolio_cfg, save_per_asset, instruments)

    try:
        xlsx_path = export_summary_xlsx(run_out)
        print(f"📊 summary.xlsx created → {xlsx_path}")
    except Exception as e:
        logging.warning("Excel export failed: %s", e)

    # ---- Tidy: move per-instrument folders into run_out/instruments ----
    try:
        dest_root = os.path.join(run_out, "instruments")
        os.makedirs(dest_root, exist_ok=True)

        # Folders we should not move
        exclude = {"portfolio", "strategies"}

        for name in sorted(os.listdir(run_out)):
            src = os.path.join(run_out, name)
            if not os.path.isdir(src):
                continue
            if name in exclude:
                continue
            # Heuristic: instrument folders use "asset_strategy" naming
            if "_" not in name:
                continue

            # Make sure it's really an instrument result folder (has marker files)
            has_marker = any(
                os.path.exists(os.path.join(src, fn))
                for fn in (
                    "details_all_bundles.csv",
                    "stats.csv",
                    "permutation_test_oos.csv",
                    "positions.csv",
                )
            )
            if not has_marker:
                continue

            dst = os.path.join(dest_root, name)
            # Overwrite if already exists (e.g., re-runs)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.move(src, dst)

        logging.info("Moved per-instrument folders into: %s", dest_root)
    except Exception as e:
        logging.warning("Tidy-up move of per-instrument folders failed: %s", e)



if __name__ == '__main__':
    main()
