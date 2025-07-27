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
        pct_cols = ["cagr","annual_vol","max_drawdown","avg_drawdown","win_rate","5th pctile","95th pctile"]
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

    inst_stats      = {}
    per_inst_results = []

    # --- 3) Per‐instrument backtests ------------------------------------------
    for inst, w in zip(instruments, weights):
        symbol = inst['symbol']
        print(f"\n=== Running backtest for {symbol} (weight {w:.2%}) ===")

        # fees for this symbol
        fee_row = fees_map[symbol]
        comm_usd, comm_pct, slip_pct = (
            fee_row['commission_usd'],
            fee_row['commission_pct'],
            fee_row['slippage_pct'],
        )

        # prepare a copy of cfg for this instrument
        inst_cfg = deepcopy(cfg)
        inst_cfg['data']['symbol']             = symbol
        inst_cfg['fees']['mapping']            = inst['fees_mapping']
        inst_cfg['strategy']                   = inst['strategy']
        inst_cfg['optimization']['param_grid'] = inst['param_grid']

        # allocate the capital slice
        inst_capital = total_capital * w
        inst_cfg['optimization']['param_grid']['capital'] = [inst_capital]

        # create per‐asset folder if needed
        out_sub = os.path.join(run_out, symbol) if save_per_asset else run_out
        os.makedirs(out_sub, exist_ok=True)

        # load data
        dl_inst = DataLoader(
            data_dir       = inst_cfg['data']['path'],
            symbol         = symbol,
            timeframe      = inst_cfg['data']['timeframe'],
            base_timeframe = inst_cfg['data'].get('base_timeframe')
        )
        df_inst = dl_inst.load()
        start, end = inst_cfg['data'].get('start'), inst_cfg['data'].get('end')
        if start or end:
            df_inst = df_inst.loc[
                (start or df_inst.index[0]) : (end or df_inst.index[-1])
            ]
        logging.info(f"{symbol}: {len(df_inst):,} bars loaded")

        # run the walk‐forward backtest
        results_inst = run_backtest(df_inst, inst_cfg)
        results_inst.to_csv(os.path.join(out_sub, 'results.csv'), index=False)

        # collect OOS diagnostics
        splits_inst = list(generate_splits(df_inst, inst_cfg['optimization']['bundles']))
        diags = []
        strat_mod = import_module(f"backtester.strategies.{inst_cfg['strategy']['module']}")
        strat_fn  = getattr(strat_mod, inst_cfg['strategy']['function'])
        for (_, row), (_, test_df) in zip(results_inst.iterrows(), splits_inst):
            params = {
                k: row[k]
                for k in results_inst.columns
                if k not in ('bundle',) and not k.startswith('oos_') and not k.startswith('_')
            }
            if 'vol_window' in params:
                params['vol_window'] = int(params['vol_window'])

            pos_df = strat_fn(test_df, **params)
            pnl_df = simulate_pnl(
                positions      = pos_df['position'],
                price          = test_df['close'],
                multiplier     = params['multiplier'],
                fx             = params['fx'],
                capital        = inst_capital,
                commission_usd = comm_usd,
                commission_pct = comm_pct,
                slippage_pct   = slip_pct,
            )

            diags.append(
                pos_df.join(pnl_df, how='left', rsuffix='_pnl')
                      .assign(bundle=row['bundle'], sample='OOS')
            )

        combined_inst = pd.concat(diags).reset_index()
        combined_inst.to_csv(os.path.join(out_sub, 'details_all_bundles.csv'), index=False)
        print(f"{symbol}: details_all_bundles.csv saved")

        # plot OOS equity & drawdown
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

        # compute per‐asset stats and save
        stats = compute_statistics(combined_inst, out_sub)
        inst_stats[symbol] = stats

        # collect for portfolio aggregation
        per_inst_results.append(combined_inst)

        # optional advanced tests
        # resp = input(f"Run permutation & bootstrap tests for {symbol}? [y/N]: ").strip().lower()
        # if resp.startswith('y'):
        #     statistical_tests(
        #         combined_inst, out_sub,
        #         bootstrap_reps  = cfg.get('tests', {}).get('bootstrap_reps', 1000),
        #         permutation_reps= cfg.get('tests', {}).get('permutation_reps', 1000)
        #     )
        # else:
        #     print(f"Skipping tests for {symbol}.")

    # --- 4) PORTFOLIO AGGREGATION via weighted returns -----------------------
    # build per-asset return series
    rets_list = []
    symbols   = [inst['symbol'] for inst in instruments]
    for sym, df_inst in zip(symbols, per_inst_results):
        df_inst = df_inst.set_index('date')
        r = df_inst['equity'].pct_change().fillna(0).rename(sym)
        rets_list.append(r)
    rets_df = pd.concat(rets_list, axis=1).fillna(0)

    # construct weight vector
    w_map = portfolio_cfg.get('weights', {})
    w_vec = np.array([w_map.get(sym, 1.0/len(symbols)) for sym in symbols])
    w_vec /= w_vec.sum()

    # portfolio returns & equity
    port_rets  = rets_df.dot(w_vec)
    port_equity= (1 + port_rets).cumprod() * total_capital

    # save portfolio equity curve
    out_port = os.path.join(run_out, 'portfolio')
    os.makedirs(out_port, exist_ok=True)
    pd.DataFrame({'equity': port_equity}, index=port_equity.index) \
      .to_csv(os.path.join(out_port, 'portfolio_equity.csv'),
              index_label='date')
    fig, ax = plt.subplots()
    ax.plot(port_equity.index, port_equity.values)
    ax.set_title("Portfolio Equity")
    ax.set_ylabel("Equity")
    ax.set_xlabel("Date")
    fig.savefig(os.path.join(out_port, 'portfolio_equity.png'), bbox_inches='tight')
    plt.close(fig)
    print("Portfolio equity saved")

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

    # 5a) Compute drawdowns as (peak‐to‐trough)/peak
    cummax = port_equity.cummax()
    raw_dd = (cummax - port_equity) / cummax

    # 5b) Determine bars where ANY instrument is in market
    pos_df = pd.concat([
        df_inst.set_index('date')['position'].rename(sym)
        for sym, df_inst in zip(symbols, per_inst_results)
    ], axis=1)
    open_pos = pos_df.abs().sum(axis=1) > 0

    # 5c) Build a portfolio‐level diagnostics DataFrame just like per‐asset
    df_port = pd.DataFrame({
        'date': port_equity.index,
        'equity': port_equity.values,
        'drawdown': raw_dd.values,
        'delta_pos': open_pos.astype(int).astype(float).values,
        'cost_usd': 0.0,
        'cost_pct': 0.0,
        'slip_cost': 0.0,
        'pnl': port_rets * total_capital,
        'sample': 'OOS',
        'bundle': 1
    })

    # 5d) Save raw bundle‐1 details and produce all portfolio charts & CSVs
    df_port.to_csv(os.path.join(out_port, 'details_all_bundles.csv'), index=False)
    # This will drop into out_port:
    #   - strategy_statistics.csv
    #   - drawdown_distribution.png
    #   - dd_duration_vs_magnitude.png
    compute_statistics(df_port, out_port)

    # 5e) Book-style permutation tests

    # Safely fetch tests configuration (avoids KeyError if 'tests' is missing)
    # Use the portfolio diagnostics DataFrame as our series
    # price_data = df_port.set_index('date').copy()
    # price_data['price_change'] = price_data['equity'].pct_change().fillna(0)

    # Build a map of system‐name → its output folder
    # --- 4e) Book-Style Permutation Tests with confirmations ----------------

    inst_folders = {}
    for inst in instruments:
        name = f"{inst['symbol']}_{inst['strategy']['module']}"
        folder = save_per_asset and os.path.join(run_out, inst['symbol']) or run_out
        inst_folders[name] = folder
    perms = cfg.get('tests', {}).get('permutations', 1000)

    # Test 1: OOS bundle permutation (per-asset)
    if input("Run Test 1 (OOS bundle permutation) for each asset? [y/N]: ").strip().lower().startswith('y'):
        for inst in instruments:
            folder = save_per_asset and os.path.join(run_out, inst['symbol']) or run_out
            p1 = test1_permutation_oos(folder, B=perms)
            pd.DataFrame({'oos_bundle_p': [p1]}) \
                .to_csv(os.path.join(folder, 'permutation_test_oos.csv'), index=False)
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
