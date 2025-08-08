import json, pathlib, pandas as pd

run = pathlib.Path("/backtests/13:26 08.08.2025")  # e.g., runs/12:34 08.08.2025
cfg = json.loads((run/"config_used.json").read_text())
initial_capital = cfg["portfolio"]["capital"]

# collect per-instrument PnL (skip the portfolio folder)
series = []
for d in run.iterdir():
    if d.is_dir() and d.name != "portfolio" and (d/"details_all_bundles.csv").exists():
        df = pd.read_csv(d/"details_all_bundles.csv", parse_dates=["date"]).set_index("date")
        if "pnl" in df:
            s = df["pnl"].rename(d.name).fillna(0.0)
            series.append(s)

# if you saved per-instrument files elsewhere, adjust the discovery logic above
combined = pd.concat(series, axis=1).fillna(0.0) if series else pd.DataFrame()
portfolio_pnl = combined.sum(axis=1) if not combined.empty else pd.Series(dtype=float)

eq_rebuilt = portfolio_pnl.cumsum() + initial_capital
eq_saved = pd.read_csv(run/"portfolio"/"portfolio_equity.csv", parse_dates=["date"]).set_index("date")["equity"]

# align and compare
eq_rebuilt, eq_saved = eq_rebuilt.align(eq_saved, fill_value=0.0)
max_abs_diff = (eq_rebuilt - eq_saved).abs().max()
print("max abs diff:", float(max_abs_diff))

port_df = pd.read_csv(run/"portfolio"/"details_all_bundles.csv", parse_dates=["date"]).set_index("date")
eq = pd.read_csv(run/"portfolio"/"portfolio_equity.csv", parse_dates=["date"]).set_index("date")["equity"]
eq, port_df = eq.align(port_df, join="inner")

# daily equity change should equal daily PnL
max_abs_diff = (eq.diff().fillna(0.0) - port_df["pnl"].fillna(0.0)).abs().max()
print("max abs diff:", float(max_abs_diff))
