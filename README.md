## To run the software: 
   ```bash
python -m backtester.cli -c strategy_config.yaml
```

# Backtester

A modular Python CLI for walk-forward backtesting, diagnostics, and portfolio backtests.


## Quickstart

1. Rename the example config:
   mv config_example.yaml strategy_config.yaml
2. Edit strategy_config.yaml to set your data.symbol, strategy.module/function, optimization.param_grid, fees mapping, and (optionally) a portfolio: block.
3. Run a backtest:
   python -m backtester.cli -c strategy_config.yaml
4. Inspect your run folder under output.root/, which now contains:
   - bundle_summary.csv, results.csv, details_all_bundles.csv
   - PNGs for equity, drawdowns, distributions
   - strategy_statistics.csv
   - (if portfolio) portfolio_equity.csv, portfolio_equity.png
   - summary.md with embedded tables & charts

## Creating a New Strategy

1. In backtester/strategies/, create a new file, e.g. mystrat.py.
2. Define a function with this signature:
   def mystrat(df: pd.DataFrame, *, span_short: int, span_long: int, vol_window: int, ... ) -> pd.DataFrame:
       # must return a DataFrame with at least:
       #   - 'position' column: integer/float position size per bar
       #   - any extra series your strategy needs
3. Point your config at it:
   strategy:
     module: 'mystrat'
     function: 'mystrat'

## Configuring strategy_config.yaml

- data
  data:
    path:       './assets_data'
    symbol:     'SP500'
    timeframe:  '1d'
    start:      '2000-01-01'
    end:        '2020-12-31'
- fees: path to JSON mapping symbols -> {commission_usd, commission_pct, slippage_pct}
- strategy: which module/function to run
- optimization:
  optimization:
    bundles:    10
    target:     max_drawdown   # or sharpe, profit_factor
    param_grid:
      span_short:     [10,16,21]
      span_long:      [50,64,80]
      vol_window:     [21]
      forecast_scale: [3.0]
      # … other params …
- portfolio (optional):
  portfolio:
    instruments:
      - symbol:       'SP500'
        capital:      100000
        fees_mapping: './fees.json'
        strategy:
          module:    'ewmac'
          function:  'ewmac'
        param_grid: { … }
      - symbol:       'EURUSD'
        capital:      100000
        fees_mapping: './fees.json'
        strategy:
          module:    'ewmac'
          function:  'ewmac'
        param_grid: { … }
  If present, the CLI builds an equal-weight portfolio across these instruments by default.

## Adding New Statistics

All bundle- and aggregate-level metrics live in backtester/utils.py -> compute_statistics(). To add another metric:

1. Edit compute_statistics(): compute your new series (e.g. Calmar ratio), then include it in the per-bundle loop or aggregate summary.
2. Optionally add a new chart at the end of compute_statistics().

## Plug-in Weighting Systems

Weights are centralized in cli.py via:
def get_portfolio_weights(instruments):
    n = len(instruments)
    weights = [1.0/n] * n         # equal-weight
    total_cap = sum(
        inst.get('capital', 100_000)
        for inst in instruments
    )
    return weights, total_cap

To implement risk-parity, momentum-tilt, or custom rules, simply replace the body of get_portfolio_weights() with your logic.

## CLI Reference

python -m backtester.cli -c strategy_config.yaml [--log DEBUG]
- -c/--config: path to your YAML or JSON config
- --log: logging level (DEBUG, INFO, WARNING, ERROR)
- Future flags: --no-plots, --no-csv, --summary-only

