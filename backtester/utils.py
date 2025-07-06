import pandas as pd
from importlib import import_module
from backtester.pnl_engine import simulate_pnl


def run_strategy(df: pd.DataFrame, config: dict, **params) -> dict:
    # load strategy fn
    mod_name  = config['strategy']['module']
    fn_name   = config['strategy']['function']
    strat_mod = import_module(f"backtester.strategies.{mod_name}")
    strat_fn  = getattr(strat_mod, fn_name)

    # drop internal keys
    strat_params = {k: v for k, v in params.items() if not k.startswith('_')}

    # 1) positions
    pos_df = strat_fn(df, **strat_params)

    # 2) simulate pnl/equity/drawdown
    pnl_df = simulate_pnl(
        positions=pos_df['position'],
        price=df['close'],
        multiplier=strat_params.get('multiplier', 1.0),
        fx=strat_params.get('fx', 1.0),
        capital=strat_params.get('capital', 100_000),
        commission_usd=strat_params.get('commission_usd', 0.0),
        commission_pct=strat_params.get('commission_pct', 0.0),
        slippage_pct=strat_params.get('slippage_pct', 0.0),
    )

    # 3) join only the PnL columns that exist
    wanted = [
        'pos_value',
        'delta_pos',
        'cost_usd',
        'cost_pct',
        'slip_cost',
        'pnl',
        'equity',
        'drawdown',
    ]
    avail = pnl_df.columns.intersection(wanted)
    diag = pos_df.join(pnl_df[avail], how='left')

    # 4) compute summary metrics
    pnl_series    = diag['pnl']
    equity_series = diag['equity']
    dd_series     = diag['drawdown']

    win    = pnl_series[pnl_series>0].sum()
    loss   = -pnl_series[pnl_series<0].sum()

    return {
        'pnl':            pnl_series.sum(),
        'sharpe':         (pnl_series.mean()/pnl_series.std()) * (252**0.5)
                          if pnl_series.std()>0 else 0.0,
        'max_drawdown':   dd_series.min(),
        'profit_factor':  (win/loss) if loss>0 else float('inf'),
        'start_equity':   equity_series.iloc[0],
        'end_equity':     equity_series.iloc[-1],
    }


