import pandas as pd


def simulate_pnl(
    positions: pd.Series,
    price:     pd.Series,
    multiplier: float,
    fx:         float,
    capital:    float = 100_000,
    commission_usd: float = 0.0,
    commission_pct: float = 0.0,
    slippage_pct:   float = 0.0,
) -> pd.DataFrame:
    """
    Simulate PnL, equity and drawdown with:
      - PnL[t] = Δprice[t] * position[t-1] * multiplier * fx
      - commission_usd per contract
      - commission_pct of notional on each trade
      - slippage_pct of notional on each trade
    """
    # 1) Compute Δprice * previous position
    price_diff = price.diff().fillna(0.0)
    prev_pos   = positions.shift(1).fillna(0.0)
    pnl_price  = price_diff * prev_pos * multiplier * fx

    # 2) Compute trade sizes
    delta_pos = positions.diff().fillna(positions.iloc[0]).abs()

    # 3) Notional traded each bar
    notional  = price * multiplier * fx * delta_pos

    # 4) Execution costs
    cost_usd  = commission_usd * delta_pos
    cost_pct  = commission_pct     * notional
    slip_cost = slippage_pct       * notional

    # 5) Net PnL after costs
    pnl       = pnl_price - cost_usd - cost_pct - slip_cost

    # 6) Equity & drawdown
    equity    = capital + pnl.cumsum()
    drawdown  = (equity.cummax() - equity) / equity.cummax()

    return pd.DataFrame({
        'position':   positions,
        'price_diff': price_diff,
        'prev_pos':   prev_pos,
        'pnl_price':  pnl_price,
        'delta_pos':  delta_pos,
        'cost_usd':   cost_usd,
        'cost_pct':   cost_pct,
        'slip_cost':  slip_cost,
        'pnl':        pnl,
        'equity':     equity,
        'drawdown':   drawdown,
    })
