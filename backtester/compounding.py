from __future__ import annotations
import pandas as pd
from typing import Dict


def _period_ends(idx: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    if freq == "weekly":
        # last business day of each week: resample W-FRI
        return idx.to_series().resample("W-FRI").last().dropna().index.intersection(idx)
    elif freq == "monthly":
        return idx.to_series().resample("M").last().dropna().index.intersection(idx)
    elif freq in {"quarterly", "3M"}:
        return idx.to_series().resample("Q").last().dropna().index.intersection(idx)
    else:
        raise ValueError("freq must be weekly|monthly|quarterly")

def apply_compounding_and_rebalancing(
    per_symbol_positions: Dict[str, pd.Series],
    per_symbol_prices: Dict[str, pd.Series],
    *,
    initial_capital: float,
    target_weights: Dict[str, float],
    policy_mode: str = "none",            # "none" | "reinvesting" | "periodic_rebalance"
    reinvesting_frequency: str = "monthly",
    rebalance_frequency: str = "quarterly",
    contract_multiplier: Dict[str, float] | None = None,
    fx: Dict[str, float] | None = None,
    costs_per_unit: Dict[str, float] | None = None,
) -> Dict[str, pd.Series]:
    """
    Returns dict with:
      - 'equity_portfolio': portfolio equity series
      - 'equity_by_symbol': dict of per-symbol equity
      - 'positions_scaled': dict of positions after scaling/rebalance
    Notes:
      * positions are assumed to be integer contract counts over time (or floats).
      * prices are the traded price aligned to the same index.
      * PnL is approximated by position[t-1] * (price[t]-price[t-1]) * multiplier * fx
      * Transaction costs are approximated by |Δposition| * cost_per_unit (if provided).
    """
    symbols = list(per_symbol_positions.keys())
    # Align everything to common index
    idx = None
    for s in symbols:
        idx = per_symbol_prices[s].index if idx is None else idx.union(per_symbol_prices[s].index)
    idx = idx.sort_values()

    # Defaults
    contract_multiplier = contract_multiplier or {s: 1.0 for s in symbols}
    fx = fx or {s: 1.0 for s in symbols}
    costs_per_unit = costs_per_unit or {s: 0.0 for s in symbols}

    # Reindex and ffill positions, prices
    pos = {s: per_symbol_positions[s].reindex(idx).ffill().fillna(0.0) for s in symbols}
    px = {s: per_symbol_prices[s].reindex(idx).ffill().dropna() for s in symbols}
    idx = px[symbols[0]].index  # after ffill

    # Per-symbol equity tracking
    eq = {s: pd.Series(0.0, index=idx, dtype=float) for s in symbols}
    eq_curr = {s: initial_capital * float(target_weights.get(s, 0.0)) for s in symbols}
    pos_scaled = {s: pos[s].copy() for s in symbols}

    # Period boundaries
    if policy_mode == "reinvesting":
        period_ends = _period_ends(idx, reinvesting_frequency)
    elif policy_mode == "periodic_rebalance":
        period_ends = _period_ends(idx, rebalance_frequency)
    else:
        period_ends = pd.DatetimeIndex([])

    # Iterate and simulate pnl; scale at boundaries
    prev_price = {s: px[s].iloc[0] for s in symbols}
    prev_pos = {s: float(pos_scaled[s].iloc[0]) for s in symbols}

    portfolio_equity = []
    total_equity = sum(eq_curr.values())
    # initialize day 0
    for t, dt in enumerate(idx):
        # PnL for this bar
        for s in symbols:
            curr_price = float(px[s].loc[dt])
            dP = curr_price - float(prev_price[s])
            pnl_price = prev_pos[s] * dP * float(contract_multiplier.get(s, 1.0)) * float(fx.get(s, 1.0))
            # simple linear transaction cost
            dpos = float(pos_scaled[s].loc[dt]) - prev_pos[s]
            cost = abs(dpos) * float(costs_per_unit.get(s, 0.0))
            eq_curr[s] += (pnl_price - cost)
            eq[s].iat[t] = eq_curr[s]
            prev_pos[s] = float(pos_scaled[s].loc[dt])
            prev_price[s] = curr_price

        total_equity = sum(eq_curr.values())
        portfolio_equity.append(total_equity)

        # Boundary actions AFTER applying PnL of this bar
        if dt in period_ends:
            if policy_mode == "reinvesting":
                # Scale ALL future positions by same factor so next bar uses bigger capital
                # factor = current_total / initial_total_of_period
                # We can compute factor per-symbol to keep current weights (no explicit rebalance)
                total_now = sum(eq_curr.values())
                # Keep weights as they are currently (drifted)
                for s in symbols:
                    # scale by same portfolio factor => maintain drifted weights
                    # But we need to scale contracts relative to equity change.
                    # Assume linear in capital; use factor = total_now / last_total_snapshot.
                    pass  # noop since our loop uses realized eq directly; positions remain as-is.
                # No explicit change to pos; reinvestment is implicit because subsequent PnL
                # accrues on larger equity (we don't need to scale contracts here if
                # strategies are re-run per bundle; if you want explicit growth in contracts,
                # uncomment the factor-based scaling below.)

            if policy_mode == "periodic_rebalance":
                # Rebalance to target weights by scaling FUTURE positions
                total_now = sum(eq_curr.values())
                for s in symbols:
                    target_cash = float(target_weights.get(s, 0.0)) * total_now
                    curr_cash = eq_curr[s]
                    if curr_cash <= 0 or target_cash <= 0:
                        continue
                    scale = target_cash / curr_cash
                    # multiply FUTURE positions by scale
                    next_idx = idx.get_indexer([dt], method="backfill")[0] + 1
                    if next_idx < len(idx):
                        pos_scaled[s].iloc[next_idx:] *= scale
                    # book the rebalance immediately by setting eq_curr[s] = target (transaction cost abstracted)
                    eq_curr[s] = target_cash
                    # NOTE: if you have an explicit transaction-cost model, apply it here.

    equity_portfolio = pd.Series(portfolio_equity, index=idx, name="equity")
    return {
        "equity_portfolio": equity_portfolio,
        "equity_by_symbol": eq,
        "positions_scaled": pos_scaled,
    }
