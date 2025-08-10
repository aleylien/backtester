from __future__ import annotations
import pandas as pd
from typing import Dict, Tuple, List


def _period_bounds(index: pd.DatetimeIndex, freq: str) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    if freq == "weekly":
        ends = index.to_series().resample("W-FRI").last().dropna()
    elif freq == "monthly":
        ends = index.to_series().resample("ME").last().dropna()   # was "M"
    elif freq in {"quarterly", "3m"}:
        ends = index.to_series().resample("QE").last().dropna()   # was "Q"
    else:
        raise ValueError("freq must be weekly|monthly|quarterly")
    bounds = []
    start = index[0]
    for dt in ends.index:
        if dt < start:
            continue
        bounds.append((start, dt))
        nxtpos = index.get_indexer([dt], method="backfill")[0] + 1
        if nxtpos < len(index):
            start = index[nxtpos]
        else:
            start = None
            break
    if start is not None and start <= index[-1]:
        bounds.append((start, index[-1]))
    # dedupe
    out = []
    for a, b in bounds:
        if out and out[-1][1] == b:
            continue
        out.append((a, b))
    return out


def scale_pnl_with_policy(
    pnl_by_symbol: Dict[str, pd.Series],
    weights: Dict[str, float],
    initial_total_capital: float,
    mode: str = "none",                   # "none" | "reinvesting" | "periodic_rebalance"
    reinvesting_frequency: str = "monthly",
    rebalance_frequency: str = "quarterly",
) -> Tuple[pd.Series, Dict[str, pd.Series]]:
    """
    Returns (portfolio_equity, scaled_pnl_by_symbol)
    - 'none': no scaling; just sum pnl across symbols → equity = cumsum + initial_total_capital
    - 'reinvesting': at each period end, multiply ALL future pnl by same factor so portfolio equity compounds
    - 'periodic_rebalance': at each period end, set each sleeve's equity to target weight of total, by scaling FUTURE pnl per symbol
    """
    # Align indices
    idx = None
    for s, ser in pnl_by_symbol.items():
        ser.index = pd.to_datetime(ser.index)
        idx = ser.index if idx is None else idx.union(ser.index)
    idx = idx.sort_values()
    pnl = {s: pnl_by_symbol[s].reindex(idx).fillna(0.0).astype(float) for s in pnl_by_symbol}
    symbols = list(pnl.keys())
    # initial per-symbol capital from target weights
    w = pd.Series(weights, index=symbols, dtype=float)
    if w.sum() == 0:
        w[:] = 1.0 / len(w)
    else:
        w /= w.sum()
    cap0 = {s: float(initial_total_capital * w[s]) for s in symbols}

    # none: simple sum
    if mode == "none":
        port_pnl = sum(pnl.values())
        eq = port_pnl.cumsum() + initial_total_capital
        return eq, pnl

    # piecewise scaling
    freq = reinvesting_frequency if mode == "reinvesting" else rebalance_frequency
    bounds = _period_bounds(idx, freq)
    scaled = {s: pnl[s].copy() for s in symbols}
    # rolling current equity snapshot at start of period
    # these track equity WITH scaling applied so far
    eq_so_far = {s: cap0[s] for s in symbols}
    eq_port = []
    out_index = []

    def total_equity_now(t_idx: int) -> float:
        # sum of (cap at start + cumsum of scaled pnl up to t_idx)
        return sum(eq_so_far[s] + scaled[s].iloc[:t_idx+1].cumsum().iloc[-1] for s in symbols)

    # iterate periods
    last_end_pos = -1
    for (a, b) in bounds:
        # indices for this slice
        i0 = idx.get_indexer([a], method="nearest")[0]
        i1 = idx.get_indexer([b], method="nearest")[0]
        # append equity path within this slice (before any scaling at its end)
        slice_equity = []
        for t in range(i0, i1+1):
            # build portfolio equity at each bar using scaled pnl so far
            tot = sum(eq_so_far[s] + scaled[s].iloc[i0:t+1].cumsum().iloc[-1] for s in symbols)
            slice_equity.append(tot)
        eq_port.extend(slice_equity)
        out_index.extend(list(idx[i0:i1+1]))

        # period end action
        if mode == "reinvesting":
            # scale ALL future pnl by same factor so that (implicitly) capital grows for next period
            eq_now = eq_port[-1]
            eq_start = eq_port[last_end_pos] if last_end_pos >= 0 else initial_total_capital
            factor = (eq_now / eq_start) if eq_start != 0 else 1.0
            if i1 + 1 < len(idx):
                for s in symbols:
                    scaled[s].iloc[i1+1:] *= factor
            # advance base equity snapshot to "now" (kept implicitly by scaled pnl)
            for s in symbols:
                eq_so_far[s] = eq_so_far[s] + scaled[s].iloc[i0:i1+1].sum()

        elif mode == "periodic_rebalance":
            # compute current per-symbol equity at period end
            eq_curr = {s: eq_so_far[s] + scaled[s].iloc[i0:i1+1].sum() for s in symbols}
            total_now = sum(eq_curr.values())
            if i1 + 1 < len(idx):
                for s in symbols:
                    target_cash = float(w[s] * total_now)
                    curr_cash = eq_curr[s]
                    if curr_cash > 0:
                        scale = target_cash / curr_cash
                        scaled[s].iloc[i1+1:] *= scale
            # set new base for next period
            eq_so_far = eq_curr

        last_end_pos = i1

    equity_portfolio = pd.Series(eq_port, index=pd.DatetimeIndex(out_index), name="equity")
    return equity_portfolio, scaled
