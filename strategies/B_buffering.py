from __future__ import annotations
import numpy as np
import pandas as pd


def _as_series(x, index: pd.Index) -> pd.Series:
    if isinstance(x, pd.Series):
        return x.reindex(index).ffill().bfill()
    return pd.Series(float(x), index=index)


def compute_buffer_contracts(
    price: pd.Series,
    ann_std: pd.Series,
    *,
    capital: float,
    idm: float,
    tau: float,
    multiplier: float,
    fx: float | pd.Series = 1.0,
    F: float = 0.10,
) -> pd.Series:
    """
    B_t = F * Capital * IDM * tau / (Multiplier * Price_t * FX_t * sigma_t)
    (Weight is already in 'capital' in your code.)

    Notes:
    - 'ann_std' is your sigma_t (annualised vol proxy).
    - If fx is scalar, it will be broadcast across the index.
    - Returns a Series of buffer width in *contracts*.
    """
    price = pd.to_numeric(price, errors="coerce")
    ann_std = pd.to_numeric(ann_std, errors="coerce")
    fx_ser = _as_series(fx, price.index)

    denom = (multiplier * price * fx_ser * ann_std).replace({0.0: np.nan})
    base = (capital * float(idm) * float(tau)) / denom
    B = float(F) * base
    return B.fillna(0.0)


def apply_buffer_to_contracts(
    N_unrounded: pd.Series,
    buffer_contracts: pd.Series,
    initial_pos: int = 0,
) -> pd.Series:
    """
    Given desired unrounded contracts N_t and buffer size B_t (contracts),
    produce an executed/rounded position series with Carver-style buffering:

      lower_t = round(N_t - B_t)
      upper_t = round(N_t + B_t)

      if pos_{t-1} < min(lower_t, upper_t): trade to lower bound
      elif pos_{t-1} > max(lower_t, upper_t): trade to upper bound
      else: hold

    Handles negative N_t correctly.
    """
    N = pd.to_numeric(N_unrounded, errors="coerce").fillna(0.0)
    B = pd.to_numeric(buffer_contracts, errors="coerce").fillna(0.0).abs()

    lower = np.round(N - B).astype(int)
    upper = np.round(N + B).astype(int)
    lo = pd.concat([lower, upper], axis=1).min(axis=1)
    hi = pd.concat([lower, upper], axis=1).max(axis=1)

    out = []
    c = int(initial_pos)
    for t in N.index:
        l = int(lo.loc[t]); h = int(hi.loc[t])
        if c < l:
            c = l
        elif c > h:
            c = h
        out.append(c)
    return pd.Series(out, index=N.index, name="position")
