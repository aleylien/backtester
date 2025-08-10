# backtester/strategies/idm.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Iterable


def _effective_corr_mat(df: pd.DataFrame, floor_negatives: bool) -> pd.DataFrame:
    rho = df.corr(min_periods=1)  # pairwise, handles NaNs
    if floor_negatives:
        rho = rho.clip(lower=0.0)
    # Fill diag with 1.0 (robustness)
    np.fill_diagonal(rho.values, 1.0)
    return rho


def _overlap_mask(df: pd.DataFrame, min_overlap: int) -> bool:
    # True if every pair has >= min_overlap overlapping obs
    n = df.shape[1]
    count = df.notna().astype(int).T @ df.notna().astype(int)
    # count_ij is #overlapping non-NaN rows for pair (i,j)
    # Require off-diagonals >= min_overlap
    ok = (count.values >= min_overlap)
    # Ignore diagonal
    np.fill_diagonal(ok, True)
    return bool(ok.all())


def compute_idm_static(
    returns_by_instrument: Dict[str, pd.Series],
    weights: Dict[str, float] | pd.Series | Iterable[float],
    *,
    floor_negatives: bool = True,
    min_overlap: int = 60,
    fallback: float = 1.0,
) -> float:
    """
    IDM = 1 / sqrt(w^T ρ w), using per-instrument OOS daily returns.
    returns_by_instrument: dict of {symbol: daily arithmetic returns (DatetimeIndex)}
    weights: dict/series/list aligned by instrument keys order.
    """
    if not returns_by_instrument:
        return float(fallback)

    df = pd.DataFrame(returns_by_instrument).astype(float)
    if df.shape[1] == 1:
        return 1.0

    if not _overlap_mask(df, min_overlap=min_overlap):
        return float(fallback)

    w = pd.Series(weights, index=df.columns, dtype=float)
    if w.isna().any():
        w = w.fillna(0.0)
    # Normalize weights to sum 1 (only relative matters here)
    if w.sum() != 0:
        w = w / w.sum()
    else:
        w = pd.Series(1.0 / len(w), index=w.index)

    rho = _effective_corr_mat(df, floor_negatives=floor_negatives)
    quad = float(np.dot(w.values, np.dot(rho.values, w.values)))
    if quad <= 0 or not np.isfinite(quad):
        return float(fallback)
    return float(1.0 / np.sqrt(quad))


def compute_idm_rolling(
    returns_by_instrument: Dict[str, pd.Series],
    weights: Dict[str, float] | pd.Series | Iterable[float],
    *,
    window: int = 250,
    floor_negatives: bool = True,
    min_overlap: int = 60,
    fallback: float = 1.0,
) -> pd.Series:
    """
    Rolling IDM as a time series (indexed like the union of input returns).
    """
    if not returns_by_instrument:
        return pd.Series(dtype=float)

    df = pd.DataFrame(returns_by_instrument).astype(float).sort_index()
    if df.shape[1] == 1:
        return pd.Series(1.0, index=df.index)

    w = pd.Series(weights, index=df.columns, dtype=float)
    if w.sum() != 0:
        w = w / w.sum()
    else:
        w = pd.Series(1.0 / len(w), index=w.index)

    out = []
    idx = []
    roll = df.rolling(window=window, min_periods=min_overlap)
    for t, sub in roll:
        if sub.shape[0] < min_overlap or sub.dropna(how="all").shape[1] < 2:
            idm_t = fallback
        else:
            if not _overlap_mask(sub, min_overlap=min_overlap):
                idm_t = fallback
            else:
                rho = _effective_corr_mat(sub, floor_negatives=floor_negatives)
                quad = float(np.dot(w.values, np.dot(rho.values, w.values)))
                idm_t = fallback if quad <= 0 or not np.isfinite(quad) else 1.0 / np.sqrt(quad)
        out.append(idm_t)
        idx.append(t)
    return pd.Series(out, index=idx).reindex(df.index).ffill().fillna(fallback)
