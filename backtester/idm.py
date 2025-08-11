from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List
from backtester.data_loader import DataLoader


def _corr_idm(returns: pd.DataFrame, weights: pd.Series, floor_negatives: bool, min_overlap: int, fallback: float) -> float:
    if returns.shape[1] <= 1:
        return 1.0
    # Require overlap
    count = returns.notna().astype(int).T @ returns.notna().astype(int)
    ok = (count.values >= min_overlap)
    np.fill_diagonal(ok, True)
    if not ok.all():
        return float(fallback)

    rho = returns.corr(min_periods=min_overlap)
    if floor_negatives:
        rho = rho.clip(lower=0.0)
    np.fill_diagonal(rho.values, 1.0)

    w = weights.copy().astype(float)
    s = w.sum()
    w = (w / s) if s != 0 else pd.Series(1.0 / len(w), index=w.index)
    quad = float(np.dot(w.values, np.dot(rho.values, w.values)))
    if not np.isfinite(quad) or quad <= 0:
        return float(fallback)
    return float(1.0 / np.sqrt(quad))


def compute_idm_map(cfg: dict, instruments: List[dict]) -> Dict[str, float]:
    """
    Returns {strategy_module: idm_scalar}. Uses close-price returns for symbols that use the same strategy.
    Config keys (optional):
    cfg['idm'] = {
        'enabled': true,
        'mode': 'static',      # only static here; rolling can be added later
        'floor_negatives': true,
        'min_overlap': 60,
        'window': 250,         # lookback in bars (on cfg['data'].timeframe)
        'returns_freq': 'daily', # daily | weekly (weekly does .resample('W-FRI').last().pct_change())
        'fallback': 1.0
    }
    If cfg['idm']['enabled'] is false or missing, returns {} and you should pass the scalar from YAML.
    """
    idm_cfg = (cfg.get('idm') or {})
    if not idm_cfg.get('enabled', False):
        return {}

    floor_neg = bool(idm_cfg.get('floor_negatives', True))
    min_overlap = int(idm_cfg.get('min_overlap', 60))
    window = int(idm_cfg.get('window', 250))
    freq = str(idm_cfg.get('returns_freq', 'daily')).lower()
    fallback = float(idm_cfg.get('fallback', 1.0))

    # Group symbols by strategy
    by_strat: Dict[str, List[str]] = {}
    for inst in instruments:
        by_strat.setdefault(inst['strategy']['module'], []).append(inst['symbol'])

    out: Dict[str, float] = {}
    for strat, symbols in by_strat.items():
        # Load closes for all symbols with this strategy
        rets_cols = {}
        for sym in symbols:
            inst_cfg = dict(cfg)  # shallow safe
            # Load per-symbol data with repo's DataLoader
            dl = DataLoader(
                data_dir=cfg['data']['path'],
                symbol=sym,
                timeframe=cfg['data']['timeframe'],
                base_timeframe=cfg['data'].get('base_timeframe')
            )
            df = dl.load().copy()
            if df.empty:
                continue
            ser = df['close'].copy()
            ser.index = pd.to_datetime(ser.index)
            if freq.startswith('week'):
                ser = ser.resample('W-FRI').last()
            rets_cols[sym] = ser.pct_change()
        if not rets_cols:
            out[strat] = fallback
            continue
        R = pd.DataFrame(rets_cols).sort_index().tail(window)
        w = pd.Series(1.0, index=list(rets_cols.keys()))
        val = _corr_idm(R, w, floor_neg, min_overlap, fallback)
        # Cap at 2.5 as requested
        out[strat] = float(min(val, 2.5))
    return out
