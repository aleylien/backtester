import os
import json
import pandas as pd


def load_symbol_currency_map(path: str) -> dict:
    """
    Load a map of symbol -> pricing currency (e.g., {'DAX':'EUR','SP500':'USD'}).
    If file missing/empty -> return empty dict. Missing symbols default to base currency in callers.
    """
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def _csv_path(fx_dir: str, pair: str, timeframe: str) -> str:
    # Expect files like EURUSD_1d.csv
    return os.path.join(fx_dir, f"{pair}_{timeframe}.csv")


def _load_pair_series(fx_dir: str, pair: str, timeframe: str) -> pd.Series:
    """
    Load FX pair CSV (Date, Open, High, Low, Close) and return Close series indexed by datetime.
    """
    path = _csv_path(fx_dir, pair, timeframe)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    date_col = "Date" if "Date" in df.columns else "date"
    close_col = "Close" if "Close" in df.columns else "close"
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    return df[close_col].astype(float)


def get_fx_series(
    symbol: str,
    base_ccy: str,
    idx: pd.Index,
    symbol_currency_map: dict,
    fx_dir: str,
    timeframe: str,
) -> pd.Series:
    """
    Return a USD conversion factor series aligned to idx (1.0 if already in base currency).
    - If symbol currency == base_ccy -> ones
    - Else load <pricing_ccy><base_ccy>_<timeframe>.csv (e.g., EURUSD_1d.csv) and FFill
    - If missing, return ones (warn upstream if desired)
    """
    # Default if symbol not in map: assume already base currency
    pricing_ccy = symbol_currency_map.get(symbol.upper(), base_ccy)
    if pricing_ccy == base_ccy:
        return pd.Series(1.0, index=idx)

    pair = f"{pricing_ccy}{base_ccy}"  # e.g., EUR + USD -> EURUSD
    try:
        ser = _load_pair_series(fx_dir, pair, timeframe)
    except FileNotFoundError:
        return pd.Series(1.0, index=idx)

    ser = ser.reindex(idx).ffill().bfill()  # <— add .bfill() for the leading gap
    return ser
