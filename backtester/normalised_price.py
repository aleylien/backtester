import numpy as np
import pandas as pd


def compute_normalised_price(
    price: pd.Series,
    *,
    method: str = "rolling",   # "rolling" | "ewm"
    window: int = 25,          # used by rolling
    span: int | None = None,   # used by ewm (defaults to window if None)
    min_periods: int | None = None,
    scale: float = 100.0,
    start_value: float = 0.0,
) -> pd.Series:
    """
    P^N_t = P^N_{t-1} + scale * (p_t - p_{t-1}) / sigma_{p,t},
    where sigma_{p,t} is the std dev of price *differences*.

    - If method == "rolling": sigma = diff.rolling(window, min_periods).std()
    - If method == "ewm":     sigma = diff.ewm(span=span or window, adjust=False).std(bias=False)

    Returns a Series named 'price' that you can feed to strategies.
    """
    p = pd.to_numeric(price, errors="coerce")
    d = p.diff()

    if method.lower() == "ewm":
        sp = int(span or window)
        sigma = d.ewm(span=sp, adjust=False).std(bias=False)
    else:
        w = int(window)
        mp = int(min_periods if min_periods is not None else max(5, w // 3))
        sigma = d.rolling(w, min_periods=mp).std()

    step = scale * (d / sigma)
    step = step.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    pn = step.cumsum()
    if start_value:
        pn = pn + float(start_value)
    pn.name = "price"
    return pn
