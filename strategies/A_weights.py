from __future__ import annotations
from typing import Dict, Sequence


def get_portfolio_weights(config: dict, instruments: Sequence[str]) -> Dict[str, float]:
    """
    MOVED OUT of your original module.
    Paste the original body here if you had custom logic.
    Default: equal weights for given instruments unless config specifies explicit weights.
    """
    # Try explicit mapping from config: e.g., config["weights"]["by_symbol"] = { "ES": 0.4, "NQ": 0.6 }
    explicit = ((config or {}).get("weights") or {}).get("by_symbol") or {}
    if explicit:
        w = {sym: float(explicit.get(sym, 0.0)) for sym in instruments}
        s = sum(w.values())
        if s > 0:
            return {k: v / s for k, v in w.items()}
    # Equal weights fallback
    n = len(instruments)
    if n == 0:
        return {}
    eq = 1.0 / n
    return {sym: eq for sym in instruments}
