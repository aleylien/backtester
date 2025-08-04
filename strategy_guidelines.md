# Strategy Integration Guidelines

Below are the **guidelines** and **absolute requirements** any strategy function must follow to plug seamlessly into this backtester framework.

## 1. Function Signature

```python
def my_strategy(
    df: pd.DataFrame,
    # any number of hyper-parameters with sensible defaults
    param1: float = …,
    param2: int   = …,
    …,
    multiplier: float = 1.0,
    fx:         float = 1.0,
) -> pd.DataFrame:
    …
```

- The first argument **must** be a DataFrame `df` containing at least a `close` column.  
- Additional keyword arguments correspond to your strategy’s tunable parameters (matching what you put in `strategy_config.yaml` under `param_grid`).

## 2. Input DataFrame Requirements

1. **Column**:  
   - `df['close']` must exist. Strategies derive prices, indicators, and forecasts from it.  
2. **Index**:  
   - Ideally a DateTimeIndex. The framework will reset it to a `date` column when saving OOS diagnostics, but your logic should assume `df.index` is ordered timestamps.  
3. **No in-place side effects**:  
   - Don’t modify `df` directly; work on copies or new Series (e.g. `price = df['close']`).

## 3. Output DataFrame Requirements

Your function returns a new DataFrame of **exactly** the same length and index as the input, with at minimum:

| Column            | Type              | Meaning                                                |
|-------------------|-------------------|--------------------------------------------------------|
| `position`        | integer (`int`)   | Number of contracts/lots to hold on each bar.          |

If you support forecasts (for “find-signal” mode or permutation tests), also include:

| Column               | Type            | Meaning                                                        |
|----------------------|-----------------|----------------------------------------------------------------|
| `raw_forecast`       | float           | Unscaled signal (e.g. normalized EMA difference).             |
| `scaled_forecast`    | float           | Forecast after applying your `forecast_scale` factor.         |
| `capped_forecast`    | float           | Final forecast clipped to your `cap` or equivalent bounds.    |

Avoid overriding reserved names:

```
['equity','drawdown','bundle','sample','pnl','cost_usd','cost_pct','slip_cost']
```

## 4. NaN & Initialization Handling

- **Rolling indicators** introduce NaNs at the start.  
- **Strategy must** leave those rows as NaN, then fill positions to 0:
  ```python
  N_unrounded = N_unrounded.fillna(0.0)
  position    = np.round(N_unrounded).astype(int)
  ```  
- Do not assign nonzero positions without sufficient data.

## 5. No P&L Simulation

- The framework handles PnL. **Do not** compute equity or pnl inside your strategy function.

## 6. Respect `multiplier` & `fx`

- Accept `multiplier` and `fx` parameters to allow the framework to pass contract sizing and currency conversion.

## 7. Forecast‑Scale Automation

- Handle `forecast_scale=None` or `0` by:
  ```python
  avg_abs = raw.abs().mean(skipna=True)
  scale   = target / (avg_abs if avg_abs!=0 else 1.0)
  ```
- Enables auto-scaling during optimization.

## 8. Timeframe‑Agnostic

- Avoid hard‑coded annualization. The framework uses `get_periods_per_year` for stats.
- If you need volatility inside the strategy, compute it generically based on `vol_window`.

## 9. Vectorization

- Use Pandas/NumPy vector operations, not Python loops, for performance.

## 10. Parameter Names & Config Matching

- Parameter names in your function must **exactly** match keys in `strategy_config.yaml` under `param_grid`.

### Minimum Skeleton

```python
def my_strategy(df, window=20, threshold=1.0, multiplier=1.0, fx=1.0):
    price = df['close']
    sma = price.rolling(window).mean()
    signal = (price - sma) / sma
    pos = (signal > threshold).astype(int)
    return pd.DataFrame({
      'raw_forecast': signal,
      'scaled_forecast': signal,
      'capped_forecast': signal.clip(-1,1),
      'position': pos
    })
```
