# Guide to Interpreting Advanced Backtest Statistical Tests

Below is a concise guide to understanding key statistical tests applied to your portfolio backtest, using your results as an example.

## 1. Sample Size Check
- **`num_nonzero_rets = 3002`**  
  You need enough non-zero return observations to support reliable inference.  
  *With ~3,000 data points, you have a solid sample size for bootstrapping and permutation tests.*

## 2. Bootstrap Confidence Intervals
Bootstrap resamples your actual return series to estimate empirical distributions of each metric.

| Metric      | 0.1 % Quantile | 95 % Quantile | Average   |
|-------------|--------------:|-------------:|----------:|
| **Mean**    | –0.000206     | 0.000146     | —         |
| **Log PF**  | –0.216        | 0.012        | —         |
| **Drawdown**| 0.071         | 0.320        | 0.2088    |

- **Reading the table**:  
  - For mean, 99.9% of bootstrap samples are **≥ –0.000206**, and 95% are **≤ 0.000146**.  
  - For drawdown, 99.9% are **≥ 7.1%**, and 95% are **≤ 32.0%**.  
- **Your actual metrics**:  
  - *Mean = 0.000005* sits well within this range → no evidence of significant outperformance.  
  - *Drawdown = 13.3%* sits between 7.1% and 32.0% → typical under the bootstrap null.

## 3. Bootstrap p‑Values (Two‑Sided)
- **`p_two_sided_mean = 0.947`**  
- **`p_two_sided_log_pf = 0.947`**  

**Definition:** Fraction of bootstrap draws whose |metric| ≥ |actual|.  
**Interpretation:**  
- 94.7% of bootstrap means (or log PFs) are as extreme as your actual values → *not statistically significant*.

## 4. Permutation Test for Drawdown (One‑Sided)
Permutation scrambles the order of returns to test order‑sensitive metrics like drawdown.

| Quantile    | Value   |
|-------------|--------:|
| 0.1 %       | 0.1002  |
| 95 %        | 0.2702  |
| **Average** | 0.1963  |

- **Actual drawdown = 13.33%**  
- **`p_one_sided_drawdown = 0.051`**  
  Only 5.1% of permuted drawdowns are as *low* as your actual drawdown → *borderline significance*.

## 5. Summary of Findings

| Metric            | Actual    | Significance (α=0.05)    |
|-------------------|----------:|-------------------------:|
| **Mean Return**   | 0.000005  | Not significant (p=0.947) |
| **Profit Factor** | 1.0061    | Not significant (p=0.947) |
| **Max Drawdown**  | 13.33%    | Borderline (p=0.051)     |

- **Return & PF**: align with randomness.
- **Drawdown**: marginally better than random shuffles.

## 6. Best Practices
1. **Check sample size** before trusting p‑values.  
2. **Use two-sided tests** for order‑invariant metrics (mean, PF).  
3. **Use permutation** for order‑dependent metrics (drawdowns).  
4. **Report quantiles & p‑values** so readers see the null distribution.  
5. **Be cautious around p≈0.05**; consider additional data or alternative methods.

