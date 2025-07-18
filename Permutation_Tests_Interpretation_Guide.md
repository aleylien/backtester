## Test 1: Fully Specified Trading System

**Metric:** p-value from permuting OOS price changes.

| p-value range | Interpretation                                                 | Next steps                                              |
|---------------|----------------------------------------------------------------|---------------------------------------------------------|
| ≤ 0.01        | **Highly significant**. Virtually impossible to see such OOS return by chance. | Consider scaling up or deploying live; perform robustness checks (e.g. alternate regimes). |
| 0.01–0.05     | **Statistically significant**. Good evidence of genuine edge.  | Increase B or test additional out-of-sample periods; monitor stability. |
| 0.05–0.10     | **Marginally significant**. Some evidence but susceptible to noise. | Collect more data; refine entry/exit rules; consider ensemble methods. |
| > 0.10        | **Not significant**. OOS returns consistent with randomness.   | Reevaluate strategy logic; simplify model; avoid live deployment. |

- **Statistical meaning:** Under the null (no true edge), p = probability of observing an OOS return ≥ actual. A small p implies low likelihood under null, so reject randomness.

---

## Test 2: Training-Process Overfitting Detection

**Metric:** p-value from permuting entire training set before optimization.

| p-value range | Interpretation                                                        | Next steps                                            |
|---------------|-----------------------------------------------------------------------|-------------------------------------------------------|
| ≤ 0.05        | **Optimization finds real signal**; noise-only trials rarely match in-sample performance. | Validate on fresh data; tune hyperparameters further. |
| > 0.05        | **Likely overfitting**; permuted noise often yields equal or better performance. | Reduce model complexity; add regularization; increase train/test splits. |

- **Statistical meaning:** Under null (no predictive patterns), the optimizer applied to noise attains performances comparable to real. Large p-value means optimizer fits noise.

---

## Test 5: Correcting for Selection Bias (Multiple Systems)

**Metrics:** For each system, two p-values:
1. **solo_p**: p-value as if the system were pre-specified.
2. **unbiased_p**: p-value accounting for selecting the best among N candidates.

| Pattern                                    | Interpretation                                                    | Next steps                                                                          |
|--------------------------------------------|-------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| **solo_p ≤ 0.05** and **unbiased_p ≤ 0.05** | Strong evidence of robust edge, even after multiple comparisons.  | Proceed to live testing; allocate capital according to risk guidelines.             |
| **solo_p ≤ 0.05** and **unbiased_p > 0.05**  | Strategy beats random chance alone, but may be top-performer luck. | Expand candidate pool; perform cross-validation; consider ensemble to reduce variance. |
| **solo_p > 0.05**                            | Even in isolation, the system isn’t significant.                  | Discard or rework; focus on new hypotheses.                                         |

- **Statistical meaning:** Unbiased_p answers: “If all systems actually have no effect, what’s the chance the best one scores ≥ observed?” A small unbiased_p lets you reject this high-bar null.

---

## Test 6: Partition Total Return into Trend, Skill & Training Bias

**Outputs:**
- **trend**: Profit from market drift.
- **bias**: The portion of return inflation due to overfitting, estimated via OOS permutations.
- **skill**: Residual genuine performance—your real alpha—after removing trend and bias.

| Component | Relative magnitude                         | Meaning & actions                                                            |
|-----------|--------------------------------------------|------------------------------------------------------------------------------|
| **trend** dominates >70% of total        | System mostly rides drift          | Hedge delta; switch to market-neutral factors; reassess directional bets.      |
| **bias** >25% of total                   | Overfitting drives most profits   | Simplify entry rules; increase out-of-sample validation; reduce lookahead bias.|
| **skill** >25% of total                  | Genuine alpha after adjustments   | High confidence: scale strategy; allocate resources to monitoring and risk controls.|

- **Statistical meaning:** By permuting OOS, true skill is destroyed; any residual above trend is by definition overfitting (bias). Subtracting reveals authentic edge.

---

### General Advice

- **Interpret p-values** in the context of B: fewer permutation trials inflate Monte Carlo error. Use B≥1,000 for moderate precision, B≥10,000 for tight thresholds.
- **Combine tests** for robust conclusions: e.g., only deploy if Test 1 and Test 2 both show p≤0.05, Test 5 unbiased_p≤0.05, and Test 6 skill exceeds a minimum threshold.
- **Sensitivity checks:** Vary oos_start, drift_rate, and model hyperparameters to ensure results aren’t regime-specific.

Use this guide to make informed, statistically grounded decisions on strategy validation and deployment.
