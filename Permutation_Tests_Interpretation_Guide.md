# APPLIED

## 2. Per-Asset Permutation Tests

| Instrument | Test 1 p | Test 2 p | Trend |   Bias  | Skill |
| :--------: | :------: | :------: | :---: | :-----: | :---: |
|    SP500   |   0.807  |   0.001  |  0.0  | –44.15% | –0.00 |
|     DAX    |   0.806  |   0.001  |  0.0  | 100.88% |  0.00 |

* **Instrument**
  The ticker or market (plus strategy) being tested.

* **Test 1 p** (OOS Bundle Permutation)
  You take each out-of-sample bundle’s P\&L for that instrument, randomly shuffle the per-bundle P\&Ls, recombine, and ask: *“How often does a scrambled total ≥ the real total?”*

  * A small p (e.g. 0.05) would indicate the real out-of-sample performance is unlikely to arise by chance.
  * Here both SP500 (0.807) and DAX (0.806) are high → their raw OOS P\&L sums are well within the distribution of random permutations.

* **Test 2 p** (Training-Process Overfit)
  You freeze the best hyper-parameters found on the in-sample portion, then permute the in-sample **price** series, re-run the optimization (on that permuted IS), and check “best-bundle” OOS P\&L. The p-value is the fraction of perms whose best OOS P\&L ≥ the original best.

  * A low p (near 0) means your optimization likely overfit—real parameters performed better than almost all random rearrangements of the in-sample data.
  * Here both series show p ≈ 0.001 → the chosen parameters found real structure that almost never appears in shuffled data.

* **Trend**
  \= (#long bars − #short bars) × drift\_rate.

  * If your system simply rode a drift or trend in the market, this quantifies that component of return.

* **Bias**
  The average “illusory” return from overfitting noise: you shuffle **only** the out-of-sample **price changes**, run your backtest each time, compute (r\_perm – trend), then average over B trials.

  * A large positive bias means a lot of the out-of-sample gains are explainable by fitting random noise.

* **Skill**
  \= original total return – trend – mean\_bias.

  * This is the residual “true alpha” once you strip out drift and overfitting bias.
  * Here both Skill \~ 0 → almost no genuine signal beyond trend or spurious fitting.

---

## 3. Multiple-System Selection Bias

|    System    | Solo p | Unbiased p |
| :----------: | :----: | :--------: |
| SP500\_ewmac |  0.817 |    0.817   |
|  DAX\_ewmac  |  0.808 |    0.001   |

* **System**
  Each asset + strategy combination you tested (you had two: SP500\_ewmac and DAX\_ewmac).

* **Solo p**
  Exactly the same as Test 1 but done separately for each system: *“If I pre-registered this one system, how often would permuted OOS P\&L ≥ the real one?”*

  * SP500\_ewmac → 81.7%
  * DAX\_ewmac   → 80.8%

* **Unbiased p**
  Here you simulate your real practice of “run all systems, then pick the best performer.” In each of B permutations you:

  1. Permute the OOS P\&L bundles.
  2. Recompute each system’s total.
  3. Identify which system “wins” on that permuted data.
  4. See if that winner’s permuted total ≥ its actual observed total.

  Counting how often that happens gives the **unbiased** p-value:

  * For SP500\_ewmac it stays 0.817 (because it was the winner even in perm trials).
  * For DAX\_ewmac it plummets to 0.001—only 0.1% of the time did the best-of-both systems on shuffled data beat its real performance.


### Why both p-values matter

* **Solo p** is the right test if you truly locked in your system *before* seeing any data.
* **Unbiased p** corrects for the fact that you compared multiple candidates and then reported the top one—so it answers:
  *“What are the odds that my best-in-class system would look this good purely by luck?”*

---

# GENERAL 
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
