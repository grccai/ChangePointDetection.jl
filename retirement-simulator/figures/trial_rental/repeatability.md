# Repeatability + statistical significance

Three checks on the v4 → v5 finding ("realistic operating costs flip
the optimum from buy-early to buy-late"):

1. **MC noise floor** — how big is the standard error of the reported
   reward at the 4000-path final eval?
2. **Cross-policy paired comparison under the realistic scenario** —
   do v4 and v5 policies differ when evaluated on the same MC paths?
3. **Optimizer-seed stability** — does the v5 optimizer find the same
   optimum across different DE seeds?

## Q1 — MC noise floor at 4000 paths

For each of the 8 published optima (4 v4 + 4 v5), evaluate against
`trial_rental_realistic.yaml` with 30 independent MC seeds. Standard
error of the weighted FIRE-prob reward is consistently:

  **SE(reward) ≈ 0.005**  (95% CI half-width ≈ 0.01)

This is much smaller than the 0.05–0.10 differences that drove the
v4 → v5 narrative. Any difference > 0.02 (~4× SE) is detectable with
30 seeds at p<0.05.

## Q2 — Paired comparison: v4 policies vs v5 policies on the realistic scenario

Each row evaluates **both** the v4 optimum (originally tuned on
`trial_rental.yaml`) and the v5 optimum (tuned on
`trial_rental_realistic.yaml`) against the realistic scenario, on
the same 30 MC seeds.

| Pair | reward_v4 (mean ± SE) | reward_v5 (mean ± SE) | Δ (v5 − v4) | paired t / p |
|---|---|---|---|---|
| bond_tent_gbm        | 4.4523 ± 0.0052 | 4.3824 ± 0.0050 | **−0.0699** | t = −21.1, p < 10⁻⁴ |
| bodie_merton_gbm     | 4.4445 ± 0.0054 | 4.4712 ± 0.0055 | +0.0267 | t = +14.1, p < 10⁻⁴ |
| bond_tent_robust     | 4.3944 ± 0.0055 | 4.3120 ± 0.0057 | **−0.0824** | t = −31.0, p < 10⁻⁴ |
| bodie_merton_robust  | 4.4420 ± 0.0056 | 4.3397 ± 0.0056 | **−0.1023** | t = −40.3, p < 10⁻⁴ |

Median ruin rate across the 30 seeds:

| Pair | ruin_v4 | ruin_v5 |
|---|---|---|
| bond_tent_gbm        | 1.43% | 0.54% |
| bodie_merton_gbm     | 1.81% | 0.27% |
| bond_tent_robust     | 0.94% | 0.36% |
| bodie_merton_robust  | 0.71% | 0.30% |

### Reading

The v4 policies (early rental purchase) **beat the v5 policies (late
purchase) on reward** in 3 of 4 cases under the realistic scenario,
with massive significance (t = −21 to −40, all p < 10⁻⁴). The
exception is bodie_merton_gbm where v5's smaller-scale early purchase
($556k vs $1.48M) genuinely improves on v4.

Both policies satisfy the 2% ruin constraint. v5 has lower ruin
across the board — but the optimizer's objective is **reward subject
to ruin ≤ 2%**, so paying 0.07–0.10 reward to drop ruin from 1% to
0.3% is not a trade the optimizer should be making.

**Conclusion: the v5 optimizer landed in a local "buy late" basin
for three of the four policy classes.** The v4 "buy early" decisions
were robust to the cost model upgrade — the cost model didn't
invalidate them, the optimizer just couldn't escape the local
optimum during the v5 run.

The original v5 narrative ("realistic costs flip the optimum to buy
late or never") is wrong. The correct narrative is: **realistic costs
do NOT flip the optimum, but the joint search space has multiple
basins of attraction, and the budget/seed combo we used for v5 fell
into the late-buy basin in 3 of 4 runs.**

## Q3 — Optimizer-seed stability (in progress)

Re-running the v5 bodie_merton-robust optimization with 4 fresh DE
seeds. Goal: see whether the late-buy basin is seed-stable, or whether
some seeds find the early-buy basin (which we now know is genuinely
better at the realistic-cost evaluation).

Results will be appended once the runs complete.

## Implications

1. The published v4 → v5 reward deltas are **real** (paired t-tests
   confirm > 4σ effect sizes), but **mis-attributed**. The drop in
   reward is from optimizer search failure, not from the cost model
   penalising leverage.

2. The single-seed DE search at popsize=8-10 × maxiter=15-18 is
   unreliable on this 11-14 dim joint search problem. Recommendations:
   - Run multiple seeds and take the best (multi-start DE).
   - Increase popsize and maxiter (with budget proportional to n²).
   - Initialize one population member from the v4 optimum to seed
     the early-buy basin.
   - Or switch to BIPOP-CMA-ES for the joint problem.

3. The "early buy" v4 decisions (age ~41, $560k-$1.5M property)
   genuinely outperform "late buy" decisions on the realistic
   cost model — the rental's leverage premium survives 8% management
   fees, capex shocks, and turnover. The v5 narrative needs revision.

## Reproducing

```bash
cd retirement-simulator/
N_SEEDS=30 PYTHONPATH=. python figures/check_repeatability.py
```

Single run (8 strategies × 30 seeds × ~1.5s each at 4000 paths) takes
~7-8 min on 4 cores.
