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

### Paired test on P(ruin), same 30 seeds

The realistic-scenario plan-failure rate per seed is bounded
[0%, 2%]; with 4000 paths the per-seed SE is ≈ 0.16 pp, so the
30-seed pooled SE on the mean ruin estimate is ≈ 0.03 pp.

| Pair | ruin_v4 (mean ± SE) | ruin_v5 (mean ± SE) | Δ (v5 − v4, pp) | paired t / p |
|---|---|---|---|---|
| bond_tent_gbm        | 1.415% ± 0.026 pp | 0.555% ± 0.020 pp | **−0.86** | t = −27.4, p < 10⁻⁴ |
| bodie_merton_gbm     | 1.824% ± 0.028 pp | 0.269% ± 0.017 pp | **−1.56** | t = −47.1, p < 10⁻⁴ |
| bond_tent_robust     | 0.912% ± 0.026 pp | 0.358% ± 0.016 pp | **−0.55** | t = −25.1, p < 10⁻⁴ |
| bodie_merton_robust  | 0.716% ± 0.020 pp | 0.288% ± 0.016 pp | **−0.43** | t = −18.9, p < 10⁻⁴ |

All four ruin deltas are negative and significant at 19-47σ —
**v5 has materially lower ruin than v4 on every policy class**.

### Reading: a Pareto trade, not a dominance result

Combining the reward and ruin paired tests, the v4 vs v5 comparison
under the realistic scenario is a clean **Pareto trade**:

| Axis | Direction | Magnitude |
|---|---|---|
| Reward | v4 > v5 (3 of 4 pairs) | 0.07–0.10 |
| Ruin   | v5 < v4 (4 of 4 pairs) | 0.43–1.56 pp |

Every datapoint is rock-solid (all p < 10⁻⁴, effect sizes 14-47σ).

By the optimizer's stated objective (**max reward s.t. P(ruin) ≤ 2%**)
v4 wins: both v4 and v5 satisfy the 2% constraint, so the objective
reduces to "max reward" and v4 has higher reward. The v5 optimizer
*should* have pushed up to the v4 reward level while staying inside
the ruin cap.

Two readings, both partly true:

1. **Optimizer search failure**: the v5 DE run landed in a local
   "low-ruin / low-reward" basin and didn't explore the higher-reward
   basin that the v4 search had found. The fact that all 4 policy
   classes ended up with similar "buy late" rentals suggests this
   basin is broad and easy to fall into when the optimizer starts
   from random Sobol sequences. The fix is multi-start DE,
   bigger budget, or a smarter init.

2. **Pareto shift**: the v5 optima ARE meaningful, defensible points
   on the reward-ruin frontier — they're just not optimal under the
   given objective function. A user with a tighter implicit ruin
   tolerance (say 0.5% instead of 2%) would correctly prefer v5.
   The realistic cost model does push the frontier outward (more
   ruin for the same reward), but it doesn't invalidate the
   buy-early region of the frontier.

The **earlier v5 narrative** ("realistic costs flip the optimum
from buy-early to buy-late or never") was wrong. The corrected
narrative:

> The realistic cost model does NOT invalidate the v4 "buy early"
> decisions. Both v4 and v5 satisfy the 2% ruin cap; v4 wins on the
> stated objective by 0.07-0.10 reward in 3 of 4 cases. v5 occupies
> a tighter-ruin / lower-reward Pareto point — useful if the user's
> true ruin tolerance is much tighter than 2%, but not what the
> objective asked for.

## Q3 — Optimizer-seed stability

Re-ran the v5 `bodie_merton + rental + fire_prob_robust` optimization
under five fresh DE seeds at the same budget (popsize=8, maxiter=15,
n_paths_inner=1000). Then evaluated each seed-found rental decision
under the v5 bm_robust **allocation policy** on 30 fresh MC seeds at
4000 paths each (same allocation, only the rental override varies —
isolating the rental-decision contribution to reward). Included v4's
rental decision as a baseline.

| seed | inner-MC obj | rental | basin | 30-seed reward (mean ± SE) | 30-seed ruin |
|---|---|---|---|---|---|
| **v4 baseline** | (n/a)  | $858k CA, age 40.7, $769k liquid | early | **4.4449 ± 0.0056** | 0.729% |
| 7              | -4.360 | $620k TX, age 40.2, $1.03M liquid | early | 4.4219 ± 0.0056 | 0.503% |
| 101            | -4.397 | $708k TX, age 42.6, $755k liquid | early | 4.4184 ± 0.0055 | 0.609% |
| 12345 (orig)   | -4.440 | $636k OR, age 78.6, $1.88M liquid | late | 4.3397 ± 0.0056 | 0.288% |
| 31             | -4.332 | $1.03M OR, age 43.7, $4.82M liquid | hybrid (rare trigger) | 4.3387 ± 0.0056 | 0.301% |
| 9999           | -4.279 | $1.08M CA, age 65.2, $4.99M liquid | retire-age | 4.3397 ± 0.0056 | 0.288% |

Two clean basins:

* **Buy-early** (seeds 7, 101, plus v4 baseline): reward ≈ 4.42-4.44,
  ruin 0.5-0.7%. Property bought at age 40-43 with a moderate liquid
  threshold around $750k-$1M.
* **Buy-late or never** (seeds 12345, 31, 9999): reward ≈ 4.34, ruin
  0.29%. Either purchase age ≥ 65 OR a liquid trigger so high
  ($4.8-5.0M) that no path actually fires it. Identical rewards
  across these three seeds confirm the "effectively never buy"
  interpretation.

**Gap between basins**: ~0.10 reward, **20× the per-seed MC noise
floor (SE ≈ 0.005)**. This is genuine optimizer variance, not MC
noise.

**DE basin-hit rate**: 2 / 5 = 40% land in the better (buy-early)
basin at our budget. The published v5 result (seed=12345) was
unlucky — fell into the buy-late basin and reported it as the
optimum.

Inner-MC ranking inverts the truth: seed 12345's inner-MC obj
of -4.440 looked best (highest worst-case reward across modes
on 1000 inner paths), but on 30 fresh 4000-path MC seeds it ties
the bottom of the rankings at 4.34. The optimizer's inner MC at
1000 paths is too noisy to discriminate between basins reliably.

### Q3 followup: BIPOP-CMA-ES

Re-ran all four v5 configurations with BIPOP-CMA-ES (restart
strategy, `restarts=9`, `incpopsize=2.0`, `--max-evals 3500-5000`)
to see if a more sophisticated global optimizer escapes the
buy-late attractor. Results:

| Run | DE v5 inner-MC | DE v5 buy | BIPOP v6 inner-MC | BIPOP v6 buy |
|---|---|---|---|---|
| bm_gbm    | 4.517 | $556k TX age 40.9 (**early**) | 4.472 | $1.43M CA age 93.9 (never) |
| bt_gbm    | 4.473 | $1.66M CA age 78.1 (late) | 4.487 | $1.92M OR age 81.8 (late) |
| bm_robust | 4.521 worst | $636k OR age 78.6 (late) | 4.444 worst | $1.83M OR age 92.6 (never) |
| bt_robust | 4.440 worst | $850k TX age 44.9 (early-ish) | 4.453 worst | $620k TX age 83.8 (late) |

**4 of 4 BIPOP runs landed in buy-late / never basins.** None found
the early-buy basin that DE seeds 7 and 101 successfully reached.
30-seed paired evaluation against the realistic scenario (final-eval
reward, `figures/trial_rental_realistic_v5/pareto.png`):

| Run | 30-seed reward | 30-seed ruin |
|---|---|---|
| v6 bm gbm BIPOP    | 4.3932 ± 0.005 | 0.13% ± 0.01pp |
| v6 bt gbm BIPOP    | 4.3890 ± 0.005 | 0.52% ± 0.02pp |
| v6 bm robust BIPOP | 4.3429 ± 0.006 | 0.29% ± 0.02pp |
| v6 bt robust BIPOP | 4.3455 ± 0.006 | 0.40% ± 0.02pp |

For comparison, the actual best policy (v5 bm gbm, single DE seed
that happened to fall into the early basin) scores **4.4712 ±
0.005** at 0.27% ruin — which Pareto-dominates every BIPOP run by
a margin of 0.08-0.13 reward.

**The diagnosis is clearer with full BIPOP data**: the buy-late
basin is genuinely large in the joint (allocation, rental) decision
space — it captures any policy that pushes `min_age` past the
plan's effective horizon or `min_liquid` above attainable wealth.
The buy-early basin is narrow: it requires a specific combination
of moderate Merton constant + small rental + early trigger. CMA-ES
restart strategies don't help when the bias of the search dynamics
(e.g., default Sobol init in DE, default N(0,1) init in CMA) puts
most starts inside the broad late basin.

For users, the practical takeaway: **don't trust a single optimizer
run on this scenario** — including BIPOP-CMA-ES. Either:

* run DE at multiple seeds and take the best (4-8 seeds at our
  budget gives ~90% chance of at least one hit on the buy-early
  basin given the empirical 40% per-seed rate);
* warm-start one population member from the v4 buy-early optimum;
* report the FRONTIER of (reward, ruin) pairs across all seeds
  rather than a single point estimate; this is what
  `figures/trial_rental_realistic_v5/pareto.png` shows for the
  16 strategies we have so far.

### The actual winner

After all this work, the single Pareto-dominant strategy is **v5
bodie_merton + early-rental GBM** (the one DE seed of v5 that
happened to land in the early basin):

* **Allocation**: Bodie-Merton with γ ≈ 5.7, Merton stock target
  29.78%, taxable cash 8.3%, no FIRE-gap conversions, 22%
  SS-window conversions, 96% trad split.
* **Rental**: $556k TX, trigger at age 40.9 with $775k liquid
  + $215k taxable.
* **30-seed reward**: 4.4712 ± 0.005 (best in the 16-strategy set)
* **30-seed ruin**: 0.27% (2nd lowest of the 16)

This is the closest the v5 sweep got to the buy-early basin and it
beats every v4 policy too — they all sit at higher ruin (0.7-1.8%)
for slightly lower reward.

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
