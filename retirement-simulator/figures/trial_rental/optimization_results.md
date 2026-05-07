# Joint optimization results: trial_rental.yaml

Three policy/objective runs against `trial_rental.yaml`, optimizing
allocation **+** rental decision (price, location, trigger) jointly via
`--optimize-rental`. All target P(W_55 ≥ $2.5M) with P(ruin) ≤ 2%.

## v1 (buggy simulator)

The first pass was run before an audit found four issues in the rental
flow (see commit `b694910`):
* (A) rental equity wasn't folded into `real_wealth`, so the FIRE-prob
  measure penalized the down-payment but credited zero equity;
* (B) the HELOC backstop was only drawn for negative rental cash flow,
  never for spending shortfalls — the ruin metric pretended it was a
  buffer;
* (C) rental income was federally taxed at a 0-baseline marginal rate
  instead of stacked on the year's wages/RMDs;
* (D) the trigger gate used inconsistent inflation deflators.

Bug A was the dominant one — it biased the optimizer hard against early
rental purchase by treating the down-payment as pure wealth destruction.

## v2 (post-fix) vs v1

| Run | Metric | v1 (buggy) | v2 (fixed) |
|---|---|---|---|
| **bond_tent + rental, GBM** | reward | 4.476 | **4.545** |
|  | P(W_55 ≥ $2.5M) | 86.6% | 87.2% |
|  | P(ruin) | 0.46% | 0.66% |
|  | median terminal real $ | $3.07M | $3.91M |
| **bodie_merton + rental, GBM** | reward | 4.472 | **4.598** |
|  | P(W_55 ≥ $2.5M) | 86.3% | 88.5% |
|  | P(ruin) | 0.12% | 0.30% |
|  | median terminal real $ | $2.48M | $4.07M |
| **bond_tent + rental, robust gbm,hist** | reward (worst) | 4.373 | **4.490** |
|  | P(W_55 ≥ $2.5M), gbm/hist | 84.8% / 82.4% | 86.8% / 84.1% |
|  | P(ruin), gbm/hist | 0.52% / 1.54% | 0.54% / 1.58% |
|  | median terminal real $ | $3.37M | $4.91M |

P(ruin) ticks up post-fix because the rental's leverage risk is now
honestly counted (HELOC backstop no longer phantom; equity properly
deflated). Net reward still higher because terminal wealth gains
dominate the small ruin increase, and both stay well inside the 2% cap.

## Rental decision: v1 → v2

| Parameter | bt GBM v1 | bt GBM v2 | bm GBM v1 | bm GBM v2 | bt robust v1 | bt robust v2 |
|---|---|---|---|---|---|---|
| price_real | $1.01M | $1.05M | $0.86M | $1.15M | $1.20M | $0.99M |
| location_state | OR | **TX** | CA | **TX** | OR | **TX** |
| trigger.min_age | 77.0 | **41.9** | 81.6 | **40.7** | 84.6 | **40.4** |
| trigger.min_liquid | $3.68M | $1.23M | $1.77M | $660k | $2.22M | $1.28M |
| trigger.min_taxable | $260k | $325k | $1.17M | $152k | $1.29M | $260k |

The verdict flips completely:

* **v1**: "buy late (77–85) or never; only if you've already won; state
  tax matters less than it should because rental tax is under-stated."
* **v2**: "buy as early as the trigger allows (~age 41), in TX (no state
  tax), with the down payment funded from $1M-ish liquid wealth."

Real estate is now an **accumulation-phase leverage play**, not an
upside-path bequest vehicle.

## Allocation policy under v2

The allocation also adapts to the new leverage:

* **bond_tent v2 GBM**: stock 85% → **31%** at age 45 (right around
  the rental purchase), 7-year span, recovers to 85% by 52. Wealth-
  responsiveness 0.99 (very strong de-risk-when-ahead). 12% taxable
  cash buffer. The deep V at the purchase moment offsets the rental's
  concentration / leverage risk.
* **bodie_merton v2 GBM**: Merton stock 32% as before, but now 13%
  taxable cash (was 6%) and 10% FIRE-gap conversions (was none) — the
  optimizer wants more liquidity buffer alongside the leveraged property.
* **bond_tent v2 robust**: stock 97% → 59% at age 54, span 22; less
  steep V than the GBM-only because historical-tail buffering already
  pulls equity exposure down. Same rental-early conclusion.

## Recommended baseline

Best run: **bodie_merton + early-TX rental, GBM**, reward 4.598 / 5.5,
P(ruin) 0.30%, median terminal $4.07M. Best balance of FIRE-prob
upside with low ruin.

For a robust version that hedges historical-mode tail risk, the
bond_tent robust v2 is the safer pick: reward 4.49 worst-case across
return models, ruin ≤ 1.6% in both modes.

## Methodology notes

* `--paths 1000–1200`, `--maxiter 15–18`, `--popsize 8–10`, `--workers 4`.
* Final eval uses 5000 paths via the CLI's built-in re-eval pass.
* All runs use `--optimize-rental` (5 extra decision dims appended to
  the policy's search vector) on top of the policy class's native
  parameters (9 for bond_tent, 6 for bodie_merton).
* Runtime: bodie_merton ~10–15 min on 4 cores; bond_tent ~25–35 min;
  robust bond_tent ~30–40 min (doubles per-eval cost across modes).
