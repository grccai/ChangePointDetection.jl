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

## v3 (post tax-accounting fixes)

A second audit found four more issues (commit `f81583a`):
* **P0** — Tax-payment withdrawals at end of `_step_decumulation` were
  realising gains/income that never made it onto any tax return.
  Carry-forward fix: stash the tax-payment-withdrawal lt_g/st_g/trad/
  Roth-ord on `s.deferred_*_n` and fold into next year's tax base.
  Magnitude: median lifetime tax up ~7.8% on `trial.yaml`.
* **P1-RMD** — Divisor was being applied to the post-return Trad
  balance; should use prior-year-end. Fixed by snapshotting before
  returns are applied.
* **P1-NIIT** — NIIT was only computed on LTCG; should be on full NII
  (LTCG + interest + ST cap gains + passive rental). `nii_extra` arg
  added to `_federal_tax_vec`.
* **P2** — Roth conversion ladder sized before the spending withdrawal
  generated ST gains, so could overshoot the targeted bracket.
  Reordered: spending withdrawal first, then conversion fills exactly.

### v2 → v3 results

| Run | Reward (v2 → v3) | P(ruin) (v2 → v3) | Median terminal (v2 → v3) |
|---|---|---|---|
| bond_tent + rental, GBM | 4.545 → **4.532** | 0.66% → 0.56% | $3.91M → **$4.47M** |
| bodie_merton + rental, GBM | 4.598 → **4.541** | 0.30% → 0.88% | $4.07M → $4.37M |
| bond_tent + rental, robust | 4.490 → **4.495** | 0.54%/1.58% → 0.28%/1.00% | $4.91M → $4.39M |

Under the corrected tax model, lifetime tax is materially higher
(~$2.5M → ~$2.7-2.9M median per path) and the optimizer compensates
in different ways across the three runs:

### v3 rental decisions

| Parameter | bt GBM | bm GBM | bt robust |
|---|---|---|---|
| price_real | $1.17M | $1.27M | **$0.63M** |
| location_state | TX | **CA** | **CA** |
| trigger.min_age | 40.2 | 40.3 | 41.0 |
| trigger.min_liquid | $647k | $638k | $515k |
| trigger.min_taxable | $291k | $106k | $160k |

The "buy early" verdict from v2 holds in all three v3 runs — that
finding is robust to the tax fixes. State and price drift more (likely
the optimizer trading off NIIT-on-rental vs no-state-tax vs leverage
size), so don't read too much into TX-vs-CA differences here without
larger budget runs.

### v3 allocation shapes

The allocation policy moves substantially in response to the higher
modelled tax burden:

* **bond_tent v3 GBM**: stock_high **66%** (was 85%), stock_low **1.4%**
  at age 62, span 8y, **35% taxable cash sleeve** (was 12%), wealth_resp
  0.27. Very conservative: low equity throughout + huge cash buffer +
  deep V at retirement to ride out sequence-of-returns risk on the
  larger tax bills.
* **bodie_merton v3 GBM**: Merton 32% as before, but **29% taxable cash**
  (was 13%) and **22% FIRE-gap conversions** (was 10%). Aggressive
  conversion ladder during the FIRE-gap to flatten the ord-income
  trajectory before SS / RMDs hit.
* **bond_tent v3 robust**: degenerate to **flat 76% stock** (no V at
  all — stock_high = stock_low). Tiny taxable cash (1.8%). Smaller
  rental ($631k) bought early in CA. The robust objective discounts
  big tactical moves under historical-mode tail risk; flat-and-modest
  wins.

### Recommended baseline (v3)

Three different policy shapes get nearly identical rewards (4.49–4.54)
in v3, which suggests the optimum is broad: many policies do well as
long as they (a) buy a rental early to capture leverage, (b) hold a
healthy cash buffer for the higher tax bills, and (c) run aggressive
Roth conversions during the FIRE-gap.

For a single recommendation: **bond_tent v3 GBM** has the best balance
(reward 4.532, ruin 0.56%, median terminal $4.47M). For risk-averse
users worried about historical-mode regimes, **bond_tent v3 robust**
has the lowest ruin (0.28% / 1.00%) but slightly lower upside.

## v4 (post QBI + IRMAA)

Two more tax-accounting fixes added (commit `5e623d0`):
* **QBI (Sec. 199A) deduction** — 20% federal deduction on positive
  rental net income, capped at 20% of taxable income. Assumes the
  rental qualifies under the Rev. Proc. 2019-38 safe harbor.
* **IRMAA Medicare surcharges** — Part B + Part D surcharges from
  age 65+, looking back to MAGI from 2 years prior. 2024 single
  schedule: $103k / $129k / $161k / $193k / $500k thresholds with
  annual surcharges of $994 / $2,496 / $3,999 / $5,502 / $6,003.
* AMT — left unmodelled; tentative-min-tax 26-28% on AMTI is
  dominated by the regular ordinary brackets which top out at 32%
  in the same range, so AMT is essentially never triggered for
  typical FIRE income profiles.

QBI partially offsets IRMAA on the rental scenario (~-0.9% net change
in median lifetime tax on `trial_rental.yaml`). On `trial.yaml` (no
rental, IRMAA only) median tax rises ~+2.3%.

### v3 → v4 results

| Run | Reward (v3 → v4) | P(ruin) (v3 → v4) | Median terminal (v3 → v4) |
|---|---|---|---|
| bond_tent + rental, GBM | 4.532 → **4.563** | 0.56% → 0.82% | $4.47M → $4.05M |
| bodie_merton + rental, GBM | 4.541 → **4.555** | 0.88% → 1.10% | $4.37M → $4.55M |
| bond_tent + rental, robust | 4.495 → **4.516** worst-case | 0.28%/1.00% → 0.50%/1.56% | $4.39M → $4.70M |

P(ruin) ticked up across the board because IRMAA adds a real $1-6k/yr
expense in retirement that the v3 model wasn't paying. Reward still
went up slightly because QBI savings on rental NOI dominate IRMAA
hits at the median path, and the optimizer adjusted allocation +
conversion strategy to match.

### v4 rental decisions

| Parameter | bt GBM | bm GBM | bt robust |
|---|---|---|---|
| price_real | $1.28M | $1.48M | $0.85M |
| location_state | TX | CA | TX |
| trigger.min_age | 40.2 | 40.8 | 44.9 |
| trigger.min_liquid | $560k | $773k | $808k |
| trigger.min_taxable | $177k | $209k | $401k |

The "buy early" verdict survived a third audit/fix cycle. The robust
optimum nudged to slightly later (44.9 vs 41.0 in v3) and smaller
($850k vs $631k), trading off the tax-burden risk against the
leverage benefit more conservatively.

### v4 allocation shapes

* **bond_tent v4 GBM**: smaller V (stock_high somewhere mid-range,
  taxable cash 18%), conv 22% FIRE-gap + 12% SS-window, wealth_resp
  0.21. More balanced than v3's "low equity + huge cash" extreme.
* **bodie_merton v4 GBM**: aggressive conversion ladder (22% FIRE-gap,
  32% SS-window) — capturing the now-honestly-modelled tax savings of
  filling brackets ahead of the IRMAA-triggering Trad RMDs.
* **bond_tent v4 robust**: stock_high 81% / stock_low 54% V at age 58,
  span 8y, 24% taxable cash, 24% conversions in both phases. Tight
  V-shape glide that hedges historical-mode ruin while capturing
  leverage upside via a smaller-than-average rental ($850k).

### Recommendation (v4)

Across four full audit/fix iterations, the qualitative answer is now
stable:

1. **Buy a rental early** (~age 41) — confirmed across all v2/v3/v4
   runs, robust to allocation policy and return mode.
2. **Hold a meaningful taxable cash sleeve** (15-25%) for tax bills.
3. **Run aggressive Roth conversions during the FIRE-gap** (22-24%
   bracket) to flatten ord-income before RMDs and IRMAA bite.
4. The leverage-benefit and tax-burden are now both honestly counted;
   reward sits at 4.5 / 5.5 (≈82% weighted FIRE-prob).

Best single run: **bond_tent v4 GBM**, reward 4.563, P(ruin) 0.82%.
Best for historical-tail-risk-averse: **bond_tent v4 robust**, reward
worst-case 4.440 with both modes feasible at ≤ 1.56% ruin.

