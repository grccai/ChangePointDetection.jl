# retirement-simulator

A tax-aware retirement portfolio simulation and allocation optimizer with a
command-line interface. Designed for users comfortable with mathematics and
software but unfamiliar with US tax law.

This project is **independent of the surrounding repository** — it shares no
code with `ChangePointDetection.jl` and lives in its own subdirectory only
because branch hosting requires it.

> **Disclaimer.** This is a quantitative model, not financial or tax advice.
> Tax constants are 2024 federal values; brackets, limits, and rules change
> annually and vary by state. Verify any conclusion with a qualified
> professional before acting on it.

## What it does

Given:
* your **birthdate**, simulation **start_date**, **retirement_date**, and
  **end_of_plan_date** (all calendar dates), filing status;
* a list of **income sources**, each with state, [start, end) calendar
  dates, annual gross dollar amount, and (optional) growth rate; multiple
  concurrent sources are summed and mid-year transitions pro-rate cleanly;
* a list of **residency periods** with state and [start, end) dates;
* savings rate and contribution policy (Traditional 401k, Roth 401k, IRAs,
  mega-backdoor Roth, employer match);
* current account balances split across taxable / Traditional / Roth, and
  inside taxable account the per-lot **cost basis** and **holding period**
  (long-term vs short-term);
* return / volatility / correlation assumptions for stocks, bonds, cash;
* expected Social Security and an annual real spending target;

it simulates your wealth path and tax bill year by year through retirement,
including:
* progressive federal ordinary income tax with the standard deduction;
* long-term capital gains stacking on top of ordinary income;
* Net Investment Income Tax (3.8% on the lesser of NII or MAGI excess);
* Social Security taxation via the IRS provisional-income method;
* **state tax for CA, OR, and WA** with separate residency-vs-employment
  apportionment (gains follow residency; wages follow employment);
* required minimum distributions (Uniform Lifetime Table, age 73 onward);
* **mega-backdoor Roth** (after-tax 401k -> Roth, up to 415(c) total);
* Roth conversion ladders that fill a target ordinary-income bracket;
* the **5-year Roth conversion clock** — fresh conversion principal is
  penalty-prone if withdrawn before age 59½ within 5 years of conversion;
* an ACA MAGI cap that constrains conversions before age 65;
* lot-level taxable-account sales preferring long-term cohorts;
* dividend / coupon yield treated as taxable annually inside the taxable
  account, while price appreciation is deferred until sale.

The simulation is **vectorised over Monte Carlo paths** with numpy: 5000
paths × 60 years runs in a couple of seconds (>3000 paths/sec on a single
core; ~180× faster than the original scalar engine).

It then **optimizes** allocations across the three accounts plus the
Traditional/Roth contribution split and Roth conversion target bracket using
differential evolution against an expected-utility objective with a CRRA
preference and a plan-failure penalty. Two location modes:
* `free` (8 vars): jointly optimize allocation and location.
* `heuristic` (4 vars): optimize overall (stock, bond, cash) and let the
  Reichenstein-style location heuristic place each asset (bonds in
  Traditional, stocks in Roth, cash in Taxable). Faster convergence,
  smaller decision space.

## Install

```bash
pip install -e .
# or, without installing:
pip install numpy scipy pyyaml typer
PYTHONPATH=. python -m retire.cli --help
```

Python 3.10+.

## Schema (date-based)

```yaml
profile:
  birthdate:        1991-04-15
  start_date:       2026-05-05      # first simulated year starts here
  retirement_date:  2041-05-05
  end_of_plan_date: 2086-05-05
  filing_status:    single

income:
  sources:
    - {state: CA, start: 2026-05-05, end: 2030-08-01, gross_annual: 200000, growth_rate: 0.03}
    - {state: WA, start: 2030-08-01, end: 2041-05-05, gross_annual: 250000, growth_rate: 0.025}
    # Concurrent sources stack; mid-year transitions pro-rate cleanly.
    - {state: OR, start: 2032-01-01, end: 2034-01-01, gross_annual:  60000}

state_taxes:                       # where you LIVE (taxes investment income)
  residency:
    - {state: CA, start: 2026-05-05, end: 2030-08-01}
    - {state: WA, start: 2030-08-01, end: 2086-05-05}
```

Each income source has (state, start, end, gross_annual, growth_rate). The
year window for simulation year `y` is `[start_date + y years, start_date +
(y+1) years)`. Wages are pro-rated by overlap days, then grown from each
source's `start` at that source's `growth_rate` to the mid-overlap point.
Wages are taxed by the source's state; investment income (capital gains,
dividends, RMDs, conversions) by the residency state(s) active that year.

If you provide the legacy age-based form (`age`, `retirement_age`,
`end_of_plan_age`), it is converted to dates anchored to today.

## Usage

```bash
# Echo the parsed config and a 25× sanity check
retire validate examples/example.yaml

# Monte Carlo the configured allocation
retire simulate-cmd examples/example.yaml --paths-csv out.csv

# Optimize allocation + contribution split + conversion bracket
retire optimize-cmd examples/example.yaml --paths 1500 --maxiter 30

# Optimize using the tax-efficient asset-location heuristic (4 vars instead
# of 8, faster convergence)
retire optimize-cmd examples/example.yaml --location-mode heuristic

# Compute the tax-efficient asset-location placement for a given overall
# allocation against the scenario's current account totals
retire location examples/example.yaml --stock 0.7 --bond 0.25

# Compute a one-off tax bill (federal + a specific state)
retire tax 200000 --ltcg 20000 --filing single --state CA
```

## Mathematical model

### Returns

Annual *real* (inflation-adjusted) arithmetic returns are sampled from a
correlated multivariate lognormal:

  log(1 + R<sub>t</sub>) ~ N(μ<sub>log</sub>, Σ),

where μ<sub>log,i</sub> = log(1 + μ<sub>arith,i</sub>) − ½σ<sub>log,i</sub>² and
σ<sub>log,i</sub>² = log(1 + σ<sub>arith,i</sub>² / (1 + μ<sub>arith,i</sub>)²)
recovers the requested arithmetic mean and standard deviation. The correlation
matrix among log returns is supplied directly. A historical block-bootstrap
mode is also implemented (`returns.block_bootstrap`) for users who prefer not
to assume lognormality; wiring it into the CLI is left as a small extension.

### Tax engine

`taxes.compute_tax` returns a `TaxBill` decomposed into federal ordinary,
federal LTCG, NIIT, and state. LTCG stacks on top of ordinary taxable income:
gains in the slice [I<sub>ord</sub>, I<sub>ord</sub> + G<sub>LT</sub>] are
taxed at the LTCG bracket rate prevailing in that slice, integrated
piecewise. Social Security taxability follows the two-tier provisional income
rule. NIIT is 0.038 × min(NII, MAGI − threshold).

State tax is in `state_taxes.py` with 2024 brackets for **CA**, **OR**, and
**WA**:

* **CA**: progressive 1%–12.3% with the 1% Mental Health Services surcharge
  folded into the 13.3% top bracket. LTCG taxed *as ordinary income* (no
  preferential CA rate).
* **OR**: progressive 4.75%–9.9%. LTCG taxed as ordinary income.
* **WA**: no income tax on wages or ordinary investment income; flat 7%
  Capital Gains Tax on long-term gains exceeding $262,000.

Multi-state apportionment is exposed via a `StateTimeline` of
`StateAssignment(state, start_age, end_age, weight=1.0)` for *residency*
and *employment*. Wages are taxed by employment-state(s); investment income
(dividends, capital gains, RMDs, conversions) is taxed by residency-state(s).
Concurrent assignments with weights model split-state work years.

### Asset location

Tax-efficient asset *location* (which account holds which asset) is a
distinct lever from asset *allocation* (overall mix). The heuristic in
`location.py` greedily places:

* bonds → Traditional first, Taxable second, Roth last;
* cash → Taxable first, Traditional second, Roth last;
* stocks → Roth first, Taxable second, Traditional last.

With this fixed location rule, the optimizer's decision space drops from 8
variables to 4 (overall stock/bond + conversion bracket + Trad/Roth split),
and convergence improves materially.

### Lot accounting

Inside the taxable account each `Lot` carries `(asset, market_value,
cost_basis, age_years)`. Sales select lowest-gain long-term lots first
(approximating "specific ID" with a tax-loss-harvesting bias). Returns are
decomposed into a yield component (dividends / coupons, taxed annually as
qualified for stocks and ordinary for bonds and cash) and a price-appreciation
component (deferred). Tax-advantaged accounts are not lot-tracked because
withdrawals are taxed by account type, not by basis.

### Annual order of operations

Accumulation:

1. Income grows by `growth_rate`.
2. Pre-tax 401k and IRA contributions reduce ordinary income.
3. Federal + state taxes are computed.
4. Roth contributions and remaining take-home are computed.
5. Excess savings (relative to `(1 − savings_rate) × gross`) flow into the
   taxable account, directed to bring it toward target allocation.
6. Returns apply: stocks/bonds appreciate; dividends are taxed.
7. Tax-advantaged accounts rebalance to target (free).
8. Lots age by one year.

Decumulation:

1. Returns apply.
2. RMD pulled from Traditional if age ≥ 73.
3. Optional Roth conversion fills the configured ordinary-income bracket,
   subject to an ACA MAGI cap (if set) before age 65.
4. The withdrawal strategy executes to fund spending net of Social Security.
5. Taxes are computed; tax owed is paid from taxable cash, falling back to
   tax-advantaged accounts.
6. Rebalance, age lots.

### Optimizer

Decision variables:

| index | meaning                                          |
|-------|--------------------------------------------------|
| 0,1   | taxable: stock fraction, bond fraction (cash = 1−stock−bond) |
| 2,3   | traditional: stock, bond                          |
| 4,5   | roth: stock, bond                                |
| 6     | Roth conversion bracket target (snapped to {None, 10%, 12%, 22%, 24%, 32%}) |
| 7     | 401k contribution split (fraction Traditional vs Roth) |

Objective (minimized as the negative):

  J = −E\[ Σ<sub>t=T<sub>ret</sub></sub><sup>T</sup> β<sup>t−T<sub>ret</sub></sup> · u(c<sub>t</sub>) + w<sub>bequest</sub> · u(W<sub>T</sub>) ] + λ · P(failure)

with CRRA utility u(c) = c<sup>1−γ</sup> / (1−γ) (γ ≠ 1) or log(c) (γ = 1).
Default γ = 3 (moderately risk-averse). Optimization uses
`scipy.optimize.differential_evolution` with Sobol initialization. Inner MC is
small (default 1500 paths) to keep evaluations fast; the final reported
allocation is re-evaluated at 5000 paths.

## What is and isn't modelled

| Modelled                                           | Not yet modelled                       |
|----------------------------------------------------|----------------------------------------|
| Federal ordinary tax brackets (2024)               | AMT                                    |
| LTCG stacking on top of ordinary                   | QBI deduction                          |
| NIIT (3.8%)                                        | IRMAA Medicare surcharges              |
| Standard deduction                                 | Detailed ACA premium tax credit calc   |
| Social Security provisional-income taxation        | HSA, FSA                               |
| **CA / OR / WA state brackets** (2024)             | Other states' brackets                 |
| **Multi-state residency / employment timelines**   | Part-year / non-resident apportionment   |
| RMDs (Uniform Lifetime Table)                      | Stochastic mortality / longevity risk  |
| Roth conversion ladder, ACA MAGI cap (cliff)       | Margin loans, leverage                 |
| **5-year Roth conversion clock** with penalty      | Tax-loss harvesting credit carryforwards |
| Long-term vs short-term capital gains              | Custom withdrawal smiles beyond Bengen |
| Cohort-aggregated basis with LT preference         | Inheritance / large bequests           |
| Dividend / coupon yield vs price appreciation      | Variable spending strategies (Guyton-Klinger) |
| Employer 401k match                                | Backdoor Roth income phase-outs        |
| **Mega-backdoor Roth (after-tax 401k)**            | Roth IRA / deductible IRA income limits |
| Bengen smile retirement spending profile           | Self-employed plans (SEP, Solo 401k)   |
| **Asset-location heuristic** (Reichenstein)        | Lump-sum bequests / inheritance        |
| CRRA + bequest objective                           |                                        |
| **Vectorised numpy simulation engine**             |                                        |

## Performance

The simulation engine is **vectorised** across Monte Carlo paths. State per
path lives in numpy arrays (taxable LT/ST cohorts as `(P, 3)`, Trad and Roth
balances as `(P, 3)`, conversion ledger as `(P, H+1)`); the only Python loop
is the year loop, where each year's operations are batched across paths.

On a single core: ~3000 paths/second for a 60-year horizon, so 5000 paths
finishes in under 2 seconds. Scales linearly until `(P, H+1)` arrays no
longer fit in cache (millions of paths). `optimize-cmd` issues a fresh MC
per evaluation; with `--paths 1500 --maxiter 30 --popsize 12` (default) the
optimizer converges in 1–3 minutes.

`--workers $(nproc)` enables scipy's parallel DE evaluations across cores.

The cohort aggregation in the taxable account loses the within-cohort
"specific ID, lowest-gain lot" refinement of the prior scalar engine. In
exchange we get a >100× speedup with negligible difference in realised LT
vs ST tax, which is the load-bearing distinction.

## Limitations and known approximations

* Roth contribution **basis** is treated as FIFO-withdrawable; the IRS
  ordering rules separate basis, conversions (5-year clock), and earnings —
  we do not enforce the conversion 5-year rule. Practical effect is small if
  conversions start ≥ 5 years before the first basis withdrawal.
* The 10% early-withdrawal penalty is applied as +10% to ordinary income on
  the relevant withdrawal — this approximates the additional tax even though
  the IRS treats it as a separate line.
* Income-based contribution phase-outs (Roth IRA, deductible Traditional
  IRA above income limits) are not modelled. The recommendation: assume a
  backdoor Roth for high earners.
* Taxes are computed on dividend income generated *during* the year and paid
  in the same period — i.e., no W-4 withholding vs Q4 estimated-tax
  difference.
* Rebalancing in the taxable account is restricted to redirecting deposits
  rather than realizing gains. This understates rebalancing tax drag if the
  drift is large; in practice this matches a buy-and-hold rebalancer.
* All MC paths share the same inflation realisation across assets within a
  given path (correct), but Social Security COLA uses realised inflation
  (also correct); however we do not separately model wage-growth shocks.

## Things you should set carefully

1. **`market.stocks.real_return` / `vol`.** Historical US large-cap real CAGR
   is ~6.5% with ~17% vol. Forward-looking estimates vary wildly; running the
   simulator at `real_return ∈ {0.04, 0.06, 0.08}` is more informative than
   any single point estimate.
2. **`profile.state_marginal_rate`.** California top marginal is ~13.3%;
   Texas/Florida 0%. Use your *effective* rate (taxes / AGI) not the top
   marginal for typical years.
3. **`spending.annual_real`.** This is the dominant lever. Test sensitivity
   at ±20%.
4. **`withdrawal.aca_magi_cap`.** Roughly 4× the federal poverty line if you
   want to stay under the cliff (varies year to year); set `null` if you do
   not buy ACA insurance pre-65.

## Testing

```bash
pip install pytest
pytest tests/
```

Tests cover the load-bearing math: progressive tax, LTCG stacking, NIIT,
Social Security taxability, RMDs, lot accounting, and return-process moments.

## File layout

```
retirement-simulator/
├── pyproject.toml
├── README.md
├── examples/
│   └── example.yaml          # documented scenario
├── retire/
│   ├── accounts.py           # Lot, TaxableAccount, TaxAdvantagedAccount, Portfolio
│   ├── cli.py                # Typer CLI entrypoint
│   ├── config.py             # YAML parsing, dataclass schema
│   ├── location.py           # tax-efficient asset-location heuristic
│   ├── optimize.py           # differential evolution allocation optimizer
│   ├── returns.py            # GBM + bootstrap return models
│   ├── simulate.py           # vectorised year-by-year simulation engine
│   ├── state_taxes.py        # CA / OR / WA brackets, multi-state timeline
│   ├── taxes.py              # 2024 federal tax math + RMD divisors
│   └── vstate.py             # batched numpy state for the vectorised engine
└── tests/
    ├── test_accounts.py
    ├── test_location.py
    ├── test_returns.py
    ├── test_simulate.py      # integration tests on the vector engine
    ├── test_state_taxes.py
    └── test_taxes.py
```
