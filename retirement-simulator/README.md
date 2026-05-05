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
* your age, income, savings rate, target retirement (FIRE) age and end-of-plan
  age, filing status, state tax rate;
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
* state tax as a flat marginal rate;
* required minimum distributions (Uniform Lifetime Table, age 73 onward);
* Roth conversion ladders that fill a target ordinary-income bracket;
* an ACA MAGI cap that constrains conversions before age 65;
* lot-level taxable-account sales preferring long-term, lowest-gain lots;
* dividend / coupon yield treated as taxable annually inside the taxable
  account, while price appreciation is deferred until sale.

It then **optimizes** allocations across the three accounts plus the
Traditional/Roth contribution split and Roth conversion target bracket using
differential evolution against an expected-utility objective with a CRRA
preference and a plan-failure penalty.

## Install

```bash
pip install -e .
# or, without installing:
pip install numpy scipy pyyaml typer
PYTHONPATH=. python -m retire.cli --help
```

Python 3.10+.

## Usage

```bash
# Echo the parsed config and a 25× sanity check
retire validate examples/example.yaml

# Monte Carlo the configured allocation
retire simulate-cmd examples/example.yaml --paths-csv out.csv

# Optimize allocation + contribution split + conversion bracket
retire optimize-cmd examples/example.yaml --paths 1500 --maxiter 30

# Compute a one-off tax bill
retire tax 200000 --ltcg 20000 --filing single --state-rate 0.05
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
| NIIT (3.8%)                                        | State *brackets* (only flat rate)      |
| Standard deduction                                 | IRMAA Medicare surcharges              |
| Social Security provisional-income taxation        | Detailed ACA premium tax credit calc   |
| RMDs (Uniform Lifetime Table)                      | HSA, mega-backdoor Roth                |
| Roth conversion ladder, ACA MAGI cap (cliff)       | Stochastic mortality / longevity risk  |
| Long-term vs short-term capital gains              | Asset location optimization (separate from allocation) |
| Lot-level basis with HIFO/lowest-gain selection    | Margin loans, leverage                 |
| Dividend / coupon yield vs price appreciation      | Tax-loss harvesting credit carryforwards |
| Employer 401k match                                | Custom withdrawal smiles beyond Bengen |
| Bengen smile retirement spending profile           | Inheritance / large bequests           |
| CRRA + bequest objective                           | Variable spending strategies (Guyton-Klinger) |

## Performance

The simulation runs about 15–25 paths/second on a single core for a 60-year
horizon. Most of the cost is `deepcopy` of the lot list at the start of each
path. This is fine for `simulate-cmd` (5000 paths × 60 years finishes in a
few minutes) but makes `optimize-cmd` slow: differential evolution needs
hundreds of objective evaluations, each its own MC. Practical knobs:

* `--paths 500 --maxiter 15 --popsize 6` — low-fidelity scan, ~10 minutes.
* `--workers $(nproc)` — DE parallelizes objective evaluations across cores.
* For serious work, profile and replace the lot-tracking inner loop with a
  vectorised representation (single ndarray of (n_lots, 3) with bulk
  operations); ~50× speedup is achievable.

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
│   ├── optimize.py           # differential evolution allocation optimizer
│   ├── returns.py            # GBM + bootstrap return models
│   ├── simulate.py           # year-by-year accumulation + decumulation engine
│   └── taxes.py              # 2024 federal tax math + RMD divisors
└── tests/
    ├── test_accounts.py
    ├── test_returns.py
    └── test_taxes.py
```
