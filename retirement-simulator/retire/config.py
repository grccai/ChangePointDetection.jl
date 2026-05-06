"""YAML config schema with validation.

The simulator runs on calendar dates, anchored to `profile.start_date`.
Each simulation year y is the window [start_date + y years, start_date + (y+1) years).
Residency periods and income sources carry explicit YYYY-MM-DD start/end
dates; partial-year overlaps are weighted by fraction of the year.

Backward-compatible age-based shorthand: if you provide `age`,
`retirement_age`, `end_of_plan_age` instead of dates, they are converted
using `start_date` (default = today) and an inferred `birthdate`.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from .accounts import (Asset, AccountType, Lot, Portfolio,
                       TaxableAccount, TaxAdvantagedAccount)
from .returns import AssetParams, MarketModel
from .state_taxes import (StateTimeline, ResidencyPeriod, IncomeSource,
                          DAYS_PER_YEAR, _add_years)
from .taxes import FilingStatus


@dataclass
class Profile:
    birthdate: _dt.date
    start_date: _dt.date
    retirement_date: _dt.date
    end_of_plan_date: _dt.date
    filing_status: FilingStatus = "single"

    def _age_on(self, d: _dt.date) -> float:
        """Age (years.fraction-of-current-year) on date `d`. Returns clean
        integer values on birthday anniversaries — important so that
        boundary checks (catchup at 50, SS claim, RMDs) trigger on the
        right simulation year regardless of leap-year accumulation."""
        int_years = d.year - self.birthdate.year
        if (d.month, d.day) < (self.birthdate.month, self.birthdate.day):
            int_years -= 1
        last_birthday = _add_years(self.birthdate, int_years)
        fractional = (d - last_birthday).days / DAYS_PER_YEAR
        return int_years + fractional

    @property
    def age(self) -> float:
        """Age in years at simulation start."""
        return self._age_on(self.start_date)

    @property
    def retirement_age(self) -> float:
        """Age at retirement_date. Returns clean integer when retirement_date
        falls on a birthday anniversary."""
        return self._age_on(self.retirement_date)

    def age_at_year(self, year_idx: int) -> float:
        """Fractional age at the start of simulation year `year_idx`."""
        return self._age_on(_add_years(self.start_date, year_idx))

    def years_to_retirement(self) -> int:
        return max(0, int(round(
            (self.retirement_date - self.start_date).days / DAYS_PER_YEAR)))

    def horizon(self) -> int:
        return int(round(
            (self.end_of_plan_date - self.start_date).days / DAYS_PER_YEAR))


@dataclass
class Contributions:
    """Annual contribution policy. Values can be either a fixed dollar amount
    or the string 'max' (-> use IRS limit). Excess savings beyond these go
    to taxable.

    `mega_backdoor_roth` is the after-tax 401k contribution that gets
    immediately converted to Roth. Limit = $69,000 (2024 415(c) total)
    minus employee pre-tax minus employer match.
    """
    trad_401k: float | str = 0.0
    roth_401k: float | str = 0.0
    trad_ira: float | str = 0.0
    roth_ira: float | str = 0.0
    mega_backdoor_roth: float | str = 0.0
    employer_match_rate: float = 0.0


@dataclass
class Savings:
    """`rate` is the fraction of *total nominal wages* (across all active
    income sources) that gets saved. Anything not saved is implied living
    expense; tax is paid out of saved-or-taken-home cash before any taxable
    deposit happens."""
    rate: float
    contributions: Contributions = field(default_factory=Contributions)


@dataclass
class SocialSecurity:
    monthly_at_67: float = 0.0   # nominal in today's dollars; cola'd by inflation
    claim_age: int = 67


@dataclass
class FlexibleSpending:
    """Flexible / variable-percentage withdrawal during retirement.

    Spending each year is the smile-adjusted baseline scaled toward the
    current floor when the portfolio is below its retirement-start trajectory:

        ratio = current_real_wealth / wealth_at_retirement_start
        scaling = clamp(1 + sensitivity * (ratio - 1), floor/base, 1)
        spend  = max(floor_real_at_age, base_real * smile_factor * scaling)

    The floor is age-dependent if `floor_change_age` and `floor_after_real`
    are set: floor = `floor_real` strictly before `floor_change_age`, then
    `floor_after_real` from that age on. (Useful for "live tight pre-
    Medicare for ACA subsidy then loosen up at 65.")"""
    floor_real: float
    sensitivity: float = 1.0
    floor_change_age: int | None = None
    floor_after_real: float | None = None

    def floor_at(self, age: float) -> float:
        if self.floor_change_age is None or self.floor_after_real is None:
            return self.floor_real
        return self.floor_after_real if age >= self.floor_change_age else self.floor_real


@dataclass
class Spending:
    """Retirement spending target. `annual_real` is in today's dollars and is
    interpreted as POST-TAX consumption that the simulator must deliver each
    year, grossing-up withdrawals to cover the tax bill.

    `working_annual_real` (optional) is the working-years post-tax living
    budget. If set, it overrides the savings-rate residual: each working
    year, anything above (taxes + contributions + `working_annual_real`)
    flows to taxable savings. Otherwise the simulator falls back to
    `(1 - savings.rate) * gross_wages` as the implied living budget.

    `flexible` (optional) enables variable-percentage withdrawal — see
    `FlexibleSpending`. When omitted, retirement spend is the smile-adjusted
    baseline regardless of portfolio state."""
    annual_real: float
    smile: Literal["flat", "bengen"] = "flat"
    working_annual_real: float | None = None
    flexible: FlexibleSpending | None = None


@dataclass
class WithdrawalPolicy:
    strategy: Literal["tax_aware", "ordered", "proportional"] = "tax_aware"
    roth_conversion_target_bracket: float | None = 0.12
    aca_magi_cap: float | None = None


@dataclass
class MarketConfig:
    stocks: AssetParams
    bonds: AssetParams
    cash: AssetParams
    inflation_mean: float = 0.025
    inflation_vol: float = 0.0
    correlation_stock_bond: float = 0.10
    correlation_stock_cash: float = 0.0
    correlation_bond_cash: float = 0.0

    def to_market_model(self) -> MarketModel:
        C = np.array([
            [1.0, self.correlation_stock_bond, self.correlation_stock_cash],
            [self.correlation_stock_bond, 1.0, self.correlation_bond_cash],
            [self.correlation_stock_cash, self.correlation_bond_cash, 1.0],
        ])
        return MarketModel(
            params={Asset.STOCK: self.stocks, Asset.BOND: self.bonds,
                    Asset.CASH: self.cash},
            correlation=C,
            inflation_mean=self.inflation_mean,
            inflation_vol=self.inflation_vol,
        )


@dataclass
class Allocation:
    stock: float
    bond: float
    cash: float

    def __post_init__(self) -> None:
        s = self.stock + self.bond + self.cash
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"allocation must sum to 1, got {s}")
        for v in (self.stock, self.bond, self.cash):
            if v < -1e-9 or v > 1 + 1e-9:
                raise ValueError("allocation fractions must be in [0,1]")

    def as_dict(self) -> dict[Asset, float]:
        return {Asset.STOCK: self.stock, Asset.BOND: self.bond,
                Asset.CASH: self.cash}


@dataclass
class TargetAllocations:
    taxable: Allocation
    traditional: Allocation
    roth: Allocation


@dataclass
class SimulationParams:
    n_paths: int = 5000
    seed: int = 42
    # 'gbm'           : correlated lognormal sampling from the configured
    #                   means/vols/correlations (default).
    # 'deterministic' : every year uses each asset's `real_return` exactly
    #                   and inflation_mean exactly. Forces n_paths=1 since
    #                   all paths would be identical. Use for fast rough
    #                   approximations / sanity checks.
    # 'historical'    : block-bootstrap from the embedded US 1928-2023
    #                   annual real returns (S&P 500, 10y T-bond, 3mo
    #                   T-bill) with realised CPI for inflation. Captures
    #                   real bull/bear sequence-of-returns dynamics.
    return_model: Literal["gbm", "deterministic", "historical",
                          "historical_ath", "historical_stretched_ath",
                          "bootstrap"] = "gbm"


@dataclass
class GompertzHazard:
    """Gompertz survival model for the benefactor's remaining lifetime.
    Hazard h(age) = b * exp(c * age); per-path arrival year for the
    inheritance is sampled by inverse-CDF given the benefactor's
    current_age.

    Defaults (`profile="healthy_65"`): tuned so qx(65) ≈ 0.005 (matching
    SOA Healthy Annuitant 2012 IAM, midway between male and female), and
    median age at death ≈ 92. Use `profile="average_65"` for population
    mortality (qx(65) ≈ 0.015, median death ≈ 84).
    """
    current_age: float = 65.0   # benefactor's age at sim_start
    profile: Literal["healthy_65", "average_65", "custom"] = "healthy_65"
    b: float | None = None       # Gompertz baseline; required if profile=custom
    c: float | None = None       # Gompertz slope; required if profile=custom

    def resolved(self) -> tuple[float, float]:
        """Return (b, c). Picks profile defaults if custom params unset."""
        if self.profile == "custom":
            if self.b is None or self.c is None:
                raise ValueError("custom Gompertz requires b and c")
            return self.b, self.c
        if self.profile == "healthy_65":
            # qx(65)=0.005, median death ≈ 92
            return 2.4e-5, 0.085
        if self.profile == "average_65":
            # population-level: qx(65)=0.015, median death ≈ 84
            return 7.0e-5, 0.085
        raise ValueError(f"unknown profile: {self.profile}")


@dataclass
class Inheritance:
    """One-time lump-sum deposit to a specified account.

    Timing: either deterministic (`date`) OR stochastic (`hazard`). Exactly
    one must be provided. With `hazard`, each MC path samples its own
    arrival year from the survival distribution; paths where the benefactor
    outlives the simulation horizon don't receive the inheritance.

    `amount_real` is in today's dollars; converted to nominal at deposit
    using realised cumulative inflation.

    Account treatment:
      * `taxable`     — fresh tax lot at basis = market value (stepped-up
                        from estate, or cash receipt). Allocated per the
                        scenario's taxable target at deposit time.
      * `traditional` — added to Trad balance. Inherited-IRA SECURE Act
                        10-year drawdown is NOT modelled.
      * `roth`        — added to Roth balance and to roth_basis (treated as
                        penalty-free principal).

    Estate / inheritance tax is NOT modelled — `amount_real` is the net
    received after any estate-side taxes."""
    amount_real: float
    account: Literal["taxable", "traditional", "roth"] = "taxable"
    date: _dt.date | None = None
    hazard: GompertzHazard | None = None

    def __post_init__(self) -> None:
        if (self.date is None) == (self.hazard is None):
            raise ValueError("Inheritance must specify exactly one of "
                             "`date` (deterministic) or `hazard` (stochastic)")


@dataclass
class RentalPurchaseTrigger:
    """Wealth-conditional purchase policy. Property is bought the first year
    ALL of these hold AND the property hasn't already been purchased.

    No `max_age` — caller can leave the trigger latent if it never fires.
    """
    min_age: float = 0.0                         # earliest age we'd buy
    min_liquid_real_wealth: float = 0.0          # total real wealth gate
    min_taxable_real_wealth: float = 0.0         # taxable-only gate so the
                                                 # downpayment doesn't have to
                                                 # come from 401k/Roth


@dataclass
class RentalProperty:
    """Rental property modeled as a separate asset on the balance sheet.

    Mechanics (per path, per year, after `trigger` fires):
      property_return            ~ correlated lognormal w/ stock & bond
                                   log-returns this year (rho_s, rho_b)
      property_value_real       *= (1 + property_return)
      rent_shock                ~ N(0, rent_shock_vol)   per year
      noi_real                   = (cap_rate * (1 + rent_shock)
                                    - expense_ratio) * property_value_real
      mortgage_payment_nominal   = locked annuity payment (fixed at purchase)
      mortgage_interest_t        = balance_t * rate_n
      taxable_rental_income      = noi_real - mortgage_interest_real
                                   (mortgage interest deductible; depreciation
                                   NOT modelled in v1)
      net_cash_flow_real         = noi_real - mortgage_payment_real
                                    - heloc_interest_real
      after-tax cash             -> deposited to taxable cash sleeve

    Tax sourcing: rental taxable income is taxed by `location_state`
    (source-based), not residency. Federal tax applies as ordinary.

    Borrow-against-equity (no sale event):
      accessible_equity = max(0, ltv_max * value_n - mortgage_n - heloc_n)
      When the simulator would otherwise mark a path failed, draw up to
      `accessible_equity` from a HELOC at `heloc_rate_nominal` instead.
      Subsequent years' cash flow services HELOC interest first.
    """
    # Purchase economics
    price_real: float
    downpayment_frac: float = 0.25
    mortgage_term_years: int = 30
    mortgage_nominal_rate: float = 0.07
    # Operating economics (fractions of property_value_real)
    cap_rate: float = 0.05
    expense_ratio: float = 0.02   # maint + insurance + property tax + vacancy
    rent_shock_vol: float = 0.05  # annual stdev of multiplicative rent shock
    # Property appreciation (excess of CPI), stochastic, correlated with stock
    # and bond log-returns.
    appreciation_real_mean: float = 0.005   # excess of CPI
    appreciation_real_vol: float = 0.10     # annual stdev (real)
    correlation_with_stock: float = 0.30    # contemporaneous log-return corr
    correlation_with_bond: float = 0.10     # ditto
    # Tax sourcing
    location_state: str = "CA"
    # Borrow-against-equity backstop
    ltv_max: float = 0.80
    heloc_rate_nominal: float = 0.08
    # Wealth-conditional purchase trigger
    trigger: RentalPurchaseTrigger = field(default_factory=RentalPurchaseTrigger)


@dataclass
class Scenario:
    profile: Profile
    state_taxes: StateTimeline
    savings: Savings
    spending: Spending
    initial_portfolio: Portfolio
    target_allocations: TargetAllocations
    market: MarketConfig
    social_security: SocialSecurity = field(default_factory=SocialSecurity)
    withdrawal: WithdrawalPolicy = field(default_factory=WithdrawalPolicy)
    simulation: SimulationParams = field(default_factory=SimulationParams)
    inheritances: list[Inheritance] = field(default_factory=list)
    rental_property: RentalProperty | None = None


# ---------- YAML helpers ----------

def _make_lot(d: dict[str, Any]) -> Lot:
    return Lot(asset=Asset(d["asset"]),
               market_value=float(d["market_value"]),
               cost_basis=float(d["cost_basis"]),
               age_years=float(d.get("age_years", 1.0)))


def _make_portfolio(d: dict[str, Any]) -> Portfolio:
    p = Portfolio()
    for lot_dict in d.get("taxable", {}).get("lots", []):
        p.taxable.lots.append(_make_lot(lot_dict))
    for asset_str, val in d.get("traditional", {}).items():
        p.traditional.balances[Asset(asset_str)] = float(val)
    roth = d.get("roth", {})
    for asset_str, val in roth.items():
        if asset_str == "basis":
            continue
        p.roth.balances[Asset(asset_str)] = float(val)
    p.roth.roth_basis = float(roth.get("basis", 0.0))
    return p


def _make_allocation(d: dict[str, Any]) -> Allocation:
    return Allocation(stock=float(d["stock"]), bond=float(d["bond"]),
                      cash=float(d["cash"]))


def _to_date(v: Any) -> _dt.date:
    """Accept date, datetime, or 'YYYY-MM-DD' string."""
    if isinstance(v, _dt.date) and not isinstance(v, _dt.datetime):
        return v
    if isinstance(v, _dt.datetime):
        return v.date()
    if isinstance(v, str):
        return _dt.date.fromisoformat(v)
    raise TypeError(f"expected date or 'YYYY-MM-DD' string, got {type(v).__name__}")


def _parse_profile(raw: dict[str, Any]) -> Profile:
    """Date-based or age-based form. Date-based fields take precedence."""
    fs = raw.get("filing_status", "single")
    if "birthdate" in raw or "start_date" in raw:
        if "birthdate" not in raw:
            raise ValueError("date-based profile requires `birthdate`")
        bd = _to_date(raw["birthdate"])
        sd = _to_date(raw.get("start_date", _dt.date.today()))
        rd = _to_date(raw["retirement_date"])
        ed = _to_date(raw["end_of_plan_date"])
        return Profile(birthdate=bd, start_date=sd, retirement_date=rd,
                       end_of_plan_date=ed, filing_status=fs)
    # Legacy age-based form
    if "age" not in raw:
        raise ValueError("profile must specify either dates (birthdate, "
                         "retirement_date, end_of_plan_date) or ages "
                         "(age, retirement_age, end_of_plan_age)")
    today = _dt.date.today()
    age = int(raw["age"])
    bd = _add_years(today, -age)
    rd = _add_years(today, int(raw["retirement_age"]) - age)
    ed = _add_years(today, int(raw["end_of_plan_age"]) - age)
    return Profile(birthdate=bd, start_date=today, retirement_date=rd,
                   end_of_plan_date=ed, filing_status=fs)


def _parse_state_timeline(raw: dict[str, Any], income_raw: dict[str, Any] | list,
                          profile: Profile) -> StateTimeline:
    """Build the StateTimeline from `state_taxes` and `income.sources` blocks.

    Backward-compatible shorthand:
      state_taxes: {state: CA}        -> CA residency for full plan
      state_taxes:
        residency:
          - {state: CA, start: 2026-05-05, end: 2030-08-01}
          - {state: WA, start: 2030-08-01, end: 2086-05-05}
    """
    timeline = StateTimeline()

    # --- Residency periods ---
    if isinstance(raw, dict) and raw.get("state"):
        # Shorthand: one state for the whole plan.
        timeline.residency.append(ResidencyPeriod(
            state=raw["state"], start=profile.start_date,
            end=_add_years(profile.end_of_plan_date, 1)))
    if isinstance(raw, dict) and "residency" in raw:
        for entry in raw["residency"]:
            timeline.residency.append(ResidencyPeriod(
                state=entry["state"],
                start=_to_date(entry["start"]),
                end=_to_date(entry["end"]),
            ))

    # --- Income sources ---
    sources_raw: list[dict[str, Any]] = []
    if isinstance(income_raw, list):
        sources_raw = income_raw
    elif isinstance(income_raw, dict):
        sources_raw = income_raw.get("sources", [])
        # Legacy single-source form: income: {current_gross, growth_rate, ...}
        if not sources_raw and "current_gross" in income_raw:
            employment_state = (raw.get("state") if isinstance(raw, dict)
                                else None) or "NONE"
            sources_raw = [{
                "state": employment_state,
                "start": profile.start_date,
                "end": profile.retirement_date,
                "gross_annual": float(income_raw["current_gross"]),
                "growth_rate": float(income_raw.get("growth_rate", 0.0)),
            }]
    for src in sources_raw:
        timeline.income_sources.append(IncomeSource(
            state=src["state"],
            start=_to_date(src["start"]),
            end=_to_date(src["end"]),
            gross_annual=float(src["gross_annual"]),
            growth_rate=float(src.get("growth_rate", 0.0)),
        ))
    return timeline


def load_scenario(path: str | Path) -> Scenario:
    """Parse a YAML scenario file."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    profile = _parse_profile(raw["profile"])

    sav_raw = raw["savings"]
    contribs = Contributions(**sav_raw.get("contributions", {}))
    savings = Savings(rate=float(sav_raw["rate"]), contributions=contribs)

    spending_raw = dict(raw["spending"])
    flex_raw = spending_raw.pop("flexible", None)
    spending = Spending(**spending_raw)
    if flex_raw:
        spending.flexible = FlexibleSpending(**flex_raw)

    portfolio = _make_portfolio(raw["initial_portfolio"])
    targets_raw = raw["target_allocations"]
    targets = TargetAllocations(
        taxable=_make_allocation(targets_raw["taxable"]),
        traditional=_make_allocation(targets_raw["traditional"]),
        roth=_make_allocation(targets_raw["roth"]),
    )

    mkt_raw = raw["market"]
    market = MarketConfig(
        stocks=AssetParams(**mkt_raw["stocks"]),
        bonds=AssetParams(**mkt_raw["bonds"]),
        cash=AssetParams(**mkt_raw["cash"]),
        inflation_mean=float(mkt_raw.get("inflation_mean", 0.025)),
        inflation_vol=float(mkt_raw.get("inflation_vol", 0.0)),
        correlation_stock_bond=float(mkt_raw.get("correlation_stock_bond", 0.10)),
        correlation_stock_cash=float(mkt_raw.get("correlation_stock_cash", 0.0)),
        correlation_bond_cash=float(mkt_raw.get("correlation_bond_cash", 0.0)),
    )

    ss = SocialSecurity(**raw.get("social_security", {}))
    wd = WithdrawalPolicy(**raw.get("withdrawal", {}))
    sim = SimulationParams(**raw.get("simulation", {}))

    timeline = _parse_state_timeline(
        raw.get("state_taxes", {}), raw.get("income", {}), profile)

    inheritances: list[Inheritance] = []
    for inh in raw.get("inheritances", []):
        kwargs = dict(amount_real=float(inh["amount_real"]),
                      account=inh.get("account", "taxable"))
        if "date" in inh:
            kwargs["date"] = _to_date(inh["date"])
        if "hazard" in inh:
            h = inh["hazard"]
            kwargs["hazard"] = GompertzHazard(
                current_age=float(h.get("current_age", 65.0)),
                profile=h.get("profile", "healthy_65"),
                b=h.get("b"), c=h.get("c"),
            )
        inheritances.append(Inheritance(**kwargs))

    rental = None
    rp_raw = raw.get("rental_property")
    if rp_raw is not None:
        tr = rp_raw.get("trigger", {})
        # Backward-compatibility: accept the old `appreciation_real` scalar
        # as the new `appreciation_real_mean` (deterministic if no _vol set).
        appr_mean = float(rp_raw.get("appreciation_real_mean",
                                       rp_raw.get("appreciation_real", 0.005)))
        rental = RentalProperty(
            price_real=float(rp_raw["price_real"]),
            downpayment_frac=float(rp_raw.get("downpayment_frac", 0.25)),
            mortgage_term_years=int(rp_raw.get("mortgage_term_years", 30)),
            mortgage_nominal_rate=float(rp_raw.get("mortgage_nominal_rate", 0.07)),
            cap_rate=float(rp_raw.get("cap_rate", 0.05)),
            expense_ratio=float(rp_raw.get("expense_ratio", 0.02)),
            rent_shock_vol=float(rp_raw.get("rent_shock_vol", 0.05)),
            appreciation_real_mean=appr_mean,
            appreciation_real_vol=float(rp_raw.get("appreciation_real_vol", 0.10)),
            correlation_with_stock=float(rp_raw.get("correlation_with_stock", 0.30)),
            correlation_with_bond=float(rp_raw.get("correlation_with_bond", 0.10)),
            location_state=str(rp_raw.get("location_state", "CA")),
            ltv_max=float(rp_raw.get("ltv_max", 0.80)),
            heloc_rate_nominal=float(rp_raw.get("heloc_rate_nominal", 0.08)),
            trigger=RentalPurchaseTrigger(
                min_age=float(tr.get("min_age", 0.0)),
                min_liquid_real_wealth=float(tr.get("min_liquid_real_wealth", 0.0)),
                min_taxable_real_wealth=float(tr.get("min_taxable_real_wealth", 0.0)),
            ),
        )

    return Scenario(
        profile=profile, state_taxes=timeline,
        savings=savings, spending=spending,
        initial_portfolio=portfolio, target_allocations=targets,
        market=market,
        social_security=ss, withdrawal=wd, simulation=sim,
        inheritances=inheritances,
        rental_property=rental,
    )
