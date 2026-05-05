"""YAML config schema with validation.

Configuration is intentionally explicit; defaults live here, not scattered
across the simulator. The `Scenario` is the single object passed to the
simulation and optimizer."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from .accounts import (Asset, AccountType, Lot, Portfolio,
                       TaxableAccount, TaxAdvantagedAccount)
from .returns import AssetParams, MarketModel
from .taxes import FilingStatus


@dataclass
class Profile:
    age: int
    retirement_age: int
    end_of_plan_age: int
    filing_status: FilingStatus = "single"
    state_marginal_rate: float = 0.0  # state income tax flat rate

    def years_to_retirement(self) -> int:
        return max(0, self.retirement_age - self.age)

    def horizon(self) -> int:
        return self.end_of_plan_age - self.age


@dataclass
class Income:
    current_gross: float
    growth_rate: float = 0.03  # nominal


@dataclass
class Contributions:
    """Annual contribution policy. Values can be either a fixed dollar amount
    or the string 'max' (-> use IRS limit). Excess savings beyond these go
    to taxable."""
    trad_401k: float | str = 0.0
    roth_401k: float | str = 0.0
    trad_ira: float | str = 0.0
    roth_ira: float | str = 0.0
    employer_match_rate: float = 0.0  # employer matches X * gross, deposited to traditional 401k


@dataclass
class Savings:
    rate: float  # of gross income
    contributions: Contributions = field(default_factory=Contributions)


@dataclass
class SocialSecurity:
    monthly_at_67: float = 0.0   # nominal in today's dollars; cola'd by inflation
    claim_age: int = 67


@dataclass
class Spending:
    annual_real: float
    smile: Literal["flat", "bengen"] = "flat"
    # bengen: -1% real per year for 10y after retirement, then flat, then +1%/yr
    # for healthcare from age 80


@dataclass
class WithdrawalPolicy:
    strategy: Literal["tax_aware", "ordered", "proportional"] = "tax_aware"
    # Roth conversion: each year, convert traditional -> Roth up to filling
    # this marginal bracket from the top. 0.12 means "fill the 12% bracket".
    # None disables conversions.
    roth_conversion_target_bracket: float | None = 0.12
    # Stop conversions before this MAGI to avoid ACA cliff (only relevant
    # before Medicare age 65). Set to None to disable.
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
    """Target allocation for a single account. Fractions sum to 1."""
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
    return_model: Literal["gbm", "bootstrap"] = "gbm"


@dataclass
class Scenario:
    profile: Profile
    income: Income
    savings: Savings
    spending: Spending
    initial_portfolio: Portfolio
    target_allocations: TargetAllocations
    market: MarketConfig
    social_security: SocialSecurity = field(default_factory=SocialSecurity)
    withdrawal: WithdrawalPolicy = field(default_factory=WithdrawalPolicy)
    simulation: SimulationParams = field(default_factory=SimulationParams)


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


def load_scenario(path: str | Path) -> Scenario:
    """Parse a YAML scenario file."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    profile = Profile(**raw["profile"])
    income = Income(**raw["income"])

    sav_raw = raw["savings"]
    contribs_raw = sav_raw.get("contributions", {})
    contribs = Contributions(**contribs_raw)
    savings = Savings(rate=float(sav_raw["rate"]), contributions=contribs)

    spending = Spending(**raw["spending"])

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

    return Scenario(
        profile=profile, income=income, savings=savings, spending=spending,
        initial_portfolio=portfolio, target_allocations=targets,
        market=market, social_security=ss, withdrawal=wd, simulation=sim,
    )
