"""Account and lot model.

Three account types — Taxable, Traditional 401k/IRA, Roth IRA/401k — and three
asset classes — stocks, bonds, cash. Inside the taxable account we track lots
(quantity at cost basis with an acquisition date) so we can distinguish
short-term vs long-term capital gains and apply lot-selection rules at sale.

Inside tax-advantaged accounts, lots are not needed because withdrawals are
taxed by account type (or not at all), independent of basis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


class Asset(str, Enum):
    STOCK = "stock"
    BOND = "bond"
    CASH = "cash"


class AccountType(str, Enum):
    TAXABLE = "taxable"
    TRADITIONAL = "traditional"   # pre-tax 401k or IRA
    ROTH = "roth"                 # Roth 401k or Roth IRA (combined here)


@dataclass
class Lot:
    """A single tax lot inside a taxable account."""
    asset: Asset
    market_value: float
    cost_basis: float
    age_years: float  # holding period in years; >= 1 -> long-term

    @property
    def is_long_term(self) -> bool:
        return self.age_years >= 1.0

    @property
    def unrealized_gain(self) -> float:
        return self.market_value - self.cost_basis

    def sell(self, dollars: float) -> tuple[float, float]:
        """Sell `dollars` of this lot's market value. Returns (realized_gain,
        proceeds). Mutates the lot. If `dollars` exceeds market_value, sells
        the whole lot."""
        if dollars <= 0 or self.market_value <= 0:
            return 0.0, 0.0
        sold = min(dollars, self.market_value)
        fraction = sold / self.market_value
        gain = self.unrealized_gain * fraction
        self.market_value -= sold
        self.cost_basis -= self.cost_basis * fraction
        return gain, sold


@dataclass
class TaxableAccount:
    lots: list[Lot] = field(default_factory=list)

    def value(self, asset: Asset | None = None) -> float:
        if asset is None:
            return sum(lot.market_value for lot in self.lots)
        return sum(lot.market_value for lot in self.lots if lot.asset == asset)

    def cost_basis(self, asset: Asset | None = None) -> float:
        if asset is None:
            return sum(lot.cost_basis for lot in self.lots)
        return sum(lot.cost_basis for lot in self.lots if lot.asset == asset)

    def deposit(self, asset: Asset, amount: float) -> None:
        """Buy `amount` of `asset`. New lot at full basis, age 0."""
        if amount <= 0:
            return
        self.lots.append(Lot(asset=asset, market_value=amount,
                             cost_basis=amount, age_years=0.0))

    def age(self, dt: float = 1.0) -> None:
        """Advance time by `dt` years. All lots age."""
        for lot in self.lots:
            lot.age_years += dt

    def apply_returns(self, asset: Asset, total_return: float,
                      yield_fraction: float) -> tuple[float, float]:
        """Apply a one-period total return decomposed into a yield (paid out
        as cash and immediately reinvested at new basis) and price
        appreciation (deferred).

        Returns (qualified_dividend_income, st_dividend_income). Currently we
        treat all yield as qualified (i.e., LTCG-rate eligible).
        """
        qualified = 0.0
        ordinary = 0.0
        new_lots: list[Lot] = []
        for lot in self.lots:
            if lot.asset != asset:
                continue
            yield_income = lot.market_value * yield_fraction
            appreciation = lot.market_value * (total_return - yield_fraction)
            lot.market_value += appreciation
            # Yield is "distributed" then reinvested; new basis lot.
            if yield_income > 0:
                new_lots.append(Lot(asset=asset, market_value=yield_income,
                                    cost_basis=yield_income, age_years=0.0))
                if asset == Asset.BOND or asset == Asset.CASH:
                    ordinary += yield_income
                else:
                    qualified += yield_income
        self.lots.extend(new_lots)
        return qualified, ordinary

    def sell_for(self, dollars: float, asset: Asset,
                 prefer_long_term: bool = True) -> tuple[float, float, float]:
        """Sell `dollars` worth of `asset`. Lot selection: lowest unrealized
        gain among (LT first if prefer_long_term, else gain/loss optimal).

        Returns (proceeds, lt_gain_realized, st_gain_realized). If the account
        cannot fully fund the request, sells what it can.
        """
        # Sort lots: long-term first if preferred, then by lowest gain
        # (HIFO-ish — actually lowest-gain-first is closer to "tax-loss
        # harvesting" / "specific ID with min gain").
        candidates = [l for l in self.lots if l.asset == asset and l.market_value > 0]
        if prefer_long_term:
            candidates.sort(key=lambda l: (not l.is_long_term, l.unrealized_gain))
        else:
            candidates.sort(key=lambda l: l.unrealized_gain)

        proceeds = 0.0
        lt_gain = 0.0
        st_gain = 0.0
        remaining = dollars
        for lot in candidates:
            if remaining <= 0:
                break
            gain, sold = lot.sell(remaining)
            proceeds += sold
            remaining -= sold
            if lot.is_long_term:
                lt_gain += gain
            else:
                st_gain += gain
        # Compact zero-value lots
        self.lots = [l for l in self.lots if l.market_value > 1e-6]
        return proceeds, lt_gain, st_gain


@dataclass
class TaxAdvantagedAccount:
    """Traditional or Roth. Lots not tracked because basis doesn't matter for
    withdrawal taxation."""
    type: AccountType
    balances: dict[Asset, float] = field(default_factory=lambda: {a: 0.0 for a in Asset})
    # Roth-only: contributions (not earnings) can be withdrawn tax/penalty free.
    roth_basis: float = 0.0

    def value(self, asset: Asset | None = None) -> float:
        if asset is None:
            return sum(self.balances.values())
        return self.balances[asset]

    def deposit(self, asset: Asset, amount: float, is_contribution: bool = True) -> None:
        if amount <= 0:
            return
        self.balances[asset] += amount
        if self.type == AccountType.ROTH and is_contribution:
            self.roth_basis += amount

    def apply_returns(self, asset: Asset, total_return: float) -> None:
        self.balances[asset] *= (1.0 + total_return)

    def withdraw(self, dollars: float, allocation: dict[Asset, float] | None = None
                 ) -> float:
        """Withdraw `dollars` from the account, drawing per `allocation`
        (proportional to current balances if None). Returns dollars actually
        withdrawn (may be less if depleted)."""
        total = self.value()
        if total <= 0 or dollars <= 0:
            return 0.0
        actual = min(dollars, total)
        if allocation is None:
            for a in Asset:
                if total > 0:
                    self.balances[a] -= actual * (self.balances[a] / total)
        else:
            for a, frac in allocation.items():
                take = actual * frac
                if self.balances[a] < take:
                    actual -= (take - self.balances[a])
                    take = self.balances[a]
                self.balances[a] -= take
        if self.type == AccountType.ROTH:
            # Withdrawals come from basis first (FIFO basis treatment).
            self.roth_basis = max(0.0, self.roth_basis - actual)
        return actual


@dataclass
class Portfolio:
    taxable: TaxableAccount = field(default_factory=TaxableAccount)
    traditional: TaxAdvantagedAccount = field(
        default_factory=lambda: TaxAdvantagedAccount(AccountType.TRADITIONAL))
    roth: TaxAdvantagedAccount = field(
        default_factory=lambda: TaxAdvantagedAccount(AccountType.ROTH))

    def total_value(self) -> float:
        return self.taxable.value() + self.traditional.value() + self.roth.value()

    def by_asset(self) -> dict[Asset, float]:
        out: dict[Asset, float] = {a: 0.0 for a in Asset}
        for a in Asset:
            out[a] += self.taxable.value(a)
            out[a] += self.traditional.value(a)
            out[a] += self.roth.value(a)
        return out

    def asset_fractions(self) -> dict[Asset, float]:
        total = self.total_value()
        if total <= 0:
            return {a: 0.0 for a in Asset}
        by = self.by_asset()
        return {a: v / total for a, v in by.items()}
