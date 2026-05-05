"""Year-by-year simulation engine.

State variables that evolve:
  * portfolio (taxable lots, traditional balances by asset, Roth balances)
  * age
  * cumulative_inflation_factor (real -> nominal conversion)
  * income_path (nominal)

Each simulated year, in order:
  1. Update inflation factor.
  2. If age < retirement_age:
       a. Compute gross income, taxes during accumulation
       b. Apply contributions: trad 401k (pre-tax), Roth 401k (post-tax),
          trad IRA, Roth IRA, employer match (-> trad 401k)
       c. Compute taxable savings residual; deposit to taxable account
       d. Apply asset returns (with dividend yield treatment in taxable)
       e. Rebalance toward target allocation per account
     Else (decumulation):
       a. Apply asset returns
       b. Compute desired spending in nominal
       c. Take RMDs if applicable
       d. Optional Roth conversion to fill bracket / under ACA cap
       e. Withdraw to cover spending net of SS using configured strategy
       f. Compute taxes due; gross-up withdrawal
       g. Rebalance
  3. Age the lots by 1 year.

We return per-path arrays of: terminal real wealth, year-end real wealth,
spending shortfall flags, total lifetime taxes, and other diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from typing import Callable

import numpy as np

from .accounts import (Asset, AccountType, Lot, Portfolio,
                       TaxableAccount, TaxAdvantagedAccount)
from .config import (Scenario, Allocation, TargetAllocations,
                     WithdrawalPolicy, Spending)
from .returns import MarketModel, sample_gbm_paths, sample_inflation
from .taxes import (TAX_2024, FilingStatus, compute_tax, TaxBill,
                    required_min_distribution, top_of_bracket,
                    progressive_tax)


@dataclass
class PathResult:
    """Per-path summary."""
    terminal_real_wealth: float
    real_wealth_by_year: np.ndarray   # length = horizon+1
    real_spending_by_year: np.ndarray # length = horizon (target real spend)
    real_shortfall_by_year: np.ndarray  # length = horizon (positive = unmet)
    lifetime_real_tax: float
    failed: bool   # ran out before end of plan


@dataclass
class SimResult:
    paths: list[PathResult]

    @property
    def n_paths(self) -> int:
        return len(self.paths)

    def terminal_quantiles(self, q: list[float] = [0.05, 0.5, 0.95]
                           ) -> dict[float, float]:
        x = np.array([p.terminal_real_wealth for p in self.paths])
        return {qi: float(np.quantile(x, qi)) for qi in q}

    def failure_rate(self) -> float:
        return sum(1 for p in self.paths if p.failed) / max(1, self.n_paths)

    def cvar_failure(self, alpha: float = 0.05) -> float:
        """Average of the worst alpha-fraction terminal wealths (real $)."""
        x = np.array([p.terminal_real_wealth for p in self.paths])
        x.sort()
        k = max(1, int(np.ceil(alpha * len(x))))
        return float(np.mean(x[:k]))


# ---------- helpers ----------

def _resolve_contrib(value: float | str, limit: float) -> float:
    if isinstance(value, str) and value == "max":
        return limit
    return float(value)


def _spending_factor(years_into_retirement: int, smile: str) -> float:
    """Multiplier on baseline real spending."""
    if smile == "flat":
        return 1.0
    if smile == "bengen":
        # Bengen smile approximation: -1%/yr first 10y, flat 10y, +1%/yr after
        if years_into_retirement < 10:
            return 1.0 - 0.01 * years_into_retirement
        if years_into_retirement < 20:
            return 0.90
        return min(1.10, 0.90 + 0.01 * (years_into_retirement - 20))
    raise ValueError(f"unknown spending smile: {smile}")


def _rebalance_account_via_buys(account: TaxAdvantagedAccount,
                                target: Allocation) -> None:
    """In tax-advantaged accounts we can rebalance freely (no tax). Just set
    each balance to target * total."""
    total = account.value()
    if total <= 0:
        return
    t = target.as_dict()
    for a in Asset:
        account.balances[a] = total * t[a]


def _rebalance_taxable(taxable: TaxableAccount, target: Allocation,
                       allow_sells: bool = False
                       ) -> tuple[float, float]:
    """Best-effort rebalance. By default only deposits redirect (tax-free).
    With allow_sells=True, will also sell overweight to fix drift; returns
    (lt_realized, st_realized).

    For now we just track the drift and only sell if drift > 5pp on any asset
    when allow_sells=True. Otherwise no action — caller is responsible for
    directing deposits."""
    if not allow_sells:
        return 0.0, 0.0
    total = taxable.value()
    if total <= 0:
        return 0.0, 0.0
    t = target.as_dict()
    lt = st = 0.0
    for a in Asset:
        cur = taxable.value(a)
        want = total * t[a]
        if cur - want > 0.05 * total:  # >5pp overweight
            excess = cur - want
            _, _lt, _st = taxable.sell_for(excess, a)
            lt += _lt
            st += _st
    return lt, st


def _direct_taxable_deposit(taxable: TaxableAccount, amount: float,
                            target: Allocation) -> None:
    """Spread a deposit across assets to push toward target allocation."""
    if amount <= 0:
        return
    total_after = taxable.value() + amount
    t = target.as_dict()
    # For each asset, want_after = total_after * t[asset].
    # Deposit need = max(0, want_after - current). May not sum to amount;
    # normalize.
    needs = {a: max(0.0, total_after * t[a] - taxable.value(a)) for a in Asset}
    s = sum(needs.values())
    if s <= 0:
        # All overweight already; just deposit per target weights
        for a in Asset:
            taxable.deposit(a, amount * t[a])
        return
    for a in Asset:
        taxable.deposit(a, amount * needs[a] / s)


def _apply_asset_returns(portfolio: Portfolio,
                         returns: dict[Asset, float],
                         model: MarketModel
                         ) -> tuple[float, float]:
    """Apply one-period returns. Returns (taxable_qualified_div_income,
    taxable_ordinary_div_income) for the year — these flow to the tax bill."""
    qual = 0.0
    ord_ = 0.0
    for a in Asset:
        r = returns[a]
        # Tax-advantaged: just compound
        portfolio.traditional.balances[a] *= (1.0 + r)
        portfolio.roth.balances[a] *= (1.0 + r)
        # Taxable: split into yield (dividend) + appreciation (deferred)
        yf = model.yield_fraction[a] if r > 0 else 0.0
        # Cap yield_fraction effect when total_return is negative (no
        # negative dividends): apply price-only return
        if r < 0:
            for lot in portfolio.taxable.lots:
                if lot.asset == a:
                    lot.market_value *= (1.0 + r)
        else:
            q, o = portfolio.taxable.apply_returns(a, r, yf)
            qual += q
            ord_ += o
    return qual, ord_


# ---------- accumulation ----------

@dataclass
class _AccumState:
    nominal_income: float
    cumulative_inflation: float


def _step_accumulation(scn: Scenario, p: Portfolio, age: int, year_idx: int,
                       returns_year: dict[Asset, float], inflation: float,
                       state: _AccumState) -> tuple[float, float]:
    """Run one accumulation year. Returns (real_wealth_end, real_taxes)."""
    targets = scn.target_allocations
    fs = scn.profile.filing_status
    state.cumulative_inflation *= (1.0 + inflation)
    state.nominal_income *= (1.0 + scn.income.growth_rate) if year_idx > 0 else 1.0

    gross = state.nominal_income
    # Contribution caps (catchup at 50+)
    catchup_401k = TAX_2024.contrib_limit_401k_catchup if age >= 50 else 0.0
    catchup_ira = TAX_2024.contrib_limit_ira_catchup if age >= 50 else 0.0
    limit_401k = TAX_2024.contrib_limit_401k + catchup_401k
    limit_ira = TAX_2024.contrib_limit_ira + catchup_ira

    c = scn.savings.contributions
    trad_401k = min(_resolve_contrib(c.trad_401k, limit_401k), limit_401k, gross)
    roth_401k = min(_resolve_contrib(c.roth_401k, limit_401k - trad_401k),
                    limit_401k - trad_401k, gross - trad_401k)
    trad_ira = min(_resolve_contrib(c.trad_ira, limit_ira), limit_ira)
    roth_ira = min(_resolve_contrib(c.roth_ira, limit_ira - trad_ira),
                   limit_ira - trad_ira)
    employer_match = c.employer_match_rate * gross

    # Deposit contributions BEFORE returns are applied this year (timing
    # convention; contributions happen at start of year).
    p.traditional.deposit(_alloc_dom_asset(targets.traditional), trad_401k + employer_match)
    p.roth.deposit(_alloc_dom_asset(targets.roth), roth_401k)
    # IRAs go to same accounts (we don't separate IRA from 401k).
    p.traditional.deposit(_alloc_dom_asset(targets.traditional), trad_ira)
    p.roth.deposit(_alloc_dom_asset(targets.roth), roth_ira, is_contribution=True)

    # Apply returns first so this year's dividend income is included in the
    # same tax bill as wages (avoids double-applying the standard deduction).
    qual_div, ord_div = _apply_asset_returns(
        p, returns_year, scn.market.to_market_model())

    ord_income = max(0.0, gross - trad_401k - trad_ira) + ord_div
    bill = compute_tax(
        ordinary_income=ord_income, ltcg_income=qual_div, ss_benefit=0.0,
        tax_exempt_interest=0.0, filing_status=fs,
        state_marginal_rate=scn.profile.state_marginal_rate, ty=TAX_2024,
    )
    bill_total = bill.total

    # After-tax take-home from wages (dividends are reinvested in the lots
    # already; tax owed on them is paid out-of-band from taxable cash).
    take_home = gross - trad_401k - bill.total - roth_401k - trad_ira - roth_ira
    implied_living = (1.0 - scn.savings.rate) * gross
    taxable_savings = max(0.0, take_home - implied_living)
    _direct_taxable_deposit(p.taxable, taxable_savings, targets.taxable)

    # Rebalance tax-advantaged accounts (free).
    _rebalance_account_via_buys(p.traditional, targets.traditional)
    _rebalance_account_via_buys(p.roth, targets.roth)

    # Age lots
    p.taxable.age(1.0)

    real_wealth = p.total_value() / state.cumulative_inflation
    real_taxes = bill_total / state.cumulative_inflation
    return real_wealth, real_taxes


def _alloc_dom_asset(target: Allocation) -> Asset:
    """For deposits into tax-advantaged accounts we just deposit into the
    most-target asset; rebalance step squares it up."""
    t = target.as_dict()
    return max(t, key=t.get)


def _withdraw_from_taxable_for_taxes(taxable: TaxableAccount, amount: float,
                                     target: Allocation
                                     ) -> tuple[float, float, float]:
    """Pay current-year tax from taxable account. Prefer cash, then bonds,
    then stocks. Returns (paid, lt_realized, st_realized) where `paid` is
    actual dollars sourced (may be less than `amount` if depleted)."""
    if amount <= 0:
        return 0.0, 0.0, 0.0
    paid = lt = st = 0.0
    for a in (Asset.CASH, Asset.BOND, Asset.STOCK):
        if amount <= 0:
            break
        avail = taxable.value(a)
        if avail <= 0:
            continue
        take = min(amount, avail)
        proc, _lt, _st = taxable.sell_for(take, a)
        paid += proc
        lt += _lt
        st += _st
        amount -= proc
    return paid, lt, st


# ---------- decumulation ----------

def _step_decumulation(scn: Scenario, p: Portfolio, age: int, year_idx: int,
                       returns_year: dict[Asset, float], inflation: float,
                       state: _AccumState,
                       prior_year_end_trad: float
                       ) -> tuple[float, float, float, float]:
    """Run one decumulation year.

    Returns (real_wealth_end, real_taxes, real_target_spend, real_shortfall).
    """
    targets = scn.target_allocations
    fs = scn.profile.filing_status
    state.cumulative_inflation *= (1.0 + inflation)

    # Apply returns FIRST (start-of-year balance grows during the year before
    # withdrawal — common convention).
    qual_div, ord_div = _apply_asset_returns(p, returns_year, scn.market.to_market_model())

    years_into_retire = age - scn.profile.retirement_age
    factor = _spending_factor(years_into_retire, scn.spending.smile)
    real_target_spend = scn.spending.annual_real * factor
    nominal_target_spend = real_target_spend * state.cumulative_inflation

    # Social Security (nominal, COLA'd by inflation since today)
    ss_nominal = 0.0
    if age >= scn.social_security.claim_age and scn.social_security.monthly_at_67 > 0:
        # Adjust claim age vs 67: simplified linear PIA adjustment. Real users
        # should set monthly_at_67 to their statement value.
        ss_nominal = (scn.social_security.monthly_at_67 * 12.0
                      * state.cumulative_inflation)

    # 1) RMDs (forced traditional withdrawal)
    rmd = required_min_distribution(age, prior_year_end_trad)
    rmd_taken = p.traditional.withdraw(rmd) if rmd > 0 else 0.0

    # Track ordinary income flowing into tax bill
    ordinary_income = ord_div + rmd_taken
    ltcg_income = qual_div

    # 2) Optional Roth conversion ladder
    wd = scn.withdrawal
    conversion = 0.0
    if (wd.roth_conversion_target_bracket is not None
            and p.traditional.value() > 0):
        # Target ordinary income at top of bracket
        target_ordinary = top_of_bracket(wd.roth_conversion_target_bracket, fs)
        # subtract std deduction since brackets apply to taxable income
        target_gross = target_ordinary + TAX_2024.std_deduction[fs]
        room = max(0.0, target_gross - (ordinary_income + ss_nominal * 0.85))
        # ACA cap (modified AGI) — enforce only before Medicare age 65
        if wd.aca_magi_cap is not None and age < 65:
            room = min(room, max(0.0, wd.aca_magi_cap * state.cumulative_inflation
                                 - (ordinary_income + ltcg_income + ss_nominal)))
        conversion = min(room, p.traditional.value())
        if conversion > 0:
            p.traditional.withdraw(conversion)
            # Convert to Roth (deposit at current target allocation)
            p.roth.deposit(_alloc_dom_asset(targets.roth), conversion,
                           is_contribution=False)
            ordinary_income += conversion

    # 3) Spending — withdraw to cover (target_spend - SS), gross-up for tax
    net_need = max(0.0, nominal_target_spend - ss_nominal)
    # Iteratively gross up: tax on incremental withdrawal depends on source.
    # We do: 2 passes. Pass 1: withdraw at face. Pass 2: top up by computed tax.
    withdrawals = _execute_withdrawal_strategy(
        p, net_need, age, scn, ordinary_income, ltcg_income, ss_nominal,
        state.cumulative_inflation,
    )
    ordinary_income += withdrawals.ordinary_added
    ltcg_income += withdrawals.ltcg_added

    bill = compute_tax(
        ordinary_income=ordinary_income, ltcg_income=ltcg_income,
        ss_benefit=ss_nominal, tax_exempt_interest=0.0,
        filing_status=fs,
        state_marginal_rate=scn.profile.state_marginal_rate, ty=TAX_2024,
    )

    # Pay taxes by drawing additional dollars from the same strategy.
    # Don't double-withdraw (we treat withdrawals.gross as already covering
    # net_need; tax is incremental).
    paid, lt_extra, st_extra = _withdraw_from_taxable_for_taxes(
        p.taxable, bill.total, targets.taxable)
    ltcg_income += lt_extra
    ordinary_income += st_extra
    remaining_tax = bill.total - paid
    if remaining_tax > 1e-6:
        # Top up from tax-advantaged. Pre-59.5 prefer Roth basis (no penalty);
        # 59.5+ prefer Traditional (already paying ordinary tax).
        if age < 59.5 and p.roth.roth_basis > 0:
            take = min(remaining_tax, p.roth.roth_basis)
            p.roth.withdraw(take)
            remaining_tax -= take
        if remaining_tax > 0 and p.traditional.value() > 0:
            take = min(remaining_tax, p.traditional.value())
            p.traditional.withdraw(take)
            remaining_tax -= take
        if remaining_tax > 0 and p.roth.value() > 0:
            take = min(remaining_tax, p.roth.value())
            p.roth.withdraw(take)
            remaining_tax -= take
        # If still remaining, plan failed to pay tax -> reflect as shortfall.

    # Funded amount = what we actually withdrew for spending + SS - any
    # tax paid out of the spending withdrawal (we already paid tax
    # separately above). So funded == withdrawals.gross + ss_nominal.
    nominal_received = withdrawals.gross + ss_nominal
    nominal_shortfall = max(0.0, nominal_target_spend - nominal_received) + remaining_tax
    real_shortfall = nominal_shortfall / state.cumulative_inflation

    # Rebalance tax-advantaged
    _rebalance_account_via_buys(p.traditional, targets.traditional)
    _rebalance_account_via_buys(p.roth, targets.roth)

    p.taxable.age(1.0)

    real_wealth = p.total_value() / state.cumulative_inflation
    real_taxes = bill.total / state.cumulative_inflation
    return real_wealth, real_taxes, real_target_spend, real_shortfall


@dataclass
class _WithdrawalResult:
    gross: float           # total nominal withdrawn from accounts
    ordinary_added: float  # added to ordinary income for the tax bill
    ltcg_added: float      # added to LTCG for the tax bill


def _execute_withdrawal_strategy(p: Portfolio, net_need: float, age: int,
                                 scn: Scenario, ord_so_far: float,
                                 ltcg_so_far: float, ss_nominal: float,
                                 inflation_factor: float) -> _WithdrawalResult:
    """Withdraw `net_need` nominal dollars (pre-tax). Returns income added
    to the tax categories.

    Strategy 'tax_aware':
      Step A: Sell taxable lots (LT first, lowest gain first). Adds realized
              LT gains to LTCG income.
      Step B: If still need money, and age >= 59.5: traditional 401k.
      Step C: If <59.5 and need more: Roth basis (penalty-free), then
              taxable ST (penalty doesn't apply), then traditional w/ penalty
              (we model the 10% penalty as additional tax).
      Step D: 59.5+: Roth (last, to preserve tax-free compounding).
    """
    if net_need <= 0:
        return _WithdrawalResult(0.0, 0.0, 0.0)
    strategy = scn.withdrawal.strategy
    targets = scn.target_allocations

    gross = 0.0
    ord_add = 0.0
    ltcg_add = 0.0
    remaining = net_need

    if strategy == "proportional":
        total = p.total_value()
        if total > 0:
            shares = {
                "tax": p.taxable.value() / total,
                "trad": p.traditional.value() / total,
                "roth": p.roth.value() / total,
            }
            # Taxable
            tx_take = remaining * shares["tax"]
            for a in (Asset.STOCK, Asset.BOND, Asset.CASH):
                if tx_take <= 0:
                    break
                avail = p.taxable.value(a)
                if avail <= 0:
                    continue
                take = min(tx_take, avail)
                _, _lt, _st = p.taxable.sell_for(take, a)
                gross += take
                ltcg_add += _lt
                ord_add += _st
                tx_take -= take
            # Traditional
            tr_take = min(remaining * shares["trad"], p.traditional.value())
            p.traditional.withdraw(tr_take)
            gross += tr_take
            ord_add += tr_take
            # Roth
            ro_take = min(remaining * shares["roth"], p.roth.value())
            p.roth.withdraw(ro_take)
            gross += ro_take
            return _WithdrawalResult(gross, ord_add, ltcg_add)

    # Tax-aware (default) and ordered
    # Step A: taxable
    for a in (Asset.STOCK, Asset.BOND, Asset.CASH):
        if remaining <= 0:
            break
        avail = p.taxable.value(a)
        if avail <= 0:
            continue
        take = min(remaining, avail)
        _, _lt, _st = p.taxable.sell_for(take, a)
        gross += take
        ltcg_add += _lt
        ord_add += _st
        remaining -= take

    # Step B: if 59.5+, traditional next
    if remaining > 0 and age >= 59.5 and p.traditional.value() > 0:
        take = min(remaining, p.traditional.value())
        p.traditional.withdraw(take)
        gross += take
        ord_add += take
        remaining -= take

    # Step C: pre-59.5 fallback — Roth basis, then traditional w/ penalty
    if remaining > 0 and age < 59.5 and p.roth.roth_basis > 0:
        take = min(remaining, p.roth.roth_basis)
        p.roth.withdraw(take)
        gross += take
        remaining -= take
    if remaining > 0 and age < 59.5 and p.traditional.value() > 0:
        take = min(remaining, p.traditional.value())
        p.traditional.withdraw(take)
        gross += take
        ord_add += take + 0.10 * take  # crude 10% penalty as extra ordinary
        remaining -= take

    # Step D: Roth (post-59.5 it's tax-free; pre-59.5 of earnings is taxed
    # — we approximate as ordinary)
    if remaining > 0 and p.roth.value() > 0:
        take = min(remaining, p.roth.value())
        p.roth.withdraw(take)
        gross += take
        if age < 59.5:
            ord_add += take + 0.10 * take
        remaining -= take

    return _WithdrawalResult(gross, ord_add, ltcg_add)


# ---------- top-level driver ----------

def simulate(scn: Scenario,
             allocations: TargetAllocations | None = None,
             ) -> SimResult:
    """Run Monte Carlo. Returns a SimResult."""
    if allocations is not None:
        # local override (used by optimizer)
        scn = _replace_allocations(scn, allocations)

    horizon = scn.profile.horizon()
    n_paths = scn.simulation.n_paths
    seed = scn.simulation.seed
    market = scn.market.to_market_model()

    returns = sample_gbm_paths(market, horizon, n_paths, seed=seed)
    inflation = sample_inflation(market, horizon, n_paths, seed=(seed or 0) + 1)

    paths: list[PathResult] = []
    for i in range(n_paths):
        p = deepcopy(scn.initial_portfolio)
        state = _AccumState(nominal_income=scn.income.current_gross,
                            cumulative_inflation=1.0)
        wealth = np.empty(horizon + 1)
        wealth[0] = p.total_value()
        spend = np.zeros(horizon)
        short = np.zeros(horizon)
        total_real_tax = 0.0
        failed = False
        prior_trad_end = p.traditional.value()

        for y in range(horizon):
            age = scn.profile.age + y
            ret_y = {Asset.STOCK: returns[Asset.STOCK][i, y],
                     Asset.BOND:  returns[Asset.BOND][i, y],
                     Asset.CASH:  returns[Asset.CASH][i, y]}
            infl_y = inflation[i, y]
            if age < scn.profile.retirement_age:
                rw, rt = _step_accumulation(scn, p, age, y, ret_y, infl_y, state)
                spend[y] = 0.0
            else:
                rw, rt, ts, sh = _step_decumulation(
                    scn, p, age, y, ret_y, infl_y, state, prior_trad_end,
                )
                spend[y] = ts
                short[y] = sh
                if p.total_value() <= 0:
                    failed = True
            total_real_tax += rt
            wealth[y + 1] = p.total_value() / state.cumulative_inflation
            prior_trad_end = p.traditional.value()
            if failed:
                # zero out remaining
                wealth[y + 1:] = 0.0
                short[y + 1:] = scn.spending.annual_real
                break

        paths.append(PathResult(
            terminal_real_wealth=wealth[-1],
            real_wealth_by_year=wealth,
            real_spending_by_year=spend,
            real_shortfall_by_year=short,
            lifetime_real_tax=total_real_tax,
            failed=failed,
        ))
    return SimResult(paths=paths)


def _replace_allocations(scn: Scenario, allocations: TargetAllocations) -> Scenario:
    """Return a copy of scn with replaced target_allocations."""
    new = deepcopy(scn)
    new.target_allocations = allocations
    return new
