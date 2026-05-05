"""Integration tests for the vectorised simulator (date-based config).

Checks invariants and consistency:
  * Wealth is non-negative.
  * Higher spending => more failures (monotonicity).
  * The 5-year clock prevents penalty-free withdrawal of fresh conversions.
  * Mega-backdoor Roth contributes more than zero when enabled.
  * State tax matters: CA-resident-during-FIRE worse than WA-resident.
"""

import datetime as dt
import math
import numpy as np
import pytest

from retire.accounts import Asset, Lot, Portfolio
from retire.config import (
    Profile, Savings, Contributions, Spending,
    SocialSecurity, WithdrawalPolicy, Allocation, TargetAllocations,
    MarketConfig, SimulationParams, Scenario,
)
from retire.returns import AssetParams
from retire.simulate import simulate
from retire.state_taxes import (StateTimeline, ResidencyPeriod, IncomeSource)
from retire.vstate import VState


D = dt.date


def _basic_scenario(spending: float = 60_000,
                    end_date: dt.date = D(2066, 1, 1),  # ~age 80
                    state_timeline: StateTimeline | None = None,
                    contributions: Contributions | None = None,
                    n_paths: int = 200) -> Scenario:
    p = Portfolio()
    p.taxable.lots.append(
        Lot(asset=Asset.STOCK, market_value=200_000,
            cost_basis=120_000, age_years=2.0))
    p.traditional.balances[Asset.STOCK] = 200_000
    p.traditional.balances[Asset.BOND] = 100_000
    p.roth.balances[Asset.STOCK] = 100_000
    p.roth.roth_basis = 50_000

    profile = Profile(
        birthdate=D(1986, 1, 1), start_date=D(2026, 1, 1),
        retirement_date=D(2041, 1, 1), end_of_plan_date=end_date,
        filing_status="single",
    )

    if state_timeline is None:
        # Default: single income source 2026-2041 in NONE (no state tax).
        state_timeline = StateTimeline(
            income_sources=[IncomeSource(
                state="NONE", start=D(2026, 1, 1), end=D(2041, 1, 1),
                gross_annual=200_000, growth_rate=0.03)],
        )

    return Scenario(
        profile=profile, state_taxes=state_timeline,
        savings=Savings(rate=0.30,
                        contributions=contributions or Contributions()),
        spending=Spending(annual_real=spending, smile="flat"),
        initial_portfolio=p,
        target_allocations=TargetAllocations(
            taxable=Allocation(0.7, 0.2, 0.1),
            traditional=Allocation(0.4, 0.6, 0.0),
            roth=Allocation(1.0, 0.0, 0.0),
        ),
        market=MarketConfig(
            stocks=AssetParams(0.06, 0.18),
            bonds=AssetParams(0.02, 0.06),
            cash=AssetParams(0.005, 0.01),
        ),
        social_security=SocialSecurity(monthly_at_67=0, claim_age=67),
        withdrawal=WithdrawalPolicy(strategy="tax_aware",
                                     roth_conversion_target_bracket=None),
        simulation=SimulationParams(n_paths=n_paths, seed=1),
    )


def test_wealth_non_negative():
    r = simulate(_basic_scenario(spending=80_000))
    for p in r.paths:
        assert (p.real_wealth_by_year >= -1e-3).all()


def test_failure_monotone_in_spending():
    r1 = simulate(_basic_scenario(spending=40_000))
    r2 = simulate(_basic_scenario(spending=80_000))
    assert r2.failure_rate() >= r1.failure_rate()


def test_state_tax_changes_outcome():
    """CA-resident-during-FIRE vs WA-resident: WA preserves more wealth."""
    ca = StateTimeline(
        residency=[ResidencyPeriod("CA", D(2026, 1, 1), D(2070, 1, 1))],
        income_sources=[IncomeSource(
            state="CA", start=D(2026, 1, 1), end=D(2041, 1, 1),
            gross_annual=200_000, growth_rate=0.03)],
    )
    wa = StateTimeline(
        residency=[ResidencyPeriod("WA", D(2026, 1, 1), D(2070, 1, 1))],
        income_sources=[IncomeSource(
            state="WA", start=D(2026, 1, 1), end=D(2041, 1, 1),
            gross_annual=200_000, growth_rate=0.03)],
    )
    r_ca = simulate(_basic_scenario(spending=40_000, state_timeline=ca))
    r_wa = simulate(_basic_scenario(spending=40_000, state_timeline=wa))
    # Compare wealth at retirement (year 15 = age 55).
    w_ca = np.mean([p.real_wealth_by_year[15] for p in r_ca.paths])
    w_wa = np.mean([p.real_wealth_by_year[15] for p in r_wa.paths])
    assert w_wa > w_ca, f"WA ({w_wa:,.0f}) should beat CA ({w_ca:,.0f})"


def test_mid_year_state_move():
    """Move from CA to WA on July 1 of a year: tax falls partway between
    full-CA-year and full-WA-year."""
    full_ca = StateTimeline(
        residency=[ResidencyPeriod("CA", D(2026, 1, 1), D(2070, 1, 1))],
        income_sources=[IncomeSource(
            state="CA", start=D(2026, 1, 1), end=D(2041, 1, 1),
            gross_annual=200_000, growth_rate=0.03)],
    )
    full_wa = StateTimeline(
        residency=[ResidencyPeriod("WA", D(2026, 1, 1), D(2070, 1, 1))],
        income_sources=[IncomeSource(
            state="WA", start=D(2026, 1, 1), end=D(2041, 1, 1),
            gross_annual=200_000, growth_rate=0.03)],
    )
    mid = StateTimeline(
        residency=[
            ResidencyPeriod("CA", D(2026, 1, 1), D(2030, 7, 1)),
            ResidencyPeriod("WA", D(2030, 7, 1), D(2070, 1, 1)),
        ],
        income_sources=[
            IncomeSource("CA", D(2026, 1, 1), D(2030, 7, 1),
                         gross_annual=200_000, growth_rate=0.03),
            IncomeSource("WA", D(2030, 7, 1), D(2041, 1, 1),
                         gross_annual=200_000, growth_rate=0.03),
        ],
    )
    r_ca = simulate(_basic_scenario(spending=40_000, state_timeline=full_ca))
    r_wa = simulate(_basic_scenario(spending=40_000, state_timeline=full_wa))
    r_mid = simulate(_basic_scenario(spending=40_000, state_timeline=mid))
    w_ca = np.mean([p.real_wealth_by_year[15] for p in r_ca.paths])
    w_wa = np.mean([p.real_wealth_by_year[15] for p in r_wa.paths])
    w_mid = np.mean([p.real_wealth_by_year[15] for p in r_mid.paths])
    assert w_ca <= w_mid <= w_wa, (
        f"mid-year move ({w_mid:,.0f}) should sit between full-CA "
        f"({w_ca:,.0f}) and full-WA ({w_wa:,.0f})")


def test_multiple_income_sources():
    """Two concurrent jobs in different states: wages stack, both get taxed."""
    tl = StateTimeline(
        residency=[ResidencyPeriod("CA", D(2026, 1, 1), D(2070, 1, 1))],
        income_sources=[
            IncomeSource("CA", D(2026, 1, 1), D(2041, 1, 1),
                         gross_annual=100_000, growth_rate=0.03),
            IncomeSource("OR", D(2026, 1, 1), D(2030, 1, 1),
                         gross_annual=50_000, growth_rate=0.0),
        ],
    )
    # Just confirm it runs and produces some wealth growth.
    r = simulate(_basic_scenario(spending=40_000, state_timeline=tl))
    assert r.paths[0].real_wealth_by_year[5] > r.paths[0].real_wealth_by_year[0]


def test_mega_backdoor_increases_roth():
    no_mbdr = _basic_scenario(spending=40_000, n_paths=50,
                               contributions=Contributions(
                                   trad_401k=23_000, mega_backdoor_roth=0))
    with_mbdr = _basic_scenario(spending=40_000, n_paths=50,
                                 contributions=Contributions(
                                     trad_401k=23_000,
                                     mega_backdoor_roth=20_000))
    r_no = simulate(no_mbdr)
    r_yes = simulate(with_mbdr)
    w_no = np.mean([p.real_wealth_by_year[15] for p in r_no.paths])
    w_yes = np.mean([p.real_wealth_by_year[15] for p in r_yes.paths])
    assert w_yes > w_no


def test_5y_clock_protects_fresh_conversions():
    from retire.vstate import withdraw_roth
    s = VState.from_portfolio(_basic_scenario().initial_portfolio,
                               n_paths=2, horizon=20,
                               starting_nominal_income=200_000)
    s.roth_conversions[:, 3] = 50_000.0
    s.roth_balance[:, 0] += 50_000.0
    s.roth_basis[:] = 0.0
    proc, ord_add, pen = withdraw_roth(s, np.array([10_000.0, 10_000.0]),
                                        year_idx=5, age=45)
    assert (pen > 0).all()
    assert math.isclose(pen[0], 1_000.0, rel_tol=1e-9)


def test_5y_clock_mature_conversion_no_penalty():
    from retire.vstate import withdraw_roth
    s = VState.from_portfolio(_basic_scenario().initial_portfolio,
                               n_paths=2, horizon=20,
                               starting_nominal_income=200_000)
    s.roth_conversions[:, 0] = 50_000.0
    s.roth_balance[:, 0] += 50_000.0
    s.roth_basis[:] = 0.0
    proc, ord_add, pen = withdraw_roth(s, np.array([10_000.0, 10_000.0]),
                                        year_idx=6, age=45)
    assert (pen == 0).all()


def test_withdraw_roth_basis_first_no_penalty():
    from retire.vstate import withdraw_roth
    s = VState.from_portfolio(_basic_scenario().initial_portfolio,
                               n_paths=1, horizon=20,
                               starting_nominal_income=200_000)
    proc, ord_add, pen = withdraw_roth(s, np.array([30_000.0]),
                                        year_idx=2, age=45)
    assert math.isclose(proc[0], 30_000.0, rel_tol=1e-9)
    assert pen[0] == 0.0
    assert ord_add[0] == 0.0
