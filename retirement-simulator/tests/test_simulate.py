"""Integration tests for the vectorised simulator.

These tests don't validate the financial model precisely (that's research-
grade work). They check invariants and consistency:
  * Wealth is non-negative.
  * Higher spending => more failures (monotonicity).
  * The 5-year clock prevents penalty-free withdrawal of fresh conversions.
  * Mega-backdoor Roth contributes more than zero when enabled.
"""

import math
import numpy as np
import pytest

from retire.accounts import Asset, Lot, Portfolio
from retire.config import (
    Profile, Income, Savings, Contributions, Spending,
    SocialSecurity, WithdrawalPolicy, Allocation, TargetAllocations,
    MarketConfig, SimulationParams, Scenario,
)
from retire.returns import AssetParams
from retire.simulate import simulate
from retire.state_taxes import StateTimeline, StateAssignment
from retire.vstate import VState


def _basic_scenario(spending: float = 60_000,
                    horizon_age: int = 80,
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

    return Scenario(
        profile=Profile(age=40, retirement_age=55,
                        end_of_plan_age=horizon_age, filing_status="single"),
        income=Income(current_gross=200_000, growth_rate=0.03),
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
        state_taxes=state_timeline or StateTimeline(),
        social_security=SocialSecurity(monthly_at_67=0, claim_age=67),
        withdrawal=WithdrawalPolicy(strategy="tax_aware",
                                     roth_conversion_target_bracket=None),
        simulation=SimulationParams(n_paths=n_paths, seed=1),
    )


def test_wealth_non_negative():
    r = simulate(_basic_scenario(spending=80_000))
    for p in r.paths:
        assert (p.real_wealth_by_year >= -1e-3).all(), \
            f"negative wealth in path: {p.real_wealth_by_year.min()}"


def test_failure_monotone_in_spending():
    # Higher spending -> at least as many failures.
    r1 = simulate(_basic_scenario(spending=40_000))
    r2 = simulate(_basic_scenario(spending=80_000))
    assert r2.failure_rate() >= r1.failure_rate()


def test_state_tax_changes_outcome():
    # Same scenario, CA vs WA residency. Use lower spending so paths
    # don't all deplete (collapsed medians don't differentiate).
    ca = StateTimeline(residency=[StateAssignment("CA", 40, 80)])
    wa = StateTimeline(residency=[StateAssignment("WA", 40, 80)])
    r_ca = simulate(_basic_scenario(spending=40_000, state_timeline=ca))
    r_wa = simulate(_basic_scenario(spending=40_000, state_timeline=wa))
    # Use mean of real wealth at retirement age (year 15 = age 55) as the
    # comparison: WA should accumulate more during working years (no
    # income tax on wages -> larger taxable savings).
    w_ca = np.mean([p.real_wealth_by_year[15] for p in r_ca.paths])
    w_wa = np.mean([p.real_wealth_by_year[15] for p in r_wa.paths])
    assert w_wa > w_ca, f"WA ({w_wa}) should beat CA ({w_ca}) at retirement"


def test_mega_backdoor_increases_roth():
    """With mega-backdoor enabled, the Roth balance at retirement age should
    be noticeably higher than without."""
    no_mbdr = _basic_scenario(spending=40_000, n_paths=50,
                               contributions=Contributions(
                                   trad_401k=23_000, mega_backdoor_roth=0))
    with_mbdr = _basic_scenario(spending=40_000, n_paths=50,
                                 contributions=Contributions(
                                     trad_401k=23_000,
                                     mega_backdoor_roth=20_000))
    r_no = simulate(no_mbdr)
    r_yes = simulate(with_mbdr)
    # Compare wealth at retirement (year 15 = age 55), before withdrawals
    # erode it.
    w_no = np.mean([p.real_wealth_by_year[15] for p in r_no.paths])
    w_yes = np.mean([p.real_wealth_by_year[15] for p in r_yes.paths])
    assert w_yes > w_no


def test_5y_clock_protects_fresh_conversions():
    """Pre-59.5 withdrawals from a Roth should hit the penalty if
    conversions are < 5 years old. We test the vstate.withdraw_roth helper
    directly because it's deterministic at that level."""
    from retire.vstate import withdraw_roth
    s = VState.from_portfolio(_basic_scenario().initial_portfolio,
                               n_paths=2, horizon=20,
                               starting_nominal_income=200_000)
    # Inject a conversion at year 3
    s.roth_conversions[:, 3] = 50_000.0
    s.roth_balance[:, 0] += 50_000.0  # increase balance accordingly
    s.roth_basis[:] = 0.0  # zero basis so withdrawal hits conversion
    # Withdraw at year 5 (only 2 years after conversion, age 45 -> early)
    proc, ord_add, pen = withdraw_roth(s, np.array([10_000.0, 10_000.0]),
                                        year_idx=5, age=45)
    # Pre-59.5 withdrawal of green conversion -> 10% penalty on principal.
    assert (pen > 0).all(), f"expected penalty, got {pen}"
    assert math.isclose(pen[0], 1_000.0, rel_tol=1e-9)


def test_5y_clock_mature_conversion_no_penalty():
    from retire.vstate import withdraw_roth
    s = VState.from_portfolio(_basic_scenario().initial_portfolio,
                               n_paths=2, horizon=20,
                               starting_nominal_income=200_000)
    s.roth_conversions[:, 0] = 50_000.0  # conversion at year 0
    s.roth_balance[:, 0] += 50_000.0
    s.roth_basis[:] = 0.0
    # Withdraw at year 6 (>= 5 years after) -> no penalty even pre-59.5
    proc, ord_add, pen = withdraw_roth(s, np.array([10_000.0, 10_000.0]),
                                        year_idx=6, age=45)
    assert (pen == 0).all()


def test_withdraw_roth_basis_first_no_penalty():
    """Direct contributions are always penalty-free regardless of clock."""
    from retire.vstate import withdraw_roth
    s = VState.from_portfolio(_basic_scenario().initial_portfolio,
                               n_paths=1, horizon=20,
                               starting_nominal_income=200_000)
    # Roth basis starts at 50k from the basic scenario; conversions at 0.
    # Withdraw 30k at age 45 (early) — should come from basis, no penalty.
    proc, ord_add, pen = withdraw_roth(s, np.array([30_000.0]),
                                        year_idx=2, age=45)
    assert math.isclose(proc[0], 30_000.0, rel_tol=1e-9)
    assert pen[0] == 0.0
    assert ord_add[0] == 0.0
