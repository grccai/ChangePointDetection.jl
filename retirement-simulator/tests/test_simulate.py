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


def test_flexible_spending_floor_enforced():
    """When portfolio crashes, spending should be cut but not below floor."""
    from retire.config import FlexibleSpending
    scn = _basic_scenario(spending=80_000, n_paths=200)
    scn.spending.flexible = FlexibleSpending(floor_real=40_000, sensitivity=1.0)
    # Crank up volatility so the drawdown distribution sweeps the floor.
    scn.market.stocks.vol = 0.40
    r = simulate(scn)
    # Across all paths and retirement years, target spend must be >= floor.
    for p in r.paths:
        # Retirement year window starts at year 15 in the basic scenario
        # (age 40 birthdate-anchored, retirement_date=2041, end 2066 -> H=40).
        ret_yrs = p.real_spending_by_year[15:]
        assert (ret_yrs >= 40_000 - 1e-3).all() or (ret_yrs == 0).all(), \
            f"floor breach: min={ret_yrs.min()}"
        # And target must be <= base (downside-only)
        assert (ret_yrs <= 80_000 + 1e-3).all()


def test_flexible_spending_scales_proportionally():
    """When ratio < 1, target = base * ratio (within floor)."""
    from retire.config import FlexibleSpending
    scn = _basic_scenario(spending=100_000, n_paths=50)
    scn.spending.flexible = FlexibleSpending(floor_real=50_000, sensitivity=1.0)
    # Force a market crash by setting low real returns
    scn.market.stocks.real_return = -0.05
    scn.market.bonds.real_return = -0.02
    scn.market.stocks.vol = 1e-9
    scn.market.bonds.vol = 1e-9
    scn.market.cash.vol = 1e-9
    r = simulate(scn)
    p = r.paths[0]
    # By year 5+ of retirement, portfolio is way down -> spending should be
    # below baseline ($100k) and approaching floor ($50k).
    late_target = p.real_spending_by_year[20]  # year 5 of retirement
    assert late_target < 100_000
    assert late_target >= 50_000


def test_inheritance_increases_wealth_at_date():
    """A $1M real inheritance at age 60 should bump real wealth by ~$1M
    in the year it lands (with deterministic returns)."""
    from retire.config import Inheritance
    scn = _basic_scenario(spending=40_000, n_paths=1)
    # Deterministic
    scn.market.stocks.vol = 1e-9
    scn.market.bonds.vol = 1e-9
    scn.market.cash.vol = 1e-9
    scn.market.inflation_vol = 0.0
    # Birthdate 1986-01-01, sim_start 2026-01-01 -> age 60 = 2046-01-01
    scn.inheritances = [Inheritance(date=dt.date(2046, 1, 1),
                                     amount_real=1_000_000,
                                     account="taxable")]
    r = simulate(scn)
    p = r.paths[0]
    # The inheritance lands at the start of year-loop y=20 (window
    # [2046-01-01, 2047-01-01)) and is reflected in real_wealth_by_year[21]
    # (end of that year). Compare to [20] (start of that year, before deposit).
    pre = p.real_wealth_by_year[20]
    post = p.real_wealth_by_year[21]
    bump = post - pre
    # Expect ~$1M of new real wealth, plus ~year of returns on it, minus
    # the year's $40k retirement spending. Loose bracket.
    assert 700_000 < bump < 1_200_000, f"bump {bump} not in expected range"


def test_no_inheritance_default():
    """Empty inheritances list = simulator runs unchanged."""
    scn = _basic_scenario(spending=40_000, n_paths=50)
    assert scn.inheritances == []
    r = simulate(scn)
    # Just check it runs
    assert len(r.paths) == 50


def test_no_flexible_spending_unchanged():
    """When .flexible is None, spending equals smile-adjusted baseline."""
    scn = _basic_scenario(spending=100_000, n_paths=100)
    scn.spending.flexible = None
    r = simulate(scn)
    # Retirement first year (age 55, year 15) target = $100k flat smile
    for p in r.paths:
        assert math.isclose(p.real_spending_by_year[15], 100_000, rel_tol=1e-9)


def test_glide_policy_degenerates_to_static():
    """Glide policy with both knots equal == static policy: simulating with
    each gives identical real wealth at every year."""
    from retire.policy import (StaticPolicy, GlidePolicy, GlidePath,
                                AccountGlide)
    static_alloc = TargetAllocations(
        taxable=Allocation(0.7, 0.2, 0.1),
        traditional=Allocation(0.4, 0.6, 0.0),
        roth=Allocation(1.0, 0.0, 0.0),
    )
    static_pol = StaticPolicy(allocations=static_alloc,
                              conversion_bracket=None,
                              trad_contribution_split=1.0)
    flat = lambda v: GlidePath([(0.0, v), (200.0, v)])
    glide_pol = GlidePolicy(
        taxable=AccountGlide(flat(0.7), flat(0.2)),
        traditional=AccountGlide(flat(0.4), flat(0.6)),
        roth=AccountGlide(flat(1.0), flat(0.0)),
        conv_during_fire_gap=None, conv_during_ss_window=None,
        trad_contribution_split=1.0, wealth_responsiveness=0.0,
        retirement_age=55.0, ss_age=67.0, rmd_age=73.0,
    )
    scn = _basic_scenario(spending=40_000)
    r_static = simulate(scn, policy=static_pol)
    r_glide = simulate(scn, policy=glide_pol)
    # Identical seed -> identical paths
    for ps, pg in zip(r_static.paths, r_glide.paths):
        assert np.allclose(ps.real_wealth_by_year, pg.real_wealth_by_year,
                           rtol=1e-9, atol=1e-3)


def test_glide_policy_changes_outcome_when_knots_differ():
    """Glide path that de-risks aggressively into retirement should give
    different outcomes from a flat 100% stock policy."""
    from retire.policy import GlidePolicy, GlidePath, AccountGlide
    flat_stock = lambda: AccountGlide(
        GlidePath([(0.0, 1.0), (200.0, 1.0)]),
        GlidePath([(0.0, 0.0), (200.0, 0.0)]),
    )
    aggressive = GlidePolicy(
        taxable=flat_stock(), traditional=flat_stock(), roth=flat_stock(),
        conv_during_fire_gap=None, conv_during_ss_window=None,
        retirement_age=55.0,
    )
    conservative = GlidePolicy(
        taxable=AccountGlide(
            GlidePath([(40.0, 1.0), (80.0, 0.2)]),  # de-risk to 20% stock
            GlidePath([(40.0, 0.0), (80.0, 0.7)])),
        traditional=AccountGlide(
            GlidePath([(40.0, 1.0), (80.0, 0.2)]),
            GlidePath([(40.0, 0.0), (80.0, 0.7)])),
        roth=AccountGlide(
            GlidePath([(40.0, 1.0), (80.0, 0.2)]),
            GlidePath([(40.0, 0.0), (80.0, 0.7)])),
        conv_during_fire_gap=None, conv_during_ss_window=None,
        retirement_age=55.0,
    )
    scn = _basic_scenario(spending=40_000, n_paths=200)
    r_agg = simulate(scn, policy=aggressive)
    r_con = simulate(scn, policy=conservative)
    # Median terminal wealth should differ; we don't assert direction since
    # both can be optimal in different return regimes — just that the policy
    # actually changed something.
    q_agg = r_agg.terminal_quantiles([0.5])[0.5]
    q_con = r_con.terminal_quantiles([0.5])[0.5]
    assert q_agg != q_con


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
