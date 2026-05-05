"""State tax tests. Spot-check the bracket data and the date-based
multi-state apportionment."""

import datetime as dt
import math
import numpy as np
import pytest

from retire.state_taxes import (
    state_tax, state_tax_vec, state_wages_tax, state_residency_tax_vec,
    progressive_tax_vec, STATES, StateTimeline, ResidencyPeriod, IncomeSource,
)
from retire.taxes import Bracket, progressive_tax


# ---------- Per-state bracket math ----------

def test_ca_zero_income():
    assert state_tax(state="CA", ordinary_income=0, ltcg_income=0,
                     filing_status="single") == 0


def test_ca_low_income():
    # Single, $20k ordinary, no LTCG. Std deduction $5,540.
    # Taxable = $14,460.
    # 1% on first $10,756 = $107.56; 2% on remaining $3,704 = $74.08.
    got = state_tax(state="CA", ordinary_income=20_000, ltcg_income=0,
                    filing_status="single")
    assert math.isclose(got, 107.56 + 74.08, rel_tol=1e-3)


def test_ca_ltcg_as_ordinary():
    bk = STATES["CA"].ordinary_brackets["single"]
    expected = progressive_tax(50_000 + 30_000 - 5_540, bk)
    got = state_tax(state="CA", ordinary_income=50_000, ltcg_income=30_000,
                    filing_status="single")
    assert math.isclose(got, expected, rel_tol=1e-9)


def test_or_brackets():
    # Single, $50k. Std ded $2,745. Taxable $47,255.
    # 4.75% on first $4,300 + 6.75% on (10,750 - 4,300) + 8.75% on (47,255 - 10,750)
    expected = 4_300 * 0.0475 + 6_450 * 0.0675 + 36_505 * 0.0875
    got = state_tax(state="OR", ordinary_income=50_000, ltcg_income=0,
                    filing_status="single")
    assert math.isclose(got, expected, rel_tol=1e-6)


def test_wa_no_wage_tax():
    assert state_tax(state="WA", ordinary_income=200_000, ltcg_income=0,
                     filing_status="single") == 0


def test_wa_cgt_above_threshold():
    got = state_tax(state="WA", ordinary_income=500_000, ltcg_income=400_000,
                    filing_status="single")
    assert math.isclose(got, 0.07 * (400_000 - 262_000), rel_tol=1e-9)


def test_state_tax_vec_matches_scalar():
    incomes = [10_000, 50_000, 200_000, 1_500_000]
    ltcgs = [0, 5_000, 50_000, 200_000]
    for s in ["CA", "OR", "WA"]:
        vec = state_tax_vec(state=s,
                            ordinary_income=np.array(incomes, dtype=float),
                            ltcg_income=np.array(ltcgs, dtype=float),
                            filing_status="single")
        for i, (oi, lg) in enumerate(zip(incomes, ltcgs)):
            scalar = state_tax(state=s, ordinary_income=oi, ltcg_income=lg,
                               filing_status="single")
            assert math.isclose(vec[i], scalar, rel_tol=1e-9)


# ---------- Date-based timeline ----------

D = dt.date


def test_residency_full_year():
    tl = StateTimeline(residency=[
        ResidencyPeriod("CA", D(2026, 1, 1), D(2030, 1, 1)),
    ])
    w = tl.residency_weights(D(2026, 1, 1), 0)
    assert w == {"CA": pytest.approx(1.0, abs=1e-3)}


def test_residency_mid_year_move():
    # Move from CA to WA on July 1, 2030 — sim year 2030 should split ~50/50.
    sim_start = D(2026, 1, 1)
    tl = StateTimeline(residency=[
        ResidencyPeriod("CA", D(2026, 1, 1), D(2030, 7, 1)),
        ResidencyPeriod("WA", D(2030, 7, 1), D(2050, 1, 1)),
    ])
    w = tl.residency_weights(sim_start, year_idx=4)  # year window 2030-01-01..2031-01-01
    # Days in year 2030 to July 1 = 181; remaining = 184. Total = 365.
    assert math.isclose(w["CA"], 181 / 365, abs_tol=0.01)
    assert math.isclose(w["WA"], 184 / 365, abs_tol=0.01)
    # Sum should be ~1.0
    assert math.isclose(w["CA"] + w["WA"], 1.0, abs_tol=0.01)


def test_residency_gap_no_state_tax():
    # Period not covered by any residency entry contributes no state tax.
    sim_start = D(2026, 1, 1)
    tl = StateTimeline(residency=[
        ResidencyPeriod("CA", D(2026, 1, 1), D(2027, 1, 1)),
        # No coverage 2027-2028
        ResidencyPeriod("WA", D(2028, 1, 1), D(2030, 1, 1)),
    ])
    w = tl.residency_weights(sim_start, year_idx=1)
    assert sum(w.values()) == pytest.approx(0.0, abs=1e-6)


def test_wages_by_state_basic():
    sim_start = D(2026, 1, 1)
    tl = StateTimeline(income_sources=[
        IncomeSource("CA", D(2026, 1, 1), D(2031, 1, 1),
                     gross_annual=100_000, growth_rate=0.0),
    ])
    out = tl.wages_by_state(sim_start, year_idx=2)  # year 2028
    assert math.isclose(out["CA"], 100_000, rel_tol=0.01)


def test_wages_grow_with_source_growth():
    sim_start = D(2026, 1, 1)
    tl = StateTimeline(income_sources=[
        IncomeSource("CA", D(2026, 1, 1), D(2031, 1, 1),
                     gross_annual=100_000, growth_rate=0.05),
    ])
    # Year 2 (2028): grown from start by ~2.5 years to mid-overlap.
    out = tl.wages_by_state(sim_start, year_idx=2)
    expected_min = 100_000 * 1.05**2.0  # ~110,250
    expected_max = 100_000 * 1.05**3.0  # ~115,762
    assert expected_min < out["CA"] < expected_max


def test_wages_two_concurrent_sources():
    sim_start = D(2026, 1, 1)
    tl = StateTimeline(income_sources=[
        IncomeSource("CA", D(2026, 1, 1), D(2030, 1, 1), gross_annual=100_000),
        IncomeSource("WA", D(2026, 1, 1), D(2030, 1, 1), gross_annual=80_000),
    ])
    out = tl.wages_by_state(sim_start, year_idx=1)
    assert math.isclose(out["CA"], 100_000, rel_tol=0.01)
    assert math.isclose(out["WA"],  80_000, rel_tol=0.01)
    assert math.isclose(tl.total_wages(sim_start, 1), 180_000, rel_tol=0.01)


def test_wages_partial_year():
    """Source ends mid-year; wages prorate."""
    sim_start = D(2026, 1, 1)
    tl = StateTimeline(income_sources=[
        IncomeSource("CA", D(2026, 1, 1), D(2026, 7, 1),
                     gross_annual=100_000),
    ])
    out = tl.wages_by_state(sim_start, year_idx=0)
    # Half-year overlap -> ~$50k
    assert 49_000 < out["CA"] < 51_000


def test_state_wages_tax_apportions_pretax():
    # $200k CA wages, $20k pretax 401k. Apportion 100% to CA.
    by_state = {"CA": 200_000.0}
    pretax = 20_000.0
    got = state_wages_tax(wages_by_state=by_state, pretax_401k=pretax,
                          filing_status="single")
    expected = state_tax(state="CA", ordinary_income=180_000, ltcg_income=0,
                         filing_status="single")
    assert math.isclose(got, expected, rel_tol=1e-9)


def test_state_wages_tax_split():
    # $100k CA + $100k WA. WA has no wage tax. Pretax $10k.
    by_state = {"CA": 100_000.0, "WA": 100_000.0}
    pretax = 10_000.0
    got = state_wages_tax(wages_by_state=by_state, pretax_401k=pretax,
                          filing_status="single")
    # CA gets half pretax = $5k. CA taxable wages = $95k. WA contributes $0.
    expected = state_tax(state="CA", ordinary_income=95_000, ltcg_income=0,
                         filing_status="single")
    assert math.isclose(got, expected, rel_tol=1e-9)


def test_residency_tax_split():
    # 50/50 between CA and WA on $50k of ordinary investment income +
    # $300k LTCG.
    P = 4
    ord_other = np.full(P, 50_000.0)
    ltcg = np.full(P, 300_000.0)
    res_w = {"CA": 0.5, "WA": 0.5}
    out = state_residency_tax_vec(residency_weights=res_w,
                                   ordinary_other=ord_other, ltcg=ltcg,
                                   filing_status="single")
    expected = 0.5 * state_tax(state="CA", ordinary_income=50_000,
                                ltcg_income=300_000, filing_status="single") \
             + 0.5 * state_tax(state="WA", ordinary_income=50_000,
                                ltcg_income=300_000, filing_status="single")
    for v in out:
        assert math.isclose(v, expected, rel_tol=1e-9)
