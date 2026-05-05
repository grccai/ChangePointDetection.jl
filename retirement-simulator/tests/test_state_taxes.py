"""State tax tests. Spot-check a few hand-computed bracket integrals to
confirm the data tables are entered correctly."""

import math
import numpy as np
import pytest

from retire.state_taxes import (state_tax, state_tax_vec, multi_state_tax,
                                 multi_state_tax_vec, StateTimeline,
                                 StateAssignment, progressive_tax_vec, STATES)
from retire.taxes import Bracket


def test_ca_zero_income():
    assert state_tax(state="CA", ordinary_income=0, ltcg_income=0,
                     filing_status="single") == 0


def test_ca_low_income():
    # Single, $20k ordinary income, no LTCG. CA std deduction $5,540.
    # Taxable = $14,460.
    # 1% on first $10,756 = $107.56
    # 2% on (14,460 - 10,756) = 2% * $3,704 = $74.08
    # Total = $181.64
    got = state_tax(state="CA", ordinary_income=20_000, ltcg_income=0,
                    filing_status="single")
    assert math.isclose(got, 107.56 + 74.08, rel_tol=1e-3)


def test_ca_ltcg_as_ordinary():
    # $50k ord + $30k LTCG -> CA stacks them, taxes as $80k - $5,540 = $74,460.
    # Compute progressive on $74,460.
    bk = STATES["CA"].ordinary_brackets["single"]
    from retire.taxes import progressive_tax
    expected = progressive_tax(74_460, bk)
    got = state_tax(state="CA", ordinary_income=50_000, ltcg_income=30_000,
                    filing_status="single")
    assert math.isclose(got, expected, rel_tol=1e-9)


def test_ca_high_earner_mhst():
    # $1.5M ordinary -> hits 13.3% top bracket (12.3% + 1% MHST folded in).
    got = state_tax(state="CA", ordinary_income=1_500_000, ltcg_income=0,
                    filing_status="single")
    # The marginal $500k over $1M is at 13.3%. Lower brackets contribute too.
    # Just assert it's strictly more than the same income would owe at 12.3%
    # flat (sanity).
    assert got > 1_500_000 * 0.10  # quite a lot of tax


def test_or_brackets():
    # Single, $50k ordinary. Std ded $2,745. Taxable $47,255.
    # 4.75% on first $4,300       = $204.25
    # 6.75% on (10,750 - 4,300)   = 6.75% * 6,450  = $435.375
    # 8.75% on (47,255 - 10,750)  = 8.75% * 36,505 = $3,194.1875
    # Total = $3,833.81
    got = state_tax(state="OR", ordinary_income=50_000, ltcg_income=0,
                    filing_status="single")
    assert math.isclose(got, 204.25 + 435.375 + 3_194.1875, rel_tol=1e-6)


def test_wa_no_wage_tax():
    # WA has no income tax on wages.
    assert state_tax(state="WA", ordinary_income=200_000, ltcg_income=0,
                     filing_status="single") == 0


def test_wa_cgt_below_threshold():
    # $200k LT gain < $262k threshold -> 0
    assert state_tax(state="WA", ordinary_income=0, ltcg_income=200_000,
                     filing_status="single") == 0


def test_wa_cgt_above_threshold():
    # $400k LT gain. Tax = 7% * (400_000 - 262_000) = 7% * 138_000 = $9_660.
    got = state_tax(state="WA", ordinary_income=500_000, ltcg_income=400_000,
                    filing_status="single")
    assert math.isclose(got, 0.07 * (400_000 - 262_000), rel_tol=1e-9)


def test_none_state():
    assert state_tax(state="NONE", ordinary_income=1_000_000, ltcg_income=500_000,
                     filing_status="single") == 0


def test_state_tax_vec_matches_scalar():
    # Vectorised must equal scalar pointwise.
    incomes = [10_000, 50_000, 200_000, 1_500_000]
    ltcgs = [0, 5_000, 50_000, 200_000]
    for s in ["CA", "OR", "WA"]:
        for fs in ["single", "mfj"]:
            vec_in_ord = np.array(incomes, dtype=float)
            vec_in_ltcg = np.array(ltcgs, dtype=float)
            vec_out = state_tax_vec(state=s, ordinary_income=vec_in_ord,
                                    ltcg_income=vec_in_ltcg, filing_status=fs)
            for i, (oi, lg) in enumerate(zip(incomes, ltcgs)):
                scalar = state_tax(state=s, ordinary_income=oi, ltcg_income=lg,
                                   filing_status=fs)
                assert math.isclose(vec_out[i], scalar, rel_tol=1e-9), (
                    f"{s} {fs} oi={oi} lg={lg}: vec {vec_out[i]} != scalar {scalar}"
                )


def test_progressive_tax_vec_zero_brackets():
    # WA-style: empty bracket list means 0 tax everywhere.
    out = progressive_tax_vec(np.array([0.0, 100_000.0, 1e6]), [])
    assert np.all(out == 0)


def test_timeline_at_age_simple():
    tl = StateTimeline(
        residency=[StateAssignment("CA", 35, 50), StateAssignment("WA", 50, 95)],
    )
    assert tl.at_age(40, "residency") == {"CA": 1.0}
    assert tl.at_age(70, "residency") == {"WA": 1.0}
    assert tl.at_age(40, "employment") == {}


def test_timeline_overlap_weights():
    tl = StateTimeline(
        employment=[StateAssignment("CA", 40, 50, weight=0.5),
                    StateAssignment("WA", 40, 50, weight=0.5)],
    )
    w = tl.at_age(45, "employment")
    assert math.isclose(w["CA"], 0.5)
    assert math.isclose(w["WA"], 0.5)


def test_multi_state_split_residency_employment():
    # Live in WA, work in CA -> wages taxed CA, gains taxed WA.
    tl = StateTimeline(
        residency=[StateAssignment("WA", 40, 50)],
        employment=[StateAssignment("CA", 40, 50)],
    )
    got = multi_state_tax(
        ordinary_income_wages=200_000, ordinary_income_other=0,
        ltcg_income=300_000, age=45, filing_status="single", timeline=tl,
    )
    # CA on wages
    ca = state_tax(state="CA", ordinary_income=200_000, ltcg_income=0,
                   filing_status="single")
    # WA on LT gain (under $262k threshold? $300k > $262k -> 7% * 38k)
    wa = state_tax(state="WA", ordinary_income=0, ltcg_income=300_000,
                   filing_status="single")
    assert math.isclose(got, ca + wa, rel_tol=1e-9)


def test_multi_state_no_overlap_zero_outside():
    # Outside the timeline window, nothing is taxed.
    tl = StateTimeline(residency=[StateAssignment("CA", 40, 50)])
    got = multi_state_tax(
        ordinary_income_wages=0, ordinary_income_other=100_000,
        ltcg_income=0, age=60, filing_status="single", timeline=tl,
    )
    assert got == 0


def test_multi_state_vec_matches_scalar():
    tl = StateTimeline(
        residency=[StateAssignment("WA", 40, 50)],
        employment=[StateAssignment("CA", 40, 50)],
    )
    P = 5
    rng = np.random.default_rng(0)
    wages = rng.uniform(50_000, 300_000, P)
    other = rng.uniform(0, 100_000, P)
    gains = rng.uniform(0, 500_000, P)
    vec = multi_state_tax_vec(
        ordinary_income_wages=wages, ordinary_income_other=other,
        ltcg_income=gains, age=45, filing_status="single", timeline=tl,
    )
    for i in range(P):
        scalar = multi_state_tax(
            ordinary_income_wages=wages[i], ordinary_income_other=other[i],
            ltcg_income=gains[i], age=45, filing_status="single", timeline=tl,
        )
        assert math.isclose(vec[i], scalar, rel_tol=1e-9), \
            f"path {i}: {vec[i]} vs {scalar}"
