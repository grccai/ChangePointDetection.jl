"""Tests for the tax module — these are the load-bearing numbers; if these
are wrong, every dollar downstream is wrong."""

import math
import pytest

from retire.taxes import (
    TAX_2024, Bracket, progressive_tax, marginal_rate, ltcg_tax,
    taxable_social_security, niit_owed, compute_tax,
    required_min_distribution, top_of_bracket, RMD_START_AGE,
)


def test_progressive_tax_basic():
    # Single, $50,000 ordinary taxable income (already net of std deduction)
    # Brackets: 10% up to 11_600; 12% up to 47_150; 22% up to 100_525
    # Expected: 11_600*0.10 + (47_150-11_600)*0.12 + (50_000-47_150)*0.22
    expected = 1160.0 + 4266.0 + 627.0
    got = progressive_tax(50_000, TAX_2024.ordinary_brackets["single"])
    assert math.isclose(got, expected, rel_tol=1e-9)


def test_progressive_tax_zero():
    assert progressive_tax(0, TAX_2024.ordinary_brackets["single"]) == 0
    assert progressive_tax(-100, TAX_2024.ordinary_brackets["single"]) == 0


def test_marginal_rate():
    bk = TAX_2024.ordinary_brackets["single"]
    assert marginal_rate(0, bk) == 0.10
    assert marginal_rate(11_600, bk) == 0.12
    assert marginal_rate(11_599, bk) == 0.10
    assert marginal_rate(150_000, bk) == 0.24
    assert marginal_rate(700_000, bk) == 0.37


def test_ltcg_zero_bracket():
    # Single, no ordinary income, $40k LTCG -> entirely 0% bracket
    bk = TAX_2024.ltcg_brackets["single"]
    assert ltcg_tax(0, 40_000, bk) == 0


def test_ltcg_stacking():
    # Single, $50k ordinary taxable income + $20k LTCG.
    # 0% LTCG bracket ends at $47,025 single. Ordinary already exceeds it.
    # All $20k of LTCG is taxed at 15%.
    bk = TAX_2024.ltcg_brackets["single"]
    got = ltcg_tax(50_000, 20_000, bk)
    assert math.isclose(got, 20_000 * 0.15, rel_tol=1e-9)


def test_ltcg_partial_zero():
    # Single, $30k ordinary + $20k LTCG.
    # 0% bracket ends at 47_025. First (47_025-30_000)=17_025 of LTCG is 0%.
    # Remaining 2_975 is at 15%.
    bk = TAX_2024.ltcg_brackets["single"]
    got = ltcg_tax(30_000, 20_000, bk)
    assert math.isclose(got, 2_975 * 0.15, rel_tol=1e-9)


def test_ss_below_threshold():
    # Single. Below $25k provisional -> 0% SS taxable.
    assert taxable_social_security(20_000, 5_000, 0, "single", TAX_2024) == 0


def test_ss_above_upper_threshold():
    # Single. High income -> 85% of SS taxable.
    got = taxable_social_security(30_000, 100_000, 0, "single", TAX_2024)
    assert math.isclose(got, 0.85 * 30_000, rel_tol=1e-9)


def test_niit():
    # Single, MAGI=$300k (threshold $200k). NII=$50k. NIIT = 3.8% * min(50k, 100k)
    got = niit_owed(300_000, 50_000, "single", TAX_2024)
    assert math.isclose(got, 0.038 * 50_000, rel_tol=1e-9)


def test_niit_below_threshold():
    assert niit_owed(150_000, 50_000, "single", TAX_2024) == 0


def test_compute_tax_consistency():
    # Sanity: total >= sum of components
    bill = compute_tax(
        ordinary_income=120_000, ltcg_income=10_000, ss_benefit=0,
        tax_exempt_interest=0, filing_status="single",
        state_marginal_rate=0.05,
    )
    assert math.isclose(
        bill.total,
        bill.federal_ordinary + bill.federal_ltcg + bill.niit + bill.state,
        rel_tol=1e-9,
    )


def test_rmd_below_age():
    assert required_min_distribution(72, 1_000_000) == 0


def test_rmd_age_73():
    # Divisor 26.5 -> $1M / 26.5
    assert math.isclose(
        required_min_distribution(73, 1_000_000), 1_000_000 / 26.5, rel_tol=1e-9
    )


def test_top_of_bracket():
    # 12% bracket ends at $47,150 single
    assert top_of_bracket(0.12, "single") == 47_150
    assert top_of_bracket(0.22, "single") == 100_525
