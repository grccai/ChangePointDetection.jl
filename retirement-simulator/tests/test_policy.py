"""Tests for the time-aware policy classes."""

import datetime as dt
import math

import numpy as np
import pytest

from retire.config import Allocation, TargetAllocations
from retire.policy import (
    StaticPolicy, GlidePolicy, GlidePath, AccountGlide, Decision,
    StateSummary, build_glide_policy, GLIDE_PARAM_BOUNDS,
)


def _ss(age: float = 40.0, year_idx: int = 5,
        wealth: float = 1_000_000, target: float = 2_000_000) -> StateSummary:
    return StateSummary(
        age=age, year_idx=year_idx, years_to_retirement=10.0,
        fire_target_real=target, median_real_wealth=wealth,
        fire_progress_ratio=wealth / target if target > 0 else 1.0,
    )


# ---------- StaticPolicy ----------

def test_static_policy_returns_same_decision():
    targets = TargetAllocations(
        taxable=Allocation(0.7, 0.2, 0.1),
        traditional=Allocation(0.4, 0.6, 0.0),
        roth=Allocation(1.0, 0.0, 0.0),
    )
    p = StaticPolicy(allocations=targets, conversion_bracket=0.12,
                     trad_contribution_split=0.5)
    d1 = p.decide(_ss(age=30, year_idx=0))
    d2 = p.decide(_ss(age=70, year_idx=40))
    assert d1.allocations is d2.allocations
    assert d1.conversion_bracket == 0.12
    assert d1.trad_contribution_split == 0.5


# ---------- GlidePath / AccountGlide ----------

def test_glide_path_endpoints():
    g = GlidePath([(30.0, 0.9), (60.0, 0.4)])
    assert math.isclose(g.at(30), 0.9)
    assert math.isclose(g.at(60), 0.4)


def test_glide_path_interior():
    g = GlidePath([(30.0, 0.9), (60.0, 0.4)])
    # halfway: 0.65
    assert math.isclose(g.at(45), 0.65, rel_tol=1e-9)


def test_glide_path_outside_clamps():
    g = GlidePath([(30.0, 0.9), (60.0, 0.4)])
    assert g.at(20) == 0.9
    assert g.at(80) == 0.4


def test_account_glide_residual_cash():
    ag = AccountGlide(stock=GlidePath([(30, 0.7), (60, 0.5)]),
                      bond=GlidePath([(30, 0.2), (60, 0.4)]))
    a = ag.allocation_at(45)  # stock 0.6, bond 0.3, cash 0.1
    assert math.isclose(a.stock, 0.6)
    assert math.isclose(a.bond, 0.3)
    assert math.isclose(a.cash, 0.1)


def test_account_glide_clipping_when_overflow():
    """If the optimizer requests stock + bond > 1, clip without going negative."""
    ag = AccountGlide(stock=GlidePath([(30, 0.9), (60, 0.9)]),
                      bond=GlidePath([(30, 0.5), (60, 0.5)]))
    a = ag.allocation_at(45)
    assert math.isclose(a.stock + a.bond + a.cash, 1.0, abs_tol=1e-9)
    assert a.stock >= 0 and a.bond >= 0 and a.cash >= 0


# ---------- GlidePolicy ----------

def test_glide_policy_no_conv_pre_retirement():
    p = GlidePolicy(
        taxable=AccountGlide(GlidePath([(30, 0.8), (90, 0.4)]),
                             GlidePath([(30, 0.1), (90, 0.5)])),
        traditional=AccountGlide(GlidePath([(30, 0.5), (90, 0.3)]),
                                 GlidePath([(30, 0.5), (90, 0.7)])),
        roth=AccountGlide(GlidePath([(30, 1.0), (90, 0.6)]),
                          GlidePath([(30, 0.0), (90, 0.4)])),
        conv_during_fire_gap=0.12, conv_during_ss_window=0.22,
        retirement_age=50.0, ss_age=67.0, rmd_age=73.0,
    )
    # Pre-retirement -> no conversion
    d = p.decide(_ss(age=40, year_idx=10))
    assert d.conversion_bracket is None


def test_glide_policy_conv_phases():
    p = GlidePolicy(
        taxable=AccountGlide(GlidePath([(30, 0.8), (90, 0.4)]),
                             GlidePath([(30, 0.1), (90, 0.5)])),
        traditional=AccountGlide(GlidePath([(30, 0.5), (90, 0.3)]),
                                 GlidePath([(30, 0.5), (90, 0.7)])),
        roth=AccountGlide(GlidePath([(30, 1.0), (90, 0.6)]),
                          GlidePath([(30, 0.0), (90, 0.4)])),
        conv_during_fire_gap=0.12, conv_during_ss_window=0.22,
        retirement_age=50.0, ss_age=67.0, rmd_age=73.0,
    )
    # FIRE gap (50..67) -> 12%
    assert p.decide(_ss(age=55, year_idx=25)).conversion_bracket == 0.12
    # SS window (67..73) -> 22%
    assert p.decide(_ss(age=70, year_idx=40)).conversion_bracket == 0.22
    # Post-RMD -> None
    assert p.decide(_ss(age=80, year_idx=50)).conversion_bracket is None


def test_glide_policy_wealth_responsiveness_de_risk_when_ahead():
    """Ahead of plan (wealth > target) -> stock fraction reduced."""
    p = GlidePolicy(
        taxable=AccountGlide(GlidePath([(30, 0.8), (90, 0.8)]),  # flat 80% stock
                             GlidePath([(30, 0.1), (90, 0.1)])),
        traditional=AccountGlide(GlidePath([(30, 0.5), (90, 0.5)]),
                                 GlidePath([(30, 0.5), (90, 0.5)])),
        roth=AccountGlide(GlidePath([(30, 1.0), (90, 1.0)]),
                          GlidePath([(30, 0.0), (90, 0.0)])),
        conv_during_fire_gap=None, conv_during_ss_window=None,
        retirement_age=50.0, wealth_responsiveness=0.5,
    )
    # Ahead of plan: ratio 1.5
    ahead = p.decide(_ss(age=45, year_idx=15, wealth=3_000_000, target=2_000_000))
    # On target: ratio 1.0
    on_target = p.decide(_ss(age=45, year_idx=15, wealth=2_000_000, target=2_000_000))
    # Behind: ratio 0.5
    behind = p.decide(_ss(age=45, year_idx=15, wealth=1_000_000, target=2_000_000))

    # At ratio 1: no shift
    assert math.isclose(on_target.allocations.taxable.stock, 0.8, abs_tol=1e-6)
    # Ahead -> de-risk -> less stock
    assert ahead.allocations.taxable.stock < on_target.allocations.taxable.stock
    # Behind -> risk up -> more stock
    assert behind.allocations.taxable.stock > on_target.allocations.taxable.stock


def test_build_glide_policy_decodes_vector():
    # 12-element vector: cash forbidden in Trad/Roth (their bond glides are
    # derived as 1 - stock_glide).
    x = [
        0.85, 0.40,   # taxable stock start, end
        0.05, 0.35,   # taxable bond
        0.50, 0.30,   # trad stock (bond = 1 - stock implicit)
        1.00, 0.60,   # roth stock (bond = 1 - stock implicit)
        2.0,          # conv FIRE-gap idx (-> 0.12)
        4.0,          # conv SS-window idx (-> 0.24)
        0.6,          # trad split
        0.3,          # wealth responsiveness
    ]
    p = build_glide_policy(x, start_age=35.0, end_age=95.0,
                          retirement_age=50.0, ss_age=67.0)
    assert math.isclose(p.taxable.stock.at(35), 0.85)
    assert math.isclose(p.taxable.stock.at(95), 0.40)
    assert p.conv_during_fire_gap == 0.12
    assert p.conv_during_ss_window == 0.24
    assert math.isclose(p.trad_contribution_split, 0.6)
    # Cash-forbidden in trad/roth: bond knot at start_age = 1 - 0.50 = 0.50.
    assert math.isclose(p.traditional.bond.at(35), 1.0 - 0.50, abs_tol=1e-9)
    # And the resulting allocation has cash = 0.
    a = p.traditional.allocation_at(35)
    assert math.isclose(a.cash, 0.0, abs_tol=1e-9)
    a_roth = p.roth.allocation_at(35)
    assert math.isclose(a_roth.cash, 0.0, abs_tol=1e-9)


def test_glide_param_bounds_length_matches_decoder():
    assert len(GLIDE_PARAM_BOUNDS) == 12


def test_three_knot_glide_decoder():
    from retire.policy import (build_three_knot_glide_policy,
                                THREE_KNOT_GLIDE_PARAM_BOUNDS)
    x = [
        0.85, 0.65, 0.30,   # taxable stock at start, retire, end
        0.05, 0.20, 0.50,   # taxable bond
        0.70, 0.40, 0.10,   # trad stock (bond = 1 - stock)
        1.00, 0.80, 0.50,   # roth stock
        2.0, 4.0,           # conv brackets (0.12, 0.24)
        0.6, 0.3,           # trad split, wealth_responsiveness
    ]
    p = build_three_knot_glide_policy(
        x, start_age=35.0, retirement_age=55.0, end_age=95.0)
    # Three-knot interpolation; the middle knot lets accumulation and
    # decumulation slopes differ.
    assert math.isclose(p.taxable.stock.at(35), 0.85)
    assert math.isclose(p.taxable.stock.at(55), 0.65)  # mid knot
    assert math.isclose(p.taxable.stock.at(95), 0.30)
    # Linear interp between mid knot (55, 0.65) and end (95, 0.30):
    # at age 75 = halfway, value = 0.475
    assert math.isclose(p.taxable.stock.at(75), 0.475, abs_tol=1e-9)
    # Cash forbidden in Trad/Roth
    assert math.isclose(p.traditional.allocation_at(55).cash, 0.0, abs_tol=1e-9)
    assert math.isclose(p.roth.allocation_at(55).cash, 0.0, abs_tol=1e-9)
    assert len(THREE_KNOT_GLIDE_PARAM_BOUNDS) == 16
