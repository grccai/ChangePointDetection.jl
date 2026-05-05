import math
import pytest

from retire.location import (tax_efficient_dollars,
                              heuristic_target_allocations,
                              overall_allocation_of)
from retire.config import Allocation, TargetAllocations


def test_basic_placement():
    # 60/30/10 overall, $100 taxable, $200 trad, $100 roth (total $400).
    # Bonds first -> trad. Need $120 bonds. Trad has $200; bonds fill $120,
    # trad still has $80.
    # Cash next -> taxable. Need $40 cash. Taxable has $100; cash fills $40,
    # taxable has $60.
    # Stock last -> roth first ($100), then taxable ($60), then trad ($80).
    out = tax_efficient_dollars(0.60, 0.30, 0.10, 100, 200, 100)
    assert math.isclose(out["traditional"]["bond"], 120.0)
    assert math.isclose(out["traditional"]["stock"], 80.0)
    assert math.isclose(out["traditional"]["cash"], 0.0)
    assert math.isclose(out["taxable"]["cash"], 40.0)
    assert math.isclose(out["taxable"]["stock"], 60.0)
    assert math.isclose(out["taxable"]["bond"], 0.0)
    assert math.isclose(out["roth"]["stock"], 100.0)
    assert math.isclose(out["roth"]["bond"], 0.0)
    assert math.isclose(out["roth"]["cash"], 0.0)


def test_placement_overall_invariant():
    # Sum of placement equals total target dollars, by asset.
    overall = (0.7, 0.2, 0.1)
    accs = (250_000, 500_000, 150_000)
    out = tax_efficient_dollars(*overall, *accs)
    total = sum(accs)
    for asset, frac in zip(("stock", "bond", "cash"), overall):
        s = sum(out[acc][asset] for acc in out)
        assert math.isclose(s, total * frac, rel_tol=1e-9)


def test_placement_account_totals_invariant():
    overall = (0.6, 0.3, 0.1)
    accs = (100_000, 200_000, 50_000)
    out = tax_efficient_dollars(*overall, *accs)
    for acc, total in zip(("taxable", "traditional", "roth"), accs):
        s = sum(out[acc].values())
        assert math.isclose(s, total, rel_tol=1e-9)


def test_heuristic_allocations_normalize():
    # Even when account is empty, allocation should be valid (sum to 1).
    targets = heuristic_target_allocations(0.6, 0.3, 0.1, 100, 0, 100)
    for a in (targets.taxable, targets.traditional, targets.roth):
        assert math.isclose(a.stock + a.bond + a.cash, 1.0, abs_tol=1e-9)


def test_overall_allocation_of_recovers():
    # Build a target from the heuristic, then overall_allocation_of should
    # recover the original target.
    overall = (0.65, 0.30, 0.05)
    accs = (200_000, 300_000, 100_000)
    targets = heuristic_target_allocations(*overall, *accs)
    recovered = overall_allocation_of(targets, *accs)
    for r, o in zip(recovered, overall):
        assert math.isclose(r, o, abs_tol=1e-6)


def test_priority_stocks_in_roth():
    # When all accounts are equal and stock fraction matches roth size,
    # all stocks should go in Roth.
    out = tax_efficient_dollars(1/3, 1/3, 1/3, 100, 100, 100)
    # Stock allocation: 100. Priority is roth first.
    assert math.isclose(out["roth"]["stock"], 100.0)
    # Bond priority: trad first
    assert math.isclose(out["traditional"]["bond"], 100.0)
    # Cash: taxable first
    assert math.isclose(out["taxable"]["cash"], 100.0)
