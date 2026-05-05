"""Tax-efficient asset location.

Given an overall (stock, bond, cash) target and current account totals,
choose which assets sit in which account to minimise long-run tax drag.

Heuristic priority (Reichenstein-style):
  * **Bonds** -> Traditional first (yield taxed as ordinary anyway, but the
    tax is deferred), then Taxable, then Roth.
  * **Cash** -> Taxable first (it's where you actually use the cash; ordinary
    yield is small in dollars), then Traditional, then Roth.
  * **Stocks** -> Roth first (highest expected long-run return; tax-free
    growth maximises the value of the shelter), then Taxable (LTCG +
    step-up-at-death), then Traditional (forced ordinary tax on withdrawal
    converts what should be LT gains into ordinary income).

Once dollar placements are computed, we convert to per-account
fractional allocations (the form the simulator's rebalancing logic expects).

This module provides two modes the optimizer can run in:
  * "free" — the optimizer chooses 6 per-account allocation parameters
    independently. Captures both location and allocation jointly.
  * "heuristic" — the optimizer chooses overall (stock, bond) and the
    location is fixed by the heuristic above. Decision space is 2 vars
    instead of 6, which trades expressiveness for sample efficiency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .accounts import Asset
from .config import Allocation, TargetAllocations


PRIORITY = {
    "bond":  ["traditional", "taxable", "roth"],
    "cash":  ["taxable", "traditional", "roth"],
    "stock": ["roth", "taxable", "traditional"],
}


def tax_efficient_dollars(overall_stock: float, overall_bond: float,
                          overall_cash: float,
                          taxable_total: float, traditional_total: float,
                          roth_total: float
                          ) -> dict[str, dict[str, float]]:
    """Greedy per-asset placement. Returns nested dict
    out[account][asset] -> dollars."""
    s = overall_stock + overall_bond + overall_cash
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"overall fractions must sum to 1, got {s}")
    total = taxable_total + traditional_total + roth_total
    target = {"stock": total * overall_stock,
              "bond":  total * overall_bond,
              "cash":  total * overall_cash}
    remaining = {"taxable": taxable_total, "traditional": traditional_total,
                 "roth": roth_total}
    out = {acc: {a: 0.0 for a in ("stock", "bond", "cash")} for acc in remaining}
    # Order matters because each asset's priority can compete for the same
    # account. Allocate bonds first (their priority is the most tax-saving),
    # then cash, then stocks fill what's left.
    for asset in ("bond", "cash", "stock"):
        rem_asset = target[asset]
        for acc in PRIORITY[asset]:
            place = min(rem_asset, remaining[acc])
            out[acc][asset] += place
            remaining[acc] -= place
            rem_asset -= place
            if rem_asset <= 1e-9:
                break
    return out


def heuristic_target_allocations(overall_stock: float, overall_bond: float,
                                 overall_cash: float,
                                 taxable_total: float, traditional_total: float,
                                 roth_total: float
                                 ) -> TargetAllocations:
    """Convert a tax-efficient dollar placement to per-account fractional
    allocations. If an account total is 0, it gets a degenerate (1,0,0)
    target by default — the simulator multiplies by the account's actual
    total each year so it doesn't matter what fraction we report."""
    out = tax_efficient_dollars(
        overall_stock, overall_bond, overall_cash,
        taxable_total, traditional_total, roth_total,
    )

    def _to_alloc(acc_dict: dict[str, float], total: float) -> Allocation:
        if total <= 0:
            return Allocation(stock=1.0, bond=0.0, cash=0.0)
        s = acc_dict["stock"] / total
        b = acc_dict["bond"] / total
        c = acc_dict["cash"] / total
        # Numerical safety: clip and renormalise
        s, b, c = max(0.0, s), max(0.0, b), max(0.0, c)
        total_frac = s + b + c
        if total_frac > 0:
            s, b, c = s / total_frac, b / total_frac, c / total_frac
        return Allocation(stock=s, bond=b, cash=c)

    return TargetAllocations(
        taxable=_to_alloc(out["taxable"], taxable_total),
        traditional=_to_alloc(out["traditional"], traditional_total),
        roth=_to_alloc(out["roth"], roth_total),
    )


def overall_allocation_of(targets: TargetAllocations,
                          taxable_total: float, traditional_total: float,
                          roth_total: float
                          ) -> tuple[float, float, float]:
    """Compute the overall (stock, bond, cash) fractions implied by per-account
    targets and current account totals."""
    total = taxable_total + traditional_total + roth_total
    if total <= 0:
        return (0.0, 0.0, 0.0)
    s = (targets.taxable.stock * taxable_total
         + targets.traditional.stock * traditional_total
         + targets.roth.stock * roth_total) / total
    b = (targets.taxable.bond * taxable_total
         + targets.traditional.bond * traditional_total
         + targets.roth.bond * roth_total) / total
    c = (targets.taxable.cash * taxable_total
         + targets.traditional.cash * traditional_total
         + targets.roth.cash * roth_total) / total
    return (s, b, c)
