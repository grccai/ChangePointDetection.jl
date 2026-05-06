"""Time-aware decision policies.

The simulator asks a Policy each year:

    decision = policy.decide(age, year_idx, portfolio_summary)

The decision carries the per-account target allocations, the Roth conversion
bracket target, and the Traditional-vs-Roth contribution split — i.e., all the
levers the optimizer can choose. A Policy can return a *function of state*,
which lets us optimize today's decisions under the assumption that future
years' decisions will be made by the same (optimized) policy.

Two concrete policy classes:

* `StaticPolicy` — single fixed value for each lever. Equivalent to the
  original optimizer's decision space; preserved as default for back-compat
  and as a baseline.

* `GlidePolicy` — life-cycle aware:
    - Each account's stock and bond fractions follow a 2-knot piecewise-linear
      *glide path* over age (cash = 1 - stock - bond).
    - The Roth conversion bracket target is *life-phase conditional*: separate
      values for the FIRE-to-Social-Security gap, for the SS-to-RMD window,
      and zero outside those phases (we don't convert while drawing wages or
      after RMDs make conversion counterproductive).
    - The Traditional vs. Roth contribution split is constant during working
      years (we keep this as a single optimizer variable; richer
      parametrizations didn't earn their cost in pilots).
    - A single `wealth_responsiveness` parameter shifts the *current-year*
      stock fraction up if the simulated portfolio is below its target glide
      and down if it is above — implementing the "ahead/behind plan"
      adjustment the user asked for.

This is a *parametric policy*: the optimizer searches over a low-dimensional
θ such that the policy π(s_t, t; θ) maximizes expected utility under the
forward simulation. It is not literally Bellman-optimal (no backward
induction over a discretised state), but the policy class is rich enough to
capture the levers that matter most: glide paths, life-phase Roth
conversions, and ahead/behind-plan risk adjustment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .config import Allocation, TargetAllocations


# ---------- Decision and state summary ----------

@dataclass
class Decision:
    """Per-year output of a Policy."""
    allocations: TargetAllocations
    conversion_bracket: float | None     # None = skip Roth conversion this year
    trad_contribution_split: float       # 0..1, fraction of 401k pool to Traditional


@dataclass
class StateSummary:
    """Per-path snapshot the policy can condition on. Held simple — most
    decisions only depend on age and FIRE progress, and for vector
    simulation we want one Decision per year (not per path), so we summarise
    portfolio state via the cross-path mean."""
    age: float
    year_idx: int
    years_to_retirement: float
    fire_target_real: float
    median_real_wealth: float       # cross-path median (or sample mean)
    fire_progress_ratio: float       # median_real_wealth / fire_target_real


# ---------- Policy Protocol ----------

class Policy(Protocol):
    def decide(self, ss: StateSummary) -> Decision: ...


# ---------- Helpers ----------

def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# Note: _clip is used both as _clip(x) → [0,1] and _clip(x, lo, hi) → [lo, hi].


def _alloc(stock: float, bond: float) -> Allocation:
    s = _clip(stock, 0.0, 1.0)
    b = _clip(bond, 0.0, 1.0 - s)
    c = max(0.0, 1.0 - s - b)
    return Allocation(stock=s, bond=b, cash=c)


# ---------- StaticPolicy ----------

@dataclass
class StaticPolicy:
    """Returns the same Decision every year. Equivalent to the prior
    optimizer's policy class."""
    allocations: TargetAllocations
    conversion_bracket: float | None
    trad_contribution_split: float = 1.0

    def decide(self, ss: StateSummary) -> Decision:
        return Decision(self.allocations, self.conversion_bracket,
                        self.trad_contribution_split)


# ---------- GlidePath ----------

@dataclass
class GlidePath:
    """Piecewise-linear interpolation between (age, value) knots, sorted by
    age. Outside [first.age, last.age] the value is held flat at the nearest
    endpoint."""
    knots: list[tuple[float, float]]

    def at(self, age: float) -> float:
        ks = self.knots
        if age <= ks[0][0]:
            return ks[0][1]
        if age >= ks[-1][0]:
            return ks[-1][1]
        for i in range(len(ks) - 1):
            (a0, v0), (a1, v1) = ks[i], ks[i + 1]
            if a0 <= age <= a1:
                t = (age - a0) / (a1 - a0)
                return v0 + t * (v1 - v0)
        return ks[-1][1]  # unreachable


@dataclass
class AccountGlide:
    """Independent glide paths for stock and bond fractions in one account.
    Cash fraction is the residual."""
    stock: GlidePath
    bond: GlidePath

    def allocation_at(self, age: float) -> Allocation:
        return _alloc(self.stock.at(age), self.bond.at(age))


# ---------- GlidePolicy ----------

@dataclass
class GlidePolicy:
    """Time- and state-aware policy. See module docstring for math."""
    taxable: AccountGlide
    traditional: AccountGlide
    roth: AccountGlide

    # Roth conversion bracket targets, conditional on life phase.
    # None for either = skip conversions in that phase.
    conv_during_fire_gap: float | None = 0.12   # retirement_age <= age < ss_age
    conv_during_ss_window: float | None = 0.12  # ss_age <= age < rmd_age

    # Contribution split during working years (0..1, fraction Traditional).
    trad_contribution_split: float = 1.0

    # Wealth-vs-target responsiveness:
    #   if median_real_wealth / fire_target_at_age > 1, reduce stock by
    #   `wealth_responsiveness * (ratio - 1)` (de-risk when ahead).
    #   if < 1, increase stock by the same magnitude (risk up when behind).
    # Set to 0 to disable.
    wealth_responsiveness: float = 0.0

    # Cached scenario constants (set at build time).
    retirement_age: float = 0.0
    ss_age: float = 67.0
    rmd_age: float = 73.0

    # ----- decision logic -----

    def decide(self, ss: StateSummary) -> Decision:
        age = ss.age
        # 1) Base allocations from glide
        tax = self.taxable.allocation_at(age)
        trad = self.traditional.allocation_at(age)
        roth = self.roth.allocation_at(age)

        # 2) Wealth-vs-target tweak: shift stock fraction by a factor of
        # (fire_progress_ratio - 1). Conservative: only apply once we have
        # a meaningful trajectory (year_idx > 2) and avoid div-by-zero.
        if self.wealth_responsiveness != 0 and ss.year_idx > 2 \
                and ss.fire_target_real > 0:
            ratio = ss.fire_progress_ratio
            shift = -self.wealth_responsiveness * (ratio - 1.0)
            tax = _alloc(tax.stock + shift, tax.bond)
            trad = _alloc(trad.stock + shift, trad.bond)
            roth = _alloc(roth.stock + shift, roth.bond)

        # 3) Roth conversion: phase-conditional
        if age < self.retirement_age:
            conv = None
        elif age < self.ss_age:
            conv = self.conv_during_fire_gap
        elif age < self.rmd_age:
            conv = self.conv_during_ss_window
        else:
            conv = None

        return Decision(
            allocations=TargetAllocations(taxable=tax, traditional=trad, roth=roth),
            conversion_bracket=conv,
            trad_contribution_split=self.trad_contribution_split,
        )


# ---------- Factory: build a GlidePolicy from a flat parameter vector ----------

GLIDE_PARAM_LAYOUT = """
GlidePolicy parameter vector (length 12). Cash is constrained to 0 in
Traditional and Roth (it makes no sense to hold cash in a tax-advantaged
account); their bond fractions are implicitly (1 - stock).

  0..1   taxable.stock  (start_value, end_value)
  2..3   taxable.bond   (start, end)               [cash = 1 - stock - bond]
  4..5   traditional.stock (start, end)            [bond = 1 - stock, cash = 0]
  6..7   roth.stock        (start, end)            [bond = 1 - stock, cash = 0]
  8      conversion bracket index for FIRE-gap phase (snapped to discrete)
  9      conversion bracket index for SS-window phase (snapped)
  10     trad contribution split (0..1)
  11     wealth_responsiveness (0..2 typically)
"""


_CONV_CANDIDATES = [None, 0.10, 0.12, 0.22, 0.24, 0.32]


def _snap_bracket(idx_raw: float) -> float | None:
    idx = int(round(idx_raw))
    idx = max(0, min(len(_CONV_CANDIDATES) - 1, idx))
    return _CONV_CANDIDATES[idx]


def build_glide_policy(x: list[float] | tuple[float, ...],
                       start_age: float, end_age: float,
                       retirement_age: float, ss_age: float = 67.0,
                       rmd_age: float = 73.0) -> GlidePolicy:
    """Decode a 12-element vector to a GlidePolicy. Cash is forbidden in
    Trad/Roth (their bond glides are derived as 1 - stock_glide); only the
    taxable account has a free bond and (residual) cash glide."""
    if len(x) != 12:
        raise ValueError(f"expected 12 params for glide policy, got {len(x)}")

    def _glide(start_v: float, end_v: float) -> GlidePath:
        return GlidePath([(start_age, _clip(start_v)),
                          (end_age, _clip(end_v))])

    # Trad/Roth: bond glide = 1 - stock glide (yields cash = 0 after _alloc).
    def _bond_complement(stock_start: float, stock_end: float) -> GlidePath:
        return GlidePath([(start_age, 1.0 - _clip(stock_start)),
                          (end_age, 1.0 - _clip(stock_end))])

    return GlidePolicy(
        taxable=AccountGlide(stock=_glide(x[0], x[1]),
                             bond=_glide(x[2], x[3])),
        traditional=AccountGlide(stock=_glide(x[4], x[5]),
                                 bond=_bond_complement(x[4], x[5])),
        roth=AccountGlide(stock=_glide(x[6], x[7]),
                          bond=_bond_complement(x[6], x[7])),
        conv_during_fire_gap=_snap_bracket(x[8]),
        conv_during_ss_window=_snap_bracket(x[9]),
        trad_contribution_split=_clip(x[10]),
        wealth_responsiveness=max(0.0, min(2.0, x[11])),
        retirement_age=retirement_age,
        ss_age=ss_age,
        rmd_age=rmd_age,
    )


GLIDE_PARAM_BOUNDS: list[tuple[float, float]] = [
    (0.0, 1.0), (0.0, 1.0),  # taxable stock start/end
    (0.0, 1.0), (0.0, 1.0),  # taxable bond
    (0.0, 1.0), (0.0, 1.0),  # traditional stock (bond = 1-stock)
    (0.0, 1.0), (0.0, 1.0),  # roth stock (bond = 1-stock)
    (0.0, 5.0),               # conv FIRE-gap idx
    (0.0, 5.0),               # conv SS-window idx
    (0.0, 1.0),               # trad split
    (0.0, 2.0),               # wealth_responsiveness
]


# ---------- Three-knot glide policy ----------

THREE_KNOT_GLIDE_PARAM_LAYOUT = """
ThreeKnotGlidePolicy parameter vector (length 16). Knots at start_age,
retirement_age, and end_of_plan_age — separates accumulation and
decumulation slopes. Cash is again forbidden in Trad/Roth.

  0..2    taxable.stock   (start, retire, end)
  3..5    taxable.bond    (start, retire, end)        [cash = 1 - stock - bond]
  6..8    traditional.stock (start, retire, end)      [bond = 1 - stock, cash = 0]
  9..11   roth.stock        (start, retire, end)      [bond = 1 - stock, cash = 0]
  12      conversion bracket idx for FIRE-gap phase
  13      conversion bracket idx for SS-window phase
  14      trad contribution split
  15      wealth_responsiveness
"""


def build_three_knot_glide_policy(
        x: list[float] | tuple[float, ...],
        start_age: float, retirement_age: float, end_age: float,
        ss_age: float = 67.0, rmd_age: float = 73.0) -> GlidePolicy:
    """Decode a 16-element vector to a GlidePolicy with three knots
    per glide (start, retirement, end). Cash is forbidden in Trad/Roth.
    Lets accumulation and decumulation glide slopes differ."""
    if len(x) != 16:
        raise ValueError(f"expected 16 params for three-knot policy, got {len(x)}")

    def _glide_3(s: float, m: float, e: float) -> GlidePath:
        return GlidePath([(start_age, _clip(s)),
                          (retirement_age, _clip(m)),
                          (end_age, _clip(e))])

    def _bond_complement(s: float, m: float, e: float) -> GlidePath:
        return GlidePath([(start_age, 1.0 - _clip(s)),
                          (retirement_age, 1.0 - _clip(m)),
                          (end_age, 1.0 - _clip(e))])

    return GlidePolicy(
        taxable=AccountGlide(stock=_glide_3(x[0], x[1], x[2]),
                             bond=_glide_3(x[3], x[4], x[5])),
        traditional=AccountGlide(stock=_glide_3(x[6], x[7], x[8]),
                                 bond=_bond_complement(x[6], x[7], x[8])),
        roth=AccountGlide(stock=_glide_3(x[9], x[10], x[11]),
                          bond=_bond_complement(x[9], x[10], x[11])),
        conv_during_fire_gap=_snap_bracket(x[12]),
        conv_during_ss_window=_snap_bracket(x[13]),
        trad_contribution_split=_clip(x[14]),
        wealth_responsiveness=max(0.0, min(2.0, x[15])),
        retirement_age=retirement_age,
        ss_age=ss_age,
        rmd_age=rmd_age,
    )


THREE_KNOT_GLIDE_PARAM_BOUNDS: list[tuple[float, float]] = [
    (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),  # taxable stock (3 knots)
    (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),  # taxable bond
    (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),  # traditional stock
    (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),  # roth stock
    (0.0, 5.0),                           # conv FIRE-gap
    (0.0, 5.0),                           # conv SS-window
    (0.0, 1.0),                           # trad split
    (0.0, 2.0),                           # wealth_responsiveness
]


# ---------- Bond Tent (Kitces & Pfau): V-shaped equity glide ----------

@dataclass
class BondTentPolicy:
    """V-shaped equity allocation:

        offset = |age - tent_age|
        stock(age) = stock_high                                  if offset >= span
                   = stock_low + (stock_high - stock_low) * offset/span    otherwise

    Idea: equity is high in accumulation, dipping to a low at the tent age
    (typically near retirement), then climbing back as the
    sequence-of-returns risk window passes. Wealth-responsiveness term is
    optional; when zero the policy is purely age-driven.

    Same stock fraction across all three accounts; bond = 1 - stock in
    Trad/Roth (cash forbidden); taxable holds an additional `taxable_cash`
    fraction with bond = 1 - stock - taxable_cash."""
    stock_high: float
    stock_low: float
    tent_age: float
    span: float
    taxable_cash: float = 0.0

    conv_during_fire_gap: float | None = None
    conv_during_ss_window: float | None = None
    trad_contribution_split: float = 1.0
    wealth_responsiveness: float = 0.0

    retirement_age: float = 0.0
    ss_age: float = 67.0
    rmd_age: float = 73.0

    def _stock_at(self, age: float) -> float:
        offset = abs(age - self.tent_age)
        if self.span <= 0:
            return self.stock_high
        if offset >= self.span:
            return self.stock_high
        return self.stock_low + (self.stock_high - self.stock_low) * (offset / self.span)

    def decide(self, ss: StateSummary) -> Decision:
        age = ss.age
        stock = self._stock_at(age)
        # Optional wealth-responsiveness on top of the tent shape
        if self.wealth_responsiveness != 0 and ss.year_idx > 2 \
                and ss.fire_target_real > 0:
            shift = -self.wealth_responsiveness * (ss.fire_progress_ratio - 1.0)
            stock = _clip(stock + shift)

        tax_cash = _clip(self.taxable_cash, 0.0, max(0.0, 1.0 - stock))
        tax = _alloc(stock, max(0.0, 1.0 - stock - tax_cash))
        trad = _alloc(stock, max(0.0, 1.0 - stock))
        roth = _alloc(stock, max(0.0, 1.0 - stock))

        if age < self.retirement_age:
            conv = None
        elif age < self.ss_age:
            conv = self.conv_during_fire_gap
        elif age < self.rmd_age:
            conv = self.conv_during_ss_window
        else:
            conv = None

        return Decision(
            allocations=TargetAllocations(taxable=tax, traditional=trad, roth=roth),
            conversion_bracket=conv,
            trad_contribution_split=self.trad_contribution_split,
        )


def build_bond_tent_policy(x: list[float] | tuple[float, ...],
                            retirement_age: float, ss_age: float = 67.0,
                            rmd_age: float = 73.0) -> BondTentPolicy:
    """Decode a 9-element vector to a BondTentPolicy.

    Parameter layout:
      0  stock_high              [0, 1]
      1  stock_low               [0, 1]    (clipped to <= stock_high)
      2  tent_age_offset         [-15, 20] relative to retirement_age
      3  span                    [3, 30]   years
      4  taxable_cash            [0, 0.4]
      5  conv FIRE-gap idx       (snapped to discrete bracket)
      6  conv SS-window idx      (snapped)
      7  trad split              [0, 1]
      8  wealth_responsiveness   [0, 2]"""
    if len(x) != 9:
        raise ValueError(f"expected 9 params for bond_tent, got {len(x)}")
    sh = _clip(x[0])
    sl = min(_clip(x[1]), sh)   # ensure stock_low <= stock_high
    return BondTentPolicy(
        stock_high=sh, stock_low=sl,
        tent_age=retirement_age + max(-15.0, min(20.0, x[2])),
        span=max(3.0, min(30.0, x[3])),
        taxable_cash=_clip(x[4], 0.0, 0.4),
        conv_during_fire_gap=_snap_bracket(x[5]),
        conv_during_ss_window=_snap_bracket(x[6]),
        trad_contribution_split=_clip(x[7]),
        wealth_responsiveness=max(0.0, min(2.0, x[8])),
        retirement_age=retirement_age,
        ss_age=ss_age,
        rmd_age=rmd_age,
    )


BOND_TENT_PARAM_BOUNDS: list[tuple[float, float]] = [
    (0.0, 1.0),     # stock_high
    (0.0, 1.0),     # stock_low
    (-15.0, 20.0),  # tent_age_offset
    (3.0, 30.0),    # span
    (0.0, 0.4),     # taxable_cash
    (0.0, 5.0),     # conv FIRE-gap idx
    (0.0, 5.0),     # conv SS-window idx
    (0.0, 1.0),     # trad split
    (0.0, 2.0),     # wealth_responsiveness
]


# ---------- CPPI (Constant Proportion Portfolio Insurance) ----------

@dataclass
class CPPIPolicy:
    """Wealth-anchored allocation:

        cushion = max(0, real_wealth - floor_real(t))
        stock_frac = clamp(multiplier * cushion / real_wealth, 0, upper_stock_cap)
        floor_real(t) = floor_real_at_start * (1 + floor_growth_rate)^t

    Mechanically protects the floor: as wealth approaches floor, stock
    fraction collapses to 0 (insurance kicks in). As wealth grows above
    floor, the multiplier amplifies risk-taking. Different *shape* of
    risk-taking from age-glide policies — entirely state-driven, not
    age-driven.

    Uses cross-path median real wealth (per the StateSummary architecture);
    decisions are broadcast to all paths each year, so this is the
    median-path CPPI approximation."""
    floor_real_at_start: float    # in real $
    floor_growth_rate: float
    multiplier: float
    upper_stock_cap: float = 1.0
    taxable_cash: float = 0.0

    conv_during_fire_gap: float | None = None
    conv_during_ss_window: float | None = None
    trad_contribution_split: float = 1.0

    retirement_age: float = 0.0
    ss_age: float = 67.0
    rmd_age: float = 73.0

    def decide(self, ss: StateSummary) -> Decision:
        age = ss.age
        floor_now = self.floor_real_at_start * (1 + self.floor_growth_rate) ** ss.year_idx
        W = max(1.0, ss.median_real_wealth)
        cushion = max(0.0, W - floor_now)
        raw_stock = self.multiplier * cushion / W
        stock = _clip(raw_stock, 0.0, self.upper_stock_cap)

        tax_cash = _clip(self.taxable_cash, 0.0, max(0.0, 1.0 - stock))
        tax = _alloc(stock, max(0.0, 1.0 - stock - tax_cash))
        trad = _alloc(stock, max(0.0, 1.0 - stock))
        roth = _alloc(stock, max(0.0, 1.0 - stock))

        if age < self.retirement_age:
            conv = None
        elif age < self.ss_age:
            conv = self.conv_during_fire_gap
        elif age < self.rmd_age:
            conv = self.conv_during_ss_window
        else:
            conv = None

        return Decision(
            allocations=TargetAllocations(taxable=tax, traditional=trad, roth=roth),
            conversion_bracket=conv,
            trad_contribution_split=self.trad_contribution_split,
        )


def build_cppi_policy(x: list[float] | tuple[float, ...],
                      retirement_age: float, ss_age: float = 67.0,
                      rmd_age: float = 73.0) -> CPPIPolicy:
    """Decode an 8-element vector to a CPPIPolicy.

    Parameter layout:
      0  floor_real_at_start ($M)   [0, 3]
      1  floor_growth_rate           [-0.02, 0.05]
      2  multiplier                  [1, 5]
      3  upper_stock_cap             [0, 1]
      4  taxable_cash                [0, 0.4]
      5  conv FIRE-gap idx           (snapped)
      6  conv SS-window idx          (snapped)
      7  trad split                  [0, 1]"""
    if len(x) != 8:
        raise ValueError(f"expected 8 params for cppi, got {len(x)}")
    return CPPIPolicy(
        floor_real_at_start=max(0.0, x[0]) * 1_000_000,
        floor_growth_rate=max(-0.02, min(0.05, x[1])),
        multiplier=max(1.0, min(5.0, x[2])),
        upper_stock_cap=_clip(x[3]),
        taxable_cash=_clip(x[4], 0.0, 0.4),
        conv_during_fire_gap=_snap_bracket(x[5]),
        conv_during_ss_window=_snap_bracket(x[6]),
        trad_contribution_split=_clip(x[7]),
        retirement_age=retirement_age,
        ss_age=ss_age,
        rmd_age=rmd_age,
    )


CPPI_PARAM_BOUNDS: list[tuple[float, float]] = [
    (0.0, 3.0),      # floor_real_at_start ($M)
    (-0.02, 0.05),   # floor_growth_rate
    (1.0, 5.0),      # multiplier
    (0.0, 1.0),      # upper_stock_cap
    (0.0, 0.4),      # taxable_cash
    (0.0, 5.0),      # conv FIRE-gap idx
    (0.0, 5.0),      # conv SS-window idx
    (0.0, 1.0),      # trad split
]


# ---------- Bodie-Merton Human-Capital Glide ----------

@dataclass
class BodieMertonPolicy:
    """Allocation derived from the Merton optimal-portfolio formula applied
    to *total* economic wealth (financial + human capital), where HC is
    treated as a bond.

        target_total_stock = (μ_excess) / (γ · σ²)   ← Merton constant
        HC(t) = PV of remaining real wages, discounted at hc_discount_rate
        total = W + HC
        stock_in_W = clamp(target_total_stock · total / W, 0, 1)

    The "glide" emerges endogenously: as HC depletes with age, the
    financial portfolio's stock fraction declines from ~100% (HC large)
    toward the Merton constant (HC = 0). Two fundamentals (γ, r_hc) do
    the work of many free knots.

    HC trajectory is pre-computed at policy build time from the scenario's
    income sources + deterministic inflation."""
    hc_by_year: np.ndarray              # (H+1,)
    target_total_stock_frac: float      # Merton constant, derived from γ and market
    taxable_cash: float = 0.0

    conv_during_fire_gap: float | None = None
    conv_during_ss_window: float | None = None
    trad_contribution_split: float = 1.0

    retirement_age: float = 0.0
    ss_age: float = 67.0
    rmd_age: float = 73.0

    def decide(self, ss: StateSummary) -> Decision:
        age = ss.age
        year_idx = ss.year_idx
        if year_idx < len(self.hc_by_year):
            HC = self.hc_by_year[year_idx]
        else:
            HC = 0.0
        W = max(1.0, ss.median_real_wealth)
        total = W + HC
        raw_stock = self.target_total_stock_frac * total / W
        stock = _clip(raw_stock)

        tax_cash = _clip(self.taxable_cash, 0.0, max(0.0, 1.0 - stock))
        tax = _alloc(stock, max(0.0, 1.0 - stock - tax_cash))
        trad = _alloc(stock, max(0.0, 1.0 - stock))
        roth = _alloc(stock, max(0.0, 1.0 - stock))

        if age < self.retirement_age:
            conv = None
        elif age < self.ss_age:
            conv = self.conv_during_fire_gap
        elif age < self.rmd_age:
            conv = self.conv_during_ss_window
        else:
            conv = None

        return Decision(
            allocations=TargetAllocations(taxable=tax, traditional=trad, roth=roth),
            conversion_bracket=conv,
            trad_contribution_split=self.trad_contribution_split,
        )


def _compute_hc_trajectory(scn, r_hc: float) -> np.ndarray:
    """For each year_idx, compute the present value (in real dollars) of all
    future real wages, discounted at r_hc."""
    horizon = scn.profile.horizon()
    sim_start = scn.profile.start_date
    timeline = scn.state_taxes
    inflation_rate = scn.market.inflation_mean
    real_wages = np.zeros(horizon)
    deflator = 1.0
    for y in range(horizon):
        deflator *= (1.0 + inflation_rate)
        nominal = timeline.total_wages(sim_start, y)
        real_wages[y] = nominal / deflator
    hc = np.zeros(horizon + 1)
    discount = (1.0 + r_hc)
    for y_now in range(horizon + 1):
        pv = 0.0
        for y in range(y_now, horizon):
            pv += real_wages[y] / discount ** (y - y_now)
        hc[y_now] = pv
    return hc


def build_bodie_merton_policy(x: list[float] | tuple[float, ...],
                               scn,
                               retirement_age: float, ss_age: float = 67.0,
                               rmd_age: float = 73.0) -> BodieMertonPolicy:
    """Decode a 6-element vector to a BodieMertonPolicy.

    Parameter layout:
      0  risk_aversion (γ)         [1.0, 10.0]   typical 2-5
      1  hc_discount_rate (r_hc)   [0.0, 0.08]   real, typical 0.02-0.04
      2  taxable_cash              [0, 0.4]
      3  conv FIRE-gap idx         (snapped)
      4  conv SS-window idx        (snapped)
      5  trad split                [0, 1]"""
    if len(x) != 6:
        raise ValueError(f"expected 6 params for bodie_merton, got {len(x)}")
    gamma = max(1.0, min(10.0, x[0]))
    r_hc = max(0.0, min(0.08, x[1]))
    # Merton constant: target stock fraction of TOTAL wealth.
    mu_stock = scn.market.stocks.real_return
    sigma_stock = scn.market.stocks.vol
    r_riskfree = scn.market.cash.real_return
    excess = mu_stock - r_riskfree
    if sigma_stock <= 0:
        target_total = 1.0 if excess > 0 else 0.0
    else:
        target_total = max(0.0, min(1.0, excess / (gamma * sigma_stock ** 2)))
    hc_traj = _compute_hc_trajectory(scn, r_hc)
    return BodieMertonPolicy(
        hc_by_year=hc_traj,
        target_total_stock_frac=target_total,
        taxable_cash=_clip(x[2], 0.0, 0.4),
        conv_during_fire_gap=_snap_bracket(x[3]),
        conv_during_ss_window=_snap_bracket(x[4]),
        trad_contribution_split=_clip(x[5]),
        retirement_age=retirement_age,
        ss_age=ss_age,
        rmd_age=rmd_age,
    )


BODIE_MERTON_PARAM_BOUNDS: list[tuple[float, float]] = [
    (1.0, 10.0),     # risk_aversion
    (0.0, 0.08),     # hc_discount_rate
    (0.0, 0.4),      # taxable_cash
    (0.0, 5.0),      # conv FIRE-gap idx
    (0.0, 5.0),      # conv SS-window idx
    (0.0, 1.0),      # trad split
]
