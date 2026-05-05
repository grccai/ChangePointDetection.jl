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
GlidePolicy parameter vector (length 16):
  0..1   taxable.stock     (start_value, end_value)
  2..3   taxable.bond      (start, end)
  4..5   traditional.stock (start, end)
  6..7   traditional.bond  (start, end)
  8..9   roth.stock        (start, end)
  10..11 roth.bond         (start, end)
  12     conversion bracket index for FIRE-gap phase (snapped to discrete)
  13     conversion bracket index for SS-window phase (snapped)
  14     trad contribution split (0..1)
  15     wealth_responsiveness (0..2 typically)
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
    """Decode a 16-element vector to a GlidePolicy. The (start_age, end_age)
    knot positions are *fixed* by the scenario (passed here); only the
    *values* at those knots are free."""
    if len(x) != 16:
        raise ValueError(f"expected 16 params for glide policy, got {len(x)}")

    def _glide(start_v: float, end_v: float) -> GlidePath:
        return GlidePath([(start_age, _clip(start_v)),
                          (end_age, _clip(end_v))])

    return GlidePolicy(
        taxable=AccountGlide(stock=_glide(x[0], x[1]),
                             bond=_glide(x[2], x[3])),
        traditional=AccountGlide(stock=_glide(x[4], x[5]),
                                 bond=_glide(x[6], x[7])),
        roth=AccountGlide(stock=_glide(x[8], x[9]),
                          bond=_glide(x[10], x[11])),
        conv_during_fire_gap=_snap_bracket(x[12]),
        conv_during_ss_window=_snap_bracket(x[13]),
        trad_contribution_split=_clip(x[14]),
        wealth_responsiveness=max(0.0, min(2.0, x[15])),
        retirement_age=retirement_age,
        ss_age=ss_age,
        rmd_age=rmd_age,
    )


GLIDE_PARAM_BOUNDS: list[tuple[float, float]] = [
    (0.0, 1.0), (0.0, 1.0),  # taxable stock start/end
    (0.0, 1.0), (0.0, 1.0),  # taxable bond
    (0.0, 1.0), (0.0, 1.0),  # traditional stock
    (0.0, 1.0), (0.0, 1.0),  # traditional bond
    (0.0, 1.0), (0.0, 1.0),  # roth stock
    (0.0, 1.0), (0.0, 1.0),  # roth bond
    (0.0, 5.0),               # conv FIRE-gap idx
    (0.0, 5.0),               # conv SS-window idx
    (0.0, 1.0),               # trad split
    (0.0, 2.0),               # wealth_responsiveness
]
