"""Vectorized simulation state.

All path-level state lives in numpy arrays whose first axis is `path`. This
replaces the per-path Python objects in `accounts.py`, which are still used
to *build* the initial state from a YAML config but are not in the simulation
hot loop.

Conventions:
  * Asset axis order: 0 stock, 1 bond, 2 cash.
  * Taxable account aggregates lots into two cohorts per asset: LT
    (held >= 1 year) and ST (held < 1 year). Within-cohort lot selection is
    not modelled (we treat each cohort as a single basket); this loses the
    "specific ID lowest-gain" refinement but recovers >50x speed and still
    distinguishes the load-bearing LT/ST tax difference.
  * Roth conversions are tracked by year-of-conversion in
    `roth_conversions[path, year_idx]` so we can enforce the 5-year clock.

Calling convention for in-place mutators: they return either nothing or
auxiliary outputs (e.g., realised gains) used by the caller for tax
computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .accounts import Asset, Portfolio


ASSET_ORDER = [Asset.STOCK, Asset.BOND, Asset.CASH]
ASSET_IDX = {a: i for i, a in enumerate(ASSET_ORDER)}
N_ASSETS = 3


@dataclass
class VState:
    """Vectorised state. All arrays' first axis is `path`."""
    n_paths: int
    horizon: int

    # Taxable account, cohorts by holding period
    tax_lt_value: np.ndarray   # (P, 3)
    tax_lt_basis: np.ndarray   # (P, 3)
    tax_st_value: np.ndarray   # (P, 3)
    tax_st_basis: np.ndarray   # (P, 3)

    # Tax-advantaged
    trad_balance: np.ndarray   # (P, 3)
    roth_balance: np.ndarray   # (P, 3)
    # Roth direct contributions (always penalty-free)
    roth_basis: np.ndarray     # (P,)
    # Roth conversions tracked by year_idx for the 5-year clock
    roth_conversions: np.ndarray  # (P, H + 1)

    # Bookkeeping
    nominal_income: np.ndarray         # (P,) — wages this year (nominal)
    cumulative_inflation: np.ndarray   # (P,) — multiplier from year 0 (real -> nominal)
    failed: np.ndarray                 # (P,) bool
    # Flexible spending: the per-path real wealth at the start of retirement.
    # Set lazily by the simulator on the first decumulation year of each path;
    # zeros until then. Used to compute the drawdown ratio that scales
    # current-year spending toward the configured floor.
    flex_baseline_wealth: np.ndarray   # (P,)

    # Outputs (filled across simulation)
    real_wealth: np.ndarray         # (P, H + 1)
    real_taxes: np.ndarray          # (P, H)
    real_target_spend: np.ndarray   # (P, H)
    real_shortfall: np.ndarray      # (P, H)
    # Detailed per-(year, account, asset) tracking. Account axis index:
    # 0 = taxable, 1 = traditional, 2 = roth.  Asset axis index follows
    # ASSET_ORDER (0=stock, 1=bond, 2=cash). Stored as REAL dollars (after
    # dividing by cumulative_inflation at end of each year).
    real_balance_by_year: np.ndarray   # (P, H + 1, 3, 3)
    real_contrib_by_year: np.ndarray   # (P, H, 3, 3) — fresh deposits only

    # ------- constructors -------

    @classmethod
    def from_portfolio(cls, portfolio: Portfolio, n_paths: int, horizon: int,
                       starting_nominal_income: float) -> "VState":
        """Initialize all `n_paths` paths from a single Portfolio specimen."""
        zeros2 = np.zeros((n_paths, N_ASSETS))
        zeros1 = np.zeros(n_paths)
        s = cls(
            n_paths=n_paths, horizon=horizon,
            tax_lt_value=zeros2.copy(), tax_lt_basis=zeros2.copy(),
            tax_st_value=zeros2.copy(), tax_st_basis=zeros2.copy(),
            trad_balance=zeros2.copy(), roth_balance=zeros2.copy(),
            roth_basis=zeros1.copy(),
            roth_conversions=np.zeros((n_paths, horizon + 1)),
            nominal_income=np.full(n_paths, float(starting_nominal_income)),
            cumulative_inflation=np.ones(n_paths),
            failed=np.zeros(n_paths, dtype=bool),
            flex_baseline_wealth=np.zeros(n_paths),
            real_wealth=np.zeros((n_paths, horizon + 1)),
            real_taxes=np.zeros((n_paths, horizon)),
            real_target_spend=np.zeros((n_paths, horizon)),
            real_shortfall=np.zeros((n_paths, horizon)),
            real_balance_by_year=np.zeros((n_paths, horizon + 1, 3, 3)),
            real_contrib_by_year=np.zeros((n_paths, horizon, 3, 3)),
        )
        # Aggregate taxable lots into LT/ST cohorts
        for lot in portfolio.taxable.lots:
            ai = ASSET_IDX[lot.asset]
            if lot.is_long_term:
                s.tax_lt_value[:, ai] += lot.market_value
                s.tax_lt_basis[:, ai] += lot.cost_basis
            else:
                s.tax_st_value[:, ai] += lot.market_value
                s.tax_st_basis[:, ai] += lot.cost_basis
        for a, ai in ASSET_IDX.items():
            s.trad_balance[:, ai] = portfolio.traditional.value(a)
            s.roth_balance[:, ai] = portfolio.roth.value(a)
        s.roth_basis[:] = portfolio.roth.roth_basis
        s.real_wealth[:, 0] = s.total_value()
        return s

    # ------- views -------

    def total_value(self) -> np.ndarray:
        """(P,) — total portfolio nominal value."""
        return (self.tax_lt_value.sum(-1) + self.tax_st_value.sum(-1)
                + self.trad_balance.sum(-1) + self.roth_balance.sum(-1))

    def taxable_total(self) -> np.ndarray:
        """(P,)"""
        return self.tax_lt_value.sum(-1) + self.tax_st_value.sum(-1)

    def trad_total(self) -> np.ndarray:
        return self.trad_balance.sum(-1)

    def roth_total(self) -> np.ndarray:
        return self.roth_balance.sum(-1)

    def mature_conversion_basis(self, year_idx: int) -> np.ndarray:
        """(P,) — sum of conversions that have aged at least 5 years.
        A conversion in year y is mature at year y+5 (>= 5 years old)."""
        cutoff = year_idx - 4   # mature if conv_year <= year_idx - 5
        if cutoff <= 0:
            return np.zeros(self.n_paths)
        return self.roth_conversions[:, :cutoff].sum(-1)

    def green_conversion_basis(self, year_idx: int) -> np.ndarray:
        """(P,) — sum of conversions in the last 5 years (still penalty-prone
        if withdrawn pre-59.5)."""
        cutoff = max(0, year_idx - 4)
        return self.roth_conversions[:, cutoff:year_idx + 1].sum(-1)

    # ------- mutators (taxable lot age step) -------

    def age_st_to_lt(self) -> None:
        """At year-end, ST cohort becomes LT.

        Approximation: ST holdings are exactly 1 year old after one
        simulation year. Real-world ST/LT boundary is the actual purchase
        date plus one year + one day; year-step granularity rounds this. The
        effect is mild — it underestimates ST holdings by up to a year for
        deposits made late in a calendar year.
        """
        self.tax_lt_value += self.tax_st_value
        self.tax_lt_basis += self.tax_st_basis
        self.tax_st_value[:] = 0.0
        self.tax_st_basis[:] = 0.0


# ---------- helpers ----------

def deposit_taxable_st(s: VState, asset_idx: int, amount: np.ndarray) -> None:
    """Buy `amount` (P,) of asset into the ST cohort. Basis = market value."""
    s.tax_st_value[:, asset_idx] += amount
    s.tax_st_basis[:, asset_idx] += amount


def deposit_taxable_st_split(s: VState, amount: np.ndarray,
                             target_fractions: np.ndarray) -> None:
    """Distribute `amount` (P,) across assets per `target_fractions` (3,)."""
    for ai in range(N_ASSETS):
        deposit_taxable_st(s, ai, amount * target_fractions[ai])


def withdraw_taxable_for_spending(s: VState, dollars: np.ndarray
                                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Withdraw `dollars` (P,) from the taxable account.

    Order: cash first, then bonds, then stocks. Within asset, LT first.
    Vectorised across paths.

    Returns (proceeds, lt_gain_realized, st_gain_realized), each (P,).
    """
    remaining = dollars.copy()
    proceeds = np.zeros(s.n_paths)
    lt_gain = np.zeros(s.n_paths)
    st_gain = np.zeros(s.n_paths)

    # cash, bond, stock
    for ai in (2, 1, 0):
        for cohort_value, cohort_basis, gain_acc in (
            (s.tax_lt_value, s.tax_lt_basis, lt_gain),
            (s.tax_st_value, s.tax_st_basis, st_gain),
        ):
            avail = cohort_value[:, ai]
            take = np.minimum(remaining, avail)
            # Fraction of cohort sold; safe-divide
            frac = np.where(avail > 0, take / np.where(avail > 0, avail, 1.0), 0.0)
            gain = frac * (cohort_value[:, ai] - cohort_basis[:, ai])
            cohort_value[:, ai] -= take
            cohort_basis[:, ai] -= cohort_basis[:, ai] * frac
            proceeds += take
            gain_acc += gain
            remaining -= take
            if not remaining.any():
                return proceeds, lt_gain, st_gain
    return proceeds, lt_gain, st_gain


def withdraw_taxable_specific_asset(s: VState, dollars: np.ndarray,
                                    asset_idx: int
                                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sell `dollars` (P,) of a specific asset, LT first, then ST."""
    remaining = dollars.copy()
    proceeds = np.zeros(s.n_paths)
    lt_gain = np.zeros(s.n_paths)
    st_gain = np.zeros(s.n_paths)
    for cohort_value, cohort_basis, gain_acc in (
        (s.tax_lt_value, s.tax_lt_basis, lt_gain),
        (s.tax_st_value, s.tax_st_basis, st_gain),
    ):
        avail = cohort_value[:, asset_idx]
        take = np.minimum(remaining, avail)
        frac = np.where(avail > 0, take / np.where(avail > 0, avail, 1.0), 0.0)
        gain = frac * (cohort_value[:, asset_idx] - cohort_basis[:, asset_idx])
        cohort_value[:, asset_idx] -= take
        cohort_basis[:, asset_idx] -= cohort_basis[:, asset_idx] * frac
        proceeds += take
        gain_acc += gain
        remaining -= take
    return proceeds, lt_gain, st_gain


def withdraw_traditional(s: VState, dollars: np.ndarray) -> np.ndarray:
    """Withdraw `dollars` (P,) from Traditional, proportional to current
    asset balance. Returns actually withdrawn (P,)."""
    avail = s.trad_balance.sum(-1)
    take = np.minimum(dollars, avail)
    # Per-asset fraction to withdraw
    frac = np.where(avail > 0, take / np.where(avail > 0, avail, 1.0), 0.0)
    s.trad_balance -= s.trad_balance * frac[:, None]
    return take


def withdraw_roth(s: VState, dollars: np.ndarray, year_idx: int,
                  age: float
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Withdraw `dollars` (P,) from Roth.

    Returns (proceeds, taxable_ord_added, penalty_added).
      taxable_ord_added: dollars taxed as ordinary income (only earnings
        withdrawn pre-59.5).
      penalty_added: 10% early-withdrawal penalty dollars.

    Withdrawal order (IRS-ish):
      1. Roth direct contributions (basis) — always penalty-free.
      2. Mature conversions (>= 5 years old) — penalty-free.
      3. Green conversions (< 5 years old) — 10% penalty on principal
         pre-59.5; principal itself is not taxed (ordinary tax was paid at
         conversion).
      4. Earnings — pre-59.5: ordinary tax + 10% penalty; post-59.5 (and
         account >= 5 years old, which we approximate as always true): free.

    The Roth balance, broken into assets, is the physical pool. We
    reduce balance by the same dollar amount taken from any of the four
    sources so the invariant
        roth_balance.sum() == roth_basis + roth_conversions.sum() + earnings
    is preserved (with earnings >= 0).
    """
    P = s.n_paths
    remaining = dollars.copy()
    proceeds = np.zeros(P)
    ord_add = np.zeros(P)
    penalty = np.zeros(P)
    is_early = age < 59.5

    def _reduce_balance(amount: np.ndarray) -> None:
        total = s.roth_balance.sum(-1)
        frac = np.where(total > 0, amount / np.where(total > 0, total, 1.0), 0.0)
        s.roth_balance -= s.roth_balance * frac[:, None]

    # Step 1: roth_basis (direct contributions).
    take = np.minimum(remaining, s.roth_basis)
    s.roth_basis -= take
    _reduce_balance(take)
    proceeds += take
    remaining -= take

    # Step 2 + 3: traverse conversion years FIFO. Each year is mature if
    # year_idx - conv_year >= 5, else green.
    for y in range(0, year_idx + 1):
        if not (remaining > 0).any():
            break
        avail_y = s.roth_conversions[:, y]
        tk = np.minimum(remaining, avail_y)
        s.roth_conversions[:, y] -= tk
        _reduce_balance(tk)
        proceeds += tk
        remaining -= tk
        is_green = (year_idx - y) < 5
        if is_green and is_early:
            penalty += 0.10 * tk

    # Step 4: earnings. Whatever remains in the balance after subtracting
    # basis + conversions is earnings.
    earnings_avail = np.maximum(
        0.0, s.roth_balance.sum(-1)
        - (s.roth_basis + s.roth_conversions.sum(-1)))
    if (remaining > 0).any() and (earnings_avail > 0).any():
        tk = np.minimum(remaining, earnings_avail)
        _reduce_balance(tk)
        proceeds += tk
        remaining -= tk
        if is_early:
            ord_add += tk
            penalty += 0.10 * tk

    return proceeds, ord_add, penalty
