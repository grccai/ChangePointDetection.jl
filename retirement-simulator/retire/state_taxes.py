"""US state income tax for CA, WA, and OR (2024) and a date-based
multi-state timeline.

Each state's tax interacts differently with federal:
  * CA: progressive ordinary brackets; LTCG taxed *as ordinary income*
        (no preferential rate). 1% Mental Health Services Tax over $1M
        is folded into the top bracket here.
  * OR: progressive ordinary brackets; LTCG taxed as ordinary income.
  * WA: no income tax on wages or interest/dividends; flat 7% Capital
        Gains Tax on long-term gains in excess of a $262,000 standard
        deduction (2023 indexed value; close to 2024). Real estate gains
        and retirement-account distributions are exempt.

Multi-state apportionment:
  * Residency periods carry a state and explicit calendar [start, end) dates.
    Investment income (capital gains, dividends, RMDs, conversions) is taxed
    by the residency state(s) active in each simulation year, weighted by
    fraction of the year resident.
  * Income sources also carry state and [start, end) dates plus an annual
    gross amount and a growth rate. Wages from each source are taxed by the
    source's state. Multiple concurrent sources (e.g., a half-time job in
    each of two states) are summed.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .taxes import Bracket, FilingStatus, progressive_tax


@dataclass(frozen=True)
class StateTaxYear:
    state: str
    year: int
    ordinary_brackets: dict[str, list[Bracket]]
    std_deduction: dict[str, float]
    # 'ordinary' -> LTCG added to ordinary base (CA, OR)
    # 'separate_flat_threshold' -> WA-style cap-gains tax
    # 'none' -> no tax on cap gains
    ltcg_treatment: Literal["ordinary", "separate_flat_threshold", "none"]
    cgt_threshold: float = 0.0
    cgt_rate: float = 0.0


# California 2024.
CA_2024 = StateTaxYear(
    state="CA", year=2024,
    ordinary_brackets={
        "single": [
            Bracket(0,          0.0100),
            Bracket(10_756,     0.0200),
            Bracket(25_499,     0.0400),
            Bracket(40_245,     0.0600),
            Bracket(55_866,     0.0800),
            Bracket(70_606,     0.0930),
            Bracket(360_659,    0.1030),
            Bracket(432_787,    0.1130),
            Bracket(721_314,    0.1230),
            Bracket(1_000_000,  0.1330),  # 12.3% + 1% MHST
        ],
        "mfj": [
            Bracket(0,          0.0100),
            Bracket(21_512,     0.0200),
            Bracket(50_998,     0.0400),
            Bracket(80_490,     0.0600),
            Bracket(111_732,    0.0800),
            Bracket(141_212,    0.0930),
            Bracket(721_318,    0.1030),
            Bracket(865_574,    0.1130),
            Bracket(1_442_628,  0.1230),
            Bracket(2_000_000,  0.1330),
        ],
    },
    std_deduction={"single": 5_540.0, "mfj": 11_080.0},
    ltcg_treatment="ordinary",
)

# Oregon 2024.
OR_2024 = StateTaxYear(
    state="OR", year=2024,
    ordinary_brackets={
        "single": [
            Bracket(0,         0.0475),
            Bracket(4_300,     0.0675),
            Bracket(10_750,    0.0875),
            Bracket(125_000,   0.0990),
        ],
        "mfj": [
            Bracket(0,         0.0475),
            Bracket(8_600,     0.0675),
            Bracket(21_500,    0.0875),
            Bracket(250_000,   0.0990),
        ],
    },
    std_deduction={"single": 2_745.0, "mfj": 5_495.0},
    ltcg_treatment="ordinary",
)

# Washington 2024.
WA_2024 = StateTaxYear(
    state="WA", year=2024,
    ordinary_brackets={"single": [], "mfj": []},
    std_deduction={"single": 0.0, "mfj": 0.0},
    ltcg_treatment="separate_flat_threshold",
    cgt_threshold=262_000.0,
    cgt_rate=0.07,
)


STATES: dict[str, StateTaxYear] = {
    "CA": CA_2024, "OR": OR_2024, "WA": WA_2024,
    "NONE": StateTaxYear(state="NONE", year=2024,
                         ordinary_brackets={"single": [], "mfj": []},
                         std_deduction={"single": 0.0, "mfj": 0.0},
                         ltcg_treatment="none"),
}


def state_tax(*, state: str, ordinary_income: float, ltcg_income: float,
              filing_status: FilingStatus) -> float:
    """Scalar state tax. Returns dollars."""
    sty = STATES[state]
    base = max(0.0, ordinary_income - sty.std_deduction[filing_status])
    if sty.ltcg_treatment == "ordinary":
        base = max(0.0, ordinary_income + ltcg_income
                   - sty.std_deduction[filing_status])
        return progressive_tax(base, sty.ordinary_brackets[filing_status])
    if sty.ltcg_treatment == "separate_flat_threshold":
        ord_tax = progressive_tax(base, sty.ordinary_brackets[filing_status])
        cgt = max(0.0, ltcg_income - sty.cgt_threshold) * sty.cgt_rate
        return ord_tax + cgt
    if sty.ltcg_treatment == "none":
        return 0.0
    raise ValueError(f"unknown LTCG treatment: {sty.ltcg_treatment}")


# ---------- Vectorised forms ----------

def _brackets_to_arrays(brackets: list[Bracket]) -> tuple[np.ndarray, np.ndarray]:
    if not brackets:
        return np.array([0.0, np.inf]), np.array([0.0])
    thresh = np.array([b.threshold for b in brackets] + [np.inf])
    rates = np.array([b.rate for b in brackets])
    return thresh, rates


def progressive_tax_vec(taxable: np.ndarray, brackets: list[Bracket]
                        ) -> np.ndarray:
    thresh, rates = _brackets_to_arrays(brackets)
    if rates.sum() == 0:
        return np.zeros_like(taxable)
    t = np.maximum(0.0, np.asarray(taxable))
    upper = thresh[1:][None, :]
    lower = thresh[:-1][None, :]
    in_bracket = np.clip(np.minimum(t[:, None], upper) - lower, 0.0, None)
    return (in_bracket * rates[None, :]).sum(axis=-1)


def state_tax_vec(*, state: str, ordinary_income: np.ndarray,
                  ltcg_income: np.ndarray, filing_status: FilingStatus
                  ) -> np.ndarray:
    sty = STATES[state]
    sd = sty.std_deduction[filing_status]
    if sty.ltcg_treatment == "none":
        return np.zeros_like(ordinary_income)
    if sty.ltcg_treatment == "ordinary":
        base = np.maximum(0.0, ordinary_income + ltcg_income - sd)
        return progressive_tax_vec(base, sty.ordinary_brackets[filing_status])
    if sty.ltcg_treatment == "separate_flat_threshold":
        base = np.maximum(0.0, ordinary_income - sd)
        ord_tax = progressive_tax_vec(base, sty.ordinary_brackets[filing_status])
        cgt = np.maximum(0.0, ltcg_income - sty.cgt_threshold) * sty.cgt_rate
        return ord_tax + cgt
    raise ValueError(f"unknown LTCG treatment: {sty.ltcg_treatment}")


# ---------- Date-based timeline ----------

DAYS_PER_YEAR = 365.25


def _add_years(d: _dt.date, years: int) -> _dt.date:
    """Add `years` calendar years to `d`. Feb 29 in a leap year maps to
    Feb 28 in non-leap years."""
    try:
        return d.replace(year=d.year + years)
    except ValueError:
        return d.replace(year=d.year + years, day=28)


def _interval_overlap_days(a_start: _dt.date, a_end: _dt.date,
                            b_start: _dt.date, b_end: _dt.date) -> int:
    """Days of overlap between [a_start, a_end) and [b_start, b_end)."""
    lo = max(a_start, b_start)
    hi = min(a_end, b_end)
    delta = (hi - lo).days
    return max(0, delta)


@dataclass
class ResidencyPeriod:
    """Where the taxpayer lives for tax purposes during [start, end)."""
    state: str
    start: _dt.date
    end: _dt.date


@dataclass
class IncomeSource:
    """A wage source. `gross_annual` is the year-1 (start-of-source) annual
    gross. The amount grows at `growth_rate` (nominal, compounded annually)
    measured from `start`.

    `state` is the state of employment that taxes the wages from this source.
    Use 'NONE' to exempt this source from state tax (e.g., contract income
    classified differently)."""
    state: str
    start: _dt.date
    end: _dt.date
    gross_annual: float
    growth_rate: float = 0.0


@dataclass
class StateTimeline:
    residency: list[ResidencyPeriod] = field(default_factory=list)
    income_sources: list[IncomeSource] = field(default_factory=list)

    # ----- year-window helpers -----

    def year_window(self, sim_start: _dt.date, year_idx: int
                    ) -> tuple[_dt.date, _dt.date]:
        """[start, end) for simulation year `year_idx` (anchored to sim_start)."""
        return (_add_years(sim_start, year_idx),
                _add_years(sim_start, year_idx + 1))

    def residency_weights(self, sim_start: _dt.date, year_idx: int
                           ) -> dict[str, float]:
        """state -> fraction of year resident in that state. Sum may be < 1
        if the timeline has gaps (those periods incur no state tax)."""
        ws, we = self.year_window(sim_start, year_idx)
        out: dict[str, float] = {}
        year_len = max(1, (we - ws).days)
        for r in self.residency:
            ovl = _interval_overlap_days(ws, we, r.start, r.end)
            if ovl > 0:
                out[r.state] = out.get(r.state, 0.0) + ovl / year_len
        return out

    def wages_by_state(self, sim_start: _dt.date, year_idx: int
                        ) -> dict[str, float]:
        """state -> nominal wages active in this state during this year.
        Wages from a source are pro-rated by fraction-of-year overlap and
        grown from the source's own start date at the source's growth rate.
        """
        ws, we = self.year_window(sim_start, year_idx)
        out: dict[str, float] = {}
        year_len = max(1, (we - ws).days)
        for src in self.income_sources:
            ovl_lo = max(ws, src.start)
            ovl_hi = min(we, src.end)
            ovl = (ovl_hi - ovl_lo).days
            if ovl <= 0:
                continue
            # Grow from source.start to mid-overlap
            mid_days = (ovl_lo - src.start).days + ovl // 2
            growth_yrs = max(0.0, mid_days / DAYS_PER_YEAR)
            grown = src.gross_annual * (1.0 + src.growth_rate) ** growth_yrs
            wages = grown * (ovl / year_len)
            out[src.state] = out.get(src.state, 0.0) + wages
        return out

    def total_wages(self, sim_start: _dt.date, year_idx: int) -> float:
        return sum(self.wages_by_state(sim_start, year_idx).values())

    def has_wages(self, sim_start: _dt.date, year_idx: int) -> bool:
        return self.total_wages(sim_start, year_idx) > 0


# ---------- Apportioned tax helpers ----------

def state_wages_tax(*, wages_by_state: dict[str, float],
                    pretax_401k: float, filing_status: FilingStatus) -> float:
    """Scalar tax owed on wages across all employment states.

    Pretax 401k contributions are apportioned across employment states by
    each state's share of total wages, reducing that state's taxable wage
    base. (This is the same approach the IRS Form W-2 + state nonresident
    forms use for split-state employees, simplified.)"""
    total = sum(wages_by_state.values())
    if total <= 0:
        return 0.0
    out = 0.0
    for state, w in wages_by_state.items():
        pretax_share = pretax_401k * (w / total)
        taxable_w = max(0.0, w - pretax_share)
        out += state_tax(state=state, ordinary_income=taxable_w,
                         ltcg_income=0.0, filing_status=filing_status)
    return out


def state_residency_tax_vec(*, residency_weights: dict[str, float],
                            ordinary_other: np.ndarray, ltcg: np.ndarray,
                            filing_status: FilingStatus) -> np.ndarray:
    """Vectorised tax on investment income (and any non-wage ordinary income
    such as RMDs, traditional withdrawals, Roth conversions) apportioned by
    residency."""
    P = ordinary_other.shape[0]
    out = np.zeros(P)
    for state, w in residency_weights.items():
        out += w * state_tax_vec(state=state, ordinary_income=ordinary_other,
                                 ltcg_income=ltcg, filing_status=filing_status)
    return out
