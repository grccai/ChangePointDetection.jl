"""US state income tax for CA, WA, and OR (2024).

Each state's tax interacts differently with federal:
  * CA: progressive ordinary brackets; LTCG taxed *as ordinary income*
        (no preferential rate). 1% Mental Health Services Tax over $1M
        is folded into the top bracket here.
  * OR: progressive ordinary brackets; LTCG taxed as ordinary income.
  * WA: no income tax on wages or interest/dividends; flat 7% Capital
        Gains Tax on long-term gains in excess of a $262,000 standard
        deduction (2023 indexed value; close to 2024). Real estate gains
        and retirement-account distributions are exempt.

Multi-state allocation:
  * Earned income (wages) is taxed by the state of *employment*.
  * Investment income (capital gains, dividends, interest, retirement
    distributions taxable to that state) is taxed by the state of
    *residence*.
  * Both timelines support multiple concurrent entries with weights, so
    you can model a year of half-time work in two states.

This module's vectorised computation is what the simulator uses; the scalar
form is exposed for tests and for the CLI's one-off `tax` command.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .taxes import Bracket, FilingStatus, progressive_tax


@dataclass(frozen=True)
class StateTaxYear:
    state: str
    year: int
    # Empty list -> no income tax in this state
    ordinary_brackets: dict[str, list[Bracket]]
    std_deduction: dict[str, float]
    # 'ordinary' -> LTCG added to ordinary base (CA, OR)
    # 'separate_flat_threshold' -> WA-style cap-gains tax
    # 'none' -> no tax on cap gains (most no-income-tax states)
    ltcg_treatment: Literal["ordinary", "separate_flat_threshold", "none"]
    cgt_threshold: float = 0.0  # for separate_flat_threshold
    cgt_rate: float = 0.0


# California 2024. The 13.3% top is 12.3% bracket + 1% Mental Health Services
# Tax. Single thresholds; MFJ brackets are (almost exactly) doubled.
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

# Washington 2024. No income tax on wages or ordinary investment income.
# Capital Gains Tax on LT gains > $262,000 (real estate, retirement-account
# distributions, and a few other categories are exempt — we don't model
# those exemptions; assume CGT applies to all simulated LT gains).
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


# ---------- Vectorised forms (used by simulator) ----------

def _brackets_to_arrays(brackets: list[Bracket]) -> tuple[np.ndarray, np.ndarray]:
    """Return (thresholds_with_inf, rates). Thresholds shape (n+1,), rates (n,)."""
    if not brackets:
        return np.array([0.0, np.inf]), np.array([0.0])
    thresh = np.array([b.threshold for b in brackets] + [np.inf])
    rates = np.array([b.rate for b in brackets])
    return thresh, rates


def progressive_tax_vec(taxable: np.ndarray, brackets: list[Bracket]
                        ) -> np.ndarray:
    """Vectorised progressive tax. `taxable` is a 1D array of incomes;
    returns array of same shape with tax owed."""
    thresh, rates = _brackets_to_arrays(brackets)
    if rates.sum() == 0:
        return np.zeros_like(taxable)
    # For each bracket i: amount = clip(min(taxable, thresh[i+1]) - thresh[i], 0, inf)
    # Sum over i of amount[i] * rates[i].
    t = np.maximum(0.0, np.asarray(taxable))
    upper = thresh[1:][None, :]      # (1, n)
    lower = thresh[:-1][None, :]     # (1, n)
    in_bracket = np.clip(np.minimum(t[:, None], upper) - lower, 0.0, None)
    return (in_bracket * rates[None, :]).sum(axis=-1)


def state_tax_vec(*, state: str, ordinary_income: np.ndarray,
                  ltcg_income: np.ndarray, filing_status: FilingStatus
                  ) -> np.ndarray:
    """Vectorised state tax. Returns array of same shape as inputs."""
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


# ---------- Multi-state timeline ----------

@dataclass
class StateAssignment:
    """One entry on a residency or employment timeline. Weights need not sum
    to 1; they're normalised at evaluation time within (start_age, end_age)."""
    state: str
    start_age: int
    end_age: int   # exclusive
    weight: float = 1.0


@dataclass
class StateTimeline:
    """Residency = where you live (taxes investment income).
    Employment = where you work (taxes wages)."""
    residency: list[StateAssignment] = field(default_factory=list)
    employment: list[StateAssignment] = field(default_factory=list)

    def at_age(self, age: int, kind: Literal["residency", "employment"]
               ) -> dict[str, float]:
        """Return {state: weight} active at this age, normalised to sum 1.
        Empty dict if nothing active (caller treats as no state tax)."""
        items = self.residency if kind == "residency" else self.employment
        active = [a for a in items if a.start_age <= age < a.end_age]
        total_w = sum(a.weight for a in active)
        if total_w <= 0:
            return {}
        out: dict[str, float] = {}
        for a in active:
            out[a.state] = out.get(a.state, 0.0) + a.weight / total_w
        return out


def multi_state_tax(*, ordinary_income_wages: float, ordinary_income_other: float,
                    ltcg_income: float, age: int, filing_status: FilingStatus,
                    timeline: StateTimeline) -> float:
    """Scalar multi-state tax for a single year.

    Wages (`ordinary_income_wages`) follow the *employment* timeline.
    Other ordinary income (RMDs, traditional withdrawals, dividends-as-ordinary,
    Roth conversions) and capital gains follow the *residency* timeline.
    """
    res = timeline.at_age(age, "residency")
    emp = timeline.at_age(age, "employment")
    total = 0.0
    for state, w in emp.items():
        # Apportion only wages to this state. Other income may be small at
        # employment-only weight; standard simplification: treat wages-only
        # taxable in employment state.
        total += w * state_tax(state=state, ordinary_income=ordinary_income_wages,
                               ltcg_income=0.0, filing_status=filing_status)
    for state, w in res.items():
        total += w * state_tax(state=state,
                               ordinary_income=ordinary_income_other,
                               ltcg_income=ltcg_income,
                               filing_status=filing_status)
    return total


def multi_state_tax_vec(*, ordinary_income_wages: np.ndarray,
                        ordinary_income_other: np.ndarray,
                        ltcg_income: np.ndarray, age: int,
                        filing_status: FilingStatus,
                        timeline: StateTimeline) -> np.ndarray:
    """Vectorised version of multi_state_tax. All income inputs are (P,)."""
    res = timeline.at_age(age, "residency")
    emp = timeline.at_age(age, "employment")
    P = ordinary_income_wages.shape[0]
    total = np.zeros(P)
    zeros = np.zeros(P)
    for state, w in emp.items():
        total += w * state_tax_vec(state=state, ordinary_income=ordinary_income_wages,
                                   ltcg_income=zeros, filing_status=filing_status)
    for state, w in res.items():
        total += w * state_tax_vec(state=state,
                                   ordinary_income=ordinary_income_other,
                                   ltcg_income=ltcg_income,
                                   filing_status=filing_status)
    return total
