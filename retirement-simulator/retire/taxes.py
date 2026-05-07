"""US federal income tax calculations.

All bracket thresholds and limits are 2024 values. Thresholds for ordinary
income brackets, LTCG brackets, NIIT, standard deduction, and contribution
limits are dataclass-based so they can be replaced with future-year values.

This module is deliberately federal-only; state tax is handled as a flat
marginal rate on ordinary income in the simulation engine.

Implemented:
  * Ordinary income tax (progressive brackets)
  * Long-term capital gains tax (stacked on top of ordinary income)
  * Net Investment Income Tax (3.8% above MAGI thresholds)
  * Standard deduction
  * Up-to-85% taxation of Social Security benefits (provisional income method)
  * RMD divisors (Uniform Lifetime Table, 2022+)

Not implemented (intentionally; document for users):
  * AMT (rare for typical FIRE profiles)
  * QBI deduction
  * State tax brackets (use flat rate input)
  * IRMAA Medicare surcharges (can be approximated by user)
  * ACA premium tax credit cliff (cliff position is exposed for the optimizer)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

FilingStatus = Literal["single", "mfj"]


@dataclass(frozen=True)
class Bracket:
    """A single tax bracket. Upper bound is inclusive in the next bracket
    (i.e., income exactly at threshold uses the lower bracket)."""
    threshold: float  # lower bound of this bracket (in $)
    rate: float       # marginal rate as a decimal (0.22 = 22%)


@dataclass(frozen=True)
class TaxYear:
    year: int
    std_deduction: dict[str, float]
    ordinary_brackets: dict[str, list[Bracket]]
    ltcg_brackets: dict[str, list[Bracket]]
    niit_threshold: dict[str, float]  # MAGI threshold above which NIIT applies
    niit_rate: float
    contrib_limit_401k: float
    contrib_limit_401k_catchup: float  # extra at 50+
    contrib_limit_ira: float
    contrib_limit_ira_catchup: float
    ss_provisional_thresholds: dict[str, tuple[float, float]]  # (lower, upper)


# 2024 tax year. Update when filing year changes.
TAX_2024 = TaxYear(
    year=2024,
    std_deduction={"single": 14_600.0, "mfj": 29_200.0},
    ordinary_brackets={
        "single": [
            Bracket(0,         0.10),
            Bracket(11_600,    0.12),
            Bracket(47_150,    0.22),
            Bracket(100_525,   0.24),
            Bracket(191_950,   0.32),
            Bracket(243_725,   0.35),
            Bracket(609_350,   0.37),
        ],
        "mfj": [
            Bracket(0,         0.10),
            Bracket(23_200,    0.12),
            Bracket(94_300,    0.22),
            Bracket(201_050,   0.24),
            Bracket(383_900,   0.32),
            Bracket(487_450,   0.35),
            Bracket(731_200,   0.37),
        ],
    },
    ltcg_brackets={
        "single": [
            Bracket(0,         0.00),
            Bracket(47_025,    0.15),
            Bracket(518_900,   0.20),
        ],
        "mfj": [
            Bracket(0,         0.00),
            Bracket(94_050,    0.15),
            Bracket(583_750,   0.20),
        ],
    },
    niit_threshold={"single": 200_000.0, "mfj": 250_000.0},
    niit_rate=0.038,
    contrib_limit_401k=23_000.0,
    contrib_limit_401k_catchup=7_500.0,
    contrib_limit_ira=7_000.0,
    contrib_limit_ira_catchup=1_000.0,
    ss_provisional_thresholds={"single": (25_000.0, 34_000.0),
                               "mfj":    (32_000.0, 44_000.0)},
)


def progressive_tax(amount: float, brackets: list[Bracket]) -> float:
    """Tax owed on `amount` given the bracket table.

    Brackets are sorted by threshold ascending. The piecewise marginal rate
    is integrated from 0 to `amount`. Returns 0 for non-positive amounts.
    """
    if amount <= 0:
        return 0.0
    tax = 0.0
    for i, b in enumerate(brackets):
        lo = b.threshold
        hi = brackets[i + 1].threshold if i + 1 < len(brackets) else float("inf")
        if amount <= lo:
            break
        taxable_in_bracket = min(amount, hi) - lo
        tax += taxable_in_bracket * b.rate
        if amount <= hi:
            break
    return tax


def marginal_rate(amount: float, brackets: list[Bracket]) -> float:
    """Marginal rate that applies to the next dollar of `amount`."""
    rate = brackets[0].rate
    for b in brackets:
        if amount >= b.threshold:
            rate = b.rate
        else:
            break
    return rate


def ltcg_tax(ordinary_taxable: float, ltcg: float,
             brackets: list[Bracket]) -> float:
    """Capital gains stack on top of ordinary taxable income.

    The 0/15/20% LTCG brackets are filled by the *combined* income, but only
    the LTCG portion is taxed at LTCG rates. Concretely, gains in the
    [ordinary_taxable, ordinary_taxable + ltcg] slice are taxed at the LTCG
    rate corresponding to that slice's position in the LTCG bracket table.
    """
    if ltcg <= 0:
        return 0.0
    tax = 0.0
    remaining = ltcg
    pos = max(ordinary_taxable, 0.0)
    for i, b in enumerate(brackets):
        hi = brackets[i + 1].threshold if i + 1 < len(brackets) else float("inf")
        if pos >= hi:
            continue
        slice_lo = max(pos, b.threshold)
        slice_hi = hi
        in_slice = min(remaining, slice_hi - slice_lo)
        if in_slice <= 0:
            continue
        tax += in_slice * b.rate
        remaining -= in_slice
        pos += in_slice
        if remaining <= 0:
            break
    return tax


def taxable_social_security(ss_benefit: float, other_income: float,
                            tax_exempt_interest: float,
                            filing_status: FilingStatus,
                            ty: TaxYear) -> float:
    """Portion of Social Security benefits subject to ordinary income tax.

    Implements the IRS "provisional income" two-tier method:
      provisional = other_income + tax_exempt_interest + 0.5 * ss_benefit
      below lower threshold -> 0% taxable
      between thresholds    -> 50% of (provisional - lower), capped at 50% of SS
      above upper threshold -> 85% of (provisional - upper) + lesser of (prior, 50% of SS)
                               capped at 85% of SS
    """
    lo, hi = ty.ss_provisional_thresholds[filing_status]
    provisional = other_income + tax_exempt_interest + 0.5 * ss_benefit
    if provisional <= lo:
        return 0.0
    if provisional <= hi:
        return min(0.5 * (provisional - lo), 0.5 * ss_benefit)
    tier1 = min(0.5 * (hi - lo), 0.5 * ss_benefit)
    tier2 = 0.85 * (provisional - hi)
    return min(tier1 + tier2, 0.85 * ss_benefit)


def niit_owed(magi: float, net_investment_income: float,
              filing_status: FilingStatus, ty: TaxYear) -> float:
    """3.8% NIIT on the lesser of NII or MAGI excess over threshold."""
    threshold = ty.niit_threshold[filing_status]
    excess = max(0.0, magi - threshold)
    return ty.niit_rate * min(net_investment_income, excess)


@dataclass
class TaxBill:
    ordinary_income: float
    ltcg_income: float
    ss_taxable: float
    taxable_after_std_ded: float  # ordinary + ss_taxable - std_ded, floored at 0
    federal_ordinary: float
    federal_ltcg: float
    niit: float
    state: float
    total: float

    def __str__(self) -> str:
        return (
            f"Ordinary income ${self.ordinary_income:>12,.0f}  "
            f"LTCG ${self.ltcg_income:>10,.0f}  SS-tax ${self.ss_taxable:>9,.0f}\n"
            f"  Federal ordinary  ${self.federal_ordinary:>10,.0f}\n"
            f"  Federal LTCG      ${self.federal_ltcg:>10,.0f}\n"
            f"  NIIT              ${self.niit:>10,.0f}\n"
            f"  State (flat)      ${self.state:>10,.0f}\n"
            f"  TOTAL             ${self.total:>10,.0f}"
        )


def compute_tax(*, ordinary_income: float, ltcg_income: float,
                ss_benefit: float, tax_exempt_interest: float,
                filing_status: FilingStatus, state_marginal_rate: float,
                ty: TaxYear = TAX_2024) -> TaxBill:
    """Compute total federal+state tax for the year.

    Inputs:
      ordinary_income        wages + traditional withdrawals + qualified divs
                             treated as ordinary (we treat dividends as
                             qualified LTCG; pass them via ltcg_income)
      ltcg_income            realized long-term capital gains + qualified
                             dividends. Short-term gains should be added to
                             ordinary_income by the caller.
      ss_benefit             gross Social Security benefit
      tax_exempt_interest    e.g. muni bond interest (rare for this tool)
      filing_status          'single' or 'mfj'
      state_marginal_rate    flat rate applied to (ordinary_income + ss_taxable
                             + ltcg_income) above standard deduction; gross
                             approximation; users in CA/NY/etc. should set
                             this to their effective rate
    """
    ss_tax = taxable_social_security(
        ss_benefit, ordinary_income + ltcg_income, tax_exempt_interest,
        filing_status, ty,
    )
    std_ded = ty.std_deduction[filing_status]

    ordinary_taxable = max(0.0, ordinary_income + ss_tax - std_ded)

    fed_ord = progressive_tax(ordinary_taxable, ty.ordinary_brackets[filing_status])
    fed_ltcg = ltcg_tax(ordinary_taxable, ltcg_income,
                        ty.ltcg_brackets[filing_status])

    magi = ordinary_income + ss_tax + ltcg_income + tax_exempt_interest
    niit = niit_owed(magi, ltcg_income, filing_status, ty)

    state_base = max(0.0, ordinary_income + ss_tax + ltcg_income - std_ded)
    state_tax = state_base * state_marginal_rate

    total = fed_ord + fed_ltcg + niit + state_tax
    return TaxBill(
        ordinary_income=ordinary_income, ltcg_income=ltcg_income,
        ss_taxable=ss_tax, taxable_after_std_ded=ordinary_taxable,
        federal_ordinary=fed_ord, federal_ltcg=fed_ltcg,
        niit=niit, state=state_tax, total=total,
    )


# IRS Uniform Lifetime Table (2022+). Divisor used for RMDs starting at 73.
# Values are factors: required distribution = prior_year_end_balance / divisor.
RMD_DIVISORS = {
    73: 26.5, 74: 25.5, 75: 24.6, 76: 23.7, 77: 22.9, 78: 22.0, 79: 21.1,
    80: 20.2, 81: 19.4, 82: 18.5, 83: 17.7, 84: 16.8, 85: 16.0, 86: 15.2,
    87: 14.4, 88: 13.7, 89: 12.9, 90: 12.2, 91: 11.5, 92: 10.8, 93: 10.1,
    94:  9.5, 95:  8.9, 96:  8.4, 97:  7.8, 98:  7.3, 99:  6.8, 100: 6.4,
    101: 6.0, 102: 5.6, 103: 5.2, 104: 4.9, 105: 4.6, 106: 4.3, 107: 4.1,
    108: 3.9, 109: 3.7, 110: 3.5,
}
RMD_START_AGE = 73


def required_min_distribution(age: int, prior_year_end_balance: float) -> float:
    """RMD for the given age. 0 below age 73."""
    if age < RMD_START_AGE or prior_year_end_balance <= 0:
        return 0.0
    divisor = RMD_DIVISORS.get(min(age, max(RMD_DIVISORS)))
    return prior_year_end_balance / divisor


def top_of_bracket(target_marginal_rate: float, filing_status: FilingStatus,
                   ty: TaxYear = TAX_2024) -> float:
    """Income (after std deduction) at the top of the bracket whose marginal
    rate is `target_marginal_rate`. Used for Roth conversion ladders that
    'fill the 12% bracket'."""
    brackets = ty.ordinary_brackets[filing_status]
    for i, b in enumerate(brackets):
        if abs(b.rate - target_marginal_rate) < 1e-9:
            if i + 1 < len(brackets):
                return brackets[i + 1].threshold
            return float("inf")
    raise ValueError(f"No bracket with rate {target_marginal_rate}")


# IRMAA (Income-Related Monthly Adjustment Amount) — surcharges on
# Medicare Part B + Part D premiums for high-income beneficiaries. Looks
# back to MAGI from TWO years prior. Standard Part B + Part D base
# premiums plus IRMAA surcharge are an out-of-pocket household expense
# that we treat as a tax-like cost in the year it's paid (age 65+).
#
# Each tier entry is (magi_lower_threshold, total_annual_surcharge).
# Sum of Part B IRMAA + Part D IRMAA (CMS 2024 single-filer schedule),
# annualised. Below the lowest threshold the surcharge is $0 and the
# user just pays standard premiums (separately).
IRMAA_2024_SINGLE: list[tuple[float, float]] = [
    (0.0,         0.0),
    (103_000.0,   12.0 * (69.90 + 12.90)),   # +$994/yr
    (129_000.0,   12.0 * (174.70 + 33.30)),  # +$2,496/yr
    (161_000.0,   12.0 * (279.50 + 53.80)),  # +$3,999/yr
    (193_000.0,   12.0 * (384.30 + 74.20)),  # +$5,502/yr
    (500_000.0,   12.0 * (419.30 + 81.00)),  # +$6,003/yr
]
IRMAA_2024_MFJ: list[tuple[float, float]] = [
    (0.0,         0.0),
    (206_000.0,   12.0 * (69.90 + 12.90)),
    (258_000.0,   12.0 * (174.70 + 33.30)),
    (322_000.0,   12.0 * (279.50 + 53.80)),
    (386_000.0,   12.0 * (384.30 + 74.20)),
    (750_000.0,   12.0 * (419.30 + 81.00)),
]
# Medicare eligibility starts at 65; IRMAA applies from then on.
MEDICARE_AGE = 65


def irmaa_surcharge(magi_two_years_ago: float, age: float,
                    filing_status: FilingStatus) -> float:
    """Annual IRMAA surcharge in nominal dollars for a beneficiary of the
    given age, given their MAGI from two years prior (the SSA lookback
    rule). Returns 0 below age 65 or below the lowest threshold."""
    if age < MEDICARE_AGE:
        return 0.0
    schedule = (IRMAA_2024_SINGLE if filing_status == "single"
                else IRMAA_2024_MFJ)
    surcharge = 0.0
    for threshold, amount in schedule:
        if magi_two_years_ago >= threshold:
            surcharge = amount
        else:
            break
    return surcharge
