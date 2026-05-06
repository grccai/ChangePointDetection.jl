"""Rental property mechanics.

This module contains the per-year rental property step and the wealth-
conditional purchase trigger. The simulator calls into here once per
simulation year; everything else (taxable account waterfall, federal/
state tax, ruin determination) lives in `simulate.py` / `state_taxes.py`.

Design notes:
  * `RentalProperty` is a single specimen shared across paths; per-path
    state lives on `VState` (rental_owned, rental_value_real,
    mortgage_balance_nominal, mortgage_payment_nominal, heloc_balance_nominal).
  * Property value is tracked in REAL dollars and grown by
    `appreciation_real` each year. Convert to nominal by multiplying by
    `cumulative_inflation` when needed (e.g., for LTV / equity).
  * Mortgage is tracked in NOMINAL dollars with a fixed nominal rate +
    locked annuity payment. Real P&I per year = nominal_payment /
    cumulative_inflation, so inflation eats the real payment automatically.
  * Mortgage interest is deductible against rental income; depreciation
    is NOT modelled (this would be a small additional tax shield).
  * HELOC is a backstop, not free liquidity: drawn only when the
    simulator would otherwise mark a path failed; balance accrues at
    `heloc_rate_nominal` and crowds out future cash flow.
"""
from __future__ import annotations

import numpy as np

from .accounts import Asset
from .config import MarketConfig, RentalProperty
from .vstate import VState


def sample_rental_paths(R: np.ndarray, market: MarketConfig,
                         rp: RentalProperty,
                         seed: int | None = None
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Sample (P, H) annual real property returns + (P, H) rent shocks.

    Property log-return is correlated with stock and bond log returns:
      z_stock, z_bond = standardized log-returns (using GBM-implied
                        moments — works for both GBM and historical
                        bootstrap, where realised log returns are folded
                        in directly)
      log_property = mu_p_log + sd_p_log * (rho_s * z_s + rho_b * z_b
                                              + sqrt(...) * z_indep)
    Variance of the convex combination is normalised to 1 using the
    stock-bond correlation (`rho_sb`) from MarketConfig.

    Rent shock is independent N(0, rent_shock_vol) per (path, year).
    """
    P, H, _ = R.shape
    eps = 1e-9
    rng = np.random.default_rng(seed)
    log_s = np.log(np.maximum(eps, 1.0 + R[:, :, 0]))
    log_b = np.log(np.maximum(eps, 1.0 + R[:, :, 1]))
    # GBM-implied log moments for standardisation
    from .returns import _arith_to_log
    mu_s_log, sd_s_log = _arith_to_log(market.stocks.real_return,
                                         market.stocks.vol)
    mu_b_log, sd_b_log = _arith_to_log(market.bonds.real_return,
                                         market.bonds.vol)
    sd_s_log = max(eps, sd_s_log)
    sd_b_log = max(eps, sd_b_log)
    z_s = (log_s - mu_s_log) / sd_s_log
    z_b = (log_b - mu_b_log) / sd_b_log
    rho_s = float(rp.correlation_with_stock)
    rho_b = float(rp.correlation_with_bond)
    rho_sb = float(market.correlation_stock_bond)
    cross = 2.0 * rho_s * rho_b * rho_sb
    var_indep = 1.0 - rho_s ** 2 - rho_b ** 2 - cross
    if var_indep < 1e-6:
        # Shrink toward feasibility while preserving sign + ratio
        denom = (rho_s ** 2 + rho_b ** 2 + abs(cross)) or 1.0
        scale = float(np.sqrt(max(0.0, (1.0 - 1e-3) / denom)))
        rho_s *= scale
        rho_b *= scale
        cross = 2.0 * rho_s * rho_b * rho_sb
        var_indep = max(1e-6, 1.0 - rho_s ** 2 - rho_b ** 2 - cross)
    z_indep = rng.standard_normal((P, H))
    z_p = rho_s * z_s + rho_b * z_b + np.sqrt(var_indep) * z_indep
    mu_p_log, sd_p_log = _arith_to_log(rp.appreciation_real_mean,
                                          max(eps, rp.appreciation_real_vol))
    log_p = mu_p_log + sd_p_log * z_p
    arith_p = np.exp(log_p) - 1.0

    rent_shocks = rng.normal(0.0, max(0.0, rp.rent_shock_vol), size=(P, H))
    return arith_p, rent_shocks


def annuity_payment_nominal(loan_n: np.ndarray, rate: float,
                              term_years: int) -> np.ndarray:
    """Standard fixed-rate annuity payment. Vectorised over (P,)."""
    if rate <= 0:
        return loan_n / term_years
    f = (1 + rate) ** term_years
    return loan_n * rate * f / (f - 1.0)


def downpayment_real(rp: RentalProperty) -> float:
    return rp.price_real * rp.downpayment_frac


def trigger_fires(s: VState, rp: RentalProperty, age: float,
                   liquid_real_wealth: np.ndarray,
                   taxable_real_wealth: np.ndarray) -> np.ndarray:
    """(P,) bool — paths for whom the purchase trigger fires this year.

    Conditions: not yet owned AND age >= min_age AND liquid >= min_liquid
    AND taxable >= max(min_taxable, downpayment) (so the downpayment can
    actually be funded from taxable without dipping into 401k/Roth).
    """
    tr = rp.trigger
    dp_real = downpayment_real(rp)
    needed_taxable = max(tr.min_taxable_real_wealth, dp_real)
    return (
        (~s.rental_owned)
        & (age >= tr.min_age)
        & (liquid_real_wealth >= tr.min_liquid_real_wealth)
        & (taxable_real_wealth >= needed_taxable)
    )


def execute_purchase(s: VState, rp: RentalProperty,
                      buyers: np.ndarray, year_idx: int) -> np.ndarray:
    """For each path in `buyers`: lock in mortgage, set property_value_real,
    mark rental_owned=True. Returns (P,) real $ that must be withdrawn from
    the taxable account using the simulator's normal waterfall (caller is
    responsible for that withdrawal — and any cap-gains tax it produces).
    """
    if not buyers.any():
        return np.zeros(s.n_paths)
    cum_infl = s.cumulative_inflation
    price_n = rp.price_real * cum_infl
    loan_n = price_n * (1.0 - rp.downpayment_frac)
    pmt_n = annuity_payment_nominal(loan_n, rp.mortgage_nominal_rate,
                                     rp.mortgage_term_years)
    s.rental_owned |= buyers
    s.rental_value_real = np.where(buyers, rp.price_real, s.rental_value_real)
    s.mortgage_balance_nominal = np.where(buyers, loan_n,
                                           s.mortgage_balance_nominal)
    s.mortgage_payment_nominal = np.where(buyers, pmt_n,
                                           s.mortgage_payment_nominal)
    dp_real = np.zeros(s.n_paths)
    dp_real[buyers] = rp.price_real * rp.downpayment_frac
    return dp_real


def step_rental_year(s: VState, rp: RentalProperty,
                     property_return_real: np.ndarray | None = None,
                     rent_shock: np.ndarray | None = None,
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advance rental property by one year, in-place mutation of mortgage
    + heloc balances + property value.

    `property_return_real` and `rent_shock` are (P,) arrays sampled by the
    caller (see `sample_rental_paths` in this module). If omitted, the
    property step is deterministic (return = appreciation_real_mean,
    no rent shock).

    Returns three (P,) arrays in REAL dollars (zeros for non-owners):
      taxable_rental_income_real :  NOI - mortgage_interest (Schedule-E)
      net_cash_flow_real         :  NOI - mortgage_payment - heloc_interest
                                    (pre federal/state tax)
      mortgage_interest_real     :  for diagnostics

    Caller is responsible for:
      - federal+state tax on `taxable_rental_income_real` (state =
        rp.location_state, source-based)
      - depositing post-tax `net_cash_flow_real` into taxable cash sleeve
        (or, if negative, debiting taxable / drawing HELOC)
    """
    P = s.n_paths
    owned = s.rental_owned
    cum_infl = s.cumulative_inflation
    zeros = np.zeros(P)

    # Property appreciation (real). Draw from sampled returns if provided.
    if property_return_real is None:
        ret_real = np.full(P, rp.appreciation_real_mean)
    else:
        ret_real = property_return_real
    s.rental_value_real = np.where(
        owned,
        s.rental_value_real * (1.0 + ret_real),
        s.rental_value_real,
    )

    # Mortgage step (nominal). Interest first, then principal, capped at balance.
    interest_n = s.mortgage_balance_nominal * rp.mortgage_nominal_rate
    pay_n = np.minimum(s.mortgage_payment_nominal,
                       s.mortgage_balance_nominal + interest_n)
    principal_n = np.clip(pay_n - interest_n, 0.0,
                          s.mortgage_balance_nominal)
    new_balance_n = s.mortgage_balance_nominal - principal_n
    s.mortgage_balance_nominal = np.where(owned, new_balance_n,
                                           s.mortgage_balance_nominal)
    # When the loan is fully paid the locked nominal payment becomes zero
    # going forward.
    paid_off = owned & (s.mortgage_balance_nominal <= 1e-6)
    s.mortgage_payment_nominal = np.where(paid_off, 0.0,
                                           s.mortgage_payment_nominal)
    interest_n = np.where(owned, interest_n, 0.0)
    pay_n = np.where(owned, pay_n, 0.0)

    # HELOC interest (nominal). Pure interest accrual; principal is paid down
    # implicitly when net cash flow is positive (handled in caller).
    heloc_int_n = s.heloc_balance_nominal * rp.heloc_rate_nominal
    s.heloc_balance_nominal = s.heloc_balance_nominal + heloc_int_n
    heloc_int_n = np.where(owned, heloc_int_n, 0.0)

    # Convert to real for cash-flow accounting
    interest_real = interest_n / cum_infl
    payment_real = pay_n / cum_infl
    heloc_int_real = heloc_int_n / cum_infl
    if rent_shock is None:
        cap_eff = rp.cap_rate
    else:
        cap_eff = rp.cap_rate * (1.0 + rent_shock)
    noi_real = np.where(
        owned,
        (cap_eff - rp.expense_ratio) * s.rental_value_real,
        zeros,
    )
    net_cash_flow_real = noi_real - payment_real - heloc_int_real
    taxable_rental_income_real = noi_real - interest_real
    return (taxable_rental_income_real, net_cash_flow_real, interest_real)


def accessible_equity_real(s: VState, rp: RentalProperty) -> np.ndarray:
    """(P,) real $ of HELOC capacity remaining."""
    cum_infl = s.cumulative_inflation
    value_n = s.rental_value_real * cum_infl
    cap = rp.ltv_max * value_n - s.mortgage_balance_nominal \
          - s.heloc_balance_nominal
    cap_real = np.maximum(0.0, cap) / cum_infl
    return np.where(s.rental_owned, cap_real, 0.0)


def repay_heloc_real(s: VState, repay_real: np.ndarray) -> np.ndarray:
    """Apply a (P,) array of real-$ HELOC paydowns. Returns the actual
    real-$ amount applied (capped by balance)."""
    cum_infl = s.cumulative_inflation
    repay_n = np.maximum(0.0, repay_real) * cum_infl
    repay_n = np.minimum(repay_n, s.heloc_balance_nominal)
    s.heloc_balance_nominal -= repay_n
    return repay_n / cum_infl


def draw_heloc_real(s: VState, draw_real: np.ndarray,
                     rp: RentalProperty) -> np.ndarray:
    """Try to draw `draw_real` (P,) real-$ from HELOC, capped by accessible
    equity. Mutates `heloc_balance_nominal`. Returns actual real-$ drawn."""
    cap_real = accessible_equity_real(s, rp)
    drawn_real = np.minimum(np.maximum(0.0, draw_real), cap_real)
    s.heloc_balance_nominal += drawn_real * s.cumulative_inflation
    return drawn_real
