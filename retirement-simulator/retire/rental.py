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
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                     np.ndarray, np.ndarray, np.ndarray]:
    """Sample annual rental shocks. Returns six (P, H) arrays:
      property_returns_real    : real property appreciation, correlated
                                  with stock + bond log-returns.
      rent_shocks              : N(0, rent_shock_vol) multiplicative.
      turnover_events          : Bernoulli(turnover.annual_prob) — 0/1.
      capex_events             : Bernoulli(capex.annual_prob) — 0/1.
      capex_magnitude_frac     : if event fires, fraction of property
                                  value lost (lognormal w/ specified mean
                                  and sigma); zero otherwise.
      market_rate_path         : (P, H) annual nominal 30-year market
                                  mortgage rate sampled from AR(1) on
                                  log-rate around refinance.market_rate_mean.

    Property log-return is correlated with stock and bond log returns:
      z_stock, z_bond = standardized log-returns (using GBM-implied
                        moments — works for both GBM and historical
                        bootstrap, where realised log returns are folded
                        in directly)
      log_property = mu_p_log + sd_p_log * (rho_s * z_s + rho_b * z_b
                                              + sqrt(...) * z_indep)
    Variance of the convex combination is normalised to 1 using the
    stock-bond correlation (`rho_sb`) from MarketConfig.
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

    # Tenant-turnover: per-(path, year) Bernoulli. The event-cost shape is
    # computed inside step_rental_year using the event flag.
    turnover_events = rng.binomial(
        1, max(0.0, min(1.0, rp.turnover.annual_prob)), size=(P, H))

    # Capex shocks: Bernoulli arrival × lognormal magnitude (as a fraction
    # of property_value_real). Lognormal calibrated so E[X] = mean_frac:
    #    X = exp(mu + sigma*Z),  E[X] = exp(mu + sigma^2/2) = mean_frac
    #    => mu = log(mean_frac) - sigma^2/2
    capex_events = rng.binomial(
        1, max(0.0, min(1.0, rp.capex.annual_prob)), size=(P, H))
    cx_mean = max(1e-9, rp.capex.mean_frac)
    cx_sigma = max(1e-9, rp.capex.lognormal_sigma)
    cx_mu = np.log(cx_mean) - 0.5 * cx_sigma ** 2
    capex_log_mag = rng.normal(cx_mu, cx_sigma, size=(P, H))
    capex_magnitude_frac = capex_events * np.exp(capex_log_mag)

    # Market mortgage-rate path: AR(1) on log-rate around the configured
    # long-run mean. Sampled even if refinance.enabled=False (it's cheap
    # and keeps the array shape stable for the simulator).
    rf = rp.refinance
    log_mean = np.log(max(1e-4, rf.market_rate_mean))
    alpha = max(0.0, min(1.0, rf.market_rate_ar1_alpha))
    log_vol = max(0.0, rf.market_rate_log_vol)
    log_rate = np.zeros((P, H))
    log_rate[:, 0] = log_mean   # start at the long-run mean
    innovations = rng.normal(0.0, log_vol, size=(P, H))
    for t in range(1, H):
        log_rate[:, t] = (log_mean
                           + alpha * (log_rate[:, t - 1] - log_mean)
                           + innovations[:, t])
    market_rate_path = np.exp(log_rate)

    return (arith_p, rent_shocks, turnover_events,
            capex_events, capex_magnitude_frac, market_rate_path)


def annuity_payment_nominal(loan_n: np.ndarray, rate, term_years: int
                             ) -> np.ndarray:
    """Standard fixed-rate annuity payment. Vectorised over (P,). `rate`
    accepts either a scalar (single locked rate at origination) or a (P,)
    array (per-path locked rate after a refinance)."""
    rate_arr = np.asarray(rate)
    if rate_arr.ndim == 0:
        if float(rate_arr) <= 0:
            return loan_n / term_years
        f = (1.0 + float(rate_arr)) ** term_years
        return loan_n * float(rate_arr) * f / (f - 1.0)
    f = (1.0 + rate_arr) ** term_years
    pmt = np.where(rate_arr > 0,
                    loan_n * rate_arr * f / np.where(f != 1.0, f - 1.0, 1.0),
                    loan_n / term_years)
    return pmt


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
    s.mortgage_rate_nominal = np.where(buyers, rp.mortgage_nominal_rate,
                                        s.mortgage_rate_nominal)
    s.mortgage_term_years_remaining = np.where(
        buyers, rp.mortgage_term_years,
        s.mortgage_term_years_remaining).astype(np.int32)
    # Refi cooldown starts at zero (immediately eligible) but only fires
    # if the rate-drop threshold is hit; in practice that's rare in the
    # year of purchase since the locked rate equals the market rate at
    # that moment for newly originated loans.
    s.refi_cooldown = np.where(buyers, 0,
                                s.refi_cooldown).astype(np.int32)
    dp_real = np.zeros(s.n_paths)
    dp_real[buyers] = rp.price_real * rp.downpayment_frac
    return dp_real


def step_rental_year(s: VState, rp: RentalProperty,
                     property_return_real: np.ndarray | None = None,
                     rent_shock: np.ndarray | None = None,
                     turnover_event: np.ndarray | None = None,
                     capex_magnitude_frac: np.ndarray | None = None,
                     market_rate: np.ndarray | None = None,
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advance rental property by one year, in-place mutation of mortgage
    + heloc balances + property value.

    Per-(path, year) shocks are sampled by the caller (see
    `sample_rental_paths`). All optional; defaults are deterministic
    (no shocks).

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

    # ----- 1) Property appreciation -----
    if property_return_real is None:
        ret_real = np.full(P, rp.appreciation_real_mean)
    else:
        ret_real = property_return_real
    s.rental_value_real = np.where(
        owned,
        s.rental_value_real * (1.0 + ret_real),
        s.rental_value_real,
    )

    # ----- 2) Refinance check -----
    # Per-path nominal market 30y mortgage rate this year. Refi fires
    # when (locked - market) >= threshold AND cooldown has expired.
    refi_closing_cost_real = np.zeros(P)
    if rp.refinance.enabled and market_rate is not None and owned.any():
        rf = rp.refinance
        eligible = owned & (s.refi_cooldown <= 0) & \
                   ((s.mortgage_rate_nominal - market_rate)
                    >= rf.rate_drop_threshold) & \
                   (s.mortgage_balance_nominal > 1.0)
        if eligible.any():
            # Closing costs paid out-of-pocket this year (not rolled in).
            closing_n = rf.closing_cost_frac * s.mortgage_balance_nominal
            refi_closing_cost_real = np.where(
                eligible, closing_n / cum_infl, 0.0)
            # Re-amortize remaining balance at market rate over a fresh
            # `new_term_years` term.
            new_pmt_n = annuity_payment_nominal(
                s.mortgage_balance_nominal, market_rate, rf.new_term_years)
            s.mortgage_rate_nominal = np.where(
                eligible, market_rate, s.mortgage_rate_nominal)
            s.mortgage_payment_nominal = np.where(
                eligible, new_pmt_n, s.mortgage_payment_nominal)
            s.mortgage_term_years_remaining = np.where(
                eligible, rf.new_term_years,
                s.mortgage_term_years_remaining).astype(np.int32)
            s.refi_cooldown = np.where(
                eligible, rf.cooldown_years,
                np.maximum(0, s.refi_cooldown - 1)).astype(np.int32)
        else:
            s.refi_cooldown = np.maximum(0, s.refi_cooldown - 1).astype(np.int32)
    else:
        # Even if refi is disabled, decrement any stale cooldown counters.
        if owned.any():
            s.refi_cooldown = np.maximum(0, s.refi_cooldown - 1).astype(np.int32)

    # ----- 3) Mortgage step (using current locked rate) -----
    interest_n = s.mortgage_balance_nominal * s.mortgage_rate_nominal
    pay_n = np.minimum(s.mortgage_payment_nominal,
                       s.mortgage_balance_nominal + interest_n)
    principal_n = np.clip(pay_n - interest_n, 0.0,
                          s.mortgage_balance_nominal)
    new_balance_n = s.mortgage_balance_nominal - principal_n
    s.mortgage_balance_nominal = np.where(owned, new_balance_n,
                                           s.mortgage_balance_nominal)
    paid_off = owned & (s.mortgage_balance_nominal <= 1e-6)
    s.mortgage_payment_nominal = np.where(paid_off, 0.0,
                                           s.mortgage_payment_nominal)
    interest_n = np.where(owned, interest_n, 0.0)
    pay_n = np.where(owned, pay_n, 0.0)
    s.mortgage_term_years_remaining = np.where(
        owned, np.maximum(0, s.mortgage_term_years_remaining - 1),
        s.mortgage_term_years_remaining).astype(np.int32)

    # ----- 4) HELOC interest -----
    heloc_int_n = s.heloc_balance_nominal * rp.heloc_rate_nominal
    s.heloc_balance_nominal = s.heloc_balance_nominal + heloc_int_n
    heloc_int_n = np.where(owned, heloc_int_n, 0.0)

    # ----- 5) Operating costs and NOI -----
    interest_real = interest_n / cum_infl
    payment_real = pay_n / cum_infl
    heloc_int_real = heloc_int_n / cum_infl

    # Effective rent multiplier: cap_rate * (1 + rent_shock), then cut by
    # the turnover-vacancy fraction (months_vacant / 12 if turnover this
    # year, else 0).
    if rent_shock is None:
        rent_mult = 1.0
    else:
        rent_mult = (1.0 + rent_shock)
    if turnover_event is None or rp.turnover.annual_prob <= 0:
        turnover_vacancy_frac = 0.0
        turnover_cost_real = np.zeros(P)
    else:
        vac_frac = max(0.0, min(1.0, rp.turnover.months_vacant / 12.0))
        turnover_vacancy_frac = turnover_event * vac_frac
        # One-time turnover cost = cost_frac × annual gross rent (using the
        # un-shocked cap rate as the baseline reference rent).
        turnover_cost_real = (turnover_event * rp.turnover.cost_frac
                               * rp.cap_rate * s.rental_value_real)
    gross_rent_real = (rp.cap_rate * rent_mult
                        * (1.0 - turnover_vacancy_frac)
                        * s.rental_value_real)
    gross_rent_real = np.where(owned, gross_rent_real, zeros)

    # Granular operating costs. The legacy `expense_ratio` lump is
    # additive on top so old YAMLs (with expense_ratio=0.025 and the
    # granular rates at 0) keep their behaviour.
    cost_frac = (rp.expense_ratio + rp.property_tax_rate
                  + rp.insurance_rate + rp.maintenance_rate)
    fixed_costs_real = cost_frac * s.rental_value_real
    mgmt_fee_real = rp.management_fee_frac * gross_rent_real
    legal_ins_real = np.where(owned, rp.legal_insurance_real, 0.0)

    # Capex shock (already a fraction of property_value with the lognormal
    # magnitude pre-applied; zero on non-event years).
    if capex_magnitude_frac is None:
        capex_cost_real = np.zeros(P)
    else:
        capex_cost_real = np.where(owned,
                                    capex_magnitude_frac * s.rental_value_real,
                                    zeros)

    # Refi closing costs accrue this year (already in real $).
    refi_closing_cost_real = np.where(owned, refi_closing_cost_real, 0.0)

    operating_costs_real = (fixed_costs_real + mgmt_fee_real
                             + legal_ins_real + turnover_cost_real
                             + capex_cost_real + refi_closing_cost_real)
    noi_real = np.where(owned, gross_rent_real - operating_costs_real, zeros)

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
