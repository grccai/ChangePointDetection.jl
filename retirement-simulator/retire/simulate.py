"""Vectorised year-by-year retirement simulation.

All Monte Carlo paths advance in lock-step inside numpy ops. Per-path Python
loops are gone except for the H year-step loop (which is unavoidable because
each year's state depends on the prior year's). For H = 60 and P = 5000
this runs in well under a second on a single core.

The order of operations in each year is documented in the README.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .accounts import Asset
from .config import (Scenario, Allocation, TargetAllocations,
                     WithdrawalPolicy, Spending)
from .policy import (Policy, StaticPolicy, Decision, StateSummary)
from .returns import (sample_gbm_paths, sample_inflation,
                      sample_deterministic_paths,
                      sample_deterministic_inflation,
                      sample_historical_paths,
                      sample_historical_ath_paths,
                      sample_historical_stretched_ath_paths)
from . import rental as _rental
from .state_taxes import (StateTimeline, state_tax_vec, state_wages_tax,
                          state_residency_tax_vec)
from .taxes import (TAX_2024, TaxYear, FilingStatus, Bracket,
                    progressive_tax, ltcg_tax, taxable_social_security,
                    niit_owed, required_min_distribution, top_of_bracket,
                    RMD_DIVISORS, RMD_START_AGE)
from .vstate import (VState, ASSET_IDX, ASSET_ORDER, N_ASSETS,
                     deposit_taxable_st, deposit_taxable_st_split,
                     withdraw_taxable_for_spending,
                     withdraw_taxable_specific_asset,
                     withdraw_traditional, withdraw_roth)


# ---------- Vectorised tax-bill helpers ----------

def _progressive_tax_vec(taxable: np.ndarray, brackets: list[Bracket]
                         ) -> np.ndarray:
    """Vectorised progressive tax. taxable: (P,) -> (P,)."""
    if not brackets:
        return np.zeros_like(taxable)
    thresh = np.array([b.threshold for b in brackets] + [np.inf])
    rates = np.array([b.rate for b in brackets])
    t = np.maximum(0.0, taxable)
    upper = thresh[1:][None, :]
    lower = thresh[:-1][None, :]
    in_b = np.clip(np.minimum(t[:, None], upper) - lower, 0.0, None)
    return (in_b * rates[None, :]).sum(-1)


def _ltcg_tax_vec(ord_taxable: np.ndarray, ltcg: np.ndarray,
                  brackets: list[Bracket]) -> np.ndarray:
    """Stack LTCG on top of ordinary taxable; tax at LTCG rates per slice."""
    if not brackets:
        return np.zeros_like(ord_taxable)
    thresh = np.array([b.threshold for b in brackets] + [np.inf])
    rates = np.array([b.rate for b in brackets])
    P = ord_taxable.shape[0]
    out = np.zeros(P)
    pos = np.maximum(0.0, ord_taxable)
    remaining = np.maximum(0.0, ltcg)
    for i in range(len(rates)):
        lo = thresh[i]
        hi = thresh[i + 1]
        # slice in this bracket: from max(pos, lo) to hi
        slice_lo = np.maximum(pos, lo)
        slice_hi = np.full(P, hi)
        in_slice = np.clip(np.minimum(remaining, slice_hi - slice_lo), 0.0, None)
        out += in_slice * rates[i]
        remaining -= in_slice
        pos += in_slice
    return out


def _ss_taxable_vec(ss_benefit: np.ndarray, other_income: np.ndarray,
                    fs: FilingStatus, ty: TaxYear) -> np.ndarray:
    lo, hi = ty.ss_provisional_thresholds[fs]
    prov = other_income + 0.5 * ss_benefit
    out = np.zeros_like(ss_benefit)
    # Tier 1: between lo and hi
    mask1 = (prov > lo) & (prov <= hi)
    out = np.where(mask1, np.minimum(0.5 * (prov - lo),
                                     0.5 * ss_benefit), out)
    # Tier 2: above hi
    mask2 = prov > hi
    tier1 = np.minimum(0.5 * (hi - lo), 0.5 * ss_benefit)
    tier2 = 0.85 * (prov - hi)
    out = np.where(mask2, np.minimum(tier1 + tier2, 0.85 * ss_benefit), out)
    return out


def _niit_vec(magi: np.ndarray, nii: np.ndarray, fs: FilingStatus,
              ty: TaxYear) -> np.ndarray:
    th = ty.niit_threshold[fs]
    excess = np.maximum(0.0, magi - th)
    return ty.niit_rate * np.minimum(nii, excess)


def _federal_tax_vec(ord_income: np.ndarray, ltcg_income: np.ndarray,
                     ss_benefit: np.ndarray, fs: FilingStatus,
                     ty: TaxYear = TAX_2024
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (federal_total, ord_taxable_after_std_ded)."""
    ss_tax = _ss_taxable_vec(ss_benefit, ord_income + ltcg_income, fs, ty)
    sd = ty.std_deduction[fs]
    ord_taxable = np.maximum(0.0, ord_income + ss_tax - sd)
    fed_ord = _progressive_tax_vec(ord_taxable, ty.ordinary_brackets[fs])
    fed_ltcg = _ltcg_tax_vec(ord_taxable, ltcg_income, ty.ltcg_brackets[fs])
    magi = ord_income + ss_tax + ltcg_income
    niit = _niit_vec(magi, ltcg_income, fs, ty)
    return fed_ord + fed_ltcg + niit, ord_taxable


# ---------- Result types (unchanged interface from the scalar version) ----------

@dataclass
class PathResult:
    terminal_real_wealth: float
    real_wealth_by_year: np.ndarray
    real_spending_by_year: np.ndarray
    real_shortfall_by_year: np.ndarray
    lifetime_real_tax: float
    failed: bool


@dataclass
class SimResult:
    paths: list[PathResult]
    # Detailed per-(path, year, account, asset) tracking. Set by simulate().
    # Useful for downstream Excel/CSV export at quantile granularity.
    real_balance_by_year: np.ndarray | None = None    # (P, H+1, 3, 3)
    real_contrib_by_year: np.ndarray | None = None    # (P, H, 3, 3)
    real_equity_by_year: np.ndarray | None = None     # (P, H+1) rental equity

    @property
    def n_paths(self) -> int:
        return len(self.paths)

    def terminal_quantiles(self, q: list[float] = [0.05, 0.5, 0.95]
                           ) -> dict[float, float]:
        x = np.array([p.terminal_real_wealth for p in self.paths])
        return {qi: float(np.quantile(x, qi)) for qi in q}

    def failure_rate(self) -> float:
        return sum(1 for p in self.paths if p.failed) / max(1, self.n_paths)

    def cvar_failure(self, alpha: float = 0.05) -> float:
        x = np.array([p.terminal_real_wealth for p in self.paths])
        x.sort()
        k = max(1, int(np.ceil(alpha * len(x))))
        return float(np.mean(x[:k]))


# ---------- helpers ----------

def _alloc_to_array(a: Allocation) -> np.ndarray:
    return np.array([a.stock, a.bond, a.cash])


def _record_balances(s: VState, year_idx: int) -> None:
    """Write end-of-year per-(account, asset) real balances into
    s.real_balance_by_year[:, year_idx, :, :]. Account axis order:
    [taxable, traditional, roth]."""
    cum_inf = s.cumulative_inflation[:, None]   # (P, 1)
    # Taxable: aggregate LT + ST cohorts per asset.
    taxable_real = (s.tax_lt_value + s.tax_st_value) / cum_inf  # (P, 3)
    trad_real = s.trad_balance / cum_inf
    roth_real = s.roth_balance / cum_inf
    s.real_balance_by_year[:, year_idx, 0, :] = taxable_real
    s.real_balance_by_year[:, year_idx, 1, :] = trad_real
    s.real_balance_by_year[:, year_idx, 2, :] = roth_real


def _deposit_inheritance(s: VState, scn: Scenario, inh, year_idx: int,
                          mask: np.ndarray) -> None:
    """Deposit a one-time inheritance into the configured account, but only
    for the paths in `mask` (boolean (P,)).

    Real -> nominal via each path's realised cumulative inflation."""
    nominal = inh.amount_real * s.cumulative_inflation * mask    # (P,)
    if inh.account == "taxable":
        tgt = _alloc_to_array(scn.target_allocations.taxable)
        for ai in range(N_ASSETS):
            amt = nominal * tgt[ai]
            s.tax_st_value[:, ai] += amt
            s.tax_st_basis[:, ai] += amt
    elif inh.account == "traditional":
        tgt = _alloc_to_array(scn.target_allocations.traditional)
        s.trad_balance += nominal[:, None] * tgt[None, :]
    elif inh.account == "roth":
        tgt = _alloc_to_array(scn.target_allocations.roth)
        s.roth_balance += nominal[:, None] * tgt[None, :]
        s.roth_basis += nominal
    else:
        raise ValueError(f"unknown inheritance account: {inh.account}")


def _spending_smile_factor(year_into_retirement: int, smile: str) -> float:
    if smile == "flat":
        return 1.0
    if smile == "bengen":
        if year_into_retirement < 10:
            return 1.0 - 0.01 * year_into_retirement
        if year_into_retirement < 20:
            return 0.90
        return min(1.10, 0.90 + 0.01 * (year_into_retirement - 20))
    raise ValueError(f"unknown spending smile: {smile}")


def _resolve_contrib_value(value: float | str, limit: float) -> float:
    if isinstance(value, str) and value == "max":
        return limit
    return float(value)


# ---------- Top-level driver ----------

def simulate(scn: Scenario,
             allocations: TargetAllocations | None = None,
             policy: Policy | None = None,
             ) -> SimResult:
    """Run a vectorised Monte Carlo. Returns SimResult.

    Decision policy resolution (in order of precedence):
      * `policy` argument (e.g., a GlidePolicy): the simulator queries
        `policy.decide(StateSummary)` each year.
      * `allocations` argument: builds a StaticPolicy that uses these
        allocations + the scenario's withdrawal/conversion settings.
      * Default: builds a StaticPolicy from scn.target_allocations and
        scn.withdrawal.roth_conversion_target_bracket.
    """
    if policy is None:
        if allocations is None:
            allocations = scn.target_allocations
        # Derive trad split from contributions (if both numeric)
        c = scn.savings.contributions
        try:
            pool = float(c.trad_401k) + float(c.roth_401k)
            split = float(c.trad_401k) / pool if pool > 0 else 1.0
        except (TypeError, ValueError):
            split = 1.0
        policy = StaticPolicy(
            allocations=allocations,
            conversion_bracket=scn.withdrawal.roth_conversion_target_bracket,
            trad_contribution_split=split,
        )

    horizon = scn.profile.horizon()
    P = scn.simulation.n_paths
    seed = scn.simulation.seed
    market = scn.market.to_market_model()

    return_model = scn.simulation.return_model
    if return_model == "deterministic":
        # All paths are identical; force a single path to skip wasted compute.
        P = 1
        returns_by_asset = sample_deterministic_paths(market, horizon, P)
        inflation = sample_deterministic_inflation(market, horizon, P)
    elif return_model in ("historical", "bootstrap"):
        returns_by_asset, inflation = sample_historical_paths(
            n_years=horizon, n_paths=P, seed=seed, block_size=1)
    elif return_model == "historical_ath":
        returns_by_asset, inflation = sample_historical_ath_paths(
            n_years=horizon, n_paths=P, seed=seed)
    elif return_model == "historical_stretched_ath":
        returns_by_asset, inflation = sample_historical_stretched_ath_paths(
            n_years=horizon, n_paths=P, seed=seed)
    else:
        # gbm (default)
        returns_by_asset = sample_gbm_paths(market, horizon, P, seed=seed)
        inflation = sample_inflation(market, horizon, P,
                                      seed=(seed or 0) + 1)
    R = np.stack([returns_by_asset[Asset.STOCK],
                  returns_by_asset[Asset.BOND],
                  returns_by_asset[Asset.CASH]], axis=-1)

    # Pre-sample correlated property returns + rent shocks. Both are (P, H)
    # arrays; they're indexed once per year inside the main loop below.
    if scn.rental_property is not None:
        property_returns_real, rent_shocks = _rental.sample_rental_paths(
            R, scn.market, scn.rental_property,
            seed=(seed or 0) + 7919)
    else:
        property_returns_real = None
        rent_shocks = None

    starting_wages = scn.state_taxes.total_wages(scn.profile.start_date, 0)
    s = VState.from_portfolio(scn.initial_portfolio, P, horizon,
                              starting_nominal_income=starting_wages)
    # Year-0 (initial) balance snapshot — same across paths because the
    # initial portfolio is a single specimen.
    _record_balances(s, year_idx=0)

    # Pre-sample per-path inheritance arrival years.
    # For deterministic Inheritance: same year_idx for every path.
    # For Gompertz hazard: sample each path independently; paths where the
    # benefactor outlives the horizon get year_idx = -1 (no deposit).
    inh_arrival = []   # list[(P,)] arrays, parallel to scn.inheritances
    inh_rng = np.random.default_rng((seed or 0) + 7)
    for inh in scn.inheritances:
        if inh.hazard is not None:
            b, c = inh.hazard.resolved()
            u = np.clip(inh_rng.uniform(size=P), 1e-12, 1 - 1e-12)
            # Inverse-CDF for age at death given alive at current_age:
            #   exp(c·x) = exp(c·a) − (c/b)·log(1 − U)
            inner = np.exp(c * inh.hazard.current_age) - (c / b) * np.log(1 - u)
            death_age = np.log(inner) / c
            years_from_now = death_age - inh.hazard.current_age
            year_idx_per_path = np.round(years_from_now).astype(int)
            # Mark out-of-horizon paths with -1
            year_idx_per_path = np.where(
                year_idx_per_path >= horizon, -1, year_idx_per_path)
            year_idx_per_path = np.maximum(0, year_idx_per_path)
        else:
            year_lo, _ = scn.state_taxes.year_window(scn.profile.start_date, 0)
            # find which sim year contains inh.date
            target = -1
            for y in range(horizon):
                lo, hi = scn.state_taxes.year_window(scn.profile.start_date, y)
                if lo <= inh.date < hi:
                    target = y
                    break
            year_idx_per_path = np.full(P, target, dtype=int)
        inh_arrival.append(year_idx_per_path)

    yf = np.array([market.yield_fraction[a] for a in ASSET_ORDER])

    fs = scn.profile.filing_status
    timeline: StateTimeline = scn.state_taxes
    sim_start = scn.profile.start_date
    retirement_age = scn.profile.retirement_age
    fire_target_real = 25.0 * scn.spending.annual_real  # 4% rule benchmark

    for y in range(horizon):
        age = scn.profile.age_at_year(y)
        ret_y = R[:, y, :]
        infl_y = inflation[:, y]
        s.cumulative_inflation *= (1.0 + infl_y)

        # ---- inheritance deposits for paths whose sampled arrival is y ----
        for inh, arrivals in zip(scn.inheritances, inh_arrival):
            mask = (arrivals == y)
            if mask.any():
                _deposit_inheritance(s, scn, inh, y, mask)

        # ---- rental property purchase trigger ----
        # Fired before the year's policy query so the policy sees the
        # post-purchase taxable balance. Cap gains from the downpayment
        # withdrawal are taxed in the same year as a separate adjustment.
        if scn.rental_property is not None:
            rp = scn.rental_property
            # Liquid wealth = ACCOUNTS only (taxable + Trad + Roth), excluding
            # rental equity which is illiquid for the purposes of funding a
            # downpayment. Both gates use the same (current) cum_infl deflator
            # so the trigger arithmetic stays internally consistent regardless
            # of the year's inflation tick.
            cum_infl_now = s.cumulative_inflation
            liquid_real = s.total_value() / cum_infl_now
            taxable_real = s.taxable_total() / cum_infl_now
            buyers = _rental.trigger_fires(s, rp, age, liquid_real, taxable_real)
            if buyers.any():
                dp_real = _rental.execute_purchase(s, rp, buyers, y)
                dp_n = dp_real * s.cumulative_inflation
                _proc, lt_g, st_g = withdraw_taxable_for_spending(s, dp_n)
                # Marginal cap-gains tax on the purchase-driven sale. Apply
                # at year-end below by stashing in pending arrays.
                rental_purchase_lt_gain_n = lt_g
                rental_purchase_st_gain_n = st_g
            else:
                rental_purchase_lt_gain_n = np.zeros(P)
                rental_purchase_st_gain_n = np.zeros(P)
        else:
            rental_purchase_lt_gain_n = np.zeros(P)
            rental_purchase_st_gain_n = np.zeros(P)

        # ---- query policy with current state summary ----
        if y == 0:
            median_real_w = float(s.real_wealth[:, 0].mean())
            median_idx = 0
        else:
            sort_idx = np.argsort(s.real_wealth[:, y])
            median_idx = int(sort_idx[P // 2])
            median_real_w = float(s.real_wealth[median_idx, y])
        progress = (median_real_w / fire_target_real
                    if fire_target_real > 0 else 1.0)
        # Realized-volatility signal: rolling std of the median path's
        # recent stock-return draws. Defaults to the configured stock vol
        # on early years before enough history accumulates.
        L = 5  # lookback window (years)
        if y >= 2:
            recent = R[median_idx, max(0, y - L):y, 0]   # stock returns
            if recent.size >= 2:
                rec_vol = float(recent.std(ddof=1))
            else:
                rec_vol = float(market.params[Asset.STOCK].vol)
        else:
            rec_vol = float(market.params[Asset.STOCK].vol)
        ss = StateSummary(
            age=age, year_idx=y,
            years_to_retirement=max(0.0, retirement_age - age),
            fire_target_real=fire_target_real,
            median_real_wealth=median_real_w,
            fire_progress_ratio=progress,
            recent_realized_vol=rec_vol,
        )
        decision = policy.decide(ss)
        targets_taxable = _alloc_to_array(decision.allocations.taxable)
        targets_trad = _alloc_to_array(decision.allocations.traditional)
        targets_roth = _alloc_to_array(decision.allocations.roth)

        if age < retirement_age:
            _step_accumulation(s, scn, age, y, ret_y, fs, timeline,
                               targets_taxable, targets_trad, targets_roth, yf,
                               decision)
        else:
            _step_decumulation(s, scn, age, y, ret_y, fs, timeline,
                               targets_taxable, targets_trad, targets_roth, yf,
                               decision)

        # ---- rental property: annual operating step ----
        if scn.rental_property is not None and s.rental_owned.any():
            rp = scn.rental_property
            pr_y = (property_returns_real[:, y]
                    if property_returns_real is not None else None)
            rs_y = (rent_shocks[:, y]
                    if rent_shocks is not None else None)
            (rental_taxable_real, rental_net_cash_real,
             _interest_real) = _rental.step_rental_year(s, rp,
                                                          property_return_real=pr_y,
                                                          rent_shock=rs_y)
            # Rental tax: Federal ordinary on rental_taxable_income; state
            # is sourced to the property's location_state regardless of
            # residency.
            rental_taxable_n = rental_taxable_real * s.cumulative_inflation
            # Federal: incremental tax of stacking rental on top of the year's
            # baseline ordinary income (wages / RMDs / conversions / dividends
            # / capital gains from the spending withdrawal). Captures the true
            # bracket / NIIT margin instead of the prior 0-baseline
            # approximation that under-stated tax in working years and RMD
            # years. Rental cash flow can also be negative (operating loss);
            # we treat negative rental income as a deduction that lowers the
            # year's federal tax bill (Schedule-E loss limited only by the
            # already-paid baseline — passive-activity loss limits are not
            # modelled, so this slightly *overstates* the deduction in some
            # high-AGI cases).
            ord_baseline_n = s.year_ord_income_n
            ltcg_baseline_n = s.year_ltcg_income_n
            ss_baseline_n = s.year_ss_nominal
            fed_with_rental, _ = _federal_tax_vec(
                np.maximum(0.0, ord_baseline_n + rental_taxable_n),
                ltcg_baseline_n, ss_baseline_n, fs)
            fed_baseline_recomp, _ = _federal_tax_vec(
                ord_baseline_n, ltcg_baseline_n, ss_baseline_n, fs)
            fed_rent_tax = fed_with_rental - fed_baseline_recomp
            # State: rental income is *source-based* to the property's state,
            # not residency. Compute standalone (no stacking with residency-
            # sourced income, which is taxed by a different state).
            state_rent_tax = state_tax_vec(
                state=rp.location_state,
                ordinary_income=np.maximum(0.0, rental_taxable_n),
                ltcg_income=np.zeros(P),
                filing_status=fs)
            # Cap-gains tax on the purchase sale (this year only)
            cg_fed_tax = np.zeros(P)
            cg_state_tax = np.zeros(P)
            if (rental_purchase_lt_gain_n.any()
                    or rental_purchase_st_gain_n.any()):
                _, cg_fed_tax = _federal_tax_vec(
                    np.zeros(P),
                    rental_purchase_lt_gain_n,
                    rental_purchase_st_gain_n, fs)
                # Residency-weighted, since cap gains are residency-sourced
                # (the seller still lives wherever they live)
                residency_w = timeline.residency_weights(sim_start, y)
                cg_state_tax = state_residency_tax_vec(
                    residency_weights=residency_w,
                    ordinary_other=rental_purchase_st_gain_n,
                    ltcg=rental_purchase_lt_gain_n,
                    filing_status=fs)
            rental_total_tax_n = (fed_rent_tax + state_rent_tax
                                   + cg_fed_tax + cg_state_tax)
            rental_total_tax_real = rental_total_tax_n / s.cumulative_inflation

            # Post-tax cash flow into / out of taxable cash sleeve
            post_tax_real = rental_net_cash_real - rental_total_tax_real
            post_tax_n = post_tax_real * s.cumulative_inflation
            # If positive, deposit to taxable cash; first try to repay HELOC
            pos = np.maximum(0.0, post_tax_real)
            if pos.any():
                applied_repay = _rental.repay_heloc_real(s, pos)
                deposit_real = pos - applied_repay
                deposit_n = deposit_real * s.cumulative_inflation
                deposit_taxable_st(s, ASSET_IDX[Asset.CASH], deposit_n)
            # If negative, debit taxable cash; backstop with HELOC
            neg_real = np.maximum(0.0, -post_tax_real)
            if neg_real.any():
                neg_n = neg_real * s.cumulative_inflation
                _proc, lt_g, st_g = withdraw_taxable_for_spending(s, neg_n)
                # Any unfunded shortfall draws on HELOC up to capacity
                still_short_n = neg_n - _proc
                still_short_real = still_short_n / s.cumulative_inflation
                if (still_short_real > 0).any():
                    _rental.draw_heloc_real(s, still_short_real, rp)

            # Record total rental tax in the year's tax bucket
            s.real_taxes[:, y] += rental_total_tax_real

        s.age_st_to_lt()
        s.real_wealth[:, y + 1] = s.total_value() / s.cumulative_inflation
        _record_balances(s, year_idx=y + 1)
        # Record property equity in real $: net of mortgage AND HELOC. This
        # is the "if I sold today and paid off both the mortgage and the
        # HELOC" residual, which is the honest contribution to total wealth.
        # Fold this same equity into real_wealth so all downstream consumers
        # (FIRE-prob optimizer measure, terminal_real_wealth, StateSummary's
        # median, visualizations) see the household's true balance-sheet
        # wealth — *not* accounts only.
        if scn.rental_property is not None and s.rental_owned.any():
            cum_infl = s.cumulative_inflation
            eq_real = (s.rental_value_real
                       - (s.mortgage_balance_nominal
                          + s.heloc_balance_nominal) / cum_infl)
            eq_real = np.where(s.rental_owned, np.maximum(0.0, eq_real), 0.0)
            s.real_equity_by_year[:, y + 1] = eq_real
            s.real_wealth[:, y + 1] += eq_real
        if age >= retirement_age:
            # Ruin: taxable+401k+roth all depleted AND no HELOC capacity left.
            # When rental is configured, the equity backstop softens ruin.
            tot = s.total_value()
            if scn.rental_property is not None and s.rental_owned.any():
                eq_real = _rental.accessible_equity_real(s, scn.rental_property)
                eq_n = eq_real * s.cumulative_inflation
                tot = tot + eq_n
            s.failed |= (tot <= 0)

    # Build per-path PathResult objects from arrays
    paths: list[PathResult] = []
    for i in range(P):
        paths.append(PathResult(
            terminal_real_wealth=float(s.real_wealth[i, -1]),
            real_wealth_by_year=s.real_wealth[i].copy(),
            real_spending_by_year=s.real_target_spend[i].copy(),
            real_shortfall_by_year=s.real_shortfall[i].copy(),
            lifetime_real_tax=float(s.real_taxes[i].sum()),
            failed=bool(s.failed[i]),
        ))
    return SimResult(paths=paths,
                     real_balance_by_year=s.real_balance_by_year.copy(),
                     real_contrib_by_year=s.real_contrib_by_year.copy(),
                     real_equity_by_year=s.real_equity_by_year.copy())


# ---------- Per-year accumulation step ----------

def _step_accumulation(s: VState, scn: Scenario, age: float, year_idx: int,
                       returns_y: np.ndarray, fs: FilingStatus,
                       timeline: StateTimeline,
                       tgt_tax: np.ndarray, tgt_trad: np.ndarray,
                       tgt_roth: np.ndarray, yf: np.ndarray,
                       decision: Decision) -> None:
    """Vectorised accumulation year, in-place mutation of s."""
    sim_start = scn.profile.start_date
    P = s.n_paths

    # 1) Wages this year by employment state (mid-year transitions handled
    # by fraction-of-year overlap).
    wages_by_state = timeline.wages_by_state(sim_start, year_idx)
    total_wages = sum(wages_by_state.values())
    s.nominal_income[:] = total_wages

    # 2) Returns + yield decomposition (taxable yield -> new ST cohort)
    s.trad_balance *= (1.0 + returns_y)
    s.roth_balance *= (1.0 + returns_y)
    pos_ret = np.maximum(0.0, returns_y)
    yield_amt_lt = s.tax_lt_value * pos_ret * yf
    yield_amt_st = s.tax_st_value * pos_ret * yf
    appreciation = returns_y - pos_ret * yf
    s.tax_lt_value *= (1.0 + appreciation)
    s.tax_st_value *= (1.0 + appreciation)
    yield_amt = yield_amt_lt + yield_amt_st
    s.tax_st_value += yield_amt
    s.tax_st_basis += yield_amt
    qual_div = yield_amt[:, 0]
    ord_div = yield_amt[:, 1] + yield_amt[:, 2]

    # 3) Contributions (subject to IRS limits + age 50 catchup)
    catchup_401k = TAX_2024.contrib_limit_401k_catchup if age >= 50 else 0.0
    catchup_ira = TAX_2024.contrib_limit_ira_catchup if age >= 50 else 0.0
    limit_401k = TAX_2024.contrib_limit_401k + catchup_401k
    limit_ira = TAX_2024.contrib_limit_ira + catchup_ira
    total_415c = 69_000.0 + catchup_401k

    c = scn.savings.contributions
    # Pool the 401k contribution and split per the policy. If the user wrote
    # numeric trad_401k + roth_401k, their sum is the pool; if either is
    # 'max', we cap at the limit. The policy's split (0..1) determines the
    # Traditional fraction.
    try:
        configured_pool = min(float(c.trad_401k) + float(c.roth_401k),
                              limit_401k, total_wages)
    except (TypeError, ValueError):
        configured_pool = min(_resolve_contrib_value(c.trad_401k, limit_401k)
                              + _resolve_contrib_value(c.roth_401k,
                                                       limit_401k),
                              limit_401k, total_wages)
    pool_401k = max(0.0, configured_pool)
    trad_401k = pool_401k * decision.trad_contribution_split
    roth_401k = pool_401k - trad_401k
    trad_ira = min(_resolve_contrib_value(c.trad_ira, limit_ira), limit_ira)
    roth_ira = min(_resolve_contrib_value(c.roth_ira, limit_ira - trad_ira),
                   limit_ira - trad_ira)
    employer_match = c.employer_match_rate * total_wages
    mbdr_room = max(0.0, total_415c - trad_401k - roth_401k - employer_match)
    if isinstance(c.mega_backdoor_roth, str) and c.mega_backdoor_roth == "max":
        mbdr = mbdr_room
    else:
        mbdr = min(float(c.mega_backdoor_roth), mbdr_room)

    pretax_trad = trad_401k + trad_ira
    s.trad_balance += pretax_trad * tgt_trad[None, :]
    s.trad_balance += employer_match * tgt_trad[None, :]
    roth_direct = roth_401k + roth_ira
    s.roth_balance += roth_direct * tgt_roth[None, :]
    s.roth_basis += roth_direct
    s.roth_balance += mbdr * tgt_roth[None, :]
    s.roth_basis += mbdr

    # Record contribution flows (real $) into the per-(year, account, asset)
    # tracking array. Trad-account inflow this year = trad_401k + trad_ira +
    # employer_match (all allocated per tgt_trad). Roth-account inflow =
    # roth_401k + roth_ira + mbdr. Taxable-account inflow is recorded after
    # the residual savings deposit below.
    inv_inf = 1.0 / s.cumulative_inflation   # nominal -> real, (P,)
    real_trad_in = (pretax_trad + employer_match) * inv_inf   # (P,)
    real_roth_in = (roth_direct + mbdr) * inv_inf
    s.real_contrib_by_year[:, year_idx, 1, :] = (
        real_trad_in[:, None] * tgt_trad[None, :])
    s.real_contrib_by_year[:, year_idx, 2, :] = (
        real_roth_in[:, None] * tgt_roth[None, :])

    # 4) Tax bill
    # Federal: ordinary base = wages_after_pretax + ord_div; LTCG = qual_div.
    wages_after_pretax = max(0.0, total_wages - pretax_trad)
    ord_income_fed = wages_after_pretax + ord_div
    ltcg_income = qual_div
    fed_tax, _ = _federal_tax_vec(ord_income_fed, ltcg_income,
                                  np.zeros(P), fs)
    # State tax: wages by employment state (apportioning pretax_401k);
    # plus residency tax on dividends.
    state_wage_tax_scalar = state_wages_tax(
        wages_by_state=wages_by_state, pretax_401k=pretax_trad,
        filing_status=fs)
    residency_w = timeline.residency_weights(sim_start, year_idx)
    state_inv_tax = state_residency_tax_vec(
        residency_weights=residency_w, ordinary_other=ord_div,
        ltcg=qual_div, filing_status=fs)
    state_tax_total = state_wage_tax_scalar + state_inv_tax  # (P,)
    bill_total = fed_tax + state_tax_total

    # Stash baseline tax inputs so the rental block (run after this step)
    # can compute incremental federal tax with bracket stacking.
    s.year_ord_income_n[:] = ord_income_fed
    s.year_ltcg_income_n[:] = ltcg_income
    s.year_ss_nominal[:] = 0.0

    # 5) Take-home and residual taxable savings
    take_home = (total_wages - trad_401k - trad_ira - roth_direct
                 - mbdr - bill_total)
    # Living budget: explicit working-years post-tax target (preferred) or
    # the savings-rate residual.
    if scn.spending.working_annual_real is not None:
        # Explicit post-tax living target, inflation-adjusted to nominal.
        implied_living = (scn.spending.working_annual_real
                          * s.cumulative_inflation)
    else:
        implied_living = (1.0 - scn.savings.rate) * total_wages
    taxable_savings = np.maximum(0.0, take_home - implied_living)
    deposit_taxable_st_split(s, taxable_savings, tgt_tax)
    real_taxable_in = taxable_savings * inv_inf      # (P,)
    s.real_contrib_by_year[:, year_idx, 0, :] = (
        real_taxable_in[:, None] * tgt_tax[None, :])

    # 6) Rebalance tax-advantaged
    _rebalance_to_target(s.trad_balance, tgt_trad)
    _rebalance_to_target(s.roth_balance, tgt_roth)

    s.real_taxes[:, year_idx] = bill_total / s.cumulative_inflation


def _rebalance_to_target(balances: np.ndarray, target: np.ndarray) -> None:
    """Rebalance a (P, 3) balance array to target (3,) within total. In-place.
    Floors at zero — depleted accounts shouldn't reflect negative drift from
    floating-point rounding."""
    total = np.maximum(0.0, balances.sum(-1, keepdims=True))
    balances[:] = total * target[None, :]


# ---------- Per-year decumulation step ----------

def _step_decumulation(s: VState, scn: Scenario, age: float, year_idx: int,
                       returns_y: np.ndarray, fs: FilingStatus,
                       timeline: StateTimeline,
                       tgt_tax: np.ndarray, tgt_trad: np.ndarray,
                       tgt_roth: np.ndarray, yf: np.ndarray,
                       decision: Decision) -> None:
    """Vectorised decumulation year, in-place mutation of s."""
    P = s.n_paths
    sim_start = scn.profile.start_date
    # 1) Returns
    s.trad_balance *= (1.0 + returns_y)
    s.roth_balance *= (1.0 + returns_y)
    pos_ret = np.maximum(0.0, returns_y)
    yield_amt = (s.tax_lt_value + s.tax_st_value) * pos_ret * yf
    appreciation = returns_y - pos_ret * yf
    s.tax_lt_value *= (1.0 + appreciation)
    s.tax_st_value *= (1.0 + appreciation)
    s.tax_st_value += yield_amt
    s.tax_st_basis += yield_amt
    qual_div = yield_amt[:, 0]
    ord_div = yield_amt[:, 1] + yield_amt[:, 2]

    # 2) Spending target (real, then nominal). Per-path because flexible
    # spending makes the target depend on each path's portfolio drawdown.
    retirement_age = scn.profile.retirement_age
    years_into_retire = max(0, int(round(age - retirement_age)))
    factor = _spending_smile_factor(years_into_retire, scn.spending.smile)
    base_real = scn.spending.annual_real * factor

    # On the first retirement year of each path, freeze the per-path
    # baseline real wealth used to compute the drawdown ratio.
    if years_into_retire == 0:
        # Use start-of-year (post-return) real wealth.
        s.flex_baseline_wealth = s.total_value() / s.cumulative_inflation

    if scn.spending.flexible is not None and years_into_retire >= 0:
        flex = scn.spending.flexible
        current_floor = flex.floor_at(age)
        current_real = s.total_value() / s.cumulative_inflation
        # ratio relative to retirement-start real wealth; safe-divide.
        baseline = np.where(s.flex_baseline_wealth > 0,
                             s.flex_baseline_wealth, 1.0)
        ratio = current_real / baseline
        # Downside-only proportional scaling, then clamp to [floor/base, 1].
        scaling = np.minimum(1.0, 1.0 + flex.sensitivity * (ratio - 1.0))
        floor_scaling = (current_floor / base_real if base_real > 0 else 0.0)
        scaling = np.maximum(scaling, floor_scaling)
        real_target = base_real * scaling
        real_target = np.maximum(real_target, current_floor)
    else:
        real_target = np.full(P, base_real)

    s.real_target_spend[:, year_idx] = real_target
    nominal_target = real_target * s.cumulative_inflation

    # 3) Social Security (in nominal; COLA'd by realised inflation)
    ss_nominal = np.zeros(P)
    if (age >= scn.social_security.claim_age
            and scn.social_security.monthly_at_67 > 0):
        ss_nominal = (scn.social_security.monthly_at_67 * 12.0
                      * s.cumulative_inflation)

    # 4) RMDs (forced traditional withdrawal)
    rmd_amt = np.zeros(P)
    age_int = int(age + 1e-6)  # robust to float wobble on birthday boundary
    if age_int >= RMD_START_AGE:
        prior_trad = s.trad_balance.sum(-1)
        divisor = RMD_DIVISORS.get(min(age_int, max(RMD_DIVISORS)), 6.0)
        rmd_amt = prior_trad / divisor
    rmd_taken = withdraw_traditional(s, rmd_amt)

    ord_income = ord_div + rmd_taken
    ltcg_income = qual_div

    # 5) Roth conversion ladder. Bracket target comes from the policy
    # (which may be life-phase conditional), with the ACA cap from
    # withdrawal config.
    wd = scn.withdrawal
    conv_bracket = decision.conversion_bracket
    if conv_bracket is not None:
        target_ord_taxable = top_of_bracket(conv_bracket, fs)
        target_gross = target_ord_taxable + TAX_2024.std_deduction[fs]
        room = np.maximum(0.0, target_gross
                          - (ord_income + ss_nominal * 0.85))
        if wd.aca_magi_cap is not None and age < 65:
            cap_nominal = wd.aca_magi_cap * s.cumulative_inflation
            room = np.minimum(room, np.maximum(
                0.0, cap_nominal - (ord_income + ltcg_income + ss_nominal)))
        conv = np.minimum(room, s.trad_balance.sum(-1))
        # Withdraw from traditional, deposit into Roth, record conversion
        actual_conv = withdraw_traditional(s, conv)
        s.roth_balance += actual_conv[:, None] * tgt_roth[None, :]
        s.roth_conversions[:, year_idx] += actual_conv
        ord_income += actual_conv

    # 6) Spending withdrawal
    net_need = np.maximum(0.0, nominal_target - ss_nominal)
    proceeds, lt_g, st_g = _execute_withdrawal_strategy(
        s, net_need, age, year_idx, scn)
    ltcg_income += lt_g
    ord_income += st_g

    # 7) Tax bill (federal + state)
    fed_tax, _ = _federal_tax_vec(ord_income, ltcg_income, ss_nominal, fs)
    # In retirement: any wages from active income sources still apply
    # (e.g., post-retirement consulting). Otherwise pure residency-based tax.
    wages_by_state = timeline.wages_by_state(sim_start, year_idx)
    if sum(wages_by_state.values()) > 0:
        state_wage_tax_scalar = state_wages_tax(
            wages_by_state=wages_by_state, pretax_401k=0.0,
            filing_status=fs)
    else:
        state_wage_tax_scalar = 0.0
    residency_w = timeline.residency_weights(sim_start, year_idx)
    state_inv_tax = state_residency_tax_vec(
        residency_weights=residency_w, ordinary_other=ord_income,
        ltcg=ltcg_income, filing_status=fs)
    bill_total = fed_tax + state_wage_tax_scalar + state_inv_tax

    # Stash baseline tax inputs so the rental block can compute incremental
    # federal tax with bracket stacking (rather than taxing rental at zero
    # baseline, which would systematically understate the marginal rate).
    s.year_ord_income_n[:] = ord_income
    s.year_ltcg_income_n[:] = ltcg_income
    s.year_ss_nominal[:] = ss_nominal

    # 8) Pay tax: another withdrawal pass for the tax dollars
    paid, lt_g2, st_g2 = withdraw_taxable_for_spending(s, bill_total)
    remaining_tax = bill_total - paid
    if (remaining_tax > 0).any():
        # Try traditional (post-59.5 only — pre-59.5 hits penalty)
        if age >= 59.5:
            pulled = withdraw_traditional(s, remaining_tax)
            remaining_tax -= pulled
        if (remaining_tax > 0).any():
            r_proc, r_ord, r_pen = withdraw_roth(s, remaining_tax, year_idx, age)
            remaining_tax -= r_proc

    # 9) Compute shortfall
    received = proceeds + ss_nominal
    shortfall = np.maximum(0.0, nominal_target - received) + remaining_tax
    s.real_shortfall[:, year_idx] = shortfall / s.cumulative_inflation

    # 9b) HELOC backstop for spending shortfalls. Drawn AFTER all account
    # waterfalls have been exhausted; reduces the recorded shortfall by the
    # amount actually drawn. Subsequent years carry the HELOC balance and
    # accrue interest in the rental block. Without this draw the ruin check
    # below was a phantom backstop — it credited accessible_equity toward
    # avoiding ruin without ever putting the cash on the household's table.
    if scn.rental_property is not None and s.rental_owned.any():
        short_real = s.real_shortfall[:, year_idx]
        if (short_real > 0).any():
            drawn_real = _rental.draw_heloc_real(
                s, short_real, scn.rental_property)
            s.real_shortfall[:, year_idx] -= drawn_real

    # 10) Rebalance tax-advantaged
    _rebalance_to_target(s.trad_balance, tgt_trad)
    _rebalance_to_target(s.roth_balance, tgt_roth)

    s.real_taxes[:, year_idx] = bill_total / s.cumulative_inflation


def _execute_withdrawal_strategy(s: VState, net_need: np.ndarray, age: int,
                                 year_idx: int, scn: Scenario
                                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tax-aware withdrawal:
        Step A: taxable (cash -> bond -> stock; LT first)
        Step B: 59.5+ -> traditional next
        Step C: <59.5 -> Roth (basis, mature conversions, green w/ penalty,
                         earnings w/ penalty + ordinary tax)
        Step D: 59.5+ -> Roth last (tax-free)
    Returns (gross_proceeds, lt_gain_added, st_gain_added) all (P,).
    Gains added are realised gains created by sales in the taxable account.
    Roth-related ordinary income from earnings is folded into the caller's
    tax computation via withdraw_roth's `ord_add`.
    """
    P = s.n_paths
    remaining = net_need.copy()
    gross = np.zeros(P)
    lt_g = np.zeros(P)
    st_g = np.zeros(P)

    # Step A
    proc, lt, st = withdraw_taxable_for_spending(s, remaining)
    gross += proc
    lt_g += lt
    st_g += st
    remaining -= proc

    # Step B: 59.5+ -> traditional
    if age >= 59.5 and (remaining > 0).any():
        pulled = withdraw_traditional(s, remaining)
        gross += pulled
        # Traditional withdrawals are ordinary income; caller's tax bill
        # already includes them via the prior `ord_income` accumulation. To
        # signal this, we add to st_g (which the caller treats as ordinary).
        # Better: use a dedicated channel. For simplicity:
        st_g += pulled
        remaining -= pulled

    # Step C / D: Roth
    if (remaining > 0).any():
        r_proc, r_ord, r_pen = withdraw_roth(s, remaining, year_idx, age)
        gross += r_proc
        st_g += r_ord + r_pen  # ordinary tax + penalty added to ord
        remaining -= r_proc

    # Step E (fallback): pre-59.5 traditional with 10% penalty
    if age < 59.5 and (remaining > 0).any():
        pulled = withdraw_traditional(s, remaining)
        gross += pulled
        st_g += pulled + 0.10 * pulled  # ordinary + penalty
        remaining -= pulled

    return gross, lt_g, st_g
