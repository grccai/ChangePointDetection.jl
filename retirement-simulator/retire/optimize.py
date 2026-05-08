"""Allocation optimizer.

Decision variables (8 by default, exposed as a flat vector to scipy):
  x[0:2]  taxable account     (stock, bond)  ; cash = 1 - stock - bond
  x[2:4]  traditional account (stock, bond)
  x[4:6]  Roth account        (stock, bond)
  x[6]    Roth conversion bracket target in {0, 0.10, 0.12, 0.22, 0.24}
          discretised by rounding (we treat as continuous and snap)
  x[7]    contribution split: fraction of 401k contribution that goes
          traditional (rest goes to Roth 401k).

We use scipy.optimize.differential_evolution for a global search since the
objective is noisy (Monte Carlo) and non-smooth (tax brackets, RMD kink).

The objective trades off median terminal real wealth and CVaR (worst-tail
real wealth) using a CRRA utility on lifetime real consumption, plus a heavy
penalty for plan failure. Specifically:

  U(c_t) = c_t^(1-gamma) / (1-gamma)   if gamma != 1 else log(c_t)

  J(allocation) = E[ sum_t beta^(t-T_ret) U(c_t) ]  -  lambda * P(failure)
  where c_t is *real* consumption actually delivered (target - shortfall).

Since target consumption is identical across allocations, the objective
reduces to differences in shortfall and terminal wealth. We add a bequest
term beta_bequest * U(W_T) so the optimizer doesn't deplete to zero.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from copy import deepcopy
from typing import Callable

import numpy as np
from scipy.optimize import differential_evolution, minimize

from .config import (Scenario, Allocation, TargetAllocations,
                     WithdrawalPolicy, RentalProperty,
                     RentalPurchaseTrigger)
from .location import heuristic_target_allocations
from .policy import (Policy, StaticPolicy, GlidePolicy,
                     build_glide_policy, GLIDE_PARAM_BOUNDS,
                     build_three_knot_glide_policy,
                     THREE_KNOT_GLIDE_PARAM_BOUNDS,
                     build_bond_tent_policy, BOND_TENT_PARAM_BOUNDS,
                     build_cppi_policy, CPPI_PARAM_BOUNDS,
                     build_bodie_merton_policy,
                     BODIE_MERTON_PARAM_BOUNDS,
                     build_multi_phase_policy, MULTI_PHASE_PARAM_BOUNDS,
                     build_vol_targeting_policy,
                     VOL_TARGETING_PARAM_BOUNDS)
from .simulate import simulate, SimResult


@dataclass
class OptimizerConfig:
    gamma: float = 3.0          # CRRA risk aversion
    beta: float = 0.97          # time discount factor (per year)
    bequest_weight: float = 5.0 # weight on terminal wealth utility
    failure_penalty: float = 1e6
    n_paths_inner: int = 1500   # MC paths per objective eval
    maxiter: int = 30
    popsize: int = 12
    workers: int = 1            # set > 1 for parallel
    polish: bool = False
    seed: int = 12345
    # 'free'      : optimize 8 vars (per-account allocations + conv + split)
    # 'heuristic' : optimize 4 vars (overall stock/bond + conv + split);
    #               location is fixed by the tax-efficient heuristic
    location_mode: str = "free"
    # 'static'           : single fixed Decision applied every year (legacy).
    # 'glide'            : 12 vars — 2-knot per-account glide (start, end);
    #                      cash forbidden in Trad/Roth.
    # 'three_knot_glide' : 16 vars — 3-knot per-account glide (start,
    #                      retirement, end); separates accumulation and
    #                      decumulation slopes; cash forbidden in Trad/Roth.
    policy_class: str = "static"
    # Objective family:
    #   'utility'   : maximize CRRA utility of consumption + bequest, with a
    #                 plan-failure penalty. (default)
    #   'fire_prob' : maximize P(real wealth at `fire_age` >= `fire_target_real`)
    #                 subject to P(ruin over full horizon) <= `ruin_max`. The
    #                 ruin constraint is a steep penalty (effectively hard).
    #   'fire_prob_weighted' : maximize a 11-year time-decayed sum of FIRE
    #                 probabilities,
    #                     sum_{i=0..10} (1 - i/10) * P(W_{fire_age+i} >= T),
    #                 subject to the same ruin constraint. Rewards hitting
    #                 the target *early*; max possible value = 5.5.
    objective: str = "utility"
    # fire_prob-specific knobs (None defaults are filled in by `optimize()`).
    fire_target_real: float | None = None    # default: 25 * scn.spending.annual_real
    fire_age: int | None = None              # default: scn.profile.retirement_age
    ruin_max: float = 0.01                   # 1% by default for fire_prob
    # Search algorithm:
    #   'differential_evolution' : default, scipy.optimize.differential_evolution
    #   'cma_es'                 : Hansen et al. CMA-ES via the cma package
    #   'bipop_cma_es'           : BIPOP-restart CMA-ES; better for noisy /
    #                              multimodal objectives, often outperforms DE
    #                              on 10-50-dim continuous problems.
    algorithm: str = "differential_evolution"
    # Total evaluation budget cap (only used by cma/bipop_cma; DE uses
    # popsize x maxiter x ndim from its own logic).
    max_evals: int | None = None
    # For 'fire_prob_robust' objective: list of return_model strings to
    # evaluate worst-case across. Default = ['gbm', 'historical'] when
    # objective='fire_prob_robust'. Doubles per-eval MC cost.
    robust_return_modes: list[str] | None = None
    # If True and the scenario has a rental_property, append 3 extra
    # decision variables to the search vector covering the rental's
    # liquid + taxable purchase gates and the property price. See
    # RENTAL_PARAM_BOUNDS / decode_rental. (location_state and
    # trigger.min_age come from the YAML.)
    optimize_rental: bool = False
    # Fixed levers that USED to be optimization variables. They are
    # plumbed into the policies directly rather than being searched. Set
    # via CLI/programmatic config to run sensitivity sweeps without
    # touching the YAML; the default values (1.0 / 0.03) match the
    # historical optimizer-default behaviour.
    #
    # `fixed_trad_split` is the fraction of the 401(k) pool that goes
    # to Traditional (rest to Roth 401(k)). YAML's
    # `savings.contributions.trad_401k: max, roth_401k: 0` already
    # implies 1.0; the optimizer's trad_split optima clustered tightly
    # at 0.92–1.00 so the search dimension was wasted.
    #
    # `fixed_r_hc` is the real discount rate applied to future wages
    # to derive the Bodie-Merton human-capital trajectory. Optima never
    # printed it (jointly identifiable with γ); fixing at 3% real keeps
    # γ alone as the risk-aversion lever.
    fixed_trad_split: float = 1.0
    fixed_r_hc: float = 0.03


def _alloc(s: float, b: float) -> Allocation:
    s = max(0.0, min(1.0, s))
    b = max(0.0, min(1.0 - s, b))
    c = max(0.0, 1.0 - s - b)
    return Allocation(stock=s, bond=b, cash=c)


def _snap_conversion(raw: float) -> float | None:
    """Snap a continuous index to a discrete conversion bracket."""
    candidates = [None, 0.10, 0.12, 0.22, 0.24, 0.32]
    idx = int(round(raw))
    idx = max(0, min(len(candidates) - 1, idx))
    return candidates[idx]


# ---------- Reduced parameter bounds (post r_hc / trad_split removal) ----
#
# Full bond_tent param vector is 9 elements (see policy.py). The reduced
# vector drops index 7 (trad_contribution_split). Index 8 (wealth_resp)
# stays — re-numbered to position 7 in the reduced vector.
#
# Full bodie_merton param vector is 6 elements. The reduced vector drops
# index 1 (r_hc) and index 5 (trad_contribution_split). Indices 2-4 are
# re-numbered to 1-3 in the reduced vector.
BOND_TENT_REDUCED_PARAM_BOUNDS: list[tuple[float, float]] = [
    (0.0, 1.0),     # 0  stock_high
    (0.0, 1.0),     # 1  stock_low
    (-15.0, 20.0),  # 2  tent_age_offset
    (3.0, 30.0),    # 3  span
    (0.0, 0.4),     # 4  taxable_cash
    (0.0, 5.0),     # 5  conv FIRE-gap idx
    (0.0, 5.0),     # 6  conv SS-window idx
    (0.0, 2.0),     # 7  wealth_responsiveness   (was index 8)
]

BODIE_MERTON_REDUCED_PARAM_BOUNDS: list[tuple[float, float]] = [
    (1.0, 10.0),    # 0  risk_aversion (γ)
    (0.0, 0.4),     # 1  taxable_cash             (was index 2)
    (0.0, 5.0),     # 2  conv FIRE-gap idx        (was index 3)
    (0.0, 5.0),     # 3  conv SS-window idx       (was index 4)
]


def _expand_bond_tent_x(x_red, fixed_trad_split: float) -> list[float]:
    """Expand the 8-element reduced bond_tent vector back into the
    9-element form `build_bond_tent_policy` expects, slotting
    `fixed_trad_split` into position 7."""
    if len(x_red) != len(BOND_TENT_REDUCED_PARAM_BOUNDS):
        raise ValueError(
            f"expected {len(BOND_TENT_REDUCED_PARAM_BOUNDS)} reduced "
            f"bond_tent params, got {len(x_red)}")
    return [float(x_red[0]), float(x_red[1]), float(x_red[2]),
            float(x_red[3]), float(x_red[4]), float(x_red[5]),
            float(x_red[6]), fixed_trad_split, float(x_red[7])]


def _expand_bodie_merton_x(x_red, fixed_r_hc: float,
                            fixed_trad_split: float) -> list[float]:
    """Expand the 4-element reduced bodie_merton vector back into the
    6-element form `build_bodie_merton_policy` expects, slotting
    `fixed_r_hc` into position 1 and `fixed_trad_split` into position 5."""
    if len(x_red) != len(BODIE_MERTON_REDUCED_PARAM_BOUNDS):
        raise ValueError(
            f"expected {len(BODIE_MERTON_REDUCED_PARAM_BOUNDS)} reduced "
            f"bodie_merton params, got {len(x_red)}")
    return [float(x_red[0]), fixed_r_hc, float(x_red[1]),
            float(x_red[2]), float(x_red[3]), fixed_trad_split]


# ---------- Rental decision variables ----------
# When OptimizerConfig.optimize_rental is True and the scenario has a
# rental_property, the search vector is extended by 3 floats appended
# after the policy-specific params:
#
#   r0  min_liquid_real_M    [0.5, 5.0]  in $M; liquid-wealth purchase gate
#   r1  min_taxable_real_M   [0.1, 2.0]  in $M; taxable-wealth purchase gate
#                                         (the simulator floors this at the
#                                         actual downpayment amount, so a
#                                         too-low value is silently raised)
#   r2  price_real_M         [0.4, 2.0]  in $M; purchase price
#
# Removed in this iteration:
#   * `state_idx` — `location_state` is now a YAML-only parameter on
#     `rental_property.location_state`; the optimizer no longer searches
#     over it. Rationale: state choice is mostly an artefact of practical
#     constraints (where you live, can manage a property, want to invest)
#     not an optimization variable.
#   * `min_age` — also a YAML-only parameter on `rental_property.trigger.
#     min_age`. Rationale: the wealth gates (min_liquid + min_taxable)
#     are usually the binding constraints; min_age was redundant once
#     wealth thresholds were realistic.
RENTAL_PARAM_BOUNDS: list[tuple[float, float]] = [
    (0.5, 5.0),       # min_liquid ($M)
    (0.1, 2.0),       # min_taxable ($M)
    (0.4, 2.0),       # price ($M)
]
RENTAL_NPARAMS = len(RENTAL_PARAM_BOUNDS)


def decode_rental(x_tail: np.ndarray | list[float],
                  rp_base: RentalProperty) -> RentalProperty:
    """Build a RentalProperty by overriding `rp_base` with the 3 trailing
    decision vars. `location_state` and `trigger.min_age` are preserved
    from `rp_base` (i.e., they come from the YAML, not the optimizer)."""
    if len(x_tail) != RENTAL_NPARAMS:
        raise ValueError(
            f"expected {RENTAL_NPARAMS} rental params, got {len(x_tail)}")
    min_liquid = float(np.clip(x_tail[0], 0.5, 5.0)) * 1_000_000
    min_taxable = float(np.clip(x_tail[1], 0.1, 2.0)) * 1_000_000
    price = float(np.clip(x_tail[2], 0.4, 2.0)) * 1_000_000
    return replace(
        rp_base,
        price_real=price,
        trigger=RentalPurchaseTrigger(
            min_age=rp_base.trigger.min_age,
            min_liquid_real_wealth=min_liquid,
            min_taxable_real_wealth=min_taxable,
        ),
    )


def _split_x(x: np.ndarray, has_rental: bool
             ) -> tuple[np.ndarray, np.ndarray | None]:
    """Split a decision vector into (policy-params, rental-params or None)."""
    if not has_rental:
        return np.asarray(x), None
    n = len(x)
    return np.asarray(x[: n - RENTAL_NPARAMS]), np.asarray(x[n - RENTAL_NPARAMS:])


def _apply_rental(scn: Scenario, x_rental: np.ndarray | None) -> None:
    """In-place override of scn.rental_property when rental decision vars
    are present. No-op otherwise."""
    if x_rental is None:
        return
    if scn.rental_property is None:
        return
    scn.rental_property = decode_rental(x_rental, scn.rental_property)


def _decode_free(x: np.ndarray) -> tuple[TargetAllocations, float | None, float]:
    """8-parameter decoding: per-account allocations + conv + split."""
    taxable = _alloc(x[0], x[1])
    traditional = _alloc(x[2], x[3])
    roth = _alloc(x[4], x[5])
    conv_target = _snap_conversion(x[6])
    trad_split = max(0.0, min(1.0, x[7]))
    return TargetAllocations(taxable, traditional, roth), conv_target, trad_split


def _decode_heuristic(x: np.ndarray, scn: Scenario
                      ) -> tuple[TargetAllocations, float | None, float]:
    """4-parameter decoding: overall (s, b) + conv + split. Location fixed
    by tax-efficient heuristic against the *initial* portfolio totals."""
    s = max(0.0, min(1.0, x[0]))
    b = max(0.0, min(1.0 - s, x[1]))
    c = max(0.0, 1.0 - s - b)
    p = scn.initial_portfolio
    targets = heuristic_target_allocations(
        s, b, c,
        p.taxable.value(), p.traditional.value(), p.roth.value(),
    )
    conv_target = _snap_conversion(x[2])
    trad_split = max(0.0, min(1.0, x[3]))
    return targets, conv_target, trad_split


def _crra(c: np.ndarray, gamma: float) -> np.ndarray:
    c = np.maximum(c, 1e-9)
    if abs(gamma - 1.0) < 1e-9:
        return np.log(c)
    return (c ** (1.0 - gamma)) / (1.0 - gamma)


def _build_policy(x: np.ndarray, scn_base: Scenario,
                  cfg: OptimizerConfig) -> Policy:
    """Decode `x` into a Policy according to cfg.policy_class /
    cfg.location_mode."""
    start_age = scn_base.profile.age
    end_age = scn_base.profile._age_on(scn_base.profile.end_of_plan_date)
    retirement_age = scn_base.profile.retirement_age
    ss_age = float(scn_base.social_security.claim_age)
    if cfg.policy_class == "glide":
        return build_glide_policy(list(x), start_age=start_age,
                                  end_age=end_age,
                                  retirement_age=retirement_age,
                                  ss_age=ss_age)
    if cfg.policy_class == "three_knot_glide":
        return build_three_knot_glide_policy(
            list(x), start_age=start_age,
            retirement_age=retirement_age,
            end_age=end_age, ss_age=ss_age)
    if cfg.policy_class == "bond_tent":
        x_full = _expand_bond_tent_x(list(x), cfg.fixed_trad_split)
        return build_bond_tent_policy(
            x_full, retirement_age=retirement_age, ss_age=ss_age)
    if cfg.policy_class == "cppi":
        return build_cppi_policy(
            list(x), retirement_age=retirement_age, ss_age=ss_age)
    if cfg.policy_class == "bodie_merton":
        x_full = _expand_bodie_merton_x(list(x), cfg.fixed_r_hc,
                                          cfg.fixed_trad_split)
        return build_bodie_merton_policy(
            x_full, scn=scn_base,
            retirement_age=retirement_age, ss_age=ss_age)
    if cfg.policy_class == "multi_phase":
        return build_multi_phase_policy(
            list(x), scn=scn_base,
            retirement_age=retirement_age, ss_age=ss_age)
    if cfg.policy_class == "vol_targeting":
        return build_vol_targeting_policy(
            list(x), retirement_age=retirement_age, ss_age=ss_age)
    # static
    if cfg.location_mode == "heuristic":
        allocations, conv_target, trad_split = _decode_heuristic(x, scn_base)
    else:
        allocations, conv_target, trad_split = _decode_free(x)
    return StaticPolicy(allocations=allocations,
                        conversion_bracket=conv_target,
                        trad_contribution_split=trad_split)


@dataclass
class _Objective:
    """Picklable callable for scipy.differential_evolution(workers=N)."""
    scn_base: Scenario
    cfg: OptimizerConfig
    horizon: int
    years_to_retire: int
    has_rental: bool = False

    def __call__(self, x: np.ndarray) -> float:
        x_pol, x_rent = _split_x(x, self.has_rental)
        policy = _build_policy(x_pol, self.scn_base, self.cfg)
        scn = deepcopy(self.scn_base)
        _apply_rental(scn, x_rent)
        scn.simulation.n_paths = self.cfg.n_paths_inner
        scn.simulation.seed = self.cfg.seed
        result = simulate(scn, policy=policy)

        n = len(result.paths)
        util = np.zeros(n)
        for i, p in enumerate(result.paths):
            real_consump = np.maximum(
                p.real_spending_by_year - p.real_shortfall_by_year, 0.0)
            betas = self.cfg.beta ** np.arange(self.horizon)
            betas[:self.years_to_retire] = 0.0
            u = _crra(np.maximum(real_consump, 1e-3), self.cfg.gamma)
            util[i] = float(np.sum(betas * u))
            util[i] += self.cfg.bequest_weight * float(_crra(
                np.array([max(p.terminal_real_wealth, 1.0)]), self.cfg.gamma)[0])

        expected_util = float(np.mean(util))
        fail_pen = self.cfg.failure_penalty * result.failure_rate()
        return -expected_util + fail_pen


@dataclass
class _FIREProbObjective:
    """Chance-constrained: maximize P(real wealth at `fire_age` >= target)
    subject to P(ruin) <= `ruin_max`. The constraint is enforced via a steep
    quadratic+linear penalty that quickly dominates the objective when
    violated, so DE searches the feasible interior."""
    scn_base: Scenario
    cfg: OptimizerConfig
    fire_age: int
    fire_target_real: float
    ruin_max: float
    year_idx_at_fire: int   # precomputed
    has_rental: bool = False

    def __call__(self, x: np.ndarray) -> float:
        x_pol, x_rent = _split_x(x, self.has_rental)
        policy = _build_policy(x_pol, self.scn_base, self.cfg)
        scn = deepcopy(self.scn_base)
        _apply_rental(scn, x_rent)
        scn.simulation.n_paths = self.cfg.n_paths_inner
        scn.simulation.seed = self.cfg.seed
        result = simulate(scn, policy=policy)

        wealth_fire = np.array([p.real_wealth_by_year[self.year_idx_at_fire]
                                 for p in result.paths])
        prob_hit = float((wealth_fire >= self.fire_target_real).mean())
        ruin = result.failure_rate()

        # Aim for tighter internal constraint than the user asked for, so
        # inner-MC noise (≈0.4pp stderr at 800 paths) doesn't let an
        # optimum slip across the boundary at the 5000-path final eval.
        internal_target = max(0.0, self.ruin_max - 0.003)
        slack = ruin - internal_target
        if slack <= 0:
            penalty = 0.0
        else:
            # Heavy barrier: a 1pp violation costs 1.0+ of objective space,
            # an order of magnitude bigger than any plausible gain in P(hit).
            penalty = 100.0 * slack + 100_000.0 * slack ** 2
        return -prob_hit + penalty


@dataclass
class _FIREProbWeightedObjective:
    """Time-decayed FIRE probability:

        reward = sum_{i=0..10} (1 - i/10) * P(W_{fire_age+i} >= target)

    subject to P(ruin) <= ruin_max via the same barrier-style penalty as
    `_FIREProbObjective`. Max reward = 5.5 (= 1 + 0.9 + 0.8 + ... + 0.0)
    if every retirement year is at-or-above target. The weights linearly
    decay to zero at age fire_age+10, so hitting the target early is
    materially better than late.
    """
    scn_base: Scenario
    cfg: OptimizerConfig
    fire_age: int
    fire_target_real: float
    ruin_max: float
    year_indices: list[int]   # year_idx for ages fire_age..fire_age+10
    weights: np.ndarray       # length 11, (1 - i/10) for i=0..10
    has_rental: bool = False

    def __call__(self, x: np.ndarray) -> float:
        x_pol, x_rent = _split_x(x, self.has_rental)
        policy = _build_policy(x_pol, self.scn_base, self.cfg)
        scn = deepcopy(self.scn_base)
        _apply_rental(scn, x_rent)
        scn.simulation.n_paths = self.cfg.n_paths_inner
        scn.simulation.seed = self.cfg.seed
        result = simulate(scn, policy=policy)

        wealth_arr = np.array([p.real_wealth_by_year for p in result.paths])
        # P(W_{fire_age+i} >= T) for each i: shape (11,)
        p_hit = np.array([
            (wealth_arr[:, idx] >= self.fire_target_real).mean()
            for idx in self.year_indices
        ])
        reward = float((self.weights * p_hit).sum())  # in [0, 5.5]

        ruin = result.failure_rate()
        internal_target = max(0.0, self.ruin_max - 0.003)
        slack = ruin - internal_target
        if slack <= 0:
            penalty = 0.0
        else:
            penalty = 100.0 * slack + 100_000.0 * slack ** 2
        return -reward + penalty


@dataclass
class _FIREProbRobustObjective:
    """Robust version: minimizes worst-case across multiple return modes.

    For each candidate policy, runs MC under EACH return model in
    `return_modes`, then:
       reward = min over modes of (time-decayed FIRE-prob reward)
       ruin   = max over modes of P(ruin) — applies to the penalty

    The result is a policy whose worst-case (across i.i.d. lognormal AND
    historical-bootstrap) outcome is as good as possible. Practical effect:
    optimizer naturally values cash/bond buffers because they reduce the
    historical-mode tail without proportionally hurting GBM-mode reward."""
    scn_base: Scenario
    cfg: OptimizerConfig
    fire_age: int
    fire_target_real: float
    ruin_max: float
    year_indices: list[int]
    weights: np.ndarray
    return_modes: list[str]      # e.g., ["gbm", "historical"]
    has_rental: bool = False

    def __call__(self, x: np.ndarray) -> float:
        x_pol, x_rent = _split_x(x, self.has_rental)
        policy = _build_policy(x_pol, self.scn_base, self.cfg)
        rewards = []
        ruins = []
        for mode in self.return_modes:
            scn = deepcopy(self.scn_base)
            _apply_rental(scn, x_rent)
            scn.simulation.n_paths = self.cfg.n_paths_inner
            scn.simulation.seed = self.cfg.seed
            scn.simulation.return_model = mode
            result = simulate(scn, policy=policy)
            wealth_arr = np.array([p.real_wealth_by_year for p in result.paths])
            p_hit = np.array([
                (wealth_arr[:, idx] >= self.fire_target_real).mean()
                for idx in self.year_indices
            ])
            rewards.append(float((self.weights * p_hit).sum()))
            ruins.append(float(result.failure_rate()))
        worst_reward = min(rewards)
        worst_ruin = max(ruins)
        internal_target = max(0.0, self.ruin_max - 0.003)
        slack = worst_ruin - internal_target
        if slack <= 0:
            penalty = 0.0
        else:
            penalty = 100.0 * slack + 100_000.0 * slack ** 2
        return -worst_reward + penalty


def _objective_for(scn_base: Scenario, cfg: OptimizerConfig):
    has_rental = bool(cfg.optimize_rental and scn_base.rental_property is not None)
    fire_objs = ("fire_prob", "fire_prob_weighted", "fire_prob_robust")
    if cfg.objective in fire_objs:
        fire_age = cfg.fire_age if cfg.fire_age is not None \
            else int(round(scn_base.profile.retirement_age))
        fire_target = cfg.fire_target_real if cfg.fire_target_real is not None \
            else 25.0 * scn_base.spending.annual_real
        horizon = scn_base.profile.horizon()
        start_age = scn_base.profile.age
        year_idx_at_fire = max(0, min(horizon,
                                       int(round(fire_age - start_age))))
        if cfg.objective == "fire_prob":
            return _FIREProbObjective(
                scn_base=scn_base, cfg=cfg,
                fire_age=fire_age, fire_target_real=fire_target,
                ruin_max=cfg.ruin_max,
                year_idx_at_fire=year_idx_at_fire,
                has_rental=has_rental,
            )
        year_indices = [
            min(horizon, max(0, int(round(fire_age + i - start_age))))
            for i in range(11)
        ]
        weights = np.array([1.0 - i / 10.0 for i in range(11)])
        if cfg.objective == "fire_prob_weighted":
            return _FIREProbWeightedObjective(
                scn_base=scn_base, cfg=cfg,
                fire_age=fire_age, fire_target_real=fire_target,
                ruin_max=cfg.ruin_max,
                year_indices=year_indices, weights=weights,
                has_rental=has_rental,
            )
        # fire_prob_robust
        modes = cfg.robust_return_modes or ["gbm", "historical"]
        return _FIREProbRobustObjective(
            scn_base=scn_base, cfg=cfg,
            fire_age=fire_age, fire_target_real=fire_target,
            ruin_max=cfg.ruin_max,
            year_indices=year_indices, weights=weights,
            return_modes=list(modes),
            has_rental=has_rental,
        )
    return _Objective(
        scn_base=scn_base, cfg=cfg,
        horizon=scn_base.profile.horizon(),
        years_to_retire=scn_base.profile.years_to_retirement(),
        has_rental=has_rental,
    )


@dataclass
class _SearchResult:
    """Adapter mimicking scipy's OptimizeResult for downstream code."""
    x: np.ndarray
    fun: float
    nit: int
    nfev: int
    message: str = ""


def _run_search(obj, bounds, cfg: OptimizerConfig) -> _SearchResult:
    """Dispatch on cfg.algorithm. Returns a _SearchResult-ish object with
    .x, .fun, .nit, .nfev, .message."""
    n = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    if cfg.algorithm == "differential_evolution":
        res = differential_evolution(
            obj, bounds=bounds, seed=cfg.seed, maxiter=cfg.maxiter,
            popsize=cfg.popsize, workers=cfg.workers, polish=cfg.polish,
            tol=1e-3, mutation=(0.5, 1.0), recombination=0.7,
            init="sobol",
            updating="deferred" if cfg.workers != 1 else "immediate",
        )
        return _SearchResult(x=res.x, fun=float(res.fun),
                              nit=int(res.nit), nfev=int(res.nfev),
                              message=str(res.message))

    if cfg.algorithm in ("cma_es", "bipop_cma_es"):
        try:
            import cma
        except ImportError as e:
            raise RuntimeError("Install the `cma` package for CMA-ES") from e
        # Box constraints via cma's bounds; rescale to roughly [0,10] range
        # so the unit-sigma works across heterogeneous parameter scales.
        scale = np.where(hi > lo, hi - lo, 1.0)
        # Initial guess: midpoint
        x0_real = (lo + hi) / 2.0
        x0_norm = (x0_real - lo) / scale  # in [0, 1]
        sigma0 = 0.25  # spans most of [0,1] in two sigma

        def obj_norm(z):
            # decode normalised z in [0,1] back to real bounds
            x = lo + np.clip(z, 0.0, 1.0) * scale
            return obj(x)

        budget = cfg.max_evals or (cfg.popsize * cfg.maxiter * n)
        opts = {
            "bounds": [[0.0] * n, [1.0] * n],
            "maxfevals": budget,
            "tolfun": 1e-3,
            "tolx": 1e-4,
            "seed": cfg.seed,
            "verbose": -9,    # silent
            "popsize": max(4 + int(3 * np.log(n)), cfg.popsize),
        }
        if cfg.algorithm == "bipop_cma_es":
            # Use cma's restart wrapper with BIPOP strategy
            best, es = cma.fmin2(
                obj_norm, x0_norm, sigma0, options=opts,
                bipop=True, restarts=9,
                incpopsize=2.0,
            )
            x_best_norm = best
            f_best = es.best.f
            nfev = int(es.countevals) if hasattr(es, "countevals") else 0
            nit = int(es.countiter) if hasattr(es, "countiter") else 0
        else:
            best, es = cma.fmin2(
                obj_norm, x0_norm, sigma0, options=opts,
            )
            x_best_norm = best
            f_best = es.best.f
            nfev = int(es.countevals) if hasattr(es, "countevals") else 0
            nit = int(es.countiter) if hasattr(es, "countiter") else 0

        x_best = lo + np.clip(np.array(x_best_norm), 0.0, 1.0) * scale
        return _SearchResult(x=x_best, fun=float(f_best),
                              nit=nit, nfev=nfev,
                              message=f"cma {cfg.algorithm} converged")

    raise ValueError(f"unknown algorithm: {cfg.algorithm}")


def optimize(scn: Scenario, cfg: OptimizerConfig | None = None
             ) -> tuple[TargetAllocations, float | None, float, dict]:
    """Run differential evolution. Returns
        (best_allocations, conversion_bracket, trad_split, diagnostics).
    """
    cfg = cfg or OptimizerConfig()
    obj = _objective_for(scn, cfg)
    if cfg.policy_class == "glide":
        bounds = list(GLIDE_PARAM_BOUNDS)
    elif cfg.policy_class == "three_knot_glide":
        bounds = list(THREE_KNOT_GLIDE_PARAM_BOUNDS)
    elif cfg.policy_class == "bond_tent":
        bounds = list(BOND_TENT_REDUCED_PARAM_BOUNDS)
    elif cfg.policy_class == "cppi":
        bounds = list(CPPI_PARAM_BOUNDS)
    elif cfg.policy_class == "bodie_merton":
        bounds = list(BODIE_MERTON_REDUCED_PARAM_BOUNDS)
    elif cfg.policy_class == "multi_phase":
        bounds = list(MULTI_PHASE_PARAM_BOUNDS)
    elif cfg.policy_class == "vol_targeting":
        bounds = list(VOL_TARGETING_PARAM_BOUNDS)
    elif cfg.location_mode == "heuristic":
        bounds = [
            (0.0, 1.0), (0.0, 1.0),  # overall stock, bond
            (0.0, 5.0),              # conversion bracket index
            (0.0, 1.0),              # trad split
        ]
    else:
        bounds = [
            (0.0, 1.0), (0.0, 1.0),  # taxable (stock, bond)
            (0.0, 1.0), (0.0, 1.0),  # traditional
            (0.0, 1.0), (0.0, 1.0),  # roth
            (0.0, 5.0),              # conversion bracket index
            (0.0, 1.0),              # trad split fraction
        ]
    has_rental = bool(cfg.optimize_rental and scn.rental_property is not None)
    if has_rental:
        bounds = bounds + list(RENTAL_PARAM_BOUNDS)
    res = _run_search(obj, bounds, cfg)
    x_pol, x_rent = _split_x(res.x, has_rental)
    policy = _build_policy(x_pol, scn, cfg)
    rental_decoded = decode_rental(x_rent, scn.rental_property) \
        if has_rental and scn.rental_property is not None else None
    if cfg.policy_class in ("glide", "three_knot_glide", "bond_tent", "cppi",
                             "bodie_merton", "multi_phase", "vol_targeting"):
        # For glide policies, "current-year" allocations come from
        # policy.decide() at age 0.
        from .policy import StateSummary
        ss0 = StateSummary(
            age=scn.profile.age, year_idx=0,
            years_to_retirement=float(scn.profile.years_to_retirement()),
            fire_target_real=25.0 * scn.spending.annual_real,
            median_real_wealth=scn.initial_portfolio.total_value(),
            fire_progress_ratio=1.0,
        )
        d0 = policy.decide(ss0)
        allocations = d0.allocations
        conv_target = d0.conversion_bracket
        trad_split = d0.trad_contribution_split
    elif cfg.location_mode == "heuristic":
        allocations, conv_target, trad_split = _decode_heuristic(x_pol, scn)
    else:
        allocations, conv_target, trad_split = _decode_free(x_pol)
    if rental_decoded is not None:
        # Mutate the caller's scenario so subsequent simulate() calls
        # (CLI final-evaluation, per-mode re-eval) use the optimized
        # rental decision.
        scn.rental_property = rental_decoded
    diag = {"obj_value": float(res.fun), "nit": int(res.nit), "nfev": int(res.nfev),
            "x": res.x.tolist(), "message": str(res.message),
            "location_mode": cfg.location_mode,
            "rental": rental_decoded,
            "policy_class": cfg.policy_class,
            "algorithm": cfg.algorithm,
            "policy": policy}
    return allocations, conv_target, trad_split, diag
