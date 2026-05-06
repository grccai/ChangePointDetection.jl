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
                     WithdrawalPolicy)
from .location import heuristic_target_allocations
from .policy import (Policy, StaticPolicy, GlidePolicy,
                     build_glide_policy, GLIDE_PARAM_BOUNDS,
                     build_three_knot_glide_policy,
                     THREE_KNOT_GLIDE_PARAM_BOUNDS,
                     build_bond_tent_policy, BOND_TENT_PARAM_BOUNDS,
                     build_cppi_policy, CPPI_PARAM_BOUNDS,
                     build_bodie_merton_policy,
                     BODIE_MERTON_PARAM_BOUNDS)
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
        return build_bond_tent_policy(
            list(x), retirement_age=retirement_age, ss_age=ss_age)
    if cfg.policy_class == "cppi":
        return build_cppi_policy(
            list(x), retirement_age=retirement_age, ss_age=ss_age)
    if cfg.policy_class == "bodie_merton":
        return build_bodie_merton_policy(
            list(x), scn=scn_base,
            retirement_age=retirement_age, ss_age=ss_age)
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

    def __call__(self, x: np.ndarray) -> float:
        policy = _build_policy(x, self.scn_base, self.cfg)
        scn = deepcopy(self.scn_base)
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

    def __call__(self, x: np.ndarray) -> float:
        policy = _build_policy(x, self.scn_base, self.cfg)
        scn = deepcopy(self.scn_base)
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

    def __call__(self, x: np.ndarray) -> float:
        policy = _build_policy(x, self.scn_base, self.cfg)
        scn = deepcopy(self.scn_base)
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


def _objective_for(scn_base: Scenario, cfg: OptimizerConfig):
    if cfg.objective in ("fire_prob", "fire_prob_weighted"):
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
            )
        # fire_prob_weighted
        year_indices = [
            min(horizon, max(0, int(round(fire_age + i - start_age))))
            for i in range(11)
        ]
        weights = np.array([1.0 - i / 10.0 for i in range(11)])
        return _FIREProbWeightedObjective(
            scn_base=scn_base, cfg=cfg,
            fire_age=fire_age, fire_target_real=fire_target,
            ruin_max=cfg.ruin_max,
            year_indices=year_indices, weights=weights,
        )
    return _Objective(
        scn_base=scn_base, cfg=cfg,
        horizon=scn_base.profile.horizon(),
        years_to_retire=scn_base.profile.years_to_retirement(),
    )


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
        bounds = list(BOND_TENT_PARAM_BOUNDS)
    elif cfg.policy_class == "cppi":
        bounds = list(CPPI_PARAM_BOUNDS)
    elif cfg.policy_class == "bodie_merton":
        bounds = list(BODIE_MERTON_PARAM_BOUNDS)
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
    res = differential_evolution(
        obj, bounds=bounds, seed=cfg.seed, maxiter=cfg.maxiter,
        popsize=cfg.popsize, workers=cfg.workers, polish=cfg.polish,
        tol=1e-3, mutation=(0.5, 1.0), recombination=0.7,
        init="sobol", updating="deferred" if cfg.workers != 1 else "immediate",
    )
    policy = _build_policy(res.x, scn, cfg)
    if cfg.policy_class in ("glide", "three_knot_glide", "bond_tent", "cppi",
                             "bodie_merton"):
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
        allocations, conv_target, trad_split = _decode_heuristic(res.x, scn)
    else:
        allocations, conv_target, trad_split = _decode_free(res.x)
    diag = {"obj_value": float(res.fun), "nit": int(res.nit), "nfev": int(res.nfev),
            "x": res.x.tolist(), "message": res.message,
            "location_mode": cfg.location_mode,
            "policy_class": cfg.policy_class,
            "policy": policy}
    return allocations, conv_target, trad_split, diag
