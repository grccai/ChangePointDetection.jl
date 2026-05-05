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
                     build_glide_policy, GLIDE_PARAM_BOUNDS)
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
    # 'static' : single fixed Decision applied every year (legacy).
    # 'glide'  : 16 vars — per-account 2-knot glide path + life-phase
    #            conversion brackets + Trad/Roth split + wealth_responsiveness.
    #            Each year's decision depends on age and FIRE progress.
    policy_class: str = "static"


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
    if cfg.policy_class == "glide":
        start_age = scn_base.profile.age
        end_age = (scn_base.profile.end_of_plan_date
                   - scn_base.profile.birthdate).days / 365.25
        retirement_age = (scn_base.profile.retirement_date
                          - scn_base.profile.birthdate).days / 365.25
        ss_age = float(scn_base.social_security.claim_age)
        return build_glide_policy(list(x), start_age=start_age,
                                  end_age=end_age,
                                  retirement_age=retirement_age,
                                  ss_age=ss_age)
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


def _objective_for(scn_base: Scenario, cfg: OptimizerConfig) -> _Objective:
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
    if cfg.policy_class == "glide":
        # For glide, "current-year" allocations are policy.decide() at age 0.
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
