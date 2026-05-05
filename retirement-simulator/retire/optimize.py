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


def _decode(x: np.ndarray) -> tuple[TargetAllocations, float | None, float]:
    def alloc(s: float, b: float) -> Allocation:
        s = max(0.0, min(1.0, s))
        b = max(0.0, min(1.0 - s, b))
        c = max(0.0, 1.0 - s - b)
        return Allocation(stock=s, bond=b, cash=c)

    taxable = alloc(x[0], x[1])
    traditional = alloc(x[2], x[3])
    roth = alloc(x[4], x[5])

    # Snap conversion target to a discrete bracket
    candidates = [None, 0.10, 0.12, 0.22, 0.24, 0.32]
    idx = int(round(x[6]))
    idx = max(0, min(len(candidates) - 1, idx))
    conv_target = candidates[idx]

    trad_split = max(0.0, min(1.0, x[7]))
    return TargetAllocations(taxable, traditional, roth), conv_target, trad_split


def _crra(c: np.ndarray, gamma: float) -> np.ndarray:
    c = np.maximum(c, 1e-9)
    if abs(gamma - 1.0) < 1e-9:
        return np.log(c)
    return (c ** (1.0 - gamma)) / (1.0 - gamma)


def _objective_for(scn_base: Scenario, cfg: OptimizerConfig
                   ) -> Callable[[np.ndarray], float]:
    horizon = scn_base.profile.horizon()
    years_to_retire = scn_base.profile.years_to_retirement()

    def obj(x: np.ndarray) -> float:
        allocations, conv_target, trad_split = _decode(x)
        scn = deepcopy(scn_base)
        scn.target_allocations = allocations
        scn.withdrawal = WithdrawalPolicy(
            strategy=scn_base.withdrawal.strategy,
            roth_conversion_target_bracket=conv_target,
            aca_magi_cap=scn_base.withdrawal.aca_magi_cap,
        )
        # Apply contribution split
        c = scn.savings.contributions
        # Total 401k pool: keep sum(trad_401k + roth_401k) constant if both
        # are numeric; otherwise interpret 'max' literally on trad and 0 roth.
        try:
            tp = float(c.trad_401k) + float(c.roth_401k)
            c.trad_401k = tp * trad_split
            c.roth_401k = tp * (1.0 - trad_split)
        except (TypeError, ValueError):
            pass

        # Run a smaller MC for speed
        scn.simulation.n_paths = cfg.n_paths_inner
        # vary seed slightly each call to reduce variance of optimizer signal
        scn.simulation.seed = cfg.seed
        result = simulate(scn)

        # Build per-path discounted utility of consumption
        n = len(result.paths)
        util = np.zeros(n)
        for i, p in enumerate(result.paths):
            real_consump = (p.real_spending_by_year
                            - p.real_shortfall_by_year)
            real_consump = np.maximum(real_consump, 0.0)
            # Only discount/utility from retirement onwards
            betas = cfg.beta ** np.arange(horizon)
            betas[:years_to_retire] = 0.0  # ignore accumulation consumption
            u = _crra(np.maximum(real_consump, 1e-3), cfg.gamma)
            util[i] = float(np.sum(betas * u))
            # Bequest
            util[i] += cfg.bequest_weight * float(_crra(
                np.array([max(p.terminal_real_wealth, 1.0)]), cfg.gamma)[0])

        expected_util = float(np.mean(util))
        fail_pen = cfg.failure_penalty * result.failure_rate()
        # We *minimize*, so return negative utility plus penalty
        return -expected_util + fail_pen

    return obj


def optimize(scn: Scenario, cfg: OptimizerConfig | None = None
             ) -> tuple[TargetAllocations, float | None, float, dict]:
    """Run differential evolution. Returns
        (best_allocations, conversion_bracket, trad_split, diagnostics).
    """
    cfg = cfg or OptimizerConfig()
    obj = _objective_for(scn, cfg)
    # Bounds: allocations in [0,1] with stock+bond <= 1 enforced inside _decode
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
    allocations, conv_target, trad_split = _decode(res.x)
    diag = {"obj_value": float(res.fun), "nit": int(res.nit), "nfev": int(res.nfev),
            "x": res.x.tolist(), "message": res.message}
    return allocations, conv_target, trad_split, diag
