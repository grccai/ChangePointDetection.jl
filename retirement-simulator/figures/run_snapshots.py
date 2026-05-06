"""Run MC/bootstrap simulations for selected (policy, return_mode) pairs and
snapshot the lightweight arrays we need for visualization.

Usage:
    PYTHONPATH=. python figures/run_snapshots.py

Writes one .npz per (strategy, return_mode) into snapshots/.
Re-running overwrites; visualization scripts load these snapshots so we can
iterate on plots without re-running MC.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

from retire.config import load_scenario
from retire.simulate import simulate
from retire.policy import (
    StaticPolicy, build_bond_tent_policy, build_bodie_merton_policy,
)
from retire.config import Allocation, TargetAllocations


SNAP_DIR = os.path.join(os.path.dirname(__file__), "..", "snapshots")
os.makedirs(SNAP_DIR, exist_ok=True)


def static_policy_from_yaml(scn) -> StaticPolicy:
    """Use trial.yaml's target_allocations as the static baseline."""
    ta = scn.target_allocations
    return StaticPolicy(
        allocations=TargetAllocations(
            taxable=Allocation(stock=ta.taxable.stock, bond=ta.taxable.bond,
                                cash=ta.taxable.cash),
            traditional=Allocation(stock=ta.traditional.stock,
                                    bond=ta.traditional.bond,
                                    cash=ta.traditional.cash),
            roth=Allocation(stock=ta.roth.stock, bond=ta.roth.bond,
                             cash=ta.roth.cash),
        ),
        conversion_bracket=None,
        trad_contribution_split=1.0,
    )


def bond_tent_known_good(scn):
    """Hand-picked from the [gbm, historical_ath] robust optimum:
       stock_high~0.95, stock_low~0.484, tent_age~61.1 (~retire+6),
       taxable_cash~0.303."""
    retirement_age = scn.profile.retirement_age
    tent_offset = 61.1 - retirement_age
    # Layout: stock_high, stock_low, tent_age_offset, span, taxable_cash,
    #         conv_fire_idx, conv_ss_idx, trad_split, wealth_resp
    x = [0.95, 0.484, tent_offset, 15.0, 0.303, 3.0, 2.5, 1.0, 0.0]
    return build_bond_tent_policy(x, retirement_age=retirement_age)


def bond_tent_stretched_robust(scn):
    """[gbm, historical_stretched_ath] DE optimum (3328 evals, 25 gens).
    Tent at retirement, stock_low ~0%, strong wealth_responsiveness."""
    retirement_age = scn.profile.retirement_age
    tent_offset = 54.8 - retirement_age
    # conv_during_fire_gap = 0.12 -> idx ~0; conv_during_ss_window = 0.24 -> idx
    # we used 0.0 / 0.24 - skipping discrete bracket snap details, use
    # representative idx values.
    x = [0.9616, 0.0018, tent_offset, 10.3, 0.3004, 0.5, 1.0, 0.999, 1.477]
    return build_bond_tent_policy(x, retirement_age=retirement_age)


def bodie_merton_default(scn):
    """gamma=3, r_hc=3%, light taxable cash."""
    retirement_age = scn.profile.retirement_age
    # Layout: gamma, r_hc, taxable_cash, conv_fire_idx, conv_ss_idx, trad_split
    x = [3.0, 0.03, 0.05, 3.0, 2.5, 1.0]
    return build_bodie_merton_policy(x, scn, retirement_age=retirement_age)


STRATEGIES = {
    "static_baseline": static_policy_from_yaml,
    "bond_tent_robust": bond_tent_known_good,
    "bond_tent_stretched": bond_tent_stretched_robust,
    "bodie_merton": bodie_merton_default,
}

RETURN_MODES = ["gbm", "historical", "historical_ath", "historical_stretched_ath"]


def run_one(strategy_name: str, policy, scn, return_mode: str,
            n_paths: int, seed: int) -> str:
    scn.simulation.return_model = return_mode
    scn.simulation.n_paths = n_paths
    scn.simulation.seed = seed
    t0 = time.time()
    result = simulate(scn, policy=policy)
    dt = time.time() - t0
    P = result.n_paths
    H1 = len(result.paths[0].real_wealth_by_year)
    wealth = np.array([p.real_wealth_by_year for p in result.paths])  # (P, H+1)
    spend = np.array([p.real_spending_by_year for p in result.paths])  # (P, H)
    failed = np.array([p.failed for p in result.paths])
    ages = np.array([scn.profile.age_at_year(y) for y in range(H1)])
    bal = result.real_balance_by_year  # (P, H+1, 3, 3)
    out = os.path.join(SNAP_DIR, f"{strategy_name}__{return_mode}.npz")
    np.savez_compressed(out,
                        wealth=wealth.astype(np.float32),
                        spending=spend.astype(np.float32),
                        balances=bal.astype(np.float32) if bal is not None
                                  else np.zeros((P, H1, 3, 3), dtype=np.float32),
                        failed=failed,
                        ages=ages.astype(np.float32),
                        strategy=strategy_name,
                        return_mode=return_mode,
                        n_paths=P,
                        failure_rate=result.failure_rate())
    print(f"  {strategy_name:<20} x {return_mode:<28} "
          f"P={P} H={H1-1} ruin={100*result.failure_rate():5.2f}% "
          f"medW55=${np.median(wealth[:, 18]) / 1e6:.2f}M  "
          f"({dt:.1f}s)")
    return out


def main():
    n_paths = int(os.environ.get("SNAP_PATHS", "4000"))
    seed = int(os.environ.get("SNAP_SEED", "42"))
    print(f"Running snapshots: {len(STRATEGIES)} strategies x "
          f"{len(RETURN_MODES)} return modes, n_paths={n_paths}")
    cfg_path = os.path.join(os.path.dirname(__file__), "..",
                             "examples", "trial.yaml")
    for strat_name, build in STRATEGIES.items():
        # One scenario per strategy (so policy build sees fresh scenario)
        scn = load_scenario(cfg_path)
        policy = build(scn)
        for mode in RETURN_MODES:
            scn_m = load_scenario(cfg_path)
            run_one(strat_name, policy, scn_m, mode, n_paths, seed)
    print("Done.")


if __name__ == "__main__":
    main()
