"""Run MC/bootstrap simulations for selected (policy, return_mode) pairs and
snapshot the lightweight arrays we need for visualization.

Usage:
    PYTHONPATH=. python figures/run_snapshots.py
    SCENARIO_ID=trial_zero_floor SCENARIO_YAML=examples/trial_zero_floor.yaml \
        PYTHONPATH=. python figures/run_snapshots.py

Writes one .npz per (strategy, return_mode) into snapshots/{SCENARIO_ID}/.
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
from retire.config import (Allocation, TargetAllocations, RentalProperty,
                            RentalPurchaseTrigger)


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _override_rental(scn, *, price_real: float, location_state: str,
                      min_age: float, min_liquid: float,
                      min_taxable: float):
    """Replace scn.rental_property with v4-optimized fields. Keeps the
    other RentalProperty fields (cap_rate, expense_ratio, mortgage rate,
    etc.) as configured in the YAML so we only override the decision
    variables."""
    if scn.rental_property is None:
        return  # no rental in scenario; no-op
    rp = scn.rental_property
    from dataclasses import replace
    scn.rental_property = replace(
        rp,
        price_real=price_real,
        location_state=location_state,
        trigger=RentalPurchaseTrigger(
            min_age=min_age,
            min_liquid_real_wealth=min_liquid,
            min_taxable_real_wealth=min_taxable,
        ),
    )


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

# v4 optimized strategies for the trial_rental scenario. Each value is a
# (policy_builder, rental_override_dict) tuple. Selected when
# SCENARIO_ID="trial_rental_v4".
def _bt_v4_gbm(scn):
    """v4 bond_tent + rental on GBM, fire_prob_weighted optimum."""
    retirement_age = scn.profile.retirement_age
    tent_offset = 68.8 - retirement_age
    # bond_tent layout: stock_high, stock_low, tent_age_offset, span,
    #                   taxable_cash, conv_fire_idx, conv_ss_idx, trad_split,
    #                   wealth_resp. Conv indices: 0=None, 1=10%, 2=12%,
    #                   3=22%, 4=24%, 5=32%.
    x = [0.8085, 0.1667, tent_offset, 29.2, 0.40, 0.0, 2.0, 0.819, 0.215]
    return build_bond_tent_policy(x, retirement_age=retirement_age)


def _bt_v4_robust(scn):
    """v4 bond_tent + rental robust [gbm,hist] optimum."""
    retirement_age = scn.profile.retirement_age
    tent_offset = 58.3 - retirement_age
    x = [0.8078, 0.5391, tent_offset, 7.8, 0.2353, 4.0, 4.0, 0.934, 0.209]
    return build_bond_tent_policy(x, retirement_age=retirement_age)


def _bm_v4_gbm(scn):
    """v4 bodie_merton + rental GBM optimum (reward 4.555).

    Layout: gamma, r_hc, taxable_cash, conv_fire_idx, conv_ss_idx,
    trad_split. The optimum's Merton target was 39.10% of total wealth;
    back-solving (mu-rf)/(gamma*sigma^2) at the trial market (mu=6%,
    rf=0.5%, sigma=18%) gives gamma=4.34. r_hc=3% real (default; the
    optimizer's choice on r_hc isn't printed but it only affects HC
    trajectory shape, not the Merton constant)."""
    retirement_age = scn.profile.retirement_age
    x = [4.34, 0.03, 0.168, 3.0, 5.0, 0.951]
    return build_bodie_merton_policy(x, scn, retirement_age=retirement_age)


def _bm_v4_robust(scn):
    """v4 bodie_merton + rental robust [gbm,hist] optimum (worst-case
    reward 4.521; ruin 0.38%/1.24%). The optimum's Merton constant was
    50.95% target stock-of-total — back-solving gamma = 3.33 at the
    trial market. r_hc = 3% (default; not printed)."""
    retirement_age = scn.profile.retirement_age
    x = [3.33, 0.03, 0.1301, 3.0, 2.0, 0.9622]
    return build_bodie_merton_policy(x, scn, retirement_age=retirement_age)


STRATEGIES_V4_RENTAL = {
    "static_baseline": (static_policy_from_yaml, None),
    "bond_tent_v4_gbm": (
        _bt_v4_gbm,
        dict(price_real=1_276_517, location_state="TX",
             min_age=40.2, min_liquid=559_512, min_taxable=177_226),
    ),
    "bodie_merton_v4_gbm": (
        _bm_v4_gbm,
        dict(price_real=1_483_607, location_state="CA",
             min_age=40.8, min_liquid=772_992, min_taxable=209_262),
    ),
    "bond_tent_v4_robust": (
        _bt_v4_robust,
        dict(price_real=850_270, location_state="TX",
             min_age=44.9, min_liquid=807_710, min_taxable=401_488),
    ),
    "bodie_merton_v4_robust": (
        _bm_v4_robust,
        dict(price_real=858_081, location_state="CA",
             min_age=40.7, min_liquid=769_316, min_taxable=382_221),
    ),
}

RETURN_MODES = ["gbm", "historical", "historical_ath", "historical_stretched_ath"]


def run_one(strategy_name: str, policy, scn, return_mode: str,
            n_paths: int, seed: int, snap_dir: str) -> str:
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
    eq = result.real_equity_by_year      # (P, H+1) — rental equity
    if eq is None:
        eq = np.zeros((P, H1), dtype=np.float32)
    out = os.path.join(snap_dir, f"{strategy_name}__{return_mode}.npz")
    np.savez_compressed(out,
                        wealth=wealth.astype(np.float32),
                        equity=eq.astype(np.float32),
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
    scenario_id = os.environ.get("SCENARIO_ID", "trial")
    scenario_yaml = os.environ.get("SCENARIO_YAML",
                                    os.path.join("examples", "trial.yaml"))
    cfg_path = scenario_yaml if os.path.isabs(scenario_yaml) \
        else os.path.join(REPO_ROOT, scenario_yaml)
    snap_dir = os.path.join(REPO_ROOT, "snapshots", scenario_id)
    os.makedirs(snap_dir, exist_ok=True)
    # SCENARIO_ID="trial_rental_v4" -> use the v4 rental-aware strategy set
    # (each strategy ships both a policy and a rental override dict).
    use_v4 = scenario_id.endswith("_v4")
    strategies = STRATEGIES_V4_RENTAL if use_v4 else \
                  {k: (v, None) for k, v in STRATEGIES.items()}
    print(f"Running snapshots: {len(strategies)} strategies x "
          f"{len(RETURN_MODES)} return modes, n_paths={n_paths}")
    print(f"  scenario_id={scenario_id}  yaml={cfg_path}")
    print(f"  snap_dir={snap_dir}")
    for strat_name, (build, rental_override) in strategies.items():
        scn = load_scenario(cfg_path)
        if rental_override is not None:
            _override_rental(scn, **rental_override)
        policy = build(scn)
        for mode in RETURN_MODES:
            scn_m = load_scenario(cfg_path)
            if rental_override is not None:
                _override_rental(scn_m, **rental_override)
            run_one(strat_name, policy, scn_m, mode, n_paths, seed, snap_dir)
    print("Done.")


if __name__ == "__main__":
    main()
