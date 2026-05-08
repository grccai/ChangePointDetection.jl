"""Repeatability + significance check for v4 vs v5 optimization findings.

For each (policy class) we evaluate BOTH the v4 optimum (tuned on
trial_rental.yaml) and the v5 optimum (tuned on
trial_rental_realistic.yaml) AGAINST the realistic scenario, using
N independent MC seeds. The realistic scenario is the more honest
test ground because it includes management fees, capex shocks,
turnover, and refinance.

Per (strategy, mode) we compute:
  reward[seed] = sum_{i=0..10} (1 - i/10) * P(W_{55+i} >= $2.5M)
  ruin[seed]   = path failure rate over the full plan

Then per pair (v4-policy, v5-policy):
  delta_reward = reward_v5[seed] - reward_v4[seed]   (paired)
  paired t-test on H0: mean(delta) = 0
  Wilcoxon signed-rank as a non-parametric backup
"""
from __future__ import annotations

import os
import sys
import time
from copy import deepcopy
from dataclasses import replace

import numpy as np
from scipy import stats

from retire.config import load_scenario, RentalPurchaseTrigger
from retire.simulate import simulate

# Import policy builders from the snapshot script (so we use the same
# parameters that produced the published optima).
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from figures.run_snapshots import (
    _bt_v4_gbm, _bt_v4_robust, _bm_v4_gbm, _bm_v4_robust,
    _bt_v5_gbm, _bt_v5_robust, _bm_v5_gbm, _bm_v5_robust,
    _override_rental,
)


# Per-policy-class pairs: (v4_builder, v4_rental_override, v5_builder, v5_override)
PAIRS = {
    "bond_tent_gbm": (
        _bt_v4_gbm,
        dict(price_real=1_276_517, location_state="TX",
             min_age=40.2, min_liquid=559_512, min_taxable=177_226),
        _bt_v5_gbm,
        dict(price_real=1_661_830, location_state="CA",
             min_age=78.1, min_liquid=2_198_419, min_taxable=1_168_111),
    ),
    "bodie_merton_gbm": (
        _bm_v4_gbm,
        dict(price_real=1_483_607, location_state="CA",
             min_age=40.8, min_liquid=772_992, min_taxable=209_262),
        _bm_v5_gbm,
        dict(price_real=556_298, location_state="TX",
             min_age=40.9, min_liquid=775_369, min_taxable=215_241),
    ),
    "bond_tent_robust": (
        _bt_v4_robust,
        dict(price_real=850_270, location_state="TX",
             min_age=44.9, min_liquid=807_710, min_taxable=401_488),
        _bt_v5_robust,
        dict(price_real=1_781_768, location_state="TX",
             min_age=58.1, min_liquid=4_666_373, min_taxable=1_671_295),
    ),
    "bodie_merton_robust": (
        _bm_v4_robust,
        dict(price_real=858_081, location_state="CA",
             min_age=40.7, min_liquid=769_316, min_taxable=382_221),
        _bm_v5_robust,
        dict(price_real=636_309, location_state="OR",
             min_age=78.6, min_liquid=1_880_187, min_taxable=731_982),
    ),
}


def reward_and_ruin(scn_yaml, build_policy, rental_override, seed,
                     fire_age=55, fire_target=2_500_000, n_paths=4000):
    scn = load_scenario(scn_yaml)
    if rental_override is not None:
        _override_rental(scn, **rental_override)
    scn.simulation.seed = int(seed)
    scn.simulation.n_paths = n_paths
    policy = build_policy(scn)
    result = simulate(scn, policy=policy)

    horizon = scn.profile.horizon()
    start_age = scn.profile.age
    year_indices = [min(horizon, max(0, int(round(fire_age + i - start_age))))
                     for i in range(11)]
    weights = np.array([1.0 - i / 10.0 for i in range(11)])
    wealth = np.array([p.real_wealth_by_year for p in result.paths])
    p_hit = np.array([(wealth[:, idx] >= fire_target).mean()
                       for idx in year_indices])
    reward = float((weights * p_hit).sum())
    ruin = float(result.failure_rate())
    term_med = float(np.median([p.terminal_real_wealth for p in result.paths]))
    return reward, ruin, term_med


def main():
    realistic = "examples/trial_rental_realistic.yaml"
    n_seeds = int(os.environ.get("N_SEEDS", "30"))
    rng = np.random.default_rng(20260508)
    seeds = rng.integers(1, 1_000_000, size=n_seeds)
    print(f"Repeatability check: {n_seeds} seeds, n_paths=4000, "
          f"scenario={realistic}\n")
    print(f"{'pair':<22}  {'reward_v4 (mean ± SE)':<24}  "
          f"{'reward_v5 (mean ± SE)':<24}  "
          f"{'Δ reward':<22}  {'paired t / p':<20}  Wilcoxon")
    print("=" * 145)
    # Stash ruin arrays for the second-pass ruin paired test below
    ruin_results = {}
    for name, (build4, rp4, build5, rp5) in PAIRS.items():
        rw4, rn4, tm4 = [], [], []
        rw5, rn5, tm5 = [], [], []
        for s in seeds:
            r4, ru4, t4 = reward_and_ruin(realistic, build4, rp4, s)
            r5, ru5, t5 = reward_and_ruin(realistic, build5, rp5, s)
            rw4.append(r4); rn4.append(ru4); tm4.append(t4)
            rw5.append(r5); rn5.append(ru5); tm5.append(t5)
        rw4 = np.array(rw4); rw5 = np.array(rw5)
        delta = rw5 - rw4
        se4 = rw4.std(ddof=1) / np.sqrt(n_seeds)
        se5 = rw5.std(ddof=1) / np.sqrt(n_seeds)
        t_stat = delta.mean() / (delta.std(ddof=1) / np.sqrt(n_seeds))
        p_val = float(2 * (1 - stats.t.cdf(abs(t_stat), df=n_seeds - 1)))
        try:
            w_stat, w_p = stats.wilcoxon(delta)
            w_str = f"p={w_p:.4f}"
        except ValueError:
            w_str = "n/a"
        print(f"{name:<22}  "
              f"{rw4.mean():.4f} ± {se4:.4f}        "
              f"{rw5.mean():.4f} ± {se5:.4f}        "
              f"{delta.mean():+.4f} (σ={delta.std(ddof=1):.4f})   "
              f"t={t_stat:+.2f}, p={p_val:.4f}  {w_str}")
        ruin_results[name] = (np.array(rn4), np.array(rn5))

    print()
    print(f"{'pair':<22}  {'ruin_v4 (mean ± SE)':<24}  "
          f"{'ruin_v5 (mean ± SE)':<24}  "
          f"{'Δ ruin (pp)':<22}  {'paired t / p':<20}  Wilcoxon")
    print("=" * 145)
    for name, (rn4, rn5) in ruin_results.items():
        delta_r = rn5 - rn4
        se_r4 = rn4.std(ddof=1) / np.sqrt(n_seeds)
        se_r5 = rn5.std(ddof=1) / np.sqrt(n_seeds)
        if delta_r.std(ddof=1) > 0:
            t_r = delta_r.mean() / (delta_r.std(ddof=1) / np.sqrt(n_seeds))
            p_r = float(2 * (1 - stats.t.cdf(abs(t_r), df=n_seeds - 1)))
            try:
                _, w_pr = stats.wilcoxon(delta_r)
                w_str_r = f"p={w_pr:.4f}"
            except ValueError:
                w_str_r = "n/a"
        else:
            t_r, p_r, w_str_r = float("nan"), float("nan"), "n/a"
        print(f"{name:<22}  "
              f"{100*rn4.mean():.3f}% ± {100*se_r4:.3f}pp        "
              f"{100*rn5.mean():.3f}% ± {100*se_r5:.3f}pp        "
              f"{100*delta_r.mean():+.3f}pp (σ={100*delta_r.std(ddof=1):.3f}pp) "
              f"t={t_r:+.2f}, p={p_r:.4f}  {w_str_r}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n[{time.time()-t0:.1f}s]")
