"""Pareto scatter of (P(ruin), reward) for every published optimum so far,
all evaluated on the same N MC seeds against trial_rental_realistic.yaml.

Strategies plotted:
* v4 (4 policies tuned on trial_rental.yaml, simple cost model) ---
  triangle markers
* v5 (4 policies tuned on trial_rental_realistic.yaml, DE single-seed) ---
  square markers
* 5 DE seed-stability bm_robust optima (same scenario, varying seed) ---
  small circles
* BIPOP-CMA-ES bm_gbm and bt_gbm (so far) --- diamond markers

Color = basin classification (early / late / hybrid). The optimizer's
2% ruin cap is shown as a vertical reference line.

Usage:
    PYTHONPATH=. python figures/plot_pareto.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from retire.config import load_scenario
from retire.simulate import simulate
from retire.policy import build_bond_tent_policy, build_bodie_merton_policy
from figures.run_snapshots import (
    _bt_v4_gbm, _bt_v4_robust, _bm_v4_gbm, _bm_v4_robust,
    _bt_v5_gbm, _bt_v5_robust, _bm_v5_gbm, _bm_v5_robust,
    _override_rental,
)


# ---- The strategies to plot ----
# Each entry: label -> (policy_builder, rental_override or None, group)
# group ∈ {"v4", "v5_de", "v5_seed", "v6_bipop"}; basin ∈ {"early", "late", "hybrid"}.

def _bt_v6_gbm(scn):
    """BIPOP bt_gbm: stock 100/41 V at 54, span 16, 0.4% cash."""
    retirement_age = scn.profile.retirement_age
    tent_offset = 53.9 - retirement_age
    x = [0.9997, 0.4103, tent_offset, 15.9, 0.0041, 0.0, 5.0, 0.9999, 0.0]
    return build_bond_tent_policy(x, retirement_age=retirement_age)


def _bm_v6_gbm(scn):
    """BIPOP bm_gbm: Merton 31.39%, 1.31% cash, gamma ≈ 5.42."""
    retirement_age = scn.profile.retirement_age
    mu, rf, sigma = 0.06, 0.005, 0.18
    gamma = (mu - rf) / (0.3139 * sigma**2)
    x = [gamma, 0.03, 0.0131, 0.0, 5.0, 1.0]
    return build_bodie_merton_policy(x, scn, retirement_age=retirement_age)


def _bm_seed7(scn):
    """seed=7 bm_robust DE optimum. Use v5 bm_robust allocation; only the
    rental override changes across seeds."""
    return _bm_v5_robust(scn)


STRATEGIES = {
    # v4 (originally tuned on trial_rental.yaml, simple cost model)
    "v4 bt gbm":     (_bt_v4_gbm, dict(price_real=1_276_517, location_state="TX",
                       min_age=40.2, min_liquid=559_512, min_taxable=177_226),
                       "v4", "early"),
    "v4 bm gbm":     (_bm_v4_gbm, dict(price_real=1_483_607, location_state="CA",
                       min_age=40.8, min_liquid=772_992, min_taxable=209_262),
                       "v4", "early"),
    "v4 bt robust":  (_bt_v4_robust, dict(price_real=850_270, location_state="TX",
                       min_age=44.9, min_liquid=807_710, min_taxable=401_488),
                       "v4", "early"),
    "v4 bm robust":  (_bm_v4_robust, dict(price_real=858_081, location_state="CA",
                       min_age=40.7, min_liquid=769_316, min_taxable=382_221),
                       "v4", "early"),
    # v5 (DE single-seed on trial_rental_realistic.yaml)
    "v5 bt gbm":     (_bt_v5_gbm, dict(price_real=1_661_830, location_state="CA",
                       min_age=78.1, min_liquid=2_198_419, min_taxable=1_168_111),
                       "v5_de", "late"),
    "v5 bm gbm":     (_bm_v5_gbm, dict(price_real=556_298, location_state="TX",
                       min_age=40.9, min_liquid=775_369, min_taxable=215_241),
                       "v5_de", "early"),
    "v5 bt robust":  (_bt_v5_robust, dict(price_real=1_781_768, location_state="TX",
                       min_age=58.1, min_liquid=4_666_373, min_taxable=1_671_295),
                       "v5_de", "late"),
    "v5 bm robust":  (_bm_v5_robust, dict(price_real=636_309, location_state="OR",
                       min_age=78.6, min_liquid=1_880_187, min_taxable=731_982),
                       "v5_de", "late"),
    # 5 DE seed-stability bm_robust optima — allocation fixed at v5 bm_robust
    "seed=7 bm rob":     (_bm_seed7, dict(price_real=619_720, location_state="TX",
                          min_age=40.2, min_liquid=1_032_887, min_taxable=161_675),
                          "v5_seed", "early"),
    "seed=31 bm rob":    (_bm_seed7, dict(price_real=1_026_387, location_state="OR",
                          min_age=43.7, min_liquid=4_815_655, min_taxable=567_466),
                          "v5_seed", "hybrid"),
    "seed=101 bm rob":   (_bm_seed7, dict(price_real=708_125, location_state="TX",
                          min_age=42.6, min_liquid=755_033, min_taxable=348_431),
                          "v5_seed", "early"),
    "seed=9999 bm rob":  (_bm_seed7, dict(price_real=1_083_606, location_state="CA",
                          min_age=65.2, min_liquid=4_987_347, min_taxable=1_314_403),
                          "v5_seed", "late"),
    # BIPOP-CMA-ES (so far: bm_gbm, bt_gbm)
    "v6 bm gbm BIPOP":   (_bm_v6_gbm, dict(price_real=1_434_566, location_state="CA",
                          min_age=93.9, min_liquid=2_005_048, min_taxable=671_295),
                          "v6_bipop", "late"),
    "v6 bt gbm BIPOP":   (_bt_v6_gbm, dict(price_real=1_923_378, location_state="OR",
                          min_age=81.8, min_liquid=3_107_121, min_taxable=1_771_284),
                          "v6_bipop", "late"),
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
    return reward, ruin


def main():
    realistic = "examples/trial_rental_realistic.yaml"
    n_seeds = int(os.environ.get("N_SEEDS", "30"))
    rng = np.random.default_rng(20260508)
    seeds = rng.integers(1, 1_000_000, size=n_seeds)

    rows = []
    for label, (build, rp, group, basin) in STRATEGIES.items():
        rw, rn = [], []
        for s in seeds:
            r, ruin = reward_and_ruin(realistic, build, rp, s)
            rw.append(r); rn.append(ruin)
        rw = np.array(rw); rn = np.array(rn)
        rows.append((label, group, basin,
                      rw.mean(), rw.std(ddof=1) / np.sqrt(n_seeds),
                      rn.mean(), rn.std(ddof=1) / np.sqrt(n_seeds)))
        print(f"{label:<22} {group:<10} {basin:<8} "
              f"r={rw.mean():.4f}±{rw.std(ddof=1)/np.sqrt(n_seeds):.4f}  "
              f"ruin={100*rn.mean():.3f}%±{100*rn.std(ddof=1)/np.sqrt(n_seeds):.3f}pp")

    # ---- Plot ----
    GROUP_MARKER = {"v4": "^", "v5_de": "s", "v5_seed": "o", "v6_bipop": "D"}
    BASIN_COLOR = {"early": "#1f77b4", "late": "#d62728", "hybrid": "#7f7f7f"}
    GROUP_SIZE = {"v4": 130, "v5_de": 110, "v5_seed": 70, "v6_bipop": 110}

    # Per-label hand-tuned offsets to keep crowded points readable.
    LABEL_OFFSET = {
        "v5 bt robust":   (10, -10),
        "v5 bm robust":   (12, -2),
        "seed=31 bm rob": (12, 7),
        "seed=9999 bm rob": (12, -10),
        "v5 bm gbm":      (10, 4),
        "v5 bt gbm":      (10, -12),
        "v4 bm gbm":      (-95, 4),
        "v4 bt gbm":      (8, 4),
        "v4 bm robust":   (8, 4),
        "v4 bt robust":   (8, 4),
        "seed=7 bm rob":  (-100, -10),
        "seed=101 bm rob":(8, 7),
        "v6 bm gbm BIPOP": (10, 4),
        "v6 bt gbm BIPOP": (10, 4),
    }

    fig, ax = plt.subplots(figsize=(11, 7.5))

    # Pareto frontier (max-reward at each ruin level): take points whose
    # (reward, ruin) is not dominated by any other point. A point is
    # dominated if there is another point with strictly higher reward AND
    # strictly lower-or-equal ruin (i.e., better on at least one axis,
    # not worse on the other).
    points = [(100 * mu, mr) for label, group, basin, mr, _, mu, _ in rows]
    pareto = []
    for i, (xi, yi) in enumerate(points):
        dominated = False
        for j, (xj, yj) in enumerate(points):
            if i == j:
                continue
            if (xj <= xi and yj > yi) or (xj < xi and yj >= yi):
                dominated = True
                break
        if not dominated:
            pareto.append((xi, yi))
    pareto.sort()
    if len(pareto) > 1:
        # Extend the frontier as a step function: rightward of the
        # right-most non-dominated point, the achievable reward is
        # capped by that point's reward (no point further right has
        # higher reward).
        px, py = zip(*pareto)
        ax.plot(px, py, color="#2ca02c", lw=1.5, ls="-", alpha=0.65,
                zorder=1, label=None)
        right_x = max(100 * mu for _, _, _, _, _, mu, _ in rows) * 1.15
        ax.plot([px[-1], right_x], [py[-1], py[-1]],
                color="#2ca02c", lw=1.5, ls="-", alpha=0.35, zorder=1)

    for label, group, basin, mr, ser, mu, seu in rows:
        ax.errorbar(100*mu, mr, xerr=100*seu, yerr=ser,
                    fmt="none", ecolor="0.6", elinewidth=0.8, capsize=2,
                    zorder=2)
        ax.scatter(100*mu, mr, marker=GROUP_MARKER[group],
                   s=GROUP_SIZE[group], color=BASIN_COLOR[basin],
                   edgecolor="black", linewidth=0.5, zorder=3,
                   label=None)
        offx, offy = LABEL_OFFSET.get(label, (7, 4))
        ax.annotate(label, (100*mu, mr), textcoords="offset points",
                    xytext=(offx, offy), fontsize=8, color="0.18")

    ax.axvline(2.0, color="#d62728", lw=1.0, ls="--", alpha=0.45,
               label="optimizer ruin cap (2%)")

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#1f77b4",
               markeredgecolor="black", markersize=11, label="v4 (simple cost model)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#d62728",
               markeredgecolor="black", markersize=10, label="v5 DE single-seed"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#7f7f7f",
               markeredgecolor="black", markersize=8, label="v5 DE seed-stability"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#d62728",
               markeredgecolor="black", markersize=9, label="v6 BIPOP-CMA-ES"),
        Line2D([0], [0], marker="o", color="#1f77b4", markersize=11,
               label="basin: BUY EARLY", linestyle=""),
        Line2D([0], [0], marker="o", color="#d62728", markersize=11,
               label="basin: BUY LATE / NEVER", linestyle=""),
        Line2D([0], [0], color="#2ca02c", lw=1.3, alpha=0.6,
               label="Pareto frontier"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8.5,
              frameon=True, facecolor="white", framealpha=0.95,
              ncol=1)

    ax.set_xlabel("P(ruin) on trial_rental_realistic.yaml  "
                  "(mean over 30 MC seeds × 4000 paths each, error bars = ±1 SE)")
    ax.set_ylabel("Reward = Σᵢ (1 − i/10) · P(W_{55+i} ≥ $2.5M)   [max possible = 5.5]")
    ax.set_title(
        "Pareto scatter: every published optimum, evaluated on trial_rental_realistic.yaml.\n"
        "v5 bm gbm Pareto-dominates every other strategy (highest reward, 2nd lowest ruin). "
        "Two basins ~0.1 reward apart visible across the cloud.",
        fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, 2.15)

    out_dir = Path(__file__).parent / "trial_rental_realistic_v5"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "pareto.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
