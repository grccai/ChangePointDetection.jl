"""Pareto + seed-stability scatter for the v7+v8 multi-seed sweep.

Reads:
  figures/trial_rental_realistic_v8/evaluations.json

Produces:
  figures/trial_rental_realistic_v8/pareto.png
  figures/trial_rental_realistic_v8/seed_stability.png

Also includes prior-published v4 / v5 / v6 optima as background context
(loaded directly from figures/run_snapshots.py + earlier sources).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from retire.config import load_scenario
from retire.simulate import simulate
from figures.run_snapshots import (
    _bt_v4_gbm, _bt_v4_robust, _bm_v4_gbm, _bm_v4_robust,
    _bt_v5_gbm, _bt_v5_robust, _bm_v5_gbm, _bm_v5_robust,
    _override_rental,
)
from figures.plot_pareto import _bm_v6_gbm, _bt_v6_gbm, _bm_v6_robust, _bt_v6_robust


REPO = Path(__file__).resolve().parent.parent
EVAL_PATH = REPO / "figures" / "trial_rental_realistic_v8" / "evaluations.json"
OUT_DIR = REPO / "figures" / "trial_rental_realistic_v8"


PRIOR_OPTIMA = {
    # v4 (originally tuned on simple cost trial_rental.yaml)
    "v4 bt gbm":    (_bt_v4_gbm, dict(price_real=1_276_517, location_state="TX",
                       min_age=40.2, min_liquid=559_512, min_taxable=177_226), "v4"),
    "v4 bm gbm":    (_bm_v4_gbm, dict(price_real=1_483_607, location_state="CA",
                       min_age=40.8, min_liquid=772_992, min_taxable=209_262), "v4"),
    "v4 bt robust": (_bt_v4_robust, dict(price_real=850_270, location_state="TX",
                       min_age=44.9, min_liquid=807_710, min_taxable=401_488), "v4"),
    "v4 bm robust": (_bm_v4_robust, dict(price_real=858_081, location_state="CA",
                       min_age=40.7, min_liquid=769_316, min_taxable=382_221), "v4"),
    # v5 (DE single-seed on realistic)
    "v5 bt gbm":    (_bt_v5_gbm, dict(price_real=1_661_830, location_state="CA",
                       min_age=78.1, min_liquid=2_198_419, min_taxable=1_168_111), "v5"),
    "v5 bm gbm":    (_bm_v5_gbm, dict(price_real=556_298, location_state="TX",
                       min_age=40.9, min_liquid=775_369, min_taxable=215_241), "v5"),
    "v5 bt robust": (_bt_v5_robust, dict(price_real=1_781_768, location_state="TX",
                       min_age=58.1, min_liquid=4_666_373, min_taxable=1_671_295), "v5"),
    "v5 bm robust": (_bm_v5_robust, dict(price_real=636_309, location_state="OR",
                       min_age=78.6, min_liquid=1_880_187, min_taxable=731_982), "v5"),
    # v6 (BIPOP-CMA-ES on realistic)
    "v6 bm gbm BIPOP":   (_bm_v6_gbm, dict(price_real=1_434_566, location_state="CA",
                       min_age=93.9, min_liquid=2_005_048, min_taxable=671_295), "v6"),
    "v6 bt gbm BIPOP":   (_bt_v6_gbm, dict(price_real=1_923_378, location_state="OR",
                       min_age=81.8, min_liquid=3_107_121, min_taxable=1_771_284), "v6"),
    "v6 bm robust BIPOP":(_bm_v6_robust, dict(price_real=1_831_802, location_state="OR",
                       min_age=92.6, min_liquid=4_520_253, min_taxable=651_834), "v6"),
    "v6 bt robust BIPOP":(_bt_v6_robust, dict(price_real=619_581, location_state="TX",
                       min_age=83.8, min_liquid=2_578_299, min_taxable=1_757_844), "v6"),
}


def evaluate_prior_optimum(builder, rp_over, mc_seeds, n_paths=4000,
                            fire_age=55, fire_target=2_500_000):
    rw, rn = [], []
    for s in mc_seeds:
        scn = load_scenario("examples/trial_rental_realistic.yaml")
        _override_rental(scn, **rp_over)
        scn.simulation.seed = int(s); scn.simulation.n_paths = n_paths
        policy = builder(scn)
        result = simulate(scn, policy=policy)
        wealth = np.array([p.real_wealth_by_year for p in result.paths])
        weights = np.array([1.0 - i / 10.0 for i in range(11)])
        horizon = scn.profile.horizon(); start_age = scn.profile.age
        idxs = [min(horizon, max(0, int(round(fire_age + i - start_age))))
                 for i in range(11)]
        p_hit = np.array([(wealth[:, idx] >= fire_target).mean() for idx in idxs])
        rw.append(float((weights * p_hit).sum()))
        rn.append(float(result.failure_rate()))
    return np.array(rw), np.array(rn)


def main():
    n_seeds = int(os.environ.get("N_SEEDS", "30"))
    rng = np.random.default_rng(20260508)
    mc_seeds = rng.integers(1, 1_000_000, size=n_seeds)

    # ---- v7+v8 evaluations (if present) ----
    if not EVAL_PATH.exists():
        print(f"No {EVAL_PATH} yet — run figures/v7v8_analysis.py first.")
        return
    v7v8 = json.loads(EVAL_PATH.read_text())

    # ---- Re-evaluate priors on the same seeds ----
    print(f"Evaluating {len(PRIOR_OPTIMA)} prior optima on {n_seeds} seeds...")
    prior = {}
    for label, (build, rp_over, group) in PRIOR_OPTIMA.items():
        rw, rn = evaluate_prior_optimum(build, rp_over, mc_seeds)
        prior[label] = (rw, rn, group)
        print(f"  {label:<22}  reward={rw.mean():.4f}   ruin={100*rn.mean():.2f}%")

    # ---- Build the unified scatter ----
    fig, ax = plt.subplots(figsize=(13, 8))
    GROUP_COLOR = {
        "v4": "#9ecae1", "v5": "#fdae6b", "v6": "#c994c7",
        "v7": "#1f77b4", "v8": "#2ca02c",
    }
    GROUP_MARKER = {
        "v4": "^", "v5": "s", "v6": "D",
        "v7": "o", "v8": "*",
    }
    GROUP_SIZE = {
        "v4": 80, "v5": 70, "v6": 80,
        "v7": 100, "v8": 130,
    }

    # Plot priors (background)
    for label, (rw, rn, group) in prior.items():
        ax.errorbar(100*rn.mean(), rw.mean(),
                    xerr=100*rn.std(ddof=1)/np.sqrt(n_seeds),
                    yerr=rw.std(ddof=1)/np.sqrt(n_seeds),
                    fmt="none", ecolor="0.7", elinewidth=0.6, capsize=2)
        ax.scatter(100*rn.mean(), rw.mean(), marker=GROUP_MARKER[group],
                   s=GROUP_SIZE[group], color=GROUP_COLOR[group],
                   edgecolor="black", linewidth=0.4, alpha=0.55, zorder=2)

    # Plot v7+v8 (foreground)
    for d in v7v8:
        rw_mean = d["mean_reward"]; rn_mean = d["mean_ruin"]
        sem_r = d["sem_reward"]; sem_ru = d["sem_ruin"]
        group = "v7" if d["seed"] == 12345 else "v8"
        ax.errorbar(100*rn_mean, rw_mean, xerr=100*sem_ru, yerr=sem_r,
                    fmt="none", ecolor="0.5", elinewidth=0.8, capsize=2)
        ax.scatter(100*rn_mean, rw_mean, marker=GROUP_MARKER[group],
                   s=GROUP_SIZE[group], color=GROUP_COLOR[group],
                   edgecolor="black", linewidth=0.6, zorder=4)
        ax.annotate(d["label"], (100*rn_mean, rw_mean),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=7, color="0.15")

    # Pareto frontier across ALL points
    all_points = []
    for label, (rw, rn, _) in prior.items():
        all_points.append((100*rn.mean(), rw.mean(), label))
    for d in v7v8:
        all_points.append((100*d["mean_ruin"], d["mean_reward"], d["label"]))
    pareto = []
    for x, y, lbl in all_points:
        dom = any((x2 <= x and y2 > y) or (x2 < x and y2 >= y)
                   for x2, y2, _ in all_points if (x2, y2, lbl) != (x, y, lbl))
        if not dom:
            pareto.append((x, y, lbl))
    pareto.sort()
    if pareto:
        px = [p[0] for p in pareto]
        py = [p[1] for p in pareto]
        ax.plot(px, py, color="#d62728", lw=1.5, ls="-", alpha=0.7,
                label="Pareto frontier", zorder=3)
        # extend right
        right = max(p[0] for p in all_points) * 1.05
        ax.plot([px[-1], right], [py[-1], py[-1]],
                color="#d62728", lw=1.5, ls="-", alpha=0.35, zorder=3)

    ax.axvline(2.0, color="#d62728", lw=0.8, ls="--", alpha=0.4)
    ax.text(2.0, ax.get_ylim()[0] + 0.005, "2% ruin cap",
            color="#d62728", fontsize=9, ha="right", va="bottom")

    handles = [
        Line2D([0],[0], marker="^", color="w", markerfacecolor="#9ecae1",
               markeredgecolor="black", markersize=10, label="v4 (simple cost, prior)"),
        Line2D([0],[0], marker="s", color="w", markerfacecolor="#fdae6b",
               markeredgecolor="black", markersize=10, label="v5 (DE single-seed, prior)"),
        Line2D([0],[0], marker="D", color="w", markerfacecolor="#c994c7",
               markeredgecolor="black", markersize=10, label="v6 (BIPOP-CMA-ES, prior)"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#1f77b4",
               markeredgecolor="black", markersize=11, label="v7 (DE reduced-dim, default seed)"),
        Line2D([0],[0], marker="*", color="w", markerfacecolor="#2ca02c",
               markeredgecolor="black", markersize=14, label="v8 (DE reduced-dim, seeds 7/31/101)"),
        Line2D([0],[0], color="#d62728", lw=1.5, label="Pareto frontier"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8.5,
              frameon=True, facecolor="white", framealpha=0.95)

    ax.set_xlabel(f"P(ruin) on trial_rental_realistic.yaml  "
                  f"(mean over {n_seeds} MC seeds × 4000 paths each, error bars = ±1 SE)")
    ax.set_ylabel("Reward = Σᵢ (1 − i/10) · P(W_{55+i} ≥ $2.5M)   [max possible = 5.5]")
    ax.set_title(
        "v7+v8 reduced-dim multi-seed sweep vs prior published optima.\n"
        "v7 = DE single-seed at reduced dims; v8 = 3 additional seeds per config.",
        fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, 2.15)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "pareto.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")

    # ---- Seed-stability per config ----
    fig2, axes = plt.subplots(1, 4, figsize=(15, 5), sharey=True)
    configs = ["bm_gbm", "bt_gbm", "bm_robust", "bt_robust"]
    for ax2, config in zip(axes, configs):
        items = [d for d in v7v8 if d["config"] == config]
        items.sort(key=lambda d: -d["mean_reward"])
        labels = [f"seed={d['seed']}" for d in items]
        rewards = [d["mean_reward"] for d in items]
        ruins = [100*d["mean_ruin"] for d in items]
        sem_r = [d["sem_reward"] for d in items]
        x = np.arange(len(items))
        ax2.bar(x, rewards, yerr=sem_r, capsize=4,
                color=["#2ca02c" if r > 4.40 else "#d62728" for r in rewards],
                edgecolor="black", linewidth=0.5)
        for i, (r, ru) in enumerate(zip(rewards, ruins)):
            ax2.text(i, r + 0.005, f"{r:.3f}\nruin={ru:.2f}%",
                     ha="center", va="bottom", fontsize=7.5)
        ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8)
        ax2.set_title(config)
        ax2.grid(True, axis="y", alpha=0.3)
        ax2.axhline(4.40, color="0.4", lw=0.6, ls="--",
                    alpha=0.5)
    axes[0].set_ylabel("30-seed mean reward (±1 SE)")
    fig2.suptitle("v7+v8 seed-stability: reward across DE seeds, sorted best→worst",
                  fontsize=11)
    fig2.tight_layout()
    out2 = OUT_DIR / "seed_stability.png"
    fig2.savefig(out2, dpi=140, bbox_inches="tight")
    plt.close(fig2)
    print(f"Wrote {out2}")


if __name__ == "__main__":
    main()
