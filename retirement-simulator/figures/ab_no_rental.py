"""Head-to-head: each v7/v8 strategy vs a "never buy" counterfactual.

For each evaluated v7/v8 strategy (same 30 MC seeds × 4000 paths as the
main sweep), runs a counterfactual where the rental purchase trigger
never fires (trigger.min_age set to 200 — past plan-end-age 95). The
allocation policy is unchanged; only the rental decision differs.

Outputs:
  - paired t-test on Δreward, Δruin per strategy
  - bar plot of Δreward and Δruin with error bars
  - summary table

This isolates the rental's value-add for a fixed allocation. Two
caveats:
  1) The optimizer's allocation was tuned WITH the rental option
     available, so the counterfactual is "same alloc, force never-buy"
     not "best alloc given no rental". The latter would require
     re-optimization.
  2) The cash sleeve in the allocation may be sized for the rental
     down-payment; carrying it under "never-buy" is mildly wasteful
     but the comparison is still apples-to-apples on the SAME alloc.
"""
from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from retire.config import load_scenario
from retire.simulate import simulate
from figures.run_snapshots import _override_rental
from figures.v7v8_analysis import (
    _parse_v7_log, _parse_v8_json, _build_policy_from_strategy,
)


REPO = Path(__file__).resolve().parent.parent


def evaluate_pair(strat, mc_seeds, n_paths=4000):
    """Return (rw_with, rn_with, rw_never, rn_never), all length-N arrays.
    Same MC seeds for both arms — paired comparison."""
    rw_w, rn_w, rw_n, rn_n = [], [], [], []
    for s in mc_seeds:
        # Arm 1: with optimized rental
        scn = load_scenario("examples/trial_rental_realistic.yaml")
        _override_rental(scn, **strat.rental_override)
        scn.simulation.seed = int(s); scn.simulation.n_paths = n_paths
        policy = _build_policy_from_strategy(strat, scn)
        result = simulate(scn, policy=policy)
        wealth = np.array([p.real_wealth_by_year for p in result.paths])
        weights = np.array([1.0 - i / 10.0 for i in range(11)])
        horizon = scn.profile.horizon(); start_age = scn.profile.age
        idxs = [min(horizon, max(0, int(round(55 + i - start_age))))
                 for i in range(11)]
        p_hit = np.array([(wealth[:, idx] >= 2_500_000).mean() for idx in idxs])
        rw_w.append(float((weights * p_hit).sum()))
        rn_w.append(float(result.failure_rate()))

        # Arm 2: never buy (trigger.min_age = 200 so no path ever fires)
        scn2 = load_scenario("examples/trial_rental_realistic.yaml")
        rp_never = dict(strat.rental_override); rp_never["min_age"] = 200.0
        _override_rental(scn2, **rp_never)
        scn2.simulation.seed = int(s); scn2.simulation.n_paths = n_paths
        policy2 = _build_policy_from_strategy(strat, scn2)
        result2 = simulate(scn2, policy=policy2)
        wealth2 = np.array([p.real_wealth_by_year for p in result2.paths])
        p_hit2 = np.array([(wealth2[:, idx] >= 2_500_000).mean() for idx in idxs])
        rw_n.append(float((weights * p_hit2).sum()))
        rn_n.append(float(result2.failure_rate()))
    return (np.array(rw_w), np.array(rn_w),
             np.array(rw_n), np.array(rn_n))


def paired_t(delta):
    if delta.std(ddof=1) <= 0:
        return float("nan"), float("nan")
    n = len(delta)
    t = delta.mean() / (delta.std(ddof=1) / np.sqrt(n))
    p = 2 * (1 - stats.t.cdf(abs(t), df=n - 1))
    return float(t), float(p)


def main():
    n_seeds = int(os.environ.get("N_SEEDS", "30"))
    rng = np.random.default_rng(20260508)
    mc_seeds = rng.integers(1, 1_000_000, size=n_seeds)

    # Load v7+v8 strategies
    strategies = []
    for cfg in ["bm_gbm", "bt_gbm", "bm_robust", "bt_robust"]:
        s = _parse_v7_log(Path(f"/tmp/opt_v7_{cfg}.log"))
        if s is not None:
            strategies.append(s)
    for cfg in ["bm_gbm", "bt_gbm", "bm_robust", "bt_robust"]:
        for seed in [7, 101, 31]:
            s = _parse_v8_json(Path(f"/tmp/opt_v8_{cfg}_seed{seed}.json"))
            if s is not None:
                strategies.append(s)
    print(f"Evaluating {len(strategies)} strategies × 2 arms × "
          f"{n_seeds} MC seeds × 4000 paths.\n")

    rows = []
    for strat in strategies:
        rw_w, rn_w, rw_n, rn_n = evaluate_pair(strat, mc_seeds)
        d_rw = rw_w - rw_n
        d_rn = rn_w - rn_n
        t_rw, p_rw = paired_t(d_rw)
        t_rn, p_rn = paired_t(d_rn)
        rows.append(dict(
            label=strat.label, config=strat.config,
            rw_with=float(rw_w.mean()), rw_with_se=float(rw_w.std(ddof=1)/np.sqrt(n_seeds)),
            rw_never=float(rw_n.mean()), rw_never_se=float(rw_n.std(ddof=1)/np.sqrt(n_seeds)),
            d_rw=float(d_rw.mean()), d_rw_se=float(d_rw.std(ddof=1)/np.sqrt(n_seeds)),
            t_rw=float(t_rw), p_rw=float(p_rw),
            ruin_with_pp=float(100*rn_w.mean()), ruin_with_pp_se=float(100*rn_w.std(ddof=1)/np.sqrt(n_seeds)),
            ruin_never_pp=float(100*rn_n.mean()), ruin_never_pp_se=float(100*rn_n.std(ddof=1)/np.sqrt(n_seeds)),
            d_ruin_pp=float(100*d_rn.mean()), d_ruin_pp_se=float(100*d_rn.std(ddof=1)/np.sqrt(n_seeds)),
            t_ruin=float(t_rn), p_ruin=float(p_rn),
        ))

    # Persist data FIRST so print/plot bugs don't cost another eval pass.
    out_path = REPO / "figures" / "trial_rental_realistic_v8" / "ab_no_rental.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {out_path}\n")

    # ---- Print summary tables ----
    print(f"{'strategy':<28}  {'reward (with)':>16}  {'reward (never)':>16}  "
          f"{'Δreward':>17}  {'paired t / p':>20}")
    print("-" * 110)
    for r in sorted(rows, key=lambda v: -v["d_rw"]):
        print(f"{r['label']:<28}  "
              f"{r['rw_with']:.4f}±{r['rw_with_se']:.4f}  "
              f"{r['rw_never']:.4f}±{r['rw_never_se']:.4f}  "
              f"{r['d_rw']:+.4f}±{r['d_rw_se']:.4f}  "
              f"t={r['t_rw']:+5.1f}, p={r['p_rw']:.4f}")

    print()
    print(f"{'strategy':<28}  {'ruin% (with)':>16}  {'ruin% (never)':>16}  "
          f"{'Δruin (pp)':>17}  {'paired t / p':>20}")
    print("-" * 110)
    for r in sorted(rows, key=lambda v: v["d_ruin_pp"]):
        print(f"{r['label']:<28}  "
              f"{r['ruin_with_pp']:.3f}±{r['ruin_with_pp_se']:.3f}pp  "
              f"{r['ruin_never_pp']:.3f}±{r['ruin_never_pp_se']:.3f}pp  "
              f"{r['d_ruin_pp']:+.3f}±{r['d_ruin_pp_se']:.3f}pp  "
              f"t={r['t_ruin']:+5.1f}, p={r['p_ruin']:.4f}")

    # ---- Plot Δreward and Δruin ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    rs = sorted(rows, key=lambda v: -v["d_rw"])
    labels = [r["label"] for r in rs]
    drws = [r["d_rw"] for r in rs]
    drwes = [r["d_rw_se"] for r in rs]
    drns = [r["d_ruin_pp"] for r in rs]
    drnes = [r["d_ruin_pp_se"] for r in rs]
    y = np.arange(len(labels))[::-1]

    ax1.barh(y, drws, xerr=drwes, capsize=3,
             color=["#2ca02c" if d > 0 else "#d62728" for d in drws],
             edgecolor="black", linewidth=0.4)
    ax1.set_yticks(y); ax1.set_yticklabels(labels, fontsize=8)
    ax1.axvline(0, color="0.4", lw=0.6)
    ax1.set_xlabel("Δ reward (with rental − never buy)")
    ax1.set_title("Reward gain from rental purchase\n(positive = rental adds value)")
    ax1.grid(True, axis="x", alpha=0.3)

    ax2.barh(y, drns, xerr=drnes, capsize=3,
             color=["#d62728" if d > 0 else "#2ca02c" for d in drns],
             edgecolor="black", linewidth=0.4)
    ax2.set_yticks(y); ax2.set_yticklabels([""] * len(labels))
    ax2.axvline(0, color="0.4", lw=0.6)
    ax2.set_xlabel("Δ ruin (pp; with rental − never buy)")
    ax2.set_title("Ruin change from rental purchase\n(positive = rental increases ruin)")
    ax2.grid(True, axis="x", alpha=0.3)

    fig.suptitle(
        "Head-to-head: same allocation, with optimized rental vs never buy.\n"
        "Paired comparison on the same 30 MC seeds × 4000 paths each.",
        fontsize=11)
    fig.tight_layout()
    out_png = REPO / "figures" / "trial_rental_realistic_v8" / "ab_no_rental.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_png}")


if __name__ == "__main__":
    main()
