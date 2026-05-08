"""Second-tier comparison: best-with-rental vs best-without-rental.

Loads:
  /tmp/opt_norental_{config}_seed{N}.json (16 no-rental optima)
  v7+v8 evaluations from figures/trial_rental_realistic_v8/evaluations.json
  ab_no_rental.json (the same-allocation never-buy arm)

For each (config, seed) pair, compares:
  Arm A: v7/v8 with-rental optimum (alloc tuned WITH rental option,
         evaluated on rental-enabled scenario with rental override)
  Arm B: no-rental optimum (alloc tuned with NO rental block,
         evaluated with min_age=200 so no path ever buys)

Both evaluated on the same 30 MC seeds × 4000 paths. Paired
t-test on Δreward = arm_A − arm_B.

Outputs:
  figures/trial_rental_realistic_v8/compare_norental.json
  figures/trial_rental_realistic_v8/compare_norental.png
"""
from __future__ import annotations

import json
import os
import sys
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
from retire.policy import build_bond_tent_policy, build_bodie_merton_policy
from figures.run_snapshots import _override_rental


REPO = Path(__file__).resolve().parent.parent


def _conv_idx(v):
    if v is None or v == "None": return 0.0
    v = float(v)
    if abs(v - 0.10) < 1e-3: return 1.0
    if abs(v - 0.12) < 1e-3: return 2.0
    if abs(v - 0.22) < 1e-3: return 3.0
    if abs(v - 0.24) < 1e-3: return 4.0
    if abs(v - 0.32) < 1e-3: return 5.0
    return 0.0


def build_norental_policy(d, scn):
    """Construct a policy from a no-rental optimization JSON dump."""
    pp = d["policy_params"]
    retirement_age = scn.profile.retirement_age
    if d["policy_class"] == "bond_tent":
        x = [pp["stock_high"], pp["stock_low"],
             pp["tent_age"] - retirement_age,
             pp["span"], pp["taxable_cash"],
             _conv_idx(pp["conv_during_fire_gap"]),
             _conv_idx(pp["conv_during_ss_window"]),
             pp["trad_split"], pp["wealth_responsiveness"]]
        return build_bond_tent_policy(x, retirement_age=retirement_age)
    if d["policy_class"] == "bodie_merton":
        # Back-solve gamma from Merton constant
        mu, rf, sigma = 0.06, 0.005, 0.18
        target = pp["target_total_stock_frac"]
        gamma = (mu - rf) / (target * sigma * sigma) if target > 0 else 10.0
        x = [gamma, 0.03, pp["taxable_cash"],
             _conv_idx(pp["conv_during_fire_gap"]),
             _conv_idx(pp["conv_during_ss_window"]),
             pp["trad_split"]]
        return build_bodie_merton_policy(x, scn=scn,
                                          retirement_age=retirement_age)
    raise ValueError(d["policy_class"])


def evaluate_norental(d, mc_seeds, n_paths=4000,
                      fire_age=55, fire_target=2_500_000):
    """Eval a no-rental strategy on the rental-enabled scenario, but
    with rental.trigger.min_age=200 so no path buys. This makes the
    no-rental optima directly comparable to the v7+v8 with-rental
    optima on the same MC paths."""
    rw, rn, tm = [], [], []
    for s in mc_seeds:
        scn = load_scenario("examples/trial_rental_realistic.yaml")
        # Disable purchases by pushing min_age past the plan horizon.
        rp = scn.rental_property
        from dataclasses import replace
        from retire.config import RentalPurchaseTrigger
        scn.rental_property = replace(rp, trigger=RentalPurchaseTrigger(
            min_age=200.0,
            min_liquid_real_wealth=rp.trigger.min_liquid_real_wealth,
            min_taxable_real_wealth=rp.trigger.min_taxable_real_wealth,
        ))
        scn.simulation.seed = int(s); scn.simulation.n_paths = n_paths
        policy = build_norental_policy(d, scn)
        result = simulate(scn, policy=policy)
        wealth = np.array([p.real_wealth_by_year for p in result.paths])
        weights = np.array([1.0 - i / 10.0 for i in range(11)])
        horizon = scn.profile.horizon(); start_age = scn.profile.age
        idxs = [min(horizon, max(0, int(round(fire_age + i - start_age))))
                 for i in range(11)]
        p_hit = np.array([(wealth[:, idx] >= fire_target).mean() for idx in idxs])
        rw.append(float((weights * p_hit).sum()))
        rn.append(float(result.failure_rate()))
        tm.append(float(np.median([p.terminal_real_wealth for p in result.paths])))
    return np.array(rw), np.array(rn), np.array(tm)


def main():
    n_seeds = int(os.environ.get("N_SEEDS", "30"))
    rng = np.random.default_rng(20260508)
    mc_seeds = rng.integers(1, 1_000_000, size=n_seeds)

    # ---- Load no-rental optima ----
    norental = {}
    for cfg in ["bm_gbm", "bt_gbm", "bm_robust", "bt_robust"]:
        for seed in [12345, 7, 31, 101]:
            p = Path(f"/tmp/opt_norental_{cfg}_seed{seed}.json")
            if p.exists():
                d = json.loads(p.read_text())
                d["label"] = f"norental {cfg} seed={seed}"
                norental[(cfg, seed)] = d

    # ---- Load v7+v8 evals ----
    v7v8_evals = json.loads(
        (REPO / "figures" / "trial_rental_realistic_v8" / "evaluations.json")
        .read_text())
    v7v8 = {}
    for d in v7v8_evals:
        v7v8[(d["config"], d["seed"])] = d

    # ---- Eval no-rental optima on the same MC seeds ----
    print(f"Evaluating {len(norental)} no-rental optima on {n_seeds} MC seeds...")
    norental_evals = {}
    for key, d in norental.items():
        rw, rn, tm = evaluate_norental(d, mc_seeds)
        norental_evals[key] = dict(
            label=d["label"], config=d["config"], seed=d["seed"],
            obj_value=d["obj_value"], policy_class=d["policy_class"],
            policy_params=d["policy_params"],
            mean_reward=float(rw.mean()),
            sem_reward=float(rw.std(ddof=1)/np.sqrt(n_seeds)),
            mean_ruin=float(rn.mean()),
            sem_ruin=float(rn.std(ddof=1)/np.sqrt(n_seeds)),
            rewards=rw.tolist(), ruins=rn.tolist(),
        )
        print(f"  {d['label']:<28}  reward={rw.mean():.4f}±{rw.std(ddof=1)/np.sqrt(n_seeds):.4f}   "
              f"ruin={100*rn.mean():.3f}±{100*rn.std(ddof=1)/np.sqrt(n_seeds):.3f}pp")

    # ---- Paired comparison ----
    print(f"\n{'(config, seed)':<22}  {'rental reward':>16}  {'no-rental reward':>18}  "
          f"{'Δreward':>15}  {'paired t / p':>20}")
    print("-" * 110)
    pairs = []
    for key in v7v8.keys():
        if key not in norental_evals:
            continue
        a = v7v8[key]; b = norental_evals[key]
        rw_a = np.array(a["rewards"]); rw_b = np.array(b["rewards"])
        rn_a = np.array(a["ruins"]); rn_b = np.array(b["ruins"])
        d_rw = rw_a - rw_b
        d_rn = rn_a - rn_b
        if d_rw.std(ddof=1) > 0:
            t = d_rw.mean() / (d_rw.std(ddof=1) / np.sqrt(n_seeds))
            p = 2 * (1 - stats.t.cdf(abs(t), df=n_seeds - 1))
        else:
            t, p = float("nan"), float("nan")
        if d_rn.std(ddof=1) > 0:
            t_r = d_rn.mean() / (d_rn.std(ddof=1) / np.sqrt(n_seeds))
            p_r = 2 * (1 - stats.t.cdf(abs(t_r), df=n_seeds - 1))
        else:
            t_r, p_r = float("nan"), float("nan")
        pairs.append(dict(
            config=a["config"], seed=a["seed"],
            rental_reward=float(rw_a.mean()),
            rental_reward_se=float(rw_a.std(ddof=1)/np.sqrt(n_seeds)),
            norental_reward=float(rw_b.mean()),
            norental_reward_se=float(rw_b.std(ddof=1)/np.sqrt(n_seeds)),
            d_reward=float(d_rw.mean()),
            d_reward_se=float(d_rw.std(ddof=1)/np.sqrt(n_seeds)),
            t_reward=float(t), p_reward=float(p),
            rental_ruin_pp=float(100*rn_a.mean()),
            norental_ruin_pp=float(100*rn_b.mean()),
            d_ruin_pp=float(100*d_rn.mean()),
            d_ruin_pp_se=float(100*d_rn.std(ddof=1)/np.sqrt(n_seeds)),
            t_ruin=float(t_r), p_ruin=float(p_r),
        ))
    for r in sorted(pairs, key=lambda v: -v["d_reward"]):
        print(f"{r['config']:<10} seed={r['seed']:<5}  "
              f"{r['rental_reward']:.4f}±{r['rental_reward_se']:.4f}  "
              f"{r['norental_reward']:.4f}±{r['norental_reward_se']:.4f}  "
              f"{r['d_reward']:+.4f}±{r['d_reward_se']:.4f}  "
              f"t={r['t_reward']:+5.1f}, p={r['p_reward']:.4f}")

    print(f"\n{'(config, seed)':<22}  {'rental ruin%':>16}  {'no-rental ruin%':>18}  "
          f"{'Δruin (pp)':>15}  {'paired t / p':>20}")
    print("-" * 110)
    for r in sorted(pairs, key=lambda v: v["d_ruin_pp"]):
        print(f"{r['config']:<10} seed={r['seed']:<5}  "
              f"{r['rental_ruin_pp']:.3f}±—pp        "
              f"{r['norental_ruin_pp']:.3f}±—pp        "
              f"{r['d_ruin_pp']:+.3f}±{r['d_ruin_pp_se']:.3f}pp  "
              f"t={r['t_ruin']:+5.1f}, p={r['p_ruin']:.4f}")

    # ---- Persist ----
    out_path = REPO / "figures" / "trial_rental_realistic_v8" / "compare_norental.json"
    out_path.write_text(json.dumps(dict(
        norental_evals={f"{k[0]}_seed{k[1]}": v for k, v in norental_evals.items()},
        pairs=pairs,
    ), indent=2))
    print(f"\nWrote {out_path}")

    # ---- Plot ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    rs = sorted(pairs, key=lambda v: -v["d_reward"])
    labels = [f"{r['config']} seed={r['seed']}" for r in rs]
    drws = [r["d_reward"] for r in rs]
    drwes = [r["d_reward_se"] for r in rs]
    drns = [r["d_ruin_pp"] for r in rs]
    drnes = [r["d_ruin_pp_se"] for r in rs]
    y = np.arange(len(labels))[::-1]
    ax1.barh(y, drws, xerr=drwes, capsize=3,
             color=["#2ca02c" if d > 0 else "#d62728" for d in drws],
             edgecolor="black", linewidth=0.4)
    ax1.set_yticks(y); ax1.set_yticklabels(labels, fontsize=8)
    ax1.axvline(0, color="0.4", lw=0.6)
    ax1.set_xlabel("Δ reward (rental opt — no-rental opt)")
    ax1.set_title("Reward gain from having rental option\n(positive = rental option valuable)")
    ax1.grid(True, axis="x", alpha=0.3)

    ax2.barh(y, drns, xerr=drnes, capsize=3,
             color=["#d62728" if d > 0 else "#2ca02c" for d in drns],
             edgecolor="black", linewidth=0.4)
    ax2.set_yticks(y); ax2.set_yticklabels([""] * len(labels))
    ax2.axvline(0, color="0.4", lw=0.6)
    ax2.set_xlabel("Δ ruin (pp; rental opt — no-rental opt)")
    ax2.set_title("Ruin change from rental option\n(positive = rental option raises ruin)")
    ax2.grid(True, axis="x", alpha=0.3)

    fig.suptitle(
        "Best-with-rental vs Best-without-rental, paired per (config, seed).\n"
        "30 MC seeds × 4000 paths. Higher dim/jitter than the same-alloc A/B.",
        fontsize=11)
    fig.tight_layout()
    out_png = REPO / "figures" / "trial_rental_realistic_v8" / "compare_norental.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
