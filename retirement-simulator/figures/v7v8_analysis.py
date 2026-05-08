"""Multi-seed v7+v8 optimization sweep analysis.

Reads:
  /tmp/opt_v7_{bm_gbm,bt_gbm,bm_robust,bt_robust}.log   (one per config, default seed)
  /tmp/opt_v8_{config}_seed{N}.json                       (one per (config,seed))

For each strategy:
  * reconstructs the policy + rental override
  * evaluates on N_SEEDS MC seeds at 4000 paths against
    examples/trial_rental_realistic.yaml
  * collects reward, ruin, terminal wealth per seed

Then:
  * Per-config: tabulates seed-stability (mean ± SE for each seed's optimum)
  * Per-config: paired-t test on best vs. worst seed
  * Across configs: Pareto frontier + scatter plot
  * Compares to prior published v4/v5/v6 optima

Output:
  figures/trial_rental_realistic_v8/pareto.png
  figures/trial_rental/repeatability_v8.md
  prints summary tables to stdout
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from retire.config import load_scenario, RentalPurchaseTrigger
from retire.simulate import simulate
from retire.policy import build_bond_tent_policy, build_bodie_merton_policy
from figures.run_snapshots import _override_rental


REPO = Path(__file__).resolve().parent.parent


@dataclass
class Strategy:
    label: str          # e.g. "v7 bm_gbm seed=12345"
    config: str         # bm_gbm | bt_gbm | bm_robust | bt_robust
    seed: int           # DE seed used for optimization
    obj_value: float    # final inner-MC objective from the optimizer
    rental_override: dict
    policy_kind: str    # "bodie_merton" | "bond_tent"
    policy_params: dict # depends on policy_kind


def _parse_v7_log(path: Path) -> Strategy | None:
    """Extract strategy from a v7 .log file (CLI output format)."""
    if not path.exists():
        return None
    txt = path.read_text()
    config = path.stem.replace("opt_v7_", "")  # bm_gbm / bt_gbm / ...

    def _findf(pat, txt=txt, default=None):
        m = re.search(pat, txt)
        return float(m.group(1)) if m else default

    def _finds(pat, txt=txt, default=None):
        m = re.search(pat, txt)
        return m.group(1).strip() if m else default

    obj = _findf(r"obj=(-?[\d.]+)") or 0.0
    price = _findf(r"price_real:\s*\$\s*([\d,]+)")
    if price is None:
        return None
    price = float(re.sub(r",", "", _finds(r"price_real:\s*\$\s*([\d,]+)") or "0"))
    state = _finds(r"location_state:\s*(\w+)") or "CA"
    min_age = _findf(r"trigger\.min_age:\s*([\d.]+)") or 40.7
    min_liquid = float(re.sub(r",", "", _finds(r"trigger\.min_liquid_real:\s*\$\s*([\d,]+)") or "0"))
    min_taxable = float(re.sub(r",", "", _finds(r"trigger\.min_taxable_real:\s*\$\s*([\d,]+)") or "0"))
    rental_override = dict(
        price_real=price, location_state=state,
        min_age=min_age, min_liquid=min_liquid, min_taxable=min_taxable,
    )
    pp = {}
    if "Bond Tent" in txt:
        kind = "bond_tent"
        pp = dict(
            stock_high=_findf(r"stock_high \(far from tent\):\s*([\d.]+)%") / 100.0,
            stock_low=_findf(r"stock_low\s*\(at tent\):\s*([\d.]+)%") / 100.0,
            tent_age=_findf(r"tent_age:\s*([\d.]+)"),
            span=_findf(r"span \(years to recover\):\s*([\d.]+)"),
            taxable_cash=_findf(r"taxable cash fraction:\s*([\d.]+)%") / 100.0,
            conv_fire=_finds(r"conv FIRE-gap:\s*(\S+)"),
            conv_ss=_finds(r"conv SS-window:\s*(\S+)"),
            trad_split=_findf(r"trad fraction\):\s*([\d.]+)%") / 100.0 or 1.0,
            wealth_resp=_findf(r"wealth_responsiveness:\s*([\d.]+)") or 0.0,
        )
    elif "Bodie-Merton" in txt:
        kind = "bodie_merton"
        merton = _findf(r"target stock of total wealth\):\s*([\d.]+)%") / 100.0
        # gamma back-solve: target = (mu - rf) / (gamma * sigma^2)
        mu, rf, sigma = 0.06, 0.005, 0.18
        gamma = (mu - rf) / (merton * sigma * sigma) if merton > 0 else 10.0
        pp = dict(
            target_total_stock_frac=merton, gamma=gamma,
            taxable_cash=_findf(r"taxable cash fraction:\s*([\d.]+)%") / 100.0,
            conv_fire=_finds(r"conv FIRE-gap:\s*(\S+)"),
            conv_ss=_finds(r"conv SS-window:\s*(\S+)"),
            trad_split=_findf(r"trad fraction\):\s*([\d.]+)%") / 100.0 or 1.0,
        )
    else:
        return None
    return Strategy(
        label=f"v7 {config} seed=12345",
        config=config, seed=12345, obj_value=obj,
        rental_override=rental_override, policy_kind=kind,
        policy_params=pp,
    )


def _parse_v8_json(path: Path) -> Strategy | None:
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    config = d["config"]
    seed = int(d["seed"])
    rp = d["rental"]
    rental_override = dict(
        price_real=rp["price_real"], location_state=rp["location_state"],
        min_age=rp["min_age"], min_liquid=rp["min_liquid_real"],
        min_taxable=rp["min_taxable_real"],
    )
    pp = d.get("policy_params", {}) or {}
    if "stock_high" in pp:
        kind = "bond_tent"
    elif "target_total_stock_frac" in pp:
        kind = "bodie_merton"
        # Add gamma back-solve for builder compatibility
        mu, rf, sigma = 0.06, 0.005, 0.18
        target = pp.get("target_total_stock_frac", 0.0)
        pp["gamma"] = (mu - rf) / (target * sigma * sigma) if target > 0 else 10.0
    else:
        return None
    return Strategy(
        label=f"v8 {config} seed={seed}",
        config=config, seed=seed, obj_value=d["obj_value"],
        rental_override=rental_override, policy_kind=kind,
        policy_params=pp,
    )


def _build_policy_from_strategy(strat: Strategy, scn):
    pp = strat.policy_params
    retirement_age = scn.profile.retirement_age
    # Conv idx mapping: None=0, 0.10=1, 0.12=2, 0.22=3, 0.24=4, 0.32=5
    def _conv_idx(v):
        if v is None or v == "None": return 0.0
        v = float(v)
        if abs(v - 0.10) < 1e-3: return 1.0
        if abs(v - 0.12) < 1e-3: return 2.0
        if abs(v - 0.22) < 1e-3: return 3.0
        if abs(v - 0.24) < 1e-3: return 4.0
        if abs(v - 0.32) < 1e-3: return 5.0
        return 0.0
    if strat.policy_kind == "bond_tent":
        x = [pp["stock_high"], pp["stock_low"],
             pp["tent_age"] - retirement_age,
             pp["span"], pp["taxable_cash"],
             _conv_idx(pp["conv_fire"]),
             _conv_idx(pp["conv_ss"]),
             pp.get("trad_split", 1.0),
             pp.get("wealth_resp", 0.0)]
        return build_bond_tent_policy(x, retirement_age=retirement_age)
    if strat.policy_kind == "bodie_merton":
        x = [pp["gamma"], 0.03, pp["taxable_cash"],
             _conv_idx(pp["conv_fire"]),
             _conv_idx(pp["conv_ss"]),
             pp.get("trad_split", 1.0)]
        return build_bodie_merton_policy(x, scn=scn,
                                          retirement_age=retirement_age)
    raise ValueError(strat.policy_kind)


def evaluate_on_seeds(strat: Strategy, mc_seeds, *,
                       fire_age=55, fire_target=2_500_000, n_paths=4000):
    scn_yaml = "examples/trial_rental_realistic.yaml"
    rw, rn, tm = [], [], []
    for s in mc_seeds:
        scn = load_scenario(scn_yaml)
        _override_rental(scn, **strat.rental_override)
        scn.simulation.seed = int(s)
        scn.simulation.n_paths = n_paths
        policy = _build_policy_from_strategy(strat, scn)
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
    n_paths = int(os.environ.get("N_PATHS", "4000"))
    rng = np.random.default_rng(20260508)
    mc_seeds = rng.integers(1, 1_000_000, size=n_seeds)

    # ---- 1) Collect strategies from v7 + v8 outputs ----
    strategies: list[Strategy] = []
    for cfg in ["bm_gbm", "bt_gbm", "bm_robust", "bt_robust"]:
        s = _parse_v7_log(Path(f"/tmp/opt_v7_{cfg}.log"))
        if s is not None:
            strategies.append(s)
    for cfg in ["bm_gbm", "bt_gbm", "bm_robust", "bt_robust"]:
        for seed in [7, 101, 31]:
            s = _parse_v8_json(Path(f"/tmp/opt_v8_{cfg}_seed{seed}.json"))
            if s is not None:
                strategies.append(s)
    print(f"Loaded {len(strategies)} strategies from v7+v8 outputs.\n")

    # ---- 2) Evaluate all on the same MC seeds ----
    print(f"Evaluating each on {n_seeds} MC seeds × {n_paths} paths...")
    results = {}
    for s in strategies:
        rw, rn, tm = evaluate_on_seeds(s, mc_seeds, n_paths=n_paths)
        results[s.label] = (s, rw, rn, tm)
        print(f"  {s.label:<28}  reward={rw.mean():.4f} ± {rw.std(ddof=1)/np.sqrt(n_seeds):.4f}   "
              f"ruin={100*rn.mean():.3f}% ± {100*rn.std(ddof=1)/np.sqrt(n_seeds):.3f}pp   "
              f"obj={s.obj_value:.3f}")

    # ---- 3) Per-config seed-stability analysis ----
    print("\n=== Per-config seed-stability ===")
    print(f"{'config':<14}  {'seeds':<6}  {'reward range':<22}  {'best seed':<14}  "
          f"{'best - worst':<22}  {'paired t / p (best vs worst)'}")
    print("-" * 130)
    by_config = {}
    for label, (s, rw, rn, tm) in results.items():
        by_config.setdefault(s.config, []).append((label, s, rw, rn, tm))
    for config, items in by_config.items():
        items.sort(key=lambda v: -v[2].mean())
        best = items[0]; worst = items[-1]
        d = best[2] - worst[2]
        if d.std(ddof=1) > 0:
            t = d.mean() / (d.std(ddof=1) / np.sqrt(n_seeds))
            p = 2 * (1 - stats.t.cdf(abs(t), df=n_seeds - 1))
            stat_str = f"t={t:+.2f}, p={p:.4f}"
        else:
            stat_str = "n/a"
        rw_means = [it[2].mean() for it in items]
        print(f"{config:<14}  {len(items):<6}  "
              f"{min(rw_means):.4f} – {max(rw_means):.4f}     "
              f"{best[0]:<14}  {d.mean():+.4f} (σ={d.std(ddof=1):.4f})   "
              f"{stat_str}")

    # ---- 4) Pareto frontier across all v7/v8 strategies ----
    print("\n=== Pareto frontier ===")
    points = [(100*rn.mean(), rw.mean(), label)
              for label, (s, rw, rn, tm) in results.items()]
    pareto = []
    for x, y, lbl in points:
        dominated = False
        for x2, y2, _ in points:
            if (x2 <= x and y2 > y) or (x2 < x and y2 >= y):
                dominated = True; break
        if not dominated:
            pareto.append((x, y, lbl))
    for x, y, lbl in sorted(pareto):
        print(f"  ruin={x:>5.2f}%   reward={y:.4f}   {lbl}")

    # ---- 5) Save evaluations to JSON for downstream plotting ----
    eval_out = REPO / "figures" / "trial_rental_realistic_v8" / "evaluations.json"
    eval_out.parent.mkdir(parents=True, exist_ok=True)
    out_data = []
    for label, (s, rw, rn, tm) in results.items():
        out_data.append(dict(
            label=label, config=s.config, seed=s.seed,
            obj_value=s.obj_value,
            policy_kind=s.policy_kind,
            policy_params=s.policy_params,
            rental_override=s.rental_override,
            rewards=rw.tolist(), ruins=rn.tolist(), terminals=tm.tolist(),
            mean_reward=float(rw.mean()),
            mean_ruin=float(rn.mean()),
            sem_reward=float(rw.std(ddof=1)/np.sqrt(n_seeds)),
            sem_ruin=float(rn.std(ddof=1)/np.sqrt(n_seeds)),
        ))
    eval_out.write_text(json.dumps(out_data, indent=2))
    print(f"\nWrote {eval_out}")


if __name__ == "__main__":
    main()
