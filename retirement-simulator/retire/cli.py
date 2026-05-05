"""Command-line interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import typer

from .accounts import Asset
from .config import load_scenario
from .location import (heuristic_target_allocations, overall_allocation_of,
                       tax_efficient_dollars)
from .simulate import simulate
from .optimize import optimize, OptimizerConfig
from .state_taxes import state_tax, STATES
from .taxes import compute_tax, TAX_2024


app = typer.Typer(add_completion=False, help="Retirement portfolio simulator and optimizer.")


def _print_summary(scn, result, label: str = "") -> None:
    q = result.terminal_quantiles([0.05, 0.25, 0.50, 0.75, 0.95])
    fail = result.failure_rate()
    real_taxes = np.array([p.lifetime_real_tax for p in result.paths])
    short = np.array([p.real_shortfall_by_year.sum() for p in result.paths])
    if label:
        print(f"\n=== {label} ===")
    print(f"Paths: {result.n_paths}")
    print(f"Plan failure rate: {100*fail:.2f}%")
    print(f"Terminal real wealth (today's $):")
    for qi, v in q.items():
        print(f"   {int(qi*100):>3}th pct  ${v:>14,.0f}")
    print(f"CVaR-5% terminal real wealth: ${result.cvar_failure(0.05):,.0f}")
    print(f"Median lifetime real tax:     ${np.median(real_taxes):,.0f}")
    print(f"Median total real shortfall:  ${np.median(short):,.0f}")


@app.command()
def simulate_cmd(
    config: Path = typer.Argument(..., exists=True, readable=True,
                                  help="YAML scenario file."),
    paths_csv: Path | None = typer.Option(None, help="Optional CSV output of "
                                          "year-by-year median real wealth"),
) -> None:
    """Run a Monte Carlo simulation of the scenario as configured."""
    scn = load_scenario(config)
    result = simulate(scn)
    _print_summary(scn, result, "Simulation")
    if paths_csv:
        wealth = np.array([p.real_wealth_by_year for p in result.paths])
        med = np.median(wealth, axis=0)
        p05 = np.quantile(wealth, 0.05, axis=0)
        p95 = np.quantile(wealth, 0.95, axis=0)
        with open(paths_csv, "w") as f:
            f.write("year,p05_real_wealth,median_real_wealth,p95_real_wealth\n")
            for y, (lo, m, hi) in enumerate(zip(p05, med, p95)):
                f.write(f"{y},{lo:.2f},{m:.2f},{hi:.2f}\n")
        print(f"\nWrote year-by-year quantiles to {paths_csv}")


@app.command()
def optimize_cmd(
    config: Path = typer.Argument(..., exists=True, readable=True),
    gamma: float = typer.Option(3.0, help="CRRA risk aversion."),
    paths: int = typer.Option(1500, help="MC paths per objective evaluation."),
    maxiter: int = typer.Option(30, help="DE generations."),
    popsize: int = typer.Option(12, help="DE population size."),
    workers: int = typer.Option(1, help="Parallel workers (>=1)."),
    final_paths: int = typer.Option(5000, help="MC paths for final report."),
    location_mode: str = typer.Option(
        "free",
        help="'free' (8 vars) or 'heuristic' (4 vars, location fixed by tax-"
             "efficient placement). Ignored when --policy=glide.",
    ),
    policy: str = typer.Option(
        "static",
        help="'static' (single fixed Decision per year) | 'glide' (2-knot "
             "glide, 12 vars, no cash in Trad/Roth) | 'three_knot_glide' "
             "(3-knot glide with mid-knot at retirement, 16 vars, no cash "
             "in Trad/Roth).",
    ),
    objective: str = typer.Option(
        "utility",
        help="'utility' (CRRA + bequest + failure penalty) | 'fire_prob' "
             "(maximize P(wealth at --fire-age >= --fire-target) subject to "
             "P(ruin) <= --ruin-max) | 'fire_prob_weighted' (time-decayed "
             "11-year sum of FIRE probabilities, max value 5.5; same ruin "
             "constraint).",
    ),
    fire_age: int = typer.Option(50, help="FIRE target age (only for fire_prob)."),
    fire_target: float = typer.Option(2_500_000, help="Real-dollar FIRE target "
                                       "(only for fire_prob)."),
    ruin_max: float = typer.Option(0.01, help="Maximum P(ruin) constraint "
                                    "(only for fire_prob)."),
) -> None:
    """Optimize allocation and contribution split for the scenario."""
    scn = load_scenario(config)
    fire_objs = {"fire_prob", "fire_prob_weighted"}
    cfg = OptimizerConfig(gamma=gamma, n_paths_inner=paths, maxiter=maxiter,
                         popsize=popsize, workers=workers,
                         location_mode=location_mode, policy_class=policy,
                         objective=objective,
                         fire_age=fire_age if objective in fire_objs else None,
                         fire_target_real=fire_target if objective in fire_objs
                                          else None,
                         ruin_max=ruin_max)
    print("Running differential evolution... (this can take a few minutes)")
    allocations, conv_bracket, trad_split, diag = optimize(scn, cfg)
    print(f"\n=== Optimal decisions (policy={diag['policy_class']}) ===")
    print(f"Year-0 allocation:")
    print(f"  Taxable     stock={allocations.taxable.stock:.2%}  "
          f"bond={allocations.taxable.bond:.2%}  cash={allocations.taxable.cash:.2%}")
    print(f"  Traditional stock={allocations.traditional.stock:.2%}  "
          f"bond={allocations.traditional.bond:.2%}  cash={allocations.traditional.cash:.2%}")
    print(f"  Roth        stock={allocations.roth.stock:.2%}  "
          f"bond={allocations.roth.bond:.2%}  cash={allocations.roth.cash:.2%}")
    print(f"Year-0 Roth conversion bracket target: {conv_bracket}")
    print(f"401k contribution split (trad fraction): {trad_split:.2%}")
    if diag["policy_class"] in ("glide", "three_knot_glide"):
        gp = diag["policy"]
        print()
        print(f"Glide path knots ({len(gp.taxable.stock.knots)} knots per asset):")
        for name, ag in [("Taxable", gp.taxable), ("Traditional", gp.traditional),
                         ("Roth", gp.roth)]:
            sk = ag.stock.knots
            bk = ag.bond.knots
            stock_str = " -> ".join(f"{v:.2%}" for _, v in sk)
            bond_str  = " -> ".join(f"{v:.2%}" for _, v in bk)
            ages_str  = "/".join(f"{a:.0f}" for a, _ in sk)
            print(f"  {name:<12} ages {ages_str:<14}   stock {stock_str}")
            print(f"  {'':<12} {'':<19}   bond  {bond_str}")
        print(f"  conversion bracket FIRE-gap:  {gp.conv_during_fire_gap}")
        print(f"  conversion bracket SS-window: {gp.conv_during_ss_window}")
        print(f"  wealth_responsiveness: {gp.wealth_responsiveness:.3f} "
              f"(>0 means de-risk when ahead of target)")
    print(f"\nOptimizer diagnostics: nfev={diag['nfev']}, nit={diag['nit']}, "
          f"obj={diag['obj_value']:.3f}")

    # Re-run with final_paths for accurate reporting using the same policy.
    scn.simulation.n_paths = final_paths
    result = simulate(scn, policy=diag["policy"])
    _print_summary(scn, result, "Final evaluation at optimum")

    if objective in ("fire_prob", "fire_prob_weighted"):
        import numpy as np
        start_age = scn.profile.age
        horizon = scn.profile.horizon()
        wealth_arr = np.array([p.real_wealth_by_year for p in result.paths])
        ruin = result.failure_rate()
        feas = "FEASIBLE" if ruin <= ruin_max else "INFEASIBLE"

        if objective == "fire_prob":
            year_idx_at_fire = max(0, min(horizon, int(round(fire_age - start_age))))
            prob_hit = float((wealth_arr[:, year_idx_at_fire] >= fire_target).mean())
            print(f"\n=== FIRE-prob objective ===")
            print(f"  P(real wealth at age {fire_age} >= ${fire_target:,.0f}): {100*prob_hit:.2f}%")
            print(f"  Wealth at age {fire_age} quantiles (real $):")
            for q in [0.05, 0.25, 0.5, 0.75, 0.95]:
                print(f"    {int(q*100):>3}th pct  ${np.quantile(wealth_arr[:, year_idx_at_fire], q):>14,.0f}")
        else:
            # fire_prob_weighted
            year_indices = [min(horizon, max(0, int(round(fire_age + i - start_age))))
                             for i in range(11)]
            weights = np.array([1.0 - i / 10.0 for i in range(11)])
            p_hit = np.array([
                (wealth_arr[:, idx] >= fire_target).mean()
                for idx in year_indices
            ])
            reward = float((weights * p_hit).sum())
            print(f"\n=== Weighted FIRE-prob objective ===")
            print(f"  Reward = sum_{{i=0..10}} (1 - i/10) * P(W_{{age {fire_age}+i}} >= ${fire_target:,.0f})")
            print(f"         = {reward:.3f}   (max possible: 5.500)")
            print(f"  Per-age FIRE probabilities (real wealth >= target):")
            for i, idx, p, w in zip(range(11), year_indices, p_hit, weights):
                print(f"    age {fire_age+i}  weight {w:.1f}   "
                      f"P(W >= target) = {100*p:.2f}%   contrib {w*p:.3f}")
        print(f"\n  P(ruin) over full plan: {100*ruin:.2f}%   "
              f"(constraint: <= {100*ruin_max:.2f}%)   --> {feas}")


@app.command()
def tax(
    ordinary: float = typer.Argument(..., help="Wages + traditional withdrawals."),
    ltcg: float = typer.Option(0.0, help="LT capital gains + qualified dividends."),
    ss: float = typer.Option(0.0, help="Annual Social Security benefit."),
    filing: str = typer.Option("single", help="single or mfj"),
    state: str = typer.Option("NONE", help=f"State: one of {sorted(STATES.keys())}"),
) -> None:
    """Compute a 2024 federal+state tax bill for given income."""
    state = state.upper()
    if state not in STATES:
        raise typer.BadParameter(f"unknown state {state}; supported: {sorted(STATES.keys())}")
    st_tax = state_tax(state=state, ordinary_income=ordinary,
                       ltcg_income=ltcg, filing_status=filing)
    bill = compute_tax(
        ordinary_income=ordinary, ltcg_income=ltcg, ss_benefit=ss,
        tax_exempt_interest=0.0, filing_status=filing,
        state_marginal_rate=0.0, ty=TAX_2024,
    )
    print(bill)
    print(f"  State ({state})    ${st_tax:>10,.0f}")
    print(f"  GRAND TOTAL       ${bill.total + st_tax:>10,.0f}")


@app.command()
def location(
    config: Path = typer.Argument(..., exists=True, readable=True),
    stock: float = typer.Option(0.70, help="Overall stock fraction."),
    bond: float = typer.Option(0.25, help="Overall bond fraction."),
    cash: float | None = typer.Option(None, help="Overall cash fraction "
                                      "(default = 1 - stock - bond)."),
) -> None:
    """Compute the tax-efficient asset-location placement for a given
    overall (stock, bond, cash) target, against the scenario's current
    account totals."""
    scn = load_scenario(config)
    if cash is None:
        cash = 1.0 - stock - bond
    p = scn.initial_portfolio
    placement = tax_efficient_dollars(
        stock, bond, cash,
        p.taxable.value(), p.traditional.value(), p.roth.value(),
    )
    print(f"Overall target:  stock {100*stock:.1f}%  bond {100*bond:.1f}%  "
          f"cash {100*cash:.1f}%  (total ${p.total_value():,.0f})")
    print()
    print(f"{'account':<12} {'stock':>14} {'bond':>14} {'cash':>14} {'total':>14}")
    for acc in ("taxable", "traditional", "roth"):
        d = placement[acc]
        tot = d["stock"] + d["bond"] + d["cash"]
        print(f"{acc:<12} ${d['stock']:>12,.0f} ${d['bond']:>12,.0f} "
              f"${d['cash']:>12,.0f} ${tot:>12,.0f}")
    targets = heuristic_target_allocations(
        stock, bond, cash,
        p.taxable.value(), p.traditional.value(), p.roth.value(),
    )
    print()
    print("Per-account fractional allocations (use these in target_allocations:):")
    for name, a in [("taxable", targets.taxable),
                    ("traditional", targets.traditional),
                    ("roth", targets.roth)]:
        print(f"  {name:<12} stock={a.stock:.4f}  bond={a.bond:.4f}  cash={a.cash:.4f}")


@app.command()
def validate(
    config: Path = typer.Argument(..., exists=True, readable=True),
) -> None:
    """Parse the YAML config and print a structured echo + sanity checks."""
    scn = load_scenario(config)
    p = scn.initial_portfolio
    pr = scn.profile
    print(f"Profile: born {pr.birthdate}, sim starts {pr.start_date} "
          f"(age {pr.age:.1f}), retires {pr.retirement_date}, "
          f"plan ends {pr.end_of_plan_date} ({pr.filing_status})")
    if scn.state_taxes.income_sources:
        print("Income sources:")
        for src in scn.state_taxes.income_sources:
            print(f"  {src.state:>4}  {src.start} -> {src.end}  "
                  f"${src.gross_annual:>10,.0f}/yr  growth {100*src.growth_rate:.2f}%/yr")
    if scn.state_taxes.residency:
        print("Residency:")
        for r in scn.state_taxes.residency:
            print(f"  {r.state:>4}  {r.start} -> {r.end}")
    print(f"Savings: {100*scn.savings.rate:.0f}% of gross")
    print(f"Spending target: ${scn.spending.annual_real:,.0f}/yr real, "
          f"smile={scn.spending.smile}")
    print(f"\nCurrent portfolio:")
    print(f"  Taxable     ${p.taxable.value():>12,.0f}  "
          f"(basis ${p.taxable.cost_basis():,.0f}, "
          f"unrealized ${p.taxable.value()-p.taxable.cost_basis():,.0f})")
    print(f"  Traditional ${p.traditional.value():>12,.0f}")
    print(f"  Roth        ${p.roth.value():>12,.0f}  "
          f"(basis ${p.roth.roth_basis:,.0f})")
    print(f"  TOTAL       ${p.total_value():>12,.0f}")
    af = p.asset_fractions()
    print(f"\nCurrent overall allocation: "
          f"stocks {100*af[Asset.STOCK]:.1f}%  "
          f"bonds {100*af[Asset.BOND]:.1f}%  "
          f"cash {100*af[Asset.CASH]:.1f}%")
    # Sanity: 25x rule
    fire_25 = scn.spending.annual_real * 25
    cur = p.total_value()
    pct = 100 * cur / fire_25
    print(f"\n4% rule (25x spending) FIRE target: ${fire_25:,.0f}")
    print(f"Current portfolio is {pct:.1f}% of FIRE target.")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
