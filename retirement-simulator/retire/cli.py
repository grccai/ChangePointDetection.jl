"""Command-line interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import typer

from .accounts import Asset
from .config import load_scenario
from .simulate import simulate
from .optimize import optimize, OptimizerConfig
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
) -> None:
    """Optimize allocation and contribution split for the scenario."""
    scn = load_scenario(config)
    cfg = OptimizerConfig(gamma=gamma, n_paths_inner=paths, maxiter=maxiter,
                         popsize=popsize, workers=workers)
    print("Running differential evolution... (this can take a few minutes)")
    allocations, conv_bracket, trad_split, diag = optimize(scn, cfg)
    print("\n=== Optimal allocations ===")
    print(f"Taxable     stock={allocations.taxable.stock:.2%}  "
          f"bond={allocations.taxable.bond:.2%}  cash={allocations.taxable.cash:.2%}")
    print(f"Traditional stock={allocations.traditional.stock:.2%}  "
          f"bond={allocations.traditional.bond:.2%}  cash={allocations.traditional.cash:.2%}")
    print(f"Roth        stock={allocations.roth.stock:.2%}  "
          f"bond={allocations.roth.bond:.2%}  cash={allocations.roth.cash:.2%}")
    print(f"Roth conversion bracket target: {conv_bracket}")
    print(f"401k contribution split (trad fraction): {trad_split:.2%}")
    print(f"\nOptimizer diagnostics: nfev={diag['nfev']}, nit={diag['nit']}, "
          f"obj={diag['obj_value']:.3f}")

    # Re-run with final_paths for accurate reporting
    scn.target_allocations = allocations
    scn.withdrawal.roth_conversion_target_bracket = conv_bracket
    scn.simulation.n_paths = final_paths
    result = simulate(scn)
    _print_summary(scn, result, "Final evaluation at optimum")


@app.command()
def tax(
    ordinary: float = typer.Argument(..., help="Wages + traditional withdrawals."),
    ltcg: float = typer.Option(0.0, help="LT capital gains + qualified dividends."),
    ss: float = typer.Option(0.0, help="Annual Social Security benefit."),
    filing: str = typer.Option("single", help="single or mfj"),
    state_rate: float = typer.Option(0.0, help="State marginal rate."),
) -> None:
    """Compute a 2024 federal+state tax bill for given income."""
    bill = compute_tax(
        ordinary_income=ordinary, ltcg_income=ltcg, ss_benefit=ss,
        tax_exempt_interest=0.0, filing_status=filing,
        state_marginal_rate=state_rate, ty=TAX_2024,
    )
    print(bill)


@app.command()
def validate(
    config: Path = typer.Argument(..., exists=True, readable=True),
) -> None:
    """Parse the YAML config and print a structured echo + sanity checks."""
    scn = load_scenario(config)
    p = scn.initial_portfolio
    print(f"Profile: age {scn.profile.age} -> retire {scn.profile.retirement_age} "
          f"-> end {scn.profile.end_of_plan_age} ({scn.profile.filing_status})")
    print(f"Income:  ${scn.income.current_gross:,.0f} growing at "
          f"{100*scn.income.growth_rate:.2f}%/yr nominal")
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
