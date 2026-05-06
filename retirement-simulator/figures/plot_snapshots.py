"""Visualizations consuming snapshot .npz files.

Produces:
  figures/fan_<strategy>__<mode>.png       wealth fan chart per cell
  figures/fan_grid.png                     3 strategies x 4 modes grid
  figures/fire_ruin_grid.png               P(FIRE>=2.5M) and P(ruin) curves
  figures/allocation_<strategy>.png        median-path allocation over time

Run after run_snapshots.py:
    PYTHONPATH=. python figures/plot_snapshots.py
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SNAP_DIR = Path(__file__).resolve().parent.parent / "snapshots"
FIG_DIR = Path(__file__).resolve().parent
FIG_DIR.mkdir(exist_ok=True)

STRATEGIES = ["static_baseline", "bond_tent_robust",
              "bond_tent_stretched", "bodie_merton"]
RETURN_MODES = ["gbm", "historical", "historical_ath", "historical_stretched_ath"]
MODE_LABELS = {
    "gbm": "GBM (lognormal)",
    "historical": "Historical (1928-2023)",
    "historical_ath": "Historical | ATH start",
    "historical_stretched_ath": "Historical | ATH + 10y CAGR>8%",
}
STRAT_LABELS = {
    "static_baseline": "Static 90/0/10",
    "bond_tent_robust": "BondTent [gbm,hist_ath]",
    "bond_tent_stretched": "BondTent [gbm,stretched_ath]",
    "bodie_merton": "Bodie-Merton HC",
}

FIRE_TARGET = 2_500_000.0
BROKE_THRESHOLD = 250_000.0  # alive but essentially SS-dependent

# Gompertz survival from current age 37 (matches scenario's birthdate +
# start_date). Uses the codebase's "healthy_65" parameters (b=2.4e-5,
# c=0.085) — calibrated to qx(65)=0.005, median death ~92.
GOMPERTZ_B = 2.4e-5
GOMPERTZ_C = 0.085


def survival_curve(ages: np.ndarray, start_age: float = 37.0) -> np.ndarray:
    """S(age | start_age) under Gompertz hazard b*exp(c*age)."""
    b, c = GOMPERTZ_B, GOMPERTZ_C
    H = (b / c) * (np.exp(c * ages) - np.exp(c * start_age))
    return np.exp(-H)


def load(strategy: str, mode: str):
    p = SNAP_DIR / f"{strategy}__{mode}.npz"
    if not p.exists():
        raise FileNotFoundError(p)
    return np.load(p, allow_pickle=False)


def fan_panel(ax, ages, wealth, title=None, ymax=None):
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    Q = np.quantile(wealth, qs, axis=0) / 1e6
    ax.fill_between(ages, Q[0], Q[4], color="#4878d0", alpha=0.18,
                    label="5-95%")
    ax.fill_between(ages, Q[1], Q[3], color="#4878d0", alpha=0.35,
                    label="25-75%")
    ax.plot(ages, Q[2], color="#1f3a6a", lw=1.6, label="median")
    ax.axhline(FIRE_TARGET / 1e6, color="#d62728", lw=0.8, ls="--",
               alpha=0.6, label="FIRE $2.5M")
    ax.axhline(0, color="0.6", lw=0.5)
    if title:
        ax.set_title(title, fontsize=9)
    if ymax is not None:
        ax.set_ylim(-0.5, ymax)
    ax.grid(True, alpha=0.25)


def fan_grid():
    rows, cols = len(STRATEGIES), len(RETURN_MODES)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows),
                              sharex=True, sharey=True)
    # global y-max from 95th percentile across all panels
    ymax = 0
    cells = {}
    for r, strat in enumerate(STRATEGIES):
        for c, mode in enumerate(RETURN_MODES):
            d = load(strat, mode)
            cells[(strat, mode)] = d
            ymax = max(ymax, np.quantile(d["wealth"], 0.95) / 1e6)
    ymax = float(np.ceil(ymax / 5.0) * 5.0)
    for r, strat in enumerate(STRATEGIES):
        for c, mode in enumerate(RETURN_MODES):
            d = cells[(strat, mode)]
            ax = axes[r, c]
            ages = d["ages"]
            ruin = float(d["failure_rate"])
            p_fire = (d["wealth"][:, 18] >= FIRE_TARGET).mean()
            title = (f"{STRAT_LABELS[strat]} | {MODE_LABELS[mode]}\n"
                     f"P(W55>=2.5M)={100*p_fire:.0f}%, P(ruin)={100*ruin:.1f}%")
            fan_panel(ax, ages, d["wealth"], title=title, ymax=ymax)
            if c == 0:
                ax.set_ylabel("real wealth ($M)")
            if r == rows - 1:
                ax.set_xlabel("age")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 0.995), fontsize=9, frameon=False)
    fig.suptitle("Real wealth fan charts: 3 strategies × 4 return scenarios",
                 fontsize=12, y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIG_DIR / "fan_grid.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fire_ruin_grid():
    """For each strategy x mode: curves of P(W>=FIRE) and P(ruin) vs age."""
    rows, cols = 2, len(RETURN_MODES)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.0 * rows),
                              sharex=True)
    colors = {"static_baseline": "#888888",
              "bond_tent_robust": "#1f77b4",
              "bond_tent_stretched": "#9467bd",
              "bodie_merton": "#2ca02c"}
    for c, mode in enumerate(RETURN_MODES):
        ax_fire = axes[0, c]
        ax_ruin = axes[1, c]
        for strat in STRATEGIES:
            d = load(strat, mode)
            ages = d["ages"]
            wealth = d["wealth"]
            p_fire = (wealth >= FIRE_TARGET).mean(axis=0)
            # P(ruin) by age = P(min(W[0..t]) <= 0) — cumulative
            ruined = (wealth <= 0).cumsum(axis=1) > 0
            p_ruin = ruined.mean(axis=0)
            ax_fire.plot(ages, p_fire, color=colors[strat],
                         label=STRAT_LABELS[strat], lw=1.6)
            ax_ruin.plot(ages, p_ruin, color=colors[strat],
                         label=STRAT_LABELS[strat], lw=1.6)
        ax_fire.set_title(MODE_LABELS[mode], fontsize=10)
        ax_fire.set_ylim(0, 1.05)
        ax_fire.grid(True, alpha=0.25)
        ax_fire.axvline(50, color="0.5", lw=0.5, ls=":")
        ax_fire.axvline(55, color="0.5", lw=0.5, ls=":")
        ax_ruin.set_ylim(0, 0.10)
        ax_ruin.grid(True, alpha=0.25)
        if c == 0:
            ax_fire.set_ylabel("P(W >= $2.5M)")
            ax_ruin.set_ylabel("P(ever ruined by age)")
        ax_ruin.set_xlabel("age")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("FIRE-target probability and cumulative ruin probability vs age",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIG_DIR / "fire_ruin_grid.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def allocation_panel(ax, ages, balances_path, title=None):
    """Stack (stock, bond, cash) totals across all accounts."""
    # balances: (H+1, 3, 3) -> sum over accounts
    by_asset = balances_path.sum(axis=1)   # (H+1, 3)
    total = by_asset.sum(axis=1).clip(min=1.0)
    frac = by_asset / total[:, None]
    ax.stackplot(ages, frac[:, 0], frac[:, 1], frac[:, 2],
                 labels=["stock", "bond", "cash"],
                 colors=["#4878d0", "#ee854a", "#d5d5d5"], alpha=0.95)
    ax.set_ylim(0, 1)
    ax.set_xlim(ages[0], ages[-1])
    if title:
        ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.2)


def median_path_idx(wealth: np.ndarray) -> int:
    """Index of the path closest to the median terminal wealth."""
    term = wealth[:, -1]
    return int(np.argsort(term)[len(term) // 2])


def allocation_grid():
    rows, cols = len(STRATEGIES), len(RETURN_MODES)
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 2.8 * rows),
                              sharex=True, sharey=True)
    for r, strat in enumerate(STRATEGIES):
        for c, mode in enumerate(RETURN_MODES):
            d = load(strat, mode)
            ages = d["ages"]
            balances = d["balances"]   # (P, H+1, 3, 3)
            idx = median_path_idx(d["wealth"])
            ax = axes[r, c]
            allocation_panel(ax, ages, balances[idx],
                             title=f"{STRAT_LABELS[strat]} | "
                                   f"{MODE_LABELS[mode]}")
            if c == 0:
                ax.set_ylabel("share of wealth")
            if r == rows - 1:
                ax.set_xlabel("age")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 0.995), fontsize=9, frameon=False)
    fig.suptitle("Asset allocation along the median wealth path",
                 fontsize=12, y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIG_DIR / "allocation_grid.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def spaghetti_grid(n_show: int = 80, seed: int = 0):
    """Individual real-wealth trajectories. n_show paths per cell, sampled
    randomly with a fixed seed so cells are comparable. Median path overlaid."""
    rng = np.random.default_rng(seed)
    rows, cols = len(STRATEGIES), len(RETURN_MODES)
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 2.8 * rows),
                              sharex=True, sharey=True)
    # Hard cap at $15M so the FIRE-target line and the bulk of paths are
    # legible; long upper tail clips. Median+IQR overlay shows central
    # tendency.
    ymax = 15.0
    cells = {(s, m): load(s, m) for s in STRATEGIES for m in RETURN_MODES}
    for r, strat in enumerate(STRATEGIES):
        for c, mode in enumerate(RETURN_MODES):
            d = cells[(strat, mode)]
            ax = axes[r, c]
            ages = d["ages"]
            W = d["wealth"] / 1e6
            P = W.shape[0]
            idx = rng.choice(P, size=min(n_show, P), replace=False)
            for i in idx:
                ax.plot(ages, W[i], color="#1f77b4", lw=0.4, alpha=0.22)
            q = np.quantile(W, [0.25, 0.5, 0.75], axis=0)
            ax.plot(ages, q[0], color="#222266", lw=0.9, ls="--",
                    alpha=0.7, label="25th")
            ax.plot(ages, q[2], color="#222266", lw=0.9, ls="--",
                    alpha=0.7, label="75th")
            ax.plot(ages, q[1], color="#d62728", lw=1.8, label="median")
            ax.axhline(FIRE_TARGET / 1e6, color="0.3", lw=0.8, ls="--",
                       alpha=0.7)
            ax.set_ylim(-0.5, ymax)
            ax.set_title(f"{STRAT_LABELS[strat]} | {MODE_LABELS[mode]}",
                         fontsize=9)
            ax.grid(True, alpha=0.25)
            if c == 0:
                ax.set_ylabel("real wealth ($M)")
            if r == rows - 1:
                ax.set_xlabel("age")
    fig.suptitle(f"Individual wealth trajectories ({n_show} sampled paths "
                 f"per cell; median in red)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIG_DIR / "spaghetti_grid.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def richbrokedead_grid():
    """Stacked area of P(state, age) for state in
        {dead, broke (alive, W<=$250k), comfortable, rich (alive, W>=$2.5M)}.
    Categories are mutually exclusive and sum to 1 at each age."""
    rows, cols = len(STRATEGIES), len(RETURN_MODES)
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 2.6 * rows),
                              sharex=True, sharey=True)
    for r, strat in enumerate(STRATEGIES):
        for c, mode in enumerate(RETURN_MODES):
            d = load(strat, mode)
            ax = axes[r, c]
            ages = d["ages"]
            W = d["wealth"]
            S = survival_curve(ages)               # P(alive | age)
            p_dead = 1.0 - S
            p_alive_rich = S * (W >= FIRE_TARGET).mean(axis=0)
            p_alive_broke = S * (W <= BROKE_THRESHOLD).mean(axis=0)
            p_alive_mid = S - p_alive_rich - p_alive_broke
            p_alive_mid = np.clip(p_alive_mid, 0, 1)
            ax.stackplot(
                ages,
                p_alive_rich, p_alive_mid, p_alive_broke, p_dead,
                labels=[f"rich (W≥${FIRE_TARGET/1e6:.1f}M)",
                        "comfortable",
                        f"broke (W≤${BROKE_THRESHOLD/1e3:.0f}k)",
                        "deceased"],
                colors=["#2ca02c", "#a8d8a8", "#d62728", "#444444"],
                alpha=0.95,
            )
            ax.set_ylim(0, 1)
            ax.set_xlim(ages[0], ages[-1])
            ax.set_title(f"{STRAT_LABELS[strat]} | {MODE_LABELS[mode]}",
                         fontsize=9)
            if c == 0:
                ax.set_ylabel("probability")
            if r == rows - 1:
                ax.set_xlabel("age")
            ax.grid(True, alpha=0.2, color="white")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 0.995), fontsize=9, frameon=False)
    fig.suptitle(
        "Outcome probability over time: rich / comfortable / broke / deceased "
        "(Gompertz healthy from age 37)", fontsize=12, y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIG_DIR / "richbrokedead_grid.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def summary_table():
    """Print a comparison table of P(FIRE) and P(ruin) by (strat, mode)."""
    print("\n" + "=" * 96)
    print(f"{'strategy':<22} {'return mode':<28} "
          f"{'medW55':>9} {'P(W50>=T)':>10} {'P(W55>=T)':>10} "
          f"{'P(ruin)':>9}")
    print("-" * 96)
    for strat in STRATEGIES:
        for mode in RETURN_MODES:
            d = load(strat, mode)
            W = d["wealth"]
            p50 = (W[:, 13] >= FIRE_TARGET).mean()
            p55 = (W[:, 18] >= FIRE_TARGET).mean()
            ruin = float(d["failure_rate"])
            print(f"{STRAT_LABELS[strat]:<22} {MODE_LABELS[mode]:<28} "
                  f"{np.median(W[:, 18])/1e6:>8.2f}M "
                  f"{100*p50:>9.1f}% {100*p55:>9.1f}% "
                  f"{100*ruin:>8.2f}%")
        print()
    print("=" * 96)


if __name__ == "__main__":
    print("Generating figures...")
    fan_grid()
    fire_ruin_grid()
    allocation_grid()
    spaghetti_grid()
    richbrokedead_grid()
    summary_table()
    print("Done.")
