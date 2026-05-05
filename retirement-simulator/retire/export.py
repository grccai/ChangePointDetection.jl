"""Excel export of recommended allocations and contributions across the
simulation horizon.

For each percentile q in {5, 25, 50, 75, 95}, build a sheet whose rows are
years and whose columns are per-(account, asset) cells. Each cell is the
q-th percentile of that quantity *across the Monte Carlo paths* simulated
under the given policy. So the q=5 sheet shows "in 5% of paths, you'd have
less than $X in (taxable, stock) at year N" — useful as a downside envelope.
The q=95 sheet is the corresponding upside envelope.

Two block-groups per sheet:
  * Balance: end-of-year real $ in each (account, asset) cell.
  * Contrib: fresh contribution flows (real $) deposited in each
    (account, asset) cell during the year. Wages-driven contributions
    only — internal flows like Roth conversions are NOT recorded (they
    just shuffle money between accounts; the balance columns reflect the
    end state).
"""

from __future__ import annotations

import datetime as _dt
import io
from pathlib import Path
from typing import Iterable

import numpy as np
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment

from .simulate import SimResult
from .vstate import VState


ACCOUNT_NAMES = ("Taxable", "Traditional", "Roth")
ASSET_NAMES = ("stock", "bond", "cash")

QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


def _path_at_total_quantile(balance_by_year: np.ndarray, q: float
                             ) -> np.ndarray:
    """For each year, return the path index whose total balance is at the
    q-th rank among all paths.

    Different years may select different paths (the rows of a sheet are
    therefore not a single coherent trajectory); each row is the snapshot
    of *whatever* path happened to land at the q-th total-balance rank
    that specific year. That's the "specific values of the simulation
    which resulted in that quantile" interpretation."""
    total = balance_by_year.sum(axis=(2, 3))   # (P, H+1)
    P, H1 = total.shape
    rank = max(0, min(P - 1, int(round(q * (P - 1)))))
    sort_idx = np.argsort(total, axis=0)        # (P, H+1)
    return sort_idx[rank, :]                    # (H+1,)


def export_allocation_xlsx(out_path: Path | str,
                            balance_by_year: np.ndarray,    # (P, H+1, 3, 3)
                            contrib_by_year: np.ndarray,    # (P, H, 3, 3)
                            ages_at_year: np.ndarray,       # (H+1,)
                            quantiles: Iterable[float] = QUANTILES,
                            policy_summary: str = "",
                            fire_target_real: float | None = None,
                            retirement_age: float | None = None,
                            png_path: Path | str | None = None) -> None:
    """Write an .xlsx file. Each quantile gets its own sheet, plus a
    'Charts' sheet with embedded visualisations.

    If `png_path` is provided, the chart PNG is also saved standalone."""
    wb = openpyxl.Workbook()
    # Drop the default empty sheet
    wb.remove(wb.active)

    bold = Font(bold=True)
    header_fill = PatternFill("solid", fgColor="DDDDDD")
    block_fills = {
        "Balance": PatternFill("solid", fgColor="EAF4FF"),
        "Contrib": PatternFill("solid", fgColor="EAFFEC"),
    }

    n_years = balance_by_year.shape[1]   # H+1

    for q in quantiles:
        sheet_name = f"{int(round(q*100))}th"
        ws = wb.create_sheet(title=sheet_name)
        if policy_summary:
            ws.cell(row=1, column=1, value=policy_summary)
            ws.cell(row=1, column=1).font = bold

        # Headers — two header rows: block (Balance/Contrib), col label
        block_row = 3
        col_row = 4
        # year and age columns
        ws.cell(row=col_row, column=1, value="year").font = bold
        ws.cell(row=col_row, column=2, value="age").font = bold

        col = 3
        # Balance block (9 columns: 3 accounts × 3 assets)
        balance_start_col = col
        for acc in ACCOUNT_NAMES:
            for asset in ASSET_NAMES:
                cell = ws.cell(row=col_row, column=col,
                               value=f"{acc.lower()}_{asset}_bal")
                cell.font = bold
                cell.fill = block_fills["Balance"]
                col += 1
        # Total balance
        ws.cell(row=col_row, column=col, value="total_bal").font = bold
        ws.cell(row=col_row, column=col).fill = block_fills["Balance"]
        total_bal_col = col
        col += 1

        # Contrib block (9 columns)
        contrib_start_col = col
        for acc in ACCOUNT_NAMES:
            for asset in ASSET_NAMES:
                cell = ws.cell(row=col_row, column=col,
                               value=f"{acc.lower()}_{asset}_contrib")
                cell.font = bold
                cell.fill = block_fills["Contrib"]
                col += 1
        ws.cell(row=col_row, column=col, value="total_contrib").font = bold
        ws.cell(row=col_row, column=col).fill = block_fills["Contrib"]
        total_contrib_col = col

        # Block headers
        ws.cell(row=block_row, column=balance_start_col,
                value="BALANCES (real $, end-of-year)").font = bold
        ws.cell(row=block_row, column=balance_start_col).fill = block_fills["Balance"]
        ws.cell(row=block_row, column=contrib_start_col,
                value="CONTRIBUTIONS (real $, fresh deposits)").font = bold
        ws.cell(row=block_row, column=contrib_start_col).fill = block_fills["Contrib"]

        # Cell values at this quantile.
        # Convention: rank paths by total balance each year; pick the path
        # at the q-th rank that year and copy its actual cell values.
        # Different years may pick different paths — rows are NOT a single
        # coherent trajectory.
        idx_at_q = _path_at_total_quantile(balance_by_year, q)  # (H+1,)
        bal_q = balance_by_year[idx_at_q, np.arange(n_years), :, :]
        # Contributions: pair year y's selected path with year y's contrib.
        H = contrib_by_year.shape[1]
        contrib_padded = np.zeros((n_years, 3, 3))
        if H > 0:
            contrib_padded[:H] = contrib_by_year[idx_at_q[:H], np.arange(H), :, :]

        for y in range(n_years):
            row = col_row + 1 + y
            ws.cell(row=row, column=1, value=int(y))
            ws.cell(row=row, column=2, value=float(ages_at_year[y]))
            # Balance cells
            c = balance_start_col
            for acc_idx in range(3):
                for asset_idx in range(3):
                    ws.cell(row=row, column=c,
                            value=float(bal_q[y, acc_idx, asset_idx]))
                    c += 1
            ws.cell(row=row, column=c, value=float(bal_q[y].sum()))
            # Contrib cells (year-y inflows; 0 for the terminal row)
            c = contrib_start_col
            for acc_idx in range(3):
                for asset_idx in range(3):
                    ws.cell(row=row, column=c,
                            value=float(contrib_padded[y, acc_idx, asset_idx]))
                    c += 1
            ws.cell(row=row, column=c, value=float(contrib_padded[y].sum()))

        # Number formatting on the dollar columns
        from openpyxl.utils import get_column_letter
        for c in range(3, total_contrib_col + 1):
            for r in range(col_row + 1, col_row + 1 + n_years):
                ws.cell(row=r, column=c).number_format = '#,##0'

        # Column widths
        ws.column_dimensions["A"].width = 6
        ws.column_dimensions["B"].width = 7
        for c in range(3, total_contrib_col + 1):
            ws.column_dimensions[get_column_letter(c)].width = 16

    # Chart sheet
    chart_png = _build_chart_png(
        balance_by_year=balance_by_year,
        contrib_by_year=contrib_by_year,
        ages_at_year=ages_at_year,
        fire_target_real=fire_target_real,
        retirement_age=retirement_age,
        title=policy_summary,
    )
    if png_path is not None:
        with open(png_path, "wb") as f:
            f.write(chart_png)

    chart_ws = wb.create_sheet(title="Charts")
    chart_ws.cell(row=1, column=1, value=policy_summary).font = bold
    img = openpyxl.drawing.image.Image(io.BytesIO(chart_png))
    img.anchor = "A3"
    chart_ws.add_image(img)

    wb.save(out_path)


def _build_chart_png(*,
                     balance_by_year: np.ndarray,    # (P, H+1, 3, 3)
                     contrib_by_year: np.ndarray,    # (P, H, 3, 3)
                     ages_at_year: np.ndarray,       # (H+1,)
                     fire_target_real: float | None,
                     retirement_age: float | None,
                     title: str = "") -> bytes:
    """Build a 3-panel chart and return its PNG bytes.

      panel 1: Total real wealth quantile fan over age
      panel 2: Median per-account real balance (stacked area)
      panel 3: Median per-(account, asset) allocation FRACTION (stacked
               area of total wealth, by 3 accounts × 3 assets)
    """
    # Lazy import so the rest of the package doesn't depend on matplotlib.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ages = ages_at_year
    n_years = balance_by_year.shape[1]
    # Total real wealth per (path, year)
    total = balance_by_year.sum(axis=(2, 3))   # (P, H+1)
    qs_pct = [5, 25, 50, 75, 95]
    qs = [np.quantile(total, q / 100.0, axis=0) for q in qs_pct]

    fig, axes = plt.subplots(3, 1, figsize=(11, 13), constrained_layout=True)
    if title:
        fig.suptitle(title, fontsize=11)

    # --- Panel 1: wealth quantile fan ---
    ax = axes[0]
    ax.fill_between(ages, qs[0], qs[4], alpha=0.18, color="C0",
                    label="5–95th pct")
    ax.fill_between(ages, qs[1], qs[3], alpha=0.30, color="C0",
                    label="25–75th pct")
    ax.plot(ages, qs[2], color="C0", lw=2.0, label="median")
    if fire_target_real is not None:
        ax.axhline(fire_target_real, color="firebrick", lw=1.0,
                   linestyle="--", label=f"FIRE target ${fire_target_real/1e6:.1f}M")
    if retirement_age is not None:
        ax.axvline(retirement_age, color="black", lw=0.8, linestyle=":",
                   label=f"retirement (age {retirement_age:.0f})")
    ax.set_yscale("log")
    ax.set_xlabel("age")
    ax.set_ylabel("Total real wealth ($)")
    ax.set_title("Total real wealth — quantile fan across MC paths")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))

    # For panels 2-3 use the same "median path per year" rule as the
    # spreadsheet's 50th sheet — pick the path at the median total balance
    # each year and report ITS cells. Each year may select a different path,
    # but each year's row is a real, internally-consistent snapshot.
    idx_med = _path_at_total_quantile(balance_by_year, 0.5)   # (H+1,)
    med_cells = balance_by_year[idx_med, np.arange(n_years), :, :]   # (H+1,3,3)

    # --- Panel 2: median-path per-account stacked area ---
    ax = axes[1]
    med_by_account = med_cells.sum(axis=2)   # (H+1, 3)
    ax.stackplot(ages, med_by_account.T, labels=ACCOUNT_NAMES,
                 colors=["#88B0E0", "#E89D6F", "#7FBE7F"], alpha=0.85)
    ax.set_xlabel("age")
    ax.set_ylabel("Real balance ($)")
    ax.set_title("Median-path per-account balance (stacked) — same path-per-year rule as spreadsheet")
    if retirement_age is not None:
        ax.axvline(retirement_age, color="black", lw=0.8, linestyle=":")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))

    # --- Panel 3: median-path per-(account, asset) FRACTION ---
    ax = axes[2]
    totals = med_cells.sum(axis=(1, 2))              # (H+1,)
    totals_safe = np.where(totals > 0, totals, 1.0)
    fractions = med_cells / totals_safe[:, None, None]
    flat = np.zeros((9, n_years))
    labels = []
    palette = ["#1F4F8B", "#A2C8FF", "#D6E5FF",   # taxable: blue family
               "#9B450F", "#F2B68A", "#FAE0CC",   # traditional: orange family
               "#1F6F36", "#A2D9A2", "#D6F0D6"]   # roth: green family
    idx = 0
    for acc_idx, acc_name in enumerate(ACCOUNT_NAMES):
        for asset_idx, asset_name in enumerate(ASSET_NAMES):
            flat[idx] = fractions[:, acc_idx, asset_idx]
            labels.append(f"{acc_name[:3].lower()}_{asset_name}")
            idx += 1
    ax.stackplot(ages, flat, labels=labels, colors=palette, alpha=0.95)
    ax.set_xlabel("age")
    ax.set_ylabel("Fraction of total wealth")
    ax.set_title("Median-path allocation composition (per account × asset) over time")
    if retirement_age is not None:
        ax.axvline(retirement_age, color="black", lw=0.8, linestyle=":")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8, ncol=3, framealpha=0.9)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return buf.getvalue()
