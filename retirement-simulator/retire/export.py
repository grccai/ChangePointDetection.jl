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


def _quantile_along_paths(arr: np.ndarray, q: float) -> np.ndarray:
    """arr shape (P, ...). Returns array of shape arr.shape[1:] with the
    per-cell qth percentile across the path axis."""
    return np.quantile(arr, q, axis=0)


def export_allocation_xlsx(out_path: Path | str,
                            balance_by_year: np.ndarray,    # (P, H+1, 3, 3)
                            contrib_by_year: np.ndarray,    # (P, H, 3, 3)
                            ages_at_year: np.ndarray,       # (H+1,)
                            quantiles: Iterable[float] = QUANTILES,
                            policy_summary: str = "") -> None:
    """Write an .xlsx file. Each sheet = one quantile."""
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

        # Per-cell percentile values.
        # Convention used in the sheet:
        #   row "year y" shows the start-of-year-y balance (= end-of-year-y-1
        #   for y > 0, or the initial state for y == 0) AND the contributions
        #   that flow IN during year y. The very last row (y = H) shows the
        #   terminal balance with no contribution (no year H).
        bal_q = _quantile_along_paths(balance_by_year, q)   # (H+1, 3, 3)
        contrib_q_inner = _quantile_along_paths(contrib_by_year, q)  # (H, 3, 3)
        contrib_padded = np.zeros((n_years, 3, 3))
        contrib_padded[:contrib_q_inner.shape[0]] = contrib_q_inner

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

    wb.save(out_path)
