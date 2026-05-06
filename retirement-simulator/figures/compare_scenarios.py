"""Print a side-by-side comparison of P(W>=T) and P(ruin) across the three
visualization scenarios. Reads existing snapshots from disk.
"""
import os
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
STRATS = ["static_baseline", "bond_tent_robust", "bond_tent_stretched", "bodie_merton"]
MODES = ["gbm", "historical", "historical_ath", "historical_stretched_ath"]

CASES = [
    ("trial",            2_500_000.0, "trial / FIRE=$2.5M"),
    ("trial",            3_000_000.0, "trial / FIRE=$3.0M"),
    ("trial_zero_floor", 2_500_000.0, "zero_floor / FIRE=$2.5M"),
]

print(f"{'strategy':<22} {'mode':<28} "
      + " | ".join(f"{lbl:>22}" for _, _, lbl in CASES))
print("-" * (22 + 30 + 25 * len(CASES)))

for strat in STRATS:
    for mode in MODES:
        cells = []
        for snap_id, fire, _ in CASES:
            d = np.load(REPO / "snapshots" / snap_id / f"{strat}__{mode}.npz")
            W = d["wealth"]
            p55 = (W[:, 18] >= fire).mean()
            ruin = float(d["failure_rate"])
            cells.append(f"P55={100*p55:5.1f}% ruin={100*ruin:5.2f}%")
        print(f"{strat:<22} {mode:<28} " + " | ".join(f"{c:>22}" for c in cells))
    print()
