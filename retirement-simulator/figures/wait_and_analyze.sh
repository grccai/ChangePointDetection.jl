#!/bin/bash
# Wait for v8 to finish, then run analysis + plots.
set -u

LOG=/tmp/v8_analysis.log
echo "Pipeline starting at $(date)" > "$LOG"

# Wait for v8 batch to finish
while [ ! -f /tmp/opt_v8_progress.log ] || ! grep -q "ALL DONE" /tmp/opt_v8_progress.log 2>/dev/null; do
  sleep 60
done
echo "v8 batch done at $(date), running analysis..." >> "$LOG"

cd "$(dirname "$0")/.." || exit 2

N_SEEDS=30 N_PATHS=4000 PYTHONPATH=. python figures/v7v8_analysis.py >> "$LOG" 2>&1
echo "--- analysis done, plotting ---" >> "$LOG"
N_SEEDS=30 PYTHONPATH=. python figures/plot_v8_pareto.py >> "$LOG" 2>&1
echo "ALL DONE at $(date)" >> "$LOG"
