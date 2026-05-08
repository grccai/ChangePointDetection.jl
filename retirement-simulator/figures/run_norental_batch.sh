#!/bin/bash
# 4 configs × 4 seeds = 16 no-rental optimizations on
# trial_realistic_NORENTAL.yaml (rental_property block removed).
set -u
cd "$(dirname "$0")/.."

PROG=/tmp/opt_norental_progress.log
echo "starting at $(date)" > "$PROG"

for cfg in bm_gbm bt_gbm bm_robust bt_robust; do
  case "$cfg" in
    bm_gbm)    policy=bodie_merton; obj=fire_prob_weighted; modes=None; paths=1200; popsize=10; maxiter=18 ;;
    bt_gbm)    policy=bond_tent;    obj=fire_prob_weighted; modes=None; paths=1200; popsize=10; maxiter=18 ;;
    bm_robust) policy=bodie_merton; obj=fire_prob_robust;   modes="['gbm','historical']"; paths=1000; popsize=8; maxiter=15 ;;
    bt_robust) policy=bond_tent;    obj=fire_prob_robust;   modes="['gbm','historical']"; paths=1000; popsize=8; maxiter=15 ;;
  esac
  for seed in 12345 7 31 101; do
    echo "=== norental ${cfg} seed=${seed} ===" >> "$PROG"
    date >> "$PROG"
    PYTHONPATH=. python -c "
from retire.config import load_scenario
from retire.optimize import optimize, OptimizerConfig
scn = load_scenario('examples/trial_realistic_NORENTAL.yaml')
cfg = OptimizerConfig(
    objective='${obj}',
    fire_age=55, fire_target_real=2_500_000, ruin_max=0.02,
    policy_class='${policy}',
    n_paths_inner=${paths}, maxiter=${maxiter}, popsize=${popsize}, workers=4,
    seed=${seed}, optimize_rental=False,
    robust_return_modes=${modes},
)
allocs, conv, split, diag = optimize(scn, cfg)
policy = diag['policy']
import json
out = {'seed': ${seed}, 'config': '${cfg}', 'obj_value': diag['obj_value'],
       'nfev': diag['nfev'], 'policy_class': diag['policy_class']}
if hasattr(policy, 'stock_high'):
    out['policy_params'] = dict(stock_high=policy.stock_high, stock_low=policy.stock_low,
        tent_age=policy.tent_age, span=policy.span, taxable_cash=policy.taxable_cash,
        conv_during_fire_gap=policy.conv_during_fire_gap,
        conv_during_ss_window=policy.conv_during_ss_window,
        trad_split=policy.trad_contribution_split,
        wealth_responsiveness=policy.wealth_responsiveness)
elif hasattr(policy, 'target_total_stock_frac'):
    out['policy_params'] = dict(target_total_stock_frac=policy.target_total_stock_frac,
        taxable_cash=policy.taxable_cash,
        conv_during_fire_gap=policy.conv_during_fire_gap,
        conv_during_ss_window=policy.conv_during_ss_window,
        trad_split=policy.trad_contribution_split)
print(json.dumps(out))
" > /tmp/opt_norental_${cfg}_seed${seed}.json 2>/tmp/opt_norental_${cfg}_seed${seed}.err
    echo "  done $(date)" >> "$PROG"
  done
done
echo "ALL DONE" >> "$PROG"
