#!/bin/bash
# v8 batch: 3 additional seeds per config (combines with v7's default seed
# to give 4 seeds × 4 configs = 16 total optima at reduced dims).
# Waits for v7's ALL DONE marker before starting.

set -u
cd "$(dirname "$0")/.."  # to retirement-simulator/

# Wait for v7 to finish
until grep -q "ALL DONE" /tmp/opt_v7_progress.log 2>/dev/null; do
  sleep 30
done
echo "v7 complete, starting v8" >> /tmp/opt_v8_progress.log

# 3 fresh seeds, picked to be diverse from default 12345
SEEDS=(7 101 31)

for seed in "${SEEDS[@]}"; do
  for config in bm_gbm bt_gbm bm_robust bt_robust; do
    case $config in
      bm_gbm)
        policy=bodie_merton; obj=fire_prob_weighted; modes=""; paths=1200; popsize=10; maxiter=18 ;;
      bt_gbm)
        policy=bond_tent;    obj=fire_prob_weighted; modes=""; paths=1200; popsize=10; maxiter=18 ;;
      bm_robust)
        policy=bodie_merton; obj=fire_prob_robust; modes="--robust-modes gbm,historical"; paths=1000; popsize=8; maxiter=15 ;;
      bt_robust)
        policy=bond_tent;    obj=fire_prob_robust; modes="--robust-modes gbm,historical"; paths=1000; popsize=8; maxiter=15 ;;
    esac
    echo "=== v8 ${config} seed=${seed} ===" >> /tmp/opt_v8_progress.log
    date >> /tmp/opt_v8_progress.log
    PYTHONPATH=. python -c "
from retire.config import load_scenario
from retire.optimize import optimize, OptimizerConfig
scn = load_scenario('examples/trial_rental_realistic.yaml')
modes = $( [ -z "$modes" ] && echo 'None' || echo "['gbm','historical']" )
cfg = OptimizerConfig(
    objective='${obj}',
    fire_age=55, fire_target_real=2_500_000, ruin_max=0.02,
    policy_class='${policy}',
    n_paths_inner=${paths}, maxiter=${maxiter}, popsize=${popsize}, workers=4,
    seed=${seed}, optimize_rental=True,
    robust_return_modes=modes,
)
allocs, conv, split, diag = optimize(scn, cfg)
rp = diag['rental']
policy = diag['policy']
import json
out = {
    'seed': ${seed},
    'config': '${config}',
    'obj_value': diag['obj_value'],
    'nfev': diag['nfev'],
    'rental': {
        'price_real': rp.price_real,
        'location_state': rp.location_state,
        'min_age': rp.trigger.min_age,
        'min_liquid_real': rp.trigger.min_liquid_real_wealth,
        'min_taxable_real': rp.trigger.min_taxable_real_wealth,
    },
    'policy_class': diag['policy_class'],
}
# Allocation params snapshot per class
if hasattr(policy, 'stock_high'):
    out['policy_params'] = dict(stock_high=policy.stock_high, stock_low=policy.stock_low,
                                  tent_age=policy.tent_age, span=policy.span,
                                  taxable_cash=policy.taxable_cash,
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
" > /tmp/opt_v8_${config}_seed${seed}.json 2>/tmp/opt_v8_${config}_seed${seed}.err
    echo "  done $(date)" >> /tmp/opt_v8_progress.log
  done
done
echo "ALL DONE" >> /tmp/opt_v8_progress.log
