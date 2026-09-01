#!/bin/bash
# Ablation Studies — v20 aligned
# Each experiment removes ONE component to measure its contribution.
# Uses v20 config (samples_per_class=5000, label_smoothing=0.1)
set -e

COMMON="--rounds 200 --data_dir data/compressed --batch_size 128 --eval_every_n_rounds 5 --eval_max_samples 100000 --exclude-minority --num_cpus 1"
RESULTS_DIR="results_v20_ablation"

echo "=========================================="
echo "  v20 ABLATION STUDIES — $(date)"
echo "=========================================="

for SCENARIO in A1_no_fairness A2_no_personal A3_fixed_depth A4_no_warmup A5_no_proximal; do
    for SEED in 42 123 456; do
        echo ">>> $SCENARIO seed $SEED starting at $(date)"
        python3 main.py --seed $SEED --scenario $SCENARIO \
            --results_dir $RESULTS_DIR/$SCENARIO $COMMON
        echo "  Final F1: $(python3 -c "
import json
with open('$RESULTS_DIR/$SCENARIO/$SCENARIO/seed_$SEED/round_metrics.json') as f:
    d=json.load(f)
print(f'{d[-1][\"avg_f1\"]:.4f}')
" 2>/dev/null || echo 'N/A')"
        echo ">>> $SCENARIO seed $SEED done at $(date)"
    done
    echo ""
done

echo "=========================================="
echo "  ALL v20 ABLATIONS DONE — $(date)"
echo "=========================================="

# Summary
python3 -c "
import json, numpy as np
print('\n  v20 ABLATION SUMMARY:')
print(f'  {\"Scenario\":20s} {\"Seed 42\":>8s} {\"Seed 123\":>8s} {\"Seed 456\":>8s} {\"Mean\":>8s} {\"Std\":>6s}')
print('-' * 60)
for scenario in ['A1_no_fairness', 'A2_no_personal', 'A3_fixed_depth', 'A4_no_warmup', 'A5_no_proximal']:
    vals = []
    for seed in [42, 123, 456]:
        try:
            path = f'results_v20_ablation/{scenario}/{scenario}/seed_{seed}/round_metrics.json'
            with open(path) as f:
                d = json.load(f)
            vals.append(d[-1]['avg_f1'])
        except:
            vals.append(0)
    print(f'  {scenario:20s} {vals[0]:>8.4f} {vals[1]:>8.4f} {vals[2]:>8.4f} {np.mean(vals):>8.4f} {np.std(vals):>6.4f}')
print(f'\n  v20 Full:            0.8648   0.8667   0.8709   0.8675 0.0026')
"
