#!/bin/bash
# v20 DP experiments — FairPFL_DP vs FedAvg_DP × 3 seeds
set -e

COMMON="--rounds 200 --data_dir data/compressed --batch_size 128 --eval_every_n_rounds 5 --eval_max_samples 100000 --exclude-minority --num_cpus 1"

echo "=========================================="
echo "  v20 DP EXPERIMENTS — $(date)"
echo "=========================================="

# --- FairPFL_DP (3 seeds) ---
for SEED in 42 123 456; do
    echo ">>> FairPFL_DP seed $SEED starting at $(date)"
    python3 main.py --seed $SEED --scenario FairPFL_DP \
        --results_dir results_v20_dp/FairPFL_DP $COMMON
    echo ">>> FairPFL_DP seed $SEED done at $(date)"
done

# --- FedAvg with DP (3 seeds) ---
# Uses the dedicated FedAvg_DP scenario in fl/baselines.py (baseline model,
# dp=True, dp_mode=manual, epsilon 5.0, clip 1.0) rather than FedAvg plus a
# CLI epsilon, and writes to the directory name the analysis scripts read.
for SEED in 42 123 456; do
    echo ">>> FedAvg_DP seed $SEED starting at $(date)"
    python3 main.py --seed $SEED --scenario FedAvg_DP \
        --results_dir results_v20_dp/FedAvg_DP_real $COMMON
    echo ">>> FedAvg_DP seed $SEED done at $(date)"
done

echo "=========================================="
echo "  ALL v20 DP EXPERIMENTS DONE — $(date)"
echo "=========================================="

# Quick summary
python3 -c "
import json, numpy as np
print('\n  DP RESULTS SUMMARY:')
for method in ['FairPFL_DP', 'FedAvg_DP']:
    vals = []
    for seed in ['42', '123', '456']:
        try:
            if method == 'FairPFL_DP':
                path = f'results_v20_dp/FairPFL_DP/FairPFL_DP/seed_{seed}/round_metrics.json'
            else:
                path = f'results_v20_dp/FedAvg_DP_real/FedAvg_DP/seed_{seed}/round_metrics.json'
            with open(path) as f:
                d = json.load(f)
            vals.append(d[-1]['avg_f1'])
        except Exception as e:
            print(f'    {method} seed {seed}: ERROR ({e})')
    if vals:
        print(f'  {method:14s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}  seeds={[round(v,4) for v in vals]}')

print(f'\n  NON-DP REFERENCE:')
print(f'  FairPFL_noDP:  0.8675 +/- 0.0026')
print(f'  FedAvg_noDP:   0.8488 +/- 0.0019')
"
