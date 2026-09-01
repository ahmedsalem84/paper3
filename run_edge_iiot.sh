#!/bin/bash
# Second-dataset evaluation: best v20 config (FairPFL_noDP) on Edge-IIoTset.
# Mirrors run_v20_ablation.sh COMMON flags EXACTLY, except:
#   --data_dir       data/compressed        -> data_edge_iiot/compressed
#   --exclude-minority (CIC 34->8->6 remap) -> --num_classes 15 --no-remap
# Edge-IIoTset labels are already contiguous 0..14 (no CIC remap applies).
# Results go to a SCRATCH dir; committed CIC results are never touched.
set -e

COMMON="--rounds 200 --data_dir data_edge_iiot/compressed --batch_size 128 --eval_every_n_rounds 5 --eval_max_samples 100000 --num_classes 15 --no-remap --num_cpus 1"
RESULTS_DIR="results_edge_iiot"
SCENARIO="FairPFL_noDP"

echo "=========================================="
echo "  EDGE-IIoTset — $SCENARIO — $(date)"
echo "=========================================="

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

echo "=========================================="
echo "  EDGE-IIoTset DONE — $(date)"
echo "=========================================="

python3 -c "
import json, numpy as np
vals = []
for seed in [42, 123, 456]:
    try:
        path = f'results_edge_iiot/FairPFL_noDP/FairPFL_noDP/seed_{seed}/round_metrics.json'
        with open(path) as f:
            d = json.load(f)
        vals.append(d[-1]['avg_f1'])
    except Exception:
        vals.append(0)
print(f'\n  Edge-IIoTset FairPFL_noDP: {vals[0]:.4f} {vals[1]:.4f} {vals[2]:.4f} '
      f'mean={np.mean(vals):.4f} std={np.std(vals):.4f}')
"
