#!/bin/bash
# Edge-IIoTset second-dataset COMPARISON SUITE — 5 baselines × 3 seeds.
# Mirrors run_baselines_6class.sh config EXACTLY, except:
#   --data_dir         data/compressed   -> data_edge_iiot/compressed
#   --exclude-minority (CIC remap)        -> --num_classes 15 --no-remap
# FairPFL_noDP (our method) is already in results_edge_iiot/ — this fills in
# the baselines so the second-dataset table is a fair head-to-head.
# Results go to SCRATCH results_edge_iiot/; committed CIC results untouched.
set -e

# Override with e.g. PYTHON=/path/to/venv/bin/python ./run_edge_iiot_baselines.sh
PYTHON="${PYTHON:-python3}"
DATA_DIR="data_edge_iiot/compressed"
RESULTS_DIR="results_edge_iiot"
ROUNDS=200
BATCH_SIZE=128
MAX_SAMPLES=500000
EVAL_MAX_SAMPLES=100000
EVAL_EVERY=5
NUM_CPUS=1

SCENARIOS=("FedAvg" "FedProx" "qFFL" "Ditto" "pFedMe")
SEEDS=(42 123 456)

mkdir -p logs
echo "============================================================"
echo "EDGE-IIoTset BASELINE SUITE — $(date)"
echo "Scenarios: ${SCENARIOS[*]} | Seeds: ${SEEDS[*]}"
echo "============================================================"

for scenario in "${SCENARIOS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_log="logs/edge_${scenario}_seed${seed}.log"
        done_check="${RESULTS_DIR}/${scenario}/${scenario}/seed_${seed}/round_metrics.json"
        if [ -f "$done_check" ]; then
            n=$($PYTHON -c "import json;print(len(json.load(open('$done_check'))))" 2>/dev/null || echo 0)
            if [ "$n" -ge 40 ]; then
                echo "[SKIP] ${scenario}/seed_${seed} — already complete ($n evals)"
                continue
            fi
        fi
        echo ""
        echo "[RUN] ${scenario} / seed=${seed} — started $(date '+%H:%M:%S')"
        $PYTHON main.py \
            --scenario "$scenario" \
            --seed "$seed" \
            --rounds "$ROUNDS" \
            --data_dir "$DATA_DIR" \
            --results_dir "${RESULTS_DIR}/${scenario}" \
            --batch_size "$BATCH_SIZE" \
            --max_samples "$MAX_SAMPLES" \
            --eval_max_samples "$EVAL_MAX_SAMPLES" \
            --eval_every_n_rounds "$EVAL_EVERY" \
            --num_classes 15 --no-remap \
            --num_cpus "$NUM_CPUS" \
            > "$run_log" 2>&1
        f1=$($PYTHON -c "import json;d=json.load(open('$done_check'));print(f'{d[-1][\"avg_f1\"]:.4f}')" 2>/dev/null || echo FAILED)
        echo "  done $(date '+%H:%M:%S') — final macro-F1: $f1"
    done
done

echo ""; echo "=== EDGE-IIoTset SUITE SUMMARY ==="
$PYTHON -c "
import json, numpy as np
scenarios=['FairPFL_noDP','FedAvg','FedProx','qFFL','Ditto','pFedMe']
print(f'{\"method\":14} {\"s42\":>7} {\"s123\":>7} {\"s456\":>7} {\"mean\":>7} {\"std\":>6}')
for sc in scenarios:
    vals=[]
    for s in [42,123,456]:
        try:
            d=json.load(open(f'results_edge_iiot/{sc}/{sc}/seed_{s}/round_metrics.json'))
            vals.append(d[-1]['avg_f1'])
        except Exception:
            vals.append(float('nan'))
    print(f'{sc:14} {vals[0]:>7.4f} {vals[1]:>7.4f} {vals[2]:>7.4f} {np.nanmean(vals):>7.4f} {np.nanstd(vals):>6.4f}')
"
