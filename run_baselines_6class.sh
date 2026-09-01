#!/bin/bash
# =============================================================
# Baseline Experiments — 6-Class Mode (Matching v11d Config)
# All baselines run with IDENTICAL data/hyperparameters to v11d
# =============================================================

set -e

PYTHON=python3
DATA_DIR="data/compressed"
RESULTS_DIR="results_baselines_6class"
ROUNDS=200
BATCH_SIZE=128
MAX_SAMPLES=500000
EVAL_MAX_SAMPLES=100000
EVAL_EVERY=5
NUM_CPUS=1

# Scenarios to run (baselines + our method for fair comparison)
SCENARIOS=("FedAvg" "FedProx" "qFFL" "Ditto" "pFedMe")

# Seeds for statistical rigor (mean ± std)
SEEDS=(42 123 456)

mkdir -p logs

echo "============================================================"
echo "BASELINE EXPERIMENTS — 6-CLASS MODE"
echo "Config: bs=$BATCH_SIZE, rounds=$ROUNDS, data=$DATA_DIR"
echo "Scenarios: ${SCENARIOS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "============================================================"

for scenario in "${SCENARIOS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_log="logs/baseline_${scenario}_seed${seed}.log"
        result_check="${RESULTS_DIR}/${scenario}/seed_${seed}/results.json"
        
        # Skip if already completed
        if [ -f "$result_check" ]; then
            echo "[SKIP] ${scenario}/seed_${seed} — already completed"
            continue
        fi
        
        echo ""
        echo "[RUN] ${scenario} / seed=${seed}"
        echo "  Log: ${run_log}"
        echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
        
        $PYTHON main.py \
            --scenario "$scenario" \
            --seed "$seed" \
            --rounds "$ROUNDS" \
            --data_dir "$DATA_DIR" \
            --results_dir "$RESULTS_DIR" \
            --batch_size "$BATCH_SIZE" \
            --max_samples "$MAX_SAMPLES" \
            --eval_max_samples "$EVAL_MAX_SAMPLES" \
            --eval_every_n_rounds "$EVAL_EVERY" \
            --exclude-minority \
            --num_cpus "$NUM_CPUS" \
            > "$run_log" 2>&1
        
        # Extract final metrics
        final_f1=$(grep '\[R.*\].*F1=' "$run_log" | tail -1 | grep -oP 'F1=\K[0-9.]+')
        echo "  Completed: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "  Final F1: ${final_f1:-FAILED}"
    done
done

echo ""
echo "============================================================"
echo "ALL BASELINES COMPLETED"
echo "============================================================"

# Summary
echo ""
echo "=== RESULTS SUMMARY ==="
for scenario in "${SCENARIOS[@]}"; do
    echo "--- ${scenario} ---"
    for seed in "${SEEDS[@]}"; do
        result="${RESULTS_DIR}/${scenario}/seed_${seed}/results.json"
        if [ -f "$result" ]; then
            f1=$($PYTHON -c "import json; d=json.load(open('$result')); print(f'{d[\"final_f1\"]:.4f}')" 2>/dev/null || echo "?")
            echo "  seed_${seed}: F1=${f1}"
        else
            echo "  seed_${seed}: NOT RUN"
        fi
    done
done
