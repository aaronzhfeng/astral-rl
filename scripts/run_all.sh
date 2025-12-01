#!/bin/bash
# run_all.sh
# Master script to run complete ASTRAL experiment suite

set -e  # Exit on error

cd "$(dirname "$0")/.."

# Activate venv (adjust path if needed)
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "======================================================"
echo "ASTRAL Complete Experiment Suite"
echo "======================================================"
echo ""
echo "This will run:"
echo "  1. Core experiments (ASTRAL vs Baseline, 3 seeds)"
echo "  2. Interpretability experiments (7 configs)"
echo "  3. Test-time adaptation tests"
echo "  4. Causal intervention experiments"
echo ""
echo "Estimated time: 2-3 hours on CPU"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

START_TIME=$(date +%s)

# 1. Core experiments
echo ""
echo "======================================================"
echo "[1/4] Core Experiments"
echo "======================================================"
./scripts/run_core_experiments.sh 500000

# 2. Interpretability experiments
echo ""
echo "======================================================"
echo "[2/4] Interpretability Experiments"
echo "======================================================"
./scripts/run_interpretability_experiments.sh 200000

# 3. TTA tests
echo ""
echo "======================================================"
echo "[3/4] Test-Time Adaptation Tests"
echo "======================================================"
./scripts/run_tta_tests.sh 20 15

# 4. Intervention experiments
echo ""
echo "======================================================"
echo "[4/4] Causal Intervention Experiments"
echo "======================================================"
./scripts/run_interventions.sh 20

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo ""
echo "======================================================"
echo "ASTRAL Experiment Suite Complete!"
echo "======================================================"
echo ""
echo "Total time: ${HOURS}h ${MINUTES}m"
echo ""
echo "Results locations:"
echo "  Training logs: results/runs/"
echo "  TTA results:   results/tta/"
echo "  Interventions: results/interventions/"
echo ""
echo "View tensorboard: tensorboard --logdir results/runs"
echo "======================================================"

