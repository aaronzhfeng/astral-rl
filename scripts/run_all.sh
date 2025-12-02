#!/bin/bash
# run_all.sh
# Master script to run complete ASTRAL experiment suite
#
# Usage:
#   ./scripts/run_all.sh        # Interactive (prompts for confirmation)
#   ./scripts/run_all.sh --yes  # Non-interactive (for nohup/background runs)

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
echo "Estimated time: 2-3 hours on CPU (faster with GPU)"
echo ""

# Check for --yes flag for non-interactive mode
if [[ "$1" != "--yes" && "$1" != "-y" ]]; then
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
else
    echo "Running in non-interactive mode (--yes)"
fi

START_TIME=$(date +%s)

# Track phase results
PHASE_RESULTS=()

# 1. Core experiments
echo ""
echo "======================================================"
echo "[1/4] Core Experiments"
echo "======================================================"
if ./scripts/run_core_experiments.sh 500000; then
    PHASE_RESULTS+=("Core Experiments: ✓ PASSED")
else
    PHASE_RESULTS+=("Core Experiments: ✗ HAD FAILURES (check output above)")
fi

# 2. Interpretability experiments
echo ""
echo "======================================================"
echo "[2/4] Interpretability Experiments"
echo "======================================================"
if ./scripts/run_interpretability_experiments.sh 200000; then
    PHASE_RESULTS+=("Interpretability: ✓ PASSED")
else
    PHASE_RESULTS+=("Interpretability: ✗ HAD FAILURES (check output above)")
fi

# 3. TTA tests
echo ""
echo "======================================================"
echo "[3/4] Test-Time Adaptation Tests"
echo "======================================================"
if ./scripts/run_tta_tests.sh 20 15; then
    PHASE_RESULTS+=("TTA Tests: ✓ PASSED")
else
    PHASE_RESULTS+=("TTA Tests: ✗ HAD FAILURES (check output above)")
fi

# 4. Intervention experiments
echo ""
echo "======================================================"
echo "[4/4] Causal Intervention Experiments"
echo "======================================================"
if ./scripts/run_interventions.sh 20; then
    PHASE_RESULTS+=("Interventions: ✓ PASSED")
else
    PHASE_RESULTS+=("Interventions: ✗ HAD FAILURES (check output above)")
fi

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
echo "PHASE SUMMARY:"
for result in "${PHASE_RESULTS[@]}"; do
    echo "  $result"
done
echo ""
echo "Results locations:"
echo "  Training logs: results/runs/"
echo "  TTA results:   results/tta/"
echo "  Interventions: results/interventions/"
echo ""
echo "View tensorboard: tensorboard --logdir results/runs"
echo "======================================================"

# Check if any phase had failures
FAILED=0
for result in "${PHASE_RESULTS[@]}"; do
    if [[ "$result" == *"FAILURES"* ]]; then
        FAILED=1
    fi
done

exit $FAILED
