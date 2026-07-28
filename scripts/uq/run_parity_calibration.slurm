#!/bin/bash
#SBATCH --job-name=parity_cal_Al12
#SBATCH --partition=hmem
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=32000
#SBATCH --time=02:00:00
#SBATCH --output=parity_cal_Al12_%A_%a.log

# Test-set parity + calibration for the three committees in aeq_cheap_vs_expensive,
# one array task per method (they are completely independent):
#
#   task 0  A_cheap_reject       cheap a_eq-only QP + phonon rejection
#   task 1  B_naive              naive POPS hypercube, no constraints
#   task 2  C_expensive_repair   full multi-volume cutting-plane repair + rejection
#
# committee_predictions is a SERIAL loop over ~1082 test configs (stride 10) and
# mutates the shared model, so it does not thread — extra cores per task buy little.
# The parallelism that helps is exactly this: three methods at once. 4 cores per task
# is enough for BLAS inside the ACE evaluation.
#
# Each task writes its own plots, raw predictions (.jls + CSVs) and a *_row.csv.
# AFTER all three finish, stitch the rows into one summary:
#
#   julia --project scripts/uq/parity_calibration_three_methods_Al_12_4_6A_2.jl merge
#
# NOTE ON PRECOMPILATION: three Julia processes starting cold will fight over the
# shared ~/.julia precompile cache and each will recompile repeatedly (we lost ~30 min
# to exactly this). If the logs fill with "missing from the cache", let one task get
# through precompilation first, then resubmit the other two — or throttle with
# --array=0-2%1 for a serial-but-clean run.

REPO=${REPO:-/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow}
echo "REPO = $REPO   method index = $SLURM_ARRAY_TASK_ID"

cd "$REPO" || exit 1     # the script resolves the test set relative to the repo root
julia -t 4 --project="$REPO" \
  "$REPO/scripts/uq/parity_calibration_three_methods_Al_12_4_6A_2.jl" "$SLURM_ARRAY_TASK_ID"
