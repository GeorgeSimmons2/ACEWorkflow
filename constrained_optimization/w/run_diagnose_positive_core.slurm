#!/bin/bash
#SBATCH --job-name=repro_w_diag
#SBATCH --partition=hmem
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=16000
#SBATCH --time=06:00:00
#SBATCH --output=repro_w_diag_%j.log

# Is the W positive-core QP's solution well defined at the tolerance it was run at?
# See the script header.  Longer walltime than 01 because the tight solve may need
# far more than 19,000 iterations.

set -euo pipefail
REPO=${REPO:-/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow}
cd "$REPO"
echo "host=$(hostname)  started=$(date)  EPS_TIGHT=${EPS_TIGHT:-1e-10}"
julia --project="$REPO" -t "${SLURM_NTASKS:-8}" \
  "$REPO/constrained_optimization/w/02_diagnose_positive_core.jl"
echo "finished=$(date)"
