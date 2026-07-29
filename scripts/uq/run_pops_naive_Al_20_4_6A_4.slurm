#!/bin/bash
#SBATCH --job-name=pops_naive_Al20
#SBATCH --partition=hmem
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=32000
#SBATCH --time=08:00:00
#SBATCH --output=pops_naive_Al20_%j.log
#
# Draw 5 completely naive POPS hypercube samples of Al_20_4_6A_4.
#
# MEMORY. POPSRegression.corrections computes leverage as diag(X*A) with X = M×K
# and A = K×M, so it materialises an M×M temporary: 146958^2 × 8 B = 172.8 GB.
# That figure depends on the number of OBSERVATIONS, not parameters -- Al_12 has
# the same 146958 rows, so this step costs the same there. Peak including the
# design matrix and C\X' is roughly 190 GB; 20 × 32 GB = 640 GB covers it with room.
#
# TIME. Dominated by readdlm on the 15 GB A.csv (tens of minutes) and the
# 5476×5476 eigendecomposition in hypercube(). 8 h is generous.
#
# Run this BEFORE the NPT array, which reads committee_naive.csv.

set -euo pipefail
REPO=/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow
cd "$REPO"

echo "host=$(hostname) cpus=${SLURM_NTASKS:-?} started=$(date)"
julia --project -t "${SLURM_NTASKS:-20}" \
  "$REPO/scripts/uq/pops_naive_committee_Al_20_4_6A_4.jl" 0.5 5
echo "finished=$(date)"
