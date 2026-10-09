#!/bin/bash
#SBATCH --job-name=repro_al12_hypercube
#SBATCH --partition=hmem
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=32000
#SBATCH --time=04:00:00
#SBATCH --output=repro_al12_hypercube_%j.log

# Stage 3: POPS hypercube around stage 2's constrained cloud, rejection-sampled.
# Writes committee_rejection_full_cloud.csv -- the CONSTRAINED panel of the top row --
# plus parity and calibration against the held-out test set.
# Args: <n_members> <test stride>.  Published run used 30 and 20.
#
# Env:  COMMITTEE_DIR (stage 1)  CLOUD_DIR (stage 2)  OUTDIR  FIGW  HC_N_MEMBERS  HC_STRIDE

set -euo pipefail
REPO=${REPO:-/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow}
cd "$REPO"
echo "host=$(hostname)  started=$(date)"
echo "COMMITTEE_DIR=${COMMITTEE_DIR:-<default>}  CLOUD_DIR=${CLOUD_DIR:-<default>}"
julia --project="$REPO" -t "${SLURM_NTASKS:-8}" \
  "$REPO/constrained_optimization/al12/03_hypercube_rejection.jl" "${HC_N_MEMBERS:-30}" "${HC_STRIDE:-20}"
echo "finished=$(date)"
