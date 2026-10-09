#!/bin/bash
#SBATCH --job-name=repro_w_core_qp
#SBATCH --partition=hmem
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=16000
#SBATCH --time=03:00:00
#SBATCH --output=repro_w_core_qp_%j.log

# The positive-core constrained ridge regression for W_20_4_5A_3 — the producer of
# `positive_core_constrained_parameters.csv`, which had none until now.
#
# MEMORY, not cores, is why this is an hmem job.  models/W_20_4_5A_3/A.csv is 5.2 GB of
# text for a 146,126 x 1,829 matrix (2.14 GB of Float64); `Ap = Diagonal(W)*A/P` is a
# second copy, and readdlm's parser peaks above both.  128 GB is generous headroom.
# The QP itself is tiny — H is 1,829 x 1,829 — and takes minutes.
#
# Env:  OUT  REF  RHO_INTERVAL   (see the script header)

set -euo pipefail
REPO=${REPO:-/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow}
cd "$REPO"

echo "host=$(hostname)  started=$(date)"
echo "REPO=$REPO  RHO_INTERVAL=${RHO_INTERVAL:-25}  OUT=${OUT:-<default>}"
julia --project="$REPO" -t "${SLURM_NTASKS:-8}" \
  "$REPO/constrained_optimization/w/01_positive_core_qp.jl"
echo "finished=$(date)"
