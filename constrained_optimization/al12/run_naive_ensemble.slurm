#!/bin/bash
#SBATCH --job-name=repro_al12_naive
#SBATCH --partition=hmem
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --mem-per-cpu=16000
#SBATCH --time=08:00:00
#SBATCH --output=repro_al12_naive_%j.log

# Stage 4: the UNCONSTRAINED (naive) POPS ensemble -- samples_naive.csv and
# min_freq_naive.csv, the left-hand panel of the top row.  Each naive member gets a
# NATIVE 4x4x4 Hessian at its OWN relaxed lattice constant (they span 3.98-4.30 A), which
# is the expensive part; the constrained side is reused from stage 3, not re-drawn.
#
# Env:  SRC (stage 3)  COMMITTEE_DIR (stage 1)  OUTDIR

set -euo pipefail
REPO=${REPO:-/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow}
cd "$REPO"
echo "host=$(hostname)  started=$(date)"
julia --project="$REPO" -t "${SLURM_NTASKS:-40}" \
  "$REPO/constrained_optimization/al12/04_naive_ensemble.jl"
echo "finished=$(date)"
