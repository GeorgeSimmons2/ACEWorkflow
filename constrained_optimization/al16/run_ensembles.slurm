#!/bin/bash
#SBATCH --job-name=repro_al16_ens
#SBATCH --partition=hmem
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --mem-per-cpu=8000
#SBATCH --time=06:00:00
#SBATCH --output=repro_al16_ens_%j.log

# Stage 1: the two pinned Al_16 ensembles (unconstrained / phonon-constrained), at the
# PUBLISHED N_MEMBERS=20 and ACCEPT_TOL=-0.05.  See ../README.md for why both matter.
#
# models/Al_16_4_6A_3_/A.csv is 1.9 GB of text (146,126 x 1,829 → 2.1 GB of Float64) and
# the leverage solve is another copy, so this is memory-led like the W job.  The 3-row
# undotted band path is the compute: BUILD_THREADS caps concurrent Hessian-row builds.
#
# Env:  N_MEMBERS ACCEPT_TOL OUTDIR BPCACHE BUILD_THREADS

set -euo pipefail
REPO=${REPO:-/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow}
cd "$REPO"
export BUILD_THREADS=${BUILD_THREADS:-16}

echo "host=$(hostname)  started=$(date)"
echo "N_MEMBERS=${N_MEMBERS:-20(default)}  ACCEPT_TOL=${ACCEPT_TOL:--0.05(default)}  OUTDIR=${OUTDIR:-<default>}"
julia --project="$REPO" -t "${SLURM_NTASKS:-40}" \
  "$REPO/constrained_optimization/al16/01_pinned_rejection_ensembles.jl"
echo "finished=$(date)"
