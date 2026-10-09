#!/bin/bash
#SBATCH --job-name=repro_al16_bands
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --mem-per-cpu=3988
#SBATCH --time=08:00:00
#SBATCH --output=repro_al16_bands_%j.log

# Stage 2: one geometry optimisation and one NATIVE 4x4x4 Hessian per member, then the
# serialised band curves that bands_four_panel/ reads.  40 members (20 per ensemble) of
# native Hessian is the expensive part; the QP-free stage 1 is not.
#
# Ends by comparing its own per-member (relaxed a, min omega) against the published
# bands_{un,}constrained.csv — the actual test of whether stage 1 recovered the
# published ensembles.
#
# Env:  SRC (stage 1 output)  PUBLISHED  OUTDIR  N_CELL  N_PER_SEG  UNSTABLE  QOI_THREADS

set -euo pipefail
REPO=${REPO:-/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow}
cd "$REPO"

echo "host=$(hostname)  started=$(date)"
echo "SRC=${SRC:-<default repro_pinned_ensembles>}"
julia --project="$REPO" -t "${SLURM_NTASKS:-40}" \
  "$REPO/constrained_optimization/al16/02_bands_two_ensembles.jl"
echo "finished=$(date)"
