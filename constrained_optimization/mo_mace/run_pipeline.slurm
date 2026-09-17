#!/bin/bash
#SBATCH --job-name=mo_mace_elastics
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --mem-per-cpu=3988
#SBATCH --time=12:00:00
#SBATCH --output=mo_mace_elastics_%j.log

# Constrain_Mb_Elastics.pdf end to end: fetch mlearn Mo → MACE-MPA-0 readout design and
# elastic constraints → OSQP readout QP → patched-model verification → before/after table
# (test E and F errors, C11/C12/C44 → models/Mo_MACE_MPA0_readout/<DESCRIPTOR>/before_after.csv).  See README.md.
#
# Step 01 dominates: one batched reverse pass over all descriptor columns per config for the
# force normal equations (~5 min per 54-atom config on a busy login node; 4 workers).  The QP has
# at most 257 variables.  FORCES=0 restores the fast energy-only build.
# Compute nodes need no internet: the Mo JSON is cached in data/Mo/ and the MPA-0
# checkpoint in ~/.cache/mace/.
#
# Env (passed through): DESCRIPTOR(perez|readout) FORCES FORCES_WEIGHT N_WORKERS C11 C12 C44 REL_TOL
#                       P_TOL A0 STRAIN_H LAMBDA EPS SCAN FITS OUTDIR

set -euo pipefail
REPO=${REPO:-/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow}
cd "$REPO"
export REPO
export OMP_NUM_THREADS=${SLURM_NTASKS:-40}
export MKL_NUM_THREADS=$OMP_NUM_THREADS

echo "host=$(hostname)  started=$(date)  threads=$OMP_NUM_THREADS  DESCRIPTOR=${DESCRIPTOR:-perez}"
bash "$REPO/constrained_optimization/mo_mace/run_pipeline.sh"
echo "finished=$(date)"
