#!/bin/bash
#SBATCH --job-name=mo_mace_elastics
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --mem-per-cpu=3988
#SBATCH --time=04:00:00
#SBATCH --output=mo_mace_elastics_%j.log

# Constrain_Mb_Elastics.pdf end to end: fetch mlearn Mo → MACE-MPA-0 readout design and
# elastic constraints → OSQP readout QP → patched-model verification.  See README.md.
#
# The CPU MACE evaluations (steps 01 and 03) dominate; the QP is 145 variables.
# Compute nodes need no internet: the Mo JSON is cached in data/Mo/ and the MPA-0
# checkpoint in ~/.cache/mace/.
#
# Env (passed through): C11 C12 C44 REL_TOL P_TOL A0 STRAIN_H LAMBDA EPS SCAN FITS OUTDIR

set -euo pipefail
REPO=${REPO:-/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow}
cd "$REPO"
export REPO
export OMP_NUM_THREADS=${SLURM_NTASKS:-40}
export MKL_NUM_THREADS=$OMP_NUM_THREADS

echo "host=$(hostname)  started=$(date)  threads=$OMP_NUM_THREADS"
bash "$REPO/constrained_optimization/mo_mace/run_pipeline.sh"
echo "finished=$(date)"
