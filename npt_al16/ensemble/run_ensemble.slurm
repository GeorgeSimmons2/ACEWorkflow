#!/bin/bash
#SBATCH --job-name=npt_al16_ens
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=3988
#SBATCH --time=30:00:00
#SBATCH --output=npt_al16/ensemble/logs/%x_%A_%a.log
# One worker = one member, all six temperatures in sequence.
#   ENSEMBLE = rls | unconstrained | constrained ; member = SLURM_ARRAY_TASK_ID
# 4 threads: the scaling probe gave 3.3 min / 500 steps (≈20 h for 6 T) and 3x the
# throughput of 20 threads per core, so all 41 workers fit on ~4 nodes at once.
# Same seed for EVERY member and temperature (SEED_MODE=same, MD_SEED=1234): common
# random numbers, so differences between members reflect θ, not trajectory noise.
set -euo pipefail
REPO=/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow; cd "$REPO"
K=$(printf %02d "$SLURM_ARRAY_TASK_ID")
if [ "$ENSEMBLE" = rls ]; then export THETA_FILE=$REPO/npt_al16/ensemble/members/rls.csv
else export THETA_FILE=$REPO/npt_al16/ensemble/members/${ENSEMBLE}_$K.csv; fi
export TAG=${ENSEMBLE}_$K OUTDIR=$REPO/npt_al16/ensemble/runs/$ENSEMBLE/member_$K
export SEED_MODE=same MD_SEED=1234
echo "host=$(hostname) ensemble=$ENSEMBLE member=$K TEMPS=$TEMPS started=$(date)"
julia --project="$REPO" -t "$SLURM_NTASKS" "$REPO/npt_al16/npt_member_Al_16.jl"
echo "finished=$(date)"
