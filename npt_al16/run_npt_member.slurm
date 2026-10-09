#!/bin/bash
#SBATCH --job-name=npt_al16
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=3988
#SBATCH --time=48:00:00
#SBATCH --array=1-2
#SBATCH --output=npt_al16/logs/npt_al16_%A_%a.log
# NPT on the softest member of each Al_16 pinned ensemble.
#   array 1 → unconstrained softest (member 2,  min ω −5.25 THz at a_eq)
#   array 2 → constrained   softest (member 4,  min ω +0.336 THz at a_eq)
# TEMPS selects temperatures ("300" first; then "100,200").  Each T is seeded MD_SEED+T,
# so temperatures from separate jobs are independent and combine in aggregate.jl.
# No undotted Hessian caches and no design matrix: memory is modest, so this runs on
# `compute` rather than the contended `hmem`.
set -euo pipefail
REPO=${REPO:-/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow}
cd "$REPO"
case "$SLURM_ARRAY_TASK_ID" in
  1) export THETA_FILE=$REPO/npt_al16/members/theta_unconstrained_softest_m02.csv TAG=unconstrained_softest_m02 ;;
  2) export THETA_FILE=$REPO/npt_al16/members/theta_constrained_softest_m04.csv   TAG=constrained_softest_m04 ;;
esac
export TEMPS=${TEMPS:-300}
export OUTDIR=${OUTDIR_BASE:-$REPO/npt_al16/runs}/$TAG
echo "host=$(hostname)  task=$SLURM_ARRAY_TASK_ID  $TAG  TEMPS=$TEMPS  started=$(date)"
julia --project="$REPO" -t "${SLURM_NTASKS:-20}" "$REPO/npt_al16/npt_member_Al_16.jl"
echo "finished=$(date)"
