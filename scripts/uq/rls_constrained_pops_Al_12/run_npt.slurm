#!/bin/bash
#SBATCH --job-name=npt_rls_Al12
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=3988
#SBATCH --time=24:00:00
#SBATCH --output=scripts/uq/rls_constrained_pops_Al_12/logs/%x_%A_%a.log
# One task = one (model, T).  MODEL = rls | constrained;  array index 1,2,3 → 200, 400, 600 K.
# Same seed for both models and every T (common random numbers), driver npt_al16/npt_member_Al_16.jl.
set -euo pipefail
REPO=${REPO:-$SLURM_SUBMIT_DIR}; cd "$REPO"   # submit from the repository root
D=$REPO/scripts/uq/rls_constrained_pops_Al_12
export MODELDIR=$REPO/models/Al_12_4_6A_2_
# NPTDIR / CON_THETA default to the runs behind the figures; run_200K.sh all points them at repro_*
if [ "$MODEL" = rls ]; then export THETA_FILE=$MODELDIR/lin_params.csv
else export THETA_FILE=${CON_THETA:-$D/results_rls/theta_rls_constrained.csv}; fi
export TEMPS=$(( 200 * SLURM_ARRAY_TASK_ID )) TAG=$MODEL OUTDIR=${NPTDIR:-$D/npt}/$MODEL SEED_MODE=same MD_SEED=1234
echo "host=$(hostname) model=$MODEL T=$TEMPS started=$(date)"
julia --project="$REPO" -t "$SLURM_NTASKS" "$REPO/npt_al16/npt_member_Al_16.jl"
echo "finished=$(date)"
