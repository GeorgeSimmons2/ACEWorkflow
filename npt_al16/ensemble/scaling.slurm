#!/bin/bash
#SBATCH --job-name=npt_al16_scal
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=3988
#SBATCH --time=02:00:00
# Thread-scaling probe: 2 temperatures x 500 steps on one member; the SECOND temperature's
# MD time excludes Julia's first-call compilation.  Threads = SLURM_NTASKS.
set -euo pipefail
REPO=/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow; cd "$REPO"
export THETA_FILE=$REPO/npt_al16/ensemble/members/constrained_01.csv
export TAG=scal_t$SLURM_NTASKS OUTDIR=$REPO/npt_al16/ensemble/scaling/t$SLURM_NTASKS
export TEMPS=100,200 N_EQUIL=100 N_PROD=400 SEED_MODE=same MD_SEED=1234
echo "threads=$SLURM_NTASKS host=$(hostname)"
julia --project="$REPO" -t "$SLURM_NTASKS" "$REPO/npt_al16/npt_member_Al_16.jl"
