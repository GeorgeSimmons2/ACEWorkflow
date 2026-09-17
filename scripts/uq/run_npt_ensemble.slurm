#!/bin/bash
#SBATCH --job-name=npt_ens_Al12
#SBATCH --partition=hmem
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=32000
#SBATCH --time=12:00:00
#SBATCH --output=npt_ens_Al12_%A_%a.log

# 6-member NPT ensemble, one array task per parameter vector, all six in parallel.
#
#   task 0   constrained mean  — a_eq pinned AND phonon-positive at all six volumes
#            (results/bandpath_undotted_multivolume/theta_mean.csv)
#   task 1-5 five members of the cheap a_eq-QP + phonon-rejection committee
#            (results/aeq_cheap_vs_expensive/committee_A_cheap_phononreject.csv),
#            chosen deterministically by subsample_seed = 20260727
#
# Identical MD seeds per temperature across all six tasks, so the ensemble spread is
# the POPS parameter uncertainty rather than MD noise.
#
# 8 cores per task x 6 tasks = 48 cores total, but as an ARRAY they schedule
# independently — you do not wait for a single 48-core allocation. The previous
# single NPT run achieved ~8.3 effective cores, so 8 per task is the right size.
# Expect ~4-6 h per task; they run concurrently if the queue allows.
#
# Each task writes results/npt_ensemble/<role>/ containing the trajectories, per-T
# analysis, theta_used.csv, and metadata.csv recording the source file, source row,
# SHA-256 of the parameter vector, its min omega at all six volumes, and every MD
# setting. Task 0 additionally writes results/npt_ensemble/MANIFEST.csv.
#
# Afterwards, combine into committee bands:
#   julia --project scripts/uq/npt_ensemble_aggregate_Al_12_4_6A_2.jl

# Defaults to the main checkout; override to run from a worktree:
#   REPO=$PWD sbatch scripts/uq/run_npt_ensemble.slurm
REPO=${REPO:-/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow}
echo "REPO = $REPO   array task = $SLURM_ARRAY_TASK_ID"

julia -t 8 --project=$REPO \
  "$REPO/scripts/uq/npt_ensemble_member_Al_12_4_6A_2.jl" "$SLURM_ARRAY_TASK_ID"
