#!/bin/bash
#SBATCH --job-name=allpoints_multivol_Al12
#SBATCH --partition=hmem
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=32000
#SBATCH --time=24:00:00
#SBATCH --output=allpoints_multivol_Al12_%j.log

# Constrain every observation passing the leverage mask (no 30-point subset), then
# rejection-sample 10 members and compare against 10 naive hypercube samples.
#
# MEASURED COST (probe job 5998781, 20 threads):
#   design matrix         M = 146958 rows, K = 91 params
#   full multi-volume repair: mean 8.6 s/member (median 12.6, max 19.8), ~780 cut rows
#   leverage_percentile = 0.5  ->  ~73479 members  ->  ~8.8 h   <- the default
#   leverage_percentile = 0.0  ->   146958 members ->  ~17.6 h
# 24 h walltime covers either, with headroom for sampling and plots.
#
# RESUMABLE.  Stage 3 checkpoints to
#   models/Al_12_4_6A_2_/results/allpoints_multivolume/constrained_cloud_checkpoint.jls
# after every block of 2000 observations.  If the job hits the walltime, simply
# resubmit this same script — it reloads the checkpoint and continues from where it
# stopped.  Delete the .jls to force a clean restart.
#
# Threading: one OSQP model per thread (the 30-point script shared a single model,
# which would race here).  Stage 3 is the only expensive stage and scales with
# --ntasks.
#
# Outputs → models/Al_12_4_6A_2_/results/allpoints_multivolume/

# Defaults to the main checkout; override to run from a worktree:
#   REPO=$PWD sbatch scripts/uq/run_allpoints_committee.slurm
REPO=${REPO:-/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow}
echo "REPO = $REPO"

julia -t 20 --project=$REPO -e "include(\"$REPO/scripts/uq/allpoints_committee_multivolume_Al_12_4_6A_2.jl\")"
