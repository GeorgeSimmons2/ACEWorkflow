#!/bin/bash
#SBATCH --job-name=npt_spread_Al12
#SBATCH --partition=hmem
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=32000
#SBATCH --time=08:00:00
#SBATCH --array=1-30%4
#SBATCH --output=constrained_optimization/npt_spread/logs/npt_spread_%A_%a.log

# NPT thermal expansion for EVERY member of the published multi-volume constrained
# committee (models/Al_12_4_6A_2_/results/bandpath_undotted_multivolume/committee_rejection.csv),
# with the protocol of the published blue series unchanged: 4x4x4, 0 Pa, 300/500/700/900 K,
# 10k equilibration + 20k production steps, MD_SEED 1234.
#
# WHY.  The figure quotes α from ONE member — the softest.  Rebuilding the committee picked a
# different softest member and α moved 2.67 → 3.61 ×10⁻⁵ K⁻¹.  A single member's α is not a
# property of the constrained ensemble; its distribution over the committee is.
#
# CONTROL.  Row 18 IS the published member (theta_used.csv, max|Δ| = 0).  MD is deterministic
# given θ and MD_SEED, so member_18 must reproduce the published summary exactly.
#
# Memory: the published run peaked at 223 GB, so this stays on hmem at the proven request.
# %4 throttles to 4 concurrent members on a two-node partition other people also use.
#
# Aggregate with:  julia --project constrained_optimization/npt_spread/aggregate.jl

set -euo pipefail
REPO=${REPO:-/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow}
cd "$REPO"
K=$(printf %02d "$SLURM_ARRAY_TASK_ID")
export THETA_FILE="$REPO/constrained_optimization/npt_spread/members/theta_rej_$K.csv"
export THETA_REF=none
export OUTDIR="$REPO/models/Al_12_4_6A_2_/results/repro_npt_spread/member_$K"
export MD_SEED=${MD_SEED:-1234}
echo "host=$(hostname)  member=$K  started=$(date)"
echo "THETA_FILE=$THETA_FILE"; echo "OUTDIR=$OUTDIR"
julia -t 20 --project="$REPO" -e "include(\"$REPO/npt_trajectories/npt_constrained_softest.jl\")"
echo "finished=$(date)"
