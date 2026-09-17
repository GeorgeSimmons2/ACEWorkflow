#!/bin/bash
#SBATCH --job-name=repro_committee_mv
#SBATCH --partition=hmem
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=32000
#SBATCH --time=04:00:00
#SBATCH --output=repro_committee_mv_%j.log

# Stage 1, constrained arm: the multi-volume phonon-constrained committee.
# Six undotted H_basis builds (a_eq -> 1.1*a_eq in 2% steps) then cutting-plane repair
# and rejection sampling.  The published run took 21 min (SLURM job 5997446); 4 h is
# generous headroom for the H_basis builds if the .jls caches are cold.
#
# Env:
#   COMMITTEE_OUT  output dir (default results/repro_bandpath_undotted_multivolume)
#   RHO_INTERVAL   pinned OSQP rho schedule (default 25) -- see the README

REPO=${REPO:-/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow}
echo "REPO          = $REPO"
echo "COMMITTEE_OUT = ${COMMITTEE_OUT:-<default>}"
echo "RHO_INTERVAL  = ${RHO_INTERVAL:-25}"

julia -t 20 --project=$REPO -e "include(\"$REPO/npt_trajectories/committee_constrained_multivolume.jl\")"
