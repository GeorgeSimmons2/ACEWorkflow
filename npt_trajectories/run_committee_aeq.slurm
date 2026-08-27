#!/bin/bash
#SBATCH --job-name=repro_committee_aeq
#SBATCH --partition=hmem
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=32000
#SBATCH --time=04:00:00
#SBATCH --output=repro_committee_aeq_%j.log

# Stage 1, unconstrained arm's dependency: the single-volume (a_eq) committee.
#
# The RED series member -- the worst UNCONSTRAINED POPS draw -- does not itself need a
# QP; it comes straight from the POPS forest and is bit-exactly reproducible.  But the
# NPT driver also reports the softest CONSTRAINED member alongside it, for the
# before/after phonon panel and the summary header, and that needs this committee's
# committee_{repaired,rejection}.csv to exist.
#
# Env:
#   COMMITTEE_OUT  output dir (default results/repro_bandpath_undotted)
#   RHO_INTERVAL   pinned OSQP rho schedule (default 25) -- see the README

REPO=${REPO:-/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow}
echo "REPO          = $REPO"
echo "COMMITTEE_OUT = ${COMMITTEE_OUT:-<default>}"
echo "RHO_INTERVAL  = ${RHO_INTERVAL:-25}"

julia -t 20 --project=$REPO -e "include(\"$REPO/npt_trajectories/committee_aeq.jl\")"
