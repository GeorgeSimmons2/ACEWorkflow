#!/bin/bash
#SBATCH --job-name=repro_npt_naive_Al12
#SBATCH --partition=hmem
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=32000
#SBATCH --time=12:00:00
#SBATCH --output=repro_npt_naive_Al12_%j.log

# Reproduces the RED series of thermal_expansion_vs_experiment/ — NPT thermal
# expansion (0 Pa, 300-900 K, 4x4x4) of the WORST UNCONSTRAINED POPS member.
#
# Pinned copy of scripts/uq/run_npt_thermal_expansion.slurm pointing at the pinned .jl.
# hmem because the forest reconstruction (dense Cf\Ap') is memory-heavy.  12 h covers
# the four NPT runs plus five undotted-Hessian builds (a0 and four a(T)).
#
# Requires results/bandpath_undotted/committee_{repaired,rejection}.csv -- see the
# README.  The member is re-derived from the POPS forest (deterministic, seed 1234) and
# the .jl checks it against the vector the published run saved, aborting if they differ.
#
# Writes to results/repro_npt_thermal_expansion_naive_worst_member/ by default so the
# published trajectories are never overwritten.  To land in the original directory:
#   OUTDIR=<abs path> sbatch npt_trajectories/run_npt_unconstrained_naive_worst.slurm

REPO=${REPO:-/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow}
echo "REPO = $REPO"
echo "OUTDIR = ${OUTDIR:-<default: results/repro_npt_thermal_expansion_naive_worst_member>}"

julia -t 20 --project=$REPO -e "include(\"$REPO/npt_trajectories/npt_unconstrained_naive_worst.jl\")"
