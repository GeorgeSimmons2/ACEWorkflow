#!/bin/bash
#SBATCH --job-name=repro_npt_con_Al12
#SBATCH --partition=hmem
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=32000
#SBATCH --time=08:00:00
#SBATCH --output=repro_npt_con_Al12_%j.log

# Reproduces the BLUE series of thermal_expansion_vs_experiment/ — NPT thermal
# expansion (0 Pa, 300-900 K, 4x4x4) of the softest member of the multi-volume
# constrained committee (member 18).
#
# Pinned copy of scripts/uq/run_npt_multivolume.slurm pointing at the pinned .jl.
# The published run took 4 h 27 min on this configuration (SLURM job 5997448); 8 h
# leaves headroom.
#
# Requires results/bandpath_undotted_multivolume/ to exist -- see the README.  The .jl
# verifies that the member it selects is byte-identical to the published one and aborts
# if it is not, so a stale or regenerated committee cannot silently change the answer.
#
# Writes to results/repro_npt_multivolume_softest/ by default so the published
# trajectories are never overwritten.  To land in the original directory instead:
#   OUTDIR=<abs path> sbatch npt_trajectories/run_npt_constrained_softest.slurm

REPO=${REPO:-/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow}
echo "REPO = $REPO"
echo "OUTDIR = ${OUTDIR:-<default: results/repro_npt_multivolume_softest>}"

julia -t 20 --project=$REPO -e "include(\"$REPO/npt_trajectories/npt_constrained_softest.jl\")"
