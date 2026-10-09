#!/bin/bash
#SBATCH --job-name=repro_al12_committee
#SBATCH --partition=hmem
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=32000
#SBATCH --time=06:00:00
#SBATCH --output=repro_al12_committee_%j.log

# Stage 1: the phonon-stable POPS committee on the undotted Hessian, N_cell=4 with a
# dense Gamma->K segment.  Produces theta_mean.csv and committee_repaired.csv, which
# stages 3 and 4 both read -- so this runs first even though stage 2 is the expensive one.
# The published run took ~20 min; the headroom is for a cold undotted H_basis build.
#
# Env:  OUTDIR  RHO_INTERVAL

set -euo pipefail
REPO=${REPO:-/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow}
cd "$REPO"
echo "host=$(hostname)  started=$(date)  RHO_INTERVAL=${RHO_INTERVAL:-25}"
julia --project="$REPO" -t "${SLURM_NTASKS:-20}" \
  "$REPO/constrained_optimization/al12/01_committee_ncell4_densek.jl"
echo "finished=$(date)"
