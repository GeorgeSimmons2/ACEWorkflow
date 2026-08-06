#!/bin/bash
#SBATCH --job-name=npt_multivol_Al12
#SBATCH --partition=hmem
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=32000
#SBATCH --time=08:00:00
#SBATCH --output=npt_multivol_Al12_%j.log

# NPT thermal expansion (0 Pa, 300–900 K, 4×4×4) on a member of the MULTI-VOLUME
# constrained committee.  Requires run_committee_multivolume.slurm to have finished
# (reads results/bandpath_undotted_multivolume/).
#
# Which member: set `npt_member` at the top of the .jl —
#   :softest  worst case (smallest min ω over the six volumes)  ← default
#   :median   typical member (the fairer "does constraining work" test)
#   an Int    that row of committee_rejection.csv
#
# Unlike the earlier NPT scripts this one reports an FCC-SURVIVAL diagnostic per
# temperature (mean coordination + median NN) and fits α ONLY over temperatures where
# the lattice was still FCC — the single-volume run silently reported "6.3% thermal
# expansion" for a cell that had transformed (coord 12 → 9.5).
#
# Outputs → models/Al_12_4_6A_2_/results/npt_multivolume_<member>/

# Defaults to the main checkout; override to run from a worktree:
#   REPO=$PWD sbatch scripts/uq/run_npt_multivolume.slurm
REPO=${REPO:-/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow}
echo "REPO = $REPO"

julia -t 20 --project=$REPO -e "include(\"$REPO/scripts/uq/npt_multivolume_member_Al_12_4_6A_2.jl\")"
