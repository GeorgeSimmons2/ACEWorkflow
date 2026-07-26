#!/bin/bash
#SBATCH --job-name=committee_multivol_Al12
#SBATCH --partition=hmem
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=32000
#SBATCH --time=12:00:00
#SBATCH --output=committee_multivol_Al12_%j.log

# Phonon-stable POPS committee for Al_12_4_6A_2_ constrained at SIX VOLUMES:
#   a ∈ a_eq · {1.00, 1.02, 1.04, 1.06, 1.08, 1.10}
#
# Motivation: the single-volume (ncell4_densek) committee is stable at a_eq but its
# softest member left FCC within 7 ps of NPT at 300 K, with phonons of −7.4 THz at the
# thermally-expanded a(T).  Constraining the whole expansion range is the direct fix.
#
# Cost: 5 new undotted H_basis builds at 4×4×4/256 atoms (a_eq is cached from the
# ncell4_densek run) at ~1–2 min each, plus ~2.1 GB of new .jls cache in results/.
# The cutting-plane repair and the rejection sampling are both ~6× heavier than the
# single-volume run and the feasible set is tighter, so 12 h with headroom.
#
# Outputs → models/Al_12_4_6A_2_/results/bandpath_undotted_multivolume/

# Defaults to the main checkout; override to run from a worktree:
#   REPO=$PWD sbatch scripts/uq/run_committee_multivolume.slurm
REPO=${REPO:-/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow}
echo "REPO = $REPO"

julia -t 20 --project=$REPO -e "include(\"$REPO/scripts/uq/bandpath_committee_undotted_Al_12_4_6A_2_multivolume.jl\")"
