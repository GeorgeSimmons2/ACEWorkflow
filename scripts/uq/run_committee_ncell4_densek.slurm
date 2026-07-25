#!/bin/bash
#SBATCH --job-name=committee_n4dk_Al12
#SBATCH --partition=hmem
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=32000
#SBATCH --time=06:00:00
#SBATCH --output=committee_n4dk_Al12_%j.log

# Regenerate the phonon-stable POPS committee for Al_12_4_6A_2_ with the fixes:
#   • N_cell=4 (converged phonons; 3×3×3 was too small for the 6 Å cutoff)
#   • dense Γ→K sampling (N_per_seg=[20,20,20,20,60])
# Builds a fresh undotted H_basis at 4×4×4 (256 atoms) — heavier than the 3×3×3 build,
# hence 6 h. Needs the per-segment fcc_band_path change in src (this branch).
# Outputs → models/Al_12_4_6A_2_/results/bandpath_undotted_ncell4_densek/

REPO=/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow

julia -t 20 --project=$REPO -e "include(\"$REPO/scripts/uq/bandpath_committee_undotted_Al_12_4_6A_2_ncell4_densek.jl\")"
