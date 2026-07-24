#!/bin/bash
#SBATCH --job-name=indHess_Al12
#SBATCH --partition=hmem
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=32000
#SBATCH --time=03:00:00
#SBATCH --output=indHess_Al12_%j.log

# Individual-Hessian phonon plots (naive hypercube vs constrained rejection) for
# Al_12_4_6A_2_.  hmem: same forest reconstruction (dense Cf\Ap') as the committee
# run.  22 native Hessians on 3×3×3 supercells + the forest + precompile fit in 3 h.
# Absolute main-repo path — run after this branch's script is in the main checkout.

REPO=/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow

julia -t 20 --project=$REPO -e "include(\"$REPO/scripts/uq/bands_individual_hessian_naive_vs_constrained_Al_12_4_6A_2.jl\")"
