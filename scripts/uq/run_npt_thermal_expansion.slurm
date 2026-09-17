#!/bin/bash
#SBATCH --job-name=npt_thermexp_Al12
#SBATCH --partition=hmem
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=32000
#SBATCH --time=12:00:00
#SBATCH --output=npt_thermexp_Al12_%j.log

# NPT thermal expansion of the worst naive committee member of Al_12_4_6A_2_.
# hmem: the forest reconstruction (dense Cf\Ap') is memory-heavy, like the committee run.
# 12 h covers 4 NPT runs (256 atoms × 30k steps) + 5 undotted-Hessian builds (a₀ + 4×a(T)).
# NOTE: absolute main-repo path — run after this branch is checked out in the main repo,
# where models/Al_12_4_6A_2_/{A,P,W,Y,lin_params}.csv live.

REPO=/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow

julia -t 20 --project=$REPO -e "include(\"$REPO/scripts/uq/npt_thermal_expansion_worst_member_Al_12_4_6A_2.jl\")"
