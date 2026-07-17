#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --mem-per-cpu=3988
#SBATCH --time=24:00:00

# Builds Al_20_4_6A_4 on the 5/10/20/50 percent subsets (full already exists).
# Al_20_4_6A_4 has 5476 basis functions (full-data A.csv is ~8 GB), so this is
# heavy even for the smaller subsets — adjust mem-per-cpu/time if the 50%
# subset needs more headroom than the full build did.

julia --project=/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow -e 'include("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/scripts/model_building/build_subsets_Al_20_4_6A_4.jl")'
