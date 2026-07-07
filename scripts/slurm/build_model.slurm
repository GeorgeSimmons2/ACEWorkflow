#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --mem-per-cpu=3988
#SBATCH --time=16:00:00

julia --project=/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow -e 'include("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/scripts/model_building/build_model.jl")'