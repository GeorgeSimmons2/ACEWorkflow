#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --mem-per-cpu=3988
#SBATCH --time=04:00:00

julia -t auto --project=/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow -e 'include("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/scripts/phonons/x_point_curvature.jl")'