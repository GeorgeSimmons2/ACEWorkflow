#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=3988
#SBATCH --time=24:00:00

julia --project=/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow -e 'include("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/scripts/defects/vacancy_md.jl")'