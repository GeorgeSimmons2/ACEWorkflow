#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=3988
#SBATCH --time=47:00:00


julia --project=/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow -e 'using Unitful; sim_pressure = 500.0*u"GPa"; constrained=true; include("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/scripts/repulsive_core/md_example.jl")'
