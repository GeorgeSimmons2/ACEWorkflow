#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=3988
#SBATCH --time=03:30:00


julia --project=/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow -e 'include("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/scripts/uq/gaussian_mm.jl")'