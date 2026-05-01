#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=3988
#SBATCH --time=01:00:00

module purge
module load GCC/11.3.0

julia --project=. POPS_err.jl
