#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=3988
#SBATCH --time=06:00:00

module purge
module load GCC/11.3.0

julia --project=. high_entropy_POPS.jl