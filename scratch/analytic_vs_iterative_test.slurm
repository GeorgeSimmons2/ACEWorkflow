#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --mem-per-cpu=3988
#SBATCH --time=10:00:00

module purge
module load GCC/11.3.0

julia -t 48 --project=. small_high_entropy_ace_model/create_model.jl