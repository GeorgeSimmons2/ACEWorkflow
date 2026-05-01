#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=hmem
#SBATCH --mem-per-cpu=32000
#SBATCH --time=2:00:00

module purge
module load GCC/13.2.0

julia --project=. POPS_err.jl
