#!/bin/bash
#SBATCH --job-name=ace_phonon_ensemble
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=3988
#SBATCH --time=36:00:00
#SBATCH --output=logs/phonon_ensemble_%j.out
#SBATCH --error=logs/phonon_ensemble_%j.err

# ntasks=1 because Python multiprocessing handles all parallelism internally.
# cpus-per-task=48 allocates one CPU per pool worker.

export JULIA_PROJECT=/storage/astro2/phupfb/PhD/acestuff/new_ACE

module purge
module load GCC/11.3.0

# Activate your Python/conda environment if needed, e.g.:
# source activate myenv
# or: source /path/to/venv/bin/activate

mkdir -p logs

cd /storage/astro2/phupfb/PhD/acestuff/new_ACE/interactive_ASE_area/acease

python3 parallel_phonon_ensemble.py
