#!/bin/bash
#SBATCH --job-name=ace_phonon_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=3988
#SBATCH --time=02:00:00
#SBATCH --output=logs/phonon_test_%j.out
#SBATCH --error=logs/phonon_test_%j.err

export JULIA_PROJECT=/storage/astro2/phupfb/PhD/acestuff/new_ACE

module purge
module load GCC/11.3.0

# Activate your Python/conda environment if needed, e.g.:
# source activate myenv
# or: source /path/to/venv/bin/activate

mkdir -p logs

cd /storage/astro2/phupfb/PhD/acestuff/new_ACE/interactive_ASE_area/acease

python3 test_parallel_phonon_ensemble.py
