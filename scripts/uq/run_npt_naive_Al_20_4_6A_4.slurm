#!/bin/bash
#SBATCH --job-name=npt_naive_Al20
#SBATCH --partition=hmem
#SBATCH --array=0-4
#SBATCH --ntasks=16
#SBATCH --mem-per-cpu=4000
#SBATCH --time=47:00:00
#SBATCH --output=npt_naive_Al20_%A_%a.log
#
# NPT temperature sweep (300/500/700/900 K) for each of the 5 naive POPS
# hypercube samples of Al_20_4_6A_4. One array task per member, all five parallel.
#
# WHY THESE NUMBERS.  Al_20_4_6A_4 has 5476 basis functions against 91 for
# Al_12_4_6A_2_, and measures ~11x slower per MD step:
#
#     3x3x3 (108 atoms)   843 ms/step   ->  4 temps x 30k steps = 28.1 h
#     4x4x4 (256 atoms)  1789 ms/step   ->  4 temps x 30k steps = 59.6 h
#                                           (8 threads, login node)
#
# The partition limit is 48 h, so the Al_12 protocol (4x4x4, 10k equil + 20k prod)
# does not fit. Of the two ways to make it fit, shrinking the cell to 3x3x3 or
# shortening the runs, this keeps the 4x4x4 cell and trims production:
#
#   * cell size is what the structural diagnostics depend on -- q6bar, coordination
#     and RDF statistics are far better with 256 atoms than 108, and 4x4x4 is what
#     the Al_12 ensembles used, so the two studies stay comparable
#   * 12 ps of production is still ample for averaging a(T); it mainly costs
#     precision on the box mean, which is reported as a standard deviation anyway
#
#   8k equil + 12k prod = 20k steps  ->  35.4 h at the 16-thread rate
#
# THREADS.  Scaling is poor -- 1789/1592/1361 ms/step at 8/16/32 threads, i.e. 4x
# the cores for a 31% gain (ACE force evaluation is largely memory-bound, and the
# MonteCarloBarostat step is serial). hmem has 2 nodes x 64 = 128 CPUs, so at 32
# threads the five array tasks need 160 CPUs and only four run concurrently --
# the fifth waits a full run, making the wall time ~60 h. At 16 threads all five
# fit in 80 CPUs and finish together in ~35 h. Fewer threads is strictly faster
# here in wall-clock terms.
#
# Memory per task is small (5476 parameters, 256 atoms, ~600 logged frames), so
# 16 x 4 GB = 64 GB per task is already generous and keeps the array schedulable.
#
# NOTE the transformations seen in the Al_12 study were stochastic nucleation
# events within the run, so a shorter production window makes a transformation
# somewhat LESS likely to be caught. Treat "FCC at N/4" as a lower bound on
# instability, not a stability certificate, and prefer the q6bar verdict.
#
# Requires committee_naive.csv from run_pops_naive_Al_20_4_6A_4.slurm.
#
# ARGS: member_index  N_super  n_equil  n_prod

set -euo pipefail
REPO=/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow
cd "$REPO"

COMM="$REPO/models/Al_20_4_6A_4/results/pops_naive_committee/committee_naive.csv"
if [ ! -f "$COMM" ]; then
  echo "ERROR: $COMM missing — run run_pops_naive_Al_20_4_6A_4.slurm first" >&2
  exit 1
fi

echo "host=$(hostname) task=${SLURM_ARRAY_TASK_ID} cpus=${SLURM_NTASKS:-?} started=$(date)"
julia --project -t "${SLURM_NTASKS:-16}" \
  "$REPO/scripts/uq/npt_naive_member_Al_20_4_6A_4.jl" \
  "${SLURM_ARRAY_TASK_ID}" 4 8000 12000
echo "finished=$(date)"
