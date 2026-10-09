#!/bin/bash
#SBATCH --job-name=repro_w_eos_render
#SBATCH --partition=hmem
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=16000
#SBATCH --time=02:00:00
#SBATCH --output=repro_w_eos_render_%j.log

# Render eos_repulsive_core's figure from the re-solved W θ, beside the published one:
#   repro_eos_tight.{pdf,png}   θ from w/02 (eps 1e-10, stopped at max_iter, lowest objective)
#   repro_eos_loose.{pdf,png}   θ from w/01 (OSQP defaults, as the original ran)
# The published figure is never touched.

set -euo pipefail
REPO=${REPO:-/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow}
cd "$REPO"
W="$REPO/models/W_20_4_5A_3"
for tag in tight loose; do
  src=$([ $tag = tight ] && echo "$W/repro_positive_core_tight.csv" || echo "$W/repro_positive_core_constrained_parameters.csv")
  echo "── $tag  θ = $src  $(date)"
  THETA_CON="$src" OUT="$REPO/eos_repulsive_core/repro_eos_$tag" \
    julia --project="$REPO" -t "${SLURM_NTASKS:-8}" eos_repulsive_core/eos_with_pair_hist.jl
done
echo "finished=$(date)"
