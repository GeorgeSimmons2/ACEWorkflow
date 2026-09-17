#!/bin/bash
#SBATCH --job-name=pops_centred_W
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=8000
#SBATCH --time=04:00:00
#SBATCH --output=pops_centred_W_%j.log
#
# Naive POPS re-centred on the repulsive-core constrained parameters, vs the RLS
# centre and vs rejection sampling, for W_20_4_5A_3.
#
# MEMORY. Unlike the Al_20 naive run, this does NOT need hmem: the script's local
# pops_cloud computes leverage row-by-row rather than as diag(X*A), so the 146126^2
# (171 GB) temporary never appears. Peak is the design matrix (2.1 GB) plus C\X'
# (2.1 GB) plus the cloud, so 64 GB is comfortable.
#
# TIME. readdlm on A.csv dominates startup; the 1829x1829 eigendecomposition in
# hypercube() is seconds. Rejection sampling in arm C is the only open-ended part
# -- if the feasible fraction is tiny it will hit max_attempts and error out
# loudly, which is itself the answer to the question the script asks.

set -euo pipefail
REPO=/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow
cd "$REPO"

echo "host=$(hostname) cpus=${SLURM_NTASKS:-?} started=$(date)"
julia --project -t "${SLURM_NTASKS:-8}" \
  "$REPO/scripts/uq/pops_naive_centred_on_constrained_W_20_4_5A_3.jl" 50 0.5
echo "finished=$(date)"
