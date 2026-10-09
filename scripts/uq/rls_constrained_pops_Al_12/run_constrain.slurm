#!/bin/bash
#SBATCH --job-name=rls_con_Al12
#SBATCH --partition=hmem
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=32000
#SBATCH --time=12:00:00
#SBATCH --output=scripts/uq/rls_constrained_pops_Al_12/logs/%x_%j.log
# Constrained RLS (ω > 0 at 1.00–1.10 a_eq) + RLS-vs-constrained bands → results_rls/
set -euo pipefail
REPO=${REPO:-$SLURM_SUBMIT_DIR}; cd "$REPO"   # submit from the repository root
julia --project="$REPO" -t "$SLURM_NTASKS" scripts/uq/rls_constrained_pops_Al_12/compare_rls_Al_12_4_6A_2.jl
