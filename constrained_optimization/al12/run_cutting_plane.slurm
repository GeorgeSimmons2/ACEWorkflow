#!/bin/bash
#SBATCH --job-name=repro_al12_cutplane
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=42
#SBATCH --mem-per-cpu=3988
#SBATCH --time=04:00:00
#SBATCH --output=repro_al12_cutplane_%j.log

# Stage 2: cutting-plane constrain the whole top-50%-leverage POPS cloud -- 73,479 of
# 146,958 observations.  Calibrated at ~74 ms/member single-threaded, so ~11 min at 16
# threads; 4 h is wide margin.  Checkpoints every 5,000 
#
# --nodes=1 IS REQUIRED.  Without it SLURM spread the 42 tasks over 6 nodes and Julia ran
# 42 threads on the batch node's 2 CPUs / 7.9 GB — 440 ms/member and an OOM kill (job 6192525).members.
#
# NOT ALL MEMBERS COME OUT STABLE -- filter on min_omega_THz in member_diagnostics.csv.
# The argument is n_members; 0 means "all".
#
# Env:  OUTDIR  RHO_INTERVAL  CP_N_MEMBERS (0 = all)

set -euo pipefail
REPO=${REPO:-/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow}
cd "$REPO"
echo "host=$(hostname)  started=$(date)  RHO_INTERVAL=${RHO_INTERVAL:-25}"
julia --project="$REPO" -t "${SLURM_NTASKS:-42}" \
  "$REPO/constrained_optimization/al12/02_cutting_plane_full_cloud.jl" "${CP_N_MEMBERS:-0}"
echo "finished=$(date)"
