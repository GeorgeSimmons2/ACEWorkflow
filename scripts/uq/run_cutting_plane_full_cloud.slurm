#!/bin/bash
#SBATCH --job-name=cutplane_cloud
#SBATCH --partition=compute
#SBATCH --ntasks=42
#SBATCH --mem-per-cpu=3988
#SBATCH --time=04:00:00
#SBATCH --output=cutplane_cloud_%j.log
#
# Cutting-plane constrain the full top-50%-leverage POPS cloud of Al_12_4_6A_2_
# (73,479 of 146,958 observations).
#
# CALIBRATION from a 2,000-member pilot at 16 threads:
#   mean 73.7 ms/member, median 26.9 ms, max 3.2 s  (single-thread cost)
#   24.5% needed >= 1 cut; mean 1.90 cuts, max 143
#   18 s wall for 2,000  ->  ~11 min for 73,479 at 16 threads
# 32 threads and 4 h give a wide margin; the job should finish well inside it.
#
# MEMORY. A.csv is 259 MB, Ap ~107 MB, the 91 x M leverage solve another 107 MB
# (freed straight after selection). Peak is a few GB, so 32 x 4 GB is generous;
# the allocation is really about getting the cores.
#
# NOT ALL MEMBERS COME OUT STABLE. The pilot had 5/2000 hit max_cuts and 1/2000
# finish below the margin. Filter on min_omega_THz in member_diagnostics.csv
# before using the committee -- do not assume every column of Theta is feasible.
#
# Outputs -> models/Al_12_4_6A_2_/results/cutting_plane_full_cloud/
#   committee_full_cloud.jls   (Theta 91 x N, ~53 MB, plus diagnostics; deserialize)
#   member_diagnostics.csv     per-member cuts / iters / min omega / status / time

set -euo pipefail
REPO=/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow
cd "$REPO"

echo "host=$(hostname) cpus=${SLURM_NTASKS:-?} started=$(date)"
julia --project -t "${SLURM_NTASKS:-32}" \
  "$REPO/scripts/uq/cutting_plane_full_cloud_Al_12_4_6A_2.jl" 0
echo "finished=$(date)"
