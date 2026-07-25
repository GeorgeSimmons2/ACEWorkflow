#!/bin/bash
#SBATCH --job-name=cachediag_Al12
#SBATCH --partition=hmem
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=32000
#SBATCH --time=01:30:00
#SBATCH --output=cachediag_Al12_%j.log

# Confirm the stale-cache phonon bug for Al_12_4_6A_2_.
# Run AFTER deleting models/Al_12_4_6A_2_/results/undotted_Hbasis_3x3x3_a4.04494.jls
# (1) undotted_hessian_test.jl — per-basis Hessian ASSEMBLY check (should pass; does
#     NOT touch the cache/Bloch path, so it will NOT catch the bug — sanity only).
# (2) diagnose_stale_cache — rebuilds the cache fresh and re-checks the committee's
#     min ω at a_mean.  Stable ⇒ the stale cache was the culprit.

REPO=/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow

echo "===== (1) per-basis Hessian assembly check ====="
julia -t 20 --project=$REPO -e "include(\"$REPO/scripts/uq/undotted_hessian_test.jl\")"

echo "===== (2) fresh-cache committee re-check (the real test) ====="
julia -t 20 --project=$REPO -e "include(\"$REPO/scripts/uq/diagnose_stale_cache_Al_12_4_6A_2.jl\")"
