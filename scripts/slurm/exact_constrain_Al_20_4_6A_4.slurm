#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=7976
#SBATCH --time=12:00:00

# Exact-constrains Al_20_4_6A_4 (C11/C12/C44 -> literature targets) across all
# 5 dataset sizes. Each fit solves a dense 5476x5476 OSQP system, so this is
# memory-heavy even though it's single-process (no Distributed workers needed
# here — OSQP/LinearAlgebra use BLAS threads instead).
# Requires scripts/model_building/build_subsets_Al_20_4_6A_4.jl to have
# completed first for the 4 subset sizes.

julia --project=/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow -e 'include("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/scripts/elasticity/exact_constrain_Al_20_4_6A_4.jl")'
