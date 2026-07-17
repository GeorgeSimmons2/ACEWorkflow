# build_subsets_Al_20_4_6A_4.jl
#
# Builds the Al_20_4_6A_4 model (totaldegree=20, smoothness=4, rcut=6Å, order=4)
# on each of the reduced-data training subsets, so it can be compared against
# the already-built full-data model in models/Al_20_4_6A_4/.
#
# Reuses the same subset files already used for the Al_*_4_6A_2 / _4_6A_3
# families (data/Al/subset_{5,10,20,50}_percent.extxyz), so the data-fraction
# comparison stays apples-to-apples across degree/order settings.
#
# Each build writes to models/Al_20_4_6A_4_subset_<p>_percent/, following the
# ACEWorkflow.Models naming convention (dataset_name="subset_<p>_percent").
#
# NB: Al_20_4_6A_4 has 5476 basis functions and the full-data design matrix
# (A.csv) is ~8 GB — even the 50% subset build is a heavy, long-running job.
# Run this via scripts/slurm/build_subsets_Al_20_4_6A_4.slurm, not
# interactively.
#
# Usage:
#   sbatch scripts/slurm/build_subsets_Al_20_4_6A_4.slurm

using Distributed
addprocs(Sys.CPU_THREADS)

@everywhere begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", ".."))
end

@everywhere using ACEWorkflow

const ELEMENT     = :Al
const TOTALDEGREE = 20
const SMOOTHNESS  = 4
const RCUT        = 6.0
const ORDER       = 4

const DATA_DIR = joinpath(@__DIR__, "..", "..", "data", "Al")
const PERCENTS = [5, 10, 20, 50]

for p in PERCENTS
    training_xyz = joinpath(DATA_DIR, "subset_$(p)_percent.extxyz")
    dataset_name = "subset_$(p)_percent"

    @info "═══ Building Al_20_4_6A_4 — $(dataset_name) ═══" training_xyz

    result = build_model(ELEMENT, TOTALDEGREE, SMOOTHNESS, RCUT, ORDER;
                          training_xyz = training_xyz,
                          dataset_name = dataset_name)

    @info "  Saved to $(result.dir)"
end

@info "All subset builds complete."
