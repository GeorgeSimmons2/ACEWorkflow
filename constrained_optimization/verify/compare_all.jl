# compare_all.jl — every reproduced parameter set against its published counterpart,
# in one table, with an explicit reproducibility tier per row.
#
# Reads only CSVs; no model, no Hessian, no solver.  Runs in seconds and is safe to run
# at any point — stages that have not run yet report "—" rather than failing.
#
#   julia --project constrained_optimization/verify/compare_all.jl
#
# Writes constrained_optimization/REPRODUCTION.md and exits non-zero if anything FAILED,
# so it works as a gate in run_pipeline.sh.
#
# Env:  OUT  (report path)

using Dates
include(joinpath(@__DIR__, "..", "lib_verify.jl"))

ROOT = rstrip(normpath(joinpath(@__DIR__, "..", "..")), '/')
AL12 = "$ROOT/models/Al_12_4_6A_2_/results"
AL16 = "$ROOT/models/Al_16_4_6A_3_/results"
WDIR = "$ROOT/models/W_20_4_5A_3"
OUT  = get(ENV, "OUT", normpath(joinpath(@__DIR__, "..", "REPRODUCTION.md")))

checks = Check[]

# ── W: the dead end this pipeline exists to close ────────────────────────────
push!(checks, check_vector("W positive-core θ", :converged,
    "$WDIR/positive_core_constrained_parameters.csv",
    "$WDIR/repro_positive_core_constrained_parameters.csv"))

# ── Al_12 stage 1: the constrained mean model, and the 30-member forest ───────
push!(checks, check_vector("Al_12 θ_mean (ncell4 densek)", :converged,
    "$AL12/bandpath_undotted_ncell4_densek/theta_mean.csv",
    "$AL12/repro_bandpath_undotted_ncell4_densek/theta_mean.csv"))
push!(checks, check_ensemble("Al_12 committee_repaired (30)", :statistical,
    "$AL12/bandpath_undotted_ncell4_densek/committee_repaired.csv",
    "$AL12/repro_bandpath_undotted_ncell4_densek/committee_repaired.csv"))
push!(checks, check_ensemble("Al_12 committee_rejection (30)", :statistical,
    "$AL12/bandpath_undotted_ncell4_densek/committee_rejection.csv",
    "$AL12/repro_bandpath_undotted_ncell4_densek/committee_rejection.csv"))

# ── Al_12 stage 2: the 73k cutting-plane cloud ───────────────────────────────
# member_diagnostics.csv is 2.9 MB of per-member rows; only its min ω distribution is
# meaningful, and only sorted (the member order carries no information).
push!(checks, check_distribution("Al_12 cutting-plane cloud, min ω", :statistical,
    "$AL12/cutting_plane_full_cloud/member_diagnostics.csv",
    "$AL12/repro_cutting_plane_full_cloud/member_diagnostics.csv";
    col = 5, unit = "THz", skipstart = 2))   # min_omega_THz, past the # comment and the header

# ── Al_12 stage 3: the constrained committee behind the top-right panel ──────
push!(checks, check_ensemble("Al_12 rejection committee (full cloud)", :statistical,
    "$AL12/cutting_plane_full_cloud/committee_rejection_full_cloud.csv",
    "$AL12/repro_cutting_plane_full_cloud/committee_rejection_full_cloud.csv"))

# ── Al_12 stage 4: the naive ensemble behind the top-left panel ──────────────
# There is no QP in this path, only a seeded sample_hypercube draw, so :exact looked
# right.  MEASURED: it is not.  max|Δ| = 9.6e-2, ‖Δ‖/‖θ‖ = 5.7e-3 — small, but not zero.
# The draw is seeded, but the box it draws from comes from an eigendecomposition of the
# leverage cloud, and that is only reproducible to linear-algebra tolerance (threaded BLAS
# reduction order, near-degenerate directions).  The npt README's "naive reproduces
# bit-exactly" was measured on the FOREST path, which does not build a hypercube.
push!(checks, check_vector("Al_12 naive ensemble (30 × 91)", :converged,
    "$AL12/naive_vs_constrained/samples_naive.csv",
    "$AL12/repro_naive_vs_constrained/samples_naive.csv"))
push!(checks, check_distribution("Al_12 naive relaxed a", :converged,
    "$AL12/naive_vs_constrained/min_freq_naive.csv",
    "$AL12/repro_naive_vs_constrained/min_freq_naive.csv"; col = 2, unit = "Å"))

# ── Al_16: the bottom row.  Its ensemble_*.csv were overwritten on 19 Aug, so the
#    published reference is the per-member (a, min ω) table that survived.
# The published stage 2 sampled N_PER_SEG=20 (101 q-points); the first rerun used
# [20,20,20,20,60] (141), which shifts every min ω down ~0.1 THz — the mean model, same θ
# in both, moves 0.408 → 0.315.  Compare against the GRID-MATCHED rerun instead, or the
# comparison measures the q-grid rather than the parameters.
AL16_REPRO = get(ENV, "AL16_REPRO", "$AL16/repro_pinned_ensembles")
for tag in ("unconstrained", "constrained")
    push!(checks, check_distribution("Al_16 $tag, native min ω (grid-matched)", :converged,
        "$AL16/pinned_ensembles/bands_$tag.csv",
        "$AL16_REPRO/bands_$tag.csv"; col = 3, unit = "THz"))
end

# ── the two NPT committees (npt_trajectories stage 1, chained by run_pipeline.sh npt) ──
# The MD member each arm selects is a single committee member, so it is NOT compared
# element-wise — only the committees' mean models and distributions are.
for (sub, what) in (("bandpath_undotted_multivolume", "multi-volume"), ("bandpath_undotted", "a_eq"))
    push!(checks, check_vector("NPT $what committee θ_mean", :converged,
        "$AL12/$sub/theta_mean.csv", "$AL12/repro_$sub/theta_mean.csv"))
    push!(checks, check_ensemble("NPT $what committee_rejection (30)", :statistical,
        "$AL12/$sub/committee_rejection.csv", "$AL12/repro_$sub/committee_rejection.csv"))
end

# ── report ───────────────────────────────────────────────────────────────────
open(OUT, "w") do io
    println(io, "# Reproduction report")
    println(io)
    println(io, "Generated ", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM"),
                " by `constrained_optimization/verify/compare_all.jl`.")
    println(io)
    println(io, """
    Tiers, and why they differ — see `lib_verify.jl` for the measurements behind them:

    - `exact` — no QP in the member path (a seeded draw, or a closed-form correction).
      Must match to machine precision; anything else is a bug, not solver noise.
    - `converged` — one well-posed QP. The iterate path may differ, the optimum does not.
    - `statistical` — a cutting-plane cascade. Members and their order are not
      reproducible; the distribution is. Never quote a member index.
    """)
    println(io)
    report(checks, io)
end

nf = report(checks)
@printf("\nreport → %s\n", OUT)
exit(nf == 0 ? 0 : 1)
