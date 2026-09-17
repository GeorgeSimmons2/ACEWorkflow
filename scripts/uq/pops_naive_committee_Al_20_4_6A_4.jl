# pops_naive_committee_Al_20_4_6A_4.jl
#
# Five COMPLETELY NAIVE POPS hypercube samples of Al_20_4_6A_4 (5476 basis
# functions), for the NPT temperature sweep in
# scripts/uq/npt_naive_member_Al_20_4_6A_4.jl.
#
# "Naive" here means exactly what it means for committee_B_naive on
# Al_12_4_6A_2_: the plain POPS pipeline with NO constraints of any kind --
# no a_eq equality, no Born rows, no phonon rejection.
#
#     cloud  = corrections(Ap, Yw, P; leverage_percentile, lambda)
#     be, bb = hypercube(cloud)
#     Θ      = rejection_sample_hypercube(be, bb, lin, θ->true; ...)
#
# The trivially-true predicate keeps the same code path the constrained runs
# use, so the only difference between arms is the predicate. This mirrors
# scripts/uq/aeq_constrained_pops_coverage_Al_12_4_6A_2.jl (committee B) line
# for line -- deliberately, so the two studies are comparable.
#
# MEMORY. A is 146958 x 5476. corrections() computes leverage as diag(X*A) with
# X = M x K and A = K x M, so it materialises an M x M matrix: 146958^2 * 8 B =
# 172.8 GB. Note this does NOT scale with the number of parameters -- Al_12 has
# the same 146958 rows and therefore the same 172.8 GB for this step. What grows
# from 91 to 5476 parameters is only C (K x K, 0.24 GB) and C\X' (K x M, 6.4 GB).
# Peak is roughly 190 GB; the hmem allocation of 20 x 32 GB covers it.
#
# The 15 GB A.csv is read by readdlm inside load_model and dominates startup
# (expect tens of minutes).
#
# OUTPUTS -> models/Al_20_4_6A_4/results/pops_naive_committee/
#   committee_naive.csv    5 x 5476, one member per ROW (NPT driver reads rows)
#   member_diagnostics.csv per-member a_eq and ||δθ||
#   metadata.csv           seeds, leverage percentile, SHA-256 of the committee
#
# Run:  julia --project -t 20 scripts/uq/pops_naive_committee_Al_20_4_6A_4.jl [lev_pct] [n_members]

using LinearAlgebra, DelimitedFiles, Statistics, Printf, Random, SHA
using ACEpotentials, ACEWorkflow, Unitful

@async while true; flush(stdout); sleep(5); end     # Julia block-buffers to files

element     = :Al
lev_pct     = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 0.5
n_members   = length(ARGS) >= 2 ? parse(Int,     ARGS[2]) : 5
sample_seed = 20260729

Random.seed!(sample_seed)

@printf("── loading Al_20_4_6A_4 (reads the 15 GB A.csv — slow) ──\n"); flush(stdout)
t0 = time()
result = load_model(element, 20, 4, 6, 4; dataset_name="full")
A, Y, P, W, lin = result.A, result.Y, result.P, result.W, result.lin_params
M, K = size(A); λ = 1.0 / M
model    = result.model
modelname = result.name              # captured before `result` is dropped below
outdir   = "$(result.dir)/results/pops_naive_committee"; mkpath(outdir)
@printf("  %s: %d rows × %d params  [%.1f min]\n", modelname, M, K, (time()-t0)/60)
flush(stdout)

# preconditioned design matrix — same convention as every other study script
Ap = Diagonal(W) * A / P
Yw = W .* Y

# Release the raw design matrix (6.4 GB). Clearing the local `A` alone does NOT
# free it: `result` is a NamedTuple that still holds a reference, so every field
# we still need must be copied out first and `result` dropped as well.
result = nothing
A = nothing
GC.gc()

@printf("── POPS corrections (leverage_percentile = %.2f, λ = %.3e) ──\n", lev_pct, λ)
@printf("   expect a %.1f GB temporary for diag(X*A)\n", M^2*8/1e9); flush(stdout)
t0 = time()
# Ap/Yw/P are already dense Float64 — pass directly rather than via Matrix(),
# which would risk an extra 6.4 GB copy of the design matrix.
cloud = corrections(Ap, Yw, P; leverage_percentile = lev_pct, lambda = λ)
@printf("  cloud %d × %d  [%.1f min]\n", size(cloud,1), size(cloud,2), (time()-t0)/60)
flush(stdout)

@printf("── hypercube ──\n"); flush(stdout)
be, bb = hypercube(cloud)
@printf("  %d directions, mean width %.4g\n", size(be,2), mean(bb[2,:] .- bb[1,:]))
flush(stdout)

@printf("── drawing %d NAIVE samples (predicate θ->true) ──\n", n_members); flush(stdout)
Θ = (rejection_sample_hypercube(be, bb, lin, θ -> true;
        number_of_committee_members = n_members,
        max_attempts = 1_000_000)[1])'          # -> n_members × K, one member per row

@assert size(Θ) == (n_members, K) "expected $n_members × $K, got $(size(Θ))"
writedlm("$outdir/committee_naive.csv", Θ, ',')

# ── per-member diagnostics ───────────────────────────────────────────────────
# a_eq is the single most informative naive-committee statistic: the constrained
# committees pin it exactly by b′(a_eq)·θ = 0, whereas the naive cloud spreads it.
# (`model` was copied out of `result` before `result` was dropped.)
@printf("\n  %-8s %12s %14s\n", "member", "‖θ−θ_RLS‖", "a_eq (Å)")
diag_rows = Vector{Vector{Float64}}()
a_rls = ACEWorkflow.relax_lattice_constant(model, element)   # with lin_params still set
for i in 1:n_members
    θ = Θ[i, :]
    ACEpotentials.Models.set_linear_parameters!(model, θ)
    aeq = try
        ACEWorkflow.relax_lattice_constant(model, element)
    catch err
        @printf("    member %d: relaxation failed (%s)\n", i, sprint(showerror, err))
        NaN
    end
    push!(diag_rows, [Float64(i), norm(θ .- lin), aeq])
    @printf("  %-8d %12.4f %14.5f\n", i, norm(θ .- lin), aeq); flush(stdout)
end
ACEpotentials.Models.set_linear_parameters!(model, lin)      # restore

D = reduce(hcat, diag_rows)'
open("$outdir/member_diagnostics.csv", "w") do io
    println(io, "# a_eq(RLS) = $a_rls Å")
    println(io, "member,norm_dtheta,a_eq_Ang")
    for r in eachrow(D)
        @printf(io, "%d,%.6f,%.6f\n", Int(r[1]), r[2], r[3])
    end
end

finite = filter(isfinite, D[:, 3])
if length(finite) >= 2
    @printf("\n  a_eq: RLS %.5f | members %.5f – %.5f  (spread %.5f Å = %.2f%%)\n",
            a_rls, minimum(finite), maximum(finite),
            maximum(finite)-minimum(finite),
            100*(maximum(finite)-minimum(finite))/minimum(finite))
end

comm_sha = bytes2hex(sha256(read("$outdir/committee_naive.csv")))
open("$outdir/metadata.csv", "w") do io
    println(io, "key,value")
    println(io, "model,$modelname")
    println(io, "n_rows,$M"); println(io, "n_params,$K")
    println(io, "leverage_percentile,$lev_pct")
    println(io, "lambda,$λ")
    println(io, "sample_seed,$sample_seed")
    println(io, "n_members,$n_members")
    println(io, "constraints,none (naive hypercube, predicate = true)")
    println(io, "a_eq_rls,$a_rls")
    println(io, "committee_sha256,$comm_sha")
end

@printf("\noutputs → %s/\n", outdir)
@printf("next: sbatch scripts/uq/run_npt_naive_Al_20_4_6A_4.slurm\n")
