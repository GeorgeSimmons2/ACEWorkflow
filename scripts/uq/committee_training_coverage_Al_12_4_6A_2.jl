# committee_training_coverage_Al_12_4_6A_2.jl
#
# Training-set coverage: our 30-datapoint-built CONSTRAINED committee vs the
# FULL naive POPS hypercube from ACEWorkflow's POPS module.
#
#   pred[i,k] = A[i,:] · θ_k    over design rows i and members k
#   covered[i] = min_k pred[i,k] < Y[i] < max_k pred[i,k]
#
# The naive baseline is the stock pipeline, not a reimplementation:
#   corrections(X, Y, Γ; leverage_percentile, lambda)   → pointwise cloud over ALL
#                                                         datapoints passing the mask
#   hypercube(cloud)                                    → unclipped bounding box
#   rejection_sample_hypercube(..., θ->true)            → members, no predicate
# with X = W·A·P⁻¹, Y = W·y, Γ = P, λ = 1/M — the same quantities the constrained
# pipeline fits in.
#
# Both sets are drawn with the same sampler and the same member count, so the only
# differences are (a) which cloud the box is fitted to and (b) whether the physics
# predicate is applied.
#
# Uses the RAW A and Y for prediction (not the weighted/preconditioned Ap, Yw) so
# intervals are in physical units, comparable to Y.
#
# Run:  julia --project -t 8 scripts/uq/committee_training_coverage_Al_12_4_6A_2.jl
#       argv[1] = committee CSV      (default: ncell4_densek rejection)
#       argv[2] = leverage_percentile for the naive cloud (default 0.5, the POPS default)

include(joinpath(@__DIR__, "..", "bandpath_phonon_uq", "lib.jl"))
using Random
Random.seed!(1234)

element, dataset = :Al, ""
result = load_model(element, 12, 4, 6, 2; dataset_name=dataset)
A, Y, lin = result.A, result.Y, result.lin_params
P = result.P
X  = Diagonal(result.W)*A/P                 # preconditioned, weighted design
Yw = result.W .* Y
M, K = size(A); λ = 1.0/M

committee_csv = length(ARGS) >= 1 ? ARGS[1] :
    "$(result.dir)/results/bandpath_undotted_ncell4_densek/committee_rejection.csv"
lev_pct = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.5

Θ_con = readdlm(committee_csv, ',')
size(Θ_con, 2) == K || error("committee has $(size(Θ_con,2)) columns, design has $K")
N = size(Θ_con, 1)
@printf("design    : %d rows × %d params\n", M, K)
@printf("committee : %s\n            %d members (built from %d datapoints)\n\n",
        committee_csv, N, N)

# ── full naive POPS cloud, straight from the module ──────────────────────────
@printf("── naive POPS pipeline (ACEWorkflow POPS module) ────────────────\n")
Γ = Matrix(P)
cloud = corrections(Matrix(X), Vector(Yw), Γ; leverage_percentile=lev_pct, lambda=λ)
@printf("  corrections(leverage_percentile=%.2f) → cloud %d × %d\n",
        lev_pct, size(cloud,1), size(cloud,2))
@printf("  ‖δθ‖ over the cloud: median %.4g, max %.4g\n",
        median(norm.(eachrow(cloud))), maximum(norm.(eachrow(cloud))))
hyp_eig, hyp_bnd = hypercube(cloud)
@printf("  hypercube: %d retained directions, mean width %.4g\n",
        size(hyp_eig,2), mean(hyp_bnd[2,:] .- hyp_bnd[1,:]))
nai_mat, _ = rejection_sample_hypercube(hyp_eig, hyp_bnd, lin, θ -> true;
                                        number_of_committee_members=N,
                                        max_attempts=1_000_000)
Θ_naive = nai_mat'

# ── coverage ─────────────────────────────────────────────────────────────────
pt = A * lin
function coverage(Θ, label)
    pred = A * Θ'
    lo = vec(minimum(pred; dims=2)); hi = vec(maximum(pred; dims=2))
    cov = (lo .< Y) .& (Y .< hi); w = hi .- lo; miss = .!cov
    relmiss = any(miss) ? median(max.(lo[miss].-Y[miss], Y[miss].-hi[miss]) ./ max.(w[miss],eps())) : 0.0
    @printf("  %-30s %7.2f%%  %10.4g %10.4g  %8d %8d  %9.3g\n",
            label, 100*count(cov)/M, median(w), mean(w),
            count(Y .<= lo), count(Y .>= hi), relmiss)
    return (; lo, hi, cov, w)
end

@printf("\n── training-set coverage (%d members each) ──────────────────────────────────\n", N)
@printf("  %-30s %8s  %10s %10s  %8s %8s  %9s\n",
        "set", "coverage", "med width", "mean width", "below", "above", "med miss/w")
r_con = coverage(Θ_con,   "constrained + rejection (30 pts)")
r_nai = coverage(Θ_naive, "naive POPS full cloud")

@printf("\n  point-model RMSE on training : %.4g\n", sqrt(mean((Y .- pt).^2)))
@printf("  |Y - point model|            : median %.4g, mean %.4g\n",
        median(abs.(Y .- pt)), mean(abs.(Y .- pt)))
@printf("  point model inside envelope  : constrained %.2f%%, naive %.2f%%\n",
        100*count((r_con.lo .< pt) .& (pt .< r_con.hi))/M,
        100*count((r_nai.lo .< pt) .& (pt .< r_nai.hi))/M)

ratio = r_nai.w ./ max.(r_con.w, eps())
@printf("\n  naive width / constrained width: median %.3g, p90 %.3g, max %.3g\n",
        median(ratio), quantile(ratio, 0.9), maximum(ratio))
@printf("  constrained covers, naive does not: %d\n", count(r_con.cov .& .!r_nai.cov))
@printf("  naive covers, constrained does not: %d\n", count(r_nai.cov .& .!r_con.cov))

stem = "$(result.dir)/results/training_coverage_vs_naivePOPS_$(basename(dirname(committee_csv)))"
writedlm("$stem.csv",
    vcat(["row" "Y" "point" "con_lo" "con_hi" "con_cov" "nai_lo" "nai_hi" "nai_cov"],
         hcat(1:M, Y, pt, r_con.lo, r_con.hi, r_con.cov, r_nai.lo, r_nai.hi, r_nai.cov)), ',')
@printf("\nper-row table → %s.csv\n", stem)
