# committee_training_coverage_Al_12_4_6A_2.jl
#
# Coverage of committees on the TRAINING data, constrained vs naive.
#
#   pred[i,k] = A[i,:] · θ_k      for every design-matrix row i and member k
#   lo[i], hi[i] = min/max over k
#   covered[i] = lo[i] < Y[i] < hi[i]
#
# Compares three sets on identical footing (same rows, same member count):
#   1. constrained + rejection   — the saved committee CSV
#   2. naive hypercube samples   — same sampler, box fitted to the naive POPS cloud,
#                                  NO predicate.  The apples-to-apples control.
#   3. naive delta forest        — the raw pointwise-correction members the box is
#                                  fitted to (shown because it is free, and it is the
#                                  cloud, not a sample of it)
#
# The naive sets are regenerated deterministically (Random.seed!(1234) and the same
# 5 leverage / 10 residual / 15 random selection as the committee scripts), since the
# ncell4_densek run saved no naive CSV.
#
# Uses the RAW design matrix A and data vector Y — not the weighted/preconditioned
# Ap, Yw used for fitting — so intervals are in physical units, comparable to Y.
#
# Run:  julia --project -t 8 scripts/uq/committee_training_coverage_Al_12_4_6A_2.jl
#       optional argv[1] = committee CSV to test (default: ncell4_densek rejection)

include(joinpath(@__DIR__, "..", "bandpath_phonon_uq", "lib.jl"))
using Random
Random.seed!(1234)

element, dataset = :Al, ""
n_lev, n_res, n_rand = 5, 10, 15

result = load_model(element, 12, 4, 6, 2; dataset_name=dataset)
A, Y, lin = result.A, result.Y, result.lin_params
P = result.P; Ap = Diagonal(result.W)*result.A/P; Yw = result.W.*result.Y; λ = 1.0/size(Ap,1)
M, K = size(A)

committee_csv = length(ARGS) >= 1 ? ARGS[1] :
    "$(result.dir)/results/bandpath_undotted_ncell4_densek/committee_rejection.csv"
Θ_con = readdlm(committee_csv, ',')                  # N × K
size(Θ_con, 2) == K || error("committee has $(size(Θ_con,2)) columns, design matrix has $K")
N = size(Θ_con, 1)
@printf("design    : %d rows × %d params\n", M, K)
@printf("committee : %s  (%d members)\n\n", committee_csv, N)

# ── rebuild the naive POPS cloud exactly as the committee scripts do ─────────
C = Symmetric(Ap'*Ap .+ λ.*(P'*P)); Cf = cholesky(C)
AtX = Cf\Matrix(Ap'); θ̃ = Cf\(Ap'*Yw)
leverage = vec(sum(Ap'.*AtX; dims=1)); residual = Yw .- Ap*θ̃
forest_member(i) = lin .+ (P \ (AtX[:, i] .* (residual[i]/leverage[i])))

lev_idx = sortperm(leverage; rev=true)[1:n_lev]
res_idx = Int[]; for i in sortperm(abs.(residual); rev=true)
    i in lev_idx && continue; push!(res_idx, i); length(res_idx) == n_res && break
end
taken = Set(vcat(lev_idx, res_idx)); rand_idx = Int[]
while length(rand_idx) < n_rand
    i = rand(1:length(Yw)); (i in taken) && continue; push!(rand_idx, i); push!(taken, i)
end
selected = vcat(lev_idx, res_idx, rand_idx)
Θ_forest = reduce(hcat, [forest_member(i) for i in selected])'      # 30 × K

# naive hypercube samples: same sampler as the constrained committee, no predicate
naive_deltas = Θ_forest .- lin'
hyp_eig, hyp_bnd = hypercube(Matrix(naive_deltas))
nai_mat, _ = rejection_sample_hypercube(hyp_eig, hyp_bnd, lin, θ -> true;
                                        number_of_committee_members=N,
                                        max_attempts=1_000_000)
Θ_naive = nai_mat'                                                   # N × K

# ── coverage of one member set ───────────────────────────────────────────────
pt = A * lin
function coverage(Θ, label)
    pred = A * Θ'
    lo   = vec(minimum(pred; dims=2)); hi = vec(maximum(pred; dims=2))
    cov  = (lo .< Y) .& (Y .< hi); w = hi .- lo
    miss = .!cov
    relmiss = if any(miss)
        d = max.(lo[miss] .- Y[miss], Y[miss] .- hi[miss])
        median(d ./ max.(w[miss], eps()))
    else 0.0 end
    @printf("  %-26s %7.2f%%  %10.4g %10.4g %10.4g  %8d %8d  %9.3g\n",
            label, 100*count(cov)/M, median(w), mean(w), maximum(w),
            count(Y .<= lo), count(Y .>= hi), relmiss)
    return (; lo, hi, cov, w, label)
end

@printf("── training-set coverage (%d members each) ───────────────────────────────────────\n", N)
@printf("  %-26s %8s  %10s %10s %10s  %8s %8s  %9s\n",
        "set", "coverage", "med width", "mean width", "max width", "below", "above", "med miss/w")
r_con = coverage(Θ_con,    "constrained + rejection")
r_nai = coverage(Θ_naive,  "naive hypercube")
r_for = coverage(Θ_forest, "naive delta forest")

@printf("\n  point-model RMSE on training : %.4g\n", sqrt(mean((Y .- pt).^2)))
@printf("  |Y - point model|            : median %.4g, mean %.4g\n",
        median(abs.(Y .- pt)), mean(abs.(Y .- pt)))
@printf("  point model inside envelope  : constrained %.2f%%, naive %.2f%%\n",
        100*count((r_con.lo .< pt) .& (pt .< r_con.hi))/M,
        100*count((r_nai.lo .< pt) .& (pt .< r_nai.hi))/M)

# width ratio: how much wider is naive, row by row?
ratio = r_nai.w ./ max.(r_con.w, eps())
@printf("\n  naive width / constrained width: median %.3g, p90 %.3g, max %.3g\n",
        median(ratio), quantile(ratio, 0.9), maximum(ratio))
@printf("  rows where constrained covers but naive does not: %d\n", count(r_con.cov .& .!r_nai.cov))
@printf("  rows where naive covers but constrained does not: %d\n", count(r_nai.cov .& .!r_con.cov))

outdir = "$(result.dir)/results"
stem = "$outdir/training_coverage_$(basename(dirname(committee_csv)))"
writedlm("$stem.csv",
    vcat(["row" "Y" "point" "con_lo" "con_hi" "con_cov" "nai_lo" "nai_hi" "nai_cov"],
         hcat(1:M, Y, pt, r_con.lo, r_con.hi, r_con.cov, r_nai.lo, r_nai.hi, r_nai.cov)), ',')
@printf("\nper-row table → %s.csv\n", stem)
