# aeq_constrained_pops_coverage_Al_12_4_6A_2.jl
#
# POPS committee constrained ONLY to share the mean equilibrium lattice constant,
# then rejection-sampled, then coverage-compared against the stock naive POPS
# pipeline (and, if present, the phonon-constrained committee).
#
# For each observation i the member solves
#
#     min ½‖Ãθ̃ − ỹ‖² + (λ/2)‖Pθ̃‖²
#     s.t.  Ã_i·θ̃ = ỹ_i        (interpolate observation i — the POPS condition)
#           b′(a_eq)·θ = 0      (equilibrium pinned to the reference a_eq)
#           b″(a_eq)·θ ≥ ε      (…and it is a minimum, not a maximum)
#
# with Ã = W·A·P⁻¹, ỹ = W·y, θ̃ = Pθ, λ = 1/M.
#
# This is ONE QP per member — no cutting-plane loop — because there are no phonon
# constraints, so it runs in minutes on a few cores rather than hours on a cluster.
# It isolates what the lattice-constant pinning alone does to committee calibration,
# separately from the Born and phonon rows.
#
# Rejection sampling cannot impose the equality b′·θ = 0 (zero acceptance probability
# under a continuous proposal), so the predicate uses the band form
#   |b′·(θ − θ_mean)| / (b″·θ_mean) ≤ aeq_tol   ≈  |Δa_eq| ≤ aeq_tol Å
# together with b″·θ ≥ ε, matching the constrained pipeline.
#
# Run locally:  julia --project -t 8 scripts/uq/aeq_constrained_pops_coverage_Al_12_4_6A_2.jl
#   argv[1] = leverage_percentile   (default 0.5; 0.0 = every observation)
#   argv[2] = n_samples             (default 30)

include(joinpath(@__DIR__, "..", "bandpath_phonon_uq", "lib.jl"))
using SparseArrays, OSQP, Random, Printf
Random.seed!(1234)
@async while true; flush(stdout); sleep(5); end

element, dataset = :Al, ""
lev_pct    = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 0.5
n_samples  = length(ARGS) >= 2 ? parse(Int, ARGS[2])     : 30
aeq_tol    = 0.1        # Å; band on the member's equilibrium shift
bpp_floor  = 1e-9       # b″·θ ≥ this
include_born = false    # true → also impose the three cubic Born rows
gc_every   = 5_000      # collect periodically; these QPs are small but numerous

result = load_model(element, 12, 4, 6, 2; dataset_name=dataset)
model, lin = result.model, result.lin_params
A, Y = result.A, result.Y
P  = result.P
Ap = Diagonal(result.W)*A/P; Yw = result.W .* Y
M, K = size(A); λ = 1.0/M
outdir = "$(result.dir)/results/aeq_constrained_pops"; mkpath(outdir)
@printf("model %s: %d rows × %d params, %d threads\n", result.name, M, K, Threads.nthreads())
flush(stdout)

# ── equation-of-state rows at the mean-model a_eq ────────────────────────────
a_eq = ACEWorkflow.relax_lattice_constant(model, element)
lattice_basis(a) = ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(
                        ACEWorkflow.Elasticity.reference_system(element; a=a), model))
b_prime  = ForwardDiff.derivative(lattice_basis, a_eq)
b_dprime = ForwardDiff.derivative(a -> ForwardDiff.derivative(lattice_basis, a), a_eq)
@printf("a_eq = %.5f Å   (b′·θ_lsq = %.3e, b″·θ_lsq = %.3e)\n",
        a_eq, dot(b_prime, lin), dot(b_dprime, lin))

ineq_rows, ineq_lower = b_dprime', [bpp_floor]
if include_born
    H_el = elastic_hessian_basis(model; element=element, a=a_eq)
    c11 = reshape(H_el,36,K)[1,:]; c12 = reshape(H_el,36,K)[7,:]; c44 = reshape(H_el,36,K)[22,:]
    ineq_rows  = vcat(c44', (c11.-c12)', (c11.+2 .*c12)', b_dprime')
    ineq_lower = [0.1, 1.0, 0.1, bpp_floor]
end
@printf("inequality rows: %d  (Born %s)\n\n", size(ineq_rows,1), include_born ? "ON" : "OFF")
flush(stdout)

# ── QP: one solve per member, no cutting planes ──────────────────────────────
Hqp = sparse(Ap'*Ap .+ λ.*(P'*P)); qqp = -(Ap'*Yw)
const POOL = [OSQP.Model() for _ in 1:Threads.nthreads()]
setup_solve(osqp, Afull, l, u) = begin
    OSQP.setup!(osqp; P=Hqp, q=qqp, A=Afull, l=l, u=u, max_iter=1_000_000,
                check_termination=25, verbose=false, eps_abs=1e-6, eps_rel=1e-6)
    P \ OSQP.solve!(osqp).x
end
constrain_i(osqp, i) = setup_solve(osqp,
    vcat(sparse(Ap[i,:]'), sparse(b_prime'/P), sparse(ineq_rows/P)),
    vcat([Yw[i]], [0.0], ineq_lower),
    vcat([Yw[i]], [0.0], fill(Inf, length(ineq_lower))))
mean_fit() = setup_solve(POOL[1],
    vcat(sparse(b_prime'/P), sparse(ineq_rows/P)),
    vcat([0.0], ineq_lower),
    vcat([0.0], fill(Inf, length(ineq_lower))))

θ_mean = mean_fit()
@printf("constrained mean: b′·θ = %.3e, b″·θ = %.4g\n", dot(b_prime,θ_mean), dot(b_dprime,θ_mean))
flush(stdout)

# ── leverage mask (same rule as POPSRegression.corrections) ──────────────────
C = Symmetric(Ap'*Ap .+ λ.*(P'*P)); Cf = cholesky(C)
AtX = Cf\Matrix(Ap'); θ̃ = Cf\(Ap'*Yw)
leverage = vec(sum(Ap'.*AtX; dims=1)); residual = Yw .- Ap*θ̃
kept = findall(leverage .>= quantile(leverage, lev_pct)); n_obs = length(kept)
@printf("leverage_percentile = %.2f → %d of %d observations\n", lev_pct, n_obs, M)
flush(stdout)

# ── constrain every kept observation ─────────────────────────────────────────
Θcon = zeros(K, n_obs); t0 = time()
for c0 in 1:gc_every:n_obs
    c1 = min(c0+gc_every-1, n_obs)
    Threads.@threads :static for j in c0:c1
        Θcon[:, j] = constrain_i(POOL[Threads.threadid()], kept[j])
    end
    GC.gc()
    el = time()-t0; f = c1/n_obs
    @printf("  %d/%d (%.1f%%) | %.1f min elapsed | est. total %.1f min | heap %.1f GB\n",
            c1, n_obs, 100f, el/60, el/60/f, Sys.maxrss()/2^30); flush(stdout)
end
@printf("constrained %d members in %.1f min\n", n_obs, (time()-t0)/60)
bad = count(j -> dot(b_dprime, Θcon[:,j]) < bpp_floor - 1e-12 || abs(dot(b_prime, Θcon[:,j])) > 1e-6, 1:n_obs)
@printf("  members violating b′=0 or b″>0 after solve: %d\n\n", bad)
writedlm("$outdir/constrained_cloud_aeq.csv", Θcon', ',')
writedlm("$outdir/theta_mean_aeq.csv", θ_mean, ',')
flush(stdout)

# ── rejection sample ─────────────────────────────────────────────────────────
K_ref = dot(θ_mean, b_dprime)
hyp_e, hyp_b = hypercube(Matrix(Θcon' .- θ_mean'))
@printf("constrained box: %d directions, mean width %.4g\n",
        size(hyp_e,2), mean(hyp_b[2,:] .- hyp_b[1,:]))
n_try = Ref(0)
pred_aeq = θ -> begin
    n_try[] += 1
    all(ineq_lower .<= ineq_rows*θ) || return false
    abs(dot(b_prime, θ .- θ_mean)/K_ref) <= aeq_tol
end
rej, _ = rejection_sample_hypercube(hyp_e, hyp_b, θ_mean, pred_aeq;
                                    number_of_committee_members=n_samples,
                                    max_attempts=5_000_000)
@printf("  accepted %d of %d proposals (%.2f%%)\n\n", n_samples, n_try[], 100*n_samples/n_try[])
writedlm("$outdir/committee_aeq_rejection.csv", rej', ',')
Θ_aeq = rej'

# ── naive POPS baseline, straight from the module ────────────────────────────
cloud = corrections(Matrix(Ap), Vector(Yw), Matrix(P); leverage_percentile=lev_pct, lambda=λ)
nh_e, nh_b = hypercube(cloud)
nai, _ = rejection_sample_hypercube(nh_e, nh_b, lin, θ->true;
                                    number_of_committee_members=n_samples, max_attempts=1_000_000)
Θ_nai = nai'
@printf("naive POPS cloud %d × %d, box %d directions, mean width %.4g\n\n",
        size(cloud,1), size(cloud,2), size(nh_e,2), mean(nh_b[2,:] .- nh_b[1,:]))

# ── coverage ─────────────────────────────────────────────────────────────────
pt = A*lin
function coverage(Θ, label)
    pr = A*Θ'; lo = vec(minimum(pr;dims=2)); hi = vec(maximum(pr;dims=2))
    cov = (lo .< Y) .& (Y .< hi); w = hi .- lo; ms = .!cov
    rel = any(ms) ? median(max.(lo[ms].-Y[ms], Y[ms].-hi[ms]) ./ max.(w[ms],eps())) : 0.0
    @printf("  %-32s %7.2f%%  %10.4g %10.4g  %8d %8d  %9.3g\n",
            label, 100*count(cov)/M, median(w), mean(w), count(Y.<=lo), count(Y.>=hi), rel)
    return (; lo, hi, cov, w)
end
@printf("── training-set coverage (%d members each) ───────────────────────────────────\n", n_samples)
@printf("  %-32s %8s  %10s %10s  %8s %8s  %9s\n",
        "set","coverage","med width","mean width","below","above","med miss/w")
r_aeq = coverage(Θ_aeq, "a_eq-constrained + rejection")
r_nai = coverage(Θ_nai, "naive POPS full cloud")
phon_csv = "$(result.dir)/results/bandpath_undotted_ncell4_densek/committee_rejection.csv"
r_phon = isfile(phon_csv) ? coverage(readdlm(phon_csv, ','), "phonon-constrained (30 pts)") : nothing

@printf("\n  point-model RMSE %.4g | |Y-pt| median %.4g mean %.4g\n",
        sqrt(mean((Y.-pt).^2)), median(abs.(Y.-pt)), mean(abs.(Y.-pt)))
ratio = r_nai.w ./ max.(r_aeq.w, eps())
@printf("  naive width / a_eq-constrained width: median %.3g, p90 %.3g\n",
        median(ratio), quantile(ratio,0.9))
@printf("  a_eq covers, naive does not: %d | naive covers, a_eq does not: %d\n",
        count(r_aeq.cov .& .!r_nai.cov), count(r_nai.cov .& .!r_aeq.cov))
println("\nOutputs → $outdir/")
