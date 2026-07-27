# aeq_constrained_pops_coverage_Al_12_4_6A_2.jl
#
# QUESTION: can we skip the expensive per-member cutting-plane repair entirely?
#
# The multi-volume pipeline solves a large OSQP with ~800 phonon cut rows for EVERY
# observation — 8.6 s each, i.e. days over the full design matrix.  The cheap
# alternative tested here is:
#
#     1. constrain each POPS member ONLY on the lattice constant (one small QP:
#        interpolate observation i, b′(a_eq)·θ = 0, b″(a_eq)·θ > 0)
#     2. fit the hypercube to that cloud
#     3. let the REJECTION SAMPLER enforce phonon positivity, using the six
#        precomputed multi-volume band-path operators as the predicate
#
# If the accepted committee is comparable to the expensively-repaired one, the
# per-member cutting plane is unnecessary and the whole method becomes cheap.
#
# THREE COMMITTEES, compared on training-set coverage (same sampler, same size):
#
#   A  cheap a_eq QP over the masked cloud   + phonon rejection      ← the proposal
#   B  naive POPS hypercube, from scratch    + NO predicate          ← the control
#   C  30 highest-leverage members, FULL multi-volume cutting-plane
#      repair, + phonon rejection                                    ← the expensive
#                                                                      reference
#
# Phonon predicate (A and C): min non-acoustic ω ≥ cut_margin at every one of
# a ∈ a_eq·{1.00,1.02,1.04,1.06,1.08,1.10}, tested softest-volume-first.
# Plus b″·θ ≥ ε and the band |b′·(θ−θ_mean)|/(b″·θ_mean) ≤ 0.1 (rejection cannot
# impose the b′·θ = 0 equality — zero acceptance probability under a continuous
# proposal — so the equality lives only in the QP that builds the cloud).
#
# Run locally:  julia --project -t 8 scripts/uq/aeq_constrained_pops_coverage_Al_12_4_6A_2.jl
#   argv[1] = leverage_percentile for the cheap cloud (default 0.5; 0.0 = all rows)
#   argv[2] = committee size                          (default 30)

include(joinpath(@__DIR__, "..", "bandpath_phonon_uq", "lib.jl"))
using SparseArrays, OSQP, Random, Printf
Random.seed!(1234)
@async while true; flush(stdout); sleep(5); end

element, dataset = :Al, ""
lev_pct   = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 0.5
n_samples = length(ARGS) >= 2 ? parse(Int, ARGS[2])     : 30
n_expensive    = 30          # highest-leverage members for committee C
N_cell_fc      = 4
N_per_seg      = [20, 20, 20, 20, 60]
vol_scales     = collect(1.00:0.02:1.10)
cut_margin_THz = 0.15
max_cuts       = 40
aeq_tol        = 0.1
bpp_floor      = 1e-9
include_born   = false       # cheap stage stays lattice-constant-only by default
gc_every       = 5_000
max_attempts   = 20_000_000

result = load_model(element, 12, 4, 6, 2; dataset_name=dataset)
model, lin = result.model, result.lin_params
A, Y = result.A, result.Y
P  = result.P
Ap = Diagonal(result.W)*A/P; Yw = result.W .* Y
M, K = size(A); λ = 1.0/M
outdir = "$(result.dir)/results/aeq_cheap_vs_expensive"; mkpath(outdir)
@printf("model %s: %d rows × %d params, %d threads\nout → %s\n\n",
        result.name, M, K, Threads.nthreads(), outdir); flush(stdout)

# ── equation-of-state rows at the mean-model a_eq ────────────────────────────
a_eq = ACEWorkflow.relax_lattice_constant(model, element)
lattice_basis(a) = ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(
                        ACEWorkflow.Elasticity.reference_system(element; a=a), model))
b_prime  = ForwardDiff.derivative(lattice_basis, a_eq)
b_dprime = ForwardDiff.derivative(a -> ForwardDiff.derivative(lattice_basis, a), a_eq)
@printf("a_eq = %.5f Å\n", a_eq); flush(stdout)

ineq_rows, ineq_lower = b_dprime', [bpp_floor]
if include_born
    H_el = elastic_hessian_basis(model; element=element, a=a_eq)
    c11 = reshape(H_el,36,K)[1,:]; c12 = reshape(H_el,36,K)[7,:]; c44 = reshape(H_el,36,K)[22,:]
    ineq_rows  = vcat(c44', (c11.-c12)', (c11.+2 .*c12)', b_dprime')
    ineq_lower = [0.1, 1.0, 0.1, bpp_floor]
end

# ── six band paths (cached) — the phonon predicate and committee C need these ─
a_list = a_eq .* vol_scales; ω2_cut = (cut_margin_THz/FREQ_THz)^2
println("── band paths at $(length(a_list)) volumes ──"); flush(stdout)
bps_any = Vector{Any}(undef, length(a_list))
for (v,a) in enumerate(a_list)
    @printf("  [%d/%d] a = %.5f Å\n", v, length(a_list), a); flush(stdout)
    bps_any[v] = bandpath_Dk(result, model, element, a, N_cell_fc; N_per_seg=N_per_seg); GC.gc()
end
bps = convert(Vector{typeof(bps_any[1])}, bps_any); nvol = length(bps)
minω_vol(θ) = [min_freq_stable(θ, bp) for bp in bps]
minω_all(θ) = minimum(min_freq_stable(θ, bp) for bp in bps)
all_soft(θ) = [(v,iq,e) for v in 1:nvol for (iq,e) in soft_modes(θ, bps[v], ω2_cut)]

# ── QP machinery ─────────────────────────────────────────────────────────────
Hqp = sparse(Ap'*Ap .+ λ.*(P'*P)); qqp = -(Ap'*Yw)
const POOL = [OSQP.Model() for _ in 1:Threads.nthreads()]
solve_qp(osqp, Af, l, u) = begin
    OSQP.setup!(osqp; P=Hqp, q=qqp, A=Af, l=l, u=u, max_iter=4_000_000,
                check_termination=25, verbose=false, eps_abs=1e-6, eps_rel=1e-6)
    P \ OSQP.solve!(osqp).x
end
# cheap: interpolation + b′=0 + b″>0 (+Born if enabled).  ONE solve.
cheap_i(osqp, i) = solve_qp(osqp,
    vcat(sparse(Ap[i,:]'), sparse(b_prime'/P), sparse(ineq_rows/P)),
    vcat([Yw[i]], [0.0], ineq_lower),
    vcat([Yw[i]], [0.0], fill(Inf, length(ineq_lower))))
# expensive: the same plus accumulated multi-volume phonon cut rows
function expensive_i(osqp, i)
    acc = Vector{Vector{Float64}}(); lo = Float64[]; nc = 0
    mat() = isempty(acc) ? zeros(0,K) : permutedims(reduce(hcat, acc))
    build(er, el) = solve_qp(osqp,
        vcat(sparse(Ap[i,:]'), sparse(b_prime'/P), sparse(vcat(ineq_rows, er)/P)),
        vcat([Yw[i]], [0.0], ineq_lower, el),
        vcat([Yw[i]], [0.0], fill(Inf, length(ineq_lower)+length(el))))
    θ = build(zeros(0,K), Float64[]); conv = true
    for it in 0:max_cuts
        soft = all_soft(θ); isempty(soft) && break
        if it == max_cuts; conv = false; break; end
        for (v,iq,e) in soft; push!(acc, cut_row(iq,e,bps[v])); push!(lo, ω2_cut); end
        nc += length(soft); θ = build(mat(), lo)
    end
    (θ, nc, conv)
end
mean_fit() = solve_qp(POOL[1],
    vcat(sparse(b_prime'/P), sparse(ineq_rows/P)),
    vcat([0.0], ineq_lower), vcat([0.0], fill(Inf, length(ineq_lower))))
θ_mean = mean_fit()
@printf("constrained mean: b′·θ = %.2e, min ω over volumes = %+.3f THz\n\n",
        dot(b_prime,θ_mean), minω_all(θ_mean)); flush(stdout)

# ── leverage (same rule as POPSRegression.corrections) ───────────────────────
C = Symmetric(Ap'*Ap .+ λ.*(P'*P)); Cf = cholesky(C)
AtX = Cf\Matrix(Ap'); θ̃ = Cf\(Ap'*Yw)
leverage = vec(sum(Ap'.*AtX; dims=1)); residual = Yw .- Ap*θ̃
kept = findall(leverage .>= quantile(leverage, lev_pct)); n_obs = length(kept)
top30 = sortperm(leverage; rev=true)[1:n_expensive]
@printf("leverage_percentile %.2f → %d of %d rows | expensive set = top %d by leverage\n\n",
        lev_pct, n_obs, M, n_expensive); flush(stdout)

# ── committee A cloud: cheap a_eq QP over every kept observation ─────────────
println("── A: cheap a_eq-only QP over $n_obs observations ──"); flush(stdout)
Θcheap = zeros(K, n_obs); t0 = time()
for c0 in 1:gc_every:n_obs
    c1 = min(c0+gc_every-1, n_obs)
    Threads.@threads :static for j in c0:c1
        Θcheap[:, j] = cheap_i(POOL[Threads.threadid()], kept[j])
    end
    GC.gc(); el = time()-t0; f = c1/n_obs
    @printf("  %d/%d (%.1f%%) | %.1f min | est. total %.1f min | heap %.1f GB\n",
            c1, n_obs, 100f, el/60, el/60/f, Sys.maxrss()/2^30); flush(stdout)
end
@printf("  done in %.1f min\n", (time()-t0)/60)
@printf("  already phonon-stable at all volumes, before any rejection: %d / %d\n\n",
        count(j -> minω_all(Θcheap[:,j]) >= cut_margin_THz-1e-6, 1:n_obs), n_obs); flush(stdout)

# ── committee C cloud: full multi-volume repair, top-30 leverage ─────────────
println("── C: FULL multi-volume cutting-plane repair on the top $n_expensive leverage rows ──"); flush(stdout)
Θexp = zeros(K, n_expensive); ncuts = zeros(Int, n_expensive); convf = fill(false, n_expensive)
t1 = time()
Threads.@threads :static for j in 1:n_expensive
    θ, nc, cv = expensive_i(POOL[Threads.threadid()], top30[j])
    Θexp[:, j] = θ; ncuts[j] = nc; convf[j] = cv
end
@printf("  done in %.1f min | cuts median %d max %d | converged %d/%d | all-volume stable %d/%d\n\n",
        (time()-t1)/60, round(Int,median(ncuts)), maximum(ncuts), count(convf), n_expensive,
        count(j -> minω_all(Θexp[:,j]) >= cut_margin_THz-1e-6, 1:n_expensive), n_expensive); flush(stdout)

# ── the shared phonon predicate ──────────────────────────────────────────────
K_ref = dot(θ_mean, b_dprime)
order = sortperm(minω_vol(θ_mean))               # softest volume first → cheapest rejection
function make_pred(centre)
    n = Ref(0); nv = zeros(Int, nvol)
    f = θ -> begin
        n[] += 1
        n[] % 200_000 == 0 && (@printf("      … %d proposals, per-volume rejects %s\n", n[], string(nv)); flush(stdout))
        all(ineq_lower .<= ineq_rows*θ) || return false
        abs(dot(b_prime, θ .- centre)/K_ref) <= aeq_tol || return false
        for v in order
            min_freq_stable(θ, bps[v]) >= cut_margin_THz-1e-6 || (nv[v]+=1; return false)
        end
        true
    end
    (f, n, nv)
end
function draw(cloud_deltas, centre, label)
    e, b = hypercube(Matrix(cloud_deltas))
    @printf("  %s box: %d directions, mean width %.4g\n", label, size(e,2), mean(b[2,:].-b[1,:])); flush(stdout)
    pf, n, nv = make_pred(centre)
    m, _ = rejection_sample_hypercube(e, b, centre, pf;
                                      number_of_committee_members=n_samples, max_attempts=max_attempts)
    @printf("  %s: accepted %d of %d proposals (%.4f%%) | per-volume rejects %s\n\n",
            label, n_samples, n[], 100*n_samples/n[], string(nv)); flush(stdout)
    m'
end

println("── sampling ──"); flush(stdout)
Θ_A = draw(Θcheap' .- θ_mean', θ_mean, "A cheap+phonon-reject")
Θ_C = draw(Θexp'   .- θ_mean', θ_mean, "C expensive+phonon-reject")

# B: naive POPS, from scratch, no constraints anywhere
cloud = corrections(Matrix(Ap), Vector(Yw), Matrix(P); leverage_percentile=lev_pct, lambda=λ)
be, bb = hypercube(cloud)
@printf("  B naive box: %d directions, mean width %.4g (cloud %d × %d)\n",
        size(be,2), mean(bb[2,:].-bb[1,:]), size(cloud,1), size(cloud,2)); flush(stdout)
Θ_B = (rejection_sample_hypercube(be, bb, lin, θ->true;
        number_of_committee_members=n_samples, max_attempts=1_000_000)[1])'

for (nm, Θ) in (("A_cheap_phononreject",Θ_A), ("B_naive",Θ_B), ("C_expensive_phononreject",Θ_C))
    writedlm("$outdir/committee_$nm.csv", Θ, ',')
end

# ── coverage + phonon check ──────────────────────────────────────────────────
pt = A*lin
function report(Θ, label)
    pr = A*Θ'; lo = vec(minimum(pr;dims=2)); hi = vec(maximum(pr;dims=2))
    cov = (lo .< Y) .& (Y .< hi); w = hi .- lo; ms = .!cov
    rel = any(ms) ? median(max.(lo[ms].-Y[ms], Y[ms].-hi[ms]) ./ max.(w[ms],eps())) : 0.0
    worst = minimum(minimum(minω_vol(Θ[k,:])) for k in 1:size(Θ,1))
    @printf("  %-30s %7.2f%%  %10.4g %10.4g  %9.3g  %+9.3f\n",
            label, 100*count(cov)/M, median(w), mean(w), rel, worst)
    (; lo, hi, cov, w)
end
@printf("\n── training-set coverage (%d members each) ─────────────────────────────────────\n", n_samples)
@printf("  %-30s %8s  %10s %10s  %9s  %9s\n",
        "committee","coverage","med width","mean width","med miss/w","worst ω")
rA = report(Θ_A, "A cheap a_eq + phonon reject")
rB = report(Θ_B, "B naive POPS (no constraints)")
rC = report(Θ_C, "C expensive repair + reject")
@printf("\n  point-model RMSE %.4g | |Y-pt| median %.4g\n", sqrt(mean((Y.-pt).^2)), median(abs.(Y.-pt)))
@printf("  width ratio  B/A %.3g   C/A %.3g   (median over rows)\n",
        median(rB.w ./ max.(rA.w,eps())), median(rC.w ./ max.(rA.w,eps())))
@printf("  A covers, C does not: %d | C covers, A does not: %d\n",
        count(rA.cov .& .!rC.cov), count(rC.cov .& .!rA.cov))
println("\nOutputs → $outdir/")
