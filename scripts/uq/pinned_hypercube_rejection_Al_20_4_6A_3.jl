# pinned_hypercube_rejection_Al_20_4_6A_3.jl
#
# pinned_hypercube_Al_20_4_6A_3.jl with the sampler swapped for rejection: the box is
# unchanged, but a proposal is accepted only if the model it defines is dynamically
# stable.  Draws BOTH committees from the same box so the cost of the constraint is
# visible, not asserted.
#
# ── WHERE THIS SITS ──────────────────────────────────────────────────────────
# different_filter.jl (unpinned, 3×3×3): 5/50 soft.  Pinning to a_eq and moving to a
# clean 4×4×4 cell left 1/50.  That last member is genuinely unphysical — it is not
# residual stress (Δa = 0 by construction) and not a periodic-image artefact (half-box
# 8.04 Å vs a 6 Å cutoff).  It is the naive hypercube reaching into a region of
# parameter space where the potential has no stable FCC crystal, and it is exactly what
# a physics-informed prior is supposed to exclude.
#
# ── THE PREDICATE ────────────────────────────────────────────────────────────
#   1.  b″·θ > 0        — a_ref is a MINIMUM of E(a), not a maximum.  One dot product.
#   2.  min ω ≥ tol     — no soft mode anywhere on the band path away from Γ.
# Cheap test first: rejection_sample_hypercube calls the predicate once per proposal,
# and (2) costs ~140 matvecs while (1) costs one.
#
# Two caveats worth stating in the paper rather than hiding:
#   • The accepted ensemble is uniform on (box ∩ stable set), NOT on the box.  Rejection
#     imposes the feasible-set geometry exactly instead of approximating it by shrinking
#     the box, which is why `hypercube` is left UNCLIPPED (percentile_clipping = 0).
#   • Only 11% of the 145 band-path q-points are commensurate with a 4×4×4 cell, so MD
#     in that cell cannot falsify most of what this predicate enforces.  The constraint
#     is stronger than the test that would refute it.
#
# ── WHY THE SAMPLES INHERIT THE PIN FOR FREE ─────────────────────────────────
# The pin is exact on every row of the forest: PCθ·b′ = 0.  So b′ is an exact null
# eigenvector of PCθ′PCθ, and hypercube()'s own `eigvals .> max·1e-8` cut discards
# it — every retained direction is orthogonal to b′.  A sample is a box combination
# of retained directions, hence b′·δθ = 0 identically.  The pin needs NO rejection —
# it survives the sampler by construction, and the tell is the kept-direction count
# dropping by exactly one versus the unpinned forest.  Rejection below is only ever
# doing work on the soft modes, which is why the acceptance rate is a clean readout of
# how much of the naive box is dynamically unphysical.
#
# Run:  julia --project -t 40 scripts/uq/pinned_hypercube_rejection_Al_20_4_6A_3.jl [n_members]
#   Needs A.csv (~15-20 GB peak).  Reuses a cached band path if one exists, otherwise
#   builds it once (~15-20 min; set BUILD_THREADS).  Safe to `include` into a REPL that
#   already ran pinned_hypercube_Al_20_4_6A_3.jl — every expensive step is skipped if
#   already loaded, and it lands straight on the two draws.

using ACEWorkflow, ACEpotentials
import ACEpotentials.Models: evaluate_basis
import AtomsCalculatorsUtilities.SitePotentials: PairList, get_neighbours, cutoff_radius
using AtomsBuilder, Unitful, ForwardDiff, StaticArrays
using LinearAlgebra, Statistics, Random, Serialization, DelimitedFiles, Printf
using CairoMakie

Random.seed!(1234)

element           = :Al
filter_percentile = 0.5        # top 50% by the ratio metric feed the hypercube
unstable_tol      = -0.05      # THz — what the DIAGNOSTIC calls unstable
qΓtol             = 5e-2
n_members         = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 50

# Acceptance threshold for the predicate.  Defaults to `unstable_tol` so "accepted"
# means exactly "not flagged by the diagnostic" — no gap between the two, and no
# member is rejected for being negative at the level of numerical noise.  Set
# ACCEPT_TOL=0.0 to demand strictly non-negative frequencies instead.
accept_tol  = parse(Float64, get(ENV, "ACCEPT_TOL", string(unstable_tol)))
max_attempts = parse(Int, get(ENV, "MAX_ATTEMPTS", string(2000 * n_members)))

MODELDIR = "models/Al_20_4_6A_3_"
RES      = "$MODELDIR/results"
outdir   = "$RES/pinned_hypercube_rejection"; mkpath(outdir)

# band-path caches, in order of preference: the forest job's, the pinned forest's,
# the naive hypercube run's, our own
BPCACHES = ["$RES/forest_phonon_stability/bandpath_4x4x4.jls",
            "$RES/pinned_forest/bandpath_4x4x4_aref.jls",
            "$RES/pinned_hypercube/bandpath_4x4x4_aref.jls",
            "$outdir/bandpath_4x4x4_aref.jls"]
BPCACHE_OWN = "$outdir/bandpath_4x4x4_aref.jls"
N_cell      = 4
N_per_seg   = [20, 20, 20, 20, 60]
const BUILD_THREADS = parse(Int, get(ENV, "BUILD_THREADS", "8"))

# ═════════════════════════════════════════════════════════════════════════════
#  3-row undotted band-path builder (only Np = 1 rows of H are ever read).
#  Verified against a full 4×4×4 cache at machine precision: rows 1.5e-16,
#  D_k(q) 3.0e-16, end-to-end min ω 1.6e-14 THz.  Duplicated from
#  pinned_forest_Al_20_4_6A_3.jl rather than `include`d so this script has no path
#  dependency — keep the copies in step if any is edited.
# ═════════════════════════════════════════════════════════════════════════════
function _site_basis_hessian(m, Rs, Zs, z0, ps, st)
    nR = length(Rs)
    x0 = collect(Float64, reinterpret(Float64, Rs))
    tovec(x) = [SVector{3,eltype(x)}(x[3i-2], x[3i-1], x[3i]) for i in 1:nR]
    Bfun(x)  = evaluate_basis(m, tovec(x), Zs, z0, ps, st)
    Hflat    = ForwardDiff.jacobian(x -> vec(ForwardDiff.jacobian(Bfun, x)), x0)
    return reshape(Hflat, length(Bfun(x0)), 3nR, 3nR)
end

function undotted_rows(sys_super, V, r)
    nlist = PairList(sys_super, cutoff_radius(V))
    Nat   = length(sys_super); D = 3
    ps, st, m = V.ps, V.st, V.model
    _, Rs0, Zs0, z00 = get_neighbours(sys_super, V, nlist, 1)
    NB = length(evaluate_basis(m, Rs0, Zs0, z00, ps, st))
    Js_r, = get_neighbours(sys_super, V, nlist, r)
    contributors = unique(vcat(r, Js_r))
    nt = clamp(BUILD_THREADS, 1, min(Threads.nthreads(), length(contributors)))
    @printf("    %d of %d atoms contribute; %d build threads, %d deep\n",
            length(contributors), Nat, nt, cld(length(contributors), nt))
    @printf("    site Hessian %.2f GB/thread → ~%.0f-%.0f GB peak\n",
            NB*(3length(Js_r))^2*8/1e9, nt*NB*(3length(Js_r))^2*8/1e9,
            2*nt*NB*(3length(Js_r))^2*8/1e9); flush(stdout)
    bufs = [zeros(D, D*Nat, NB) for _ in 1:nt]
    @sync for t in 1:nt
        Threads.@spawn begin
            Hl = bufs[t]
            for i in contributors[t:nt:end]
                Js, Rs, Zs, z0 = get_neighbours(sys_super, V, nlist, i)
                Hi = _site_basis_hessian(m, Rs, Zs, z0, ps, st)
                Ji = (i-1)*D .+ (1:D); is_ref = (i == r)
                for (α1, j1) in enumerate(Js)
                    hit1 = (j1 == r)
                    (hit1 || is_ref) || continue
                    A1 = (α1-1)*D .+ (1:D)
                    for (α2, j2) in enumerate(Js)
                        A2 = (α2-1)*D .+ (1:D); J2 = (j2-1)*D .+ (1:D)
                        @views for kb in 1:NB
                            blk = Hi[kb, A1, A2]
                            if hit1
                                Hl[:, J2, kb] .+= blk; Hl[:, Ji, kb] .-= blk
                            end
                            if is_ref
                                Hl[:, J2, kb] .-= blk; Hl[:, Ji, kb] .+= blk
                            end
                        end
                    end
                end
                Hi = nothing
            end
        end
    end
    H = bufs[1]
    for t in 2:nt; H .+= bufs[t]; bufs[t] = zeros(0,0,0); end
    return H
end

function bandpath_rows(model, element, a, N_cell; N_per_seg=N_per_seg)
    sys_prim, sys_super = bulk_prim_super(element; a=a, N_cell=N_cell)
    fc = precompute_force_constants(sys_prim, sys_super, model)
    fc.Np == 1 || error("this builder assumes a 1-atom primitive cell; got Np = $(fc.Np)")
    r = fc.p2s_map[1] + 1
    @printf("  building 3-row undotted Hessian at a = %.5f Å (%d atoms) …\n", a, length(sys_super))
    t = @elapsed Hrows = undotted_rows(sys_super, model, r)
    @printf("    done in %.1f min\n", t/60); flush(stdout)
    q_list, x_vals, x_ticks, labels, _ = fcc_band_path(fc.L; N_per_seg=N_per_seg)
    NB = size(Hrows, 3)
    Rmi = Vector{SVector{3,Float64}}(undef, fc.Ns)
    for k in 1:fc.Ns
        R_cart  = fc.L * (fc.frac_super[k] - fc.frac_prim[1])
        R_sfrac = fc.Linv_super * R_cart; R_sfrac = R_sfrac .- round.(R_sfrac)
        Rmi[k]  = fc.L_super * R_sfrac
    end
    mass = fc.masses[1]
    Bq = Vector{Matrix{ComplexF64}}(undef, length(q_list))
    Threads.@threads for iq in eachindex(q_list)
        q = q_list[iq]; M = zeros(ComplexF64, 9, NB)
        for k in 1:fc.Ns
            eph = exp(im * dot(q, Rmi[k])); cols = 3(k-1) .+ (1:3)
            @views for kb in 1:NB
                blk = Hrows[:, cols, kb]
                for β in 1:3, α in 1:3
                    M[3(β-1)+α, kb] += blk[α, β] * eph
                end
            end
        end
        M ./= mass
        for kb in 1:NB
            D3 = reshape(@view(M[:, kb]), 3, 3); H3 = (D3 .+ D3') ./ 2
            M[:, kb] = vec(H3)
        end
        Bq[iq] = M
    end
    return (; Bq, q_list, x_vals, x_ticks, labels, Np=fc.Np, qnorm=norm.(q_list), a_ref=a)
end

# ── phonon helpers ───────────────────────────────────────────────────────────
# ω = sign(ω²)·√|ω²|, so dynamical instabilities read as negative frequencies.
function bands(θ, bp)
    F = Matrix{Float64}(undef, 3bp.Np, length(bp.Bq))
    for iq in eachindex(bp.Bq)
        ev = eigvals(Hermitian(reshape(bp.Bq[iq] * θ, 3bp.Np, 3bp.Np)))
        F[:, iq] = sign.(ev) .* sqrt.(abs.(ev)) .* FREQ_THz
    end
    return F
end

# minimum frequency away from Γ, and the q index carrying it.  A function, not a
# top-level loop: at top level a `for` opens a soft scope, so assigning a name that
# also exists as a global makes a fresh local and reading it after throws.
function min_freq_stable(θ, bp, keep)
    mw = Inf; im_ = 0
    for iq in keep
        ev = eigvals(Hermitian(reshape(bp.Bq[iq] * θ, 3bp.Np, 3bp.Np)))
        w  = minimum(sign.(ev) .* sqrt.(abs.(ev)) .* FREQ_THz)
        w < mw && (mw = w; im_ = iq)
    end
    return mw, im_
end

# ═════════════════════════════════════════════════════════════════════════════
#  Model.  Skipped if a previous script already loaded it into this REPL.
# ═════════════════════════════════════════════════════════════════════════════
if !@isdefined(result) || result === nothing || !hasproperty(result, :A)
    result = load_model(element, 20, 4, 6, 3; dataset_name="")
end
model      = result.model
lin_params = result.lin_params
n_params   = length(lin_params)
P, W, Y    = result.P, result.W, result.Y
if !@isdefined(Ap) || size(Ap) != size(result.A)
    Ap = Diagonal(W) * result.A / P
end
Yw = W .* Y
N  = size(Ap, 1)
@printf("Model %s: %d params, %d observations, %d threads\n",
        result.name, n_params, N, Threads.nthreads()); flush(stdout)

# ── band path (cached if possible; only ONE geometry is ever needed) ─────────
if !@isdefined(bp) || bp === nothing || !hasproperty(bp, :a_ref)
    hit = findfirst(isfile, BPCACHES)
    global bp = if hit !== nothing
        println("band path: reusing $(BPCACHES[hit])"); deserialize(BPCACHES[hit])
    else
        println("band path: no cache found — building one (this is the only build needed)")
        a0 = ACEWorkflow.relax_lattice_constant(model, element)
        @printf("  a_ref = %.17g Å; box %.2f Å, half-box %.2f Å vs 6 Å cutoff\n",
                a0, N_cell*a0, N_cell*a0/2); flush(stdout)
        b = bandpath_rows(model, element, a0, N_cell)
        serialize(BPCACHE_OWN, b); println("  cached → $BPCACHE_OWN")
        b
    end
else
    println("band path: already in the session")
end
a_ref = bp.a_ref
keep  = findall(bp.qnorm .>= qΓtol)
@printf("band path: %d q-points (%d after the near-Γ cut), %d branches, a_ref = %.17g Å\n",
        length(bp.Bq), length(keep), 3bp.Np, a_ref); flush(stdout)

# ── the pin direction and its second-order partner ──────────────────────────
lattice_basis(a) = ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(
                       ACEWorkflow.Elasticity.reference_system(element; a=a), model))
b_prime        = ForwardDiff.derivative(lattice_basis, a_ref)
b_double_prime = ForwardDiff.derivative(a -> ForwardDiff.derivative(lattice_basis, a), a_ref)
@printf("b′·lin_params = %.3e (~0 ⇒ a_ref is the mean model's relaxed lattice constant),\n",
        dot(b_prime, lin_params))
@printf("b″·lin_params = %+.6e (> 0 ⇒ minimum, not maximum)\n", dot(b_double_prime, lin_params))

# ═════════════════════════════════════════════════════════════════════════════
#  Pinned delta forest.  Closed form: one equality on a quadratic objective is a
#  rank-one correction to the inverse, so this costs one extra solve, not N QPs.
#  Verified against explicitly solved KKT systems to 3.9e-16.
# ═════════════════════════════════════════════════════════════════════════════
g = transpose(P) \ b_prime          # b′·(P⁻¹δ) = (P⁻ᵀb′)·δ = g·δ

need_forest = !@isdefined(PCx) || !@isdefined(lev_c) || !@isdefined(C) ||
              size(PCx) != (N, n_params) || maximum(abs.(PCx * g)) > 1e-6 * norm(g)
if need_forest
    @isdefined(PCx) && size(PCx) == (N, n_params) &&
        @warn "a PCx of the right shape was in the session but is NOT pinned — rebuilding"
    λ  = 1.0 / N
    C  = Symmetric(P'*P .* λ .+ Ap'*Ap)
    Cf = cholesky(C)
    @printf("C: %d×%d, cond ≈ %.3e\n", n_params, n_params, cond(Matrix(C))); flush(stdout)

    Ainv = Cf \ Matrix(Ap')                       # p × N
    x̃    = Cf \ (Ap' * Yw)
    lev  = vec(sum(Ap' .* Ainv; dims=1))

    u  = Cf \ g
    gu = dot(g, u)
    v  = Ap * u
    abs(gu) > 1e-30 || error("g′u ≈ 0 — pin direction lies in the null space of C⁻¹")

    x̃_c   = x̃ .- u .* (dot(g, x̃)/gu)              # constrained mean fit
    lev_c = lev .- (v .^ 2) ./ gu                 # constrained leverage
    r_c   = Yw .- Ap * x̃_c
    @printf("constrained leverage: min %.4e (unconstrained %.4e), %d non-positive\n",
            minimum(lev_c), minimum(lev), count(<=(0), lev_c))

    scale = r_c ./ lev_c
    global PCx = transpose(Ainv .* transpose(scale))
    PCx .-= (scale .* v ./ gu) * transpose(u)
    Ainv = nothing; GC.gc()
else
    println("pinned forest: PCx, C and lev_c already in the session, pin verified — reusing")
end
@printf("\nPIN CHECK (forest): max |b′·δ| = %.3e over all %d corrections  (‖g‖ = %.3e)\n",
        maximum(abs.(PCx * g)), N, norm(g)); flush(stdout)

# ── ratio filter: the same metric as different_filter.jl, on the PINNED forest ──
# identity form — the naive PC*C*PC′ is 147k² = 173 GB and is never formed.
# NOTE the mask will not equal different_filter.jl's: the metric is computed on the
# pinned corrections and the constrained leverage, so a different half is selected.
S_pc = transpose(PCx) * PCx
G_pc = C * S_pc * C
filter_metric = sqrt.(vec(sum((PCx * G_pc) .* PCx; dims=2))) ./ lev_c
mask = filter_metric .>= quantile(filter_metric, filter_percentile)
S_pc = nothing; G_pc = nothing; GC.gc()
@printf("ratio filter: %d of %d corrections in the top %.0f%% → these feed the hypercube\n",
        count(mask), N, 100*(1-filter_percentile)); flush(stdout)

# ═════════════════════════════════════════════════════════════════════════════
#  θ-space corrections → hypercube → samples.
#  different_filter.jl builds the box in θ-space (Gamma \ … before hypercube), so
#  the P⁻¹ is applied here too rather than being folded into the band path — same
#  box, same sampler, only the corrections are pinned.
# ═════════════════════════════════════════════════════════════════════════════
# x = Pθ, so δθ = P⁻¹δx; with corrections stored one per ROW that is δx′P⁻ᵀ, i.e.
# right-division by P′.  Identical to different_filter.jl's `Gamma \ pc'` then transpose.
pwise = PCx[mask, :]                              # n_masked × p, x-space
@printf("θ-space conversion: %d × %d (%.2f GB)\n", size(pwise)..., prod(size(pwise))*8/1e9)
flush(stdout)
pwise = pwise / transpose(P)                      # → θ-space
GC.gc()

@printf("PIN CHECK (θ-space): max |b′·δθ| = %.3e over %d filtered corrections\n",
        maximum(abs.(pwise * b_prime)), size(pwise, 1)); flush(stdout)

eig_pops, bound_pops = hypercube(pwise)
n_dir = size(eig_pops, 2)
@printf("\nhypercube: %d / %d directions kept (unpinned would keep %d — the pin removes\n",
        n_dir, n_params, n_params)
@printf("           exactly one, b′ itself, as an exact null eigenvector of PCθ′PCθ)\n")
@printf("           max |b′·eigvec| over kept directions = %.3e  ← 0 ⇒ every draw is pinned\n",
        maximum(abs.(transpose(eig_pops) * b_prime))); flush(stdout)

# ═════════════════════════════════════════════════════════════════════════════
#  Two draws from the SAME box: naive, then rejection-constrained.
# ═════════════════════════════════════════════════════════════════════════════
# Both are seeded identically, so the naive committee reproduces
# pinned_hypercube_Al_20_4_6A_3.jl exactly and the comparison below is against the
# run you already have, not a fresh sample with its own noise.

# early-exit stability test.  Cheap check first: b″·θ > 0 is one dot product and
# disqualifies a member outright (a_ref would be a MAXIMUM of E(a), so the member has
# no FCC volume minimum and its band structure describes a state it cannot hold).
n_rej_K = Ref(0); n_rej_ω = Ref(0); n_prop = Ref(0)
function is_physical(θ)
    n_prop[] += 1
    if dot(b_double_prime, θ) <= 0
        n_rej_K[] += 1; return false
    end
    for iq in keep
        ev = eigvals(Hermitian(reshape(bp.Bq[iq] * θ, 3bp.Np, 3bp.Np)))
        if minimum(sign.(ev) .* sqrt.(abs.(ev)) .* FREQ_THz) < accept_tol
            n_rej_ω[] += 1; return false      # early exit: no need to finish the path
        end
    end
    return true
end

# per-committee diagnostics, as a function — a top-level `for` opens a soft scope and
# reading an accumulator after it throws if the name also exists as a global
function evaluate_committee(mem, bp, keep)
    n = length(mem)
    mf = Vector{Float64}(undef, n); iqm = Vector{Int}(undef, n)
    Threads.@threads for i in 1:n
        mw, im_ = min_freq_stable(mem[i], bp, keep)
        mf[i] = mw; iqm[i] = im_
    end
    pin = [dot(b_prime, θ) for θ in mem]
    K   = [dot(b_double_prime, θ) for θ in mem]
    return mf, iqm, pin, K, -pin ./ K
end

println("\n── draw 1 of 2: NAIVE (no rejection) ──")
Random.seed!(1234)
samples_n, δθ_n = sample_hypercube(eig_pops, bound_pops, lin_params;
                                   number_of_committee_members=n_members)
mem_n = [samples_n[:, i] for i in 1:size(samples_n, 2)]

println("\n── draw 2 of 2: REJECTION (b″·θ > 0 and min ω ≥ $accept_tol THz) ──")
flush(stdout)
Random.seed!(1234)
t_rej = @elapsed samples_r, δθ_r =
    rejection_sample_hypercube(eig_pops, bound_pops, lin_params, is_physical;
                               number_of_committee_members=n_members,
                               max_attempts=max_attempts)
mem_r = [samples_r[:, i] for i in 1:size(samples_r, 2)]
@printf("  %d proposals for %d members in %.2f s (%.1f ms/proposal)\n",
        n_prop[], n_members, t_rej, 1000t_rej/max(n_prop[],1))
@printf("  acceptance %.2f%%  →  %.2f%% of the naive box is dynamically unphysical\n",
        100n_members/n_prop[], 100*(1 - n_members/n_prop[]))
@printf("  rejected by b″·θ ≤ 0 : %d\n", n_rej_K[])
@printf("  rejected by soft mode: %d\n", n_rej_ω[])
flush(stdout)

# ═════════════════════════════════════════════════════════════════════════════
#  Compare
# ═════════════════════════════════════════════════════════════════════════════
minf_n, iq_n, pin_n, K_n, Δa_n = evaluate_committee(mem_n, bp, keep)
minf_r, iq_r, pin_r, K_r, Δa_r = evaluate_committee(mem_r, bp, keep)
minf_mean, _ = min_freq_stable(lin_params, bp, keep)
unst_n = minf_n .< unstable_tol
unst_r = minf_r .< unstable_tol

@printf("\nmean model : min ω = %+.4f THz\n", minf_mean)
@printf("NAIVE      : min ω ∈ [%+.4f, %+.4f], median %+.4f — %d/%d unstable\n",
        minimum(minf_n), maximum(minf_n), median(minf_n), count(unst_n), n_members)
@printf("REJECTION  : min ω ∈ [%+.4f, %+.4f], median %+.4f — %d/%d unstable %s\n",
        minimum(minf_r), maximum(minf_r), median(minf_r), count(unst_r), n_members,
        count(unst_r) == 0 ? "✓" : "← PREDICATE LEAKED, investigate")
count(unst_r) == 0 ||
    @warn "a member passed the predicate but is flagged by the diagnostic — accept_tol ($accept_tol) is looser than unstable_tol ($unstable_tol)?"

# the pin is untouched by rejection: it was never the thing being rejected
@printf("\npin, both committees: max |b′·θ| = %.3e (naive) / %.3e (rejected)\n",
        maximum(abs.(pin_n)), maximum(abs.(pin_r)))
@printf("Δa, both committees : max |Δa| = %.3e / %.3e Å  (unpinned spanned ±1.5e-2 Å)\n",
        maximum(abs.(Δa_n)), maximum(abs.(Δa_r)))

# ── what did the constraint cost? ───────────────────────────────────────────
# Rejection makes the ensemble uniform on (box ∩ stable set) rather than on the box.
# tr(δθ) is the total parameter variance the committee carries; the per-parameter ratio
# says whether the change is uniform or concentrated in a few directions.
#
# NOTE: a ratio above 1 in some coordinates is NOT a bug.  Rejection removes a region,
# not an outer shell, so a marginal variance can rise if the removed part sat near the
# centre of that coordinate — and with only n_members draws the sd estimate is noisy at
# the O(1/√2n) level anyway (≈10% at 50).  Read the trace, treat the per-parameter
# spread as indicative, and don't quote either as a converged number.
sd_n = sqrt.(diag(δθ_n)); sd_r = sqrt.(diag(δθ_r))
@printf("\ncommittee spread: tr(δθ) %.6e → %.6e  (%.2f%% of the naive variance retained)\n",
        tr(δθ_n), tr(δθ_r), 100*tr(δθ_r)/tr(δθ_n))
@printf("  per-parameter sd ratio: median %.4f, min %.4f, max %.4f over %d params\n",
        median(sd_r ./ sd_n), minimum(sd_r ./ sd_n), maximum(sd_r ./ sd_n), n_params)
@printf("  (sampling noise on each sd at n = %d is ≈%.0f%%, so only the trace is worth quoting)\n",
        n_members, 100/sqrt(2n_members))
flush(stdout)

if count(unst_n) > 0
    seglab = ["Γ→X", "X→U", "U→L", "L→Γ", "Γ→K"]
    segof(iq) = clamp(searchsortedlast(bp.x_ticks, bp.x_vals[iq]), 1, length(seglab))
    println("\nwhere the naive committee went soft (what rejection removed):")
    for i in findall(unst_n)
        @printf("  member %2d: min ω = %+.4f THz on %s at q = %s\n",
                i, minf_n[i], seglab[segof(iq_n[i])],
                string(round.(bp.q_list[iq_n[i]]; digits=4)))
    end
end
flush(stdout)

# ── figure: naive vs rejected, same axes ────────────────────────────────────
BLU = RGBf(0.0,0.447,0.698); GRY = RGBAf(0.45,0.45,0.45,0.30); RED = RGBAf(0.80,0.15,0.15,0.45)
GRN = RGBAf(0.0,0.55,0.35,0.30)
TITLE, LAB, TICK = 13, 12, 11

F_n = [bands(θ, bp) for θ in mem_n]
F_r = [bands(θ, bp) for θ in mem_r]
F_mean = bands(lin_params, bp)
lo = min(minimum(minimum.(F_n)), minimum(minimum.(F_r)), minimum(F_mean))
hi = max(maximum(maximum.(F_n)), maximum(maximum.(F_r)), maximum(F_mean))
pad = 0.06*(hi-lo)

fig = Figure(size=(540, 360), figure_padding=(6,10,4,6))
function panel(gp, F_all, unst, ttl, basecol)
    ax = Axis(gp; xlabel="Wave vector", ylabel="Frequency (THz)", title=ttl,
              titlesize=TITLE, xlabelsize=LAB, ylabelsize=LAB,
              xticklabelsize=TICK, yticklabelsize=TICK,
              xticks=(bp.x_ticks, bp.labels), xgridvisible=false, ygridvisible=false,
              xtickalign=1, ytickalign=1, xticksize=4, yticksize=4)
    for (i, Fθ) in enumerate(F_all)
        col = unst[i] ? RED : basecol
        for b in 1:3bp.Np; lines!(ax, bp.x_vals, Fθ[b,:]; color=col, linewidth=0.8); end
    end
    for b in 1:3bp.Np; lines!(ax, bp.x_vals, F_mean[b,:]; color=BLU, linewidth=1.8); end
    hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=1.0)
    vlines!(ax, bp.x_ticks; color=(:black,0.22), linewidth=0.7)
    xlims!(ax, first(bp.x_vals), last(bp.x_vals)); ylims!(ax, min(lo-pad,-0.4), hi+pad)
    text!(ax, 0.97, 0.97; text="$(count(unst))/$n_members unstable", space=:relative,
          align=(:right,:top), fontsize=TICK-1, color=:gray30)
    return ax
end
panel(fig[1,1], F_n, unst_n, "naive hypercube", GRY)
ax2 = panel(fig[1,2], F_r, unst_r, "rejection: b″·θ>0, min ω ≥ $(accept_tol) THz", GRN)
ax2.ylabel = ""
save("$outdir/bands_pinned_rejection.pdf", fig)
save("$outdir/bands_pinned_rejection.png", fig; px_per_unit=4)

fig2 = Figure(size=(300, 240), figure_padding=(6,10,4,6))
axh = Axis(fig2[1,1]; xlabel="min ω (THz)", ylabel="count", title="stability margin",
           titlesize=TITLE, xlabelsize=LAB, ylabelsize=LAB,
           xticklabelsize=TICK, yticklabelsize=TICK,
           xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1)
hist!(axh, minf_n; bins=30, color=(RED,0.55), strokewidth=0.5, strokecolor=:white,
      label="naive")
hist!(axh, minf_r; bins=30, color=(GRN,0.65), strokewidth=0.5, strokecolor=:white,
      label="rejected")
vlines!(axh, [accept_tol]; color=:black, linestyle=:dash, linewidth=1.0)
vlines!(axh, [minf_mean]; color=BLU, linewidth=1.8)
axislegend(axh; position=:lt, labelsize=TICK-1, framevisible=false)
save("$outdir/margin_pinned_rejection.pdf", fig2)
save("$outdir/margin_pinned_rejection.png", fig2; px_per_unit=4)

writedlm("$outdir/samples_rejected.csv", samples_r', ',')
writedlm("$outdir/samples_naive.csv", samples_n', ',')
writedlm("$outdir/min_freq_rejected.csv", hcat(minf_r, iq_r, pin_r, K_r, Δa_r), ',')
writedlm("$outdir/min_freq_naive.csv",    hcat(minf_n, iq_n, pin_n, K_n, Δa_n), ',')
serialize("$outdir/pinned_hypercube_rejection.jls",
          (; samples_r, samples_n, δθ_r, δθ_n, minf_r, minf_n, iq_r, iq_n,
             pin_r, pin_n, K_r, K_n, Δa_r, Δa_n, a_ref, accept_tol, unstable_tol,
             n_proposals=n_prop[], n_rej_K=n_rej_K[], n_rej_ω=n_rej_ω[],
             n_dir, n_masked=count(mask), seed=1234))
println("\nfigures → $outdir/{bands,margin}_pinned_rejection.{pdf,png}")
println("data    → $outdir/samples_{rejected,naive}.csv, min_freq_{rejected,naive}.csv")
println("          columns: min ω, argmin q index, b′·θ, b″·θ, Δa")
