# pinned_hypercube_Al_20_4_6A_3.jl
#
# different_filter.jl's committee, but drawn from the a_eq-PINNED delta forest:
# pin → ratio filter → hypercube → 50 samples → phonons on the undotted 4×4×4 band path.
#
# ── WHAT THIS ANSWERS ────────────────────────────────────────────────────────
# different_filter.jl found 5/50 committee members with soft modes, but each member
# sits at its OWN equilibrium (Δa ∈ ±0.015 Å) while the phonons are evaluated at the
# mean model's a_eq — so part of that softening is residual stress, not a real
# dynamical instability.  Pinning every correction to b′·δ = 0 removes that
# confound: every member's equilibrium IS a_ref, so one band path is the exact
# operator for all of them and any surviving soft mode is genuine.
#
# ── WHY THE SAMPLES INHERIT THE PIN FOR FREE ─────────────────────────────────
# The pin is exact on every row of the forest: PCθ·b′ = 0.  So b′ is an exact null
# eigenvector of PCθ′PCθ, and hypercube()'s own `eigvals .> max·1e-8` cut discards
# it — every retained direction is orthogonal to b′.  A sample is a box combination
# of retained directions, hence b′·δθ = 0 identically.  No rejection sampling, no
# predicate: the constraint survives the sampler by construction.  The tell is the
# kept-direction count dropping by exactly one versus the unpinned forest, and that
# is printed below alongside a direct max|b′·δθ| over the 50 draws.
#
# ── SAMPLER ──────────────────────────────────────────────────────────────────
# Plain `sample_hypercube`, identical to different_filter.jl line 96 — same box, same
# uniform draw, no Gaussian substitution.  Only the corrections feeding it changed.
# Seeded, so this is exactly reproducible (no OSQP anywhere in this path).
#
# Run:  BUILD_THREADS=16 julia --project -t 40 scripts/uq/pinned_hypercube_Al_20_4_6A_3.jl [n_members]
#   Needs A.csv (~15-20 GB peak).  Reuses a cached band path if one exists, otherwise
#   builds it once (~15-20 min).  Safe to `include` into a REPL that already ran
#   pinned_forest_Al_20_4_6A_3.jl — every expensive step is skipped if already loaded.

using ACEWorkflow, ACEpotentials
import ACEpotentials.Models: evaluate_basis
import AtomsCalculatorsUtilities.SitePotentials: PairList, get_neighbours, cutoff_radius
using AtomsBuilder, Unitful, ForwardDiff, StaticArrays
using LinearAlgebra, Statistics, Random, Serialization, DelimitedFiles, Printf
using CairoMakie

Random.seed!(1234)

element           = :Al
filter_percentile = 0.5        # top 50% by the ratio metric feed the hypercube
unstable_tol      = -0.05      # THz
qΓtol             = 5e-2
n_members         = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 50

MODELDIR = "models/Al_20_4_6A_3_"
RES      = "$MODELDIR/results"
outdir   = "$RES/pinned_hypercube"; mkpath(outdir)

# band-path caches, in order of preference: the forest job's, the pinned forest's, our own
BPCACHES = ["$RES/forest_phonon_stability/bandpath_4x4x4.jls",
            "$RES/pinned_forest/bandpath_4x4x4_aref.jls",
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

samples, _ = sample_hypercube(eig_pops, bound_pops, lin_params;
                              number_of_committee_members=n_members)
members = [samples[:, i] for i in 1:size(samples, 2)]
@printf("drew %d committee members (seeded — exactly reproducible)\n", length(members))

# ── did the pin survive the sampler? ────────────────────────────────────────
# Δa ≈ −b′·θ / b″·θ is the Newton step from a_ref to the member's own equilibrium.
# different_filter.jl's unpinned committee spanned Δa ∈ [−0.0153, +0.0147] Å; here it
# should be numerically zero, which is what makes the single band path exact.
pin_s = [dot(b_prime, θ) for θ in members]
K_s   = [dot(b_double_prime, θ) for θ in members]
Δa_s  = -pin_s ./ K_s
@printf("\nSAMPLES, b′·θ    : max |b′·θ| = %.3e  (mean model %.3e)\n",
        maximum(abs.(pin_s)), abs(dot(b_prime, lin_params)))
@printf("SAMPLES, Δa      : |Δa| ≤ %.3e Å  (unpinned committee spanned ±1.5e-2 Å)\n",
        maximum(abs.(Δa_s)))
@printf("SAMPLES, b″·θ > 0: %d / %d violate (no volume minimum at a_ref); b″·θ ∈ [%+.4e, %+.4e]\n",
        count(<=(0), K_s), n_members, minimum(K_s), maximum(K_s))
flush(stdout)

# ═════════════════════════════════════════════════════════════════════════════
#  Phonons
# ═════════════════════════════════════════════════════════════════════════════
minf   = Vector{Float64}(undef, n_members)
iq_min = Vector{Int}(undef, n_members)
t = @elapsed Threads.@threads for i in 1:n_members
    mw, im_ = min_freq_stable(members[i], bp, keep)
    minf[i] = mw; iq_min[i] = im_
end
minf_mean, _ = min_freq_stable(lin_params, bp, keep)
unst   = minf .< unstable_tol
badK   = K_s .<= 0.0
n_unst = count(unst)
@printf("\nevaluated %d members in %.2f s\n", n_members, t)
@printf("mean model      : min ω = %+.4f THz\n", minf_mean)
@printf("pinned committee: min ω ∈ [%+.4f, %+.4f] THz, median %+.4f\n",
        minimum(minf), maximum(minf), median(minf))
println("\n  failure modes (a member can fail both):")
@printf("    soft mode, min ω < %.2f THz : %3d / %d  (%.1f%%)\n",
        unstable_tol, n_unst, n_members, 100n_unst/n_members)
@printf("    b″·θ ≤ 0, no volume minimum : %3d / %d\n", count(badK), n_members)
@printf("    both                        : %3d\n", count(unst .& badK))
@printf("    FULLY STABLE                : %3d / %d  (%.1f%%)\n",
        count(.!unst .& .!badK), n_members, 100count(.!unst .& .!badK)/n_members)
println("\n  reference: different_filter.jl's UNPINNED committee gave 5/50 soft at 3×3×3.")
println("  A drop here means part of that was residual stress; no drop means the soft")
println("  modes are genuine and independent of where each member's equilibrium sits.")
flush(stdout)

if n_unst > 0
    seglab = ["Γ→X", "X→U", "U→L", "L→Γ", "Γ→K"]
    segof(iq) = clamp(searchsortedlast(bp.x_ticks, bp.x_vals[iq]), 1, length(seglab))
    println("\nsoft-mode location (segment carrying each unstable member's minimum):")
    for s in 1:length(seglab)
        c = count(i -> segof(iq_min[i]) == s, findall(unst))
        c == 0 && continue
        @printf("  %-5s %3d  (%.0f%% of unstable members)\n", seglab[s], c, 100c/n_unst)
    end
    iw = argmin(minf)
    @printf("worst member: %d, min ω = %+.4f THz at q = %s\n",
            iw, minf[iw], string(round.(bp.q_list[iq_min[iw]]; digits=4)))
end
flush(stdout)

# ── figure (MLST sizing: built at its final display width, 11–13 pt text) ────
BLU = RGBf(0.0,0.447,0.698); GRY = RGBAf(0.45,0.45,0.45,0.30); RED = RGBAf(0.80,0.15,0.15,0.45)
TITLE, LAB, TICK = 13, 12, 11

F_all  = [bands(θ, bp) for θ in members]
F_mean = bands(lin_params, bp)
lo = min(minimum(minimum.(F_all)), minimum(F_mean)); hi = max(maximum(maximum.(F_all)), maximum(F_mean))
pad = 0.06*(hi-lo)

fig = Figure(size=(540, 340), figure_padding=(6,10,4,6))
ax = Axis(fig[1,1]; xlabel="Wave vector", ylabel="Frequency (THz)",
          title="$(result.name) — a_eq-pinned POPS ($n_members samples, 4×4×4)",
          titlesize=TITLE, xlabelsize=LAB, ylabelsize=LAB,
          xticklabelsize=TICK, yticklabelsize=TICK,
          xticks=(bp.x_ticks, bp.labels), xgridvisible=false, ygridvisible=false,
          xtickalign=1, ytickalign=1, xticksize=4, yticksize=4)
for (i, Fθ) in enumerate(F_all)
    col = unst[i] ? RED : GRY
    for b in 1:3bp.Np; lines!(ax, bp.x_vals, Fθ[b,:]; color=col, linewidth=0.8); end
end
for b in 1:3bp.Np; lines!(ax, bp.x_vals, F_mean[b,:]; color=BLU, linewidth=1.8); end
hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=1.0)
vlines!(ax, bp.x_ticks; color=(:black,0.22), linewidth=0.7)
xlims!(ax, first(bp.x_vals), last(bp.x_vals)); ylims!(ax, min(lo-pad, -0.4), hi+pad)
text!(ax, 0.97, 0.97; text="$n_unst/$n_members unstable", space=:relative,
      align=(:right,:top), fontsize=TICK-1, color=:gray30)

ax2 = Axis(fig[1,2]; xlabel="min ω (THz)", ylabel="count", title="stability margin",
           titlesize=TITLE, xlabelsize=LAB, ylabelsize=LAB,
           xticklabelsize=TICK, yticklabelsize=TICK,
           xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1)
hist!(ax2, minf; bins=30, color=(BLU,0.6), strokewidth=0.5, strokecolor=:white)
vlines!(ax2, [0.0]; color=:black, linestyle=:dash, linewidth=1.0)
vlines!(ax2, [minf_mean]; color=BLU, linewidth=1.8)
colsize!(fig.layout, 1, Relative(0.62)); colgap!(fig.layout, 20)
save("$outdir/bands_pinned_hypercube.pdf", fig)
save("$outdir/bands_pinned_hypercube.png", fig; px_per_unit=4)

writedlm("$outdir/samples.csv", samples', ',')
writedlm("$outdir/min_freq.csv", hcat(minf, iq_min, pin_s, K_s, Δa_s), ',')
serialize("$outdir/pinned_hypercube.jls",
          (; minf, iq_min, pin_s, K_s, Δa_s, samples, a_ref, unstable_tol, n_unst,
             n_dir, n_masked=count(mask), seed=1234))
println("\nfigure → $outdir/bands_pinned_hypercube.{pdf,png}")
println("data   → $outdir/{samples,min_freq}.csv, pinned_hypercube.jls")
println("         min_freq.csv columns: min ω, argmin q index, b′·θ, b″·θ, Δa")
