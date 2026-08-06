# pinned_forest_Al_20_4_6A_3.jl
#
# Phonon stability of EVERY member of the POPS delta forest, on the undotted 4x4x4
# band path.  No hypercube, no sampling — this is the corrections themselves.
#
# ── WHY PIN ──────────────────────────────────────────────────────────────────
# Pinning is not a fix, it is what makes the sweep possible.  The undotted Hessian
# H_k = ∂²B_k/∂r² is built at ONE geometry, so Σ_k θ_k D_k(q) is only the right
# operator for a member whose equilibrium is at that geometry.  different_filter.jl's
# corrections are unpinned and span Δa ≈ ±0.015 Å, so each would need its own build
# (or a quasiharmonic shift).  Constraining every correction to b′·δ = 0 puts them all
# at a_ref, and then a single cached band path serves the entire forest at ~ms/member.
#
# ── THE PIN COSTS ONE SOLVE, NOT N QPs ───────────────────────────────────────
# In the preconditioned variable x = Pθ (Ap = W·A/P, C = λP'P + Ap'Ap, corrections
# mapped back by P\), the pin b′·θ = 0 is g′x = 0 with g = P' \ b′.  One equality on a
# quadratic objective is a rank-one correction to the inverse:
#
#     u = C \ g ,  v = Ap·u ,  gu = g′u
#     M Ap′ = Ainv − u v′/gu        Ainv = C \ Ap′  (the filter already computes it)
#     h^c   = h − v²/gu             constrained leverage
#     x̃_c   = x̃ − u (g′x̃)/gu        constrained mean fit
#     δ_i   = (Ainv[:,i] − u·v_i/gu) · r_i^c / h_i^c        r^c = Yw − Ap·x̃_c
#
# g′δ_i = (u′Ap′[:,i] − v_i)·s = 0 identically — exact, not converged.  Verified
# against explicitly solved KKT systems (pin AND exact interpolation of observation i)
# to 3.9e-16 in scratch; an OSQP-per-member route at 1829 params would have been
# ~400x the cost of the 91-param Al_12 run.
#
# ── NOTE ON UNITS ────────────────────────────────────────────────────────────
# The corrections stay in x-space throughout: P⁻¹ is folded into the band path ONCE
# (Bq/P) instead of converting 147k × 1829 corrections into θ-space, which would cost
# a 1829×1829 solve against every member and another 2.15 GB.  Since
# θ = lin_params + P⁻¹δ, we have D(q) = Bq·lin_params + (Bq/P)·δ.
#
# Run:  julia --project -t 40 scripts/uq/pinned_forest_Al_20_4_6A_3.jl [n_members]
#   Needs A.csv (~15-20 GB peak).  No Hessian build — the band path comes from the
#   cache written by forest_phonon_stability_Al_20_4_6A_3.jl.

using ACEWorkflow, ACEpotentials
import ACEpotentials.Models: evaluate_basis
import AtomsCalculatorsUtilities.SitePotentials: PairList, get_neighbours, cutoff_radius
using AtomsBuilder, Unitful, ForwardDiff, StaticArrays
using LinearAlgebra, Statistics, Random, Serialization, DelimitedFiles, Printf
using CairoMakie

Random.seed!(1234)

element           = :Al
filter_percentile = 0.5        # reported as a breakdown only — the sweep covers everything
unstable_tol      = -0.05      # THz
qΓtol             = 5e-2
n_plot            = 400
n_arg             = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 0

MODELDIR = "models/Al_20_4_6A_3_"
RES      = "$MODELDIR/results"
BPCACHE  = "$RES/forest_phonon_stability/bandpath_4x4x4.jls"
outdir   = "$RES/pinned_forest"; mkpath(outdir)

BPCACHE_OWN = "$outdir/bandpath_4x4x4_aref.jls"     # built here if the other job's isn't ready
N_cell      = 4
N_per_seg   = [20, 20, 20, 20, 60]
const BUILD_THREADS = parse(Int, get(ENV, "BUILD_THREADS", "8"))

# ═════════════════════════════════════════════════════════════════════════════
#  3-row undotted band-path builder (only Np = 1 rows of H are ever read).
#  Verified against a full 4x4x4 cache at machine precision: rows 1.5e-16,
#  D_k(q) 3.0e-16, end-to-end min ω 1.6e-14 THz.  Duplicated from
#  forest_phonon_stability_Al_20_4_6A_3.jl rather than `include`d so this script
#  has no path dependency — keep the two in step if either is edited.
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

result     = load_model(element, 20, 4, 6, 3; dataset_name="")
model      = result.model
lin_params = result.lin_params
n_params   = length(lin_params)
P, W, Y    = result.P, result.W, result.Y
Ap = Diagonal(W) * result.A / P
Yw = W .* Y
N  = size(Ap, 1)
@printf("Model %s: %d params, %d observations, %d threads\n",
        result.name, n_params, N, Threads.nthreads()); flush(stdout)

# ── band path (cached) + the pin direction ──────────────────────────────────
# Reuse a cached band path if one exists (the forest job's, or this script's own from
# a previous run); otherwise build it.  Only ONE geometry is needed here — every member
# is pinned to a_ref, so there is no dD/da and no second build.
bp = if isfile(BPCACHE)
    println("band path: reusing $BPCACHE"); deserialize(BPCACHE)
elseif isfile(BPCACHE_OWN)
    println("band path: reusing $BPCACHE_OWN"); deserialize(BPCACHE_OWN)
else
    println("band path: no cache found — building one (this is the only build needed)")
    a0 = ACEWorkflow.relax_lattice_constant(model, element)
    @printf("  a_ref = %.17g Å; box %.2f Å, half-box %.2f Å vs 6 Å cutoff\n",
            a0, N_cell*a0, N_cell*a0/2); flush(stdout)
    b = bandpath_rows(model, element, a0, N_cell)
    serialize(BPCACHE_OWN, b); println("  cached → $BPCACHE_OWN")
    b
end
a_ref = bp.a_ref
@printf("band path: %d q-points, %d branches, a_ref = %.17g Å\n",
        length(bp.Bq), 3bp.Np, a_ref)

lattice_basis(a) = ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(
                       ACEWorkflow.Elasticity.reference_system(element; a=a), model))
b_prime        = ForwardDiff.derivative(lattice_basis, a_ref)
b_double_prime = ForwardDiff.derivative(a -> ForwardDiff.derivative(lattice_basis, a), a_ref)
@printf("b′·lin_params  = %.3e  (~0: a_ref is the mean model's relaxed lattice constant,\n", dot(b_prime, lin_params))
println("                         so pinning the deltas pins every member's total b′·θ)")
@printf("b″·lin_params  = %.6e  (> 0 ⇒ a_ref is a minimum, not a maximum)\n",
        dot(b_double_prime, lin_params))

λ  = 1.0 / N
C  = Symmetric(P'*P .* λ .+ Ap'*Ap)
Cf = cholesky(C)
@printf("C: %d×%d, cond ≈ %.3e\n", n_params, n_params, cond(Matrix(C))); flush(stdout)

Ainv = Cf \ Matrix(Ap')                       # p × N
x̃    = Cf \ (Ap' * Yw)
lev  = vec(sum(Ap' .* Ainv; dims=1))

g  = transpose(P) \ b_prime
u  = Cf \ g
gu = dot(g, u)
v  = Ap * u
abs(gu) > 1e-30 || error("g′u ≈ 0 — pin direction lies in the null space of C⁻¹")

x̃_c   = x̃ .- u .* (dot(g, x̃)/gu)
lev_c = lev .- (v .^ 2) ./ gu
r_c   = Yw .- Ap * x̃_c
@printf("\nconstrained leverage: min %.4e (unconstrained %.4e), %d non-positive\n",
        minimum(lev_c), minimum(lev), count(<=(0), lev_c))
@printf("‖x̃_c − x̃‖/‖x̃‖ = %.4e\n", norm(x̃_c .- x̃)/norm(x̃)); flush(stdout)

# pinned corrections, x-space, one row per observation — the WHOLE forest
scale = r_c ./ lev_c
PCx   = transpose(Ainv .* transpose(scale))
PCx .-= (scale .* v ./ gu) * transpose(u)
Ainv = nothing; GC.gc()

# ── the pin, verified.  b′·(P⁻¹δ) = (P⁻ᵀb′)·δ = g·δ, so this is the θ-space pin ──
pin_resid = PCx * g
@printf("\nPIN CHECK: max |b′·δ| = %.3e over all %d corrections  (‖g‖ = %.3e)\n",
        maximum(abs.(pin_resid)), N, norm(g))
@printf("           → every member's equilibrium is at a_ref; the single cached\n")
@printf("             4×4×4 band path is the exact operator for all of them\n")

# ── second-order condition: b″·θ > 0 ────────────────────────────────────────
# b′·θ = 0 only makes a_ref STATIONARY.  b″·θ > 0 is what makes it a minimum; a member
# with b″·θ ≤ 0 sits at a maximum of E(a) and has no FCC volume minimum at all, so its
# band structure at a_ref describes a structure the member does not actually hold.
# That is a distinct failure from a soft mode at finite q, and is counted separately.
#
# Note this is an INEQUALITY, so unlike the b′ pin it cannot be imposed by the rank-one
# projector — enforcing it would need a per-member QP (or a second equality projection
# onto b″·θ = ε for the violators only, which stays closed form as a 2×2 system).  Here
# it is measured, not imposed.
k_dd   = transpose(P) \ b_double_prime          # b″·(P⁻¹δ) = (P⁻ᵀb″)·δ
K_all  = dot(b_double_prime, lin_params) .+ PCx * k_dd
n_badK = count(<=(0), K_all)
@printf("\nSECOND-ORDER b″·θ > 0: %d / %d violate (%.4f%%)\n", n_badK, N, 100n_badK/N)
@printf("           b″·θ ∈ [%+.4e, %+.4e], median %+.4e (mean model %+.4e)\n",
        minimum(K_all), maximum(K_all), median(K_all), dot(b_double_prime, lin_params))
n_badK > 0 && println("           ↑ these have NO volume minimum at a_ref — reported apart from soft modes")
flush(stdout)

# ── ratio filter: reported as a breakdown, NOT used to subset the sweep ─────
# identity form — the naive PC*C*PC′ is 147k² = 173 GB and is never formed
S_pc = transpose(PCx) * PCx
G_pc = C * S_pc * C
filter_metric = sqrt.(vec(sum((PCx * G_pc) .* PCx; dims=2))) ./ lev_c
mask = filter_metric .>= quantile(filter_metric, filter_percentile)
@printf("ratio filter: %d of %d in the top %.0f%% (the subset that feeds the hypercube)\n",
        count(mask), N, 100*(1-filter_percentile)); flush(stdout)

# ── fold P⁻¹ into the band path once, and precompute the mean model's D(q) ──
nq   = length(bp.Bq)
BqP  = [bp.Bq[iq] / P for iq in 1:nq]                 # 9 × p, acts on x-space deltas
base = [bp.Bq[iq] * lin_params for iq in 1:nq]        # mean model contribution
keep = findall(bp.qnorm .>= qΓtol)                    # skip near-Γ (acoustic → 0)
@printf("folded P⁻¹ into %d q-points; %d retained after the near-Γ cut\n", nq, length(keep))
flush(stdout)

# ── sweep every correction ─────────────────────────────────────────────────
sel   = n_arg > 0 ? (1:min(n_arg, N)) : (1:N)
n_sel = length(sel)
n_arg > 0 && @warn "PILOT: first $n_sel corrections by observation index — biased, do not extrapolate a rate"

minω   = fill(NaN, n_sel)
iq_min = zeros(Int, n_sel)
println("\n── sweeping $n_sel pinned corrections ──"); flush(stdout)
t = @elapsed Threads.@threads for t_ in 1:n_sel
    δ = @view PCx[sel[t_], :]
    mw = Inf; im_ = 0
    for iq in keep
        ev = eigvals(Hermitian(reshape(base[iq] .+ BqP[iq]*δ, 3, 3)))
        w  = minimum(sign.(ev) .* sqrt.(abs.(ev)) .* FREQ_THz)
        w < mw && (mw = w; im_ = iq)
    end
    minω[t_] = mw; iq_min[t_] = im_
end
@printf("  swept in %.1f s (%.3f ms/member)\n", t, 1000t/n_sel); flush(stdout)

# function, not a bare `begin` block: at top level a `for` loop opens a soft scope, so
# assigning to a name that also exists as a global makes it a fresh local and reading it
# after the loop throws UndefVarError.
function min_freq_of(vecs, keep)
    mw = Inf
    for iq in keep
        ev = eigvals(Hermitian(reshape(vecs[iq], 3, 3)))
        mw = min(mw, minimum(sign.(ev) .* sqrt.(abs.(ev)) .* FREQ_THz))
    end
    return mw
end
minω_mean = min_freq_of(base, keep)

unst   = minω .< unstable_tol          # soft mode at finite q
badK   = K_all[sel] .<= 0.0            # no volume minimum at a_ref
msel   = mask[sel]
n_unst = count(unst)
ok     = .!unst .& .!badK

@printf("\nmean model      : min ω = %+.4f THz\n", minω_mean)
@printf("pinned forest   : min ω ∈ [%+.4f, %+.4f], median %+.4f\n",
        minimum(minω), maximum(minω), median(minω))
println("\n  failure modes (a member can fail both):")
@printf("    soft mode, min ω < %.2f THz : %7d / %d  (%.4f%%)\n",
        unstable_tol, n_unst, n_sel, 100n_unst/n_sel)
@printf("    b″·θ ≤ 0, no volume minimum : %7d / %d  (%.4f%%)\n",
        count(badK), n_sel, 100count(badK)/n_sel)
@printf("    both                        : %7d\n", count(unst .& badK))
@printf("    FULLY STABLE                : %7d / %d  (%.4f%%)\n",
        count(ok), n_sel, 100count(ok)/n_sel)
println("\n  split by the ratio filter:")
@printf("    top-%.0f%% (feeds the hypercube) : %d/%d soft (%.4f%%), %d/%d b″≤0\n",
        100*(1-filter_percentile), count(unst .& msel), count(msel),
        100*count(unst .& msel)/max(count(msel),1), count(badK .& msel), count(msel))
@printf("    bottom                        : %d/%d soft (%.4f%%), %d/%d b″≤0\n",
        count(unst .& .!msel), count(.!msel),
        100*count(unst .& .!msel)/max(count(.!msel),1), count(badK .& .!msel), count(.!msel))
if count(msel) > 0 && count(.!msel) > 0
    r_in  = count(unst .& msel)/count(msel)
    r_out = count(unst .& .!msel)/max(count(.!msel),1)
    @printf("    → the filter %s soft members (%.2f× enrichment)\n",
            r_in > r_out ? "CONCENTRATES" : "depletes", r_out > 0 ? r_in/r_out : Inf)
end
flush(stdout)

# ── where in the zone do the soft modes sit? ───────────────────────────────
if n_unst > 0
    seglab = ["Γ→X", "X→U", "U→L", "L→Γ", "Γ→K"]
    segof(iq) = clamp(searchsortedlast(bp.x_ticks, bp.x_vals[iq]), 1, length(seglab))
    println("\nsoft-mode location (segment carrying each unstable member's minimum):")
    for s in 1:length(seglab)
        c = count(i -> segof(iq_min[i]) == s, findall(unst))
        c == 0 && continue
        @printf("  %-5s %6d  (%.1f%% of unstable members)\n", seglab[s], c, 100c/n_unst)
    end
    @printf("worst member: obs %d, min ω = %+.4f THz at q = %s\n",
            sel[argmin(minω)], minimum(minω), string(round.(bp.q_list[iq_min[argmin(minω)]]; digits=4)))
end
flush(stdout)

# ── figure ─────────────────────────────────────────────────────────────────
BLU = RGBf(0.0,0.447,0.698); GRY = RGBAf(0.45,0.45,0.45,0.20); RED = RGBAf(0.80,0.15,0.15,0.55)
TITLE, LAB, TICK = 13, 12, 11
bands_of(δ) = begin
    F = Matrix{Float64}(undef, 3, nq)
    for iq in 1:nq
        ev = eigvals(Hermitian(reshape(base[iq] .+ BqP[iq]*δ, 3, 3)))
        F[:, iq] = sign.(ev) .* sqrt.(abs.(ev)) .* FREQ_THz
    end
    F
end
# show a random subsample plus every unstable member (they are the point)
idx = unique(vcat(findall(unst), randperm(n_sel)[1:min(n_plot, n_sel)]))

fig = Figure(size=(540, 350), figure_padding=(6,10,4,6))
ax = Axis(fig[1,1]; xlabel="Wave vector", ylabel="Frequency (THz)",
          title="a_eq-pinned delta forest, 4×4×4", titlesize=TITLE,
          xlabelsize=LAB, ylabelsize=LAB, xticklabelsize=TICK, yticklabelsize=TICK,
          xticks=(bp.x_ticks, bp.labels), xgridvisible=false, ygridvisible=false,
          xtickalign=1, ytickalign=1, xticksize=4, yticksize=4)
for t_ in idx
    F = bands_of(@view PCx[sel[t_], :])
    col = unst[t_] ? RED : GRY
    for b in 1:3; lines!(ax, bp.x_vals, F[b,:]; color=col, linewidth=0.6); end
end
Fm = bands_of(zeros(n_params))
for b in 1:3; lines!(ax, bp.x_vals, Fm[b,:]; color=BLU, linewidth=1.8); end
hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=1.0)
vlines!(ax, bp.x_ticks; color=(:black,0.22), linewidth=0.7)
xlims!(ax, first(bp.x_vals), last(bp.x_vals))
text!(ax, 0.97, 0.97; text="$n_unst/$n_sel unstable", space=:relative,
      align=(:right,:top), fontsize=TICK-1, color=:gray30)

ax2 = Axis(fig[1,2]; xlabel="min ω (THz)", ylabel="density", title="whole forest",
           titlesize=TITLE, xlabelsize=LAB, ylabelsize=LAB,
           xticklabelsize=TICK, yticklabelsize=TICK, yscale=log10,
           xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1)
hist!(ax2, minω; bins=80, normalization=:pdf, color=(BLU,0.6),
      strokewidth=0.3, strokecolor=:white)
vlines!(ax2, [0.0]; color=:black, linestyle=:dash, linewidth=1.0)
vlines!(ax2, [minω_mean]; color=BLU, linewidth=1.8)
colsize!(fig.layout, 1, Relative(0.62)); colgap!(fig.layout, 20)
save("$outdir/pinned_forest.pdf", fig)
save("$outdir/pinned_forest.png", fig; px_per_unit=4)

writedlm("$outdir/pinned_min_freq.csv", hcat(collect(sel), minω, iq_min, msel), ',')
serialize("$outdir/pinned_forest.jls",
          (; minω, iq_min, mask=msel, sel=collect(sel), a_ref, unstable_tol, n_unst))
println("\nfigure  → $outdir/pinned_forest.{pdf,png}")
println("min ω   → $outdir/pinned_min_freq.csv  (obs, min ω, argmin q index, in-filter)")
