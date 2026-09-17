# different_filter.jl — ratio-filtered POPS corrections, then the phonon bands of
# the resulting samples evaluated on the undotted per-basis Hessian.
#
# Run:  julia --project -t <nthreads> scripts/uq/different_filter.jl

using ACEWorkflow        # load_model, hypercube, sample_hypercube, bulk_prim_super,
                         # precompute_force_constants, dynamical_matrix_from_fc, FREQ_THz, Elasticity
using ACEpotentials      # Models.set_linear_parameters!, Models.potential_energy_basis
using AtomsBuilder       # Chemistry.symmetry  (picks the fcc/bcc/hcp high-symmetry path)
using LinearAlgebra      # Diagonal, diag, dot, norm
using Statistics         # quantile, median
using Serialization      # deserialize  (the cached undotted Hessian)
using DelimitedFiles     # writedlm
using Printf             # @printf
using ForwardDiff        # b′ = ∂_a B(a), b″ = ∂²_a B(a) for the a_eq-shift diagnostic
using Unitful            # u"eV", ustrip
using CairoMakie         # Figure, Axis, lines!, hist!, save

# ── phonon helpers (self-contained; no include, nothing outside the list above) ──
# These are byte-identical in behaviour to the versions in
# scripts/bandpath_phonon_uq/lib.jl — inlined here so this script has no path
# dependency on that file and can be pasted into a REPL in any order.

# high-symmetry path for the crystal structure (diamond shares the FCC zone)
band_path_for(sym, L; N_per_seg) =
    sym in (:fcc, :diamond) ? fcc_band_path(L; N_per_seg) :
    sym == :bcc             ? bcc_band_path(L; N_per_seg) :
    sym == :hcp             ? hcp_band_path(L; N_per_seg) :
    error("unknown structure $sym — expected :fcc/:diamond/:bcc/:hcp")

# frequencies (THz), 3Np × n_q, over the full path.  ω = sign(ω²)·√|ω²|, so
# dynamical instabilities show up as negative frequencies rather than complex ones.
function bands(θ, bp)
    F = Matrix{Float64}(undef, 3bp.Np, length(bp.Bq))
    for iq in eachindex(bp.Bq)
        ev = eigvals(Hermitian(reshape(bp.Bq[iq] * θ, 3bp.Np, 3bp.Np)))
        F[:, iq] = sign.(ev) .* sqrt.(abs.(ev)) .* FREQ_THz
    end
    return F
end

# minimum frequency away from Γ (the acoustic branches → 0 at Γ by construction,
# so including it would make every model look marginally stable)
function min_freq_stable(θ, bp; qΓtol = 5e-2)
    m = Inf
    for iq in eachindex(bp.Bq)
        bp.qnorm[iq] < qΓtol && continue
        ev = eigvals(Hermitian(reshape(bp.Bq[iq] * θ, 3bp.Np, 3bp.Np)))
        m = min(m, minimum(sign.(ev) .* sqrt.(abs.(ev)) .* FREQ_THz))
    end
    return m
end

result = load_model(:Al, 20, 4, 6, 3; dataset_name="")
A = result.A
P = result.P
Y = result.Y
W = result.W

Ap = Diagonal(W) * A / P
Yw = Y .* W

function ratio_filtered_corrections(X::AbstractMatrix{Float64}, Y::Vector{Float64}, Gamma::AbstractMatrix{Float64}; filter_percentile::Float64 = 0.5, lambda::Float64 = 1.0 / size(X,1), coeffs=nothing, return_mask::Bool = false)
    C      = (Gamma' * Gamma .* lambda .+ X' * X)
    A      = C \ X'
    leverage = diag(X * A)
    if (coeffs == nothing)
        coeffs = C \ (X' * Y)
    end
    errors = Y .- (X * coeffs)
    pointwise_corrections = A' .* (errors ./ leverage)
    # filter_metric = norm.(eachrow(pointwise_corrections * C * pointwise_corrections')) ./ leverage
    #
    # ^ mathematically what we want, but PC*C*PC' is N×N = 147k×147k (173 GB, ~4e16
    # flops).  It never has to be formed.  With v_i = C·pc_i and S = PC'PC:
    #     ‖(PC C PC')[i,:]‖² = Σ_j (v_i'pc_j)² = v_i' S v_i = pc_i' (C S C) pc_i
    # so the row norms are a quadratic form in p=1829 dimensions.  Exact, not an
    # approximation — agrees with the naive expression to 5e-16 on random tests.
    S_pc  = pointwise_corrections' * pointwise_corrections     # p × p
    G_pc  = C * S_pc * C                                       # p × p, symmetric PSD
    filter_metric = sqrt.(vec(sum((pointwise_corrections * G_pc) .* pointwise_corrections; dims=2))) ./ leverage
    percentile_threshold = quantile(filter_metric, filter_percentile)
    mask = filter_metric .>= percentile_threshold
    pointwise_corrections = A[:,mask]'
    pointwise_corrections = pointwise_corrections .* (errors[mask] ./ leverage[mask])
    pointwise_corrections = Gamma \ pointwise_corrections'
    if (return_mask == true)
        return pointwise_corrections', mask
    else
        return pointwise_corrections'
    end
end

pwise, mask = ratio_filtered_corrections(Ap, Yw, P; return_mask=true)
eig_pops, bound_pops = hypercube(pwise)
samples, _ = sample_hypercube(eig_pops, bound_pops, result.lin_params)

HESSIAN_PATH = "/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/models/Al_20_4_6A_3_/results/undotted_Hbasis_3x3x3_a4.01962.jls"

# Reload only if it isn't already sitting in the session — deserializing this is
# 1.5 GB and ~a minute, so re-running the block below must not pay for it again.
if !@isdefined(Hessian) || Hessian === nothing
    Hessian = deserialize(HESSIAN_PATH)
end
@assert Hessian isa NamedTuple && haskey(Hessian, :H_basis) "Hessian cache should be (H_basis, a_eq, N_cell); got $(typeof(Hessian))"

# ═══════════════════════════════════════════════════════════════════════════════
#  Phonons of the ratio-filtered samples on the undotted Hessian
# ═══════════════════════════════════════════════════════════════════════════════
# H_k = ∂²B_k/∂r² is geometry-only, so at fixed a it is built ONCE and each
# sample's dispersion is D(q;θ) = Σ_k θ_k D_k(q) — a matvec plus a 3×3 eigen.
#
# CAVEAT worth reading before interpreting the bands: `sample_hypercube` here is
# the NAIVE sampler — no b′·θ = 0 pin, no Born rows, no phonon predicate.  Every
# member is therefore evaluated at the MEAN model's a_eq, not at its own
# equilibrium, so part of any softening is just residual stress.  The implied
# a_eq shift per member is reported below so that contribution is visible.

element   = :Al
a_eq      = Hessian.a_eq                     # 4.01962 Å — the geometry H_k was built at
N_cell    = Hessian.N_cell                   # 3 → 108-atom cubic supercell, N3 = 324
N_per_seg = [20, 20, 20, 20, 60]             # dense Γ→K, where soft modes concentrate
unstable_tol = -0.05                         # THz; below this a member is "dynamically unstable"
qΓtol     = 5e-2                             # skip near-Γ (acoustic → 0) in the min-ω metric

model      = result.model
lin_params = result.lin_params
outdir     = "$(result.dir)/results/different_filter"; mkpath(outdir)
@printf("Model %s: %d params, %d samples, %d threads.  a_eq = %.5f Å, N_cell = %d\n",
        result.name, length(lin_params), size(samples, 2), Threads.nthreads(), a_eq, N_cell)
flush(stdout)

# ── band-path D_k(q) from the ALREADY-LOADED cache ───────────────────────────
# Deliberately not bandpath_Dk(): that calls undotted_Hbasis(), which would
# deserialize a second 1.5 GB copy of the same array we are holding.
function bandpath_from_Hbasis(result, model, element, a, N_cell, Hb; N_per_seg=20)
    sys_prim, sys_super = bulk_prim_super(element; a=a, N_cell=N_cell)
    ACEpotentials.Models.set_linear_parameters!(model, result.lin_params)
    fc0 = precompute_force_constants(sys_prim, sys_super, model)
    Np  = fc0.Np; N3 = 3*length(sys_super); np_ = size(Hb, 3)
    size(Hb, 1) == N3 || error("cached H_basis is $(size(Hb,1))×$(size(Hb,2)) but a=$a, N_cell=$N_cell needs $N3 — wrong cache")
    np_ == length(result.lin_params) || error("cached H_basis has $np_ basis functions, model has $(length(result.lin_params))")
    q_list, x_vals, x_ticks, labels, _ =
        band_path_for(AtomsBuilder.Chemistry.symmetry(element), fc0.L; N_per_seg=N_per_seg)
    Bq = Vector{Matrix{ComplexF64}}(undef, length(q_list))
    Threads.@threads for iq in eachindex(q_list)
        M = Matrix{ComplexF64}(undef, (3Np)^2, np_)
        for k in 1:np_
            M[:, k] = vec(dynamical_matrix_from_fc(merge(fc0, (H = @view(Hb[:, :, k]),)), q_list[iq]))
        end
        Bq[iq] = M
    end
    return (; Bq, q_list, x_vals, x_ticks, labels, Np, qnorm = norm.(q_list))
end

t_bp = @elapsed bp = bandpath_from_Hbasis(result, model, element, a_eq, N_cell, Hessian.H_basis;
                                          N_per_seg=N_per_seg)
@printf("Band-path D_k(q): %d q-points, %d branches  [%.1f s]\n", length(bp.Bq), 3bp.Np, t_bp)
# `Hessian` is deliberately NOT freed here: everything below needs only `bp`, but
# dropping it mid-script means any later error leaves you with Hessian === nothing
# and a 1.5 GB reload to get back.  Free it by hand if memory is tight:
#     Hessian = nothing; GC.gc()
flush(stdout)

# ── evaluate: mean model + every sample ──────────────────────────────────────
members  = [samples[:, i] for i in 1:size(samples, 2)]
F_mean   = bands(lin_params, bp)
minf     = [min_freq_stable(θ, bp; qΓtol=qΓtol) for θ in members]
minf_mean = min_freq_stable(lin_params, bp; qΓtol=qΓtol)
n_unstable = count(<(unstable_tol), minf)

@printf("\nmean model      : min ω = %+.4f THz, ω ∈ [%.3f, %.3f] THz\n",
        minf_mean, minimum(F_mean), maximum(F_mean))
@printf("filtered samples: min ω ∈ [%+.4f, %+.4f] THz, median %+.4f\n",
        minimum(minf), maximum(minf), median(minf))
@printf("                  %d / %d dynamically UNSTABLE (min ω < %.2f THz)\n",
        n_unstable, length(minf), unstable_tol)

# how far each member's own equilibrium sits from the geometry the phonons use:
# Newton step Δa ≈ −b′·θ / b″·θ  (zero by construction only for a_eq-constrained fits)
lattice_basis(a) = ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(
                       ACEWorkflow.Elasticity.reference_system(element; a=a), model))
b_prime        = ForwardDiff.derivative(lattice_basis, a_eq)
b_double_prime = ForwardDiff.derivative(a -> ForwardDiff.derivative(lattice_basis, a), a_eq)
Δa = [-dot(b_prime, θ) / dot(b_double_prime, θ) for θ in members]
@printf("implied a_eq shift: Δa ∈ [%+.4f, %+.4f] Å (|Δa|/a max %.2f%%) — mean model %+.5f Å\n",
        minimum(Δa), maximum(Δa), 100*maximum(abs.(Δa))/a_eq,
        -dot(b_prime, lin_params)/dot(b_double_prime, lin_params))
flush(stdout)

# ── figure (MLST sizing: built at its final display width, 11–13 pt text) ────
BLU = RGBf(0.0, 0.447, 0.698); GRY = RGBAf(0.45, 0.45, 0.45, 0.30); RED = RGBAf(0.80, 0.15, 0.15, 0.45)
TITLE, LAB, TICK = 13, 12, 11

F_all = [bands(θ, bp) for θ in members]
lo = min(minimum(minimum.(F_all)), minimum(F_mean)); hi = max(maximum(maximum.(F_all)), maximum(F_mean))
pad = 0.06*(hi-lo)

fig = Figure(size=(540, 340), figure_padding=(6, 10, 4, 6))
ax = Axis(fig[1, 1]; xlabel="Wave vector", ylabel="Frequency (THz)",
          title="$(result.name) — ratio-filtered POPS ($(length(members)) samples)",
          titlesize=TITLE, xlabelsize=LAB, ylabelsize=LAB,
          xticklabelsize=TICK, yticklabelsize=TICK,
          xticks=(bp.x_ticks, bp.labels), xgridvisible=false, ygridvisible=false,
          xtickalign=1, ytickalign=1, xticksize=4, yticksize=4)
for (θi, Fθ) in enumerate(F_all)
    col = minf[θi] < unstable_tol ? RED : GRY
    for b in 1:3bp.Np; lines!(ax, bp.x_vals, Fθ[b, :]; color=col, linewidth=0.8); end
end
for b in 1:3bp.Np; lines!(ax, bp.x_vals, F_mean[b, :]; color=BLU, linewidth=1.8); end
hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=1.0)
vlines!(ax, bp.x_ticks; color=(:black, 0.22), linewidth=0.7)
xlims!(ax, first(bp.x_vals), last(bp.x_vals)); ylims!(ax, min(lo-pad, -0.4), hi+pad)
text!(ax, 0.97, 0.97; text="$n_unstable/$(length(members)) unstable", space=:relative,
      align=(:right, :top), fontsize=TICK-1, color=:gray30)

ax2 = Axis(fig[1, 2]; xlabel="min ω (THz)", ylabel="count",
           title="stability margin", titlesize=TITLE, xlabelsize=LAB, ylabelsize=LAB,
           xticklabelsize=TICK, yticklabelsize=TICK,
           xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1)
hist!(ax2, minf; bins=30, color=(BLU, 0.6), strokewidth=0.5, strokecolor=:white)
vlines!(ax2, [0.0]; color=:black, linestyle=:dash, linewidth=1.0)
vlines!(ax2, [minf_mean]; color=BLU, linewidth=1.8)
colsize!(fig.layout, 1, Relative(0.62)); colgap!(fig.layout, 20)
save("$outdir/bands_different_filter.pdf", fig)
save("$outdir/bands_different_filter.png", fig; px_per_unit=4)

writedlm("$outdir/samples.csv", samples', ',')
writedlm("$outdir/min_freq.csv", minf, ',')
@printf("\nfigure → %s/bands_different_filter.{pdf,png}\n", outdir)
println("samples + min ω → $outdir/{samples,min_freq}.csv")
