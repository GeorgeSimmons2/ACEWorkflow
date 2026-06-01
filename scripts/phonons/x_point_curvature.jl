# x_point_curvature.jl
#
# Demonstrates the equivalence between two independent ways to compute the
# phonon frequency at the FCC X-point:
#
#  (1) Dynamical-matrix route
#      Build D(q_X) from supercell force constants → eigenvalue ω²_dyn.
#
#  (2) AD curvature route
#      Define B(d) : ℝ → ℝ^{N_params} = ACE design matrix of the 4-atom
#      FCC conventional cell with atoms displaced according to the X-point
#      eigenvector scaled by amplitude d  (d = 0 → equilibrium).
#      Compute the curvature operator  C = d²B/dd²|_{d=0}  via
#      ForwardDiff (inner, first derivative) + central finite differences
#      (outer, second derivative).  This avoids nested dual numbers.
#
#      Since the conventional cell contains N_cells = 4 primitive cells,
#      the extensive curvature satisfies:
#
#         C · θ  =  N_cells · ω²_dyn
#         ω²_AD  :=  (C · θ) / N_cells   ≈   ω²_dyn            [eV/Å²/amu]
#
# The script prints every X-point mode and the relative difference between
# the two routes, verifying that the design matrix exactly encodes ω².
#
# Usage:
#   julia --project scripts/phonons/x_point_curvature.jl

using LinearAlgebra, StaticArrays, Printf, ForwardDiff, Dates, DelimitedFiles
using AtomsBuilder, AtomsCalculators, Unitful, AtomsBase
using AtomsCalculatorsUtilities.SitePotentials: hessian
using ACEWorkflow, ACEpotentials

# ─────────────────────────────────────────────────────────────────────────────
#  Physical constants
# ─────────────────────────────────────────────────────────────────────────────
const _eV_J    = 1.602176634e-19
const _Å_m     = 1.0e-10
const _amu_kg  = 1.66053906660e-27
const FREQ_THz = sqrt(_eV_J / (_Å_m^2 * _amu_kg)) / (2π * 1e12)   # ≈ 15.633

# ─────────────────────────────────────────────────────────────────────────────
#  Force constant / dynamical-matrix machinery
#  (self-contained copy of the core from negative_phonon_example.jl)
# ─────────────────────────────────────────────────────────────────────────────

function _lattice_mat(cell_vectors)
    a, b, c = cell_vectors
    SMatrix{3,3,eltype(a)}(hcat(a, b, c))
end

"""
    precompute_force_constants(sys_prim, sys_super, model)

Compute the real-space force constants by evaluating the Hessian of
`sys_super` (which must be large enough that FC decay within the supercell).
Returns a NamedTuple consumed by `dynamical_matrix_from_fc`.
"""
function precompute_force_constants(sys_prim, sys_super, model)
    H  = ustrip.(hessian(sys_super, model))
    Np = length(sys_prim)
    Ns = length(sys_super)

    L_prim  = SMatrix{3,3,Float64}(ustrip.(_lattice_mat(sys_prim.cell.cell_vectors)))
    L_super = SMatrix{3,3,Float64}(ustrip.(_lattice_mat(sys_super.cell.cell_vectors)))
    Linv_p  = inv(L_prim)
    Linv_s  = inv(L_super)

    frac_prim  = [Linv_p * SVector{3,Float64}(ustrip.(sys_prim[i].position))  for i in 1:Np]
    frac_super = [Linv_p * SVector{3,Float64}(ustrip.(sys_super[k].position)) for k in 1:Ns]

    s2p_map = Vector{Int}(undef, Ns)
    for k in 1:Ns
        matched = false
        for i in 1:Np
            d = mod.(frac_super[k], 1.0) .- mod.(frac_prim[i], 1.0)
            d = d .- round.(d)
            if norm(d) < 1e-6
                s2p_map[k] = i - 1
                matched = true
                break
            end
        end
        matched || error("Supercell atom $k did not match any primitive atom")
    end

    p2s_map = Vector{Int}(undef, Np)
    for i in 1:Np
        candidates = findall(s2p_map .== (i - 1))
        r_ref = SVector{3,Float64}(ustrip.(sys_prim[i].position))
        dists = [norm(SVector{3,Float64}(ustrip.(sys_super[k].position)) .- r_ref)
                 for k in candidates]
        p2s_map[i] = candidates[argmin(dists)] - 1
    end

    masses = [ustrip(sys_prim[i].mass) for i in 1:Np]
    return (; H, frac_prim, frac_super, masses,
              L=L_prim, L_super, Linv_super=Linv_s, Np, Ns, p2s_map, s2p_map)
end

"""
    dynamical_matrix_from_fc(fc_data, q_cart)

Build the (3Np × 3Np) mass-weighted dynamical matrix at Cartesian wavevector
`q_cart` (Å⁻¹) from precomputed force constants.
"""
function dynamical_matrix_from_fc(fc_data, q_cart::AbstractVector{<:Real})
    (; H, frac_prim, frac_super, masses, L, L_super, Linv_super, Np, Ns, p2s_map, s2p_map) = fc_data

    Dq = zeros(ComplexF64, 3Np, 3Np)
    for i in 0:Np-1, j in 0:Np-1
        dm = zeros(ComplexF64, 3, 3)
        for k in 0:Ns-1
            s2p_map[k+1] == j || continue
            R_cart  = L * (frac_super[k+1] - frac_prim[j+1])
            R_sfrac = Linv_super * R_cart
            R_sfrac = R_sfrac .- round.(R_sfrac)    # minimum image in supercell
            R_mi    = L_super * R_sfrac
            eph     = exp(im * dot(q_cart, R_mi))
            for α in 1:3, β in 1:3
                dm[α, β] += H[3*p2s_map[i+1]+α, 3k+β] * eph
            end
        end
        ms = sqrt(masses[i+1] * masses[j+1])
        Dq[3i+1:3i+3, 3j+1:3j+3] .= dm ./ ms
    end
    # Enforce Hermitian symmetry
    for i in 1:3Np, j in i:3Np
        Dq[i,j] = (Dq[i,j] + conj(Dq[j,i])) / 2
        Dq[j,i] = conj(Dq[i,j])
    end
    return Dq
end

# ─────────────────────────────────────────────────────────────────────────────
#  Load model
# ─────────────────────────────────────────────────────────────────────────────
println("Loading model Al_12_4_6A_3 …")
result = load_model(:Al, 20, 4, 6, 3)
model  = result.model
θ      = result.lin_params
using Random
Random.seed!(1234)

# ── POPS ─────────────────────────────────────────────────────────────────────
# pops_corrections = readdlm("$(result.dir)/pops_corrections.csv", ',')
pops_corrections = corrections(Diagonal(result.W) * result.A / result.P, result.W .* result.Y, result.P; leverage_percentile=0.0)
pops_samples     = [vec(pops_corrections[i,:]) .+ θ for i=1:size(pops_corrections, 1)]
@printf("  N_params = %d\n", length(θ))

# ─────────────────────────────────────────────────────────────────────────────
#  Build systems
#   - sys_prim  : 1-atom primitive FCC cell  (defines the dynamical matrix)
#   - sys_conv  : 4-atom conventional FCC cell  (displacement cell for AD)
#   - sys_super : 108-atom 3×3×3 conventional supercell  (force constants)
# ─────────────────────────────────────────────────────────────────────────────
println("Relaxing lattice constant …")
a0 = ACEWorkflow.relax_lattice_constant(model, :Al)
@printf("  a₀ = %.6f Å\n", a0)

sys_prim  = bulk(:Al; a=a0 * u"Å")
sys_conv  = bulk(:Al; a=a0 * u"Å", cubic=true)
sys_super = bulk(:Al; a=a0 * u"Å", cubic=true) * (3, 3, 3)

N_prim_cells = length(sys_conv)   # 4 (FCC primitive cell has 1 atom)

@printf("  Primitive cell  : %d atom(s)\n", length(sys_prim))
@printf("  Displacement cell: %d atoms (conventional FCC, = %d prim. cells)\n",
        length(sys_conv), N_prim_cells)
@printf("  Force-const. cell: %d atoms (3×3×3 conventional)\n", length(sys_super))

# ─────────────────────────────────────────────────────────────────────────────
#  Force constants
# ─────────────────────────────────────────────────────────────────────────────
println("\nPrecomputing force constants (Hessian of 108-atom supercell) …")
fc_data = precompute_force_constants(sys_prim, sys_super, model)
println("  done.")

# ─────────────────────────────────────────────────────────────────────────────
#  X-point wavevector (Cartesian, Å⁻¹)
#
#  FCC primitive reciprocal vectors: B = 2π inv(Lᵀ_prim), columns = b₁ b₂ b₃.
#  X-point in primitive fractional coordinates: [0, ½, ½].
#
#  For the FCC primitive cell:
#    q_X = B * [0, 0.5, 0.5] = (2π/a, 0, 0)  in Cartesian
#
#  This makes q_X commensurate with the conventional cell:
#    exp(i q_X · A_k) = 1  for all conventional lattice vectors A_k,
#  so the conventional cell can host the X-point displacement pattern.
# ─────────────────────────────────────────────────────────────────────────────
B_rec = 2π * inv(transpose(fc_data.L))   # reciprocal basis, columns = b_i
q_X   = B_rec * [0.0, 0.5, 0.5]

@printf("\nX-point: q_X = (%.5f, %.5f, %.5f) Å⁻¹  [= (2π/a, 0, 0)]\n", q_X...)
@printf("  Check commensurate: q_X · a_conv = %.4f × π  (should be 2)\n",
        dot(q_X, [a0, 0.0, 0.0]) / π)

# ─────────────────────────────────────────────────────────────────────────────
#  Dynamical matrix at X  →  eigenvalues + eigenvectors
# ─────────────────────────────────────────────────────────────────────────────
println("\nBuilding D(q_X) from force constants …")
Dq_X   = dynamical_matrix_from_fc(fc_data, q_X)
F      = eigen(Hermitian(Dq_X))
ω2_X   = real.(F.values)                                   # [eV/Å²/amu]
vecs_X = F.vectors                                         # columns = eigenvectors
freqs_X = sign.(ω2_X) .* sqrt.(abs.(ω2_X)) .* FREQ_THz    # THz

println("\n  X-point phonon modes (from D(q_X)):")
println("  ┌──────┬─────────────────────┬──────────────┐")
println("  │ Mode │  ω² [eV/Å²/amu]     │  ω [THz]     │")
println("  ├──────┼─────────────────────┼──────────────┤")
for μ in 1:3
    @printf("  │  %d   │  %+.8f       │  %+.6f  │\n",
            μ, ω2_X[μ], freqs_X[μ])
end
println("  └──────┴─────────────────────┴──────────────┘")

# ─────────────────────────────────────────────────────────────────────────────
#  Atomic displacements  (mode μ, amplitude d)
#
#  The eigenvectors of the 1-atom primitive D(q_X) are 3D complex vectors.
#  At the FCC X-point, D(q_X) is real (time-reversal + inversion: -q_X ~ q_X),
#  so the eigenvectors can be chosen real.
#
#  Displacement of atom i in the conventional cell for mode μ, amplitude d:
#    u_i(d) = Re[ e_μ · exp(i q_X · r_i) ] / √m  ×  d    [Å]
#
#  For the conventional cell q_X · r_i ∈ {0, π}:
#    atoms at x=0  →  cos(0) = +1   →  u_i = +e_μ/√m · d
#    atoms at x=a/2 →  cos(π) = -1  →  u_i = −e_μ/√m · d
# ─────────────────────────────────────────────────────────────────────────────
m_Al   = ustrip(sys_conv[1].mass)   # amu (same for all Al atoms)
r_conv = [SVector{3,Float64}(ustrip.(sys_conv[i].position)) for i in 1:N_prim_cells]

println("\n  Conventional-cell atom positions and Bloch phases at q_X:")
println("  ┌──────┬─────────────────────────────┬───────────┐")
println("  │ Atom │  r [Å]                      │ cos(q·r)  │")
println("  ├──────┼─────────────────────────────┼───────────┤")
for i in 1:N_prim_cells
    phi = dot(q_X, r_conv[i])
    @printf("  │  %d   │  (%5.3f, %5.3f, %5.3f)    │  %+.4f   │\n",
            i, r_conv[i]..., cos(phi))
end
println("  └──────┴─────────────────────────────┴───────────┘")

println("\n  Displacement vectors per mode (d = 1 Å√amu):")
for μ in 1:3
    e_μ = vecs_X[:, μ]
    @printf("\n  Mode %d  [ω = %.4f THz]:\n", μ, freqs_X[μ])
    @printf("    eigenvector e_μ = (%.6f, %.6f, %.6f)  + i(%.6f, %.6f, %.6f)\n",
            real.(e_μ)..., imag.(e_μ)...)
    for i in 1:N_prim_cells
        u_i = real.(e_μ .* exp(im * dot(q_X, r_conv[i]))) ./ sqrt(m_Al)
        @printf("    Atom %d: u = (%+.6f, %+.6f, %+.6f)  [|u| = %.5f Å]\n",
                i, u_i..., norm(u_i))
    end
end

# ─────────────────────────────────────────────────────────────────────────────
#  Design matrix as a function of displacement amplitude
#
#  B(d; μ) : ℝ → ℝ^{N_params}
#    = potential_energy_basis of the conventional cell displaced by mode μ × d.
#
#  Closure captures: sys_conv, r_conv, disps_μ (Float64 displacement directions).
#  Only d enters as a Dual number — ACEpotentials.Models.potential_energy_basis
#  propagates it through the neighbour-list computation.
# ─────────────────────────────────────────────────────────────────────────────
function make_displaced_design(sys0, r0, displacements, mdl)
    # Precompute cell vectors once (Float64, outside ForwardDiff) so the
    # closure never calls sys0.cell.cell_vectors with a Dual argument.
    # eachindex(FlexibleSystem) in AtomsBase 0.5 returns property-key Symbols,
    # so we explicitly use 1:length(sys0) for integer atom indexing.
    _cv   = sys0.cell.cell_vectors
    _nat  = length(sys0)
    _spec = [sys0[i].species for i in 1:_nat]
    function B(d)
        atoms = Vector{Atom}(undef, _nat)
        for i in 1:_nat
            rnew = r0[i] .+ d .* displacements[i]
            atoms[i] = Atom(_spec[i], collect(rnew .* u"Å"), missing)
        end
        sys_new = periodic_system(atoms, _cv)
        return ustrip.(ACEpotentials.Models.potential_energy_basis(sys_new, mdl))
    end
    return B
end

# ─────────────────────────────────────────────────────────────────────────────
#  Curvature operator C_μ = d²B(d; μ)/dd²|_{d=0}
#
#  Strategy:
#    1.  dB/dd at ±h  via ForwardDiff.derivative (scalar → vector, works with
#        ACEpotentials since positions are Dual-compatible).
#    2.  d²B/dd²      via central finite differences on step 1.
#        This avoids nested dual numbers in the ACEpotentials call stack.
#
#  Comparison:
#    ω²_AD := (C_μ · θ) / N_cells   ≈   ω²_dyn      [eV/Å²/amu]
#
#  The N_cells = 4 factor appears because potential_energy_basis sums over all
#  N_cells primitive cells in the conventional supercell (extensive quantity).
# ─────────────────────────────────────────────────────────────────────────────
h_fd = 1e-4   # finite-difference step for the outer second derivative [Å]

println("\n\n═══════════════════════════════════════════════════════════════════")
println("  Comparison: ω² from dynamical matrix  vs  AD curvature operator")
println("═══════════════════════════════════════════════════════════════════\n")

for μ in 1:3
    e_μ = vecs_X[:, μ]

    # Precompute displacement directions for all atoms in the conventional cell.
    # These are Float64 — only d is a Dual when ForwardDiff runs.
    disps_μ = SVector{3,Float64}[
        real.(e_μ .* exp(im * dot(q_X, r_conv[i]))) ./ sqrt(m_Al)
        for i in 1:N_prim_cells
    ]

    B_func = make_displaced_design(sys_conv, r_conv, disps_μ, model)

    # First derivative  dB/dd  via ForwardDiff
    dB(d) = ForwardDiff.derivative(B_func, d)

    # Second derivative (curvature operator, length N_params):
    #   C_μ[k] = d²B_k/dd²|_{d=0}  ≈  (dB_k(+h) − dB_k(−h)) / (2h)
    print("  Mode $μ: computing curvature operator … ")
    C_μ = (dB(+h_fd) .- dB(-h_fd)) ./ (2h_fd)
    println("done.  ||C_μ|| = $(round(norm(C_μ), sigdigits=5))")

    # ω² from the AD route
    ω2_AD = dot(C_μ, θ) / N_prim_cells

    # Reference ω² from the dynamical matrix
    ω2_dyn = ω2_X[μ]

    tag = ω2_dyn < -1e-8 ? "  ← IMAGINARY" : ""

    println()
    @printf("  ┌─────────────────────────────────────────────────────────\n")
    @printf("  │  Mode %d\n", μ)
    @printf("  │  Dynamical matrix:  ω²_dyn = %+.8f eV/Å²/amu  →  ω = %+.5f THz%s\n",
            ω2_dyn, freqs_X[μ], tag)
    @printf("  │  AD curvature:      C_μ·θ / %d = %+.8f eV/Å²/amu\n",
            N_prim_cells, ω2_AD)
    @printf("  │  Raw C_μ·θ         (extensive) = %+.8f eV/Å²/amu\n",
            dot(C_μ, θ))
    if abs(ω2_dyn) > 1e-10
        @printf("  │  Relative error    = %+.3e\n", (ω2_AD - ω2_dyn) / abs(ω2_dyn))
    else
        @printf("  │  Absolute error    = %+.3e  (ω² ≈ 0)\n", ω2_AD - ω2_dyn)
    end
    @printf("  └─────────────────────────────────────────────────────────\n\n")
end

# ─────────────────────────────────────────────────────────────────────────────
#  Summary: the curvature operator as a 1×N_params row vector
#
#  C_μ is the exact linear operator such that:
#    d²E_conv / dd² = C_μ · θ     (for any θ, not just the fitted one)
#
#  Dividing by N_cells gives the per-primitive-cell curvature = ω².
#  This is the Hessian contribution of phonon mode μ at q_X expressed
#  directly in the ACE basis — useful for constrained fitting / POPS.
# ─────────────────────────────────────────────────────────────────────────────
println("═══════════════════════════════════════════════════════════════════")
println("Done.")

# ─────────────────────────────────────────────────────────────────────────────
#  POPS ensemble: X-point ω² for every committee member
#
#  Assumes `pops_samples` is already defined as a Vector{Vector{Float64}},
#  each element being a full parameter vector θ_i of length N_params.
#
#  For each member i:
#    1. Set model → θ_i, relax lattice constant → a_i.
#    2. Build the conventional cell at a_i; recompute r_conv_i.
#       The X-point displacement directions (disps_μ) are the SAME for every
#       a_i: q_X · r_j ∈ {0, π} for all atoms regardless of a_i (FCC geometry),
#       so the Bloch phases remain ±1 and e_μ is symmetry-fixed for the X-point.
#    3. Compute C_μ_i = d²B_i/dd²  (curvature of design matrix; θ-independent).
#    4. ω²_μ_i = C_μ_i · θ_i / N_prim_cells.
#    5. Record sign → imaginary if ω²_μ_i < 0.
#
#  The model is restored to the original mean parameters after the loop.
# ─────────────────────────────────────────────────────────────────────────────

N_committee   = length(pops_samples)
ω2_ensemble   = Matrix{Float64}(undef, 3, N_committee)   # ω²[mode, member]
a_ensemble    = Vector{Float64}(undef, N_committee)       # relaxed a_i per member

println("\n\n═══════════════════════════════════════════════════════════════════")
println("  POPS ensemble: X-point ω² for each committee member")
@printf( "  N_committee = %d  |  nthreads = %d\n", N_committee, Threads.nthreads())
println("═══════════════════════════════════════════════════════════════════\n")

# Displacement directions are a_i-independent for the FCC X-point (phases = ±1),
# so precompute once from the mean-model eigenvectors.
const_disps = Vector{Vector{SVector{3,Float64}}}(undef, 3)
for μ in 1:3
    e_μ = vecs_X[:, μ]
    const_disps[μ] = SVector{3,Float64}[
        real.(e_μ .* exp(im * dot(q_X, r_conv[j]))) ./ sqrt(m_Al)
        for j in 1:N_prim_cells
    ]
end

# ── Phase 1: lattice relaxation (serial) ─────────────────────────────────────
# relax_lattice_constant uses a geometry optimizer (Optim CG) that is not
# thread-safe: concurrent calls share internal line-search state and can cause
# the cell matrix to diverge (InexactError: Int64(~1e58) in NeighbourLists).
# Relaxation is cheap compared to the curvature computation, so run it serially.
println("  Phase 1 / 2 — lattice relaxation (serial) …")
lc_file = joinpath(result.dir, "results", "lattice_constants.csv")
if isfile(lc_file)
    a_ensemble .= vec(readdlm(lc_file, ',', Float64))
    @printf("  Loaded %d lattice constants from cache: %s\n\n", N_committee, lc_file)
else
    t_relax = @elapsed for i in 1:N_committee
        ACEpotentials.Models.set_linear_parameters!(model, pops_samples[i])
        a_ensemble[i] = ACEWorkflow.relax_lattice_constant(model, :Al)
        ACEpotentials.Models.set_linear_parameters!(model, θ)   # restore mean params
        @printf("    Member %3d: a = %.5f Å\n", i, a_ensemble[i])
    end
    mkpath(dirname(lc_file))
    writedlm(lc_file, a_ensemble, ',')
    @printf("  Relaxation done in %.2f s  →  saved to %s\n\n", t_relax, lc_file)
end

# ── Phase 2: curvature computation (threaded) ─────────────────────────────────
# potential_energy_basis is θ-independent (basis only, no linear params needed).
# One model copy per thread avoids races on any internal basis-evaluation buffers.
println("  Phase 2 / 2 — curvature computation ($(Threads.nthreads()) threads) …")
models_thread = [deepcopy(model) for _ in 1:Threads.nthreads()]
_print_lock = ReentrantLock()

t_curv = @elapsed Threads.@threads for i in 1:N_committee
    tid   = Threads.threadid()
    mdl_t = models_thread[tid]
    θ_i   = pops_samples[i]
    a_i   = a_ensemble[i]

    t_i = @elapsed begin
    sys_conv_i = bulk(:Al; a=a_i * u"Å", cubic=true)
    r_conv_i   = [SVector{3,Float64}(ustrip.(sys_conv_i[j].position)) for j in 1:N_prim_cells]

    for μ in 1:3
        B_i     = make_displaced_design(sys_conv_i, r_conv_i, const_disps[μ], mdl_t)
        dB_i(d) = ForwardDiff.derivative(B_i, d)
        C_μ_i   = (dB_i(+h_fd) .- dB_i(-h_fd)) ./ (2h_fd)
        ω2_ensemble[μ, i] = dot(C_μ_i, θ_i) / N_prim_cells
    end
    end # @elapsed

    lock(_print_lock) do
        @printf("  Member %3d [t%d]: a = %.5f Å  |  ω² = [%+.5f, %+.5f, %+.5f] eV/Å²/amu  (%5.2f s)%s\n",
                i, tid, a_i,
                ω2_ensemble[1,i], ω2_ensemble[2,i], ω2_ensemble[3,i],
                t_i,
                any(ω2_ensemble[:, i] .< 0) ? "  ← imaginary" : "")
    end
end
@printf("\n  Curvature time: %.2f s  (%.2f s / member)\n", t_curv, t_curv / N_committee)

# Save ensemble frequencies to CSV
let freq_file = joinpath(result.dir, "results", "xpoint_ensemble_frequencies.csv")
    mkpath(dirname(freq_file))
    open(freq_file, "w") do io
        println(io, "# X-point phonon frequencies for POPS ensemble")
        println(io, "# Generated: ", Dates.now())
        println(io, "# member,a_Ang,freq_T1_THz,freq_T2_THz,freq_L_THz,omega2_T1_eVA2amu,omega2_T2_eVA2amu,omega2_L_eVA2amu")
        for i in 1:N_committee
            fi = sign.(ω2_ensemble[:, i]) .* sqrt.(abs.(ω2_ensemble[:, i])) .* FREQ_THz
            @printf(io, "%d,%.6f,%.6f,%.6f,%.6f,%.10f,%.10f,%.10f\n",
                    i, a_ensemble[i], fi[1], fi[2], fi[3],
                    ω2_ensemble[1,i], ω2_ensemble[2,i], ω2_ensemble[3,i])
        end
    end
    @printf("  Ensemble frequencies written to: %s\n", freq_file)
end

# ── Mode-crossing check ───────────────────────────────────────────────────────
# A crossing occurs when the longitudinal mode (row 1) has a lower ω² than
# either transverse mode (rows 2 or 3), i.e. the L branch dips below a T branch.
let crossing_file = joinpath(result.dir, "results", "mode_crossings.csv")
    crossing_indices = Int[]
    for i in 1:N_committee
        ω2_L  = ω2_ensemble[3, i]
        ω2_T1 = ω2_ensemble[1, i]
        ω2_T2 = ω2_ensemble[2, i]
        if ω2_L < ω2_T1 || ω2_L < ω2_T2
            push!(crossing_indices, i)
        end
    end
    mkpath(dirname(crossing_file))
    writedlm(crossing_file, crossing_indices, ',')
    @printf("  Mode crossings (L < T): %d / %d members  →  saved to %s\n",
            length(crossing_indices), N_committee, crossing_file)
    if 0 < length(crossing_indices) <= 20
        println("    Crossing members: ", join(crossing_indices, ", "))
    end
end

# 5. Summary
println("\n  ── Imaginary-mode summary ──────────────────────────────────────")
for μ in 1:3
    imag_idx = findall(ω2_ensemble[μ, :] .< 0)
    @printf("  Mode %d (%s): %d / %d members imaginary",
            μ, μ < 3 ? "transverse" : "longitudinal",
            length(imag_idx), N_committee)
    if 0 < length(imag_idx) <= 20
        print("  (members: ", join(imag_idx, ", "), ")")
    end
    println()
end

println("\n═══════════════════════════════════════════════════════════════════")
println("Ensemble done.")

# ─────────────────────────────────────────────────────────────────────────────
#  Verification: for members with any imaginary mode, cross-check ω² against
#  the force-constant / dynamical-matrix route (same as the mean-model check).
# ─────────────────────────────────────────────────────────────────────────────
imag_members = sort(unique(vcat([findall(ω2_ensemble[μ, :] .< 0) for μ in 1:3]...)))

# Save indices of imaginary members to file
let outfile = joinpath(@__DIR__, "imaginary_pops_indices.txt")
    open(outfile, "w") do io
        println(io, "# Indices of POPS committee members with at least one imaginary X-point mode")
        println(io, "# Format: one index per line (1-based)")
        println(io, "# Generated: ", Dates.now())
        for i in imag_members
            println(io, i)
        end
    end
    @printf("\n  Imaginary-member indices written to: %s  (%d entries)\n",
            outfile, length(imag_members))
end

if isempty(imag_members)
    println("\nNo imaginary modes found in ensemble — no verification needed.")
else
    println("\n\n═══════════════════════════════════════════════════════════════════")
    println("  Verification: D(q_X) eigendecomposition for imaginary members")
    println("═══════════════════════════════════════════════════════════════════\n")

    for i in imag_members
        θ_i = pops_samples[i]
        a_i = a_ensemble[i]

        N_cell = Integer(ceil(12.0 / ustrip(a_i)))

        # Build systems and force constants for this member
        ACEpotentials.Models.set_linear_parameters!(model, θ_i)
        sys_prim_i  = bulk(:Al; a=a_i * u"Å")
        sys_super_i = bulk(:Al; a=a_i * u"Å", cubic=true) * (N_cell, N_cell, N_cell)
        @printf("  Member %d (a = %.5f Å): computing force constants …", i, a_i)
        fc_i   = precompute_force_constants(sys_prim_i, sys_super_i, model)
        ACEpotentials.Models.set_linear_parameters!(model, θ)   # restore
        println(" done.")

        q_X_i  = 2π / a_i .* SVector{3,Float64}(1.0, 0.0, 0.0)
        Dq_i   = dynamical_matrix_from_fc(fc_i, q_X_i)
        ω2_dyn_i = real.(eigen(Hermitian(Dq_i)).values)

        println()
        @printf("  ┌─────────────────────────────────────────────────────────\n")
        @printf("  │  Member %d\n", i)
        @printf("  │  %-12s  %+20s  %+20s  %+12s\n",
                "Mode", "ω²_dyn [eV/Å²/amu]", "ω²_AD  [eV/Å²/amu]", "rel. error")
        for μ in 1:3
            rel = abs(ω2_dyn_i[μ]) > 1e-10 ?
                  (ω2_ensemble[4-μ,i] - ω2_dyn_i[μ]) / abs(ω2_dyn_i[μ]) : NaN
            tag = ω2_dyn_i[μ] < -1e-8 ? " ←IMAG" : ""
            @printf("  │  Mode %d        %+18.8f    %+18.8f    %+.3e%s\n",
                    μ, ω2_dyn_i[μ], ω2_ensemble[4-μ,i], rel, tag)
        end
        @printf("  └─────────────────────────────────────────────────────────\n\n")
    end

    println("═══════════════════════════════════════════════════════════════════")
    println("Verification done.")
end
