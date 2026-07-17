using LinearAlgebra, Statistics, DelimitedFiles, Printf
using SparseArrays, StaticArrays
using Test
using ACEpotentials, ForwardDiff, Unitful, ACEWorkflow
using OSQP

element    = :Al

# ── Load model ────────────────────────────────────────────────────────────────
result     = load_model(element, 14, 4, 6, 2; dataset_name="subset_20_percent")
model      = result.model
A          = result.A
Y          = result.Y
P          = result.P
W          = result.W
lin_params = result.lin_params

Ap = Diagonal(W) * A / P
Yw = W .* Y

# ── Equilibrium lattice constant and Hessian basis ────────────────────────────
println("Relaxing equilibrium lattice constant ...")
a_eq = ACEWorkflow.relax_lattice_constant(model, element)
@printf("a_eq = %.6f Å\n", a_eq)

println("Computing strain Hessian basis at a_eq ...")
C, H_eq, _ = ACEWorkflow.Elasticity.strain_hessian_GPa(model, element; a=a_eq)

# ── Lattice basis derivative for equilibrium constraint ───────────────────────
function lattice_basis(a_val)
    sys = ACEWorkflow.Elasticity.reference_system(element; a=a_val)
    ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(sys, model))
end

println("Computing b′ ...")
b_prime = ForwardDiff.derivative(lattice_basis, a_eq)

println("Computing b″ ...")
b_double_prime = ForwardDiff.derivative(a -> ForwardDiff.derivative(lattice_basis, a), a_eq)

# ── Constrained ridge regression ──────────────────────────────────────────────
function constrained_ridge_regression(X_train, Y_train, Gamma, constraint_matrix, constraint_bounds; lambda = 1.0 / size(X_train, 1))
    H = (X_train' * X_train .+ (lambda .* Gamma' * Gamma))
    b = - X_train' * Y_train
    osqp_model = OSQP.Model()
    OSQP.setup!(osqp_model; P=sparse(H), q=b, A=sparse(constraint_matrix / Gamma), l=constraint_bounds[1], u=constraint_bounds[2],
                max_iter=5_000_000_000, check_termination=1_000, verbose=true, eps_abs=1e-4, eps_rel=1e-4)
    results = OSQP.solve!(osqp_model)
    return Gamma \ results.x
end

# ── Constraint vectors ────────────────────────────────────────────────────────
constraint_1 = H_eq[1,1,:]
C11_lower    = dot(constraint_1, lin_params)
C11_upper    = dot(constraint_1, lin_params)

constraint_2 = H_eq[1,2,:]
C12_lower    = dot(constraint_2, lin_params)
C12_upper    = dot(constraint_2, lin_params)

constraint_4 = H_eq[4,4,:]
C44_lower    = 0.1#dot(constraint_4, lin_params)
C44_upper    = Inf#dot(constraint_4, lin_params)

# L    = SMatrix{3,3,Float64}(ustrip.(ACEWorkflow.Elasticity.lattice_matrix(ACEWorkflow.Elasticity.reference_system(element; a=a_eq).cell.cell_vectors)))
# V    = abs(det(L))
# conversion_factor = 160.21766208 / V

# constraint_11 = H_eq[1,1,:]
# C11_lower     = 116.3 / conversion_factor
# C11_upper     = 116.3 / conversion_factor

# constraint_12 = H_eq[1,2,:]
# C12_lower     = 64.8  / conversion_factor
# C12_upper     = 64.8  / conversion_factor

# constraint_44 = H_eq[4,4,:]
# C44_lower     = 30.9  / conversion_factor
# C44_upper     = 30.9  / conversion_factor

# Born stability: C11 + 2*C12 > 0
C11_plus_2C12_constraint = constraint_1 .+ 2 .* constraint_2
C11_plus_2C12_lower      = 0.1
C11_plus_2C12_upper      = Inf

C12_less_than_C11_constraint = constraint_1 .- constraint_2
C12_less_than_C11_lower      = 1.0
C12_less_than_C11_upper      = Inf

# Lattice constant constraint: b′(a_eq) · θ = 0
# Ensures the correction δθ does not shift the equilibrium lattice constant.
# b_prime must be precomputed via del_lattice_constant_design or ForwardDiff on lattice_basis.
# The RHS is -b_prime · lin_params ≈ 0 (nominal is already at equilibrium).
lattice_eq_constraint = b_prime                          # n_params-vector
lattice_eq_rhs        = -dot(b_prime, lin_params)        # ≈ 0
lattice_eq_lower      = 0.0
lattice_eq_upper      = 0.0

# Lattice curvature constraint: b″(a_eq) · θ > 0
# E(a) ≈ E(a_eq) + ½·(b″·θ)·(a-a_eq)² + ... near equilibrium, so this is
# exactly the condition that a_eq is a minimum (not a maximum/saddle) of
# E(a) — equivalent to requiring the bulk modulus B = (C11+2C12)/3 > 0.
# Bounded away from 0 (not just > 0) for the same numerical-safety reason
# as the other strict-inequality Born constraints above.
lattice_curvature_constraint = b_double_prime            # n_params-vector
lattice_curvature_lower      = 1e-9
lattice_curvature_upper      = Inf

all_constraints = vcat(constraint_1',
                       constraint_4',
                       C12_less_than_C11_constraint',
                       C11_plus_2C12_constraint',
                       lattice_eq_constraint',
                       lattice_curvature_constraint')
lower_bounds    = [C11_lower, C44_lower,
                   C12_less_than_C11_lower, C11_plus_2C12_lower,
                   lattice_eq_lower, lattice_curvature_lower]
upper_bounds    = [C11_upper, C44_upper,
                   C12_less_than_C11_upper, C11_plus_2C12_upper,
                   lattice_eq_upper, lattice_curvature_upper]
# all_constraints = vcat(constraint_4')
# lower_bounds    = [C44_lower]
# upper_bounds    = [C44_upper]

constraints = (lower_bounds, upper_bounds)

constrained_ridge_teta = constrained_ridge_regression(Ap, Yw, P, all_constraints, constraints)

using OSQP, ACEWorkflow

function constrained_pops(X_train, Y_train, Gamma, constraint_matrix, constraint_bounds; members_to_constrain=1:length(Y_train))
    constrained_pops_parameters = zeros(length(members_to_constrain), size(X_train, 2))
    H = (X_train' * X_train .+ (1.0 / (size(X_train, 1)) .* Gamma' * Gamma))
    b = - X_train' * Y_train

    model = OSQP.Model()

    for idx in 1:length(members_to_constrain)
        i = members_to_constrain[idx]
        A_full = vcat(X_train[i,:]', constraint_matrix)
        l_full = vcat([Y_train[i]], constraint_bounds[1])
        u_full = vcat([Y_train[i]], constraint_bounds[2])
        A_sparse = sparse(A_full / Gamma)

        OSQP.setup!(model; P=sparse(H), q=b, A=A_sparse, l=l_full, u=u_full,
                    max_iter=500_000, check_termination=10, verbose=false, eps_abs=5e-4, eps_rel=5e-4)

        if (mod(idx, 100) == 0)
            println(idx)
        end
        results = OSQP.solve!(model)
        constrained_pops_parameters[idx,:] = Gamma \ results.x
    end
    return constrained_pops_parameters
end

constrained_pops_delta = constrained_pops(Ap, Yw, P, all_constraints, constraints)
# using Random
# random_selections = rand(1:size(constrained_pops_delta, 1), 10)
# sub_samples = constrained_pops_delta[random_selections, :]
pops_eig, pops_bound = hypercube(constrained_pops_delta; percentile_clipping=0.4)

pops_samples, _ = sample_hypercube(pops_eig, pops_bound, zeros(length(constrained_ridge_teta)))
writedlm("$(result.dir)/constrained_pops_samples.csv", pops_samples, ',')
pops_samples    = [pops_samples[:,i] for i=1:size(pops_samples,2)]

ACEpotentials.Models.set_linear_parameters!(model, constrained_ridge_teta)
a_eq = ACEWorkflow.relax_lattice_constant(model, element)

# ── eV → GPa conversion ───────────────────────────────────────────────────────
sys0      = ACEWorkflow.Elasticity.reference_system(element; a=a_eq)
L0        = SMatrix{3,3,Float64}(ustrip.(ACEWorkflow.Elasticity.lattice_matrix(sys0.cell.cell_vectors)))
V         = abs(det(L0))
eV_to_GPa = 160.2176621 / V

# ── Elastic constants from constrained solution ───────────────────────────────
C11_c = dot(H_eq[1,1,:], constrained_ridge_teta) * eV_to_GPa
C12_c = dot(H_eq[1,2,:], constrained_ridge_teta) * eV_to_GPa
C44_c = dot(H_eq[4,4,:], constrained_ridge_teta) * eV_to_GPa

C11_nom = C11_lower * eV_to_GPa   # nominal (from lin_params)
C12_nom = C12_lower * eV_to_GPa
C44_nom = dot(constraint_4, lin_params) * eV_to_GPa

println()
println("── Elastic constants ───────────────────────────────────────")
@printf("           %10s  %10s\n", "nominal", "constrained")
@printf("  C11  =   %8.3f    %8.3f  GPa\n", C11_nom, C11_c)
@printf("  C12  =   %8.3f    %8.3f  GPa\n", C12_nom, C12_c)
@printf("  C44  =   %8.3f    %8.3f  GPa\n", C44_nom, C44_c)
@printf("  C11-C12  = %8.3f  GPa  (Born: must be > 0)\n", C11_c - C12_c)
@printf("  C11+2C12 = %8.3f  GPa  (Born: must be > 0)\n", C11_c + 2*C12_c)

println()
println("── Constraint checks ───────────────────────────────────────")

# 1. C11 equality
@printf("  C11 equality:          |Δ| = %.3e GPa", abs(C11_c - C11_nom))
@test abs(C11_c - C11_nom) < 0.1
println(abs(C11_c - C11_nom) < 0.1 ? "  ✓" : "  ✗")

# # 2. C44 equality
# @printf("  C44 equality:          |Δ| = %.3e GPa", abs(C44_c - C44_nom))
# @test abs(C44_c - C44_nom) < 0.1
# println(abs(C44_c - C44_nom) < 0.1 ? "  ✓" : "  ✗")

# 3. Born: C11 > C12
@printf("  Born C11 - C12:        %.3f GPa", C11_c - C12_c)
@test C11_c - C12_c > 0
println(C11_c - C12_c > 0 ? "  ✓" : "  ✗")

# 4. Born: C11 + 2*C12 > 0
@printf("  Born C11 + 2*C12:      %.3f GPa", C11_c + 2*C12_c)
@test C11_c + 2*C12_c > 0
println(C11_c + 2*C12_c > 0 ? "  ✓" : "  ✗")

# 5. Lattice constant: b′·c = 0  (constraint enforces (b_prime/P)·θ = b_prime·c = 0)
lattice_residual = dot(b_prime, constrained_ridge_teta)
@printf("  Lattice b′·c:          %.3e eV/Å", lattice_residual)
@test abs(lattice_residual) < 1e-4
println(abs(lattice_residual) < 1e-4 ? "  ✓" : "  ✗")

# 6. Lattice curvature: b″·c > 0  (a_eq is a minimum, not a maximum, of E(a))
lattice_curvature = dot(b_double_prime, constrained_ridge_teta)
@printf("  Lattice b″·c (curvature): %.3e eV/Å²", lattice_curvature)
@test lattice_curvature > 0
println(lattice_curvature > 0 ? "  ✓" : "  ✗")

# ACEpotentials.save_model(model, "$(result.dir)/constrained_model.json")

using StaticArrays, LinearAlgebra, AtomsBuilder
using AtomsCalculators, Unitful, AtomsBase
using ACEpotentials: potential_energy
using AtomsCalculatorsUtilities.SitePotentials: hessian
using Arpack: eigs
using CairoMakie, ForwardDiff
using Printf
using DelimitedFiles

# ─────────────────────────────────────────────────────────────────────────────
#  Physical constants & THz conversion factor
#
#  ω²  has units  eV / (Å² · amu)  after mass-weighting the Hessian.
#  Converting:
#    ω[rad/s] = √( ω²[eV/Å²/amu] × eV_to_J / (Å_to_m² × amu_to_kg) )
#    ω[THz]   = ω[rad/s] / (2π × 10¹²)
#  → FREQ_THz ≈ 15.633  THz / √(eV/Å²/amu)
# ─────────────────────────────────────────────────────────────────────────────
const _eV_J   = 1.602176634e-19     # J eV⁻¹
const _Å_m    = 1.0e-10             # m Å⁻¹
const _amu_kg = 1.66053906660e-27   # kg amu⁻¹
const FREQ_THz = sqrt(_eV_J / (_Å_m^2 * _amu_kg)) / (2π * 1e12)  # ≈ 15.633
const THz_to_meV = 4.135667696e-15 * 1e12 * 1e3  # h [eV·s] × 10¹² [Hz/THz] × 10³ [meV/eV] ≈ 4.136

# ─────────────────────────────────────────────────────────────────────────────
#  Dynamical matrix  (self-contained copy of the core of get_hessian_phonopy.jl)
#
#  Builds the 3Nat × 3Nat mass-weighted D(q) from the real-space Hessian of
#  `sys`, using the phonopy Bloch-transform convention.
#
#  q_cart : wavevector in Cartesian coordinates (Å⁻¹).
# ─────────────────────────────────────────────────────────────────────────────

function _lattice_mat(cell_vectors)
    a, b, c = cell_vectors
    SMatrix{3,3,eltype(a)}(hcat(a, b, c))
end

function _accumulate_dm!(dm, num_patom, num_satom, fc, q_frac, svecs, multi, p2s_map,
                         i, j, k)
    i_pair = k * num_patom + i
    m_pair = multi[i_pair + 1, 1]
    adrs   = multi[i_pair + 1, 2]

    cos_ph = 0.0; sin_ph = 0.0
    for l in 1:m_pair
        ph = 2π * sum(q_frac[m] * svecs[adrs + l, m] for m in 1:3)
        cos_ph += cos(ph) / m_pair
        sin_ph += sin(ph) / m_pair
    end
    for l in 1:3, m in 1:3
        fc_idx = p2s_map[i + 1] * num_satom * 9 + k * 9 + (l - 1) * 3 + m
        dm[l, m] += fc[fc_idx] * complex(cos_ph, sin_ph)
    end
end

"""
    precompute_force_constants(sys_prim, sys_super, model)

Compute the real-space force constants by calling `hessian` once on `sys_super`.
`sys_super` must be at least 2× the ACE cutoff in each direction so that Φ(i,j+R)
decays to zero before the supercell boundary — this ensures the minimum-image
vector gives the individual term Φ, not the PBC-folded sum Σ_R Φ.

Builds:
  - `p2s_map[i]` (0-based): which supercell atom is the origin-cell image of
    primitive atom i (used to index into H).
  - `s2p_map[k]` (0-based): which primitive atom class does supercell atom k
    belong to (used to select images for the Bloch sum).
  - `frac_super`: positions of all supercell atoms in PRIMITIVE fractional coords
    (not wrapped), so R = frac_super[k] − frac_prim[j] gives the correct lattice
    vector for the Bloch phase exp(2πi q_frac · R).

Returns a NamedTuple used by `dynamical_matrix_from_fc`.
"""
function precompute_force_constants(sys_prim, sys_super, model)
    H   = ustrip.(hessian(sys_super, model))
    Np  = length(sys_prim)
    Ns  = length(sys_super)

    L_prim   = SMatrix{3,3,Float64}(ustrip.(_lattice_mat(sys_prim.cell.cell_vectors)))
    L_super  = SMatrix{3,3,Float64}(ustrip.(_lattice_mat(sys_super.cell.cell_vectors)))
    Linv_p   = inv(L_prim)
    Linv_s   = inv(L_super)

    # Positions in PRIMITIVE fractional coordinates (NOT mod-wrapped)
    frac_prim  = [Linv_p * SVector{3,Float64}(ustrip.(sys_prim[i].position))  for i in 1:Np]
    frac_super = [Linv_p * SVector{3,Float64}(ustrip.(sys_super[k].position)) for k in 1:Ns]

    # s2p_map[k] (0-based): which primitive atom class does supercell atom k match?
    s2p_map = Vector{Int}(undef, Ns)
    for k in 1:Ns
        matched = false
        for i in 1:Np
            d = mod.(frac_super[k], 1.0) .- mod.(frac_prim[i], 1.0)
            d = d .- round.(d)
            if norm(d) < 1e-6
                s2p_map[k] = i - 1   # 0-based
                matched = true
                break
            end
        end
        matched || error("Supercell atom $k did not match any primitive atom")
    end

    # p2s_map[i] (0-based): supercell index of the origin-cell image of primitive atom i.
    # Choose the image whose Cartesian position is closest to sys_prim[i].
    p2s_map = Vector{Int}(undef, Np)
    for i in 1:Np
        candidates = findall(s2p_map .== (i - 1))
        r_ref = SVector{3,Float64}(ustrip.(sys_prim[i].position))
        dists = [norm(SVector{3,Float64}(ustrip.(sys_super[k].position)) .- r_ref)
                 for k in candidates]
        p2s_map[i] = candidates[argmin(dists)] - 1   # 0-based
    end

    masses = [ustrip(sys_prim[i].mass) for i in 1:Np]
    return (; H, frac_prim, frac_super, masses, L=L_prim, L_super, Linv_super=Linv_s, Np, Ns, p2s_map, s2p_map)
end

"""
    dynamical_matrix_from_fc(fc_data, q_cart) → Matrix{ComplexF64}

Build the (3Np × 3Np) primitive-cell dynamical matrix at Cartesian wavevector
`q_cart` (Å⁻¹) using the precomputed supercell force constants.

Bloch transform:  D_ij(q) = Σ_k [s2p_map[k]==j] Φ(p2s_map[i], k) exp(2πi q_frac·R_k) / √(m_i m_j)
where R_k = frac_super[k] − frac_prim[j]  (lattice vector, NOT wrapped).
"""
function dynamical_matrix_from_fc(fc_data, q_cart::AbstractVector{<:Real})
    (; H, frac_prim, frac_super, masses, L, L_super, Linv_super, Np, Ns, p2s_map, s2p_map) = fc_data

    Dq = zeros(ComplexF64, 3Np, 3Np)
    for i in 0:Np-1, j in 0:Np-1
        dm = zeros(ComplexF64, 3, 3)
        for k in 0:Ns-1
            s2p_map[k + 1] == j || continue
            # Displacement from origin-image of j to supercell atom k (Cartesian)
            R_cart = L * (frac_super[k + 1] - frac_prim[j + 1])
            # Apply minimum-image convention in SUPERCELL fractional coordinates.
            # The hessian is computed with PBC, so H[origin_i, k] reflects the
            # interaction via the nearest periodic image of k — which may cross
            # the supercell boundary.  Without this wrapping, atoms near the far
            # edge of [0, L) get phase q·L instead of q·(−δ), causing oscillations.
            R_sfrac = Linv_super * R_cart
            R_sfrac = R_sfrac .- round.(R_sfrac)   # wrap to (−½, ½]
            R_mi    = L_super * R_sfrac
            phase   = dot(q_cart, R_mi)
            eph     = exp(im * phase)
            for α in 1:3, β in 1:3
                dm[α, β] += H[3 * p2s_map[i + 1] + α, 3k + β] * eph
            end
        end
        ms = sqrt(masses[i + 1] * masses[j + 1])
        Dq[3i+1:3i+3, 3j+1:3j+3] .= dm ./ ms
    end

    for i in 1:3Np, j in i:3Np
        Dq[i, j] = (Dq[i, j] + conj(Dq[j, i])) / 2
        Dq[j, i] = conj(Dq[i, j])
    end
    return Dq
end

# Convenience wrapper (single q, computes Hessian fresh — use for one-offs only)
dynamical_matrix_ace(sys_prim, sys_super, model, q_cart) =
    dynamical_matrix_from_fc(precompute_force_constants(sys_prim, sys_super, model), q_cart)

# ─────────────────────────────────────────────────────────────────────────────
#  DFTK-style frequency extraction
#
#  Mirrors `_phonon_modes` in DFTK.jl/phonon.jl:
#    signs = sign.(real(eigenvalues))
#    frequencies = signs .* sqrt.(abs.(real(eigenvalues)))
#  but in THz instead of atomic units.
# ─────────────────────────────────────────────────────────────────────────────
"""
    eigenvalues_to_freq_THz(ω2::AbstractVector) → Vector{Float64}

Convert mass-weighted dynamical-matrix eigenvalues ω² [eV/Å²/amu] to
frequencies [THz] using the DFTK sign convention:
  ω = sign(ω²) · √|ω²| · FREQ_THz
Imaginary (unstable) modes have negative frequency.
"""
eigenvalues_to_freq_THz(ω2) = sign.(ω2) .* sqrt.(abs.(ω2)) .* FREQ_THz

"""
    dq_eigensystem(Dq) → (freqs::Vector{Float64}, vecs::Matrix{ComplexF64})

Diagonalise Hermitian `Dq` and return
  - `freqs`: frequencies in THz (DFTK sign convention: imaginary → negative)
  - `vecs`:  eigenvectors as columns, in the same sorted order as `freqs`.
"""
function dq_eigensystem(Dq::Matrix{ComplexF64})
    F    = eigen(Hermitian(Dq))
    ω2   = real.(F.values)          # sorted ascending by LinearAlgebra.eigen
    vecs = F.vectors                # columns = eigenvectors
    return eigenvalues_to_freq_THz(ω2), vecs
end

# ─────────────────────────────────────────────────────────────────────────────
#  FCC Brillouin-zone path
#
#  Path: Γ → X → U → L → Γ → K  (all segments connected, no discontinuity)
#
#  High-symmetry points in Cartesian coordinates (Å⁻¹) for the PRIMITIVE
#  FCC cell (1 atom), expressed in units of 2π/a:
#    Γ = (0,    0,    0  )
#    X = (0,    1,    0  )   zone-face centre
#    U = (1/4,  1,    1/4)   X-face edge (U and K are the same irreducible
#    K = (3/4,  3/4,  0  )   point approached from different BZ faces)
#    L = (1/2,  1/2,  1/2)   hexagonal face centre
# ─────────────────────────────────────────────────────────────────────────────
"""
    fcc_band_path(a; N_per_seg=30)

Return `(q_list, x_coords, x_ticks, tick_labels, seg_starts)` for the FCC path
Γ → X → U → L → Γ → K.
`a` is the conventional lattice constant (Å); `N_per_seg` points per segment.
`seg_starts[iq]` is true only for the first q-point (used by branch tracking).
"""
function fcc_band_path(L; N_per_seg=30)
    B = 2π * inv(transpose(L))   # reciprocal lattice vectors as columns

    frac = (
        Γ = [0.0,   0.0,   0.0],
        X = [0.0,   0.5,   0.5],
        U = [0.25,  0.625, 0.625],
        Lp = [0.5,  0.5,   0.5],
        K = [0.375, 0.75,  0.375],
    )

    pts = (
        Γ = B * frac.Γ,
        X = B * frac.X,
        U = B * frac.U,
        L = B * frac.Lp,
        K = B * frac.K,
    )

    segs   = [(:Γ, :X), (:X, :U), (:U, :L), (:L, :Γ), (:Γ, :K)]
    labels = ["Γ", "X", "U", "L", "Γ", "K"]

    q_list     = Vector{Float64}[]
    x_vals     = Float64[]
    seg_starts = Bool[]
    x_ticks    = Float64[0.0]
    x = 0.0

    for (s, (l1, l2)) in enumerate(segs)
        q1 = getfield(pts, l1)
        q2 = getfield(pts, l2)

        seg_len = norm(q2 - q1)
        is_last = (s == length(segs))

        ts = is_last ? range(0.0, 1.0, N_per_seg + 1) :
                       range(0.0, 1.0, N_per_seg + 1)[1:end-1]

        for (ti, t) in enumerate(ts)
            q = q1 .+ t .* (q2 .- q1)
            push!(q_list, q)
            push!(x_vals, x + t * seg_len)
            push!(seg_starts, s == 1 && ti == 1)
        end

        x += seg_len
        push!(x_ticks, x)
    end

    return q_list, x_vals, x_ticks, labels, seg_starts
end

# ─────────────────────────────────────────────────────────────────────────────
#  Branch tracking
#
#  Eigenvalues sorted by magnitude can jump between physically distinct branches
#  whenever two bands are nearly degenerate (e.g. the two TA branches along
#  Γ→X are exactly degenerate by symmetry).  We track by eigenvector overlap:
#  at each q-point we assign mode j at q_n to the mode i at q_{n-1} that
#  maximises |⟨ψ_i(q_{n-1}) | ψ_j(q_n)⟩|², then reorder both freqs and vecs.
# ─────────────────────────────────────────────────────────────────────────────
function _track_branches!(freqs::Matrix{Float64}, vecs::Array{ComplexF64,3},
                          seg_starts::Vector{Bool})
    Nmodes, Nq = size(freqs)
    for iq in 2:Nq
        seg_starts[iq] && continue   # path restart — reset tracking
        # Overlap matrix O[i,j] = |⟨ψ_i(q_{n-1}) | ψ_j(q_n)⟩|²
        O = abs2.(vecs[:, :, iq-1]' * vecs[:, :, iq])
        # Greedy assignment: repeatedly pick (i,j) with largest overlap
        used_prev = falses(Nmodes)
        used_curr = falses(Nmodes)
        order = zeros(Int, Nmodes)   # order[i] = which column of curr goes to row i
        for _ in 1:Nmodes
            best = -Inf; bi = 0; bj = 0
            for i in 1:Nmodes, j in 1:Nmodes
                (used_prev[i] || used_curr[j]) && continue
                if O[i, j] > best
                    best = O[i, j]; bi = i; bj = j
                end
            end
            order[bi] = bj
            used_prev[bi] = true
            used_curr[bj] = true
        end
        freqs[:, iq]    = freqs[order, iq]        # reorder rows
        vecs[:, :, iq]  = vecs[:, order, iq]      # reorder eigenvector columns
    end
end

# ─────────────────────────────────────────────────────────────────────────────
#  Band-structure sweep
# ─────────────────────────────────────────────────────────────────────────────
"""
    compute_phonon_bands(sys_prim, sys_super, model, a; N_per_seg=30, n_modes=nothing)

Compute phonon frequencies along the standard FCC path Γ→X→U|K→Γ→L.
Eigenvalues are tracked across q-points to produce smooth, non-crossing bands.

Returns `(x_vals, freqs, x_ticks, labels)` where
`freqs` is a `Nmodes × Nq` matrix of frequencies in THz.
"""
function compute_phonon_bands(sys_prim, sys_super, model, a; N_per_seg=30, n_modes=nothing)
    # Hessian computed once on the supercell — gives individual Φ(i,j+R) terms
    print("  Precomputing force constants (Hessian of supercell) …")
    fc_data = precompute_force_constants(sys_prim, sys_super, model)
    println(" done.")
    q_list, x_vals, x_ticks, labels, seg_starts = fcc_band_path(fc_data.L; N_per_seg)
    Np     = length(sys_prim)
    Ntotal = 3 * Np
    Nout   = isnothing(n_modes) ? Ntotal : n_modes
    Nq     = length(q_list)
    freqs  = Matrix{Float64}(undef, Nout, Nq)
    evecs  = Array{ComplexF64,3}(undef, Ntotal, Nout, Nq)  # (dim, mode, q)

    mode_str = isnothing(n_modes) ? "all $Ntotal" : "lowest $n_modes of $Ntotal"
    println("  Primitive cell : $Np atoms  →  $Ntotal branches")
    println("  Supercell      : $(length(sys_super)) atoms (for force constants)")
    println("  Modes          : $mode_str")
    println("  q-path : Γ→X→U→L→Γ→K  ($Nq q-points, $N_per_seg per segment)")

    for (iq, q) in enumerate(q_list)
        iq % 10 == 0 && print("\r  Computing q-point $iq / $Nq …")
        Dq = dynamical_matrix_from_fc(fc_data, q)
        f, v = dq_eigensystem(Dq)
        freqs[:, iq]    = isnothing(n_modes) ? f      : f[1:Nout]
        evecs[:, :, iq] = isnothing(n_modes) ? v      : v[:, 1:Nout]
    end
    println("\r  Done. ($Nq q-points)                  ")

    # Reorder modes by eigenvector continuity (overlap) rather than eigenvalue sort order
    _track_branches!(freqs, evecs, seg_starts)

    return x_vals, freqs, x_ticks, labels
end

# ─────────────────────────────────────────────────────────────────────────────
#  Scatter-point plot
# ─────────────────────────────────────────────────────────────────────────────
"""
    plot_phonon_bands(x_vals, freqs, x_ticks, labels; title, linewidth)

Plot phonon frequencies as piecewise-linear bands (one line per mode row).
Branches that are entirely negative (imaginary) are drawn in red;
all others in blue.
"""
function plot_phonon_bands(x_vals, freqs, x_ticks, labels;
                           title="",
                           linewidth=1.5,
                           fig_size = (425, 263), 
                           fontsize = 12)
    Nmodes, Nq = size(freqs)

    fig = Figure(size=fig_size, fontsize=fontsize)
    ax  = Axis(fig[1, 1];
               xlabel       = "Wave vector",
               ylabel       = "Frequency (THz)",
               title        = title,
               xticks       = (x_ticks, labels),
               xgridvisible = false)

    for b in 1:Nmodes
        branch = freqs[b, :]
        color  = minimum(branch) < 0 ? RGBAf(0.8, 0.1, 0.1, 0.9) : RGBAf(0.2, 0.4, 0.7, 0.9)
        lines!(ax, x_vals, branch; color, linewidth)
    end

    hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=0.8)
    vlines!(ax, x_ticks; color=(:black, 0.3), linewidth=0.8)

    return fig
end

"""
    plot_phonon_energy(x_vals, freqs, x_ticks, labels; title, linewidth)

Same as `plot_phonon_bands` but with the y-axis in meV (ℏω) rather than THz.
Imaginary (unstable) modes are drawn in red; stable modes in blue.
"""
function plot_phonon_energy(x_vals, freqs, x_ticks, labels;
                            title="",
                            linewidth=1.5,
                            fig_size = (425, 263), 
                            fontsize = 10)
    energy = freqs .* (THz_to_meV / 1000)   # THz → eV
    Nmodes, Nq = size(energy)

    emin = floor(minimum(energy) / 0.01) * 0.01
    emax = ceil(maximum(energy)  / 0.01) * 0.01

    fig = Figure(size=fig_size,
                 fontsize=fontsize)
    ax  = Axis(fig[1, 1];
               xlabel       = "Wave vector",
               ylabel       = "Energy (eV)",
               title        = title,
               xticks       = (x_ticks, labels),
               yticks       = emin:0.01:emax,
               xgridvisible = false)

    for b in 1:Nmodes
        branch = energy[b, :]
        color  = minimum(branch) < 0 ? RGBAf(0.8, 0.1, 0.1, 0.9) : RGBAf(0.2, 0.4, 0.7, 0.9)
        lines!(ax, x_vals, branch; color, linewidth)
    end

    hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=0.8)
    vlines!(ax, x_ticks; color=(:black, 0.3), linewidth=0.8)

    return fig
end

"""
    plot_phonon_comparison(x_julia, freqs_julia, x_ticks, labels,
                           x_phonopy, energies_phonopy;
                           title="Phonon comparison — ACE Julia vs Phonopy",
                           linewidth=1.5)

Overlay phonon band structures from the Julia (ACE dynamical-matrix) and
Phonopy (ASE) implementations on the same axes, in eV.

Arguments:
  - `x_julia`           : x-coordinates from `compute_phonon_bands` (length Nq_j)
  - `freqs_julia`       : Nmodes × Nq_j matrix in THz from `compute_phonon_bands`
  - `x_ticks`, `labels` : high-symmetry tick positions and labels (from Julia path)
  - `x_phonopy`         : x-coordinates from Phonopy/ASE (length Nq_p), as a Vector
  - `energies_phonopy`  : Nq_p × Nbands matrix in eV (as written by np.savetxt)

Julia bands are drawn in blue (stable) / red (imaginary).
Phonopy bands are drawn in semi-transparent orange.
"""
function plot_phonon_comparison(x_julia, freqs_julia, x_ticks, labels,
                                x_phonopy, energies_phonopy;
                                title="Phonon comparison — ACE Julia vs Phonopy",
                                linewidth=1.5)
    energy_julia = freqs_julia .* (THz_to_meV / 1000)   # THz → eV
    Nmodes_j, Nq_j = size(energy_julia)

    # phonopy: rows = k-points, cols = bands  →  transpose for branch-wise iteration
    energy_ph = energies_phonopy'   # Nbands × Nq_p
    Nbands_p  = size(energy_ph, 1)
    x_ph      = vec(x_phonopy)

    emin = floor(min(minimum(energy_julia), minimum(energy_ph)) / 0.01) * 0.01
    emax = ceil( max(maximum(energy_julia), maximum(energy_ph)) / 0.01) * 0.01

    fig = Figure(size=(800, 520))
    ax  = Axis(fig[1, 1];
               xlabel       = "Wave vector",
               ylabel       = "Energy (eV)",
               title        = title,
               xticks       = (x_ticks, labels),
               yticks       = emin:0.01:emax,
               xgridvisible = false)

    # Phonopy bands (drawn first so Julia sits on top)
    for b in 1:Nbands_p
        lines!(ax, x_ph, energy_ph[b, :];
               color=RGBAf(0.85, 0.45, 0.05, 0.55), linewidth=linewidth)
    end

    # Julia bands
    for b in 1:Nmodes_j
        branch = energy_julia[b, :]
        color  = minimum(branch) < 0 ? RGBAf(0.8, 0.1, 0.1, 0.9) : RGBAf(0.2, 0.4, 0.7, 0.9)
        lines!(ax, x_julia, branch; color, linewidth)
    end

    hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=0.8)
    vlines!(ax, x_ticks; color=(:black, 0.3), linewidth=0.8)

    # Legend
    elem_julia   = LineElement(color=RGBAf(0.2, 0.4, 0.7, 0.9), linewidth=linewidth)
    elem_phonopy = LineElement(color=RGBAf(0.85, 0.45, 0.05, 0.55), linewidth=linewidth)
    Legend(fig[1, 2], [elem_julia, elem_phonopy], ["Julia", "Phonopy (ASE)"])

    return fig
end

# ─────────────────────────────────────────────────────────────────────────────
#  Script
# ─────────────────────────────────────────────────────────────────────────────

using ACEWorkflow
using LinearAlgebra, ACEpotentials
import ACEWorkflow: phonon_committee  

function phonon_committee(model, coeffs_committee, result; N_per_seg=30, N_cell=2, orig_coeffs=nothing)
    # Save original coefficients so we can restore them after the loop
    if (orig_coeffs == nothing)
        orig_coeffs = result.lin_params
    end
    N = length(coeffs_committee)
    all_freqs = Vector{Matrix{Float64}}(undef, N + 1)
    x_vals_out = nothing
    x_ticks_out = nothing
    labels_out = nothing

    for i in 0:N
        if i > 0
            ACEpotentials.Models.set_linear_parameters!(model, coeffs_committee[i])
        end

        a_eq     = ACEWorkflow.relax_lattice_constant(model, :Al)
        sys_prim  = bulk(:Al; a=a_eq*u"Å")
        sys_super = bulk(:Al; a=a_eq*u"Å", cubic=true) * (N_cell, N_cell, N_cell)

        println("\n--- Committee member $i / $N ---")
        x_vals, freqs, x_ticks, labels = compute_phonon_bands(
            sys_prim, sys_super, model, a_eq; N_per_seg, n_modes=nothing)

        ω_min = round(minimum(freqs), sigdigits=4)
        ω_max = round(maximum(freqs), sigdigits=4)
        n_imag = count(freqs .< 0)
        println("  Frequency range : $ω_min … $ω_max THz")
        n_imag > 0 && println("  Imaginary modes : $n_imag (shown in red)")

        all_freqs[i + 1] = freqs
        if i == 0
            x_vals_out  = x_vals
            x_ticks_out = x_ticks
            labels_out  = labels
        end
    end

    # Restore original (mean) model

    ACEpotentials.Models.set_linear_parameters!(model, orig_coeffs)

    # ── Plot (THz) ──────────────────────────────────────────────────────────
    function _committee_plot(unit_label, scale)
        fig = Figure(size=(750, 500))
        ax  = Axis(fig[1, 1];
                   xlabel       = "Wave vector",
                   ylabel       = unit_label == "THz" ? "Frequency (THz)" : "Energy (eV)",
                   title        = "Al phonon bands — ACE committee",
                   xticks       = (x_ticks_out, labels_out),
                   xgridvisible = false)

        if unit_label == "eV"
            energies_all = [f .* scale for f in all_freqs]
            emin = floor(minimum(minimum.(energies_all)) / 0.01) * 0.01
            emax = ceil( maximum(maximum.(energies_all)) / 0.01) * 0.01
            ax.yticks = emin:0.01:emax
        end

        # Committee members (indices 2..end) — light grey
        for freqs in all_freqs[2:end]
            data = freqs .* scale
            Nmodes = size(data, 1)
            for b in 1:Nmodes
                lines!(ax, x_vals_out, data[b, :];
                       color=RGBAf(0.6, 0.6, 0.6, 0.4), linewidth=1.0)
            end
        end

        # Mean model (index 1) — blue/red on top
        mean_data = all_freqs[1] .* scale
        Nmodes = size(mean_data, 1)
        for b in 1:Nmodes
            branch = mean_data[b, :]
            color  = minimum(branch) < 0 ? RGBAf(0.8, 0.1, 0.1, 0.95) :
                                            RGBAf(0.2, 0.4, 0.7, 0.95)
            lines!(ax, x_vals_out, branch; color, linewidth=2.0)
        end

        hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=0.8)
        vlines!(ax, x_ticks_out; color=(:black, 0.3), linewidth=0.8)
        return fig
    end

    fig_thz = _committee_plot("THz", 1.0)
    save("$(result.dir)/results/phonon_committee_samples_THz_$(N_cell)x$(N_cell)x$(N_cell).png", fig_thz)
    display(fig_thz)
    println("Saved: phonon_committee_THz.png")

    fig_ev = _committee_plot("eV", THz_to_meV / 1000)
    save("$(result.dir)/results/phonon_committee_samples_eV_$(N_cell)x$(N_cell)x$(N_cell).png", fig_ev)
    display(fig_ev)
    println("Saved: phonon_committee_eV.png")

    # ── Save data ────────────────────────────────────────────────────────────
    # x_vals: length-Nq vector
    writedlm("$(result.dir)/results/phonon_committee_x_vals_$(N_cell)x$(N_cell)x$(N_cell).csv",
             x_vals_out, ',')
    # freqs: stacked (N+1)*Nmodes × Nq matrix; rows 1:Nmodes = mean model,
    #        rows Nmodes+1:2*Nmodes = member 1, etc.
    stacked = reduce(vcat, all_freqs)   # ((N+1)*Nmodes) × Nq
    writedlm("$(result.dir)/results/phonon_committee_freqs_THz_$(N_cell)x$(N_cell)x$(N_cell).csv",
             stacked, ',')
    println("Saved: phonon_committee_x_vals and phonon_committee_freqs_THz CSVs")

    return x_vals_out, all_freqs, x_ticks_out, labels_out
end

ACEpotentials.Models.set_linear_parameters!(model, constrained_ridge_teta); 
x_vals_out, all_freqs, x_ticks_out, labels_out = phonon_committee(model, pops_samples, result; N_cell=4, orig_coeffs=constrained_ridge_teta)