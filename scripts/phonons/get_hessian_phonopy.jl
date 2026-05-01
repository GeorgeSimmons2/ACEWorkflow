using StaticArrays, LinearAlgebra, AtomsBuilder
using ForwardDiff
using AtomsCalculators, Unitful
using AtomsBase
using ACEpotentials: potential_energy
import AtomsCalculatorsUtilities.SitePotentials: PairList, cutoff_radius, get_neighbours, energy_unit
using Arpack: eigs

# ============================================================
#  Phonopy-style dynamical matrix helpers
#  (ported from phonon_displacements.jl / phonopy dynmat.c)
#
#  Convention (matching phonopy):
#    - q is in FRACTIONAL coordinates of the primitive cell reciprocal lattice
#      (i.e. no 2π prefactor, same as phonopy's `run(q)`)
#    - phase = exp(2πi · q_frac · svec_frac)
#    - fc[i0*Nsuper*9 + k0*9 + (α-1)*3 + β]  (all 0-indexed)
# ============================================================

"""
    get_dm!(dm, num_patom, num_satom, fc, q, svecs, multi,
            p2s_map, charge_sum, i, j, k)

Accumulate the 3×3 dynamical-matrix block contribution from supercell
atom k into `dm`.  All atom indices (i, j, k) are 0-based.
"""
function get_dm!(
    dm::Matrix{ComplexF64},
    num_patom::Int,
    num_satom::Int,
    fc::Vector{Float64},
    q::Vector{Float64},
    svecs::Matrix{Float64},
    multi::Matrix{Int},
    p2s_map::Vector{Int},
    charge_sum::Union{Array{Float64,3},Nothing},
    i::Int, j::Int, k::Int
)
    i_pair  = k * num_patom + i
    m_pair  = multi[i_pair + 1, 1]   # multiplicity
    adrs    = multi[i_pair + 1, 2]   # 0-based address into svecs

    cos_phase = 0.0
    sin_phase = 0.0
    for l in 1:m_pair
        phase = 0.0
        for m in 1:3
            phase += q[m] * svecs[adrs + l, m]
        end
        phase *= 2π
        cos_phase += cos(phase) / m_pair
        sin_phase += sin(phase) / m_pair
    end

    for l in 1:3, m in 1:3
        fc_idx  = p2s_map[i + 1] * num_satom * 9 + k * 9 + (l - 1) * 3 + m
        fc_elem = fc[fc_idx]
        if charge_sum !== nothing
            fc_elem += charge_sum[i + 1, j + 1, l, m]
        end
        dm[l, m] += fc_elem * complex(cos_phase, sin_phase)
    end
end

"""
    get_dynmat_ij!(dynamical_matrix, num_patom, num_satom, fc, q,
                   svecs, multi, mass, s2p_map, p2s_map, charge_sum, i, j)

Compute and store the 3×3 block `[i*3+1:(i+1)*3, j*3+1:(j+1)*3]` of the
dynamical matrix.  i, j are 0-based.
"""
function get_dynmat_ij!(
    dynamical_matrix::Matrix{ComplexF64},
    num_patom::Int,
    num_satom::Int,
    fc::Vector{Float64},
    q::Vector{Float64},
    svecs::Matrix{Float64},
    multi::Matrix{Int},
    mass::Vector{Float64},
    s2p_map::Vector{Int},
    p2s_map::Vector{Int},
    charge_sum::Union{Array{Float64,3},Nothing},
    i::Int, j::Int
)
    mass_sqrt = sqrt(mass[i + 1] * mass[j + 1])
    dm = zeros(ComplexF64, 3, 3)

    for k in 0:(num_satom - 1)
        if s2p_map[k + 1] != p2s_map[j + 1]
            continue
        end
        get_dm!(dm, num_patom, num_satom, fc, q, svecs, multi,
                p2s_map, charge_sum, i, j, k)
    end

    for k in 1:3, l in 1:3
        dynamical_matrix[i * 3 + k, j * 3 + l] = dm[k, l] / mass_sqrt
    end
end

"""
    make_hermitian!(mat, n)

Enforce Hermitian symmetry: mat = (mat + mat†) / 2.
"""
function make_hermitian!(mat::Matrix{ComplexF64}, n::Int)
    for i in 1:n, j in i:n
        mat[i, j] = (mat[i, j] + conj(mat[j, i])) / 2
        mat[j, i] = conj(mat[i, j])
    end
end

"""
    get_dynamical_matrix_at_q!(dynamical_matrix, num_patom, num_satom,
                                fc, q, svecs, multi, mass,
                                s2p_map, p2s_map [, charge_sum, hermitianize])

Fill `dynamical_matrix` (size `3*num_patom × 3*num_patom`) at q-point `q`
(fractional primitive-cell coordinates, no 2π).
"""
function get_dynamical_matrix_at_q!(
    dynamical_matrix::Matrix{ComplexF64},
    num_patom::Int,
    num_satom::Int,
    fc::Vector{Float64},
    q::Vector{Float64},
    svecs::Matrix{Float64},
    multi::Matrix{Int},
    mass::Vector{Float64},
    s2p_map::Vector{Int},
    p2s_map::Vector{Int},
    charge_sum::Union{Array{Float64,3},Nothing} = nothing,
    hermitianize::Bool = true
)
    dynamical_matrix .= 0

    for i in 0:(num_patom - 1), j in 0:(num_patom - 1)
        get_dynmat_ij!(dynamical_matrix, num_patom, num_satom, fc, q,
                       svecs, multi, mass, s2p_map, p2s_map, charge_sum, i, j)
    end

    if hermitianize
        make_hermitian!(dynamical_matrix, num_patom * 3)
    end

    return dynamical_matrix
end

# ============================================================
#  Utility functions (unchanged from get_Hessian.jl)
# ============================================================

function Voigt_strain_to_3x3(strain_vector::SVector{6,T}) where {T}
    s1, s2, s3, s4, s5, s6 = strain_vector
    return @SMatrix [
        s1       0.5*s6   0.5*s5
        0.5*s6   s2       0.5*s4
        0.5*s5   0.5*s4   s3
    ]
end

function lattice_matrix(cell_vectors)
    a, b, c = cell_vectors
    return SMatrix{3,3,eltype(a)}(hcat(a, b, c))
end

lattice_tuple(L) = (L[:,1], L[:,2], L[:,3])

function fractional_positions_unitless(sys)
    L0 = SMatrix{3,3,Float64}(ustrip.(lattice_matrix(sys.cell.cell_vectors)))
    Linv = inv(L0)
    fracs = Vector{SVector{3,Float64}}(undef, length(sys))
    for i in 1:length(sys)
        r = SVector{3,Float64}(ustrip.(sys[i].position))
        fracs[i] = Linv * r
    end
    return fracs
end

function rebuild_periodic_system_unitful(sys; Lnew_unitless, fracs)
    Lnew = Lnew_unitless .* u"Å"
    atoms_new = Vector{Atom}(undef, length(sys))
    for i in 1:length(sys)
        at = sys[i]
        rnew = Lnew * fracs[i]
        atoms_new[i] = Atom(at.species, collect(rnew), missing)
    end
    return periodic_system(atoms_new, lattice_tuple(Lnew))
end

function strained_cell_energy(ε::SVector{6,T}; model, element) where {T}
    sys0 = bulk(element)
    ϵ = Voigt_strain_to_3x3(ε)
    F = ϵ + one(ϵ)
    L0 = SMatrix{3,3,T}(ustrip.(lattice_matrix(sys0.cell.cell_vectors)))
    L1 = F * L0
    fracs = fractional_positions_unitless(sys0)
    sys1 = rebuild_periodic_system_unitful(sys0; Lnew_unitless=L1, fracs=fracs)
    return potential_energy(sys1, model)
end

function strained_cell_design(ε::SVector{6,T}; model, element) where {T}
    sys0 = bulk(element)
    ϵ = Voigt_strain_to_3x3(ε)
    F = ϵ + one(ϵ)
    L0 = SMatrix{3,3,T}(ustrip.(lattice_matrix(sys0.cell.cell_vectors)))
    L1 = F * L0
    fracs = fractional_positions_unitless(sys0)
    sys1 = rebuild_periodic_system_unitful(sys0; Lnew_unitless=L1, fracs=fracs)
    return ACEpotentials.Models.potential_energy_basis(sys1, model)
end

# ============================================================
#  Dynamical matrix (phonopy-style, replaces original)
#
#  q_cart : wavevector in Cartesian coordinates (1/Å).
#           Internally converted to fractional for phonopy helpers.
#           Pass [0,0,0] for Γ-point / commensurate-supercell use.
# ============================================================

using AtomsCalculatorsUtilities.SitePotentials: hessian

function dynamical_matrix(sys, model, q_cart)
    H   = ustrip.(hessian(sys, model))
    Nat = length(sys)
    N3  = 3 * Nat

    # Lattice matrix L (columns = lattice vectors, in Å)
    L    = SMatrix{3,3,Float64}(ustrip.(lattice_matrix(sys.cell.cell_vectors)))
    Linv = inv(L)

    # Convert q: Cartesian (1/Å)  →  fractional (no 2π)
    # q_frac = L^T * q_cart / (2π),  because q_cart · R = 2π * q_frac · frac_R
    q_frac = Vector{Float64}(transpose(L) * q_cart / (2π))

    # ---- Force-constant vector ----
    # Layout: fc[i0*Nat*9 + k0*9 + (α-1)*3 + β]  =  H[3*i0+α, 3*k0+β]
    fc = zeros(Float64, Nat * Nat * 9)
    for i0 in 0:(Nat-1), k0 in 0:(Nat-1), α in 1:3, β in 1:3
        fc[i0 * Nat * 9 + k0 * 9 + (α-1)*3 + β] = H[3*i0 + α, 3*k0 + β]
    end

    # ---- Fractional positions (in supercell units) ----
    fracs = [Linv * SVector{3,Float64}(ustrip.(sys[i].position)) for i in 1:Nat]

    # ---- Shortest-vector table ----
    # One image per pair: svec(k0, i0) = wrap(x_k - x_i) to [-0.5, 0.5]
    # svecs row index (1-based) = k0*Nat + i0 + 1
    # multi[k0*Nat + i0 + 1, :] = [multiplicity=1, address=k0*Nat+i0 (0-based)]
    svecs = zeros(Float64, Nat * Nat, 3)
    multi = zeros(Int,     Nat * Nat, 2)
    for k0 in 0:(Nat-1), i0 in 0:(Nat-1)
        i_pair = k0 * Nat + i0
        fd     = fracs[k0 + 1] - fracs[i0 + 1]
        fd     = fd .- round.(fd)                  # wrap to [-0.5, 0.5]
        svecs[i_pair + 1, :] = fd
        multi[i_pair + 1, :] = [1, i_pair]         # multiplicity, address
    end

    # ---- Primitive == supercell maps (identity, 0-indexed values) ----
    p2s_map = collect(0:Nat-1)
    s2p_map = collect(0:Nat-1)

    masses = [ustrip(sys[i].mass) for i in 1:Nat]

    # ---- Build dynamical matrix ----
    Dq = zeros(ComplexF64, N3, N3)
    get_dynamical_matrix_at_q!(Dq, Nat, Nat, fc, q_frac, svecs, multi,
                                masses, s2p_map, p2s_map, nothing, true)
    return Dq
end

# ============================================================
#  Remaining functions (unchanged from get_Hessian.jl)
# ============================================================

function unstable_modes(Dq; tol=1e-8, n_check=nothing)
    if isnothing(n_check)
        # Full eigendecomposition — exact but O(N³)
        F  = eigen(Hermitian(Dq))
        ω2 = F.values   # real by construction for Hermitian
        inds = findall(ω2 .< -tol)
        return ω2[inds], F.vectors[:, inds]
    else
        # Partial eigendecomposition — only the n_check lowest eigenvalues
        # Much faster for large matrices when only a few unstable modes exist.
        # Requires Arpack.jl: ]add Arpack
        n      = size(Dq, 1)
        ncv    = min(n, max(4 * n_check + 1, 40))   # larger Krylov subspace
        vals, vecs = eigs(Hermitian(Dq), nev=n_check, ncv=ncv,
                          which=:SR, ritzvec=true, maxiter=10*n)
        ω2   = real(vals)
        inds = findall(ω2 .< -tol)
        return ω2[inds], vecs[:, inds]
    end
end

function apply_mode(sys, mode, A)
    atoms_new = Vector{Atom}(undef, length(sys))
    for i in 1:length(sys)
        at     = sys[i]
        r      = ustrip.(at.position)
        mode_i = real.(mode[(3*(i-1)+1):(3*i)])
        rnew   = r .+ A .* mode_i ./ sqrt(ustrip(at.mass))
        atoms_new[i] = Atom(at.species, collect(rnew .* u"Å"), missing)
    end
    return periodic_system(atoms_new, sys.cell.cell_vectors)
end

function apply_mode_design(model, sys, mode, q, A)
    atoms_new = Vector{Atom}(undef, length(sys))
    for i in 1:length(sys)
        at     = sys[i]
        r      = ustrip.(at.position)
        phase  = exp(im * dot(q, r))
        mode_i = real.(mode[(3*(i-1)+1):(3*i)] .* phase)
        rnew   = r .+ A .* mode_i ./ sqrt(ustrip(at.mass))
        atoms_new[i] = Atom(at.species, collect(rnew .* u"Å"), missing)
    end
    sys_new = periodic_system(atoms_new, sys.cell.cell_vectors)
    return ustrip.(ACEpotentials.Models.potential_energy_basis(sys_new, model))
end

function apply_mode_energy(model, sys, mode, q, A)
    atoms_new = Vector{Atom}(undef, length(sys))
    for i in 1:length(sys)
        at     = sys[i]
        r      = ustrip.(at.position)
        phase  = exp(im * dot(q, r))
        mode_i = real.(mode[(3*(i-1)+1):(3*i)] .* phase)
        rnew   = r .+ A .* mode_i ./ sqrt(ustrip(at.mass))
        atoms_new[i] = Atom(at.species, collect(rnew .* u"Å"), missing)
    end
    sys_new = periodic_system(atoms_new, sys.cell.cell_vectors)
    return ustrip.(potential_energy(sys_new, model))
end

function apply_mode_design(model, sys, mode, q, A)
    atoms_new = Vector{Atom}(undef, length(sys))
    for i in 1:length(sys)
        at     = sys[i]
        r      = ustrip.(at.position)
        phase  = exp(im * dot(q, r))
        mode_i = real.(mode[(3*(i-1)+1):(3*i)] .* phase)
        rnew   = r .+ A .* mode_i ./ sqrt(ustrip(at.mass))
        atoms_new[i] = Atom(at.species, collect(rnew .* u"Å"), missing)
    end
    sys_new = periodic_system(atoms_new, sys.cell.cell_vectors)
    return ustrip.(ACEpotentials.Models.potential_energy_basis(sys_new, model))
end

# ============================================================
#  Script section
# ============================================================

N = 4
sys = bulk(:Al, cubic=true) * (N, N, N)

# model_copy = deepcopy(model)
# ACEpotentials.Models.set_linear_parameters!(model, committee[:,1])
# model, _ = ACEpotentials.load_model("Al_bad_c66_model.json")

function second_deriv_of_model(model, q, sys)
    Dq = dynamical_matrix(sys, model, q)
    unst_eigs, unst_vecs = unstable_modes(Dq)
    second_derivs = []
    for i in 1:size(unst_vecs, 2)
        mode      = unst_vecs[:, i]
        E_design(A) = ustrip.(apply_mode_design(model, sys, mode, q, A))
        let h = 1e-5
            dE_design(A)   = ForwardDiff.derivative(E_design, A)
            global second_deriv = (dE_design(h) - dE_design(-h)) / (2h)
        end
        push!(second_derivs, second_deriv)
    end
    return second_derivs
end

function second_deriv_of_model_dotted(model, q, sys)
    Dq = dynamical_matrix(sys, model, q)
    unst_eigs, unst_vecs = unstable_modes(Dq)
    second_derivs = []
    for i in 1:size(unst_vecs, 2)
        mode       = unst_vecs[:, i]
        E_dotted(A) = ustrip.(apply_mode_energy(model, sys, mode, q, A))
        let h = 1e-5
            dE_dotted(A)   = ForwardDiff.derivative(E_dotted, A)
            global second_deriv = (dE_dotted(h) - dE_dotted(-h)) / (2h)
        end
        push!(second_derivs, second_deriv)
    end
    return second_derivs
end

# ============================================================
#  Wrapper: (sys, q, model, A) → energy of mode-displaced structure
# ============================================================

"""
    mode_displaced_energy(sys, q, model, A; mode_index=1)

Displace `sys` by scalar amplitude `A` along the `mode_index`-th unstable
phonon eigenvector at Cartesian wavevector `q` (1/Å), and return the
potential energy (eV, unitless Float64).

Pipeline:
  1. `dynamical_matrix(sys, model, q)`   — phonopy-style D(q)
  2. `unstable_modes(Dq)`               — negative-eigenvalue eigenvectors
  3. `apply_mode_energy(..., A)`         — displace atoms, evaluate E

Throws if no unstable modes exist or `mode_index` is out of range.
"""
function mode_displaced_energy(sys, q, model, A; mode_index=1)
    Dq = dynamical_matrix(sys, model, q)
    ω2, vecs = unstable_modes(Dq)

    if isempty(ω2)
        error("No unstable modes found at q = $q")
    end
    if mode_index > size(vecs, 2)
        error("mode_index=$mode_index but only $(size(vecs,2)) unstable mode(s) found")
    end

    mode = vecs[:, mode_index]
    return apply_mode_energy(model, sys, mode, q, A)
end

function mode_displaced_design(sys, q, model, A; mode_index=1)
    Dq = dynamical_matrix(sys, model, q)
    ω2, vecs = unstable_modes(Dq)

    if isempty(ω2)
        error("No unstable modes found at q = $q")
    end
    if mode_index > size(vecs, 2)
        error("mode_index=$mode_index but only $(size(vecs,2)) unstable mode(s) found")
    end

    mode = vecs[:, mode_index]
    return apply_mode_design(model, sys, mode, q, A)
end
# ============================================================
#  Test for mode_displaced_energy
# ============================================================

function test_mode_displaced_energy(model, sys, q; tol=1e-9)
    println("\n========================================")
    println("  mode_displaced_energy tests")
    println("========================================")

    # ---- Test 1: A=0 returns equilibrium energy ----
    println("\nTest 1: A=0 returns equilibrium energy ...")
    E0_ref = ustrip.(potential_energy(sys, model))
    E0     = mode_displaced_energy(sys, q, model, 0.0)
    err1   = abs(E0 - E0_ref)
    println("  |E(A=0) - E_ref| = $err1 eV")
    @assert err1 < tol "Test 1 FAILED: E(A=0) ≠ E_ref (err=$err1)"
    println("  ✓ PASSED")

    # ---- Test 2: E(A) = E(-A)  (inversion symmetry about equilibrium) ----
    println("\nTest 2: E(A) = E(-A) (even function) ...")
    A_test = 0.02
    Ep =  mode_displaced_energy(sys, q, model,  A_test)
    Em =  mode_displaced_energy(sys, q, model, -A_test)
    err2 = abs(Ep - Em)
    println("  |E(+A) - E(-A)| = $err2 eV")
    @assert err2 < tol "Test 2 FAILED: energy not symmetric (err=$err2)"
    println("  ✓ PASSED")

    # ---- Test 3: second derivative sign matches unstable eigenvalue ----
    println("\nTest 3: second derivative is negative (unstable mode) ...")
    h  = 1e-4
    d2 = (mode_displaced_energy(sys, q, model,  h) -
          2 * mode_displaced_energy(sys, q, model, 0.0) +
          mode_displaced_energy(sys, q, model, -h)) / h^2
    println("  d²E/dA² ≈ $d2 eV/Å²")
    @assert d2 < 0 "Test 3 FAILED: second derivative not negative (d2=$d2)"
    println("  ✓ PASSED")

    println("\n========================================")
    println("  All tests PASSED ✓")
    println("========================================\n")
    return true
end

"""
    test_partial_vs_full_eigen(model, sys, q; n=5, tol=1e-6)

Check that the `n` lowest eigenvalues and eigenvectors returned by
`unstable_modes(...; n_check=n)` agree with those from the full
eigendecomposition at wavevector `q`.

Eigenvectors are compared up to an overall phase (multiply each column
of the partial result by the conjugate of the phase of its first element
relative to the full result).
"""
function test_partial_vs_full_eigen(model, sys, q; n=1, tol=1e-6)
    println("\n========================================")
    println("  Partial vs full eigendecomposition test")
    println("  n=$n lowest modes at q = $q")
    println("========================================")

    Dq = dynamical_matrix(sys, model, q)

    # Full decomposition
    ω2_full, vecs_full = unstable_modes(Dq)
    # Take the n most-negative eigenvalues (sorted ascending by eigen)
    n_use = min(n, length(ω2_full))
    if n_use == 0
        println("  No unstable modes found — skipping eigenvector test.")
        return true
    end

    # Partial decomposition — ask for n_use most negative
    ω2_part, vecs_part = unstable_modes(Dq; n_check=n_use)

    # ---- Test 1: eigenvalue agreement ----
    println("\nTest 1: Eigenvalues agree to tol=$tol ...")
    for i in 1:n_use
        err = abs(ω2_full[i] - ω2_part[i])
        println("  mode $i: full=$(round(ω2_full[i], sigdigits=6))  " *
                "partial=$(round(ω2_part[i], sigdigits=6))  |Δ|=$err")
        @assert err < tol "Test 1 FAILED at mode $i: |Δλ| = $err"
    end
    println("  ✓ PASSED")

    # ---- Test 2: eigenvector agreement (up to global phase per mode) ----
    println("\nTest 2: Eigenvectors agree (up to global phase) ...")
    for i in 1:n_use
        u_full = vecs_full[:, i]
        u_part = vecs_part[:, i]
        # Find phase: φ = conj(u_part[j]) * u_full[j] / |...|  for first nonzero j
        j = findfirst(abs.(u_full) .> 1e-12)
        if isnothing(j)
            continue
        end
        phase = conj(u_part[j]) * u_full[j]
        phase /= abs(phase)
        u_part_aligned = u_part .* phase
        err = norm(u_full - u_part_aligned)
        println("  mode $i: ||u_full - u_part|| = $err")
        @assert err < tol "Test 2 FAILED at mode $i: ||Δu|| = $err"
    end
    println("  ✓ PASSED")

    println("\n========================================")
    println("  All tests PASSED ✓")
    println("========================================\n")
    return true
end

# ============================================================
#  Script: energy curve along mode_index=1 at q_U
# ============================================================

using CairoMakie

# U-point in Cartesian (1/Å): q = (2π/a) * [1, 1/4, 1/4]
a_Al = 4.05   # Å
q_U  = [1.0, 0.25, 0.25] .* (2π / a_Al)
q_G  = [0.0, 0.0, 0.0]

# Amplitude sweep (Å * sqrt(amu))
A_vals = range(-0.3, 0.3, length=10)

function energy_curve(sys, q, model; mode_index=1)
    # Compute D(q) and extract the mode ONCE, then sweep amplitudes
    Dq = dynamical_matrix(sys, model, q)
    ω2, vecs = unstable_modes(Dq)

    if isempty(ω2)
        error("No unstable modes at q=$q — cannot plot energy curve")
    end
    if mode_index > size(vecs, 2)
        error("mode_index=$mode_index but only $(size(vecs,2)) unstable mode(s) at q=$q")
    end

    println("  q=$q  →  mode $mode_index eigenvalue ω² = $(round(ω2[mode_index], sigdigits=5))")

    mode   = vecs[:, mode_index]
    E0     = ustrip.(potential_energy(sys, model))
    E_vals = [apply_mode_energy(model, sys, mode, q, A) for A in A_vals]
    return E_vals .- E0
end

# ΔE_G = energy_curve(sys, q_G, model)
ΔE_U = energy_curve(sys, q_U, model)

fig = Figure(size = (600, 700))

ax1 = Axis(fig[1, 1],
    xlabel = "Displacement amplitude A  (Å√amu)",
    ylabel = "ΔE  (eV)",
    title  = "U-point  q = (1, ¼, ¼) · 2π/a")
lines!(ax1, collect(A_vals), ΔE_U, linewidth = 2)
hlines!(ax1, [0.0], color = :grey, linestyle = :dash, linewidth = 1)

# ax2 = Axis(fig[2, 1],
#     xlabel = "Displacement amplitude A  (Å√amu)",
#     ylabel = "ΔE  (eV)",
#     title  = "Γ-point  q = (0, 0, 0)")
# lines!(ax2, collect(A_vals), ΔE_G, linewidth = 2, color = Makie.wong_colors()[2])
# hlines!(ax2, [0.0], color = :grey, linestyle = :dash, linewidth = 1)

# save("mode_energy_curve_q_U_vs_Gamma.png", fig)
save("mode_energy_curve_q_U.png", fig)
display(fig)
