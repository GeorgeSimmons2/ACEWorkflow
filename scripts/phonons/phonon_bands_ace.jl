# phonon_bands_ace.jl
#
# Phonon band structure from an ACE potential, plotted as scatter points
# (one dot per eigenvalue at each q-point, no interpolation).
#
# Dynamical-matrix convention mirrors DFTK.jl/phonon.jl:
#   ω = sign(ω²) · √|ω²|   →  negative frequency = imaginary (unstable) mode
#
# Units: THz   (converted from the natural eV/Å²/amu eigenvalue units)
#
# Default system: bulk(:Al, cubic=true)  — 4-atom conventional FCC cell, 12 branches.
# For converged force constants swap in a supercell, e.g.
#   sys = bulk(:Al, cubic=true) * (3,3,3)   → 108 atoms, 324 branches (zone-folded)
#
# Usage (assumes `model` is already loaded in the session):
#   model, _ = ACEpotentials.load_model("../../models/Al_20_4_6A_3/Al_20_4_6A_3.json")
#   include("phonon_bands_ace.jl")

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

# ── Load model (adjust path as needed) ──────────────────────────────────────
result = load_model(:Al, 12, 4, 6, 2; dataset_name="subset_10_percent")
model  = result.model

A = result.A
Y = result.Y
W = result.W
P = result.P


con_model, _ = ACEpotentials.load_model("$(result.dir)/constrained_model.json")
constrained_params = vec(vcat(con_model.ps[1], con_model.ps[2]))

Ap = Diagonal(W) * A / P
Yw = Y .* W
pops_corr = corrections(Ap, Yw, P)
hyp_eig, hyp_bound = hypercube(pops_corr)
samples, _ = sample_hypercube(hyp_eig, hyp_bound, result.lin_params; number_of_committee_members=5)
samples = [samples[:,i] for i=1:size(samples,2)]
con_samples, _ = sample_hypercube(hyp_eig, hyp_bound, constrained_params; number_of_committee_members=50)
con_samples = [con_samples[:,i] for i=1:size(con_samples,2)]
con_forest_result = born_stability_committee(con_model, con_samples, result)
con_samples = con_samples[con_forest_result[:stable][2:end]]

x_vals_out, all_freqs, _, _ = phonon_committee(model, con_samples, result; N_per_seg=30)

# ── System ───────────────────────────────────────────────────────────────────
# Primitive cell: 1-atom FCC primitive cell — the D(q) matrix is 3×3, giving
# the 3 acoustic branches of the FCC crystal with no zone-folding artifacts.
# Using the conventional 4-atom cell here would make X ≡ Γ (a reciprocal
# lattice vector of the conventional cell), producing a flat-looking dispersion.
#
# Supercell: 3×3×3 conventional cell (∼12.15 Å sides) — must be ≥ 2× the ACE
# cutoff (6 Å) so that Φ(i, j+R) decays to zero before the boundary.

N         = 5
a_eq      = ACEWorkflow.relax_lattice_constant(model, :Al)
sys_prim  = bulk(:Al; a=a_eq*u"Å")                         # 1 atom, 3 branches
sys_super = bulk(:Al; a=a_eq*u"Å", cubic=true) * (N,N,N)   # 256 atoms, Hessian source

println("\n=== Phonon band structure (ACE) ===")
x_vals, freqs, x_ticks, labels = compute_phonon_bands(sys_prim, sys_super, model, a_eq;
                                                    N_per_seg=30, n_modes=nothing)

ω_min = round(minimum(freqs), sigdigits=4)
ω_max = round(maximum(freqs), sigdigits=4)
n_imag = count(freqs .< 0)
println("  Frequency range : $ω_min … $ω_max THz")
n_imag > 0 && println("  Imaginary modes : $n_imag (shown in red)")

fig = plot_phonon_bands(x_vals, freqs, x_ticks, labels
                         )
save("$(result.dir)/results/mean_phonon_bands_ace_sample_scatter_$(N)x$(N)x$(N).png", fig)
display(fig)
using ACEpotentials
model, _ = ACEpotentials.load_model("$(result.dir)/exact_constrained_model.json")

# ── System ───────────────────────────────────────────────────────────────────
# Primitive cell: 1-atom FCC primitive cell — the D(q) matrix is 3×3, giving
# the 3 acoustic branches of the FCC crystal with no zone-folding artifacts.
# Using the conventional 4-atom cell here would make X ≡ Γ (a reciprocal
# lattice vector of the conventional cell), producing a flat-looking dispersion.
#
# Supercell: 3×3×3 conventional cell (∼12.15 Å sides) — must be ≥ 2× the ACE
# cutoff (6 Å) so that Φ(i, j+R) decays to zero before the boundary.

N         = 5
a_eq      = ACEWorkflow.relax_lattice_constant(con_model, :Al)
sys_prim  = bulk(:Al; a=a_eq*u"Å")                         # 1 atom, 3 branches
sys_super = bulk(:Al; a=a_eq*u"Å", cubic=true) * (N,N,N)   # 256 atoms, Hessian source

println("\n=== Phonon band structure (ACE) ===")
x_vals, freqs, x_ticks, labels = compute_phonon_bands(sys_prim, sys_super, con_model, a_eq;
                                                      N_per_seg=30, n_modes=nothing)

ω_min = round(minimum(freqs), sigdigits=4)
ω_max = round(maximum(freqs), sigdigits=4)
n_imag = count(freqs .< 0)
println("  Frequency range : $ω_min … $ω_max THz")
n_imag > 0 && println("  Imaginary modes : $n_imag (shown in red)")

fig = plot_phonon_bands(x_vals, freqs, x_ticks, labels
                         )
save("$(result.dir)/results/exact_constrained_mean_phonon_bands_ace_sample_scatter_$(N)x$(N)x$(N).png", fig)
display(fig)
# println("  Saved: phonon_bands_ace_scatter.png")

# fig_e = plot_phonon_energy(x_vals, freqs, x_ticks, labels;
#                             title     = "Al phonon bands — ACE",
#                             linewidth = 1.5)
# save("$(result.dir)/results/phonon_energy_ace_$(N)x$(N)x$(N).png", fig_e)
# display(fig_e)
# println("  Saved: phonon_energy_ace.png")
# writedlm("$(result.dir)/results/phonon_energy_ace_$(N)x$(N)x$(N).csv", freqs .* (THz_to_meV / 1000), ',')
# writedlm("$(result.dir)/results/phonon_x_vals_ace_$(N)x$(N)x$(N).csv", x_vals, ',')



"""
    phonon_committee(model, coeffs_committee, result; N_per_seg=30)

Compute phonon bands for the mean model (i=0) and each committee member,
returning `(x_vals, all_freqs, x_ticks, labels)` where `all_freqs` is a
`Vector` of `Nmodes × Nq` matrices (index 1 = mean model, 2..N+1 = committee).

Also saves two overlay plots to `result.dir/results/`:
  - `phonon_committee_THz.png`  — frequency (THz)
  - `phonon_committee_eV.png`   — energy (eV), 0.01 eV tick spacing

Committee members are drawn in light grey; the mean model is drawn in blue/red.
"""
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

using Printf
"""
    born_stability_committee(model, coeffs_committee, result)

For the mean model (i=0) and each committee member, relax the lattice constant,
compute the cubic elastic tensor C (GPa) via the strain-Hessian basis, and
evaluate the three Born stability criteria for cubic crystals:
  (i)   C11 − C12 > 0
  (ii)  C11 + 2 C12 > 0
  (iii) C44 > 0

Prints a summary table and returns `(; C_list, a_list, stable)` where
`C_list[1]` is the mean model, `C_list[2:end]` are committee members.
"""
function born_stability_committee(model, coeffs_committee, result)
    orig_coeffs = result.lin_params
    N = length(coeffs_committee)

    C_list = Vector{Matrix{Float64}}(undef, N + 1)
    a_list = Vector{Float64}(undef, N + 1)
    stable = Vector{Bool}(undef, N + 1)

    println("\n=== Born stability — ACE committee ===")

    for i in 0:N
        label = i == 0 ? "mean" : "member $i"
        θ = i == 0 ? orig_coeffs : coeffs_committee[i]
        if i > 0
            ACEpotentials.Models.set_linear_parameters!(model, θ)
        end

        print("  [$label] relaxing … ")
        a_i = ACEWorkflow.relax_lattice_constant(model, :Al)
        @printf("a = %.6f Å\n", a_i)

        # Volume from the same reference system that elastic_hessian_basis uses
        sys0 = ACEWorkflow.Elasticity.reference_system(:Al; a=a_i)
        L0   = SMatrix{3,3,Float64}(
                   ustrip.(ACEWorkflow.Elasticity.lattice_matrix(sys0.cell.cell_vectors)))
        V_i        = abs(det(L0))
        eV_to_GPa  = 160.2176621 / V_i

        H_i = ACEWorkflow.elastic_hessian_basis(model; element=:Al, a=a_i)
        C_i = dropdims(sum(H_i .* reshape(θ, 1, 1, :); dims=3); dims=3) .* eV_to_GPa

        C_list[i + 1] = C_i
        a_list[i + 1] = a_i

        C11, C12, C44 = C_i[1,1], C_i[1,2], C_i[4,4]
        b1 = C11 - C12  > 0
        b2 = C11 + 2C12 > 0
        b3 = C44        > 0
        stable[i + 1]  = b1 && b2 && b3

        @printf("         C11=%7.2f  C12=%7.2f  C44=%7.2f  GPa\n", C11, C12, C44)
        @printf("         (i) C11−C12=%+.2f  (ii) C11+2C12=%+.2f  (iii) C44=%+.2f  → %s\n",
                C11-C12, C11+2C12, C44,
                stable[i + 1] ? "STABLE" : "*** UNSTABLE ***")
    end

    # Restore mean model
    ACEpotentials.Models.set_linear_parameters!(model, orig_coeffs)

    # Summary table
    println()
    println(repeat('─', 72))
    @printf("  %-12s  %8s  %8s  %8s  %8s  %s\n",
            "Member", "a (Å)", "C11", "C12", "C44", "Born stable?")
    println("  ", repeat('-', 68))
    for i in 0:N
        label = i == 0 ? "mean" : "member $i"
        C = C_list[i + 1]
        @printf("  %-12s  %8.5f  %8.2f  %8.2f  %8.2f  %s\n",
                label, a_list[i + 1], C[1,1], C[1,2], C[4,4],
                stable[i + 1] ? "✓" : "✗  UNSTABLE")
    end
    n_stable = count(stable)
    println(repeat('─', 72))
    @printf("  %d / %d members satisfy all Born stability criteria.\n",
            n_stable, N + 1)
    println(repeat('─', 72))

    return (; C_list, a_list, stable)
end

"""
    born_stability_mean(model, result)

Check Born stability for the mean model only (no committee or forest).
Relaxes the lattice constant once, computes the elastic tensor via the
strain-Hessian basis, and prints the three cubic Born criteria.

Returns `(; C, a_eq, stable)`.
"""
function born_stability_mean(model, θ)
    println("\n=== Born stability — mean model ===")

    print("  Relaxing … ")
    a_eq = ACEWorkflow.relax_lattice_constant(model, :Al)
    @printf("a_eq = %.6f Å\n", a_eq)

    sys0 = ACEWorkflow.Elasticity.reference_system(:Al; a=a_eq)
    L0   = SMatrix{3,3,Float64}(
               ustrip.(ACEWorkflow.Elasticity.lattice_matrix(sys0.cell.cell_vectors)))
    V0        = abs(det(L0))
    eV_to_GPa = 160.2176621 / V0

    H = ACEWorkflow.elastic_hessian_basis(model; element=:Al, a=a_eq)
    C = dropdims(sum(H .* reshape(θ, 1, 1, :); dims=3); dims=3) .* eV_to_GPa

    C11, C12, C44 = C[1,1], C[1,2], C[4,4]
    b1 = C11 - C12  > 0
    b2 = C11 + 2C12 > 0
    b3 = C44        > 0
    stable = b1 && b2 && b3

    println(repeat('─', 60))
    @printf("  C11 = %8.2f GPa\n", C11)
    @printf("  C12 = %8.2f GPa\n", C12)
    @printf("  C44 = %8.2f GPa\n", C44)
    println(repeat('─', 60))
    @printf("  (i)   C11 − C12   = %+.2f GPa  %s\n", C11-C12,     b1 ? "✓" : "✗")
    @printf("  (ii)  C11 + 2C12  = %+.2f GPa  %s\n", C11+2C12,    b2 ? "✓" : "✗")
    @printf("  (iii) C44         = %+.2f GPa  %s\n", C44,          b3 ? "✓" : "✗")
    println(repeat('─', 60))
    println("  Born stable? ", stable ? "YES ✓" : "NO ✗")
    println(repeat('─', 60))

    return (; C, a_eq, stable)
end

"""
    born_stability_forest(model, forest_vecs, result; verbose=false)

Fast Born stability check for a large set of POPS correction vectors (δθ).

For each member the strain-Hessian basis is linearly updated to account for
the shift in lattice constant implied by the correction, using a 2nd-order
implicit-function-theorem (IFT) approximation:

  δa ≈ δa₁ + δa₂
    δa₁ = −(b′ · δθ) / K
    δa₂ = −((b″ · δθ)·δa₁ + ½(θ_eq·b‴)·δa₁²) / K
    K   = θ_eq · b″

  H(a_eq + δa) ≈ H(a_eq) + δa · (dH/da)|_{a_eq}

All three IFT inputs (b′, b″, b‴, dH/da) are precomputed once; per-member
cost is two dot products + a rank-1 matrix update, keeping the forest sweep fast.

The procedure:
  1. Relax once at the mean model to get `a_eq`.
  2. Precompute the 6×6 strain-Hessian basis H and its lattice derivative dH/da.
  3. Precompute b′, b″, b‴ (ForwardDiff) and K = θ_eq·b″.
  4. For each δθ, compute δa (2nd-order IFT), form H_approx = H + δa·(dH/da),
     contract with θ = θ_eq + δθ, and check the cubic Born criteria.

Returns `(; stable, C11, C12, C44)` where each is a length-N vector.
"""
function born_stability_forest(model, forest_vecs, result; verbose=false)
    N = length(forest_vecs)
    println("\n=== Born stability — POPS delta forest (N=$N) ===")

    # ── Step 1: relax once at the mean model ────────────────────────────────
    print("  Relaxing mean model … ")
    a_eq = ACEWorkflow.relax_lattice_constant(model, :Al)
    @printf("a_eq = %.6f Å\n", a_eq)

    # ── Step 2: precompute strain-Hessian basis and its lattice derivative ───
    print("  Precomputing strain-Hessian basis … ")
    H_basis = ACEWorkflow.elastic_hessian_basis(model; element=:Al, a=a_eq)  # 6×6×Nbasis

    sys0 = ACEWorkflow.Elasticity.reference_system(:Al; a=a_eq)
    L0   = SMatrix{3,3,Float64}(
               ustrip.(ACEWorkflow.Elasticity.lattice_matrix(sys0.cell.cell_vectors)))
    V0        = abs(det(L0))
    eV_to_GPa = 160.2176621 / V0
    println("done.  V = $(round(V0, sigdigits=5)) Å³,  eV→GPa = $(round(eV_to_GPa, sigdigits=6))")

    print("  Precomputing dH/da … ")
    dH_da_fn = ACEWorkflow.Elasticity.strain_hessian_lattice_constant_derivative_ad(model, :Al; a=a_eq)
    dH_da    = dH_da_fn(a_eq)
    println("done.")

    # ── Step 3: IFT ingredients — b′, b″, b‴ at a_eq ───────────────────────
    print("  Precomputing b′, b″, b‴ for IFT lattice update … ")
    function _lattice_basis(a_val)
        sys = ACEWorkflow.Elasticity.reference_system(:Al; a=a_val)
        ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(sys, model))
    end
    b_prime        = ForwardDiff.derivative(_lattice_basis, a_eq)
    b_double_prime = ForwardDiff.derivative(
                         a -> ForwardDiff.derivative(_lattice_basis, a), a_eq)
    b_triple_prime = ForwardDiff.derivative(
                         a -> ForwardDiff.derivative(
                             a2 -> ForwardDiff.derivative(_lattice_basis, a2), a), a_eq)
    θ_mean          = result.lin_params
    K               = dot(θ_mean, b_double_prime)
    scalar_b_triple = dot(θ_mean, b_triple_prime)
    println("done.  K = $(round(K, sigdigits=5)) eV/Å²")

    # Mean model check
    C_mean = dropdims(sum(H_basis .* reshape(θ_mean, 1, 1, :); dims=3); dims=3) .* eV_to_GPa
    @printf("  Mean model:  C11=%7.2f  C12=%7.2f  C44=%7.2f  GPa\n",
            C_mean[1,1], C_mean[1,2], C_mean[4,4])

    # ── Step 4: check each forest member ─────────────────────────────────────
    C11_vec = Vector{Float64}(undef, N)
    C12_vec = Vector{Float64}(undef, N)
    C44_vec = Vector{Float64}(undef, N)
    stable  = Vector{Bool}(undef, N)

    for (k, δθ) in enumerate(forest_vecs)
        k % 1000 == 0 && print("\r  Checking member $k / $N …")

        # 2nd-order IFT lattice-constant shift
        δa_1 = -dot(b_prime, δθ) / K
        δa_2 = -(dot(b_double_prime, δθ) * δa_1 + 0.5 * scalar_b_triple * δa_1^2) / K
        δa   = δa_1 + δa_2

        # Linearly updated Hessian basis and full parameter vector
        H    = H_basis .+ δa .* dH_da
        θ    = θ_mean .+ δθ

        C = dropdims(sum(H .* reshape(θ, 1, 1, :); dims=3); dims=3) .* eV_to_GPa
        C11_vec[k] = C[1,1]
        C12_vec[k] = C[1,2]
        C44_vec[k] = C[4,4]
        stable[k]  = (C[1,1] - C[1,2] > 0) && (C[1,1] + 2C[1,2] > 0) && (C[4,4] > 0)
        if verbose && !stable[k]
            @printf("\n  *** UNSTABLE member %d:  C11=%7.2f  C12=%7.2f  C44=%7.2f\n",
                    k, C[1,1], C[1,2], C[4,4])
        end
    end
    println("\r  Done. ($N members checked)              ")

    n_stable   = count(stable)
    n_unstable = N - n_stable
    println(repeat('─', 60))
    @printf("  Stable   : %d / %d  (%.2f%%)\n", n_stable,   N, 100n_stable/N)
    @printf("  Unstable : %d / %d  (%.2f%%)\n", n_unstable, N, 100n_unstable/N)
    println(repeat('─', 60))

    # Distribution summary
    @printf("  C11  range: [%.2f, %.2f] GPa  (mean=%.2f)\n",
            minimum(C11_vec), maximum(C11_vec), sum(C11_vec)/N)
    @printf("  C12  range: [%.2f, %.2f] GPa  (mean=%.2f)\n",
            minimum(C12_vec), maximum(C12_vec), sum(C12_vec)/N)
    @printf("  C44  range: [%.2f, %.2f] GPa  (mean=%.2f)\n",
            minimum(C44_vec), maximum(C44_vec), sum(C44_vec)/N)
    dC = C11_vec .- C12_vec
    @printf("  C11−C12 range: [%.2f, %.2f] GPa  (min margin=%.2f)\n",
            minimum(dC), maximum(dC), minimum(dC))
    bmod = (C11_vec .+ 2 .* C12_vec)
    @printf("  C11+2C12 range: [%.2f, %.2f] GPa  (min margin=%.2f)\n",
            minimum(bmod), maximum(bmod), minimum(bmod))
    println(repeat('─', 60))

    return (; stable, C11=C11_vec, C12=C12_vec, C44=C44_vec)
end

"""
    phonon_xpoint_stability_forest(model, forest_vecs, result; verbose=false, N_conv=3)

Fast phonon stability check at the X-point of the FCC Brillouin zone for a
large POPS correction forest.

At the X-point every FCC crystal has exactly 3 phonon modes (1 LA + 2 TA),
none of which are acoustic zeros.  A member is stable if all three frequencies
are non-negative (positive real).

Strategy (all expensive work done once, before the forest loop):

  1. Relax once at the mean model to get `a_eq`.
  2. Build the primitive-cell dynamical-matrix basis at X:
       D_basis_X[:,:,k] = D(X) when θ = eₖ (k-th unit vector).
     Cost: n_params × one supercell Hessian + Bloch transform.
  3. Compute dD/da|_{a_eq} via finite differences (2 extra supercell Hessians).
  4. Precompute IFT ingredients b′, b″, b‴ and K = θ_eq · b″.

  Per-member cost (forest loop):
    - 2nd-order IFT δa: two dot-products.
    - D(X) update: D_basis ⊗ (θ_mean + δθ) + δa · dD_da  (3×3 contraction).
    - Diagonalise 3×3 Hermitian matrix.

Returns `(; stable, freqs_THz)` where `freqs_THz` is an `N × 3` matrix of
X-point frequencies in THz for each member.
"""
function phonon_xpoint_stability_forest(model, forest_vecs, result;
                                        verbose=false, N_conv=3)
    N        = length(forest_vecs)
    θ_mean   = result.lin_params
    n_params = length(θ_mean)
    println("\n=== Phonon X-point stability — POPS delta forest (N=$N) ===")

    # ── Step 1: relax mean model ─────────────────────────────────────────────
    print("  Relaxing mean model … ")
    a_eq = ACEWorkflow.relax_lattice_constant(model, :Al)
    @printf("a_eq = %.6f Å\n", a_eq)

    sys_prim  = bulk(:Al; a=a_eq*u"Å")
    sys_super = bulk(:Al; a=a_eq*u"Å", cubic=true) * (N_conv, N_conv, N_conv)

    # X-point in Cartesian (Å⁻¹): [0, 2π/a, 0]
    q_X = [0.0, 2π / a_eq, 0.0]

    # ── Step 2: D(X) basis ───────────────────────────────────────────────────
    print("  Precomputing supercell geometry … ")
    fc_mean = precompute_force_constants(sys_prim, sys_super, model)
    D_mean  = dynamical_matrix_from_fc(fc_mean, q_X)
    println("done.")

    println("  Building D(X) basis ($n_params basis functions) …")
    D_basis_X   = zeros(ComplexF64, 3, 3, n_params)
    orig_params = copy(θ_mean)
    e_k         = zeros(n_params)
    for k in 1:n_params
        k % 50 == 0 && print("\r    $k / $n_params …")
        fill!(e_k, 0.0); e_k[k] = 1.0
        ACEpotentials.Models.set_linear_parameters!(model, e_k)
        H_k = ustrip.(hessian(sys_super, model))
        D_basis_X[:, :, k] = dynamical_matrix_from_fc(merge(fc_mean, (; H=H_k)), q_X)
    end
    ACEpotentials.Models.set_linear_parameters!(model, orig_params)
    println("\r  D(X) basis done.                              ")

    D_check = dropdims(sum(D_basis_X .* reshape(θ_mean, 1, 1, :); dims=3); dims=3)
    @printf("  Basis reconstruction error: %.2e\n", norm(D_check - D_mean))

    # ── Step 3: dD/da at X via central finite differences ───────────────────
    print("  Computing dD/da (finite differences) … ")
    ε_a   = 1e-4
    fc_p  = precompute_force_constants(bulk(:Al; a=(a_eq+ε_a)*u"Å"),
                                       bulk(:Al; a=(a_eq+ε_a)*u"Å", cubic=true) * (N_conv, N_conv, N_conv),
                                       model)
    fc_m  = precompute_force_constants(bulk(:Al; a=(a_eq-ε_a)*u"Å"),
                                       bulk(:Al; a=(a_eq-ε_a)*u"Å", cubic=true) * (N_conv, N_conv, N_conv),
                                       model)
    dD_da = (dynamical_matrix_from_fc(fc_p, [0.0, 2π/(a_eq+ε_a), 0.0]) -
             dynamical_matrix_from_fc(fc_m, [0.0, 2π/(a_eq-ε_a), 0.0])) / (2ε_a)
    println("done.")

    # ── Step 4: IFT ingredients ──────────────────────────────────────────────
    print("  Precomputing b′, b″, b‴ for IFT lattice update … ")
    function _lb_xpt(a_val)
        sys = ACEWorkflow.Elasticity.reference_system(:Al; a=a_val)
        ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(sys, model))
    end
    b_prime        = ForwardDiff.derivative(_lb_xpt, a_eq)
    b_double_prime = ForwardDiff.derivative(a -> ForwardDiff.derivative(_lb_xpt, a), a_eq)
    b_triple_prime = ForwardDiff.derivative(
                         a -> ForwardDiff.derivative(
                             a2 -> ForwardDiff.derivative(_lb_xpt, a2), a), a_eq)
    K               = dot(θ_mean, b_double_prime)
    scalar_b_triple = dot(θ_mean, b_triple_prime)
    println("done.  K = $(round(K, sigdigits=5)) eV/Å²")

    # Mean model X-point check
    f_mean, _ = dq_eigensystem(Matrix{ComplexF64}(Hermitian((D_mean + D_mean') / 2)))
    @printf("  Mean model X-pt freqs: [%.3f, %.3f, %.3f] THz\n",
            f_mean[1], f_mean[2], f_mean[3])

    # ── Step 5: forest sweep ─────────────────────────────────────────────────
    stable    = Vector{Bool}(undef, N)
    freqs_THz = Matrix{Float64}(undef, N, 3)

    for (k, δθ) in enumerate(forest_vecs)
        k % 1000 == 0 && print("\r  Checking member $k / $N …")

        # 2nd-order IFT lattice-constant shift
        δa_1 = -dot(b_prime, δθ) / K
        δa_2 = -(dot(b_double_prime, δθ) * δa_1 + 0.5 * scalar_b_triple * δa_1^2) / K
        δa   = δa_1 + δa_2

        θ = θ_mean .+ δθ
        D = dropdims(sum(D_basis_X .* reshape(θ, 1, 1, :); dims=3); dims=3) .+ δa .* dD_da
        f, _ = dq_eigensystem(Matrix{ComplexF64}(Hermitian((D + D') / 2)))
        freqs_THz[k, :] = f[1:3]
        stable[k] = all(f[1:3] .>= 0)

        if verbose && !stable[k]
            @printf("\n  *** UNSTABLE member %d: X freqs = [%.3f, %.3f, %.3f] THz\n",
                    k, f[1], f[2], f[3])
        end
    end
    println("\r  Done. ($N members checked)              ")

    n_stable   = count(stable)
    n_unstable = N - n_stable
    println(repeat('─', 60))
    @printf("  Stable   : %d / %d  (%.2f%%)\n", n_stable,   N, 100n_stable/N)
    @printf("  Unstable : %d / %d  (%.2f%%)\n", n_unstable, N, 100n_unstable/N)
    println(repeat('─', 60))

    for j in 1:3
        col = freqs_THz[:, j]
        @printf("  Mode %d range: [%.3f, %.3f] THz  (mean=%.3f)\n",
                j, minimum(col), maximum(col), sum(col)/N)
    end
    println(repeat('─', 60))

    return (; stable, freqs_THz)
end

# using DelimitedFiles
# forest_mat  = readdlm("$(result.dir)/pops_corrections.csv", ',')
# forest_vecs = [forest_mat[i, :] for i in 1:size(forest_mat, 1)]   # δθ corrections only
# # committee_vecs = forest_vecs
# # x_vals_out, all_freqs, _, _ = phonon_committee(model, committee_vecs, result; N_per_seg=30)
# forest_result = born_stability_forest(model, forest_vecs, result)
# println("\nAll stable? ", all(forest_result.stable))

# xpt_result = phonon_xpoint_stability_forest(model, forest_vecs, result)
# println("\nAll X-point stable? ", all(xpt_result.stable))