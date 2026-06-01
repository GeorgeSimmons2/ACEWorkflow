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
#  Dynamical matrix  (real-space Hessian → Bloch-transform)
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
    fcc_band_path(L; N_per_seg=30)

Return `(q_list, x_coords, x_ticks, tick_labels, seg_starts)` for the FCC path
Γ → X → U → L → Γ → K.
`L` is the 3×3 primitive lattice matrix (columns = lattice vectors, in Å);
`N_per_seg` gives the number of q-points per segment.
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
    compute_phonon_bands(sys_prim, sys_super, model; N_per_seg=30, n_modes=nothing)

Compute phonon frequencies along the standard FCC path Γ→X→U→L→Γ→K.
Eigenvalues are tracked across q-points to produce smooth, non-crossing bands.

Returns `(x_vals, freqs, x_ticks, labels)` where
`freqs` is a `Nmodes × Nq` matrix of frequencies in THz.
"""
function compute_phonon_bands(sys_prim, sys_super, model; N_per_seg=30, n_modes=nothing)
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

    _track_branches!(freqs, evecs, seg_starts)

    return x_vals, freqs, x_ticks, labels
end

# ─────────────────────────────────────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────────────────────────────────────
"""
    plot_phonon_bands(x_vals, freqs, x_ticks, labels; title, linewidth)

Plot phonon frequencies as piecewise-linear bands (one line per mode row).
Branches that pass through negative values (imaginary modes) are drawn in red;
all others in blue.
"""
function plot_phonon_bands(x_vals, freqs, x_ticks, labels;
                           title="Phonon band structure — ACE",
                           linewidth=1.5)
    Nmodes, Nq = size(freqs)

    fig = Figure(size=(750, 500))
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

Same as `plot_phonon_bands` but with the y-axis in eV (ℏω) rather than THz.
Imaginary (unstable) modes are drawn in red; stable modes in blue.
"""
function plot_phonon_energy(x_vals, freqs, x_ticks, labels;
                            title="Phonon band structure — ACE",
                            linewidth=1.5)
    energy = freqs .* (THz_to_meV / 1000)   # THz → eV
    Nmodes, Nq = size(energy)

    emin = floor(minimum(energy) / 0.01) * 0.01
    emax = ceil(maximum(energy)  / 0.01) * 0.01

    fig = Figure(size=(750, 500))
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
