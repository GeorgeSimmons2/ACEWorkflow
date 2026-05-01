"""
Julia implementation of phonopy's dynamical matrix calculation.
Translates C code from dynmat.c to Julia for computing phonon dispersion relations.
"""

using LinearAlgebra
using StaticArrays

const PI = π

"""
    get_dynamical_matrix_at_q!(
        dynamical_matrix::Array{ComplexF64, 2},
        num_patom::Int,
        num_satom::Int,
        fc::Vector{Float64},
        q::Vector{Float64},
        svecs::Matrix{Float64},
        multi::Matrix{Int},
        mass::Vector{Float64},
        s2p_map::Vector{Int},
        p2s_map::Vector{Int},
        charge_sum::Union{Array{Float64, 3}, Nothing} = nothing,
        hermitianize::Bool = false,
        use_threading::Bool = false
    )

Compute the dynamical matrix at a specific q-point.

# Arguments
- `dynamical_matrix`: Output array [num_patom*3, num_patom*3] (will be filled with complex values)
- `num_patom`: Number of atoms in primitive cell
- `num_satom`: Number of atoms in supercell
- `fc`: Force constants array [num_patom, num_satom, 3, 3] (flattened)
- `q`: q-point in crystallographic coordinates [3]
- `svecs`: Shortest vectors [num_svecs, 3]
- `multi`: Multiplicities array [num_satom, num_patom, 2]
- `mass`: Atomic masses [num_patom]
- `s2p_map`: Supercell to primitive atom mapping
- `p2s_map`: Primitive to supercell atom mapping
- `charge_sum`: Optional long-range correction term [num_patom, num_patom, 3, 3]
- `hermitianize`: Whether to enforce Hermitian symmetry
- `use_threading`: Whether to use multithreading
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
    charge_sum::Union{Array{Float64, 3}, Nothing} = nothing,
    hermitianize::Bool = false,
    use_threading::Bool = false
)
    # Fill output matrix
    dynamical_matrix .= 0
    
    if use_threading
        Threads.@threads for ij in 0:(num_patom * num_patom - 1)
            i = div(ij, num_patom)
            j = mod(ij, num_patom)
            get_dynmat_ij!(
                dynamical_matrix, num_patom, num_satom, fc, q, svecs,
                multi, mass, s2p_map, p2s_map, charge_sum, i, j
            )
        end
    else
        for i in 0:(num_patom - 1)
            for j in 0:(num_patom - 1)
                get_dynmat_ij!(
                    dynamical_matrix, num_patom, num_satom, fc, q, svecs,
                    multi, mass, s2p_map, p2s_map, charge_sum, i, j
                )
            end
        end
    end
    
    if hermitianize
        make_hermitian!(dynamical_matrix, num_patom * 3)
    end
    
    return dynamical_matrix
end

"""
    get_dynmat_ij!(
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
        charge_sum::Union{Array{Float64, 3}, Nothing},
        i::Int,
        j::Int
    )

Compute the 3×3 dynamical matrix block for atoms i and j.
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
    charge_sum::Union{Array{Float64, 3}, Nothing},
    i::Int,
    j::Int
)
    # Convert to 0-indexing for this function
    mass_sqrt = sqrt(mass[i + 1] * mass[j + 1])
    
    # Initialize 3x3 complex matrix for this block
    dm = zeros(ComplexF64, 3, 3)
    
    # Sum over all lattice points
    for k in 0:(num_satom - 1)
        if s2p_map[k + 1] != p2s_map[j + 1]
            continue
        end
        get_dm!(dm, num_patom, num_satom, fc, q, svecs, multi, p2s_map,
                charge_sum, i, j, k)
    end
    
    # Store in output matrix with mass normalization
    for k in 1:3
        for l in 1:3
            # Row-major indexing: (i*3+k, j*3+l)
            row = i * 3 + k
            col = j * 3 + l
            dynamical_matrix[row, col] = dm[k, l] / mass_sqrt
        end
    end
end

"""
    get_dm!(
        dm::Matrix{ComplexF64},
        num_patom::Int,
        num_satom::Int,
        fc::Vector{Float64},
        q::Vector{Float64},
        svecs::Matrix{Float64},
        multi::Matrix{Int},
        p2s_map::Vector{Int},
        charge_sum::Union{Array{Float64, 3}, Nothing},
        i::Int,
        j::Int,
        k::Int
    )

Core calculation: compute dynamical matrix elements using force constants and phases.
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
    charge_sum::Union{Array{Float64, 3}, Nothing},
    i::Int,
    j::Int,
    k::Int
)
    # Get multiplicity and address info
    i_pair = k * num_patom + i
    m_pair = multi[i_pair + 1, 1]
    adrs = multi[i_pair + 1, 2]
    
    # Calculate average phase factors
    cos_phase = 0.0
    sin_phase = 0.0
    
    for l in 1:m_pair
        phase = 0.0
        for m in 1:3
            phase += q[m] * svecs[adrs + l, m]
        end
        cos_phase += cos(phase * 2 * PI) / m_pair
        sin_phase += sin(phase * 2 * PI) / m_pair
    end
    
    # Add force constant contributions
    for l in 1:3
        for m in 1:3
            # Force constant indexing: fc[p2s_map[i], k, l, m]
            # Flattened: fc_index = p2s_map[i] * num_satom * 9 + k * 9 + (l-1)*3 + m
            fc_index = p2s_map[i + 1] * num_satom * 9 + k * 9 + (l - 1) * 3 + m
            fc_elem = fc[fc_index]
            
            # Add long-range correction if present
            if charge_sum !== nothing
                fc_elem += charge_sum[i + 1, j + 1, l, m]
            end
            
            dm[l, m] += fc_elem * complex(cos_phase, sin_phase)
        end
    end
end

"""
    make_hermitian!(mat::Matrix{ComplexF64}, num_band::Int)

Enforce Hermitian symmetry on the matrix.
"""
function make_hermitian!(mat::Matrix{ComplexF64}, num_band::Int)
    for i in 1:num_band
        for j in i:num_band
            # Average diagonal and symmetric elements
            mat[i, j] = (mat[i, j] + conj(mat[j, i])) / 2
            mat[j, i] = conj(mat[i, j])
        end
    end
end

"""
    get_q_cart(q::Vector{Float64}, reciprocal_lattice::Matrix{Float64})

Convert q-point from crystallographic to Cartesian coordinates.

# Arguments
- `q`: q-point in crystallographic coordinates [3]
- `reciprocal_lattice`: Reciprocal lattice vectors in column format [3, 3]

# Returns
- q_cart: q-point in Cartesian coordinates [3]
"""
function get_q_cart(q::Vector{Float64}, reciprocal_lattice::Matrix{Float64})
    q_cart = zeros(Float64, 3)
    for i in 1:3
        for j in 1:3
            q_cart[i] += reciprocal_lattice[i, j] * q[j]
        end
    end
    return q_cart
end

# ============================================================================
# Wrapper function for multiple q-points
# ============================================================================

"""
    get_dynamical_matrices_at_qpoints(
        qpoints::Matrix{Float64},
        fc::Vector{Float64},
        svecs::Matrix{Float64},
        multi::Matrix{Int},
        mass::Vector{Float64},
        s2p_map::Vector{Int},
        p2s_map::Vector{Int},
        num_patom::Int,
        num_satom::Int;
        charge_sum::Union{Array{Float64, 3}, Nothing} = nothing,
        hermitianize::Bool = false,
        use_threading::Bool = false
    )

Compute dynamical matrices for multiple q-points.

# Arguments
- `qpoints`: q-points in crystallographic coordinates [n_qpoints, 3]
- All other arguments as in get_dynamical_matrix_at_q!

# Returns
- Array of dynamical matrices [n_qpoints, num_patom*3, num_patom*3] (complex)
"""
function get_dynamical_matrices_at_qpoints(
    qpoints::Matrix{Float64},
    fc::Vector{Float64},
    svecs::Matrix{Float64},
    multi::Matrix{Int},
    mass::Vector{Float64},
    s2p_map::Vector{Int},
    p2s_map::Vector{Int},
    num_patom::Int,
    num_satom::Int;
    charge_sum::Union{Array{Float64, 3}, Nothing} = nothing,
    hermitianize::Bool = false,
    use_threading::Bool = false
)
    n_qpoints = size(qpoints, 1)
    n_modes = num_patom * 3
    
    dynamical_matrices = zeros(ComplexF64, n_qpoints, n_modes, n_modes)
    
    if use_threading
        Threads.@threads for iq in 1:n_qpoints
            get_dynamical_matrix_at_q!(
                view(dynamical_matrices, iq, :, :),
                num_patom, num_satom, fc, qpoints[iq, :], svecs,
                multi, mass, s2p_map, p2s_map, charge_sum,
                hermitianize, false  # Don't nest threading
            )
        end
    else
        for iq in 1:n_qpoints
            get_dynamical_matrix_at_q!(
                view(dynamical_matrices, iq, :, :),
                num_patom, num_satom, fc, qpoints[iq, :], svecs,
                multi, mass, s2p_map, p2s_map, charge_sum,
                hermitianize, false
            )
        end
    end
    
    return dynamical_matrices
end

# ============================================================================
# High-level wrapper: supercell + q + model + amplitude → energy
# Depends on: dynamical_matrix, unstable_modes, apply_mode_energy (get_Hessian.jl)
# ============================================================================

"""
    phonon_mode_energy(sys, q_cart, model, A; mode_index=1)

Displace `sys` by amplitude `A` along the `mode_index`-th unstable phonon mode
at Cartesian wavevector `q_cart` (1/Å), and return the potential energy (eV).

Steps:
  1. Build the mass-weighted dynamical matrix D(q) from the Hessian.
  2. Extract the unstable eigenmodes (negative eigenvalues).
  3. Apply displacement `u_i = A * Re(e_i * exp(iq·r_i)) / sqrt(m_i)` to each atom.
  4. Evaluate and return the potential energy.

If there are no unstable modes, throws an error.
"""
function phonon_mode_energy(sys, q_cart, model, A; mode_index=1)
    Dq = dynamical_matrix(sys, model, q_cart)
    ω2, vecs = unstable_modes(Dq)

    if isempty(ω2)
        error("No unstable modes found at q = $q_cart")
    end
    if mode_index > size(vecs, 2)
        error("mode_index=$mode_index but only $(size(vecs,2)) unstable mode(s) found")
    end

    mode = vecs[:, mode_index]
    return apply_mode_energy(model, sys, mode, q_cart, A)
end