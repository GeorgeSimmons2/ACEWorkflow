
using StaticArrays
using Unitful, LinearAlgebra
using AtomsCalculators: potential_energy, forces, virial;
using ACEpotentials;
using AtomsBase: FlexibleSystem, FastSystem, position;
using AtomsBase;
using StaticArrays;
using AtomsBuilder, AtomsCalculators, AtomsBase;


# -------------------------
# Strain definitions
# -------------------------

# C11: uniaxial strain along x
function strain_C11(cell, δ)
    ε = @SMatrix [1+δ 0 0;
                  0 1 0;
                  0 0 1]
    return Tuple(SVector{3}(ε * v) for v in cell)
end

# C12: coupled strain (x stretched, y compressed)
function strain_C12(cell, δ)
    ε = @SMatrix [1+δ 0 0;
                  0 1-δ 0;
                  0 0 1]
    return Tuple(SVector{3}(ε * v) for v in cell)
end

# C44: shear strain (xy)
function strain_C44(cell, δ)
    ε = @SMatrix [1  δ/2 0;
                  δ/2 1 0;
                  0  0   1]
    return Tuple(SVector{3}(ε * v) for v in cell)
end

# -------------------------
# Stress evaluation
# -------------------------

function compute_stress(model, make_cell, cell_vectors, δs, deformation)
    stresses = Matrix{Float64}[]

    for δ in δs
        strained_cell = deformation(cell_vectors, δ)
        sys = make_cell(strained_cell)

        V = LinearAlgebra.dot(sys.cell.cell_vectors[1],
                LinearAlgebra.cross(sys.cell.cell_vectors[2], sys.cell.cell_vectors[3]))

        σ = virial(sys, model) ./ V  # stress tensor (units: eV/Å^3)

        push!(stresses, Matrix(ustrip.(σ)))
    end

    return stresses
end

# -------------------------
# Linear fit helper
# -------------------------

function linear_fit(x, y)
    X = hcat(ones(length(x)), x)
    coeffs = X \ y
    return coeffs
end

# -------------------------
# Main driver
# -------------------------

function compute_cubic_elastic_constants_local(
    model,
    make_cell,
    cell_vectors;
    δs = collect(range(-0.005, 0.005, length=7))
)

    δs = collect(Float64.(δs))

    # ---- C11 ----
    S = compute_stress(model, make_cell, cell_vectors, δs, strain_C11)
    σxx = [-S[i][1,1] for i in eachindex(S)]
    C11 = linear_fit(δs, σxx)[2]

    # ---- C12 ----
    σyy = [-S[i][2,2] for i in eachindex(S)]
    C12 = linear_fit(δs, σyy)[2]

    # ---- C44 ----
    S = compute_stress(model, make_cell, cell_vectors, δs, strain_C44)
    σxy = [-S[i][1,2] for i in eachindex(S)]
    C44 = linear_fit(δs, σxy)[2]

    # Convert to GPa
    conv = 160.2176621

    return (
        C11 = C11 * conv,
        C12 = C12 * conv,
        C44 = C44 * conv
    )
end
