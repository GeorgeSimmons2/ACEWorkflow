using LinearAlgebra, StaticArrays, ForwardDiff
using AtomsBase
using AtomsCalculators
using ACEpotentials

# --- Voigt strain (6) -> symmetric small strain tensor (3x3)
function voigt_to_eps(ε::SVector{6,T}) where {T}
    @SMatrix [ ε[1]   ε[6]/2  ε[5]/2
               ε[6]/2 ε[2]    ε[4]/2
               ε[5]/2 ε[4]/2  ε[3]   ]
end

# --- Build strained system: x' = (I+Eps)*x, a' = (I+Eps)*a
function strained_system(sys0::AtomsBase.AbstractSystem, εv::AbstractVector)
    ε = SVector{6, eltype(εv)}(εv)
    Eps = voigt_to_eps(ε)
    G = I + Eps

    sys = deepcopy(sys0)  # simplest; if you have an immutable system, rebuild instead

    # cell vectors as tuple of SVectors
    a = cell_vectors(sys0)  # (a1,a2,a3)
    a′ = (G * a[1], G * a[2], G * a[3])
    set_cell_vectors!(sys, a′)

    # positions
    X = position(sys0, :)  # vector of SVector{3}
    X′ = [G * x for x in X]
    set_position!(sys, :, X′)

    return sys
end

volume_from_cellvecs(a) = abs(dot(a[1], cross(a[2], a[3])))

# ============================================================
# A) Born elastic tensor C (6×6) for the current parameters in calc
# ============================================================
function born_C_voigt(sys0, calc)
    ε0 = zeros(6)

    function W(εv)
        sysε = strained_system(sys0, εv)
        E = potential_energy(sysε, calc)  # AtomsCalculators interface
        V = volume_from_cellvecs(cell_vectors(sysε))
        return E / V
    end

    return ForwardDiff.hessian(W, ε0)  # 6×6
end

# ============================================================
# B) Operator O (6×6×nθ) so that Cαβ = dot(O[α,β,:], θ)
#    Uses ACEpotentials.Models.potential_energy_basis
# ============================================================
function born_operator_voigt(sys0, calc)
    ε0 = zeros(6)

    # feature density f(ε) = Φ(ε)/V(ε)
    function f(εv)
        sysε = strained_system(sys0, εv)
        Φ = ACEpotentials.Models.potential_energy_basis(sysε, calc)  # = energy basis vector
        V = volume_from_cellvecs(cell_vectors(sysε))
        return Φ ./ V
    end

    f0 = f(ε0)
    nθ = length(f0)

    O = Array{eltype(f0)}(undef, 6, 6, nθ)
    for k in 1:nθ
        fk(εv) = f(εv)[k]
        O[:,:,k] .= ForwardDiff.hessian(fk, ε0)
    end
    return O
end

# apply operator:
elastic_from_operator(O, θ) = [dot(view(O,i,j,:), θ) for i in 1:6, j in 1:6]
