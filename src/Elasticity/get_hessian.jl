using StaticArrays, LinearAlgebra, AtomsBuilder
using ForwardDiff
using AtomsCalculators, Unitful
using AtomsBase
using ACEpotentials: potential_energy
using GeometryOptimization
import AtomsCalculatorsUtilities.SitePotentials: PairList, cutoff_radius, get_neighbours, energy_unit  

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

function reference_system(element; a=nothing)
    if isnothing(a)
        return bulk(element)
    else
        a_u = a isa Unitful.Quantity ? a : a * u"Å"
        return bulk(element; a=a_u)
    end
end

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
    # Attach units to lattice (Å)
    Lnew = Lnew_unitless .* u"Å"

    atoms_new = Vector{Atom}(undef, length(sys))
    for i in 1:length(sys)
        at = sys[i]
        rnew = Lnew * fracs[i]                      # rnew is SVector{3,Quantity}
        atoms_new[i] = Atom(at.species, collect(rnew), missing)
    end

    return periodic_system(atoms_new, lattice_tuple(Lnew))
end

function strained_cell_energy(ε::SVector{6,T}; model, element, a=nothing) where {T}
    sys0 = reference_system(element; a=a)

    ϵ = Voigt_strain_to_3x3(ε)
    F = ϵ + one(ϵ)

    # reference lattice, *unitless numeric in Å*
    L0 = SMatrix{3,3,T}(ustrip.(lattice_matrix(sys0.cell.cell_vectors)))

    # strained lattice, still unitless
    L1 = F * L0

    # fractional coords (unitless)
    fracs = fractional_positions_unitless(sys0)

    # rebuild with unitful lattice + unitful positions
    sys1 = rebuild_periodic_system_unitful(sys0; Lnew_unitless=L1, fracs=fracs)

    return potential_energy(sys1, model)
end

N=2

function build_model_basis(r; N::Integer = 2)
    powers = 0:N
    r_vec  = collect(r)                 # preserves eltype(r)
    n      = length(r_vec)
    T      = eltype(r_vec)

    X = Matrix{T}(undef, n, N + 1)
    @inbounds for (j, p) in enumerate(powers)
        X[:, j] .= r_vec .^ p
    end
    return X
end

function strained_cell_design(ε::SVector{6,T}; model, element, a=nothing) where {T}
    sys0 = reference_system(element; a=a)

    ϵ = Voigt_strain_to_3x3(ε)
    F = ϵ + one(ϵ)

    # reference lattice, *unitless numeric in Å*
    L0 = SMatrix{3,3,T}(ustrip.(lattice_matrix(sys0.cell.cell_vectors)))

    # strained lattice, still unitless
    L1 = F * L0

    # fractional coords (unitless)
    fracs = fractional_positions_unitless(sys0)

    # rebuild with unitful lattice + unitful positions
    sys1 = rebuild_periodic_system_unitful(sys0; Lnew_unitless=L1, fracs=fracs)

    return ACEpotentials.Models.potential_energy_basis(sys1, model)
end

function elastic_hessian_energy(model; element=:Al, a=nothing, ε0=nothing)
    ε_ref = isnothing(ε0) ? @SVector(zeros(6)) : ε0
    Eε = ε -> ustrip(strained_cell_energy(ε; model=model, element=element, a=a))
    return ForwardDiff.hessian(Eε, ε_ref)
end

function elastic_hessian_basis(model; element=:Al, a=nothing, ε0=nothing)
    ε_ref = isnothing(ε0) ? @SVector(zeros(6)) : ε0
    f(ε) = strained_cell_design(ε; model=model, element=element, a=a)
    nparams = length(f(ε_ref))
    H_6x6xd = Array{Float64}(undef, 6, 6, nparams)
    for k in 1:nparams
        g(ε) = ustrip.(f(ε)[k])
        H_6x6xd[:, :, k] = ForwardDiff.hessian(g, ε_ref)
    end
    return H_6x6xd
end

using AtomsCalculatorsUtilities.SitePotentials: hessian

function dynamical_matrix(sys, model, q)
    H = ustrip.(hessian(sys, model))

    Nat = length(sys)
    N3 = 3 * Nat

    nlist = PairList(sys, cutoff_radius(model))

    Dq = zeros(ComplexF64, N3, N3)

    for i in 1:Nat
        mi = ustrip(sys[i].mass)

        Js, Rs, _, _ = get_neighbours(sys, model, nlist, i)

        # off-diagonal neighbour contributions
        for (k, j) in enumerate(Js)
            mj = ustrip(sys[j].mass)

            Rij = Rs[k]
            phase = exp(im * dot(q, Rij))

            for α in 1:3, β in 1:3
                I = 3*(i-1) + α
                J = 3*(j-1) + β

                Dq[I,J] += H[I,J] * phase / sqrt(mi * mj)
            end
        end

        # diagonal block (corrected)
        for α in 1:3, β in 1:3
            I = 3*(i-1) + α
            J = 3*(i-1) + β

            Dq[I,J] += H[I,J] / mi
        end
    end

    return Dq
end

function unstable_modes(Dq; tol=1e-8)
    F = eigen(Dq)
    ω2 = real(F.values)

    inds = findall(ω2 .< -tol)

    return ω2[inds], F.vectors[:, inds]
end

function apply_mode(sys, mode, A)
    atoms_new = Vector{Atom}(undef, length(sys))

    for i in 1:length(sys)
        at = sys[i]
        r = ustrip.(at.position)
        mode_i = real.(mode[(3*(i-1)+1):(3*i)])   # 3-vector for atom i
        rnew = r .+ A .* mode_i ./ sqrt(ustrip(at.mass))
        atoms_new[i] = Atom(at.species, collect(rnew .* u"Å"), missing)
    end

    return periodic_system(atoms_new, sys.cell.cell_vectors)
end

function apply_mode_design(model, sys, mode, q, A)
    atoms_new = Vector{Atom}(undef, length(sys))

    for i in 1:length(sys)
        at = sys[i]
        r = ustrip.(at.position)
        phase = exp(im * dot(q, r))
        mode_i = real.(mode[(3*(i-1)+1):(3*i)] .* phase)   # plane-wave weighted 3-vector
        rnew = r .+ A .* mode_i ./ sqrt(ustrip(at.mass))
        atoms_new[i] = Atom(at.species, collect(rnew .* u"Å"), missing)
    end

    sys_new = periodic_system(atoms_new, sys.cell.cell_vectors)
    return ustrip.(ACEpotentials.Models.potential_energy_basis(sys_new, model))
end

function apply_mode_energy(model, sys, mode, q, A)
    atoms_new = Vector{Atom}(undef, length(sys))

    for i in 1:length(sys)
        at = sys[i]
        r = ustrip.(at.position)
        phase = exp(im * dot(q, r))
        mode_i = real.(mode[(3*(i-1)+1):(3*i)] .* phase)   # plane-wave weighted 3-vector
        rnew = r .+ A .* mode_i ./ sqrt(ustrip(at.mass))
        atoms_new[i] = Atom(at.species, collect(rnew .* u"Å"), missing)
    end

    sys_new = periodic_system(atoms_new, sys.cell.cell_vectors)
    return ustrip.(potential_energy(sys_new, model))
end

# ACEpotentials.Models.set_linear_parameters!(model, committee[:,1])
# model, _ = ACEpotentials.load_model("Al_bad_c66_model.json")
function second_deriv_of_model(model, q, sys)
    Dq = dynamical_matrix(sys, model, q)
    unst_eigs, unst_vecs = unstable_modes(Dq) 
    second_derivs = []
    for i=1:size(unst_eigs, 2)
        mode = unst_vecs[:,i]
        E_design(A) = ustrip.(apply_mode_design(model, sys, mode, q, A))
        # Second derivative: central finite differences on the AD first derivative
        # (avoids dual-of-dual types propagating into ACEpotentials internals)
        let h = 1e-5
            dE_design(A) = ForwardDiff.derivative(E_design, A)
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
    for i=1:size(unst_vecs, 2)
        mode = unst_vecs[:,i]
        E_dotted(A) = ustrip.(apply_mode_energy(model, sys, mode, q, A))
        # Second derivative: central finite differences on the AD first derivative
        # (avoids dual-of-dual types propagating into ACEpotentials internals)
        let h = 1e-5
            dE_dotted(A) = ForwardDiff.derivative(E_dotted, A)
            global second_deriv = (dE_dotted(h) - dE_dotted(-h)) / (2h)
        end
        push!(second_derivs, second_deriv)
    end
    return second_derivs
end

# sys=bulk()
# nlist = PairList(sys, cutoff_radius(V))


# Nat = length(sys)
# D = n_dimensions(sys)

# # learn what the types are 
# Js, Rs, Zs, z0 = get_neighbours(sys, V, nlist, 1) 
# # E1 = AtomsCalculatorsUtilities.eval_site(V, Rs, Zs, z0) 

# ace_model = V.model

# function radii!(rs, Rs::AbstractVector{SVector{D, T}}) where {D, T <: Real}
#     @assert length(rs) >= length(Rs)
#     @inbounds for i = 1:length(Rs)
#         rs[i] = norm(Rs[i])
#     end
#     return rs
# end

# const P4ML = Polynomials4ML

# function evaluate_design(model, 
#                          Rs::AbstractVector{SVector{3, T}}, Zs, Z0, 
#                          ps, st) where {T}

#     i_z0 = ACEpotentials.Models._z2i(model.rbasis, Z0)

#     if length(Rs) == 0 
#         return model.Vref.E0[Z0]
#     end 

#     # get the radii 
#     rs = radii!(zeros(length(Rs)), Rs) 

#     # evaluate the radial basis
#     Rnl = ACEpotentials.Models.evaluate_batched(model.rbasis, rs, Z0, Zs, 
#                                         ps.rbasis, st.rbasis)

#     # evaluate the Y basis
#     Ylm = model.ybasis(Rs)

#     # equivariant tensor product
#     # Note: SparseACEbasis returns BB as tuple for multi-L, we use [1] for L=0
#     BB = EquivariantTensors.evaluate(model.tensor, Rnl, Ylm, NamedTuple(), NamedTuple())
#     B = BB[1]

#     # contract with params
#     val = dot(B, (@view ps.WB[:, i_z0]))
    
#     # ------------------- 
#     #  pair potential 
#     if model.pairbasis != nothing 
#         Rpair = ACEpotentials.Models.evaluate_batched(model.pairbasis, rs, Z0, Zs, 
#                                 ps.pairbasis, st.pairbasis)
#         Apair = sum(Rpair, dims=1)[:]
#         val += dot(Apair, (@view ps.Wpair[:, i_z0]))
#         B = vcat(B, Apair)
#     end
#     # ------------------- 
#     #  Vref
#     val += ACEpotentials.Models.eval_site(model.Vref, Rs, Zs, Z0)
#     # ------------------- 
                
#     return B
# end

# # function hessian(sys, V::SitePotential; 
# #                        domain=1:length(sys), 
# #                        executor=ThreadedEx(), 
# #                        nlist=nothing, 
# #                        kwargs...
# #                        )
# #    if isnothing(nlist)
# #       nlist = PairList(sys, cutoff_radius(V))
# #    end

# #    Nat = length(sys)
# #    D = n_dimensions(sys)

# #    # learn what the types are 
# #    Js, Rs, Zs, z0 = get_neighbours(sys, V, nlist, 1)
# #    # 
# #    E1 = evaluate_design(V.model, Rs, Zs, z0, V.ps, V.st)
# #    TF = typeof(E1)
# #    hU = energy_unit(V) / length_unit(V)^2
# #    TFU = typeof(E1 * hU)
# #    # assuming here that the type of everything else will be the same?
# #    # need to think whether this is a restriction. 
# #    # also need to do something about fucking units 

# #    H = zeros(TFU, D*Nat, D*Nat)

# #    for i in domain 
# #       Js, Rs, Zs, z0 = get_neighbours(sys, V, nlist, i) 
# #       Hi = hessian_site(V, Rs, Zs, z0)

# #       Nr = length(Js)
# #       Ji = (i - 1) * D .+ (1:D)
# #       for (α1, j1) in enumerate(Js), (α2, j2) in enumerate(Js)
# #          A1 = (α1-1) * D .+ (1:D)
# #          A2 = (α2-1) * D .+ (1:D)
# #          J1 = (j1-1) * D .+ (1:D)
# #          J2 = (j2-1) * D .+ (1:D)
# #          H[J1, J2] += Hi[A1, A2] .* hU
# #          H[J1, Ji] -= Hi[A1, A2] .* hU
# #          H[Ji, J2] -= Hi[A1, A2] .* hU
# #          H[Ji, Ji] += Hi[A1, A2] .* hU
# #       end
# #    end

# #    return H
# # end

# function hessian_tensor(sys, V::ACEpotentials.SitePotential; 
#                         domain=1:length(sys), 
#                         executor=ThreadedEx(), 
#                         nlist=nothing, 
#                         kwargs...
#                         )
#    if isnothing(nlist)
#       nlist = PairList(sys, cutoff_radius(V))
#    end

#    Nat = length(sys)
#    D = n_dimensions(sys)
#    n_sites = length(domain)  # number of sites in domain

#    # learn what the types are 
#    Js, Rs, Zs, z0 = get_neighbours(sys, V, nlist, 1) 
#    E1 = eval_site(V, Rs, Zs, z0) 
#    TF = typeof(E1)
#    hU = energy_unit(V) / length_unit(V)^2
#    TFU = typeof(E1 * hU)

#    # Create 3-tensor: (3N × 3N × n_sites)
#    H = zeros(TFU, D*Nat, D*Nat, n_sites)

#    for (idx, i) in enumerate(domain)
#       Js, Rs, Zs, z0 = get_neighbours(sys, V, nlist, i) 
#       Hi = hessian_site(V, Rs, Zs, z0)

#       Nr = length(Js)
#       Ji = (i - 1) * D .+ (1:D)
#       for (α1, j1) in enumerate(Js), (α2, j2) in enumerate(Js)
#          A1 = (α1-1) * D .+ (1:D)
#          A2 = (α2-1) * D .+ (1:D)
#          J1 = (j1-1) * D .+ (1:D)
#          J2 = (j2-1) * D .+ (1:D)
#          H[J1, J2, idx] = Hi[A1, A2] .* hU
#          H[J1, Ji, idx] -= Hi[A1, A2] .* hU
#          H[Ji, J2, idx] -= Hi[A1, A2] .* hU
#          H[Ji, Ji, idx] += Hi[A1, A2] .* hU
#       end
#    end

#    return H
# end


# function block_hessian_tensor(sys, V::ACEpotentials.SitePotential; 
#                               domain=1:length(sys), 
#                               executor=ThreadedEx(), 
#                               nlist=nothing, 
#                               kwargs...
#                               )
#    if isnothing(nlist)
#       nlist = PairList(sys, cutoff_radius(V))
#    end

#    Nat = length(sys)
#    D = n_dimensions(sys)
#    n_sites = length(domain)

#    # learn what the types are 
#    Js, Rs, Zs, z0 = get_neighbours(sys, V, nlist, 1) 
#    E1 = eval_site(V, Rs, Zs, z0) 
#    TF = typeof(E1)
#    hU = energy_unit(V) / length_unit(V)^2
#    TFU = typeof(E1 * hU)

#    # Create 3-tensor of blocks: (Nat × Nat × n_sites)
#    H = zeros(SMatrix{D, D, TFU}, Nat, Nat, n_sites)

#    for (idx, i) in enumerate(domain)
#       Js, Rs, Zs, z0 = get_neighbours(sys, V, nlist, i) 
#       Hi = block_hessian_site(V, Rs, Zs, z0)

#       nRs = length(Js)
#       for (α1, j1) in enumerate(Js), (α2, j2) in enumerate(Js)
#          H[j1, j2, idx] = Hi[α1, α2] * hU
#          H[j1, i, idx] -= Hi[α1, α2] * hU
#          H[i, j2, idx] -= Hi[α1, α2] * hU
#          H[i, i, idx] += Hi[α1, α2] * hU
#       end
#    end

#    return H
# end
# # using Test
# # using StaticArrays
# # using Unitful
# # using AtomsBase: bulk, rattle!
# # using LinearAlgebra
# # using AtomsCalculators
# # using Folds

# # """
# #     _globify(Abl::AbstractMatrix{SMatrix{D, D, T}}) 
# # Convert block matrix format to dense matrix format.
# # """
# # function _globify(Abl::AbstractMatrix{SMatrix{D, D, T}}) where {D, T} 
# #    A = zeros(T, D * size(Abl, 1), D * size(Abl, 2))
# #    for j1 = 1:size(Abl, 1), j2 = 1:size(Abl, 2) 
# #       J1 = D * (j1-1) .+ (1:D)
# #       J2 = D * (j2-1) .+ (1:D)
# #       A[J1, J2] .= Abl[j1, j2]
# #    end
# #    return A 
# # end

# # """
# #     _globify_tensor(Abl::AbstractArray{SMatrix{D, D, T}, 3})
# # Convert block tensor format to dense tensor format.
# # """
# # function _globify_tensor(Abl::AbstractArray{SMatrix{D, D, T}, 3}) where {D, T} 
# #    n_sites = size(Abl, 3)
# #    Nat = size(Abl, 1)
# #    A = zeros(T, D * Nat, D * Nat, n_sites)
# #    for k = 1:n_sites, j1 = 1:Nat, j2 = 1:Nat
# #       J1 = D * (j1-1) .+ (1:D)
# #       J2 = D * (j2-1) .+ (1:D)
# #       A[J1, J2, k] .= Abl[j1, j2, k]
# #    end
# #    return A 
# # end

# # function rand_Al_struct(r = 0.1) 
# #    rattle!(bulk(:Al, cubic=true), r)   
# # end

# # @testset "Al Hessian Tensor Consistency" begin
   
# #    # Your ACEpotentials model
# #    V = model  # from your scope
   
# #    @testset "Per-site Hessian Contract" begin
# #       """
# #       Test that summing the tensor slices recovers the original contracted Hessian.
# #       """
# #       sys_test = rand_Al_struct()
# #       domain = 1:length(sys_test)
      
# #       # Compute original contracted Hessian using AtomsCalculators
# #       H_original = hessian(sys_test, V; domain=domain)
      
# #       # Compute tensor Hessian (using new function)
# #       H_tensor = hessian_tensor(sys_test, V; domain=domain)
      
# #       # Contract tensor by summing over all sites
# #       H_contracted = sum(H_tensor[:, :, i] for i in 1:size(H_tensor, 3))
      
# #       # Check they match
# #       @test H_original ≈ H_contracted atol=1e-8
# #       println("✓ Per-site Hessian Contract: PASSED")
# #    end

# #    @testset "Block Hessian Tensor Contract" begin
# #       """
# #       Test block version: summing block tensor recovers block hessian.
# #       """
# #       sys_test = rand_Al_struct()
# #       domain = 1:length(sys_test)
      
# #       # Compute original contracted block Hessian
# #       Hbl_original = block_hessian(sys_test, V; domain=domain)
      
# #       # Compute block tensor Hessian (using new function)
# #       Hbl_tensor = block_hessian_tensor(sys_test, V; domain=domain)
      
# #       # Contract tensor by summing over all sites
# #       Hbl_contracted = sum(Hbl_tensor[:, :, i] for i in 1:size(Hbl_tensor, 3))
      
# #       @test Hbl_original ≈ Hbl_contracted
# #       println("✓ Block Hessian Tensor Contract: PASSED")
# #    end

# #    @testset "Tensor and Block Tensor Equivalence" begin
# #       """
# #       Test that block tensor converts to dense tensor correctly.
# #       """
# #       sys_test = rand_Al_struct()
# #       domain = 1:length(sys_test)
      
# #       # Compute both versions
# #       H_tensor = hessian_tensor(sys_test, V; domain=domain)
# #       Hbl_tensor = block_hessian_tensor(sys_test, V; domain=domain)
      
# #       # Convert block tensor to dense
# #       H_from_blocks = _globify_tensor(Hbl_tensor)
      
# #       @test H_tensor ≈ H_from_blocks atol=1e-10
# #       println("✓ Tensor and Block Tensor Equivalence: PASSED")
# #    end

# #    @testset "Block Format Consistency (Baseline)" begin
# #       """
# #       Verify that dense and block formats are equivalent.
# #       """
# #       sys_test = rand_Al_struct()
# #       domain = 1:length(sys_test)
      
# #       H = hessian(sys_test, V; domain=domain)
# #       Hbl = block_hessian(sys_test, V; domain=domain)
      
# #       @test H ≈ _globify(Hbl) atol=1e-10
# #       println("✓ Block Format Consistency: PASSED")
# #    end

# #    @testset "Tensor Shape Validation" begin
# #       """
# #       Verify the tensor has the correct dimensions.
# #       """
# #       sys_test = rand_Al_struct()
# #       domain = 1:length(sys_test)
      
# #       H_tensor = hessian_tensor(sys_test, V; domain=domain)
# #       Hbl_tensor = block_hessian_tensor(sys_test, V; domain=domain)
      
# #       Nat = length(sys_test)
# #       D = 3
# #       n_sites = length(domain)
      
# #       # Check dense tensor shape
# #       @test size(H_tensor) == (D*Nat, D*Nat, n_sites)
      
# #       # Check block tensor shape  
# #       @test size(Hbl_tensor) == (Nat, Nat, n_sites)
      
# #       println("✓ Tensor Shape Validation: PASSED")
# #    end

# # end



"""
    rebuild_sys_from_flat(r_flat, sys)

Rebuild a periodic system from a flat position vector `r_flat` (unitless, Å),
keeping species and cell vectors from `sys`.
"""
function rebuild_sys_from_flat(r_flat, sys)
    Nat = length(sys)
    atoms_new = Vector{Atom}(undef, Nat)
    for i in 1:Nat
        ri = SVector{3}(r_flat[3i-2:3i]) .* u"Å"
        atoms_new[i] = Atom(sys[i].species, collect(ri), missing)
    end
    return periodic_system(atoms_new, sys.cell.cell_vectors)
end

"""
    hessian_basis(sys, model)

Returns array of shape `(3Nat, 3Nat, n_params)` where slice `[:,:,p]` is the
Hessian of the p-th ACE basis function w.r.t. atomic positions.
Uses ForwardDiff on `potential_energy_basis`.
"""
function hessian_basis(sys, model)
    Nat = length(sys)
    N3  = 3 * Nat

    # flat, unitless (Å) position vector
    r0 = reduce(vcat, [ustrip.(sys[i].position) for i in 1:Nat])

    # energy_basis_flat: R^{3Nat} -> R^{n_params}
    function energy_basis_flat(r_flat)
        sys_new = rebuild_sys_from_flat(r_flat, sys)
        return ACEpotentials.Models.potential_energy_basis(sys_new, model)
    end

    # First Jacobian to get n_params
    J0 = ForwardDiff.jacobian(energy_basis_flat, r0)   # (n_params, N3)
    n_params = size(J0, 1)

    H_basis = zeros(N3, N3, n_params)

    for p in 1:n_params
        # Hessian of p-th basis function = Jacobian of p-th row of J
        Hp = ForwardDiff.jacobian(
            r -> ForwardDiff.jacobian(energy_basis_flat, r)[p, :], r0
        )                                               # (N3, N3)
        H_basis[:, :, p] = Hp
    end

    return H_basis   # units: eV/Å²  (ACE basis is in eV)
end

"""
    dynamical_matrix_basis_at_k(kvec, sys, model)

Returns a matrix `Bk` of size `(N3*N3, n_params)` (complex) such that:

    D(k, θ) = reshape(Bk * θ, N3, N3)

is the mass-weighted dynamical matrix at Brillouin-zone vector `kvec` (1/Å).
Stability requires all eigenvalues of `D(k, θ) ≥ 0` for all k.
"""
function dynamical_matrix_basis_at_k(kvec::AbstractVector, sys, model)
    Nat = length(sys)
    N3  = 3 * Nat

    masses    = [ustrip(sys[i].mass) for i in 1:Nat]
    positions = [SVector{3,Float64}(ustrip.(sys[i].position)) for i in 1:Nat]

    H_basis = hessian_basis(sys, model)    # (N3, N3, n_params)
    n_params = size(H_basis, 3)

    Dk_basis = zeros(ComplexF64, N3, N3, n_params)

    for I in 1:Nat, J in 1:Nat
        phase  = exp(im * dot(kvec, positions[J] - positions[I]))
        mIJ    = sqrt(masses[I] * masses[J])
        Ir     = (I-1)*3 .+ (1:3)
        Jr     = (J-1)*3 .+ (1:3)

        for p in 1:n_params
            Dk_basis[Ir, Jr, p] .+= H_basis[Ir, Jr, p] .* (phase / mIJ)
        end
    end

    # flatten to (N3^2, n_params)
    Bk = reshape(Dk_basis, N3*N3, n_params)
    return Bk
end

"""
    dynamical_matrix_at_k(kvec, sys, model, θ)

Evaluate the dynamical matrix `D(k)` at `kvec` for parameter vector `θ`.
This is **linear** in θ: `D = reshape(Bk * θ, N3, N3)`.
Returns a Hermitian matrix.
"""
function dynamical_matrix_at_k(kvec::AbstractVector, sys, model, θ::AbstractVector)
    N3  = 3 * length(sys)
    Bk  = dynamical_matrix_basis_at_k(kvec, sys, model)
    Dk  = reshape(Bk * θ, N3, N3)
    return Hermitian((Dk + Dk') / 2)
end

# ============================================================
#   Tests
# ============================================================

function test_phonon_design_matrix(model; element=:Al, tol=1e-6)
    println("\n========================================")
    println("  Phonon design matrix tests")
    println("========================================")

    sys = bulk(element)
    θ   = ACEpotentials.Models.get_linear_parameters(model)
    Nat = length(sys)
    N3  = 3 * Nat
    a   = ustrip(sys.cell.cell_vectors[1][1])   # lattice constant (Å)

    # ----------------------------------------------------------
    # Test 1: D(k,θ) at Γ matches mass-weighted Hessian
    # ----------------------------------------------------------
    println("\nTest 1: D(Γ, θ) ≈ mass-weighted Hessian ...")
    kvec_gamma = [0.0, 0.0, 0.0]

    H_ref  = ustrip.(hessian(sys, model))       # (N3, N3), units eV/Å²
    masses = [ustrip(sys[i].mass) for i in 1:Nat]
    D_ref  = zeros(N3, N3)
    for i in 1:N3, j in 1:N3
        mi = masses[cld(i,3)]
        mj = masses[cld(j,3)]
        D_ref[i,j] = H_ref[i,j] / sqrt(mi * mj)
    end

    Dk_gamma = dynamical_matrix_at_k(kvec_gamma, sys, model, θ)
    err1 = norm(real.(Matrix(Dk_gamma)) - D_ref) / norm(D_ref)
    imag_err1 = norm(imag.(Matrix(Dk_gamma)))

    println("  Relative error (real part): $err1")
    println("  Imaginary part norm:        $imag_err1")
    @assert err1 < tol       "Test 1 FAILED: real part mismatch  (err=$err1)"
    @assert imag_err1 < tol  "Test 1 FAILED: non-zero imaginary part"
    println("  ✓ PASSED")

    # ----------------------------------------------------------
    # Test 2: Bk is linear in θ  (D(k, α*θ1 + β*θ2) = α*D(k,θ1) + β*D(k,θ2))
    # ----------------------------------------------------------
    println("\nTest 2: Linearity in θ ...")
    kvec_test = [π/a, 0.0, 0.0]
    θ1 = θ
    θ2 = rand(length(θ))
    α, β = 0.3, 1.7

    Dk1  = dynamical_matrix_at_k(kvec_test, sys, model, θ1)
    Dk2  = dynamical_matrix_at_k(kvec_test, sys, model, θ2)
    Dk12 = dynamical_matrix_at_k(kvec_test, sys, model, α*θ1 + β*θ2)
    err2 = norm(Matrix(Dk12) - (α*Matrix(Dk1) + β*Matrix(Dk2))) / norm(Matrix(Dk12))
    println("  Relative error: $err2")
    @assert err2 < tol "Test 2 FAILED: not linear in θ (err=$err2)"
    println("  ✓ PASSED")

    # ----------------------------------------------------------
    # Test 3: Dk is Hermitian
    # ----------------------------------------------------------
    println("\nTest 3: D(k, θ) is Hermitian ...")
    kvec_test2 = [π/a, π/a, 0.0]
    Dk3 = dynamical_matrix_at_k(kvec_test2, sys, model, θ)
    err3 = norm(Matrix(Dk3) - Matrix(Dk3)') / norm(Matrix(Dk3))
    println("  ||D - D†|| / ||D|| = $err3")
    @assert err3 < tol "Test 3 FAILED: not Hermitian (err=$err3)"
    println("  ✓ PASSED")

    # ----------------------------------------------------------
    # Test 4: eigenvalues of D(Γ,θ) match get_phonon_directions
    # ----------------------------------------------------------
    println("\nTest 4: Eigenvalues at Γ match get_phonon_directions ...")
    vals_ref, _ = get_phonon_directions(sys, model)
    vals_new     = sort(real.(eigvals(Matrix(Dk_gamma))))
    vals_ref_s   = sort(vals_ref)
    err4 = norm(vals_new - vals_ref_s) / (norm(vals_ref_s) + 1e-30)
    println("  Relative error in eigenvalues: $err4")
    @assert err4 < tol "Test 4 FAILED: eigenvalue mismatch (err=$err4)"
    println("  ✓ PASSED")

    println("\n========================================")
    println("  All tests PASSED ✓")
    println("========================================\n")
    return true
end

# ============================================================
#   Cell relaxation and strain Hessian helpers
# ============================================================

"""
    relax_lattice_constant(model, element; a_lo=2.0, a_hi=6.0)

Find the equilibrium lattice constant for `element` by minimising energy per
atom with respect to the scalar lattice parameter `a` (Å) using Brent's method
over the interval `[a_lo, a_hi]`.
"""
function relax_lattice_constant(model, element::Symbol)
    sys = reference_system(element)
    res = minimize_energy!(sys, model; variablecell=true)
    optsys = res.system
    return optsys.cell.cell_vectors[1][2] * 2
end

"""
    strain_hessian_GPa(model, element; a=nothing)

Build the 6×6 Voigt strain Hessian operator and contract it with the current
model parameters to return the elastic tensor in GPa.

If `a` is not provided the lattice constant is relaxed via
`relax_lattice_constant` first.

Returns a `NamedTuple` with fields:
- `C`       – 6×6 elastic tensor in GPa
- `H_basis` – 6×6×n_params operator in eV (reusable for parameter-set sampling)
- `a_eq`    – equilibrium lattice constant used (Å)
"""
function strain_hessian_GPa(model, element::Symbol; a=nothing)
    a_eq = isnothing(a) ? relax_lattice_constant(model, element) : Float64(a)
    sys0 = reference_system(element; a=a_eq)
    L    = SMatrix{3,3,Float64}(ustrip.(lattice_matrix(sys0.cell.cell_vectors)))
    V    = abs(det(L))                                           # unit-cell volume, Å³
    θ    = ACEpotentials.Models.get_linear_parameters(model)
    H    = elastic_hessian_basis(model; element=element, a=a_eq) # (6,6,n_params) eV
    C_eV = dropdims(sum(H .* reshape(θ, 1, 1, :); dims=3); dims=3) # (6,6) eV
    C    = C_eV .* (160.2176621 / V)                             # GPa
    return (C = C, H_basis = H, a_eq = a_eq)
end