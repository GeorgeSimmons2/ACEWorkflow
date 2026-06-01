# Use the Models helper — builds or loads from models/<name>/
# Adjust the 5 hyperparameter variables below and run this script.

include("../../src/POPS/POPSRegression.jl")
include("../../src/Models/Models.jl")
using Main.Models

# ── Model hyperparameters ─────────────────────────────────────────────────────
# Naming convention: {Element}_{totaldegree}_{smoothness}_{cutoff}_{order}
# e.g.  Al_20_5_6A_3  →  element=Al, totaldeg=20, smooth=5, rcut=6Å, order=3
element     = :Al
totaldegree = 12
order       = 3
rcut        = 6.0
smoothness  = 4

training_xyz = joinpath(@__DIR__, "..", "..", "data", "manual_df_train_Al.xyz")

result = load_model(element, totaldegree, smoothness, rcut, order;
                    training_xyz=training_xyz)

model        = result.model
A            = result.A
Y            = result.Y
P            = result.P
W            = result.W
lin_params   = result.lin_params
pops_corrections = result.pops_corrections

hypercube_eigs, hypercube_bounds = POPSRegression.hypercube(POPS_corrections)
committee, dΘ = POPSRegression.sample_hypercube(hypercube_eigs, hypercube_bounds, lin_params)

Priors = [ACEpotentials._make_prior(model,i,nothing) for i=2:6]
for P_ in Priors
    Ap= Diagonal(W) * A / P_
    Y = readdlm("high_entropy_POPS/Y.csv", ',')
    Y = Y[:,1]
    Y = W .* Y
    A_reg = (Ap'*Ap .+ (P_'*P_ .* (1.0 / size(Ap, 1))))
    push!(lin_params_priors, P_ \ (A_reg \ Ap' * Y))
end
# co_ps_vec = [committee[:, i] for i = 1:size(committee,2)]
# ACEpotentials.Models.set_committee!(model, co_ps_vec)

# # using Distributions
# # multivariate_normal = MvNormal(lin_params, dΘ + 1e-6 * I)
# # samples = rand(multivariate_normal, 10)

# include("get_Hessian.jl")

using GeometryOptimization
atoms = bulk(:Al)
results = minimize_energy!(atoms, model; variablecell=true)
optsystem = results.system
a_ref = optsystem.cell.cell_vectors[1][2]*2

using Test
@test strained_cell_energy(@SVector [0.0,0.0,0.0,0.0,0.0,0.0];
                           model=model, element=:Al, a=a_ref) ==
      ACEpotentials.potential_energy(bulk(:Al; a=a_ref), model)

ε0 = @SVector zeros(6)

volume(sys) = dot(cross(sys.cell.cell_vectors[1], sys.cell.cell_vectors[2]), sys.cell.cell_vectors[3])
unitless_volume(sys) = ustrip(volume(sys))

V = unitless_volume(bulk(:Al; a=a_ref))
V_ref = unitless_volume(bulk(:Al; a=a_ref))
# H_GPa = 160.2176621 .* H ./ V

H_6x6xd = Array{Float64}(undef, 6, 6, length(lin_params))
f(ε) = strained_cell_design(ε; model=model, element=:Al, a=a_ref)
for k in 1:length(lin_params)
    g(ε) = ustrip.(f(ε)[k])
    H_6x6xd[:, :, k] = ForwardDiff.hessian(g, ε0)
end

# Keep a human-readable CSV dump per lattice constant (same tensor, flattened by writedlm)
a_tag = replace(string(round(ustrip(u"Å", a_ref), digits=6)), "." => "p")
writedlm("high_entropy_POPS/undotted_Hessian_a_$(a_tag).csv", H_6x6xd, ',')
writedlm("undotted_Hessian.csv", H_6x6xd, ',')

function compute_elastic_tensor_from_H(H_6x6xd_slice::Matrix, POPS_delta::Vector)
    """
    Compute the full 6×6 elastic tensor (Voigt notation) from the Hessian tensor.
    
    H_6x6xd_slice is a 6×6 slice from the H_6x6xd tensor
    POPS_delta is the parameter vector offset
    
    Applies Voigt convention corrections:
    - C[i,i] for i in 4:6 (shear): multiply by 4
    - C[i,j] mixed normal-shear: multiply by 2
    """
    # Contract the Hessian with parameters
    H_mat = H_6x6xd_slice * POPS_delta
    
    # Create corrected elastic tensor
    C = zeros(6, 6)
    for i in 1:6
        for j in 1:6
            C[i,j] = H_mat[i,j]
            
            # Apply Voigt convention corrections
            shear_i = (i >= 4)
            shear_j = (j >= 4)
            
            if shear_i && shear_j
                # Shear-shear: factor of 4
                C[i,j] *= 4.0
            elseif shear_i || shear_j
                # Mixed normal-shear: factor of 2
                C[i,j] *= 2.0
            end
            # Normal-normal: no correction
        end
    end
    
    return C
end

function dot_with_H_full(POPS_delta, H_6x6xd; apply_voigt_correction=false)
    """
    Extract elastic constants from Hessian with optional Voigt corrections.
    
    NOTE: Voigt correction factor of 4 is NOT NEEDED here because:
    - Your Voigt input components already represent engineering shear strains γ = 2ε
    - The 0.5 factor in Voigt_strain_to_3x3 converts γ → ε for the 3×3 tensor
    - d²E/dγ² computed by ForwardDiff IS the physical elastic constant
    
    ONLY apply corrections if computing d²E/dε² (mathematical strain derivatives).
    
    Returns:
    - C11s_big, C12s_big, C44s_big: the cubic averages
    - full_tensors: list of 6×6 elastic tensors
    """
    C11s_big = []; C12s_big = []; C44s_big = []; full_tensors = []

    for i=1:size(POPS_delta,1)
        # Reshape and contract to get 6x6 Hessian for this parameter set
        H_list = [reshape(H_6x6xd[:,:,k], 6, 6) for k in 1:size(H_6x6xd, 3)]
        H_contracted = sum(H_list[k] .* POPS_delta[i,k] for k in 1:length(POPS_delta[i,:]))
        
        C_mat = H_contracted  # No correction needed
        
        push!(full_tensors, C_mat)
        
        # Extract cubic averages
        C11 = (C_mat[1,1] + C_mat[2,2] + C_mat[3,3]) / 3
        C12 = (C_mat[1,2] + C_mat[1,3] + C_mat[2,3]) / 3
        C44 = (C_mat[4,4] + C_mat[5,5] + C_mat[6,6]) / 3
        
        push!(C11s_big, C11)
        push!(C12s_big, C12)
        push!(C44s_big, C44)
    end
    
    return (C11s_big, C12s_big, C44s_big, full_tensors)
end

POPS_delta = POPS_corrections .+ lin_params'

C11s_big, C12s_big, C44s_big, _ = dot_with_H_full(POPS_corrections .+ lin_params', H_6x6xd)
negative_indices     = findall(C44s_big .< 0.0)
negative_POPS_deltas = POPS_delta[negative_indices, :]
# writedlm("high_entropy_POPS/negative_C44_POPS_coeffs.csv", negative_POPS_deltas, ',')


# function constrained_OLS(X_train, Y_train, constraint_matrix, constraint_value)
#     constrained_pops_parameters = zeros(length(Y_train), size(X_train, 2))
#     H = X_train' * X_train
#     b = -X_train' * Y_train
#     A_full = constraint_matrix
#     l_full = [constraint_value]
#     u_full = [constraint_value]
#     model = OSQP.Model()
#     OSQP.setup!(model; P=sparse(H), q=b, A=sparse(A_full), l=l_full, u=u_full, 
#                 alpha=0.5, verbose=true, max_iter=5000, check_termination=1000)
#     results = OSQP.solve!(model)
#     return results.x
# end
# POPS_ensemble = POPS_corrections .+ lin_params'
# # Here we are checking if any potentials give negative elastic constants. Turns out they do. So, we must constrain!!
# im_indices = []
# function test_for_negative_eigvals(POPS_ensemble, H_6x6xd, im_indices)
#     for i = 1:size(POPS_ensemble, 1)
#         H_temp = reshape(H_6x6xd, 36, :) * POPS_ensemble[i,:]
#         H_temp = reshape(H_temp, 6, 6)
#         eigs_temp = eigvals(H_temp)
#         if (typeof(eigs_temp) == Vector{ComplexF64} || sum(real(eigs_temp) .>= 0.0) != length(eigs_temp))
#             push!(im_indices, i)
#             println(i, " is imaginary")
#             println(eigs_temp)
#         end
#     end
#     return im_indices
# end
# # If you check with the POPS corrections of the "naughty" POPS members gone from the corrections matrix, you still have bad actors!
# # So, it is important to reject these unphysical members, even if we constrain something about our ensemble.
# naughty_POPS_inds = setdiff(collect(1:size(POPS_ensemble, 1)), im_indices)
# hypercube_eigs, hypercube_bounds = POPSRegression.hypercube(POPS_corrections[naughty_POPS_inds, :])
# committee, dΘ = POPSRegression.sample_hypercube(hypercube_eigs, hypercube_bounds, lin_params; number_of_committee_members=10000)
# co_ps_vec = [committee[:, i] for i = 1:size(committee,2)]
# hyp_im_indices =[]
# for i = 1:length(co_ps_vec)
#     H_temp = reshape(H_6x6xd, 36, :) * co_ps_vec[i]
#     H_temp = reshape(H_temp, 6, 6)
#     eigs_temp = eigvals(H_temp)
#     if (typeof(eigs_temp) == Vector{ComplexF64} || sum(real(eigs_temp) .>= 0.0) != length(eigs_temp))
#         push!(hyp_im_indices, i)
#         println(i, " is imaginary")
#         println(eigs_temp)
#     end
# end


# function constrained_pops(X_train, Y_train, constraint_matrix, lower_constraint_value, upper_constraint_value; members_to_constrain=1:length(Y_train))
#     constrained_pops_parameters = zeros(length(collect(members_to_constrain)), size(X_train, 2))
#     H = X_train' * X_train
#     b = - X_train' * Y_train
#     j=0
#     for i=members_to_constrain
#         j+=1
#         A_full = vcat(constraint_matrix, X_train[i,:]')
#         l_full = vcat(lower_constraint_value, Y_train[i])
#         u_full = vcat(upper_constraint_value, Y_train[i])
#         println(size(A_full), size(l_full))
#         model = OSQP.Model()
#         OSQP.setup!(model; P=sparse(H), q=b, A=sparse(A_full), l=l_full, u=u_full, 
#                     alpha=0.5, verbose=true, max_iter=5_000, check_termination=500)
#         results = OSQP.solve!(model)
#         constrained_pops_parameters[j,:] = results.x
#     end
#     return constrained_pops_parameters
# end

# all_C_ij = vcat(H_6x6xd[1,1,:]', H_6x6xd[1,2,:]', H_6x6xd[4,4,:]')
# lower_c = [0.0, 0.0, 0.0]
# upper_c = [Inf, Inf, Inf]
# constrained_pops_parameters = constrained_pops(Ap, Y, all_C_ij / P, lower_c, upper_c; members_to_constrain=im_indices)

# H_temp = reshape(H_6x6xd, 36, :) * (P \ constrained_pops_parameters')
# for i = 1:size(H_temp, 2)
#     H_temp_ = reshape(H_temp[:,i], 6, 6)
#     println(H_temp_[1,1])
#     println(H_temp_[1,2])
#     println(H_temp_[4,4])
# end


# POPS_corrections[im_indices, :] .= (P \ constrained_pops_parameters')'
# hypercube_eigs, hypercube_bounds = POPSRegression.hypercube(POPS_corrections)

# function rejection_sample_hypercube(eigvecs::AbstractMatrix{Float64}, bounds::AbstractMatrix{Float64}, coeffs::Vector{Float64}, inequality_constraints, lower_c, higher_c; number_of_committee_members::Int64 = 1000)
#     lower, upper = bounds[1, :], bounds[2, :]

#     U = rand(Float64, (number_of_committee_members, size(lower, 1)))

#     committee = eigvecs * (lower[:, :]' .+ (upper .- lower)[:,:]' .* U)'
#     δθ        = committee * committee' ./ size(committee, 2)
#     for i=1:number_of_committee_members
#         temp_coeffs = coeffs .+ committee[:,i]
#         inequality_satisfied = sum(lower_c .<= inequality_constraints * temp_coeffs .<= higher_c) == length(lower_c)
#         if (inequality_satisfied)
#             committee = coeffs[:,:] .+ committee
#         end
#     end
#     return committee, δθ  
# end
# constrained_committee, δθ = rejection_sample_hypercube(hypercube_eigs, hypercube_bounds, lin_params, all_C_ij, lower_c, upper_c)
# constrained_co_ps_vec = [constrained_committee[:, i] for i = 1:size(constrained_committee,2)]

function constrained_ridge_regression(X_train, Y_train, Gamma, constraint_matrix, constraint_bounds)
    H = (X_train' * X_train .+ (1.0 / (size(X_train, 1)) .* Gamma' * Gamma))
    b = - X_train' * Y_train
    model = OSQP.Model()
    OSQP.setup!(model; P=sparse(H), q=b, A=sparse(constraint_matrix / Gamma), l=constraint_bounds[1], u=constraint_bounds[2],
                max_iter=500_000, check_termination=1_000, verbose=true)
    results = OSQP.solve!(model)
    return Gamma \ results.x
end
    
function constrained_pops(X_train, Y_train, Gamma, constraint_matrix, constraint_bounds; members_to_constrain=1:length(Y_train))
    constrained_pops_parameters = zeros(length(members_to_constrain),size(X_train, 2))#length(collect(members_to_constrain)), size(X_train, 2))
    H = (X_train' * X_train .+ (1.0 / (size(X_train, 1)) .* Gamma' * Gamma))
    b = - X_train' * Y_train
    Threads.@threads for idx in 1:length(members_to_constrain)
        i = members_to_constrain[idx]
        A_full = vcat(X_train[i,:]', constraint_matrix) 
        l_full = vcat([Y_train[i]], constraint_bounds[1])
        u_full = vcat([Y_train[i]], constraint_bounds[2])
        model = OSQP.Model()
        OSQP.setup!(model; P=sparse(H), q=b, A=sparse(A_full / Gamma), l=l_full, u=u_full,
                    max_iter=500_000, check_termination=200, verbose=false)
        results = OSQP.solve!(model)
        constrained_pops_parameters[idx,:] = Gamma \ results.x
    end
    return constrained_pops_parameters
end

# numer_of_constrained = length(Y)

# regularized_OLS = P \ ((Ap'*Ap .+ P'*P .* (1.0/size(Ap,1)))\(Ap' * Y))
# first_unconstrained_pops = ((POPSRegression.corrections(Ap, Y, P; leverage_percentile=0.0)[1:numer_of_constrained,:])' .+ regularized_OLS)'
# constrained_pops_parameters = base_constrained_pops(Ap, Y, P; members_to_constrain=1:numer_of_constrained)

# writedlm("small_high_entropy_ace_model/constrained_pops.csv", constrained_pops_parameters, ',')
# writedlm("small_high_entropy_ace_model/unconstrained_pops.csv", first_unconstrained_pops, ',')

# @test maximum(abs.(first_unconstrained_pops .- constrained_pops_parameters)) < 1e-5

# # include("/storage/astro2/phupfb/PhD/acestuff/new_ACE/small_high_entropy_ace_model/test_analytic_against_constrained.jl")
