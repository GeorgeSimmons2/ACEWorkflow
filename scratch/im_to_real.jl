function test_for_negative_elastic_consts(POPS_ensemble, H_6x6xd, negative_indices)
    for i = 1:size(POPS_ensemble, 1)
        H_temp = reshape(H_6x6xd, 36, :) * POPS_ensemble[i,:]
        H_temp = reshape(H_temp, 6, 6)
        volume(sys) = dot(cross(sys.cell.cell_vectors[1], sys.cell.cell_vectors[2]), sys.cell.cell_vectors[3])
        unitless_volume(sys) = ustrip(volume(sys))

        V = unitless_volume(bulk(:Al))
        H_GPa = 160.2176621 .* H_temp ./ V
        if (H_GPa[1,1] < 0.0 || H_GPa[1,2] < 0.0 || H_GPa[4,4] < 0.0)
            push!(negative_indices, i)
            println("Negative elastic constant: ")
            println(i)
        end
    end
    return negative_indices
end
negative_inds = test_for_negative_elastic_consts(pops_delta, H_6x6xd, [])
chosen_ind = negative_inds[end]
chosen_vec = pops_delta[chosen_ind,:]

function constrained_pops(X_train, Y_train, Gamma, constraint_matrix, constraint_lower, constraint_upper; members_to_constrain=1:length(Y_train))
    constrained_pops_parameters = zeros(length(members_to_constrain),size(X_train, 2))#length(collect(members_to_constrain)), size(X_train, 2))
    H = (X_train' * X_train .+ (1.0 / (size(X_train, 1)) .* Gamma' * Gamma))
    b = - X_train' * Y_train
    for (idx, i) in enumerate(members_to_constrain)
        A_full = vcat(X_train[i,:]', constraint_matrix)
        l_full = vcat(Y_train[i], constraint_lower)
        u_full = vcat(Y_train[i], constraint_upper)
        model = OSQP.Model()
        OSQP.setup!(model; P=sparse(H), q=b, A=sparse(A_full), l=l_full, u=u_full,
                    max_iter=500_000, check_termination=1_000, verbose=true)
        results = OSQP.solve!(model)
        constrained_pops_parameters[idx,:] = Gamma \ results.x
    end
    return constrained_pops_parameters
end

constraints_matrix = vcat(vcat(H_6x6xd[4,4,:]', H_6x6xd[1,1,:]'), H_6x6xd[1,2,:]') / P
constraints_matrix = H_6x6xd[4,4,:]' / P
constrained_chosen_vec = constrained_pops(Ap, Y, P, constraints_matrix, [0.0], [Inf]; members_to_constrain=[chosen_ind])

ACEpotentials.Models.set_linear_parameters!(model, constrained_chosen_vec[1,:])
ACEpotentials.save_model(model, "constrained_pops_model.json")
ACEpotentials.Models.set_linear_parameters!(model, pops_delta[chosen_ind,:])
ACEpotentials.save_model(model, "unconstrained_pops_model.json")
reshape(reshape(H_6x6xd, 36, :) * constrained_chosen_vec[1,:], 6,6)