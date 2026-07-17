using OSQP, ACEWorkflow

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
                    max_iter=500_000, check_termination=10, verbose=false)
        results = OSQP.solve!(model)
        constrained_pops_parameters[idx,:] = Gamma \ results.x
    end
    return constrained_pops_parameters
end

using OSQP, ACEWorkflow

function constrained_pops(X_train, Y_train, Gamma, constraint_matrix, constraint_bounds; members_to_constrain=1:length(Y_train))
    constrained_pops_parameters = zeros(length(members_to_constrain), size(X_train, 2))
    H = (X_train' * X_train .+ (1.0 / (size(X_train, 1)) .* Gamma' * Gamma))
    b = - X_train' * Y_train

    model = OSQP.Model()
    first = true

    for idx in 1:length(members_to_constrain)
        i = members_to_constrain[idx]
        A_full = vcat(X_train[i,:]', constraint_matrix)
        l_full = vcat([Y_train[i]], constraint_bounds[1])
        u_full = vcat([Y_train[i]], constraint_bounds[2])
        A_sparse = sparse(A_full / Gamma)

        if first
            OSQP.setup!(model; P=sparse(H), q=b, A=A_sparse, l=l_full, u=u_full,
                        max_iter=500_000, check_termination=10, verbose=false)
            first = false
        else
            OSQP.update!(model; Ax=A_sparse.nzval, l=l_full, u=u_full)
        end

        results = OSQP.solve!(model)
        constrained_pops_parameters[idx,:] = Gamma \ results.x
    end
    return constrained_pops_parameters
end