module POPSRegression
using LinearAlgebra, Statistics, Random
export corrections, hypercube, sample_hypercube


function corrections(X::AbstractMatrix{Float64}, Y::Vector{Float64}, Gamma::AbstractMatrix{Float64}; leverage_percentile::Float64 = 0.5, lambda::Float64 = 1.0 / size(X,1))
    C      = (Gamma' * Gamma .* lambda .+ X' * X)
    A      = C \ X'
    leverage = diag(X * A)
    coeffs = C \ (X' * Y)
    errors = Y .- (X * coeffs)
    leverage_threshold = quantile(leverage, leverage_percentile)
    mask = leverage .>= leverage_threshold
    pointwise_corrections = A[:,mask]'
    pointwise_corrections = pointwise_corrections .* (errors[mask] ./ leverage[mask])
    pointwise_corrections = Gamma \ pointwise_corrections'
    return pointwise_corrections'
end

function hypercube(pointwise_corrections::AbstractMatrix{Float64}; percentile_clipping::Float64 = 0.0)
    eig = eigen(Symmetric(pointwise_corrections' * pointwise_corrections))
    eigvals = eig.values
    eigvecs = eig.vectors

    mask = eigvals .> maximum(eigvals) * 1e-8
    eigvecs = eigvecs[:, mask]
    eigvals = eigvals[mask]

    projections = eigvecs
    projected = pointwise_corrections * projections

    lower = [quantile(projected[:, j], percentile_clipping / 100) for j in 1:size(projected, 2)]
    upper = [quantile(projected[:, j], 1.0 - percentile_clipping / 100) for j in 1:size(projected, 2)]

    bounds = vcat(lower', upper')

    return eigvecs, bounds
end

function sample_hypercube(eigvecs::AbstractMatrix{Float64}, bounds::AbstractMatrix{Float64}, coeffs::Vector{Float64}; number_of_committee_members::Int64 = 50)
    lower, upper = bounds[1, :], bounds[2, :]

    U = rand(Float64, (number_of_committee_members, size(lower, 1)))

    committee = eigvecs * (lower[:, :]' .+ (upper .- lower)[:,:]' .* U)'
    δθ        = committee * committee' ./ size(committee, 2)

    committee = coeffs[:,:] .+ committee
    
    return committee, δθ  
end

function rejection_sample_hypercube(eigvecs::AbstractMatrix{Float64}, bounds::AbstractMatrix{Float64}, coeffs::Vector{Float64}, 
                                    constraint_matrix::AbstractVecOrMat, constraint_bounds::Tuple; 
                                    number_of_committee_members::Int64 = 50)
    
    lower_constraint, upper_constraint = constraint_bounds
    lower, upper = bounds[1, :], bounds[2, :]

    perturbations = zeros(Float64, length(coeffs), number_of_committee_members)
    count = 0
    while count < number_of_committee_members
        u = rand(Float64, size(lower, 1))
        δ = eigvecs * (lower .+ (upper .- lower) .* u)
        Ax = constraint_matrix * (coeffs .+ δ)
        if all(lower_constraint .< Ax) && all(Ax .< upper_constraint)
            count += 1
            perturbations[:, count] = δ
        end
    end

    δθ        = perturbations * perturbations' ./ number_of_committee_members
    committee = coeffs[:, :] .+ perturbations
    
    return committee, δθ  
end

function rejection_sample_hypercube(eigvecs::AbstractMatrix{Float64}, bounds::AbstractMatrix{Float64}, coeffs::Vector{Float64}, 
                                    constraint_bounds::Tuple, constraint_matrix::AbstractMatrix{Float64}; 
                                    number_of_committee_members::Int64 = 50)
    
    lower_constraint, upper_constraint = constraint_bounds
    lower, upper = bounds[1, :], bounds[2, :]

    perturbations = zeros(Float64, length(coeffs), number_of_committee_members)
    count = 0
    while count < number_of_committee_members
        u = rand(Float64, size(lower, 1))
        δ = eigvecs * (lower .+ (upper .- lower) .* u)
        Ax = constraint_matrix * (coeffs .+ δ)
        if all(lower_constraint .< Ax) && all(Ax .< upper_constraint)
            count += 1
            perturbations[:, count] = δ
        end
    end

    δθ        = perturbations * perturbations' ./ number_of_committee_members
    committee = coeffs[:, :] .+ perturbations
    
    return committee, δθ  
end

end