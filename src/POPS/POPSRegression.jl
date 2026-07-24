module POPSRegression
using LinearAlgebra, Statistics, Random
export corrections, hypercube, sample_hypercube, rejection_sample_hypercube
export gaussian_proposal, sample_gaussian, rejection_sample_gaussian


function corrections(X::AbstractMatrix{Float64}, Y::Vector{Float64}, Gamma::AbstractMatrix{Float64}; leverage_percentile::Float64 = 0.5, lambda::Float64 = 1.0 / size(X,1), coeffs=nothing)
    C      = (Gamma' * Gamma .* lambda .+ X' * X)
    A      = C \ X'
    leverage = diag(X * A)
    if (coeffs == nothing)
        coeffs = C \ (X' * Y)
    end
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

"""
    rejection_sample_hypercube(eigvecs, bounds, coeffs, is_feasible::Function;
                               number_of_committee_members=50,
                               max_attempts=10_000*number_of_committee_members)

Predicate form: `is_feasible(c)::Bool` is called with the full coefficient
vector c = coeffs + δ of each proposal and may implement any acceptance test —
linear constraints, eigenvalue positivity of a parameter-contracted operator,
etc.  Put cheap checks first inside the predicate; it is called once per
proposal.  All other behaviour matches the linear-constraint form.
"""
function rejection_sample_hypercube(eigvecs::AbstractMatrix{Float64}, bounds::AbstractMatrix{Float64}, coeffs::Vector{Float64},
                                    is_feasible::Function;
                                    number_of_committee_members::Int64 = 50,
                                    max_attempts::Int64 = 10_000 * number_of_committee_members)
    lower, upper = bounds[1, :], bounds[2, :]

    perturbations = zeros(Float64, length(coeffs), number_of_committee_members)
    accepted = 0
    attempts = 0
    while accepted < number_of_committee_members
        if attempts >= max_attempts
            error("rejection_sample_hypercube: accepted only $accepted / " *
                  "$number_of_committee_members members after $attempts proposals. " *
                  "The feasible fraction of the hypercube is very small — check that " *
                  "the mean coefficients satisfy the constraints, or raise max_attempts.")
        end
        attempts += 1
        u = rand(Float64, size(lower, 1))
        δ = eigvecs * (lower .+ (upper .- lower) .* u)
        if is_feasible(coeffs .+ δ)
            accepted += 1
            perturbations[:, accepted] = δ
        end
    end
    @info "rejection_sample_hypercube: accepted $number_of_committee_members / $attempts proposals " *
          "($(round(100 * number_of_committee_members / attempts; digits=2))% acceptance)"

    δθ        = perturbations * perturbations' ./ number_of_committee_members
    committee = coeffs[:, :] .+ perturbations

    return committee, δθ
end

"""
    rejection_sample_hypercube(eigvecs, bounds, coeffs, constraint_matrix, constraint_bounds;
                               number_of_committee_members=50,
                               max_attempts=10_000*number_of_committee_members)

Draw committee members from the axis-aligned hypercube proposal in the POPS
eigenbasis and accept only draws whose full coefficient vector c = coeffs + δ
satisfies the physical constraints l ≤ A c ≤ u.

`bounds` should come from the **unclipped** proposal, `hypercube(...)` with the
default `percentile_clipping = 0.0`: the feasible-set geometry is imposed
exactly by rejection rather than approximated by shrinking the box, so the
accepted ensemble is uniformly distributed over (hypercube ∩ feasible set).

`constraint_bounds` is a tuple `(l, u)` of lower/upper bound vectors, one entry
per row of `constraint_matrix`. Rows with `l[i] == u[i]` (equality constraints)
have zero acceptance probability under a continuous proposal and raise an
`ArgumentError` — enforce equalities in the constrained fit that generates the
proposal, and pass only the inequality rows here.

Returns `(committee, δθ)` in the same format as [`sample_hypercube`](@ref):
`committee` is `length(coeffs) × N` with members as columns, and `δθ` is the
second-moment matrix of the accepted perturbations.
"""
function rejection_sample_hypercube(eigvecs::AbstractMatrix{Float64}, bounds::AbstractMatrix{Float64}, coeffs::Vector{Float64},
                                    constraint_matrix::AbstractMatrix, constraint_bounds::Tuple;
                                    kwargs...)
    lower_constraint, upper_constraint = constraint_bounds
    if any(lower_constraint .== upper_constraint)
        throw(ArgumentError(
            "constraint rows with equal lower and upper bounds (equality constraints) " *
            "can never be satisfied by rejection sampling from a continuous proposal; " *
            "pass only the inequality rows."))
    end
    is_feasible = c -> begin
        Ac = constraint_matrix * c
        all(lower_constraint .<= Ac) && all(Ac .<= upper_constraint)
    end
    return rejection_sample_hypercube(eigvecs, bounds, coeffs, is_feasible; kwargs...)
end

"""
    gaussian_proposal(pointwise_corrections) -> (eigvecs, sigmas)

Eigenbasis of the correction cloud with per-direction standard deviations of
the projections, for use with [`sample_gaussian`](@ref).

Why this exists: the axis-aligned bounding box used by [`hypercube`](@ref)
sets each direction's range from the min/max projection over ALL corrections.
In d dimensions a uniform box draw combines near-extreme values in every
direction simultaneously, so its typical norm is ~√d × a per-direction
extreme — orders of magnitude larger than any actual correction when d is a
few hundred.  A Gaussian matched to the cloud's covariance draws members at
the cloud's true scale instead.
"""
function gaussian_proposal(pointwise_corrections::AbstractMatrix{Float64})
    eig = eigen(Symmetric(pointwise_corrections' * pointwise_corrections))
    eigvals, eigvecs = eig.values, eig.vectors

    mask    = eigvals .> maximum(eigvals) * 1e-8
    eigvecs = eigvecs[:, mask]

    projected = pointwise_corrections * eigvecs
    sigmas    = vec(std(projected; dims=1))

    return eigvecs, sigmas
end

"""
    sample_gaussian(eigvecs, sigmas, coeffs; number_of_committee_members=50, scale=1.0)

Draw committee members `coeffs .+ eigvecs * (scale .* sigmas .* z)`, z ~ N(0,I).
Returns `(committee, δθ)` in the same format as [`sample_hypercube`](@ref).
"""
function sample_gaussian(eigvecs::AbstractMatrix{Float64}, sigmas::AbstractVector{Float64}, coeffs::Vector{Float64};
                         number_of_committee_members::Int64 = 50, scale::Float64 = 1.0)
    Z = randn(length(sigmas), number_of_committee_members)
    perturbations = eigvecs * ((scale .* sigmas) .* Z)
    δθ            = perturbations * perturbations' ./ number_of_committee_members
    committee     = coeffs[:, :] .+ perturbations
    return committee, δθ
end

"""
    rejection_sample_gaussian(eigvecs, sigmas, coeffs, is_feasible::Function; kwargs...)
    rejection_sample_gaussian(eigvecs, sigmas, coeffs, constraint_matrix, constraint_bounds; kwargs...)

Gaussian-proposal analogue of [`rejection_sample_hypercube`](@ref): draw from
the covariance-matched proposal and keep members passing the feasibility test.
"""
function rejection_sample_gaussian(eigvecs::AbstractMatrix{Float64}, sigmas::AbstractVector{Float64}, coeffs::Vector{Float64},
                                   is_feasible::Function;
                                   number_of_committee_members::Int64 = 50,
                                   max_attempts::Int64 = 10_000 * number_of_committee_members,
                                   scale::Float64 = 1.0)
    perturbations = zeros(Float64, length(coeffs), number_of_committee_members)
    accepted = 0
    attempts = 0
    while accepted < number_of_committee_members
        if attempts >= max_attempts
            error("rejection_sample_gaussian: accepted only $accepted / " *
                  "$number_of_committee_members members after $attempts proposals.")
        end
        attempts += 1
        δ = eigvecs * ((scale .* sigmas) .* randn(length(sigmas)))
        if is_feasible(coeffs .+ δ)
            accepted += 1
            perturbations[:, accepted] = δ
        end
    end
    @info "rejection_sample_gaussian: accepted $number_of_committee_members / $attempts proposals " *
          "($(round(100 * number_of_committee_members / attempts; digits=2))% acceptance)"

    δθ        = perturbations * perturbations' ./ number_of_committee_members
    committee = coeffs[:, :] .+ perturbations
    return committee, δθ
end

function rejection_sample_gaussian(eigvecs::AbstractMatrix{Float64}, sigmas::AbstractVector{Float64}, coeffs::Vector{Float64},
                                   constraint_matrix::AbstractMatrix, constraint_bounds::Tuple;
                                   kwargs...)
    lower_constraint, upper_constraint = constraint_bounds
    if any(lower_constraint .== upper_constraint)
        throw(ArgumentError("equality rows cannot be enforced by rejection sampling; pass only inequality rows."))
    end
    is_feasible = c -> begin
        Ac = constraint_matrix * c
        all(lower_constraint .<= Ac) && all(Ac .<= upper_constraint)
    end
    return rejection_sample_gaussian(eigvecs, sigmas, coeffs, is_feasible; kwargs...)
end

# Alternate argument order (constraint_bounds before constraint_matrix), kept
# for backwards compatibility with existing scripts.
rejection_sample_hypercube(eigvecs::AbstractMatrix{Float64}, bounds::AbstractMatrix{Float64}, coeffs::Vector{Float64},
                           constraint_bounds::Tuple, constraint_matrix::AbstractMatrix; kwargs...) =
    rejection_sample_hypercube(eigvecs, bounds, coeffs, constraint_matrix, constraint_bounds; kwargs...)

# Single constraint row given as a plain vector.
rejection_sample_hypercube(eigvecs::AbstractMatrix{Float64}, bounds::AbstractMatrix{Float64}, coeffs::Vector{Float64},
                           constraint_row::AbstractVector, constraint_bounds::Tuple; kwargs...) =
    rejection_sample_hypercube(eigvecs, bounds, coeffs, reshape(constraint_row, 1, :), constraint_bounds; kwargs...)

end
