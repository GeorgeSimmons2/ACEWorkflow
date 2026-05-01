using OSQP
using SparseArrays
using LinearAlgebra, Statistics

# Define problem data
P = sparse([4. 1.; 1. 2.])
q = [1.; 1.]
A = sparse([1. 1.; 1. 0.; 0. 1.])
l = [1.; 0.; 0.]
u = [1.; 0.7; 0.7]

# Crate OSQP object
prob = OSQP.Model()

# Setup workspace and change alpha parameter
OSQP.setup!(prob; P=P, q=q, A=A, l=l, u=u, alpha=1)

# Solve problem
results = OSQP.solve!(prob)

using Test

@testset "QP Tests" begin
    P = sparse([4. 1.; 1. 2.])
    q = [1.; 1.]
    A = sparse([1. 1.; 1. 0.; 0. 1.])
    l = [1.; 0.; 0.]
    u = [1.; 0.7; 0.7]

    # Crate OSQP object
    prob = OSQP.Model()

    # Setup workspace and change alpha parameter
    OSQP.setup!(prob; P=P, q=q, A=A, l=l, u=u, alpha=1)

    # Solve problem
    results = OSQP.solve!(prob)
    x = results.x
    println(isapprox.(A * x .- u, zero(length(u)), rtol=1e-1))
    @test sum((A * x .- l .> zeros(length(l)) .|| A * x .- l .≈ zero(length(l)))) == size(A, 1)
    @test sum((A * x .- u .< zeros(length(u)) .|| isapprox.(A * x .- u, zero(length(u)), rtol=1))) == size(A, 1)
end    


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
using OSQP
using SparseArrays
using LinearAlgebra

# Constrained corrections that satisfy inequality constraints
function constrained_corrections(
    X::AbstractMatrix{Float64}, 
    Y::Vector{Float64}, 
    Gamma::AbstractMatrix{Float64},
    A_ineq::AbstractMatrix{Float64},
    l_ineq::Vector{Float64},
    u_ineq::Vector{Float64};
    leverage_percentile::Float64 = 0.5,
    lambda::Float64 = 1.0 / size(X,1)
)
    # Compute standard corrections first
    C = (Gamma' * Gamma .* lambda .+ X' * X)
    A = C \ X'
    leverage = diag(X * A)
    coeffs = C \ (X' * Y)
    errors = Y .- (X * coeffs)
    
    leverage_threshold = quantile(leverage, leverage_percentile)
    mask = leverage .>= leverage_threshold
    pointwise_corrections = A[:,mask]'
    pointwise_corrections = pointwise_corrections .* (errors[mask] ./ leverage[mask])
    pointwise_corrections = Gamma \ pointwise_corrections'
    
    # Now solve QP: minimize ||θ - θ_pops||² subject to inequality constraints
    n_params = size(pointwise_corrections, 2)
    
    # QP objective: minimize (θ - θ_nominal)' P (θ - θ_nominal)
    # Converted to: minimize (1/2) θ' P θ + q' θ where q = -P θ_nominal
    P_qp = sparse(2.0 * I(n_params))
    q_qp = -2.0 * vec(mean(pointwise_corrections, dims=1))
    
    # Inequality constraints: l_ineq ≤ A_ineq θ ≤ u_ineq
    A_qp = sparse(A_ineq)
    
    # Solve with OSQP
    prob = OSQP.Model()
    OSQP.setup!(prob; P=P_qp, q=q_qp, A=A_qp, l=l_ineq, u=u_ineq, verbose=false)
    results = OSQP.solve!(prob)
    
    return results.x, pointwise_corrections
end

# Example usage
P = sparse([4. 1.; 1. 2.])
q = [1.; 1.]
A_ineq = sparse([1. 1.; 1. 0.; 0. 1.])
l_ineq = [1.; 0.; 0.]
u_ineq = [1.; 0.7; 0.7]

prob = OSQP.Model()
OSQP.setup!(prob; P=P, q=q, A=A_ineq, l=l_ineq, u=u_ineq, alpha=1)
results = OSQP.solve!(prob)

using Test

@testset "QP Tests" begin
    x = results.x
    println("Solution satisfies constraints: ", isapprox.(A_ineq * x, u_ineq, rtol=1e-1))
    @test all((A_ineq * x .>= l_ineq) .| isapprox.(A_ineq * x, l_ineq, rtol=1e-6))
    @test all((A_ineq * x .<= u_ineq) .| isapprox.(A_ineq * x, u_ineq, rtol=1e-1))
end