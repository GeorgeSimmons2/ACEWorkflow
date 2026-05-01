using OSQP
using SparseArrays
using LinearAlgebra
using ExtXYZ
using Statistics
using CairoMakie
using Printf
using DelimitedFiles
using ACEpotentials

function corrections(X, Y, Gamma; leverage_percentile=0.0)
    C = (Gamma' * Gamma ./ size(X,1) .+ X' * X)
    A = C \ X'
    leverage = diag(X * A)
    coeffs = C \ (X' * Y)
    errors = Y .- (X * coeffs)
    
    leverage_threshold = quantile(leverage, leverage_percentile)
    mask = leverage .>= leverage_threshold
    pointwise_corrections = A[:,mask]'
    pointwise_corrections = pointwise_corrections .* (errors[mask] ./ leverage[mask])
    
    return pointwise_corrections / Gamma, Gamma \ coeffs
end

function constrained_pops_update(X_i, y_i, theta_nominal, A_constraints, l_constraints, u_constraints)
    n_params = length(theta_nominal)
    
    # QP: minimize ||δθ||² where θ = theta_nominal + δθ (correction)
    # Constraint: X_i'(θ_nominal + δθ) = y_i  =>  X_i' δθ = y_i - X_i'θ_nominal
    # And: A (θ_nominal + δθ) ∈ [l, u]  =>  A δθ ∈ [l - Aθ_nominal, u - Aθ_nominal]
    
    P_qp = sparse(I(n_params))
    q_qp = zeros(n_params)
    
    # Equality constraint: X_i' δθ = y_i - X_i'θ_nominal
    A_eq = reshape(X_i, 1, n_params)
    residual = y_i - dot(X_i, theta_nominal)
    
    # Shift inequality constraints by nominal coefficients
    A_ineq_shifted = A_constraints
    l_ineq_shifted = l_constraints - A_constraints * theta_nominal
    u_ineq_shifted = u_constraints - A_constraints * theta_nominal
    
    # Combine constraints
    A_combined = vcat(A_eq, A_ineq_shifted)
    l_combined = vcat([residual], l_ineq_shifted)
    u_combined = vcat([residual], u_ineq_shifted)
    
    # Solve QP for correction
    prob = OSQP.Model()
    OSQP.setup!(prob; P=P_qp, q=q_qp, A=sparse(A_combined), l=l_combined, u=u_combined,
                verbose=false, eps_abs=1e-4, eps_rel=1e-4, max_iter=100)
    results = OSQP.solve!(prob)
    
    # Return the correction (not the full coefficient)
    return results.x
end

# Iterative constrained POPS
# For each training point, generate a constrained correction ensemble member
# Returns: matrix of constrained corrections (each row = one member's corrections)
function iterative_constrained_pops(X, Y, Gamma, A_constraints, l_constraints, u_constraints; H=nothing, nominal_coeffs=nothing)
    if (nominal_coeffs != nothing)
        # Get nominal unconstrained POPS solution
        _, nominal_coeffs = corrections(X, Y, Gamma; leverage_percentile=0.0)
    end

    n_params = size(X, 2)
    n_points = size(X, 1)
    constrained_corrections = []
    
    # For each training point, solve constrained QP for that point's correction
    for i in 1:n_points
        X_i = X[i, :]
        y_i = Y[i]
        delta_theta = constrained_pops_update(X_i, y_i, nominal_coeffs, A_constraints, l_constraints, u_constraints)
        push!(constrained_corrections,  delta_theta)
    end
    
    # Return corrections as matrix (each row = one member's corrections)
    return hcat(constrained_corrections...)' / Gamma
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
suffix = "_14_4"

A_constraints = readdlm("/storage/astro2/phupfb/PhD/acestuff/new_ACE/ground_state_lattice_design_matrix_14_4_e_is_minus_12.810869094250535.csv", ',')
l_constraints = [-12.810869094250535]
u_constraints = [-12.810869094250535]

W = readdlm("/storage/astro2/phupfb/PhD/acestuff/new_ACE/W_14_4.csv", ',')[:,1]
A = readdlm("/storage/astro2/phupfb/PhD/acestuff/new_ACE/A_14_4.csv", ',')
Y = readdlm("/storage/astro2/phupfb/PhD/acestuff/new_ACE/Y_14_4.csv", ',')[:,1]
P = readdlm("/storage/astro2/phupfb/PhD/acestuff/new_ACE/P_14_4.csv", ',')

Ap = Diagonal(W) * A / P
Y = W .* Y
coeffs = P \ (Ap \ Y)



corrections_with_constraints = iterative_constrained_pops(Ap, Y, P, A_constraints, l_constraints, u_constraints; H=nothing, nominal_coeffs=coeffs)'

# pointwise_corrections = corrections_with_constraints#readdlm("/storage/astro2/phupfb/PhD/acestuff/new_ACE/POPS_corrections_stratified_sampling.csv", ',')
model, _ = ACEpotentials.load_model("Al_model$(suffix).json")
# hypercube_eigs, hypercube_bounds = hypercube(corrections_with_constraints; percentile_clipping=0.5)
# committee, _ = sample_hypercube(hypercube_eigs, hypercube_bounds, coeffs; number_of_committee_members=500)
co_ps_vec = [corrections_with_constraints[:,i] .+ coeffs for i = 1:size(corrections_with_constraints,2)]
ACEpotentials.Models.set_linear_parameters!(model, coeffs)
ACEpotentials.Models.set_committee!(model, co_ps_vec)
test = ExtXYZ.load("high_entropy_pops/manual_df_test_Al.xyz")
test_E = []
test_E_predictions = []
test_co_E_predictions = []
ev_val = counter = 0  
co_E_range = []
using Unitful
using AtomsCalculators
for (i, at) in enumerate(test[51:500])

    energy = at[:dft_energy]

    push!(test_E, energy)
    E, co_E = ustrip.(@committee AtomsCalculators.potential_energy(at, model))
    push!(test_E_predictions, E)
    push!(co_E_range, std(co_E))
    println(i)
end
using Statistics
itr = test_E
RMSE_ = sqrt.(sum(abs2.(itr .- mean(itr))) / (length(itr) - 1))
# ev_val /= counter
# ev_val *= 100
using CairoMakie

# # Create figure and axis
fig = CairoMakie.Figure()
ax = CairoMakie.Axis(fig[1, 1],
    xlabel = "Test errors / eV",
    ylabel = "Uncertainty estimates / eV"
)

# Data
xdata = abs.(test_E .- test_E_predictions)
ydata = co_E_range ./ 2

# Scatter plot
CairoMakie.scatter!(ax, xdata, ydata)

# Add identity line
max_val = maximum([maximum(xdata), maximum(ydata)])
CairoMakie.lines!(ax, [0, max_val], [0, max_val],
    linestyle = :dash,
    label = "Identity"
)

# Add EV label (invisible dummy line)
CairoMakie.lines!(ax, [NaN], [NaN],
    label = "EV = $(ev_val)%",
    color = :white
)

# Axis limits and legend
padding = 0.05 * max_val
CairoMakie.xlims!(ax, 0, max_val + padding)
CairoMakie.ylims!(ax, 0, max_val + padding)
CairoMakie.axislegend(ax, position = :rb)
clipping = "0"
# Save to file
CairoMakie.save("./constrained_errors.png", fig)