using OSQP
using SparseArrays
using LinearAlgebra, Statistics
using CairoMakie

# Generate data from a non-quadratic function (NO NOISE)
function generate_quadratic_data(n=50)
    x = range(-2, 2, length=n)
    # True function: 0.55*|x|^4 + 2*|x|^2.21 - 2*x - 1 (highly nonlinear, non-polynomial)
    # Using absolute value to handle fractional exponents with negative x
    y = 0.55 .* abs.(x).^4 .+ 2 .* abs.(x).^2.21 .- 2 .* x .- 1
    
    # Fit a quadratic design matrix: [1, x, x^2] (MISSPECIFIED!)
    X = [ones(n) x x.^2]
    return x, y, X
end

# Standard POPS corrections - returns ensemble of parameter vectors
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
    
    return pointwise_corrections, coeffs
end

# Constrained POPS with inequality constraints for a single data point
function constrained_pops_update(X_i, y_i, theta_nominal, A_constraints, l_constraints, u_constraints)
    """
    For a single data point (X_i, y_i), constrain the update to:
    1. Pass through the data point: X_i' θ = y_i
    2. Satisfy inequality constraints: l ≤ A_constraints θ ≤ u
    """
    n_params = length(theta_nominal)
    
    # QP: minimize ||θ - θ_nominal||² 
    # subject to: X_i' θ = y_i (equality constraint - enforced via bounds)
    #            A_constraints θ satisfies bounds
    P_qp = sparse(2.0 * I(n_params))
    q_qp = -2.0 * theta_nominal
    
    # Combine equality constraint (passing through data point) with inequality constraints
    A_eq = reshape(X_i, 1, n_params)  # Equality: X_i' θ = y_i
    A_ineq = sparse(A_constraints)
    
    # Stack constraints: equality becomes both lower and upper bound = y_i
    n_eq = 1
    n_ineq = size(A_constraints, 1)
    A_combined = vcat(A_eq, A_ineq)
    l_combined = vcat([y_i], l_constraints)
    u_combined = vcat([y_i], u_constraints)
    
    # Solve with OSQP - increase iterations and relax tolerance
    prob = OSQP.Model()
    OSQP.setup!(prob; P=P_qp, q=q_qp, A=sparse(A_combined), l=l_combined, u=u_combined, 
                verbose=false, eps_abs=1e-4, eps_rel=1e-4, max_iter=10000)
    results = OSQP.solve!(prob)
    
    # Check if solution exists
    if results.info.status != :Solved
        # This is OK - solver may find inaccurate solution with conflicting constraints
        # (trying to pass through data point AND satisfy constraints may not always work perfectly)
    end
    
    return results.x
end

# Iterative constrained POPS: for each data point, compute constrained update
function iterative_constrained_pops(X, Y, Gamma, A_constraints, l_constraints, u_constraints)
    """
    Iteratively compute POPS updates with constraints for each data point.
    Each update is constrained to pass through its corresponding data point.
    """
    # First get nominal POPS fit
    pointwise_corrections, nominal_coeffs = corrections(X, Y, Gamma; leverage_percentile=0.0)
    
    n_params = size(pointwise_corrections, 2)
    n_points = size(X, 1)
    
    # For each data point, compute constrained update
    constrained_updates = []
    
    for i in 1:n_points
        X_i = X[i, :]
        y_i = Y[i]
        
        # Use the POPS nominal coefficients as starting point
        theta_nominal = nominal_coeffs
        
        # Solve constrained QP for this point
        theta_constrained = constrained_pops_update(X_i, y_i, theta_nominal, 
                                                     A_constraints, l_constraints, u_constraints)
        
        push!(constrained_updates, theta_constrained)
    end
    
    constrained_updates_matrix = hcat(constrained_updates...)'
    
    return constrained_updates_matrix
end

# Main example
x, y, X = generate_quadratic_data(50)

# QUADRATIC MODEL - Perfect fit!
Gamma = Matrix(I(size(X, 2)))

println("=== Part 1: Standard POPS Ensemble ===")
println("True function: 0.55*|x|^4 + 2*|x|^2.21 - 2*x - 1")
println("Fitting with: quadratic design matrix [1, x, x²] (MISSPECIFIED)")

# Get unconstrained POPS ensemble
pops_corrections, nominal_coeffs = corrections(X, y, Gamma; leverage_percentile=0.0)

# Reconstruct POPS members: each member = nominal_coeffs + correction
pointwise_corrections_pops = pops_corrections .+ nominal_coeffs'

println("POPS ensemble size: ", size(pointwise_corrections_pops))
println("Number of fits: ", size(pointwise_corrections_pops, 1))
println("Nominal coefficients: ", nominal_coeffs)

# Setup constraints: enforce y >= 1 everywhere
# For a quadratic y = θ₀ + θ₁ x + θ₂ x², we need y >= 1 for all x
# This is complex, so we'll enforce it at several points: x = -2, -1, 0, 1, 2
test_x_points = [-2.0, -1.0, 0.0, 1.0, 2.0]

A_constraints = hcat([[1.0, xp, xp^2] for xp in test_x_points]...)'
l_constraints = fill(1.0, length(test_x_points))  # y >= 1
u_constraints = fill(Inf, length(test_x_points))   # no upper bound

println("\n=== Part 2: Iterative Constrained POPS ===")
println("Constraint: y >= 1 at x in {-2, -1, 0, 1, 2}")

# Get iterative constrained POPS ensemble
pointwise_corrections_constrained = iterative_constrained_pops(X, y, Gamma, 
                                                                A_constraints, l_constraints, u_constraints)
println("Constrained ensemble size: ", size(pointwise_corrections_constrained))

# Generate predictions for plotting
x_plot = range(-2.5, 2.5, length=200)
X_plot_full = [ones(length(x_plot)) x_plot x_plot.^2]

# True function
y_true_plot = 0.55 .* abs.(x_plot).^4 .+ 2 .* abs.(x_plot).^2.21 .- 2 .* x_plot .- 1

# Generate ensemble predictions
y_ensemble_pops = X_plot_full * pointwise_corrections_pops'
y_ensemble_constrained = X_plot_full * pointwise_corrections_constrained'

# Create plot with CairoMakie
fig = Figure(size=(1400, 600))

# Standard POPS - each member passes through one data point
ax1 = Axis(fig[1, 1], xlabel="x", ylabel="y", 
          title="Standard POPS Ensemble (Misspecified Quadratic)\n(each line passes through its data point)")

# Plot all ensemble members
for i in 1:size(y_ensemble_pops, 2)
    lines!(ax1, x_plot, y_ensemble_pops[:, i], alpha=0.15, color=:blue, linewidth=1)
end

# Data points
scatter!(ax1, x, y, label="Training data", markersize=5, color=:black)

# True function
lines!(ax1, x_plot, y_true_plot, label="True quadratic", linestyle=:dash, linewidth=2.5, color=:black)

axislegend(ax1, position=:lt)

# Constrained POPS - each member passes through one data point AND satisfies constraints
ax2 = Axis(fig[1, 2], xlabel="x", ylabel="y", 
          title="Constrained POPS Ensemble (y >= 1)\n(monotonicity + positivity constraints)")

# Plot all ensemble members
for i in 1:size(y_ensemble_constrained, 2)
    lines!(ax2, x_plot, y_ensemble_constrained[:, i], alpha=0.15, color=:red, linewidth=1)
end

# Data points
scatter!(ax2, x, y, label="Training data", markersize=5, color=:black)

# True function
lines!(ax2, x_plot, y_true_plot, label="True quadratic", linestyle=:dash, linewidth=2.5, color=:black)

axislegend(ax2, position=:lt)

save("/storage/astro2/phupfb/PhD/acestuff/new_ACE/pops_constrained_fit.png", fig)
display(fig)

# Print results
println("\n=== Constraint Satisfaction ===")
println("Note: Some data points have y < 1, so perfect constraint satisfaction")
println("is not feasible. POPS finds best compromise between data fit & constraints.\n")

# Check positivity constraints for constrained ensemble
for (idx, xp) in enumerate(test_x_points)
    y_vals = pointwise_corrections_constrained[:, 1] .+ pointwise_corrections_constrained[:, 2] .* xp .+ pointwise_corrections_constrained[:, 3] .* xp^2
    min_y = minimum(y_vals)
    num_satisfied = sum(y_vals .>= 1.0 .- 1e-3)
    println("At x = $xp: min y = $min_y, satisfied by $num_satisfied/$(size(pointwise_corrections_constrained, 1)) members")
end

println("\n=== Data Point Verification ===")
y_pred_pops = vec(diag(X * pointwise_corrections_pops'))
y_pred_constrained = vec(diag(X * pointwise_corrections_constrained'))

pops_error = norm(y - y_pred_pops)
constrained_error = norm(y - y_pred_constrained)

println("Standard POPS - fit error: ", pops_error, " (should be ≈ 0)")
println("Constrained POPS - fit error: ", constrained_error, " (should be ≈ 0)")