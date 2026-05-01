using OSQP
using SparseArrays
using LinearAlgebra, Statistics
using CairoMakie

# Generate data from a sinusoidal function (highly oscillatory - BAD for polynomials!)
function generate_sinusoid_data(n=50)
    x = range(0, 4*π, length=n)
    # True function: sin(x) + 0.3*cos(2*x) (oscillatory, hard to fit with polynomials)
    y = sin.(x) .+ 0.3 .* cos.(2 .* x)
    
    # Fit a polynomial of degree 5: [1, x, x^2, x^3, x^4, x^5]
    # This is MISSPECIFIED for sinusoidal data!
    X = hcat([ones(n), x, x.^2, x.^3, x.^4, x.^5]...)
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
    
    return results.x
end

# Iterative constrained POPS: for each data point, compute constrained update
function iterative_constrained_pops(X, Y, Gamma, A_constraints, l_constraints, u_constraints)
    """
    Iteratively compute POPS updates with constraints for each data point.
    Each update is constrained to pass through its corresponding data point.
    """
    # First get nominal POPS fit
    pops_corrections, nominal_coeffs = corrections(X, Y, Gamma; leverage_percentile=0.0)
    
    n_params = size(X, 2)
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
x, y, X = generate_sinusoid_data(50)

# Polynomial regularization
Gamma = Matrix(I(size(X, 2)))

println("=== Part 1: Standard POPS Ensemble ===")
println("True function: sin(x) + 0.3*cos(2*x)")
println("Fitting with: polynomial degree 5 (MISSPECIFIED - polynomials can't represent oscillations well)")

# Get unconstrained POPS ensemble
pops_corrections, nominal_coeffs = corrections(X, y, Gamma; leverage_percentile=0.0)

# Reconstruct POPS members: each member = nominal_coeffs + correction
pointwise_corrections_pops = pops_corrections .+ nominal_coeffs'

println("POPS ensemble size: ", size(pointwise_corrections_pops))
println("Number of fits: ", size(pointwise_corrections_pops, 1))

# Setup constraints: enforce smoothness and limit extreme oscillations
# Constraint 1: dy/dx <= 2 at x = 2π (slope should be reasonable)
# Constraint 2: dy/dx >= -2 at x = π (slope should be reasonable)
# Constraint 3: polynomial value should stay within [-2, 2] at evaluation points
# (since sin(x) + 0.3*cos(2x) is roughly in [-1.3, 1.3])

eval_x_points = [π, 2π, 3π]  # Evaluation points for value constraints

# Build constraint matrix for value bounds: -1.5 <= p(x) <= 1.5
A_value_rows = []
for xp in eval_x_points
    # [1, x, x^2, x^3, x^4, x^5]
    push!(A_value_rows, [1.0, xp, xp^2, xp^3, xp^4, xp^5])
end

n_value_constraints = length(eval_x_points)
A_value = hcat(A_value_rows...)' 

# Constraint: dy/dx <= 2 at x = π
# dy/dx = θ₁ + 2*θ₂*x + 3*θ₃*x² + 4*θ₄*x³ + 5*θ₅*x⁴
xp_slope_pos = π
A_slope_pos = reshape([0.0, 1.0, 2*xp_slope_pos, 3*xp_slope_pos^2, 4*xp_slope_pos^3, 5*xp_slope_pos^4], 1, 6)

xp_slope_neg = 2*π
A_slope_neg = reshape([0.0, 1.0, 2*xp_slope_neg, 3*xp_slope_neg^2, 4*xp_slope_neg^3, 5*xp_slope_neg^4], 1, 6)

# Combine all constraints
A_constraints = vcat(A_value, A_slope_pos, A_slope_neg)
l_constraints = vcat(fill(-1.5, n_value_constraints), [-Inf], [-2.0])  # value bounds + slope bounds
u_constraints = vcat(fill(1.5, n_value_constraints), [2.0], [Inf])

println("\n=== Part 2: Iterative Constrained POPS ===")
println("Constraints:")
println("  Value: -1.5 <= p(x) <= 1.5 at x ∈ {π, 2π, 3π}")
println("  Slope: dy/dx <= 2 at x = π")
println("  Slope: dy/dx >= -2 at x = 2π")

# Get iterative constrained POPS ensemble
pointwise_corrections_constrained = iterative_constrained_pops(X, y, Gamma, 
                                                                A_constraints, l_constraints, u_constraints)
println("Constrained ensemble size: ", size(pointwise_corrections_constrained))

# Generate predictions for plotting
x_plot = range(0, 4*π, length=300)
X_plot_full = hcat([ones(length(x_plot)), x_plot, x_plot.^2, x_plot.^3, x_plot.^4, x_plot.^5]...)

# True function
y_true_plot = sin.(x_plot) .+ 0.3 .* cos.(2 .* x_plot)

# Generate ensemble predictions
y_ensemble_pops = X_plot_full * pointwise_corrections_pops'
y_ensemble_constrained = X_plot_full * pointwise_corrections_constrained'

# Create plot with CairoMakie
fig = Figure(size=(1400, 600))

# Standard POPS - each member passes through one data point
ax1 = Axis(fig[1, 1], xlabel="x", ylabel="y", 
          title="Standard POPS Ensemble (Misspecified Polynomial)\\n(each line passes through its data point)")

# Plot all ensemble members
for i in 1:size(y_ensemble_pops, 2)
    lines!(ax1, x_plot, y_ensemble_pops[:, i], alpha=0.1, color=:blue, linewidth=1)
end

# Data points
scatter!(ax1, x, y, label="Training data", markersize=5, color=:black)

# True function
lines!(ax1, x_plot, y_true_plot, label="True sin + 0.3*cos(2x)", linestyle=:dash, linewidth=2.5, color=:darkred)

# Add constraint bounds
hlines!(ax1, [1.5, -1.5], label="Value constraint bounds", linestyle=:dot, linewidth=1.5, color=:green, alpha=0.5)

axislegend(ax1, position=:lt)

# Constrained POPS - each member passes through one data point AND satisfies constraints
ax2 = Axis(fig[1, 2], xlabel="x", ylabel="y", 
          title="Constrained POPS Ensemble\\n(smoothness & oscillation constraints)")

# Plot all ensemble members
for i in 1:size(y_ensemble_constrained, 2)
    lines!(ax2, x_plot, y_ensemble_constrained[:, i], alpha=0.1, color=:red, linewidth=1)
end

# Data points
scatter!(ax2, x, y, label="Training data", markersize=5, color=:black)

# True function
lines!(ax2, x_plot, y_true_plot, label="True sin + 0.3*cos(2x)", linestyle=:dash, linewidth=2.5, color=:darkred)

# Add constraint bounds
hlines!(ax2, [1.5, -1.5], label="Value constraint bounds", linestyle=:dot, linewidth=1.5, color=:green, alpha=0.5)

axislegend(ax2, position=:lt)

save("/storage/astro2/phupfb/PhD/acestuff/new_ACE/pops_polynomial_sinusoid.png", fig)
display(fig)

# Print results
println("\n=== Constraint Satisfaction ===")
println("Value constraints (should be in [-1.5, 1.5]):\n")

for (idx, xp) in enumerate(eval_x_points)
    idx_closest = findmin(abs.(x_plot .- xp))[2]
    x_row = vec(X_plot_full[idx_closest, :])  # Ensure it's a 1D vector
    y_vals = x_row' * pointwise_corrections_constrained'  # Should give 1x50
    y_vals = vec(y_vals)  # Convert to 1D vector
    min_y = minimum(y_vals)
    max_y = maximum(y_vals)
    num_satisfied = sum((y_vals .>= -1.5) .& (y_vals .<= 1.5))
    println("  At x = $xp: y ∈ [$min_y, $max_y], satisfied by $num_satisfied/$(size(pointwise_corrections_constrained, 1)) members")
end

println("\nSlope constraints:\n")
# dy/dx at π
deriv_π = pointwise_corrections_constrained[:, 2] .+ 2 * π .* pointwise_corrections_constrained[:, 3] .+ 
          3 * π^2 .* pointwise_corrections_constrained[:, 4] .+ 4 * π^3 .* pointwise_corrections_constrained[:, 5] .+ 
          5 * π^4 .* pointwise_corrections_constrained[:, 6]
num_satisfied_π = sum(deriv_π .<= 2.0 .+ 1e-3)
println("  At x = π: dy/dx ∈ [$(minimum(deriv_π)), $(maximum(deriv_π))], dy/dx <= 2 satisfied by $num_satisfied_π/$(size(pointwise_corrections_constrained, 1)) members")

# dy/dx at 2π
deriv_2π = pointwise_corrections_constrained[:, 2] .+ 2 * 2π .* pointwise_corrections_constrained[:, 3] .+ 
           3 * (2π)^2 .* pointwise_corrections_constrained[:, 4] .+ 4 * (2π)^3 .* pointwise_corrections_constrained[:, 5] .+ 
           5 * (2π)^4 .* pointwise_corrections_constrained[:, 6]
num_satisfied_2π = sum(deriv_2π .>= -2.0 .- 1e-3)
println("  At x = 2π: dy/dx ∈ [$(minimum(deriv_2π)), $(maximum(deriv_2π))], dy/dx >= -2 satisfied by $num_satisfied_2π/$(size(pointwise_corrections_constrained, 1)) members")

println("\n=== Data Point Verification ===")
y_pred_pops = vec(diag(X * pointwise_corrections_pops'))
y_pred_constrained = vec(diag(X * pointwise_corrections_constrained'))

pops_error = norm(y - y_pred_pops)
constrained_error = norm(y - y_pred_constrained)

println("Standard POPS - fit error: ", pops_error, " (should be ≈ 0)")
println("Constrained POPS - fit error: ", constrained_error, " (should be ≈ 0)")
