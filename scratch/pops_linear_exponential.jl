using OSQP
using SparseArrays
using LinearAlgebra, Statistics
using CairoMakie
using Printf

# Generate data from exponential function
function generate_exponential_data(n=30)
    x = range(-2, 2, length=n)
    # True function: e^(0.5*x) - exponential, always positive, never goes negative
    y = exp.(0.5 .* x)
    
    # Fit a LINEAR model: [1, x] - CATASTROPHICALLY misspecified for exponential!
    X = hcat([ones(n), x]...)
    return x, y, X
end

# Standard POPS corrections
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

# Constrained POPS for a single data point
function constrained_pops_update(X_i, y_i, theta_nominal, A_constraints, l_constraints, u_constraints)
    n_params = length(theta_nominal)
    
    # QP: minimize ||θ - θ_nominal||² 
    # subject to: X_i' θ = y_i (equality)
    #            A_constraints θ satisfies bounds
    P_qp = sparse(2.0 * I(n_params))
    q_qp = -2.0 * theta_nominal
    
    A_eq = reshape(X_i, 1, n_params)
    A_ineq = sparse(A_constraints)
    
    n_eq = 1
    n_ineq = size(A_constraints, 1)
    A_combined = vcat(A_eq, A_ineq)
    l_combined = vcat([y_i], l_constraints)
    u_combined = vcat([y_i], u_constraints)
    
    prob = OSQP.Model()
    OSQP.setup!(prob; P=P_qp, q=q_qp, A=sparse(A_combined), l=l_combined, u=u_combined, 
                verbose=false, eps_abs=1e-4, eps_rel=1e-4, max_iter=10000)
    results = OSQP.solve!(prob)
    
    return results.x
end

# Iterative constrained POPS
function iterative_constrained_pops(X, Y, Gamma, A_constraints, l_constraints, u_constraints)
    pops_corrections, nominal_coeffs = corrections(X, Y, Gamma; leverage_percentile=0.0)
    
    n_params = size(X, 2)
    n_points = size(X, 1)
    
    constrained_updates = []
    
    for i in 1:n_points
        X_i = X[i, :]
        y_i = Y[i]
        theta_nominal = nominal_coeffs
        
        theta_constrained = constrained_pops_update(X_i, y_i, theta_nominal, 
                                                     A_constraints, l_constraints, u_constraints)
        
        push!(constrained_updates, theta_constrained)
    end
    
    constrained_updates_matrix = hcat(constrained_updates...)'
    
    return constrained_updates_matrix
end

# Main example
x, y, X = generate_exponential_data(30)

# Polynomial regularization (identity)
Gamma = Matrix(I(size(X, 2)))

println("=" ^ 70)
println("POPS WITH CATASTROPHIC MODEL MISSPECIFICATION")
println("=" ^ 70)
println("True function: e^(0.5*x) [EXPONENTIAL - always positive]")
println("Model: Linear [1, x] [LINEAR - can go negative]")
println("=" ^ 70)

# Get unconstrained POPS ensemble
pops_corrections, nominal_coeffs = corrections(X, y, Gamma; leverage_percentile=0.0)
pointwise_corrections_pops = pops_corrections .+ nominal_coeffs'

println("\nPart 1: UNCONSTRAINED POPS")
println("-" ^ 70)
println("POPS ensemble size: $(size(pointwise_corrections_pops))")
println("Nominal coefficients: [θ₀, θ₁] = ", nominal_coeffs)

# Test points for constraint: evaluate over full range
test_x = collect(range(-2, 2, length=15))

# Constraint: y >= 0.1 at all test points (since e^(-1) ≈ 0.368, this is reasonable)
A_test_rows = [[1.0, xp] for xp in test_x]
A_constraints = hcat(A_test_rows...)'
l_constraints = fill(0.1, length(test_x))  # y >= 0.1
u_constraints = fill(Inf, length(test_x))

println("\nPart 2: CONSTRAINED POPS")
println("-" ^ 70)
println("Constraint: y >= 0.1 at ", length(test_x), " evaluation points")
println("           x ∈ [$(minimum(test_x)), $(maximum(test_x))]")

# Get constrained POPS ensemble
pointwise_corrections_constrained = iterative_constrained_pops(X, y, Gamma, 
                                                                A_constraints, l_constraints, u_constraints)
println("Constrained ensemble size: $(size(pointwise_corrections_constrained))")

# Generate predictions for plotting and evaluation
x_plot = range(-2.5, 2.5, length=200)
X_plot_full = hcat([ones(length(x_plot)), x_plot]...)

# True function
y_true_plot = exp.(0.5 .* x_plot)

# Ensemble predictions
y_ensemble_pops = X_plot_full * pointwise_corrections_pops'
y_ensemble_constrained = X_plot_full * pointwise_corrections_constrained'

# Compute MAE on TEST DATA (not training)
x_test = range(-2.3, 2.3, length=50)
X_test = hcat([ones(length(x_test)), x_test]...)
y_test_true = exp.(0.5 .* x_test)

# For MAE, use ensemble mean
y_pops_mean = vec(mean(X_test * pointwise_corrections_pops', dims=2))
y_constrained_mean = vec(mean(X_test * pointwise_corrections_constrained', dims=2))

mae_pops = mean(abs.(y_test_true - y_pops_mean))
mae_constrained = mean(abs.(y_test_true - y_constrained_mean))
improvement = (mae_pops - mae_constrained) / mae_pops * 100

println("\n" ^ 70)
println("RESULTS")
println("=" ^ 70)
println("Test MAE (Unconstrained POPS): ", @sprintf("%.6f", mae_pops))
println("Test MAE (Constrained POPS):   ", @sprintf("%.6f", mae_constrained))
println("Improvement:                    ", @sprintf("%.2f%%", improvement))
println("=" ^ 70)

# Check how many members go negative in unconstrained
y_test_pops_all = X_test * pointwise_corrections_pops'
num_negative_pops = sum(y_test_pops_all .< 0)
percent_negative_pops = num_negative_pops / length(y_test_pops_all) * 100

y_test_constrained_all = X_test * pointwise_corrections_constrained'
num_negative_constrained = sum(y_test_constrained_all .< 0)
percent_negative_constrained = num_negative_constrained / length(y_test_constrained_all) * 100

println("\nUnconstrained ensemble: $(num_negative_pops) predictions < 0 ($(round(percent_negative_pops, digits=1))%)")
println("Constrained ensemble:   $(num_negative_constrained) predictions < 0 ($(round(percent_negative_constrained, digits=1))%)")

# Create visualization
fig = Figure(size=(1400, 600))

# UNCONSTRAINED
ax1 = Axis(fig[1, 1], xlabel="x", ylabel="y", 
          title="Unconstrained POPS: Linear model on Exponential data\\n(ensemble goes NEGATIVE!)")

# Plot all ensemble members
for i in 1:size(y_ensemble_pops, 2)
    lines!(ax1, x_plot, y_ensemble_pops[:, i], alpha=0.05, color=:blue, linewidth=1)
end

# Ensemble mean
lines!(ax1, x_plot, vec(mean(y_ensemble_pops, dims=2)), label="Ensemble mean", 
       linewidth=2.5, color=:darkblue)

# Data points
scatter!(ax1, x, y, label="Training data", markersize=6, color=:black)

# True function
lines!(ax1, x_plot, y_true_plot, label="True e^(0.5x)", linestyle=:dash, linewidth=2.5, color=:darkred)

# Zero line
hlines!(ax1, [0], label="y=0 (reality)", linestyle=:dot, linewidth=1.5, color=:gray, alpha=0.7)

# Constraint bound
hlines!(ax1, [0.1], label="Constraint: y ≥ 0.1", linestyle=:dashdot, linewidth=1.5, color=:green, alpha=0.7)

axislegend(ax1, position=:lt, fontsize=9)
ylims!(ax1, -0.5, 5)

# CONSTRAINED
ax2 = Axis(fig[1, 2], xlabel="x", ylabel="y", 
          title="Constrained POPS: Linear model with positivity constraint\\n(ensemble stays POSITIVE!)")

# Plot all ensemble members
for i in 1:size(y_ensemble_constrained, 2)
    lines!(ax2, x_plot, y_ensemble_constrained[:, i], alpha=0.05, color=:red, linewidth=1)
end

# Ensemble mean
lines!(ax2, x_plot, vec(mean(y_ensemble_constrained, dims=2)), label="Ensemble mean", 
       linewidth=2.5, color=:darkred)

# Data points
scatter!(ax2, x, y, label="Training data", markersize=6, color=:black)

# True function
lines!(ax2, x_plot, y_true_plot, label="True e^(0.5x)", linestyle=:dash, linewidth=2.5, color=:darkgreen)

# Zero line
hlines!(ax2, [0], label="y=0", linestyle=:dot, linewidth=1.5, color=:gray, alpha=0.7)

# Constraint bound
hlines!(ax2, [0.1], label="Constraint: y ≥ 0.1", linestyle=:dashdot, linewidth=1.5, color=:green, alpha=0.7)

axislegend(ax2, position=:lt, fontsize=9)
ylims!(ax2, -0.5, 5)

save("/storage/astro2/phupfb/PhD/acestuff/new_ACE/pops_linear_exponential.png", fig)
display(fig)

# Data point verification
y_pred_pops = vec(diag(X * pointwise_corrections_pops'))
y_pred_constrained = vec(diag(X * pointwise_corrections_constrained'))

pops_error = norm(y - y_pred_pops)
constrained_error = norm(y - y_pred_constrained)

println("\n" ^ 70)
println("TRAINING DATA VERIFICATION")
println("=" ^ 70)
println("Standard POPS - fit error: $pops_error (should be ≈ 0)")
println("Constrained POPS - fit error: $constrained_error (should be ≈ 0)")

# Summary
println("\n" ^ 70)
println("KEY INSIGHT")
println("=" ^ 70)
println("Unconstrained: Linear model produces negative predictions (physically wrong!)")
println("               Tries to average through exponential data → bad fit where it matters")
println("Constrained:   Enforces positivity → forces sensible linear approximations")
println("               Result: $(round(improvement, digits=1))% LOWER MAE on test data!")
println("=" ^ 70)
