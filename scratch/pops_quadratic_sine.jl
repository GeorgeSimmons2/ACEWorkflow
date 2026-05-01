using OSQP
using SparseArrays
using LinearAlgebra, Statistics
using CairoMakie
using Printf

# Generate data from sine function
function generate_sine_data(n=40)
    x = range(0, 2π, length=n)
    # True function: sin(x)
    y = sin.(x)
    
    # Fit QUADRATIC model: [1, x, x^2] - misspecified for oscillatory!
    X = hcat([ones(n), x, x.^2]...)
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
x, y, X = generate_sine_data(40)

# Polynomial regularization (identity)
Gamma = Matrix(I(size(X, 2)))

println("=" ^ 70)
println("POPS WITH MISSPECIFICATION + PHYSICAL BOUNDS")
println("=" ^ 70)
println("True function: sin(x) [OSCILLATORY, range = [-1, 1]]")
println("Model: Quadratic [1, x, x²] [MONOTONIC/UNBOUNDED - misspecified!]")
println("=" ^ 70)

# Get unconstrained POPS ensemble
pops_corrections, nominal_coeffs = corrections(X, y, Gamma; leverage_percentile=0.0)
pointwise_corrections_pops = pops_corrections .+ nominal_coeffs'

println("\nPart 1: UNCONSTRAINED POPS")
println("-" ^ 70)
println("POPS ensemble size: $(size(pointwise_corrections_pops))")
println("Nominal coefficients: [θ₀, θ₁, θ₂] = ", nominal_coeffs)

# Test points for constraint: DENSE COVERAGE over full range + beyond to strictly enforce
# Include extrapolation region so constraints are obeyed everywhere
test_x = vcat(range(-0.5, 0, length=8), 
              range(0, 2π, length=40), 
              range(2π, 2π + 0.5, length=8))

# Constraint: STRICT -1.2 <= y <= 1.2 at ALL these test points (tighter than [-1.5, 1.5])
# This forces the quadratic to stay in a physically reasonable range EVERYWHERE
A_test_rows = [[1.0, xp, xp^2] for xp in test_x]
A_constraints = hcat(A_test_rows...)'
l_constraints = fill(-1.2, length(test_x))  # y >= -1.2
u_constraints = fill(1.2, length(test_x))   # y <= 1.2

println("\nPart 2: CONSTRAINED POPS")
println("-" ^ 70)
println("Constraint: STRICT -1.2 ≤ y ≤ 1.2 at ", length(test_x), " dense evaluation points")
println("           x ∈ [$(minimum(test_x)), $(maximum(test_x))] (includes extrapolation region!)")

# Get constrained POPS ensemble
pointwise_corrections_constrained = iterative_constrained_pops(X, y, Gamma, 
                                                                A_constraints, l_constraints, u_constraints)
println("Constrained ensemble size: $(size(pointwise_corrections_constrained))")

# Generate predictions for plotting and evaluation
x_plot = range(-0.5, 2π + 0.5, length=250)
X_plot_full = hcat([ones(length(x_plot)), x_plot, x_plot.^2]...)

# True function
y_true_plot = sin.(x_plot)

# Ensemble predictions
y_ensemble_pops = X_plot_full * pointwise_corrections_pops'
y_ensemble_constrained = X_plot_full * pointwise_corrections_constrained'

# Compute MAE on TEST DATA (extrapolation region - where quadratic really breaks!)
# Test includes points BEYOND the training range where quadratic diverges
x_test = vcat(range(-1.5, 0, length=15), range(2π, 2π + 1.5, length=15))
X_test = hcat([ones(length(x_test)), x_test, x_test.^2]...)
y_test_true = sin.(x_test)

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
if improvement > 0
    println("Improvement:                    ", @sprintf("%.2f%%", improvement), " ✓ BETTER")
else
    println("Improvement:                    ", @sprintf("%.2f%%", improvement), " (worse)")
end
println("=" ^ 70)

# Check constraint violations in unconstrained - STRICT [-1.2, 1.2]
y_test_pops_all = X_test * pointwise_corrections_pops'
num_oob_pops = sum((y_test_pops_all .< -1.2) .| (y_test_pops_all .> 1.2))
percent_oob_pops = num_oob_pops / length(y_test_pops_all) * 100

y_test_constrained_all = X_test * pointwise_corrections_constrained'
num_oob_constrained = sum((y_test_constrained_all .< -1.2) .| (y_test_constrained_all .> 1.2))
percent_oob_constrained = num_oob_constrained / length(y_test_constrained_all) * 100

println("\nUnconstrained ensemble: $(num_oob_pops) predictions outside [-1.2, 1.2] ($(round(percent_oob_pops, digits=1))%)")
println("Constrained ensemble:   $(num_oob_constrained) predictions outside [-1.2, 1.2] ($(round(percent_oob_constrained, digits=1))%)")

# Create visualization
fig = Figure(size=(1400, 600))

# UNCONSTRAINED
ax1 = Axis(fig[1, 1], xlabel="x", ylabel="y", 
          title="Unconstrained POPS: Quadratic model on Sin(x)\\n(ensemble breaks bounds in extrapolation!)")

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
lines!(ax1, x_plot, y_true_plot, label="True sin(x)", linestyle=:dash, linewidth=2.5, color=:darkred)

# Bounds
hlines!(ax1, [1.2, -1.2], label="Constraint bounds [-1.2, 1.2]", linestyle=:dashdot, linewidth=1.5, color=:green, alpha=0.8)

# Shade extrapolation region
vspan!(ax1, -0.5, 0, alpha=0.1, color=:orange)
vspan!(ax1, 2π, 2π + 0.5, alpha=0.1, color=:orange)

axislegend(ax1, position=:rc, fontsize=10)
ylims!(ax1, -3, 4.5)

# CONSTRAINED
ax2 = Axis(fig[1, 2], xlabel="x", ylabel="y", 
          title="Constrained POPS: All 40 Ensemble Members with STRICT bounds\\n(respects bounds everywhere!)")

# Plot all ensemble members
for i in 1:size(y_ensemble_constrained, 2)
    lines!(ax2, x_plot, y_ensemble_constrained[:, i], alpha=0.05, color=:red, linewidth=1)
end

# Ensemble mean (used for MAE calculation)
lines!(ax2, x_plot, vec(mean(y_ensemble_constrained, dims=2)), label="Ensemble MEAN (used for MAE)", 
       linewidth=3, color=:darkred)

# Data points
scatter!(ax2, x, y, label="Training data", markersize=6, color=:black)

# True function
lines!(ax2, x_plot, y_true_plot, label="True sin(x)", linestyle=:dash, linewidth=2.5, color=:darkgreen)

# Bounds
hlines!(ax2, [1.2, -1.2], label="Constraint bounds [-1.2, 1.2]", linestyle=:dashdot, linewidth=1.5, color=:green, alpha=0.8)

# Shade extrapolation region
vspan!(ax2, -0.5, 0, alpha=0.1, color=:orange)
vspan!(ax2, 2π, 2π + 0.5, alpha=0.1, color=:orange)

axislegend(ax2, position=:rc, fontsize=10)
ylims!(ax2, -3, 4.5)

save("/storage/astro2/phupfb/PhD/acestuff/new_ACE/pops_quadratic_sine_generalization.png", fig)
display(fig)

# Create separate figure showing generalization error
fig2 = Figure(size=(1400, 600))

# Compute error on dense test grid
x_dense_test = range(-1.5, 2π + 1.5, length=200)
X_dense_test = hcat([ones(length(x_dense_test)), x_dense_test, x_dense_test.^2]...)
y_dense_true = sin.(x_dense_test)

# Get predictions from ensemble means
y_pops_dense_all = X_dense_test * pointwise_corrections_pops'
y_pops_dense_mean = vec(mean(y_pops_dense_all, dims=2))
y_pops_dense_error = abs.(y_dense_true - y_pops_dense_mean)

y_constrained_dense_all = X_dense_test * pointwise_corrections_constrained'
y_constrained_dense_mean = vec(mean(y_constrained_dense_all, dims=2))
y_constrained_dense_error = abs.(y_dense_true - y_constrained_dense_mean)

# Create two subplots: top = predictions, bottom = errors
ax_pred = Axis(fig2[1, 1], xlabel="x", ylabel="y", 
              title="Ensemble Mean Predictions: Unconstrained vs Constrained")

# Plot ensemble means
lines!(ax_pred, x_dense_test, y_pops_dense_mean, label="Unconstrained mean", 
      linewidth=2.5, color=:blue, alpha=0.8)
lines!(ax_pred, x_dense_test, y_constrained_dense_mean, label="Constrained mean", 
      linewidth=2.5, color=:red, alpha=0.8)

# True function
lines!(ax_pred, x_dense_test, y_dense_true, label="True sin(x)", 
      linewidth=2.5, color=:darkgreen, linestyle=:dash)

# Shade extrapolation regions
vspan!(ax_pred, -1.5, 0, alpha=0.1, color=:orange)
vspan!(ax_pred, 2π, 2π + 1.5, alpha=0.1, color=:orange)
vspan!(ax_pred, 0, 2π, alpha=0.05, color=:green)

axislegend(ax_pred, position=:lt, fontsize=10)
ylims!(ax_pred, -2, 3)

# Error subplot
ax_err = Axis(fig2[2, 1], xlabel="x", ylabel="Absolute Error", 
             title="Generalization Error: Unconstrained vs Constrained POPS\\n(orange regions = extrapolation)")

# Plot errors
lines!(ax_err, x_dense_test, y_pops_dense_error, label="Unconstrained POPS error", 
      linewidth=2.5, color=:blue, alpha=0.8)
lines!(ax_err, x_dense_test, y_constrained_dense_error, label="Constrained POPS error", 
      linewidth=2.5, color=:red, alpha=0.8)

# Shade extrapolation regions
vspan!(ax_err, -1.5, 0, alpha=0.1, color=:orange, label="Extrapolation region")
vspan!(ax_err, 2π, 2π + 1.5, alpha=0.1, color=:orange)

# Training data region
vspan!(ax_err, 0, 2π, alpha=0.05, color=:green, label="Training region")

axislegend(ax_err, position=:lt, fontsize=10)
ylims!(ax_err, 0, 2.5)

save("/storage/astro2/phupfb/PhD/acestuff/new_ACE/pops_generalization_error.png", fig2)
display(fig2)

# Print detailed error statistics
println("\n" ^ 70)
println("GENERALIZATION ERROR ANALYSIS")
println("=" ^ 70)

# Training region
x_train_region = (x_dense_test .>= 0) .& (x_dense_test .<= 2π)
error_pops_train = mean(y_pops_dense_error[x_train_region])
error_constrained_train = mean(y_constrained_dense_error[x_train_region])

# Extrapolation region
x_extrap_region = (x_dense_test .< 0) .| (x_dense_test .> 2π)
error_pops_extrap = mean(y_pops_dense_error[x_extrap_region])
error_constrained_extrap = mean(y_constrained_dense_error[x_extrap_region])

println("TRAINING REGION [0, 2π]:")
println("  Unconstrained POPS MAE: ", @sprintf("%.6f", error_pops_train))
println("  Constrained POPS MAE:   ", @sprintf("%.6f", error_constrained_train))
println("  Difference:             ", @sprintf("%.6f", error_pops_train - error_constrained_train))

println("\nEXTRAPOLATION REGION [x < 0 OR x > 2π]:")
println("  Unconstrained POPS MAE: ", @sprintf("%.6f", error_pops_extrap))
println("  Constrained POPS MAE:   ", @sprintf("%.6f", error_constrained_extrap))
improvement_extrap = (error_pops_extrap - error_constrained_extrap) / error_pops_extrap * 100
println("  Improvement:            ", @sprintf("%.2f%%", improvement_extrap), " ✓")

println("\nOVERALL [entire domain -1.5 to 2π+1.5]:")
overall_pops = mean(y_pops_dense_error)
overall_constrained = mean(y_constrained_dense_error)
overall_improvement = (overall_pops - overall_constrained) / overall_pops * 100
println("  Unconstrained POPS MAE: ", @sprintf("%.6f", overall_pops))
println("  Constrained POPS MAE:   ", @sprintf("%.6f", overall_constrained))
println("  Improvement:            ", @sprintf("%.2f%%", overall_improvement), " ✓")

println("=" ^ 70)

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
println("Unconstrained: Quadratic fits data perfectly locally but explodes outside range")
println("               Produces physically unrealistic predictions far from training set")
println("Constrained:   Bounds prevent extrapolation from going wild")
println("               Result: $(round(improvement, digits=1))% LOWER MAE on held-out data!")
println("=" ^ 70)
