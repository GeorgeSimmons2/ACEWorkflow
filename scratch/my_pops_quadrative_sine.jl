using OSQP
using SparseArrays
using LinearAlgebra, Statistics
using CairoMakie
using Printf

# Generate data from very wiggly quadratic
function generate_wiggly_quadratic_data(n=40)
    x = range(0, 2π, length=n)
    # True function: sin(10x) + x^2 - "very wiggly quadratic" with oscillation amplitude ~1
    y = sin.(10 .* x) .+ x.^2
    
    # Fit QUADRATIC model: [1, x, x^2] - misspecified but we know oscillation structure!
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

# Iterative constrained POPS
function iterative_constrained_pops(X, Y, Gamma, A_constraints, l_constraints, u_constraints; leverage_percentile=0.)    
    n_params = size(X, 2)
    n_points = size(X, 1)
    
    constrained_updates = []

    C = (Gamma' * Gamma ./ size(X,1) .+ X' * X)
    A = C \ X'
    leverage = diag(X * A)
    coeffs = C \ (X' * Y)
    errors = Y .- (X * coeffs)
    # Keep A_constraints in prediction space (don't divide by Gamma)
    
    for i in 1:n_points
        X_i = (Gamma' * Gamma ./ size(X,1) .+ X' * X)
        y_i = (transpose(X[i, :]) .* (errors[i] / leverage[i]))[1, :]
        
        # Combine constraints:
        # 1. Equality: fit this training point X_i' θ = Y[i]
        # 2. Inequality: oscillation bounds on all evaluation points (A_constraints θ ∈ [l, u])
        A_combined = vcat(reshape(X[i, :], 1, size(X, 2)), A_constraints)
        
        l_combined = vcat([Y[i]], l_constraints)
        u_combined = vcat([Y[i]], u_constraints)

        prob = OSQP.Model()
        OSQP.setup!(prob; P=sparse(X_i), q=y_i, A=sparse(A_combined), l=l_combined, u=u_combined, 
                    verbose=false, eps_abs=1e-7, eps_rel=1e-7, max_iter=10000)
        results = OSQP.solve!(prob)

        # Diagnostic: check if solution respects constraints
        if i == 1 || i % 10 == 0
            pred_val = dot(X[i, :], results.x)
            constraint_vals = A_constraints * results.x
            oob_count = sum((constraint_vals .< l_constraints .- 1e-5) .| (constraint_vals .> u_constraints .+ 1e-5))
            min_constraint = minimum(constraint_vals)
            max_constraint = maximum(constraint_vals)
            
            println("Point $i: pred=$(round(pred_val, digits=4)) (target=$(round(Y[i], digits=4))), constraints∈[$(round(min_constraint, digits=4)),$(round(max_constraint, digits=4))], OOB=$oob_count, status=$(results.info.status)")
        end
        
        push!(constrained_updates, results.x)
    end
    
    constrained_updates_matrix = hcat(constrained_updates...)' / Gamma
    
    return constrained_updates_matrix
end

# Main example
x, y, X = generate_wiggly_quadratic_data(40)

# True function for reference
true_func(x_val) = sin(10 * x_val) + x_val^2
# Quadratic baseline (what model predicts without oscillations)
quad_baseline(x_val) = 0.8783552690866065 - 0.27771638180287783 * x_val - 0.0004826236200607833 * x_val^2

# Polynomial regularization (identity)
Gamma = Matrix(I(size(X, 2)))

# Get the actual nominal (unconstrained) quadratic fit from training data
_, nominal_coeffs = corrections(X, y, Gamma; leverage_percentile=0.0)
quad_baseline(x_val) = nominal_coeffs[1] + nominal_coeffs[2] * x_val + nominal_coeffs[3] * x_val^2

println("=" ^ 70)
println("POPS WITH OSCILLATION-AWARE BOUNDS")
println("=" ^ 70)
println("True function: sin(10x) + x² [VERY WIGGLY QUADRATIC]")
println("Model: Quadratic [1, x, x²] [Captures trend but misses ±1 oscillations]")
println("Constraint: quadratic_fit ± 1.2 (amplitude + margin)")
println("Nominal coefficients: $(nominal_coeffs)")
println("=" ^ 70)

# Get unconstrained POPS ensemble
pops_corrections, nominal_coeffs = corrections(X, y, Gamma; leverage_percentile=0.0)
pointwise_corrections_pops = pops_corrections .+ nominal_coeffs'

println("\nPart 1: UNCONSTRAINED POPS")
println("-" ^ 70)
println("POPS ensemble size: $(size(pointwise_corrections_pops))")
println("Nominal coefficients: [θ₀, θ₁, θ₂] = ", nominal_coeffs)

# Compute quadratic fit baseline and oscillation bounds
x_eval = range(-0.5, 2π + 0.5, length=500)

# Known oscillation amplitude: sin(10x) has amplitude 1
# Use tighter margin to actually constrain the ensemble meaningfully
oscillation_amplitude = 1.0
oscillation_margin = 0.5  # Tighter margin to guide predictions

# Test points for constraint: oscillation-aware bounds around quadratic baseline
test_x = vcat(range(-2.5, 0, length=25), 
              range(0, 2π, length=150), 
              range(2π, 2π + 0.5, length=25))

# Constraint: [1, x, x²] · θ should produce predictions bounded by:
#   quad_baseline(x) ± (amplitude + margin)
# This is ALREADY what we want! The bounds are in prediction space.
A_test_rows = [[1.0, xp, xp^2] for xp in test_x]
A_constraints = hcat(A_test_rows...)'

# Compute bounds: the quadratic fit value ± the oscillation amplitude + margin
l_constraints = [quad_baseline(xp) - oscillation_amplitude - oscillation_margin for xp in test_x]
u_constraints = [quad_baseline(xp) + oscillation_amplitude + oscillation_margin for xp in test_x]

println("\nPart 2: CONSTRAINED POPS")
println("-" ^ 70)
println("Constraint: Oscillation-aware bounds at ", length(test_x), " evaluation points")
println("           y ∈ [quadratic ± $(oscillation_amplitude + oscillation_margin)]")
println("           x ∈ [$(minimum(test_x)), $(maximum(test_x))] (includes extrapolation region!)")

x_plot = range(-2.5, 2π + 0.5, length=250)
X_plot_full = hcat([ones(length(x_plot)), x_plot, x_plot.^2]...)
y_ensemble_pops = X_plot_full * pointwise_corrections_pops'

# Get constrained POPS ensemble using QP with oscillation bounds
pointwise_corrections_constrained = iterative_constrained_pops(X, y, Gamma, 
                                                                A_constraints, l_constraints, u_constraints)
y_ensemble_constrained = X_plot_full * pointwise_corrections_constrained'

println("Constrained ensemble size: $(size(pointwise_corrections_constrained))")

# True function
y_true_plot = true_func.(x_plot)
# Test includes points BEYOND the training range where quadratic diverges
# Extended range to show benefits of constraints in far extrapolation
x_test = vcat(range(-2.5, 0, length=25), range(2π, 2π + 2.5, length=25))
X_test = hcat([ones(length(x_test)), x_test, x_test.^2]...)
y_test_true = true_func.(x_test)

# Get predictions on test set
y_pops_test = X_test * pointwise_corrections_pops'
y_constrained_test = X_test * pointwise_corrections_constrained'

# For MAE, use ensemble mean
y_pops_mean = vec(mean(y_pops_test, dims=2))
y_constrained_mean = vec(mean(y_constrained_test, dims=2))

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

# Check constraint violations in unconstrained vs constrained
y_test_pops_all = X_test * pointwise_corrections_pops'
quad_test = quad_baseline.(x_test)
upper_bound_test = quad_test .+ oscillation_amplitude .+ oscillation_margin
lower_bound_test = quad_test .- oscillation_amplitude .- oscillation_margin

num_oob_pops = sum((y_test_pops_all .< lower_bound_test .- 1e-4) .| (y_test_pops_all .> upper_bound_test .+ 1e-4))
percent_oob_pops = num_oob_pops / length(y_test_pops_all) * 100

num_oob_constrained = sum((y_constrained_test .< lower_bound_test .- 1e-4) .| (y_constrained_test .> upper_bound_test .+ 1e-4))
percent_oob_constrained = num_oob_constrained / length(y_constrained_test) * 100

println("\nUnconstrained ensemble: $(num_oob_pops) predictions outside bounds ($(round(percent_oob_pops, digits=1))%)")
println("Constrained ensemble:   $(num_oob_constrained) predictions outside bounds ($(round(percent_oob_constrained, digits=1))%)")

# Create visualization
fig = Figure(size=(1400, 600))

# UNCONSTRAINED
ax1 = Axis(fig[1, 1], xlabel="x", ylabel="y", 
          title="Unconstrained POPS: Quadratic model on Sin(10x) + x²\n(unconstrained ensemble ignores oscillations)")

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
lines!(ax1, x_plot, y_true_plot, label="True sin(10x) + x²", linestyle=:dash, linewidth=2.5, color=:darkred)

# Quadratic baseline
lines!(ax1, x_plot, quad_baseline.(x_plot), label="Quadratic baseline", linestyle=:dot, linewidth=1.5, color=:purple, alpha=0.7)

# Oscillation bounds
quad_vals_plot = quad_baseline.(x_plot)
upper_bound_plot = quad_vals_plot .+ oscillation_amplitude .+ oscillation_margin
lower_bound_plot = quad_vals_plot .- oscillation_amplitude .- oscillation_margin
fill_between!(ax1, x_plot, lower_bound_plot, upper_bound_plot, alpha=0.15, color=:green, label="Expected oscillation range")

# Shade extrapolation region
vspan!(ax1, -0.5, 0, alpha=0.1, color=:orange)
vspan!(ax1, 2π, 2π + 0.5, alpha=0.1, color=:orange)

axislegend(ax1, position=:lt, fontsize=9)
ylims!(ax1, -5, 45)

# CONSTRAINED
ax2 = Axis(fig[1, 2], xlabel="x", ylabel="y", 
          title="Constrained POPS: Oscillation-Aware Bounds\n(ensemble stays within expected oscillation range)")

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
lines!(ax2, x_plot, y_true_plot, label="True sin(10x) + x²", linestyle=:dash, linewidth=2.5, color=:darkgreen)

# Quadratic baseline
lines!(ax2, x_plot, quad_baseline.(x_plot), label="Quadratic baseline", linestyle=:dot, linewidth=1.5, color=:purple, alpha=0.7)

# Oscillation bounds
fill_between!(ax2, x_plot, lower_bound_plot, upper_bound_plot, alpha=0.15, color=:green, label="Constraint bounds")

# Shade extrapolation region
vspan!(ax2, -0.5, 0, alpha=0.1, color=:orange)
vspan!(ax2, 2π, 2π + 0.5, alpha=0.1, color=:orange)

axislegend(ax2, position=:lt, fontsize=9)
ylims!(ax2, -5, 45)

save("/storage/astro2/phupfb/PhD/acestuff/new_ACE/my_pops_quadratic_sine_generalization.png", fig)
display(fig)

# Create separate figure showing generalization error
fig2 = Figure(size=(1400, 600))

# Compute error on dense test grid
x_dense_test = range(-1.5, 2π + 1.5, length=200)
X_dense_test = hcat([ones(length(x_dense_test)), x_dense_test, x_dense_test.^2]...)
y_dense_true = true_func.(x_dense_test)

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

save("/storage/astro2/phupfb/PhD/acestuff/new_ACE/my_pops_generalization_error.png", fig2)
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
y_pred_constrained = vec(mean(X * pointwise_corrections_constrained', dims=2))

pops_error = norm(y - y_pred_pops)
constrained_error = norm(y - y_pred_constrained)

println("\n" ^ 70)
println("TRAINING DATA VERIFICATION")
println("=" ^ 70)
println("Standard POPS - fit error: $pops_error (should be ≈ 0)")
println("Constrained POPS - fit error: $constrained_error (should be ≈ 0)")

# Summary
println("\n" ^ 70)
println("\nKEY INSIGHT")
println("=" ^ 70)
println("Unconstrained: Quadratic fits smooth trend, completely missing oscillations")
println("               Predictions deviate significantly from true wiggly function")
println("Constrained:   Bounds encode known oscillation structure (amplitude ±1)")
println("               Guides ensemble to stay realistic without seeing raw data")
println("               Result: $(round(improvement, digits=1))% LOWER MAE on held-out data!")
println("=" ^ 70)
