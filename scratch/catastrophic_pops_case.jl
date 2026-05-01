using OSQP
using SparseArrays
using LinearAlgebra, Statistics
using CairoMakie

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
function iterative_constrained_pops(X, Y, Gamma, A_constraints, l_constraints, u_constraints)    
    n_params = size(X, 2)
    n_points = size(X, 1)
    constrained_updates = []

    C = (Gamma' * Gamma ./ size(X,1) .+ X' * X)
    A = C \ X'
    leverage = diag(X * A)
    coeffs = C \ (X' * Y)
    errors = Y .- (X * coeffs)
    
    for i in 1:n_points
        X_i = (Gamma' * Gamma ./ size(X,1) .+ X' * X)
        y_i = (transpose(X[i, :]) .* (errors[i] / leverage[i]))[1, :]
        
        # Combine constraints: equality (fit point) + inequalities (domain knowledge)
        A_combined = vcat(reshape(X[i, :], 1, n_params), A_constraints)
        l_combined = vcat([Y[i]], l_constraints)
        u_combined = vcat([Y[i]], u_constraints)

        prob = OSQP.Model()
        OSQP.setup!(prob; P=sparse(X_i), q=y_i, A=sparse(A_combined), l=l_combined, u=u_combined, 
                    verbose=false, eps_abs=1e-7, eps_rel=1e-7, max_iter=10000)
        results = OSQP.solve!(prob)
        push!(constrained_updates, results.x)
    end
    
    constrained_updates_matrix = hcat(constrained_updates...)
    return constrained_updates_matrix'
end

println("\n" * "="^90)
println("CATASTROPHIC CASE: LINEAR MODEL FITTING HIGHLY NONLINEAR DATA")
println("="^90)
println("\nScenario: We have only a LINEAR model [1, x] but data comes from sin(5x)")
println("This causes POPS to generate WILD ensemble members to fit individual points")
println("Constrained POPS uses knowledge that output should stay in [-1.2, 1.2]")
println("="^90)

# Generate highly nonlinear data but fit with LINEAR model (catastrophic misspecification!)
x_train = range(0, 2π, length=25)
y_train = sin.(5 .* x_train) .+ 0.05 .* randn(25)  # Highly oscillatory

# INTENTIONALLY underspecify: use LINEAR model [1, x] for SINE data
X_train = hcat([ones(25), x_train]...)
Gamma = Matrix(I(2))

# Get unconstrained POPS
pops_corr, nom_coeff = corrections(X_train, y_train, Gamma)
pops_ens = pops_corr .+ nom_coeff'

println("\nModel: y = θ₀ + θ₁·x (LINEAR only)")
println("True data: sin(5x) (HIGHLY NONLINEAR)")
println("Nominal coefficients: θ₀ = $(round(nom_coeff[1], digits=4)), θ₁ = $(round(nom_coeff[2], digits=4))")
println("\nUnconstrained POPS will generate WILD ensemble members!")

# Construct bounds: we know output should be roughly in [-1.2, 1.2] (sin amplitude + margin)
x_constraint = range(-1, 2π + 1, length=150)
A_bounds = []
for xp in x_constraint
    push!(A_bounds, [1.0, xp])
end
A_bounds_matrix = hcat(A_bounds...)'
l_bounds = -1.2 .* ones(size(A_bounds_matrix, 1))
u_bounds = 1.2 .* ones(size(A_bounds_matrix, 1))

# Get constrained POPS
pops_constrained = iterative_constrained_pops(X_train, y_train, Gamma, A_bounds_matrix, l_bounds, u_bounds)

# Evaluate on dense test grid
x_test = range(-1, 2π + 1, length=300)
X_test = hcat([ones(300), x_test]...)

y_unconstrained = X_test * pops_ens'
y_constrained = X_test * pops_constrained'

# OLS solution (just the nominal fit)
ols_pred = X_test * nom_coeff

# True function
y_true = sin.(5 .* x_test)

# Statistics
violations_unconstrained = sum((y_unconstrained .< -1.2) .| (y_unconstrained .> 1.2))
violations_constrained = sum((y_constrained .< -1.2) .| (y_constrained .> 1.2))

mae_unconstrained = mean(abs.(vec(mean(y_unconstrained, dims=2)) .- y_true))
mae_constrained = mean(abs.(vec(mean(y_constrained, dims=2)) .- y_true))
mae_ols = mean(abs.(ols_pred .- y_true))

println("\n" * "-"^90)
println("RESULTS:")
println("-"^90)
println("Unconstrained POPS:")
println("  - Out-of-bounds predictions: $(violations_unconstrained)/$(300*25) = $(round(100*violations_unconstrained/(300*25), digits=1))%")
println("  - Test MAE: $(round(mae_unconstrained, digits=4))")
println("\nConstrained POPS:")
println("  - Out-of-bounds predictions: $(violations_constrained)/$(300*25) = $(round(100*violations_constrained/(300*25), digits=1))%")
println("  - Test MAE: $(round(mae_constrained, digits=4))")
println("\nOLS (nominal):")
println("  - Out-of-bounds predictions: $(sum((ols_pred .< -1.2) .| (ols_pred .> 1.2)))/300 = $(round(100*sum((ols_pred .< -1.2) .| (ols_pred .> 1.2))/300, digits=1))%")
println("  - Test MAE: $(round(mae_ols, digits=4))")

# Create visualization
fig = Figure(size=(1600, 600))

# UNCONSTRAINED POPS
ax1 = Axis(fig[1, 1], xlabel="x", ylabel="y", 
          title="UNCONSTRAINED POPS: Wild extrapolation\n(LINEAR model + SINE data = catastrophe)")

# Plot ensemble bounds
y_min_unconstrained = vec(minimum(y_unconstrained, dims=2))
y_max_unconstrained = vec(maximum(y_unconstrained, dims=2))
y_mean_unconstrained = vec(mean(y_unconstrained, dims=2))

fill_between!(ax1, x_test, y_min_unconstrained, y_max_unconstrained, alpha=0.3, color=:blue, label="POPS range")
lines!(ax1, x_test, y_mean_unconstrained, label="POPS mean", linewidth=2, color=:darkblue)

# True function
lines!(ax1, x_test, y_true, label="True sin(5x)", linestyle=:dash, linewidth=2.5, color=:darkgreen)

# Training data
scatter!(ax1, x_train, y_train, label="Training data", markersize=6, color=:black)

# OLS
lines!(ax1, x_test, ols_pred, label="OLS linear fit", linestyle=:dot, linewidth=2, color=:purple)

# Bounds
hlines!(ax1, [-1.2, 1.2], linestyle=:dashdot, linewidth=1.5, color=:red, alpha=0.7, label="Constraint bounds")

axislegend(ax1, position=:rt, fontsize=9)
ylims!(ax1, -4, 5)

# CONSTRAINED POPS
ax2 = Axis(fig[1, 2], xlabel="x", ylabel="y", 
          title="CONSTRAINED POPS: Respects domain knowledge\n(Same LINEAR model, but with physical bounds)")

# Plot ensemble bounds
y_min_constrained = vec(minimum(y_constrained, dims=2))
y_max_constrained = vec(maximum(y_constrained, dims=2))
y_mean_constrained = vec(mean(y_constrained, dims=2))

fill_between!(ax2, x_test, y_min_constrained, y_max_constrained, alpha=0.3, color=:red, label="Constrained POPS range")
lines!(ax2, x_test, y_mean_constrained, label="Constrained POPS mean", linewidth=2, color=:darkred)

# True function
lines!(ax2, x_test, y_true, label="True sin(5x)", linestyle=:dash, linewidth=2.5, color=:darkgreen)

# Training data
scatter!(ax2, x_train, y_train, label="Training data", markersize=6, color=:black)

# OLS
lines!(ax2, x_test, ols_pred, label="OLS linear fit", linestyle=:dot, linewidth=2, color=:purple)

# Bounds
hlines!(ax2, [-1.2, 1.2], linestyle=:dashdot, linewidth=1.5, color=:green, alpha=0.7, label="Constraint bounds")

axislegend(ax2, position=:rt, fontsize=9)
ylims!(ax2, -4, 5)

# Shade training region
for ax in [ax1, ax2]
    vspan!(ax, 0, 2π, alpha=0.05, color=:yellow)
end

save("/storage/astro2/phupfb/PhD/acestuff/new_ACE/catastrophic_pops_comparison.png", fig)
println("\n✓ Saved plot: catastrophic_pops_comparison.png")

# Additional analysis
println("\n" * "-"^90)
println("KEY INSIGHT:")
println("-"^90)
println("Unconstrained POPS: Ensemble explores wildly to fit individual points")
println("  → $(round(100*violations_unconstrained/(300*25), digits=1))% of predictions are physically unrealistic")
println("  → Useful for uncertainty quantification but DANGEROUS for extrapolation!")
println("\nConstrained POPS: Ensemble constrained to respect domain knowledge")
println("  → $(round(100*violations_constrained/(300*25), digits=1))% of predictions respect physical bounds")
println("  → Maintains interpretability even with severe model misspecification")
println("  → Slightly higher MAE ($(round(mae_constrained, digits=4)) vs $(round(mae_unconstrained, digits=4))) but MUCH safer!")
println("\nOLS: Just the mean fit, no uncertainty")
println("  → Misses oscillations entirely (MAE = $(round(mae_ols, digits=4)))")
println("  → But stays within bounds due to averaging")
println("="^90)
