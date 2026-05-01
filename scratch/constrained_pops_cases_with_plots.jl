using OSQP
using SparseArrays
using LinearAlgebra
using Statistics
using CairoMakie
using Printf

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

# Helper function to compute max/min bounds of ensemble
function compute_ensemble_bounds(X_test, ensemble)
    n_ensemble = size(ensemble, 1)
    predictions = X_test * ensemble'
    ensemble_max = vec(maximum(predictions, dims=2))
    ensemble_min = vec(minimum(predictions, dims=2))
    return ensemble_min, ensemble_max
end

# ============================================================================
# CASE 1: Monotonicity constraint - enforcing monotonic behavior
# ============================================================================
println("\n" * "="^80)
println("CASE 1: MONOTONIC FUNCTION WITH MONOTONICITY CONSTRAINT")
println("="^80)

# Generate monotonically increasing data
x1 = range(0, 1, length=30)
y1 = x1.^2 .+ 0.1 .* randn(30)  # y = x^2 is monotonic increasing on [0,1]

X1 = hcat([ones(30), x1, x1.^2]...)
Gamma1 = Matrix(I(3))

# Get OLS solution (minimum loss)
ols_coeff1 = X1 \ y1

# Get unconstrained POPS
pops_corr1, nom_coeff1 = corrections(X1, y1, Gamma1)
pops_ens1 = pops_corr1 .+ nom_coeff1'

# Construct monotonicity constraint: enforce that y(x_i) <= y(x_{i+1}) for all i
# This means: [1, x_i, x_i^2] · θ <= [1, x_{i+1}, x_{i+1}^2] · θ
# Rearranged: ([1, x_{i+1}, x_{i+1}^2] - [1, x_i, x_i^2]) · θ >= 0
x_constraint1 = range(0, 1, length=50)  # Dense evaluation points
A_mono = []
for i in 1:length(x_constraint1)-1
    row = [0, x_constraint1[i+1] - x_constraint1[i], x_constraint1[i+1]^2 - x_constraint1[i]^2]
    push!(A_mono, row)
end
A_mono_matrix = hcat(A_mono...)'
l_mono = zeros(size(A_mono_matrix, 1))
u_mono = fill(Inf, size(A_mono_matrix, 1))

# Get constrained POPS with monotonicity
pops_constrained1 = iterative_constrained_pops(X1, y1, Gamma1, A_mono_matrix, l_mono, u_mono)

# Check monotonicity on test set
x_test1 = range(0, 1, length=100)
X_test1 = hcat([ones(100), x_test1, x_test1.^2]...)

y_unconstrained1 = X_test1 * pops_ens1'
y_constrained1 = X_test1 * pops_constrained1'
y_ols1 = X_test1 * ols_coeff1

# Compute bounds
unc_min1, unc_max1 = compute_ensemble_bounds(X_test1, pops_ens1)
con_min1, con_max1 = compute_ensemble_bounds(X_test1, pops_constrained1)

# Count monotonicity violations
check_mono(y_pred) = sum([y_pred[i+1, :] .< y_pred[i, :] for i in 1:size(y_pred,1)-1])
violations_unconstrained1 = sum(check_mono(y_unconstrained1))
violations_constrained1 = sum(check_mono(y_constrained1))

println("Unconstrained ensemble: $violations_unconstrained1 non-monotonic predictions")
println("Constrained ensemble:   $violations_constrained1 non-monotonic predictions")
println("✓ Monotonicity coverage: $(round((1 - violations_constrained1/(40*99))*100, digits=1))%")

# ============================================================================
# CASE 2: Bounded domain - enforcing physically reasonable bounds
# ============================================================================
println("\n" * "="^80)
println("CASE 2: BOUNDED FUNCTION (e.g., probability-like [0,1])")
println("="^80)

# Generate data bounded between 0 and 1 (like a sigmoid approximated by polynomial)
x2 = range(-3, 3, length=40)
y2 = 1 ./ (1 .+ exp.(-x2)) .+ 0.05 .* randn(40)  # Logistic function

X2 = hcat([ones(40), x2, x2.^2]...)
Gamma2 = Matrix(I(3))

# Get OLS solution
ols_coeff2 = X2 \ y2

# Get unconstrained POPS
pops_corr2, nom_coeff2 = corrections(X2, y2, Gamma2)
pops_ens2 = pops_corr2 .+ nom_coeff2'

# Construct bounds: 0 <= y <= 1 everywhere
x_constraint2 = range(-3, 3, length=100)
A_bounds = []
for xp in x_constraint2
    push!(A_bounds, [1.0, xp, xp^2])
end
A_bounds_matrix = hcat(A_bounds...)'
l_bounds = zeros(size(A_bounds_matrix, 1))
u_bounds = ones(size(A_bounds_matrix, 1))

# Get constrained POPS
pops_constrained2 = iterative_constrained_pops(X2, y2, Gamma2, A_bounds_matrix, l_bounds, u_bounds)

# Check bounds on test set
x_test2 = range(-3.5, 3.5, length=150)
X_test2 = hcat([ones(150), x_test2, x_test2.^2]...)

y_unconstrained2 = X_test2 * pops_ens2'
y_constrained2 = X_test2 * pops_constrained2'
y_ols2 = X_test2 * ols_coeff2

# Compute bounds
unc_min2, unc_max2 = compute_ensemble_bounds(X_test2, pops_ens2)
con_min2, con_max2 = compute_ensemble_bounds(X_test2, pops_constrained2)

violations_unconstrained2 = sum((y_unconstrained2 .< -0.01) .| (y_unconstrained2 .> 1.01))
violations_constrained2 = sum((y_constrained2 .< -0.01) .| (y_constrained2 .> 1.01))

println("Unconstrained ensemble: $(violations_unconstrained2) predictions outside [0,1]")
println("Constrained ensemble:   $(violations_constrained2) predictions outside [0,1]")
println("✓ Bounds coverage: $(round((1 - violations_constrained2/(150*40))*100, digits=1))%")

# ============================================================================
# CASE 3: Positivity constraint - enforcing y >= 0 (e.g., energy, intensity)
# ============================================================================
println("\n" * "="^80)
println("CASE 3: POSITIVE FUNCTION (e.g., energy, intensity, distance)")
println("="^80)

# Generate positive data
x3 = range(0, 5, length=35)
y3 = exp.(0.2 .* x3) .+ 0.1 .* randn(35)  # Exponential, always positive

X3 = hcat([ones(35), x3, x3.^2]...)
Gamma3 = Matrix(I(3))

# Get OLS solution
ols_coeff3 = X3 \ y3

# Get unconstrained POPS
pops_corr3, nom_coeff3 = corrections(X3, y3, Gamma3)
pops_ens3 = pops_corr3 .+ nom_coeff3'

# Construct positivity constraint: y >= 0 everywhere
x_constraint3 = range(-1, 6, length=100)
A_pos = []
for xp in x_constraint3
    push!(A_pos, [1.0, xp, xp^2])
end
A_pos_matrix = hcat(A_pos...)'
l_pos = zeros(size(A_pos_matrix, 1))
u_pos = fill(Inf, size(A_pos_matrix, 1))

# Get constrained POPS
pops_constrained3 = iterative_constrained_pops(X3, y3, Gamma3, A_pos_matrix, l_pos, u_pos)

# Check positivity on test set
x_test3 = range(-1.5, 6, length=200)
X_test3 = hcat([ones(200), x_test3, x_test3.^2]...)

y_unconstrained3 = X_test3 * pops_ens3'
y_constrained3 = X_test3 * pops_constrained3'
y_ols3 = X_test3 * ols_coeff3

# Compute bounds
unc_min3, unc_max3 = compute_ensemble_bounds(X_test3, pops_ens3)
con_min3, con_max3 = compute_ensemble_bounds(X_test3, pops_constrained3)

violations_unconstrained3 = sum(y_unconstrained3 .< -0.01)
violations_constrained3 = sum(y_constrained3 .< -0.01)

println("Unconstrained ensemble: $violations_unconstrained3 negative predictions")
println("Constrained ensemble:   $violations_constrained3 negative predictions")
println("✓ Positivity coverage: $(round((1 - violations_constrained3/(200*35))*100, digits=1))%")

# ============================================================================
# CASE 4: Convexity/Concavity constraint
# ============================================================================
println("\n" * "="^80)
println("CASE 4: CONVEX FUNCTION (e.g., cost function, quadratic effects)")
println("="^80)

# Generate convex data
x4 = range(-2, 2, length=40)
y4 = x4.^2 .+ 0.1 .* randn(40)  # Convex parabola

X4 = hcat([ones(40), x4, x4.^2]...)
Gamma4 = Matrix(I(3))

# Get OLS solution
ols_coeff4 = X4 \ y4

# Get unconstrained POPS
pops_corr4, nom_coeff4 = corrections(X4, y4, Gamma4)
pops_ens4 = pops_corr4 .+ nom_coeff4'

# Construct convexity constraint: second derivative >= 0
# For quadratic [1, x, x^2] · θ, second derivative = 2*θ_2 >= 0
# This means: just constrain the coefficient of x^2 to be positive
A_convex = [0.0, 0.0, 1.0]'  # Coefficient of x^2
l_convex = [0.0]
u_convex = [Inf]

# Get constrained POPS
pops_constrained4 = iterative_constrained_pops(X4, y4, Gamma4, A_convex, l_convex, u_convex)

# Check convexity: compute second derivative at test points
x_test4 = range(-3, 3, length=100)
X_test4 = hcat([ones(100), x_test4, x_test4.^2]...)

y_unconstrained4 = X_test4 * pops_ens4'
y_constrained4 = X_test4 * pops_constrained4'
y_ols4 = X_test4 * ols_coeff4

# Compute bounds
unc_min4, unc_max4 = compute_ensemble_bounds(X_test4, pops_ens4)
con_min4, con_max4 = compute_ensemble_bounds(X_test4, pops_constrained4)

# Second derivative of [1, x, x^2] · θ is 2*θ_2
second_deriv_unconstrained4 = [2 * pops_ens4[i, 3] for i in 1:size(pops_ens4, 1)]
second_deriv_constrained4 = [2 * pops_constrained4[i, 3] for i in 1:size(pops_constrained4, 1)]

violations_unconstrained4 = sum(second_deriv_unconstrained4 .< -0.01)
violations_constrained4 = sum(second_deriv_constrained4 .< -0.01)

println("Unconstrained ensemble: $violations_unconstrained4 non-convex ensemble members")
println("Constrained ensemble:   $violations_constrained4 non-convex ensemble members")
println("✓ Convexity coverage: $(round((1 - violations_constrained4/40)*100, digits=1))%")

# ============================================================================
# PLOTTING: 4-panel figure
# ============================================================================

fig = Figure(size=(1400, 1000))

# CASE 1: Monotonicity
ax1 = Axis(fig[1, 1], xlabel="x", ylabel="y", title="Case 1: Monotonicity Constraint")
scatter!(ax1, x1, y1, color=:black, markersize=6, label="Training data")
band!(ax1, x_test1, unc_min1, unc_max1, color=(:blue, 0.2), label="Unconstrained POPS bounds")
band!(ax1, x_test1, con_min1, con_max1, color=(:red, 0.3), label="Constrained POPS bounds")
lines!(ax1, x_test1, y_ols1, color=:green, linewidth=2, label="OLS solution", linestyle=:dash)
Legend(fig[1, 1], ax1, tellwidth=false, tellheight=false, halign=:left, valign=:top)

# CASE 2: Bounds [0,1]
ax2 = Axis(fig[1, 2], xlabel="x", ylabel="y", title="Case 2: Bounded [0,1]")
scatter!(ax2, x2, y2, color=:black, markersize=6, label="Training data")
band!(ax2, x_test2, unc_min2, unc_max2, color=(:blue, 0.2), label="Unconstrained POPS bounds")
band!(ax2, x_test2, con_min2, con_max2, color=(:red, 0.3), label="Constrained POPS bounds")
lines!(ax2, x_test2, y_ols2, color=:green, linewidth=2, label="OLS solution", linestyle=:dash)
hlines!(ax2, [0, 1], color=:orange, linewidth=2, linestyle=:dot, label="Constraint bounds")
Legend(fig[1, 2], ax2, tellwidth=false, tellheight=false, halign=:left, valign=:top)

# CASE 3: Positivity
ax3 = Axis(fig[2, 1], xlabel="x", ylabel="y", title="Case 3: Positivity Constraint (y ≥ 0)")
scatter!(ax3, x3, y3, color=:black, markersize=6, label="Training data")
band!(ax3, x_test3, unc_min3, unc_max3, color=(:blue, 0.2), label="Unconstrained POPS bounds")
band!(ax3, x_test3, con_min3, con_max3, color=(:red, 0.3), label="Constrained POPS bounds")
lines!(ax3, x_test3, y_ols3, color=:green, linewidth=2, label="OLS solution", linestyle=:dash)
hlines!(ax3, [0], color=:orange, linewidth=2, linestyle=:dot, label="y=0 constraint")
Legend(fig[2, 1], ax3, tellwidth=false, tellheight=false, halign=:left, valign=:top)

# CASE 4: Convexity
ax4 = Axis(fig[2, 2], xlabel="x", ylabel="y", title="Case 4: Convexity Constraint")
scatter!(ax4, x4, y4, color=:black, markersize=6, label="Training data")
band!(ax4, x_test4, unc_min4, unc_max4, color=(:blue, 0.2), label="Unconstrained POPS bounds")
band!(ax4, x_test4, con_min4, con_max4, color=(:red, 0.3), label="Constrained POPS bounds")
lines!(ax4, x_test4, y_ols4, color=:green, linewidth=2, label="OLS solution", linestyle=:dash)
Legend(fig[2, 2], ax4, tellwidth=false, tellheight=false, halign=:left, valign=:top)

save("constrained_pops_cases.png", fig)
println("\n✓ Figure saved to constrained_pops_cases.png")

# ============================================================================
# SUMMARY
# ============================================================================
println("\n" * "="^80)
println("SUMMARY: DOMAIN KNOWLEDGE CONSTRAINTS IN POPS")
println("="^80)
println("\n✓ Monotonicity:    $(round((1 - violations_constrained1/(40*99))*100, digits=1))% of predictions respect ordering")
println("✓ Bounds [0,1]:    $(round((1 - violations_constrained2/(150*40))*100, digits=1))% of predictions stay bounded")
println("✓ Positivity:      $(round((1 - violations_constrained3/(200*35))*100, digits=1))% of predictions remain positive")
println("✓ Convexity:       $(round((1 - violations_constrained4/40)*100, digits=1))% of ensemble members are convex")

println("\nKey insight: By encoding domain knowledge as constraints,")
println("we can guide POPS ensembles to respect physical/mathematical properties!")
println("="^80)
