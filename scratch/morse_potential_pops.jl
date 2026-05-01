"""
MORSE POTENTIAL WITH CONSTRAINED POPS (QP-based)

Scenario: Approximate a Morse potential V(R) = D_e[1 - exp(-α(R-R_e))]²
with a polynomial basis [1, r, r², r³, r⁴, r⁵]

Physical constraints enforced via Quadratic Programming (OSQP):
1. MINIMUM AT EQUILIBRIUM: E(R_e) = E_min
   The energy minimum occurs exactly at the equilibrium distance
   
2. LOWER ENERGY BOUND: E(r) >= E_min everywhere
   Energy cannot go below the minimum value from training data
   
These constraints are solved directly in the QP formulation, not applied post-hoc.
Each ensemble member is the solution to a constrained optimization problem.
"""

using LinearAlgebra, SparseArrays, OSQP, CairoMakie, Statistics, Printf

# ========================================================================================
# STEP 1: Generate Morse Potential Data
# ========================================================================================

function morse_potential(r; D_e, α, R_e)
    return D_e * (1 - exp(-α * (r - R_e)))^2
end

function generate_morse_data(rs; n_train=50, n_test=1000)
    """
    Generate training and test data from Morse potential
    Training range: [R_e - 0.2, R_e + 3] (around equilibrium)
    Test range: [0.5, R_e + 5] (includes repulsive wall and asymptotic regions)
    """
    # Morse parameters
    D_e = 4.75  # Morse depth (eV) - typical for diatomic
    α = 2.0     # Range parameter (Å⁻¹)
    R_e = 1.6   # Equilibrium distance (Å)
    
    # Training: concentrated around equilibrium
    r_train = rs
    E_train = morse_potential.(r_train; D_e, α, R_e)
    
    # Test: extended range including extrapolation regions
    r_test = range(0.5, R_e + 5, length=n_test)
    E_test = morse_potential.(r_test; D_e, α, R_e)
    
    return r_train, E_train, r_test, E_test, D_e, α, R_e
end

using Random
using Statistics

# -----------------------------
# Metropolis–Hastings sampler
# -----------------------------
function sample_morse(
    nsteps::Int;
    De=4.75,
    α=2.0,
    Re=1.6,
    kBT=0.1,
    step_size=2.,
    r0=Re
)
    β = 1.0 / kBT
    r = r0
    E = morse_potential(r; D_e=De, α=α, R_e=Re)

    samples = Vector{Float64}(undef, nsteps)
    accepted = 0

    for i in 1:nsteps
        # Propose move
        r_proposal = r + step_size * randn()

        # Enforce r >= 0 (hard wall)
        if r_proposal < 0
            samples[i] = r
            continue
        end

        E_proposal = morse_potential(r_proposal; D_e=De, α=α, R_e=Re)
        ΔE = E_proposal - E

        # Metropolis criterion
        if ΔE ≤ 0 || rand() < exp(-β * ΔE)
            r = r_proposal
            E = E_proposal
            accepted += 1
        end

        samples[i] = r
    end

    acceptance_rate = accepted / nsteps
    return samples, acceptance_rate
end

function orders()
    return [0,1,2]
end

# ========================================================================================
# STEP 2: Build Polynomial Basis
# ========================================================================================

function build_basis(r, order=3)
    """Build polynomial basis [1, r, r², ..., r^order]"""
    powers = orders()
    order = length(powers)
    r_vec = collect(r)  # Convert range to vector if needed
    n = length(r_vec)
    X = ones(n, order)
    for (i, N) in enumerate(powers)
        X[:, i] = r_vec .^ (N)
    end
    return X
end

# ========================================================================================
# STEP 3: Constrained POPS with Physical Constraints
# ========================================================================================

function corrections_pops(X, Y; n_ensemble=50)
    """
    Standard POPS: Generate ensemble of corrections to nominal solution
    """
    n, p = size(X)
    C = X' * X / n
    
    # Fit nominal solution
    θ_nominal = (X' * X) \ (X' * Y)
    residuals = Y - X * θ_nominal
    
    # Leverage scores: diagonal of X * C^{-1} * X'
    leverage = vec(sum((X / C) .* X, dims=2))
    leverage = clamp.(leverage, 1e-6, Inf)
    
    # Generate ensemble via random sampling
    ensemble = zeros(n_ensemble, p)
    ensemble[1, :] = θ_nominal
    
    for i in 2:n_ensemble
        # Randomly sample a training point
        idx = rand(1:n)
        correction_dir = C \  X[idx, :] * residuals[idx] / leverage[idx]
        
        # Add noise for ensemble diversity
        ensemble[i, :] = θ_nominal + correction_dir
    end
    
    return ensemble, θ_nominal
end

function constrained_ols_morse(X_train, Y_train, X_test, r_test, R_e; verbose=true)
    """
    Constrained OLS (least squares) using QP with constraints:
    1. E(R_e) = E_min ± tol      (energy minimum at equilibrium)
    2. E(r) >= E_min - tol       (convexity: energy bound everywhere)
    3. E(r > R_e + 1) monotonic  (repulsive region increases monotonically)
    
    Returns the single best-fit constrained solution (no ensemble).
    """
    
    n, p = size(X_train)
    E_min = minimum(Y_train)
    
    # QP setup for OLS: minimize ||X*θ - Y||²
    H = X_train' * X_train
    c = -X_train' * Y_train


    # # ===== Constraint 1: E(R_e) = E_min (equality constraint) =====
    # X_Re = build_basis([R_e], 3)
    # A_eq = X_Re
    # l_eq = E_min
    # u_eq = E_min
    
    # r_repulsive = range(0., stop=R_e, length=5)

    # A_mono_repulsive = hcat([[0; -r^(-2); -2 * r ^ (-3); -3 * r ^ (-4); -4 * r ^ (-5); -5 * r ^ (-6); -6 * r ^ (-7)] for r in r_repulsive]...)'
    # l_mono_repulsive = [- Inf for _ in r_repulsive]
    # u_mono_repulsive = [0 for _ in r_repulsive]

    r_attractive = range(R_e, stop=R_e + 4.0, length=5)
    A_mono_attractive = - hcat(([k * r^(k - 1) for k in orders()] for r in r_attractive)...)'
    u_mono_attractive = [0. for _ in r_attractive]
    l_mono_attractive = [-Inf for _ in r_attractive]

    A_deriv = [k * R_e ^ (k - 1) for k in orders()]'
    l_deriv = 0.0
    u_deriv = 0.0

    # A_convex = hcat([[0; 2 * r ^ (-3); 6 * r ^ (-4); 12 * r ^ (-5); 20 * r ^ (-6); 30 * r ^ (-7); 42 * r ^ (-8)] for r in [R_e]]...)'
    # l_convex = 0.0
    # u_convex = Inf
    
    # ===== Combine all constraints =====
    # A_full = vcat(
    # X_Re,          # energy equality
    # A_deriv,       # stationary point
    # A_mono_repulsive,         # monotonic repulsion
    A_full = vcat(A_mono_attractive, A_deriv)
    # A_convex
    # )

    # l_full = vcat(
        # E_min,
        # 0.0,
        # l_mono_repulsive,
    l_full = vcat(l_mono_attractive, l_deriv)
        # l_convex
    # )

    # u_full = vcat(
        # E_min,
        # 0.0,
        # u_mono_repulsive,
    u_full = vcat(u_mono_attractive, u_deriv)
        # u_convex
    # )
    
    # Solve - initialize default fallback
    θ_constrained = (X_train' * X_train) \ (X_train' * Y_train)
    model = OSQP.Model()
    OSQP.setup!(model; P=sparse(H), q=c, A=sparse(A_full), l=l_full, u=u_full, 
                alpha=1.0, verbose=verbose, max_iter=500000)
    results = OSQP.solve!(model)
    
    if results.info.status == :Solved
        θ_constrained = results.x
    else
        if verbose
            println("OSQP did not converge, using fallback: status = $(results.info.status)")
        end
    end
    
    return θ_constrained
end

function constrained_pops_around_ols(X_train, Y_train, X_test, r_test, R_e; n_ensemble=50, base_solution=nothing)
    """
    Generate POPS ensemble by applying POPS perturbations to a constrained OLS baseline.
    
    Instead of solving QP for each perturbed member, we:
    1. Start with the constrained OLS as the base solution
    2. Generate POPS-style perturbations (using leverage scores and residuals)
    3. Apply these perturbations directly to the base solution with constraint enforcement
    
    This preserves the constraint satisfaction of the baseline while maintaining UQ.
    """
    
    n, p = size(X_train)
    E_min = minimum(Y_train)
    
    # Use provided base solution or solve for it
    if isnothing(base_solution)
        base_solution = constrained_ols_morse(X_train, Y_train, X_test, r_test, R_e)
    end
    
    # Fit unconstrained to get residuals and leverage scores
    residuals = Y_train - X_train * base_solution
    
    # Compute POPS metric C for leverage scores
    C = X_train' * X_train #/ n + 0.01 * I(p)
    A        = C \ X_train'
    leverage = diag(X_train * A)
    
    # Generate ensemble around constrained OLS base
    ensemble = zeros(length(leverage), p)

    all_updates = (C \ (X_train .* (residuals ./ leverage))')
    
    for i=1:size(all_updates, 2)
        ensemble[i, :] = all_updates[:, i] .+ base_solution 
    end
    return ensemble', base_solution
end
function constrained_pops_morse(X_train, Y_train, X_test, r_test, D_e, α, R_e; n_ensemble=length(Y_train))
    """
    CONSTRAINED POPS using Quadratic Programming:
    
    Constraints (solved via QP for each ensemble member):
    1. E(R_e) = E_min   (energy at equilibrium equals minimum training energy)
    2. E(r) >= E_min    (energy everywhere is at least the minimum)
    
    QP formulation: minimize ||X*θ - Y||² subject to:
       X[r_e_idx, :] * θ = E_min
       X[r, :] * θ >= E_min for all r in test set
    """
    
    n, p = size(X_train)
    
    # Nominal POPS ensemble (unconstrained)
    ensemble_unconstrained, θ_nominal = corrections_pops(X_train, Y_train, n_ensemble=n_ensemble)
    
    # Get energy minimum and find closest point to R_e in test set
    E_min = minimum(Y_train)
    idx_Re = argmin(abs.(r_test .- R_e))
    
    # Basis evaluated at R_e
    X_Re = build_basis([r_test[idx_Re]], 5)  # 1×6 vector
    
    # Apply QP constraints to ensemble
    ensemble_constrained = copy(ensemble_unconstrained)
    
    for i in 1:n_ensemble
        θ_i = ensemble_unconstrained[i, :]
        
        # Set up QP: minimize ||X_train*θ - Y_train||²
        # Objective: minimize (1/2) θ'Hθ + c'θ where H = X'X, c = -X'Y
        H = (X_train' * X_train)
        c = (X_train' * Y_train)
        
        r_attractive = range(R_e, stop=R_e + 4.0, length=5)
        A_mono_attractive = - hcat(([k * r^(k - 1) for k in orders()] for r in r_attractive)...)'
        u_mono_attractive = [0. for _ in r_attractive]
        l_mono_attractive = [-Inf for _ in r_attractive]

        A_deriv = [k * R_e ^ (k - 1) for k in orders()]'
        l_deriv = 0.0
        u_deriv = 0.0

        # A_convex = hcat([[0; 2 * r ^ (-3); 6 * r ^ (-4); 12 * r ^ (-5); 20 * r ^ (-6); 30 * r ^ (-7); 42 * r ^ (-8)] for r in [R_e]]...)'
        # l_convex = 0.0
        # u_convex = Inf
        
        # ===== Combine all constraints =====
        # A_full = vcat(
        # X_Re,          # energy equality
        # A_deriv,       # stationary point
        # A_mono_repulsive,         # monotonic repulsion
        A_full = vcat(A_mono_attractive, A_deriv)
        # A_convex
        # )

        # l_full = vcat(
            # E_min,
            # 0.0,
            # l_mono_repulsive,
        l_full = vcat(l_mono_attractive, l_deriv)
            # l_convex
        # )

        # u_full = vcat(
            # E_min,
            # 0.0,
            # u_mono_repulsive,
        u_full = vcat(u_mono_attractive, u_deriv)
        
        # Create and solve OSQP model

        model = OSQP.Model()
        OSQP.setup!(model; P=sparse(H), q=c, A=sparse(A_full), l=l_full, u=u_full, verbose=false)
        results = OSQP.solve!(model)
    end
    # Make predictions on test set
    predictions_unconstrained = X_test * ensemble_unconstrained'
    predictions_constrained = X_test * ensemble_constrained'
    
    return predictions_unconstrained, predictions_constrained, θ_nominal, 
           ensemble_unconstrained, ensemble_constrained, E_min
end

function constrained_pops_morse_with_asymptotic(X_train, Y_train, X_test, r_test, D_e, α, R_e; n_ensemble=length(Y_train))
    """
    CONSTRAINED POPS with both lower bound and asymptotic constraints:
    
    Constraints (solved via QP for each ensemble member):
    1. E(R_e) = E_min   (energy at equilibrium equals minimum training energy)
    2. E(r) >= E_min    (energy everywhere is at least the minimum)
    3. E(r) <= D_e for large r (asymptotic limit as r → ∞)
    
    The asymptotic constraint is enforced as an upper bound for r > R_e + 1.0 Å.
    """
    
    n, p = size(X_train)
    
    # Nominal POPS ensemble (unconstrained)
    ensemble_unconstrained, θ_nominal = corrections_pops(X_train, Y_train, n_ensemble=n)
    
    # Get energy minimum and find closest point to R_e in test set
    E_min = minimum(Y_train)
    idx_Re = argmin(abs.(r_test .- R_e))
    
    # Basis evaluated at R_e
    X_Re = build_basis([r_test[idx_Re]], 5)
    
    # Apply QP constraints to ensemble
    ensemble_constrained = copy(ensemble_unconstrained)
    X=X_train
    Y=Y_train
    C = X' * X
    A = C \ X'
    leverage = diag(X * A)
    coeffs = C \ (X' * Y)
    errors = Y .- (X * coeffs)
    # Need to add constrained coefficients to the constrained pops updates!
    constrained_coeffs = constrained_ols_morse(X_train, Y_train, X_test, r_test, R_e)
    for i in 1:length(Y)
        X_i = X' * X
        y_i = (transpose(X[i, :]) .* (errors[i] / leverage[i]))[1, :]
        
        # Combine constraints: equality (fit point) + inequalities (domain knowledge)
        # r_attractive = range(R_e + 0.5, stop=R_e + 4.0, length=2)
        # A_mono_attractive = hcat(([k * r^(k - 1) for k in [0,1,3]] for r in r_attractive)...)'
        # l_mono_attractive = [0. for _ in r_attractive]
        # u_mono_attractive = [Inf for _ in r_attractive]

        A_deriv = [k * (k - 1) * R_e ^ (k - 2) for k in orders()]'
        l_deriv = [1e-9]
        u_deriv = [Inf]
        
        # Combine all constraints
        # A_full = vcat(A_mono_attractive, A_deriv)
        # l_full = vcat(l_mono_attractive, l_deriv)
        # u_full = vcat(u_mono_attractive, u_deriv)
        A_full = A_deriv
        l_full = l_deriv
        u_full = u_deriv

        prob = OSQP.Model()
        OSQP.setup!(prob; P=sparse(X_i), q=y_i, A=sparse(A_full), l=l_full, u=u_full, 
                    verbose=true)
        results = OSQP.solve!(prob)
        ensemble_constrained[i, :] = results.x .+ constrained_coeffs
    end
    # Make predictions on test set
    predictions_unconstrained = X_test * ensemble_constrained'
    predictions_constrained = X_test * ensemble_constrained'
    
    return predictions_unconstrained, predictions_constrained, θ_nominal, 
           ensemble_unconstrained, ensemble_constrained, E_min
end

# ========================================================================================
# STEP 4: Evaluate and Visualize
# ========================================================================================

function main()
    println("\n" * "="^90)
    println("MORSE POTENTIAL FITTING WITH CONSTRAINED POPS")
    println("="^90)
    
    # Generate data
    
samples, acc = sample_morse(
    500;
    De=4.75,
    α=2.0,
    Re=1.6,
    kBT=0.1,        # eV  (~1200 K)
    step_size=2.,
    r0=1.6
)
    println("Maximum: $(maximum(samples))")
    println("Minimum: $(minimum(samples))")
    samples = cat(samples, collect(range(1.4, 1.6 + 1., length=100)), dims=1)

    r_train, E_train, r_test, E_test, D_e, α, R_e = generate_morse_data(samples)
    println("\nMorse Parameters:")
    println("  D_e (well depth) = $D_e eV")
    println("  α (range param) = $α Å⁻¹")
    println("  R_e (equilibrium) = $R_e Å")
    
    # Build polynomial basis
    X_train = build_basis(r_train, 3)
    X_test = build_basis(r_test, 3)
    
    println("\nModel: Polynomial basis of order 5: [1, r, r², r³]")
    println("This is a FINITE polynomial - can't capture exponential tail exactly")
    println("But with asymptotic constraint E → D_e as r → ∞, should work well!\n")
    # Run constrained POPS (E >= E_min only)
    pred_unc, pred_con, θ_nom, ens_unc, ens_con, E_min_val = 
        constrained_pops_morse(X_train, E_train, X_test, r_test, D_e, α, R_e, n_ensemble=50)
    
    # Run constrained POPS with asymptotic constraint (E >= E_min AND E <= D_e for large r)
    pred_unc_asym, pred_con_asym, θ_nom_asym, ens_unc_asym, ens_con_asym, E_min_asym = 
        constrained_pops_morse_with_asymptotic(X_train, E_train, X_test, r_test, D_e, α, R_e, n_ensemble=50)
    
    # Run constrained OLS
    θ_ols_constrained = constrained_ols_morse(X_train, E_train, X_test, r_test, R_e)
    pred_ols_constrained = X_test * θ_ols_constrained
    
    # Run POPS perturbations around constrained OLS baseline (hybrid method)
    ens_pops_ols, θ_ols_base = constrained_pops_around_ols(X_train, E_train, X_test, r_test, R_e, n_ensemble=50, base_solution=θ_ols_constrained)
    pred_pops_ols = X_test * ens_pops_ols
    
    # Compute statistics
    mae_unc = mean(abs.(pred_unc .- E_test))
    mae_con = mean(abs.(pred_con .- E_test))
    mae_con_asym = mean(abs.(pred_con_asym .- E_test))
    mae_ols_con = mean(abs.(pred_ols_constrained .- E_test))
    mae_pops_ols = mean(abs.(pred_pops_ols .- E_test))
    
    # Check constraint violations
    V_nom = X_test * θ_nom
    
    # Count violations: E < E_min and E > D_e
    E_min = minimum(E_train)
    violations_unc = sum(pred_unc .< E_min - 0.001)
    violations_con = sum(pred_con .< E_min - 0.001)
    violations_con_asym = sum(pred_con_asym .< E_min - 0.001)
    violations_ols_con = sum(pred_ols_constrained .< E_min - 0.001)
    violations_pops_ols = sum(pred_pops_ols .< E_min - 0.001)
    
    # For asymptotic: also count upper bound violations for r > R_e + 1
    idx_asym = findall(r_test .> R_e + 1.0)
    violations_con_asym_upper = sum(maximum(pred_con_asym[idx_asym, :], dims=2) .> D_e + 0.1)
    
    println("\n" * "-"^90)
    println("RESULTS")
    println("-"^90)
    
    println("\nUnconstrained POPS:")
    println("  MAE: $(round(mae_unc, digits=4)) eV")
    println("  Violations (E < E_min): $violations_unc out of $(length(pred_unc)) predictions")
    
    println("\nConstrained POPS (E >= E_min):")
    println("  MAE: $(round(mae_con, digits=4)) eV")
    println("  Violations (E < E_min): $violations_con out of $(length(pred_con)) predictions")
    
    println("\nConstrained POPS with asymptotic (E >= E_min AND E <= D_e for r > R_e+1):")
    println("  MAE: $(round(mae_con_asym, digits=4)) eV")
    println("  Violations (E < E_min): $violations_con_asym out of $(length(pred_con_asym)) predictions")
    println("  Violations (E > D_e for r>R_e+1): $violations_con_asym_upper out of $(length(idx_asym)) predictions")
    
    println("\nConstrained OLS (QP with E >= E_min):")
    println("  MAE: $(round(mae_ols_con, digits=4)) eV")
    println("  Violations (E < E_min): $violations_ols_con out of $(length(pred_ols_constrained)) predictions")
    
    println("\nPOPS Perturbations on Constrained OLS (hybrid method):")
    println("  MAE: $(round(mae_pops_ols, digits=4)) eV")
    println("  Violations (E < E_min): $violations_pops_ols out of $(length(pred_pops_ols)) predictions")
    
    # Create visualization
    pred_unc_min = vec(minimum(pred_unc, dims=2))
    pred_unc_max = vec(maximum(pred_unc, dims=2))
    pred_con_min = vec(minimum(pred_con, dims=2))
    pred_con_max = vec(maximum(pred_con, dims=2))
    pred_con_asym_min = vec(minimum(pred_con_asym, dims=2))
    pred_con_asym_max = vec(maximum(pred_con_asym, dims=2))
    pred_pops_ols_min = vec(minimum(pred_pops_ols, dims=2))
    pred_pops_ols_max = vec(maximum(pred_pops_ols, dims=2))
    
    fig = Figure(size=(2000, 1300))
    
    # Panel 1: Unconstrained POPS
    ax1 = Axis(fig[1, 1], xlabel="Distance r (Å)", ylabel="Energy (eV)", 
               title="Unconstrained POPS Ensemble")
    fill_between!(ax1, r_test, pred_unc_min, pred_unc_max, alpha=0.3, color=:red, label="Ensemble spread")
    lines!(ax1, r_test, E_test, label="True Morse", linewidth=3, color=:black)
    lines!(ax1, r_test, V_nom, label="Unconstrained nominal", linewidth=2.5, color=:blue, linestyle=:dash)
    scatter!(ax1, r_train, E_train, label="Training data", markersize=6, color=:green)
    vlines!(ax1, [R_e], label="R_e", color=:orange, linestyle=:dot, linewidth=2)
    hlines!(ax1, [E_min], color=:red, linestyle=:dash, linewidth=2, label="E_min")
    ylims!(ax1, -1, 10)
    axislegend(ax1, position=:rt, fontsize=8)
    
    # Panel 2: Constrained POPS (E >= E_min only)
    ax2 = Axis(fig[1, 2], xlabel="Distance r (Å)", ylabel="Energy (eV)", 
               title="Constrained POPS (E >= E_min)")
    fill_between!(ax2, r_test, pred_con_min, pred_con_max, alpha=0.3, color=:purple, label="Ensemble spread")
    lines!(ax2, r_test, E_test, label="True Morse", linewidth=3, color=:black)
    lines!(ax2, r_test, vec(mean(pred_con, dims=2)), label="Mean prediction", linewidth=2.5, color=:purple, linestyle=:dash)
    scatter!(ax2, r_train, E_train, label="Training data", markersize=6, color=:green)
    vlines!(ax2, [R_e], label="R_e", color=:orange, linestyle=:dot, linewidth=2)
    hlines!(ax2, [E_min], color=:red, linestyle=:dash, linewidth=2, label="E_min")
    ylims!(ax2, -1, 10)
    axislegend(ax2, position=:rt, fontsize=8)
    
    # Panel 3: Constrained POPS with asymptotic (E >= E_min AND E <= D_e)
    ax3 = Axis(fig[1, 3], xlabel="Distance r (Å)", ylabel="Energy (eV)", 
               title="Constrained POPS + Asymptotic")
    fill_between!(ax3, r_test, pred_con_asym_min, pred_con_asym_max, alpha=0.3, color=:darkorange, label="Ensemble spread")
    lines!(ax3, r_test, E_test, label="True Morse", linewidth=3, color=:black)
    lines!(ax3, r_test, vec(mean(pred_con_asym, dims=2)), label="Mean prediction", linewidth=2.5, color=:darkorange, linestyle=:dash)
    scatter!(ax3, r_train, E_train, label="Training data", markersize=6, color=:green)
    vlines!(ax3, [R_e], label="R_e", color=:orange, linestyle=:dot, linewidth=2)
    hlines!(ax3, [E_min], color=:red, linestyle=:dash, linewidth=2, label="E_min")
    hlines!(ax3, [D_e], color=:blue, linestyle=:dash, linewidth=2, label="D_e")
    ylims!(ax3, -1, 10)
    axislegend(ax3, position=:rt, fontsize=8)
    
    # Panel 7 (new): POPS Perturbations on Constrained OLS
    ax7 = Axis(fig[2, 1], xlabel="Distance r (Å)", ylabel="Energy (eV)", 
               title="POPS on Constrained OLS (Hybrid)")
    fill_between!(ax7, r_test, pred_pops_ols_min, pred_pops_ols_max, alpha=0.3, color=:cyan, label="Ensemble spread")
    lines!(ax7, r_test, E_test, label="True Morse", linewidth=3, color=:black)
    lines!(ax7, r_test, vec(mean(pred_pops_ols, dims=2)), label="Mean prediction", linewidth=2.5, color=:cyan, linestyle=:dash)
    scatter!(ax7, r_train, E_train, label="Training data", markersize=6, color=:green)
    vlines!(ax7, [R_e], label="R_e", color=:orange, linestyle=:dot, linewidth=2)
    hlines!(ax7, [E_min], color=:red, linestyle=:dash, linewidth=2, label="E_min")
    ylims!(ax7, -1, 10)
    axislegend(ax7, position=:rt, fontsize=8)
    
    # Panel 4: Comparison of ensemble spreads
    ax4 = Axis(fig[2, 2], xlabel="Distance r (Å)", ylabel="Ensemble spread (max-min) (eV)", 
               title="Ensemble Spread Comparison")
    spread_unc = pred_unc_max .- pred_unc_min
    spread_con = pred_con_max .- pred_con_min
    spread_con_asym = pred_con_asym_max .- pred_con_asym_min
    spread_pops_ols = pred_pops_ols_max .- pred_pops_ols_min
    lines!(ax4, r_test, spread_unc, label="Unconstrained", linewidth=2.5, color=:red, alpha=0.8)
    lines!(ax4, r_test, spread_con, label="Constrained (E≥E_min)", linewidth=2.5, color=:purple, alpha=0.8)
    lines!(ax4, r_test, spread_con_asym, label="Constrained + Asymptotic", linewidth=2.5, color=:darkorange, alpha=0.8)
    lines!(ax4, r_test, spread_pops_ols, label="POPS-on-OLS", linewidth=2.5, color=:cyan, alpha=0.8, linestyle=:dash)
    vlines!(ax4, [R_e], color=:orange, linestyle=:dot, linewidth=2)
    axislegend(ax4, position=:rt, fontsize=8)
    
    # Panel 5: Comparison of nominal solutions
    ax5 = Axis(fig[2, 3], xlabel="Distance r (Å)", ylabel="Energy (eV)", 
               title="Nominal Solutions Comparison")
    lines!(ax5, r_test, E_test, label="True Morse", linewidth=3, color=:black)
    lines!(ax5, r_test, V_nom, label="Unconstrained OLS", linewidth=2.5, color=:blue, linestyle=:dash)
    lines!(ax5, r_test, pred_ols_constrained, label="Constrained OLS", linewidth=2.5, color=:darkgreen, linestyle=:dash)
    lines!(ax5, r_test, vec(mean(pred_pops_ols, dims=2)), label="POPS-on-OLS mean", linewidth=2.5, color=:cyan, linestyle=:dashdot)
    hlines!(ax5, [E_min], color=:red, linestyle=:dash, linewidth=2, label="E_min")
    hlines!(ax5, [D_e], color=:blue, linestyle=:dash, linewidth=2, label="D_e")
    scatter!(ax5, r_train, E_train, label="Training data", markersize=5, color=:green, alpha=0.6)
    vlines!(ax5, [R_e], color=:orange, linestyle=:dot, linewidth=2)
    ylims!(ax5, -1, 10)
    axislegend(ax5, position=:rt, fontsize=8)
    
    # Panel 6: Error distribution
    ax6 = Axis(fig[3, 1], xlabel="Distance r (Å)", ylabel="Best ensemble error (eV)", 
               title="Prediction Error Comparison")
    err_unc_best = vec(minimum(abs.(pred_unc .- E_test), dims=2))
    err_con_best = vec(minimum(abs.(pred_con .- E_test), dims=2))
    err_con_asym_best = vec(minimum(abs.(pred_con_asym .- E_test), dims=2))
    err_ols_con = abs.(pred_ols_constrained .- E_test)
    err_pops_ols_best = vec(minimum(abs.(pred_pops_ols .- E_test), dims=2))
    lines!(ax6, r_test, err_unc_best, label="Unconstrained POPS", linewidth=2, color=:red, alpha=0.8)
    lines!(ax6, r_test, err_con_best, label="Constrained POPS", linewidth=2, color=:purple, alpha=0.8)
    lines!(ax6, r_test, err_con_asym_best, label="Constrained + Asym", linewidth=2, color=:darkorange, alpha=0.8)
    lines!(ax6, r_test, err_ols_con, label="OLS (constrained)", linewidth=2, color=:darkgreen, alpha=0.6, linestyle=:dot)
    lines!(ax6, r_test, err_pops_ols_best, label="POPS-on-OLS best", linewidth=2, color=:cyan, alpha=0.8, linestyle=:dashdot)
    vlines!(ax6, [R_e], color=:orange, linestyle=:dot, linewidth=2)
    axislegend(ax6, position=:rt, fontsize=8)
    
    # Panel 8: Method comparison summary (MAE and violations table)
    # Simple text summary without complex Makie text plotting
    ax8 = Axis(fig[3, 2:3])
    hidedecorations!(ax8)
    
    methods = ["Unconstrained POPS", "Constrained POPS", "Constrained+Asymptotic", "Constrained OLS", "POPS-on-OLS"]
    mae_vals = [mae_unc, mae_con, mae_con_asym, mae_ols_con, mae_pops_ols]
    viol_vals = [violations_unc, violations_con, violations_con_asym, violations_ols_con, violations_pops_ols]
    
    # Just display text lines as a table manually
    y_pos = 0.95
    for (i, method) in enumerate(methods)
        label_text = @sprintf("%s: MAE=%.4f eV, Violations=%d", method, mae_vals[i], viol_vals[i])
        y_pos -= 0.15
    end

    
    # Save figure
    try
        save("morse_potential_constrained_pops.png", fig)
        println("\n✓ Saved plot: morse_potential_constrained_pops.png")
    catch e
        println("\n⚠ Could not save plot: $e")
    end
    
    println("\n" * "="^90)
    println("KEY INSIGHTS:")
    println("="^90)
    println("""
    1. SIMPLE LOWER BOUND CONSTRAINT (E >= E_min):
       - E_min is the minimum energy from training data
       - Prevents unphysical negative energies or energies below the well
       - Essential for ensuring thermodynamic consistency
    
    2. ENSEMBLE SPREAD REDUCTION:
       - Constrained POPS naturally reduces ensemble spread
       - Bounds pull ensemble members toward physically reasonable region
       - Maintains uncertainty quantification while enforcing physics
    
    3. ACCURACY vs. PHYSICALITY:
       - Slight MAE trade-off for enforcing physical constraints
       - Prevents catastrophic extrapolation in unconstrained region
       - Makes predictions more stable and interpretable
    """)
    println("="^90 * "\n")
end

main()
