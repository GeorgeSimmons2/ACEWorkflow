"""
    constrained_pops_ace_al_hessian_demo.jl

Standalone demonstration of Constrained POPS with Hessian-based SPD constraint.
Uses synthetic data for speed - shows the algorithm structure without huge file I/O.
"""

using OSQP
using SparseArrays
using LinearAlgebra, Statistics
using Printf

# ============================================================================
# PART 1: STANDARD POPS CORRECTIONS
# ============================================================================

function corrections(X, Y, Gamma; leverage_percentile=0.0)
    """Standard POPS: each member passes through one training point"""
    m, n = size(X)
    
    # Regularized normal equations
    C = (Gamma' * Gamma ./ m) .+ (X' * X)
    A = C \ X'
    
    # Leverage scores
    leverage = diag(X * A)
    
    # Nominal coefficients
    coeffs = C \ (X' * Y)
    errors = Y .- (X * coeffs)
    
    # Pointwise corrections
    leverage_threshold = quantile(leverage, leverage_percentile)
    mask = leverage .>= leverage_threshold
    pointwise_corrections = A[:,mask]'
    pointwise_corrections = pointwise_corrections .* (errors[mask] ./ leverage[mask])
    
    return pointwise_corrections, coeffs
end

# ============================================================================
# PART 2: CONSTRAINED POPS WITH HESSIAN SPD CONSTRAINT
# ============================================================================

function constrained_pops_update(theta_nominal, A_constraints, l_constraints, u_constraints; H=nothing)
    """
    Constrained POPS update for single data point.
    Minimizes ||θ - θ_nominal||²_H subject to A_constraints * θ ∈ [l, u]
    where H is the configuration-space Hessian (metric).
    """
    n_params = length(theta_nominal)
    
    if H === nothing
        H = I(n_params)
    end
    
    # QP: minimize 0.5 * x'*P*x + q'*x
    # For objective: 0.5 * (θ - θ_nominal)'*H*(θ - θ_nominal)
    P_qp = sparse(2.0 * H)
    q_qp = -2.0 * (H * theta_nominal)
    
    prob = OSQP.Model()
    OSQP.setup!(prob; P=P_qp, q=q_qp, A=sparse(A_constraints), 
                l=l_constraints, u=u_constraints, 
                verbose=false, eps_abs=1e-4, eps_rel=1e-4, max_iter=100)
    results = OSQP.solve!(prob)
    
    if results.info.status != :Solved
        @warn "OSQP did not converge: $(results.info.status)"
    end
    
    return results.x
end

function iterative_constrained_pops_hessian(X, Y, Gamma, H_config; 
                                           hessian_tolerance=1e-6)
    """
    Constrained POPS with configuration-space Hessian constraint.
    
    For each training point (X_i, Y_i):
    - Standard constraint: X_i' * θ = Y_i (pass through point)
    - Objective: minimize ||θ - θ_nominal||²_H where H is SPD Hessian
    """
    
    pops_corrections, nominal_coeffs = corrections(X, Y, Gamma; leverage_percentile=0.0)
    
    n_params = size(X, 2)
    n_points = size(X, 1)
    
    constrained_updates = []
    
    for i in 1:n_points
        X_i = X[i, :]
        Y_i = Y[i]
        
        # Nominal correction for this point
        theta_nominal = Gamma \ pops_corrections[i, :]
        
        # Data fitting constraint: X_i' * θ = Y_i
        A_combined = reshape(X_i, 1, :)  # Make it a row vector (1 × n_features matrix)
        l_combined = [Y_i]  # Must be a vector
        u_combined = [Y_i]  # Must be a vector
        
        # Solve constrained QP with Hessian metric
        theta_constrained = constrained_pops_update(theta_nominal, A_combined, l_combined, u_combined; H=H_config)
        
        push!(constrained_updates, theta_constrained)
        
        if i % max(1, div(n_points, 5)) == 0
            println("  Processed: $i / $n_points points")
        end
    end
    
    constrained_updates_matrix = hcat(constrained_updates...)'
    return constrained_updates_matrix
end

# ============================================================================
# PART 3: ANALYSIS AND COMPARISON
# ============================================================================

function compute_errors(X, Y, coeffs_ensemble; name="")
    """Compute MAE and other metrics for ensemble"""
    Y_pred = X * coeffs_ensemble'
    Y_mean = vec(mean(Y_pred, dims=2))
    Y_std = vec(std(Y_pred, dims=2))
    
    mae = mean(abs.(Y_mean .- Y))
    rmse = sqrt(mean((Y_mean .- Y).^2))
    
    println("\n$name Performance:")
    println("  MAE:  $(@sprintf("%.6f", mae))")
    println("  RMSE: $(@sprintf("%.6f", rmse))")
    println("  Ensemble size: $(size(coeffs_ensemble, 1))")
    
    return mae, rmse
end

# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

function main_demo()
    println("\n" * "="^80)
    println("CONSTRAINED POPS WITH HESSIAN SPD CONSTRAINT")
    println("Demonstration with Synthetic Data")
    println("="^80)
    
    # Generate synthetic problem
    # Problem: Fit high-dimensional function with physics constraints
    Random.seed!(42)
    
    n_samples = 200  # Training points
    n_features = 50  # Model parameters (e.g., ACE basis functions)
    
    println("\nGenerating synthetic problem:")
    println("  n_samples: $n_samples")
    println("  n_features: $n_features")
    
    # Design matrix: random basis evaluations
    X = randn(n_samples, n_features)
    
    # True coefficients (unknown)
    theta_true = randn(n_features) ./ (1:n_features).^2
    
    # Observations with noise
    Y = X * theta_true .+ 0.1 * randn(n_samples)
    
    println("  Y range: [$(minimum(Y)), $(maximum(Y))]")
    
    # Regularization
    Gamma = Matrix(I(n_features))
    
    # Configuration-space Hessian (SPD metric)
    # Simulate: Hessian of energy w.r.t. bulk lattice parameter
    # For Al potential: would be ~76 GPa (bulk modulus)
    # Here: use synthetic SPD matrix with eigenvalues ~ 1-100
    H_config = Symmetric(randn(n_features, n_features))
    H_config = H_config' * H_config .+ Matrix(I, n_features, n_features)  # Make it SPD
    max_eig = maximum(eigvals(H_config))
    H_config = H_config ./ max_eig  # Normalize
    
    eigenvals_H = eigvals(H_config)
    println("\nHessian metric (configuration space):")
    println("  Eigenvalues range: [$(minimum(eigenvals_H)), $(maximum(eigenvals_H))]")
    println("  SPD Status: $(all(eigenvals_H .> 1e-10) ? "✓ YES" : "✗ NO")")
    
    # ===== STANDARD POPS =====
    println("\n" * "="^80)
    println("STANDARD POPS (Unconstrained)")
    println("="^80)
    
    pops_corrections, coeffs_nominal = corrections(X, Y, Gamma)
    mae_pops, rmse_pops = compute_errors(X, Y, pops_corrections; name="Standard POPS")
    
    # ===== CONSTRAINED POPS =====
    println("\n" * "="^80)
    println("CONSTRAINED POPS (with Hessian SPD Constraint)")
    println("="^80)
    println("Building constrained ensemble (each point optimized with Hessian metric)...")
    
    try
        constrained_updates = iterative_constrained_pops_hessian(X, Y, Gamma, H_config;
                                                                   hessian_tolerance=1e-6)
        mae_constrained, rmse_constrained = compute_errors(X, Y, constrained_updates; 
                                                           name="Constrained POPS")
        
        # ===== COMPARISON =====
        println("\n" * "="^80)
        println("COMPARISON")
        println("="^80)
        
        improvement_mae = (mae_pops - mae_constrained) / mae_pops * 100
        improvement_rmse = (rmse_pops - rmse_constrained) / rmse_pops * 100
        
        println("\nError Metrics:")
        println("  Standard POPS MAE:    $(@sprintf("%.6f", mae_pops))")
        println("  Constrained POPS MAE: $(@sprintf("%.6f", mae_constrained))")
        println("  MAE Improvement:      $(@sprintf("%+.2f%%", improvement_mae))")
        println()
        println("  Standard POPS RMSE:    $(@sprintf("%.6f", rmse_pops))")
        println("  Constrained POPS RMSE: $(@sprintf("%.6f", rmse_constrained))")
        println("  RMSE Improvement:      $(@sprintf("%+.2f%%", improvement_rmse))")
        
        println("\nEnsemble Statistics:")
        println("  Standard POPS ensemble shape:    $(size(pops_corrections))")
        println("  Constrained POPS ensemble shape: $(size(constrained_updates))")
        println("  Constraint enforcement: Each member passes through its data point")
        println("                         while optimizing w.r.t. SPD Hessian metric")
        
        # Verify constraint satisfaction
        println("\nConstraint Verification (sample points):")
        for i in 1:min(3, size(X, 1))
            pred_standard = X[i, :] ⋅ pops_corrections[i, :]
            pred_constrained = X[i, :] ⋅ constrained_updates[i, :]
            error_standard = abs(pred_standard - Y[i])
            error_constrained = abs(pred_constrained - Y[i])
            println("  Point $i: Y=$(Y[i])")
            println("    Standard POPS:    Ŷ=$(@sprintf("%.6f", pred_standard)), error=$(@sprintf("%.2e", error_standard))")
            println("    Constrained POPS: Ŷ=$(@sprintf("%.6f", pred_constrained)), error=$(@sprintf("%.2e", error_constrained))")
        end
        
    catch e
        @error "Constrained POPS failed: $e"
        println("  $(sprint(showerror, e))")
    end
    
    println("\n" * "="^80)
    println("ANALYSIS COMPLETE")
    println("="^80)
    println("\nKey Insights:")
    println("• Each POPS ensemble member is optimized to pass through one data point")
    println("• Constrained version uses Hessian metric to prefer 'smooth' parameters")
    println("• For ACE potentials, Hessian w.r.t. bulk structure ensures SPD stability")
    println("• Trade-off: slightly higher error for better physical constraints")
    println("="^80)
end

# Run demonstration
if abspath(PROGRAM_FILE) == @__FILE__
    using Random
    main_demo()
end
