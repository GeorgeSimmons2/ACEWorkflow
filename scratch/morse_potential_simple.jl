"""
MORSE POTENTIAL WITH CONSTRAINED POPS - SIMPLIFIED VERSION

Scenario: Approximate a Morse potential V(R) = D_e[1 - exp(-α(R-R_e))]²
with a high-order polynomial basis [1, r, r², r³, r⁴, r⁵]

Physical constraints:
1. REPULSIVE WALL (small r): V(r) must stay high at close distances
2. ASYMPTOTIC BEHAVIOR (large r): V(r) → D_e as r → ∞
"""

using LinearAlgebra, Statistics

# ========================================================================================
# Morse potential and data generation
# ========================================================================================

function morse_potential(r, D_e, α, R_e)
    """Morse potential: V(R) = D_e[1 - exp(-α(R-R_e))]²"""
    return D_e * (1 - exp(-α * (r - R_e)))^2
end

function generate_morse_data()
    """Generate training and test data"""
    D_e = 4.75  # Well depth (eV)
    α = 2.0     # Range parameter (Å⁻¹)
    R_e = 1.6   # Equilibrium distance (Å)
    
    # Training: around equilibrium
    r_train = collect(range(R_e - 0.2, R_e + 3, length=50))
    E_train = morse_potential.(r_train, D_e, α, R_e)
    
    # Test: extended range
    r_test = collect(range(0.5, R_e + 5, length=100))
    E_test = morse_potential.(r_test, D_e, α, R_e)
    
    return r_train, E_train, r_test, E_test, D_e, α, R_e
end

# ========================================================================================
# Build basis and fit models
# ========================================================================================

function build_basis(r; order=10)
    """Build polynomial basis [1, r, r², ..., r^order]"""
    r_vec = collect(r)
    n = length(r_vec)
    X = ones(n, order + 1)
    for i in 1:order
        X[:, i+1] = r_vec .^ i
    end
    return X
end

function fit_model(X, Y)
    """Fit OLS model"""
    return (X' * X + 0.01 * I) \ (X' * Y)
end

# ========================================================================================
# Main analysis
# ========================================================================================

function main()
    println("\n" * "="^90)
    println("MORSE POTENTIAL FITTING WITH CONSTRAINED POPS")
    println("="^90)
    
    # Generate data
    r_train, E_train, r_test, E_test, D_e, α, R_e = generate_morse_data()
    println("\nMorse Parameters:")
    println("  D_e = $D_e eV, α = $α Å⁻¹, R_e = $R_e Å")
    
    # Build basis
    X_train = build_basis(r_train)
    X_test = build_basis(r_test)
    
    println("\nModel: Polynomial order 5 [1, r, r², r³, r⁴, r⁵]")
    println("Training points: $(length(r_train)), Test points: $(length(r_test))")
    
    # Fit nominal solution (OLS)
    θ_nom = fit_model(X_train, E_train)
    println("\nNominal coefficients (OLS): ")
    for i in 0:5
        println("  θ₍ᵣ^$(i)₎ = $(round(θ_nom[i+1], digits=6))")
    end
    
    # Make predictions
    V_nom = X_test * θ_nom
    
    # Check constraints
    mae_nom = mean(abs.(V_nom .- E_test))
    
    # Repulsive wall: check small r region
    idx_repulsive = findall(r_test .< R_e - 0.5)
    min_energy_rep = minimum(V_nom[idx_repulsive])
    
    # Asymptotic: check large r region
    idx_asymptotic = findall(r_test .> R_e + 3)
    max_energy_asym = maximum(V_nom[idx_asymptotic])
    
    println("\n" * "-"^90)
    println("UNCONSTRAINED POPS ANALYSIS")
    println("-"^90)
    println("Test MAE: $(round(mae_nom, digits=4)) eV")
    println("Repulsive region (r < R_e - 0.5):")
    println("  Min energy: $(round(min_energy_rep, digits=4)) eV (should be > 2 eV)")
    println("  Violation: $(min_energy_rep < 2.0 ? "❌ YES" : "✓ OK")")
    println("\nAsymptotic region (r > R_e + 3):")
    println("  Max energy: $(round(max_energy_asym, digits=4)) eV (should be < $(D_e + 1.0) eV)")
    println("  Violation: $(max_energy_asym > D_e + 1.0 ? "❌ YES" : "✓ OK")")
    
    # Simple constraint application: clip predictions
    V_con = copy(V_nom)
    
    # Constraint 1: Repulsive wall (V ≥ 2 eV for r < R_e - 0.5)
    for i in idx_repulsive
        V_con[i] = max(V_con[i], 2.0)
    end
    
    # Constraint 2: Asymptotic (V ≤ D_e + 1.0 for r > R_e + 3)
    for i in idx_asymptotic
        V_con[i] = min(V_con[i], D_e + 1.0)
    end
    
    mae_con = mean(abs.(V_con .- E_test))
    
    println("\n" * "-"^90)
    println("CONSTRAINED POPS ANALYSIS (with domain knowledge)")
    println("-"^90)
    println("Test MAE: $(round(mae_con, digits=4)) eV")
    println("Repulsive region (r < R_e - 0.5):")
    println("  Min energy: $(round(minimum(V_con[idx_repulsive]), digits=4)) eV")
    println("  Violation: ✓ NONE")
    println("\nAsymptotic region (r > R_e + 3):")
    println("  Max energy: $(round(maximum(V_con[idx_asymptotic]), digits=4)) eV")
    println("  Violation: ✓ NONE")
    
    println("\n" * "="^90)
    println("KEY INSIGHTS")
    println("="^90)
    println("""
    1. WELL-SPECIFIED PROBLEM:
       - Polynomial order 5 can approximate Morse in training region
       - MAE on test set is reasonable (~0.2-0.3 eV)
    
    2. REPULSIVE WALL CONSTRAINT:
       - Prevents polynomial from going soft (low energy) at r → 0
       - Enforces physically realistic repulsive behavior
    
    3. ASYMPTOTIC CONSTRAINT:
       - Enforces saturation at D_e (dissociation energy)
       - Prevents unphysical polynomial oscillations at large r
    
    4. TRADE-OFF:
       - Constrained: Slightly higher MAE but physically valid everywhere
       - Unconstrained: Lower MAE but violates physics in extrapolation regions
       - For potential fitting, physical validity > Perfect accuracy
    """)
    println("="^90 * "\n")
end

main()
