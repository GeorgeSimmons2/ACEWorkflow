using LinearAlgebra, Statistics, DelimitedFiles, Printf
using SparseArrays, StaticArrays
using Test
using ACEpotentials, ForwardDiff, Unitful, ACEWorkflow
using OSQP

# ── Load model ────────────────────────────────────────────────────────────────
result     = load_model(:Al, 20, 5, 6.0, 3)
model      = result.model
A          = result.A
Y          = result.Y
P          = result.P
W          = result.W
lin_params = result.lin_params

Ap = Diagonal(W) * A / P
Yw = W .* Y

# ── Equilibrium lattice constant and Hessian basis ────────────────────────────
println("Relaxing equilibrium lattice constant ...")
a_eq = ACEWorkflow.relax_lattice_constant(model, :Al)
@printf("a_eq = %.6f Å\n", a_eq)

println("Computing strain Hessian basis at a_eq ...")
_, H_eq, _ = ACEWorkflow.Elasticity.strain_hessian_GPa(model, :Al; a=a_eq)

# ── Lattice basis derivative for equilibrium constraint ───────────────────────
function lattice_basis(a_val)
    sys = ACEWorkflow.Elasticity.reference_system(:Al; a=a_val)
    ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(sys, model))
end

println("Computing b′ ...")
b_prime = ForwardDiff.derivative(lattice_basis, a_eq)

# ── Constrained ridge regression ──────────────────────────────────────────────
function constrained_ridge_regression(X_train, Y_train, Gamma, constraint_matrix, constraint_bounds; lambda = 1.0 / size(X_train, 1))
    H = (X_train' * X_train .+ (lambda .* Gamma' * Gamma))
    b = - X_train' * Y_train
    osqp_model = OSQP.Model()
    OSQP.setup!(osqp_model; P=sparse(H), q=b, A=sparse(constraint_matrix / Gamma), l=constraint_bounds[1], u=constraint_bounds[2],
                max_iter=500_000, check_termination=1_000, verbose=true)
    results = OSQP.solve!(osqp_model)
    return Gamma \ results.x
end

# ── Constraint vectors ────────────────────────────────────────────────────────
constraint_1 = H_eq[1,1,:]
C11_lower    = dot(constraint_1, lin_params)
C11_upper    = dot(constraint_1, lin_params)

constraint_2 = H_eq[1,2,:]
C12_lower    = dot(constraint_2, lin_params)
C12_upper    = dot(constraint_2, lin_params)

constraint_4 = H_eq[4,4,:]
C44_lower    = dot(constraint_4, lin_params)
C44_upper    = dot(constraint_4, lin_params)

# Born stability: C11 + 2*C12 > 0
C11_plus_2C12_constraint = constraint_1 .+ 2 .* constraint_2
C11_plus_2C12_lower      = 2.0
C11_plus_2C12_upper      = Inf

C12_less_than_C11_constraint = constraint_1 .- constraint_2
C12_less_than_C11_lower      = 2.0
C12_less_than_C11_upper      = Inf

# Lattice constant constraint: b′(a_eq) · θ = 0
# Ensures the correction δθ does not shift the equilibrium lattice constant.
# b_prime must be precomputed via del_lattice_constant_design or ForwardDiff on lattice_basis.
# The RHS is -b_prime · lin_params ≈ 0 (nominal is already at equilibrium).
lattice_eq_constraint = b_prime                          # n_params-vector
lattice_eq_rhs        = -dot(b_prime, lin_params)        # ≈ 0
lattice_eq_lower      = 0.0
lattice_eq_upper      = 0.0

all_constraints = vcat(constraint_1',
                       constraint_4',
                       C12_less_than_C11_constraint',
                       C11_plus_2C12_constraint',
                       lattice_eq_constraint')
lower_bounds    = [C11_lower, C44_lower,
                   C12_less_than_C11_lower, C11_plus_2C12_lower,
                   lattice_eq_lower]
upper_bounds    = [C11_upper, C44_upper,
                   C12_less_than_C11_upper, C11_plus_2C12_upper,
                   lattice_eq_upper]

constraints = (lower_bounds, upper_bounds)

constrained_ridge_teta = constrained_ridge_regression(Ap, Yw, P, all_constraints, constraints; lambda=0.0)

# ── eV → GPa conversion ───────────────────────────────────────────────────────
sys0      = ACEWorkflow.Elasticity.reference_system(:Al; a=a_eq)
L0        = SMatrix{3,3,Float64}(ustrip.(ACEWorkflow.Elasticity.lattice_matrix(sys0.cell.cell_vectors)))
V         = abs(det(L0))
eV_to_GPa = 160.2176621 / V

# ── Elastic constants from constrained solution ───────────────────────────────
C11_c = dot(H_eq[1,1,:], constrained_ridge_teta) * eV_to_GPa
C12_c = dot(H_eq[1,2,:], constrained_ridge_teta) * eV_to_GPa
C44_c = dot(H_eq[4,4,:], constrained_ridge_teta) * eV_to_GPa

C11_nom = C11_lower * eV_to_GPa   # nominal (from lin_params)
C12_nom = C12_lower * eV_to_GPa
C44_nom = C44_lower * eV_to_GPa

println()
println("── Elastic constants ───────────────────────────────────────")
@printf("           %10s  %10s\n", "nominal", "constrained")
@printf("  C11  =   %8.3f    %8.3f  GPa\n", C11_nom, C11_c)
@printf("  C12  =   %8.3f    %8.3f  GPa\n", C12_nom, C12_c)
@printf("  C44  =   %8.3f    %8.3f  GPa\n", C44_nom, C44_c)
@printf("  C11-C12  = %8.3f  GPa  (Born: must be > 0)\n", C11_c - C12_c)
@printf("  C11+2C12 = %8.3f  GPa  (Born: must be > 0)\n", C11_c + 2*C12_c)

println()
println("── Constraint checks ───────────────────────────────────────")

# 1. C11 equality
@printf("  C11 equality:          |Δ| = %.3e GPa", abs(C11_c - C11_nom))
@test abs(C11_c - C11_nom) < 0.1
println(abs(C11_c - C11_nom) < 0.1 ? "  ✓" : "  ✗")

# 2. C44 equality
@printf("  C44 equality:          |Δ| = %.3e GPa", abs(C44_c - C44_nom))
@test abs(C44_c - C44_nom) < 0.1
println(abs(C44_c - C44_nom) < 0.1 ? "  ✓" : "  ✗")

# 3. Born: C11 > C12
@printf("  Born C11 - C12:        %.3f GPa", C11_c - C12_c)
@test C11_c - C12_c > 0
println(C11_c - C12_c > 0 ? "  ✓" : "  ✗")

# 4. Born: C11 + 2*C12 > 0
@printf("  Born C11 + 2*C12:      %.3f GPa", C11_c + 2*C12_c)
@test C11_c + 2*C12_c > 0
println(C11_c + 2*C12_c > 0 ? "  ✓" : "  ✗")

# 5. Lattice constant: b′·c = 0  (constraint enforces (b_prime/P)·θ = b_prime·c = 0)
lattice_residual = dot(b_prime, constrained_ridge_teta)
@printf("  Lattice b′·c:          %.3e eV/Å", lattice_residual)
@test abs(lattice_residual) < 1e-4
println(abs(lattice_residual) < 1e-4 ? "  ✓" : "  ✗")

ACEpotentials.save_model(model, "$(result.dir)/constrained_model.json")