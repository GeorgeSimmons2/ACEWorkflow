using LinearAlgebra, Statistics, DelimitedFiles, Printf
using SparseArrays, StaticArrays
using Test
using ACEpotentials, ForwardDiff, Unitful, ACEWorkflow, AtomsBuilder, StaticArrays
using OSQP

# ── Load model ────────────────────────────────────────────────────────────────
result     = load_model(:Al, 20, 4, 6, 4)
model      = result.model
A          = result.A
Y          = result.Y
P          = result.P
W          = result.W
lin_params = result.lin_params
element    = :Al

Ap = Diagonal(W) * A / P
Yw = W .* Y

# ── Equilibrium lattice constant and Hessian basis ────────────────────────────
println("Relaxing equilibrium lattice constant ...")
a_eq = ACEWorkflow.relax_lattice_constant(model, :Al)
@printf("a_eq = %.6f Å\n", a_eq)

println("Computing strain Hessian basis at a_eq ...")
C, H_eq, _ = ACEWorkflow.Elasticity.strain_hessian_GPa(model, :Al; a=a_eq)

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
                max_iter=5_000_000_000, check_termination=1_000, verbose=true, eps_abs=1e-9, eps_rel=1e-9)
    results = OSQP.solve!(osqp_model)
    return Gamma \ results.x
end

L    = SMatrix{3,3,Float64}(ustrip.(ACEWorkflow.Elasticity.lattice_matrix(ACEWorkflow.Elasticity.reference_system(element; a=a_eq).cell.cell_vectors)))
V    = abs(det(L))
conversion_factor = 160.21766208 / V

# ── Constraint vectors ────────────────────────────────────────────────────────
constraint_11 = H_eq[1,1,:]
C11_lower     = 116.3 / conversion_factor
C11_upper     = 116.3 / conversion_factor

constraint_12 = H_eq[1,2,:]
C12_lower     = 64.8  / conversion_factor
C12_upper     = 64.8  / conversion_factor

constraint_44 = H_eq[4,4,:]
C44_lower     = 30.9  / conversion_factor
C44_upper     = 30.9  / conversion_factor

# Lattice constant constraint: b′(a_eq) · θ = 0
# Ensures the correction δθ does not shift the equilibrium lattice constant.
# b_prime must be precomputed via del_lattice_constant_design or ForwardDiff on lattice_basis.
# The RHS is -b_prime · lin_params ≈ 0 (nominal is already at equilibrium).
lattice_eq_constraint = b_prime                          # n_params-vector
lattice_eq_rhs        = -dot(b_prime, lin_params)        # ≈ 0
lattice_eq_lower      = 0.0
lattice_eq_upper      = 0.0

all_constraints = vcat(constraint_11',
                       constraint_12',
                       constraint_44',
                       lattice_eq_constraint')
lower_bounds    = [C11_lower, C12_lower, C44_lower,
                   lattice_eq_lower]
upper_bounds    = [C11_upper, C12_upper, C44_upper,
                   lattice_eq_upper]

constraints = (lower_bounds, upper_bounds)

constrained_ridge_teta = constrained_ridge_regression(Ap, Yw, P, all_constraints, constraints)

ACEpotentials.Models.set_linear_parameters!(model, constrained_ridge_teta)
a_eq = ACEWorkflow.relax_lattice_constant(model, :Al)

# ── eV → GPa conversion ───────────────────────────────────────────────────────
sys0      = ACEWorkflow.Elasticity.reference_system(:Al; a=a_eq)
L0        = SMatrix{3,3,Float64}(ustrip.(ACEWorkflow.Elasticity.lattice_matrix(sys0.cell.cell_vectors)))
V         = abs(det(L0))
eV_to_GPa = 160.2176621 / V

# ── Elastic constants from constrained solution ───────────────────────────────
C11_c = dot(H_eq[1,1,:], constrained_ridge_teta) * eV_to_GPa
C12_c = dot(H_eq[1,2,:], constrained_ridge_teta) * eV_to_GPa
C44_c = dot(H_eq[4,4,:], constrained_ridge_teta) * eV_to_GPa

C11_nom = dot(H_eq[1,1,:], lin_params) * eV_to_GPa
C12_nom = dot(H_eq[1,2,:], lin_params) * eV_to_GPa
C44_nom = dot(H_eq[4,4,:], lin_params) * eV_to_GPa

println()
println("── Elastic constants ───────────────────────────────────────")
@printf("           %10s  %10s\n", "nominal", "constrained")
@printf("  C11  =   %8.3f    %8.3f  GPa\n", C11_nom, C11_c)
@printf("  C12  =   %8.3f    %8.3f  GPa\n", C12_nom, C12_c)
@printf("  C44  =   %8.3f    %8.3f  GPa\n", C44_nom, C44_c)
@printf("  C11-C12  = %8.3f  GPa  (Born: must be > 0)\n", C11_c - C12_c)
@printf("  C11+2C12 = %8.3f  GPa  (Born: must be > 0)\n", C11_c + 2*C12_c)

ACEpotentials.save_model(model, "$(result.dir)/exact_constrained_model.json")