using LinearAlgebra, Statistics, DelimitedFiles, Printf
using SparseArrays, StaticArrays
using Test
using ACEpotentials, ForwardDiff, Unitful, ACEWorkflow
using OSQP

element    = :Al

# ── Load model ────────────────────────────────────────────────────────────────
result     = load_model(element, 14, 4, 6, 2; dataset_name="subset_20_percent")
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
a_eq = ACEWorkflow.relax_lattice_constant(model, element)
@printf("a_eq = %.6f Å\n", a_eq)

println("Computing strain Hessian basis at a_eq ...")
C, H_eq, _ = ACEWorkflow.Elasticity.strain_hessian_GPa(model, element; a=a_eq)

# ── Lattice basis derivative for equilibrium constraint ───────────────────────
function lattice_basis(a_val)
    sys = ACEWorkflow.Elasticity.reference_system(element; a=a_val)
    ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(sys, model))
end

println("Computing b′ ...")
b_prime = ForwardDiff.derivative(lattice_basis, a_eq)

println("Computing b″ ...")
b_double_prime = ForwardDiff.derivative(a -> ForwardDiff.derivative(lattice_basis, a), a_eq)

# ── Constrained ridge regression ──────────────────────────────────────────────
function constrained_ridge_regression(X_train, Y_train, Gamma, constraint_matrix, constraint_bounds; lambda = 1.0 / size(X_train, 1))
    H = (X_train' * X_train .+ (lambda .* Gamma' * Gamma))
    b = - X_train' * Y_train
    osqp_model = OSQP.Model()
    OSQP.setup!(osqp_model; P=sparse(H), q=b, A=sparse(constraint_matrix / Gamma), l=constraint_bounds[1], u=constraint_bounds[2],
                max_iter=5_000_000_000, check_termination=1_000, verbose=true, eps_abs=1e-4, eps_rel=1e-4)
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
C44_lower    = 0.1#dot(constraint_4, lin_params)
C44_upper    = Inf#dot(constraint_4, lin_params)

# L    = SMatrix{3,3,Float64}(ustrip.(ACEWorkflow.Elasticity.lattice_matrix(ACEWorkflow.Elasticity.reference_system(element; a=a_eq).cell.cell_vectors)))
# V    = abs(det(L))
# conversion_factor = 160.21766208 / V

# constraint_11 = H_eq[1,1,:]
# C11_lower     = 116.3 / conversion_factor
# C11_upper     = 116.3 / conversion_factor

# constraint_12 = H_eq[1,2,:]
# C12_lower     = 64.8  / conversion_factor
# C12_upper     = 64.8  / conversion_factor

# constraint_44 = H_eq[4,4,:]
# C44_lower     = 30.9  / conversion_factor
# C44_upper     = 30.9  / conversion_factor

# Born stability: C11 + 2*C12 > 0
C11_plus_2C12_constraint = constraint_1 .+ 2 .* constraint_2
C11_plus_2C12_lower      = 0.1
C11_plus_2C12_upper      = Inf

C12_less_than_C11_constraint = constraint_1 .- constraint_2
C12_less_than_C11_lower      = 1.0
C12_less_than_C11_upper      = Inf

# Lattice constant constraint: b′(a_eq) · θ = 0
# Ensures the correction δθ does not shift the equilibrium lattice constant.
# b_prime must be precomputed via del_lattice_constant_design or ForwardDiff on lattice_basis.
# The RHS is -b_prime · lin_params ≈ 0 (nominal is already at equilibrium).
lattice_eq_constraint = b_prime                          # n_params-vector
lattice_eq_rhs        = -dot(b_prime, lin_params)        # ≈ 0
lattice_eq_lower      = 0.0
lattice_eq_upper      = 0.0

# Lattice curvature constraint: b″(a_eq) · θ > 0
# E(a) ≈ E(a_eq) + ½·(b″·θ)·(a-a_eq)² + ... near equilibrium, so this is
# exactly the condition that a_eq is a minimum (not a maximum/saddle) of
# E(a) — equivalent to requiring the bulk modulus B = (C11+2C12)/3 > 0.
# Bounded away from 0 (not just > 0) for the same numerical-safety reason
# as the other strict-inequality Born constraints above.
lattice_curvature_constraint = b_double_prime            # n_params-vector
lattice_curvature_lower      = 1e-9
lattice_curvature_upper      = Inf

all_constraints = vcat(constraint_1',
                       constraint_4',
                       C12_less_than_C11_constraint',
                       C11_plus_2C12_constraint',
                       lattice_eq_constraint',
                       lattice_curvature_constraint')
lower_bounds    = [C11_lower, C44_lower,
                   C12_less_than_C11_lower, C11_plus_2C12_lower,
                   lattice_eq_lower, lattice_curvature_lower]
upper_bounds    = [C11_upper, C44_upper,
                   C12_less_than_C11_upper, C11_plus_2C12_upper,
                   lattice_eq_upper, lattice_curvature_upper]
# all_constraints = vcat(constraint_4')
# lower_bounds    = [C44_lower]
# upper_bounds    = [C44_upper]

constraints = (lower_bounds, upper_bounds)

constrained_ridge_teta = constrained_ridge_regression(Ap, Yw, P, all_constraints, constraints)

using OSQP, ACEWorkflow

function constrained_pops(X_train, Y_train, Gamma, constraint_matrix, constraint_bounds; members_to_constrain=1:length(Y_train))
    constrained_pops_parameters = zeros(length(members_to_constrain), size(X_train, 2))
    H = (X_train' * X_train .+ (1.0 / (size(X_train, 1)) .* Gamma' * Gamma))
    b = - X_train' * Y_train

    model = OSQP.Model()

    for idx in 1:length(members_to_constrain)
        i = members_to_constrain[idx]
        A_full = vcat(X_train[i,:]', constraint_matrix)
        l_full = vcat([Y_train[i]], constraint_bounds[1])
        u_full = vcat([Y_train[i]], constraint_bounds[2])
        A_sparse = sparse(A_full / Gamma)

        OSQP.setup!(model; P=sparse(H), q=b, A=A_sparse, l=l_full, u=u_full,
                    max_iter=500_000, check_termination=10, verbose=false, eps_abs=5e-4, eps_rel=5e-4)

        if (mod(idx, 100) == 0)
            println(idx)
        end
        results = OSQP.solve!(model)
        constrained_pops_parameters[idx,:] = Gamma \ results.x
    end
    return constrained_pops_parameters
end

constrained_pops_delta = constrained_pops(Ap, Yw, P, all_constraints, constraints)
using Random
random_selections = rand(1:size(constrained_pops_delta, 1), 10)
sub_samples = constrained_pops_delta[random_selections, :]
pops_eig, pops_bound = hypercube(constrained_pops_delta; percentile_clipping=0.4)

pops_samples, _ = sample_hypercube(pops_eig, pops_bound, zeros(length(constrained_ridge_teta)))
# writedlm("$(result.dir)/constrained_pops_samples.csv", pops_samples, ',')
pops_samples    = [pops_samples[:,i] for i=1:size(pops_samples,2)]
accepts = [dot(b_double_prime, pops_samples[i]) for i=1:length(pops_samples)] .> 0
pops_samples = pops_samples[accepts]
writedlm("$(result.dir)/accepted_constrained_pops_samples.csv", pops_samples, ',')
ACEpotentials.Models.set_linear_parameters!(model, constrained_ridge_teta)
a_eq = ACEWorkflow.relax_lattice_constant(model, element)

# ── eV → GPa conversion ───────────────────────────────────────────────────────
sys0      = ACEWorkflow.Elasticity.reference_system(element; a=a_eq)
L0        = SMatrix{3,3,Float64}(ustrip.(ACEWorkflow.Elasticity.lattice_matrix(sys0.cell.cell_vectors)))
V         = abs(det(L0))
eV_to_GPa = 160.2176621 / V

# ── Elastic constants from constrained solution ───────────────────────────────
C11_c = dot(H_eq[1,1,:], constrained_ridge_teta) * eV_to_GPa
C12_c = dot(H_eq[1,2,:], constrained_ridge_teta) * eV_to_GPa
C44_c = dot(H_eq[4,4,:], constrained_ridge_teta) * eV_to_GPa

C11_nom = C11_lower * eV_to_GPa   # nominal (from lin_params)
C12_nom = C12_lower * eV_to_GPa
C44_nom = dot(constraint_4, lin_params) * eV_to_GPa

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

# # 2. C44 equality
# @printf("  C44 equality:          |Δ| = %.3e GPa", abs(C44_c - C44_nom))
# @test abs(C44_c - C44_nom) < 0.1
# println(abs(C44_c - C44_nom) < 0.1 ? "  ✓" : "  ✗")

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

# 6. Lattice curvature: b″·c > 0  (a_eq is a minimum, not a maximum, of E(a))
lattice_curvature = dot(b_double_prime, constrained_ridge_teta)
@printf("  Lattice b″·c (curvature): %.3e eV/Å²", lattice_curvature)
@test lattice_curvature > 0
println(lattice_curvature > 0 ? "  ✓" : "  ✗")

# ACEpotentials.save_model(model, "$(result.dir)/constrained_model.json")