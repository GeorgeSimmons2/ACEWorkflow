# exact_constrain_Al_20_4_6A_4.jl
#
# Generalizes exact_elastic_constraints.jl (originally hardcoded to a single
# Al_14_4_6A_2 subset) into a loop over all five Al_20_4_6A_4 dataset sizes
# (5/10/20/50/full).
#
# For each dataset size, fits a constrained ridge regression that pins
# C11, C12, C44 to fixed literature targets (GPa) — same targets as
# exact_elastic_constraints.jl — while keeping the model at its own relaxed
# equilibrium lattice constant (b'(a_eq)·θ = 0). Saves
# models/Al_20_4_6A_4[_subset_<p>_percent]/exact_constrained_model.json.
#
# Requires Stage 1 (build_subsets_Al_20_4_6A_4.jl) to have completed for the
# 4 subset sizes; the full model already exists.
#
# Al_20_4_6A_4 has 5476 basis functions, so the OSQP constrained ridge solve
# operates on a dense 5476×5476 system per dataset size — heavy. Run via
# scripts/slurm/exact_constrain_Al_20_4_6A_4.slurm, not interactively.
#
# Usage:
#   sbatch scripts/slurm/exact_constrain_Al_20_4_6A_4.slurm

using LinearAlgebra, Statistics, DelimitedFiles, Printf
using SparseArrays, StaticArrays
using ACEpotentials, ForwardDiff, Unitful, ACEWorkflow, AtomsBuilder
using OSQP

const ELEMENT     = :Al
const TOTALDEGREE = 20
const SMOOTHNESS  = 4
const RCUT        = 6.0
const ORDER       = 4

# Literature elastic-constant targets for Al (GPa) — same as
# scripts/elasticity/exact_elastic_constraints.jl
const C11_TARGET = 116.3
const C12_TARGET = 64.8
const C44_TARGET = 30.9

const DATASET_NAMES = ["subset_5_percent", "subset_10_percent",
                        "subset_20_percent", "subset_50_percent", "full"]

function constrained_ridge_regression(X_train, Y_train, Gamma, constraint_matrix, constraint_bounds;
                                       lambda = 1.0 / size(X_train, 1))
    H = (X_train' * X_train .+ (lambda .* Gamma' * Gamma))
    b = - X_train' * Y_train
    osqp_model = OSQP.Model()
    OSQP.setup!(osqp_model; P=sparse(H), q=b, A=sparse(constraint_matrix / Gamma),
                l=constraint_bounds[1], u=constraint_bounds[2],
                max_iter=5_000_000_000, check_termination=1_000, verbose=true,
                eps_abs=1e-9, eps_rel=1e-9)
    results = OSQP.solve!(osqp_model)
    return Gamma \ results.x
end

"""
    exact_constrain_model(dataset_name)

Load the Al_20_4_6A_4 model for `dataset_name` ("subset_5_percent", ...,
"full"), fit the C11/C12/C44-targeted constrained ridge regression, and save
`exact_constrained_model.json` next to the nominal model.
"""
function exact_constrain_model(dataset_name::String)
    println("\n" * repeat('═', 78))
    println("  Al_20_4_6A_4 — $dataset_name")
    println(repeat('═', 78))

    result     = load_model(ELEMENT, TOTALDEGREE, SMOOTHNESS, RCUT, ORDER; dataset_name=dataset_name)
    model      = result.model
    A          = result.A
    Y          = result.Y
    P          = result.P
    W          = result.W
    lin_params = result.lin_params

    Ap = Diagonal(W) * A / P
    Yw = W .* Y

    println("Relaxing equilibrium lattice constant ...")
    a_eq = ACEWorkflow.relax_lattice_constant(model, ELEMENT)
    @printf("a_eq = %.6f Å\n", a_eq)

    println("Computing strain Hessian basis at a_eq ...")
    C, H_eq, _ = ACEWorkflow.Elasticity.strain_hessian_GPa(model, ELEMENT; a=a_eq)

    function lattice_basis(a_val)
        sys = ACEWorkflow.Elasticity.reference_system(ELEMENT; a=a_val)
        ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(sys, model))
    end

    println("Computing b′ ...")
    b_prime = ForwardDiff.derivative(lattice_basis, a_eq)

    L = SMatrix{3,3,Float64}(ustrip.(ACEWorkflow.Elasticity.lattice_matrix(
            ACEWorkflow.Elasticity.reference_system(ELEMENT; a=a_eq).cell.cell_vectors)))
    V = abs(det(L))
    conversion_factor = 160.21766208 / V

    constraint_11 = H_eq[1,1,:]
    constraint_12 = H_eq[1,2,:]
    constraint_44 = H_eq[4,4,:]

    # Lattice constant constraint: b′(a_eq) · θ = 0, so the correction does
    # not shift the equilibrium lattice constant away from a_eq.
    lattice_eq_constraint = b_prime

    all_constraints = vcat(constraint_11', constraint_12', constraint_44', lattice_eq_constraint')
    lower_bounds    = [C11_TARGET / conversion_factor, C12_TARGET / conversion_factor,
                        C44_TARGET / conversion_factor, 0.0]
    upper_bounds    = [C11_TARGET / conversion_factor, C12_TARGET / conversion_factor,
                        C44_TARGET / conversion_factor, 0.0]

    constrained_ridge_teta = constrained_ridge_regression(Ap, Yw, P, all_constraints,
                                                            (lower_bounds, upper_bounds))

    ACEpotentials.Models.set_linear_parameters!(model, constrained_ridge_teta)
    a_eq_new = ACEWorkflow.relax_lattice_constant(model, ELEMENT)

    sys0      = ACEWorkflow.Elasticity.reference_system(ELEMENT; a=a_eq_new)
    L0        = SMatrix{3,3,Float64}(ustrip.(ACEWorkflow.Elasticity.lattice_matrix(sys0.cell.cell_vectors)))
    V0        = abs(det(L0))
    eV_to_GPa = 160.2176621 / V0

    C11_c = dot(H_eq[1,1,:], constrained_ridge_teta) * eV_to_GPa
    C12_c = dot(H_eq[1,2,:], constrained_ridge_teta) * eV_to_GPa
    C44_c = dot(H_eq[4,4,:], constrained_ridge_teta) * eV_to_GPa

    C11_nom = dot(H_eq[1,1,:], lin_params) * eV_to_GPa
    C12_nom = dot(H_eq[1,2,:], lin_params) * eV_to_GPa
    C44_nom = dot(H_eq[4,4,:], lin_params) * eV_to_GPa

    println()
    println("── Elastic constants ───────────────────────────────────────")
    @printf("           %10s  %10s  %10s\n", "nominal", "constrained", "target")
    @printf("  C11  =   %8.3f    %8.3f    %8.3f  GPa\n", C11_nom, C11_c, C11_TARGET)
    @printf("  C12  =   %8.3f    %8.3f    %8.3f  GPa\n", C12_nom, C12_c, C12_TARGET)
    @printf("  C44  =   %8.3f    %8.3f    %8.3f  GPa\n", C44_nom, C44_c, C44_TARGET)
    @printf("  C11-C12  = %8.3f  GPa  (Born: must be > 0)\n", C11_c - C12_c)
    @printf("  C11+2C12 = %8.3f  GPa  (Born: must be > 0)\n", C11_c + 2*C12_c)

    for (label, target, got) in (("C11", C11_TARGET, C11_c), ("C12", C12_TARGET, C12_c), ("C44", C44_TARGET, C44_c))
        Δ = abs(got - target)
        Δ > 0.1 && @warn "  [$dataset_name] $label off target by $(round(Δ, sigdigits=4)) GPa"
    end

    out_json = joinpath(result.dir, "exact_constrained_model.json")
    ACEpotentials.save_model(model, out_json)
    @info "Saved $out_json"

    return (; dataset_name, dir=result.dir, a_eq=a_eq_new, C11=C11_c, C12=C12_c, C44=C44_c)
end

summary = [exact_constrain_model(name) for name in DATASET_NAMES]

println("\n" * repeat('═', 78))
println("  Summary — Al_20_4_6A_4 exact-constrained models")
println(repeat('═', 78))
@printf("  %-20s  %10s  %8s  %8s  %8s\n", "dataset", "a_eq (Å)", "C11", "C12", "C44")
for s in summary
    @printf("  %-20s  %10.5f  %8.2f  %8.2f  %8.2f\n", s.dataset_name, s.a_eq, s.C11, s.C12, s.C44)
end
