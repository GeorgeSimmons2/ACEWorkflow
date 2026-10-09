# ─────────────────────────────────────────────────────────────────────────────
# 01_positive_core_qp.jl — the producer of the W positive-core constrained θ.
#
# `models/W_20_4_5A_3/positive_core_constrained_parameters.csv` backs the top panel
# of `eos_repulsive_core/` (the "ACE constrained" curve).  Until now it had NO
# runnable producer: in every script that reads it —
#
#   eos_repulsive_core/eos_with_pair_hist.jl:117
#   scripts/repulsive_core/ZBL_core_ACE_correction.jl:83
#   scripts/repulsive_core/ZBL_core_ACE_correction_uncertainty.jl:79
#   scripts/uq/pops_naive_centred_on_constrained_W_20_4_5A_3.jl:67
#
# the line that would generate it is commented out and replaced by `readdlm`:
#
#   # ace_positive_core_constrained_parameters = constrained_ridge_regression(Ap, Yw, Gamma, constraint_matrix, bounds)
#
# This script restores that call.  Everything feeding it — `bulk_energy_basis`, the
# lattice grid, `constraint_matrix`, `bounds`, and `constrained_ridge_regression`
# itself — is lifted VERBATIM from ZBL_core_ACE_correction.jl:1-62, which still builds
# all of it and then throws it away.  No physics, no fit and no constraint is changed.
#
# THE CONSTRAINT.  E_bulk(a) = bulk_basis(a)·θ ≥ 0 at each of 50 lattice constants on
# LinRange(0.21, 2.18) Å — a positive repulsive core out to 2.18 Å, well below where
# the training data lives (the bottom panel of the figure is exactly that check).
#
# Differences from the original are marked [REPRO] and listed in ../README.md.
# Outputs go to a repro_ path; the published CSV is never touched.
#
# Run:  sbatch constrained_optimization/w/run_positive_core_qp.slurm
#   or  julia --project -t 8 constrained_optimization/w/01_positive_core_qp.jl
#
# Env:  OUT           output CSV (default models/W_20_4_5A_3/repro_positive_core_constrained_parameters.csv)
#       REF           published CSV to compare against ("none" to skip)
#       RHO_INTERVAL  OSQP adaptive_rho_interval; 0 = OSQP's timing-derived default,
#                     which is what the original ran.  See ../README.md.
# ─────────────────────────────────────────────────────────────────────────────

using ACEpotentials, EmpiricalPotentials, ACEWorkflow, OSQP, SparseArrays
using ACEpotentials.Models: potential_energy_basis
using AtomsBase
using StaticArrays
using StatsBase
using Unitful, ExtXYZ
using AtomsBuilder, LinearAlgebra, Statistics
using DelimitedFiles, Printf          # [REPRO] hoisted: the original uses readdlm ~30
                                      #         lines before its `using DelimitedFiles`
import AtomsCalculators: potential_energy

element     = :W
totdeg      = 20
prior_param = 4
rcut        = 5
v           = 3

RHO_INTERVAL = parse(Int, get(ENV, "RHO_INTERVAL", "25"))

result     = load_model(element, totdeg, prior_param, rcut, v)
model      = result.model
OUT        = get(ENV, "OUT", "$(result.dir)/repro_positive_core_constrained_parameters.csv")
REF        = get(ENV, "REF", "$(result.dir)/positive_core_constrained_parameters.csv")

lattice_constants = LinRange(0.21, 2.18, 50)

# ── verbatim from ZBL_core_ACE_correction.jl:17-27 ───────────────────────────
function bulk_energy_basis(element::Symbol, lattice_constants::AbstractVector{Float64}, model)
    energy_bases = []
    for a in lattice_constants
        ats = bulk(element, a=a*u"Å")
        E   = potential_energy_basis(ats, model)
        push!(energy_bases, ustrip.(E))
    end
    return energy_bases
end

# ── verbatim from ZBL_core_ACE_correction.jl:51-59, plus the pinned rho ───────
function constrained_ridge_regression(X_train, Y_train, Gamma, constraint_matrix, constraint_bounds)
    H = (X_train' * X_train .+ (1.0 / (size(X_train, 1)) .* Gamma' * Gamma))
    b = - X_train' * Y_train
    m = OSQP.Model()
    OSQP.setup!(m; P=sparse(H), q=b, A=sparse(constraint_matrix / Gamma),
                l=constraint_bounds[1], u=constraint_bounds[2],
                max_iter=500_000, check_termination=1_000, verbose=true,
                adaptive_rho_interval=RHO_INTERVAL)   # [REPRO] pinned; 0 = original
    results = OSQP.solve!(m)
    @printf("OSQP status: %s   iters: %d   obj: %.6e   pri_res: %.3e   dua_res: %.3e\n",
            string(results.info.status), results.info.iter, results.info.obj_val,
            results.info.pri_res, results.info.dua_res)
    results.info.status == :Solved ||
        error("OSQP did not solve the positive-core QP (status $(results.info.status)) — " *
              "the published θ cannot be claimed reproduced from a non-optimal point")
    return Gamma \ results.x, results
end

@printf("model %s: %d params, %d observations\n",
        result.name, length(result.lin_params), length(result.Y))
flush(stdout)

t_basis = @elapsed bulk_bases = bulk_energy_basis(element, collect(lattice_constants), model)
@printf("bulk energy bases: %d lattice constants in %.1f s\n", length(bulk_bases), t_basis)

constraint_matrix = Matrix(reduce(hcat, bulk_bases)')
bounds = (Vector{Float64}(zeros(length(bulk_bases))),
          Vector{Float64}(ones(length(bulk_bases)) .* Inf))

Gamma = result.P
Ap    = Diagonal(result.W) * result.A / Gamma
Yw    = result.W .* result.Y
@printf("Ap %d x %d, constraint matrix %d x %d, rho interval %d\n",
        size(Ap)..., size(constraint_matrix)..., RHO_INTERVAL); flush(stdout)

t_qp = @elapsed θ, info = constrained_ridge_regression(Ap, Yw, Gamma, constraint_matrix, bounds)
@printf("QP solved in %.1f s\n", t_qp)

# ── did the constraint actually bind? ────────────────────────────────────────
E_con = constraint_matrix * θ
E_unc = constraint_matrix * result.lin_params
@printf("\nbulk energy on the constrained grid (must be >= 0):\n")
@printf("  constrained   min %+.6e  at a = %.4f Å   (%d of %d negative)\n",
        minimum(E_con), lattice_constants[argmin(E_con)], count(<(0), E_con), length(E_con))
@printf("  unconstrained min %+.6e  at a = %.4f Å   (%d of %d negative)\n",
        minimum(E_unc), lattice_constants[argmin(E_unc)], count(<(0), E_unc), length(E_unc))
@printf("  → the constraint is doing work on %d lattice constants\n", count(<(0), E_unc))

writedlm(OUT, θ, ',')
@printf("\nθ → %s  (%d coefficients)\n", OUT, length(θ))

# ── compare against the published vector ─────────────────────────────────────
if REF != "none" && isfile(REF)
    θ_pub = vec(readdlm(REF, ','))
    length(θ_pub) == length(θ) ||
        error("published θ has $(length(θ_pub)) coefficients, reproduced has $(length(θ))")
    Δ = abs.(θ .- θ_pub)
    @printf("\n══ vs published %s ══\n", basename(REF))
    @printf("  max |Δ|      = %.6e\n", maximum(Δ))
    @printf("  median |Δ|   = %.6e\n", median(Δ))
    @printf("  ‖Δ‖₂         = %.6e   (‖θ_pub‖₂ = %.6e, relative %.3e)\n",
            norm(Δ), norm(θ_pub), norm(Δ)/norm(θ_pub))
    @printf("  max |Δ E_bulk| on the constrained grid = %.6e eV\n",
            maximum(abs.(constraint_matrix * (θ .- θ_pub))))
else
    println("\n(no published reference to compare against: REF=$REF)")
end
