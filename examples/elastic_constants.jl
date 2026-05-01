# examples/elastic_constants.jl
#
# Demonstrates the single-model elastic-constant workflow:
#
#   1. Load an ACE model from models/<name>/
#   2. Relax the lattice constant (scalar Brent minimisation)
#   3. Build the strain Hessian *basis* operator H_basis (6×6×n_params, eV)
#   4. Contract H_basis with the model parameters θ → energy Hessian (eV)
#   5. Divide by unit-cell volume and convert to GPa
#   6. Report C11, C12, C44 and save outputs to models/<name>/results/
#
# H_basis is saved to disk so it can be reused – e.g. to propagate POPS
# parameter samples or a committee through to elastic-constant distributions
# without rebuilding the operator.

using ACEWorkflow          # load_model, relax_lattice_constant,
                           # elastic_hessian_basis, strain_hessian_GPa
using ACEpotentials        # ACEpotentials.Models.get_linear_parameters
using LinearAlgebra        # det
using DelimitedFiles       # writedlm
using Unitful

# ── 1. Load model ──────────────────────────────────────────────────────────────
#   Loads JSON + A/Y/P/W/lin_params from models/Al_20_5_6A_3/.
#   Pass  training_xyz = "path/to/data.xyz"  to build from scratch if absent.
result = load_model(:Al, 20, 5, 6.0, 3)
model  = result.model
θ      = result.lin_params
println("Loaded:  $(result.name)   ($(length(θ)) parameters)")

# ── 2. Relax lattice constant ──────────────────────────────────────────────────
#   Minimises energy-per-atom w.r.t. scalar `a` using Brent's method.
a_eq = relax_lattice_constant(model, :Al)
println("a₀      = $(ustrip(a_eq)) Å")

# ── 3. Strain Hessian basis ────────────────────────────────────────────────────
#   H_basis[α, β, k] = ∂²Φₖ/∂εα∂εβ |_{ε=0, a=a_eq}
#   Φₖ  = energy of the k-th ACE basis function evaluated at the relaxed cell.
#   Units: eV  (Voigt strain indices α,β ∈ {1…6}).
println("Building strain Hessian basis (ForwardDiff over $(length(θ)) params)…")
H_basis = elastic_hessian_basis(model; element=:Al, a=a_eq)   # (6, 6, n_params)
println("  H_basis shape: $(size(H_basis))")

# ── 4. Contract with model parameters → Voigt elastic tensor in GPa ────────────
#   Cαβ = (1/V₀) ∑_k θ_k  ∂²Φₖ/∂εα∂εβ
#   Convert:  1 eV/Å³ = 160.2176621 GPa
using AtomsBuilder, StaticArrays, Unitful

sys0 = AtomsBuilder.bulk(:Al; a=a_eq)
cv   = sys0.cell.cell_vectors
L    = hcat(ustrip.(cv[1]), ustrip.(cv[2]), ustrip.(cv[3]))  # 3×3, Å
V    = abs(det(L))                                           # primitive cell volume, Å³
conv = 160.2176621 / V                                       # eV → GPa

H_energy = dropdims(sum(H_basis .* reshape(θ, 1, 1, :); dims=3); dims=3)  # (6,6) eV
C        = H_energy .* conv                                                # (6,6) GPa

# ── 5. Cubic elastic constants ─────────────────────────────────────────────────
C11 = C[1, 1];   C12 = C[1, 2];   C44 = C[4, 4]
B   = (C11 + 2C12) / 3            # Voigt bulk modulus

println("\n─── Single-model elastic constants ───────────────────────────────────")
println("  a_eq   =  Å\n",  a_eq)
println("  C11    =  GPa\n", C11)
println("  C12    =  GPa\n", C12)
println("  C44    =  GPa\n", C44)
println("  B      =  GPa\n", B)
println("  Zener  = \n",     2C44 / (C11 - C12))

# Stability check (Born criteria for cubic crystal)
stable = C44 > 0 && (C11 - C12) > 0 && (C11 + 2C12) > 0
println("  Born stability: $(stable ? "✓ STABLE" : "✗ UNSTABLE")")

# ── 6. Save outputs to models/<name>/results/ ──────────────────────────────────
out_dir = joinpath(result.dir, "results")
mkpath(out_dir)

writedlm(joinpath(out_dir, "elastic_tensor.csv"),
         C, ',')
writedlm(joinpath(out_dir, "strain_hessian_basis.csv"),
         reshape(H_basis, 36, :), ',')   # stored as (36, n_params); reload with reshape(...,6,6,:)

open(joinpath(out_dir, "elastic_summary.txt"), "w") do io
    println(io, "model  = $(result.name)")
    println(io, "a_eq   = $(a_eq) Å")
    println(io, "V_cell = $(V) Å³")
    println(io, "C11    = $(C11) GPa")
    println(io, "C12    = $(C12) GPa")
    println(io, "C44    = $(C44) GPa")
    println(io, "B      = $(B) GPa")
    println(io, "Zener  = $(2C44 / (C11 - C12))")
    println(io, "stable = $(stable)")
end

println("\nSaved to: $(out_dir)/")
println("  elastic_tensor.csv          – 6×6 C matrix in GPa")
println("  strain_hessian_basis.csv    – (36 × n_params); reshape to (6,6,n_params) on load")
println("  elastic_summary.txt         – scalar summary")
