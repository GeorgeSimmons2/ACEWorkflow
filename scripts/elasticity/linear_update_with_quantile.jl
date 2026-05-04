using LinearAlgebra, Statistics, DelimitedFiles, Printf
using ACEpotentials, ForwardDiff, Unitful, ACEWorkflow

result     = load_model(:Al, 20, 5, 6.0, 3)
model      = result.model
lin_params = result.lin_params
n_params   = length(lin_params)

# ── Load and quantile-filter pops_corrections ─────────────────────────────────
println("Loading pops_corrections.csv ...")
POPS = readdlm(joinpath(result.dir, "pops_corrections.csv"), ',', Float64)
println("  size = $(size(POPS, 1)) × $(size(POPS, 2))")

η     = 0.05
lower = [quantile(POPS[:, j], η)       for j in 1:n_params]
upper = [quantile(POPS[:, j], 1.0 - η) for j in 1:n_params]
keep  = [all(lower[j] <= POPS[i, j] <= upper[j] for j in 1:n_params)
         for i in 1:size(POPS, 1)]
POPS_q = POPS[keep, :]
println("  After η=0.05 quantile filter: $(size(POPS_q, 1)) members")

norms_q  = [norm(@view POPS_q[i, :]) for i in 1:size(POPS_q, 1)]
best_idx = argmax(norms_q)
δθ_1     = POPS_q[best_idx, :]
θ_1      = lin_params + δθ_1
@printf("  Worst-case member ‖δθ‖ = %.4e  (‖δθ‖/‖θ_eq‖ = %.4e)\n",
        norms_q[best_idx], norms_q[best_idx] / norm(lin_params))

# ── Precompute b′, b″, b‴ at a_eq ────────────────────────────────────────────
a_eq = ACEWorkflow.relax_lattice_constant(model, :Al)
@printf("a_eq = %.6f Å\n", a_eq)

function lattice_basis(a_val)
    sys = ACEWorkflow.Elasticity.reference_system(:Al; a=a_val)
    ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(sys, model))
end

println("Computing b′, b″, b‴ ...")
b_prime        = ForwardDiff.derivative(lattice_basis, a_eq)
b_double_prime = ForwardDiff.derivative(
                     a_val -> ForwardDiff.derivative(lattice_basis, a_val), a_eq)
b_triple_prime = ForwardDiff.derivative(
                     a_val -> ForwardDiff.derivative(
                         a_val2 -> ForwardDiff.derivative(lattice_basis, a_val2),
                         a_val), a_eq)

K               = dot(lin_params, b_double_prime)   # θ_eq · b″
scalar_b_triple = dot(lin_params, b_triple_prime)   # θ_eq · b‴

# ── Exact lattice constant via geometry opt ───────────────────────────────────
println("Relaxing member lattice constant ...")
ACEpotentials.Models.set_linear_parameters!(model, θ_1)
a_1 = ACEWorkflow.relax_lattice_constant(model, :Al)
ACEpotentials.Models.set_linear_parameters!(model, lin_params)
δa_exact = a_1 - a_eq
@printf("a_1 = %.6f Å   δa_exact = %+.6f Å\n", a_1, δa_exact)

# ── IFT approximations ────────────────────────────────────────────────────────
δa_1   = -dot(b_prime, δθ_1) / K
δa_ift = δa_1

bpp_dot_dθ = dot(b_double_prime, δθ_1)
δa_2    = -(bpp_dot_dθ * δa_1 + 0.5 * scalar_b_triple * δa_1^2) / K
δa_ift2 = δa_1 + δa_2

println()
@printf("  δa_exact    = %+.6f Å\n", δa_exact)
@printf("  δa_ift 1st  = %+.6f Å   |err| = %.2e Å  (%.1f%%)\n",
        δa_ift,  abs(δa_ift  - δa_exact), 100*abs(δa_ift  - δa_exact)/abs(δa_exact))
@printf("  δa_ift 2nd  = %+.6f Å   |err| = %.2e Å  (%.1f%%)\n",
        δa_ift2, abs(δa_ift2 - δa_exact), 100*abs(δa_ift2 - δa_exact)/abs(δa_exact))

# ── Hessian comparison ────────────────────────────────────────────────────────
# Unit-cell volume for eV → GPa
using StaticArrays
sys0      = ACEWorkflow.Elasticity.reference_system(:Al; a=a_eq)
L0        = SMatrix{3,3,Float64}(ustrip.(ACEWorkflow.Elasticity.lattice_matrix(sys0.cell.cell_vectors)))
V         = abs(det(L0))
eV_to_GPa = 160.2176621 / V

contract_hessian(H, θ) =
    dropdims(sum(H .* reshape(θ, 1, 1, :); dims=3); dims=3) .* eV_to_GPa

# Precompute H and dH/da at a_eq
println("\nComputing H_eq and dH/da|_AD ...")
H_eq     = ACEWorkflow.elastic_hessian_basis(model; element=:Al, a=a_eq)
dH_ad_fn = ACEWorkflow.Elasticity.strain_hessian_lattice_constant_derivative_ad(model, :Al; a=a_eq)
dH_da_ad = dH_ad_fn(a_eq)

# Naive: rebuild H at exact a_1
println("Computing H_naive at a_1 (exact, slow) ...")
t_naive = @elapsed H_naive = ACEWorkflow.elastic_hessian_basis(model; element=:Al, a=a_1)
@printf("  [%.2f s]\n", t_naive)

# Approximation: linear H update using 2nd-order IFT δa
t_approx = @elapsed H_approx = H_eq .+ δa_ift2 .* dH_da_ad

# Contract with θ_1 → elastic tensors
C_naive  = contract_hessian(H_naive,  θ_1)
C_approx = contract_hessian(H_approx, θ_1)

rel_err_H = norm(H_approx - H_naive) / (norm(H_naive) + 1e-30)
ΔC_norm   = norm(C_approx - C_naive)

fmt_row(C) = @sprintf("[C11=%.4f, C12=%.4f, C44=%.4f] GPa", C[1,1], C[1,2], C[4,4])

println("\n── HESSIAN ACCURACY ─────────────────────────────────────────────────")
@printf("  H basis rel err (approx vs naive):  %.3e\n", rel_err_H)
@printf("  ‖ΔC‖ (approx vs naive):             %.3e GPa\n", ΔC_norm)
println()
println("  C (naive,  H rebuilt at a_1):        ", fmt_row(C_naive))
println("  C (approx, 2nd-order IFT + lin H):  ", fmt_row(C_approx))
println()
@printf("  Naive  time (H rebuild):             %.2f s\n", t_naive)
@printf("  Approx time (matrix add):            %.2e s\n", t_approx)
@printf("  Speedup:                             %.0fx\n", t_naive / (t_approx + 1e-30))

