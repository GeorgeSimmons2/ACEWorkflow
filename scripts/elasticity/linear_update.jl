# scripts/elasticity/linear_update.jl
#
# Benchmarks the linear-update approximation for the strain-Hessian basis
# when an ensemble member has a different equilibrium lattice constant.
#
# For one POPS ensemble member with parameter corrections δθ:
#   θ_1   = lin_params + δθ          (member's full parameters)
#   a_1   = relax_lattice_constant(model_θ1)   ← exact, via geometry opt
#   a_ift ≈ a_eq − (b′·δθ)/(θ_eq·b″)          ← 1st-order IFT, no opt
#
# Three methods to approximate H(a_1):
#   Naive:  elastic_hessian_basis at a_1              (exact, slow)
#   FD:     H_basis + (a_1 − a_eq) × dH/da|_FD       (precomputed)
#   AD:     H_basis + (a_1 − a_eq) × dH/da|_AD       (precomputed)
#
# The elastic tensor C(θ_1) = Σ_k H_k θ_1[k] × (eV→GPa) is the final QoI.

using LinearAlgebra
using StaticArrays
using Printf
using Statistics
using DelimitedFiles
using ACEpotentials
using ForwardDiff
using Unitful
using ACEWorkflow

# ── 1. Load model ─────────────────────────────────────────────────────────────
result     = load_model(:Al, 20, 5, 6.0, 3)
model      = result.model
lin_params = result.lin_params
n_params   = length(lin_params)

# ── 2. Find worst-case member after quantile filtering (η = 0.05) ────────────
# The full matrix is needed for per-column quantiles; cache result across
# repeated include() calls with the @isdefined guard.
pops_path = joinpath(result.dir, "pops_corrections.csv")
if !@isdefined(δθ_1)
    print("  Loading pops_corrections.csv ... ")
    t_load = @elapsed POPS = readdlm(pops_path, ',', Float64)
    @printf("[%.1f s]  size = %d × %d\n", t_load, size(POPS, 1), size(POPS, 2))

    print("  Applying quantile filter (η = 0.05) ... ")
    t_filter = @elapsed begin
        η      = 0.05
        lower  = [quantile(POPS[:, j], η)       for j in 1:n_params]
        upper  = [quantile(POPS[:, j], 1.0 - η) for j in 1:n_params]
        keep   = [all(lower[j] <= POPS[i, j] <= upper[j] for j in 1:n_params)
                  for i in 1:size(POPS, 1)]
        POPS_q = POPS[keep, :]
    end
    @printf("[%.1f s]  %d → %d members after filtering\n",
            t_filter, size(POPS, 1), size(POPS_q, 1))

    print("  Finding worst-case member (max ‖δθ‖ in filtered set) ... ")
    norms_q  = [norm(@view POPS_q[i, :]) for i in 1:size(POPS_q, 1)]
    best_idx = argmax(norms_q)
    δθ_1     = POPS_q[best_idx, :]
    @printf("‖δθ‖ = %.4e\n", norms_q[best_idx])
else
    println("  Using cached δθ_1 (skipping load/filter)")
end
@assert length(δθ_1) == n_params "δθ length $(length(δθ_1)) ≠ n_params $n_params"
θ_1 = lin_params + δθ_1

println("="^70)
println("Model: $(result.name)   n_params = $n_params")
@printf("‖δθ‖ / ‖θ_eq‖ = %.4e\n", norm(δθ_1) / norm(lin_params))
println("="^70)

# ── 3. Precomputation at nominal a_eq ─────────────────────────────────────────
println("\n── PRECOMPUTATION (one-time cost) ──────────────────────────────────")

print("  relax_lattice_constant (nominal) ... ")
t_relax_eq = @elapsed a_eq = ACEWorkflow.relax_lattice_constant(model, :Al)
@printf("a_eq = %.6f Å  [%.2f s]\n", a_eq, t_relax_eq)

print("  elastic_hessian_basis at a_eq ....... ")
t_H_eq = @elapsed H_eq = ACEWorkflow.elastic_hessian_basis(model; element=:Al, a=a_eq)
@printf("[%.2f s]\n", t_H_eq)

print("  dH/da via FD (build + eval) ......... ")
t_fd = @elapsed begin
    dH_fd_fn = ACEWorkflow.Elasticity.strain_hessian_lattice_constant_derivative(
                    model, :Al; a=a_eq)
    dH_da_fd = dH_fd_fn(a_eq)
end
@printf("[%.2f s]\n", t_fd)

print("  dH/da via AD (build closure) ........ ")
t_ad_build = @elapsed dH_ad_fn = ACEWorkflow.Elasticity.strain_hessian_lattice_constant_derivative_ad(
                model, :Al; a=a_eq)
@printf("[%.2f s]\n", t_ad_build)

print("  dH/da via AD (eval only) ............ ")
t_ad_eval = @elapsed dH_da_ad = dH_ad_fn(a_eq)
@printf("[%.2f s]\n", t_ad_eval)
t_ad = t_ad_build + t_ad_eval

# Energy basis derivatives for IFT lattice constant update
# b′(a) = d/da [potential_energy_basis(bulk(Al; a=a))]
print("  b′, b″, b‴ at a_eq (IFT precompute) ... ")
t_ift_pre = @elapsed begin
    function lattice_basis(a_val)
        sys = ACEWorkflow.Elasticity.reference_system(:Al; a=a_val)
        ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(sys, model))
    end
    b_prime        = ForwardDiff.derivative(lattice_basis, a_eq)
    b_double_prime = ForwardDiff.derivative(
                         a_val -> ForwardDiff.derivative(lattice_basis, a_val),
                         a_eq)
    b_triple_prime = ForwardDiff.derivative(
                         a_val -> ForwardDiff.derivative(
                             a_val2 -> ForwardDiff.derivative(lattice_basis, a_val2),
                             a_val),
                         a_eq)
    denom_ift        = dot(lin_params, b_double_prime)   # θ_eq · b″  (scalar spring constant)
    scalar_b_triple  = dot(lin_params, b_triple_prime)   # θ_eq · b‴
end
@printf("[%.2f s]\n", t_ift_pre)

# Unit-cell volume for eV → GPa conversion
sys0     = ACEWorkflow.Elasticity.reference_system(:Al; a=a_eq)
L0       = SMatrix{3,3,Float64}(
               ustrip.(ACEWorkflow.Elasticity.lattice_matrix(sys0.cell.cell_vectors)))
V        = abs(det(L0))          # Å³
eV_to_GPa = 160.2176621 / V

# ── 4. Per-member computations ────────────────────────────────────────────────
println("\n── PER-MEMBER COMPUTATION ──────────────────────────────────────────")

# 4a. Exact geometry optimization for member
print("  relax_lattice_constant (member, geometry opt) ... ")
ACEpotentials.Models.set_linear_parameters!(model, θ_1)
t_relax_1 = @elapsed a_1 = ACEWorkflow.relax_lattice_constant(model, :Al)
ACEpotentials.Models.set_linear_parameters!(model, lin_params)   # restore
@printf("a_1 = %.6f Å  [%.2f s]\n", a_1, t_relax_1)

δa_exact = a_1 - a_eq

# 4b. IFT approximations for lattice constant (no geometry opt)
print("  δa via IFT 1st order (1 dot product) ............. ")
t_ift_apply = @elapsed begin
    δa_1  = -dot(b_prime, δθ_1) / denom_ift
    δa_ift = δa_1
    a_ift  = a_eq + δa_ift
end
@printf("a_ift = %.6f Å  [%.2e s]\n", a_ift, t_ift_apply)

print("  δa via IFT 2nd order (2 dot products) ............ ")
t_ift2_apply = @elapsed begin
    bpp_dot_dθ = dot(b_double_prime, δθ_1)           # 2nd dot product
    δa_2  = -(bpp_dot_dθ * δa_1 + 0.5 * scalar_b_triple * δa_1^2) / denom_ift
    δa_ift2 = δa_1 + δa_2
    a_ift2  = a_eq + δa_ift2
end
@printf("a_ift2 = %.6f Å  [%.2e s]\n", a_ift2, t_ift2_apply)

println()
@printf("  δa_exact    = %+.6f Å\n", δa_exact)
@printf("  δa_ift 1st  = %+.6f Å   |err| = %.2e Å\n", δa_ift,  abs(a_ift  - a_1))
@printf("  δa_ift 2nd  = %+.6f Å   |err| = %.2e Å\n", δa_ift2, abs(a_ift2 - a_1))

# 4c. Naive Hessian rebuild at a_1
print("\n  elastic_hessian_basis (NAIVE rebuild at a_1) .... ")
t_naive = @elapsed H_naive = ACEWorkflow.elastic_hessian_basis(
                                  model; element=:Al, a=a_1)
@printf("[%.2f s]\n", t_naive)

# 4d. Linear updates (matrix addition — microseconds)
print("  H_basis + δa × dH/da_FD  (FD update) ........... ")
t_fd_apply = @elapsed H_fd_exact_da = H_eq .+ δa_exact .* dH_da_fd
@printf("[%.2e s]\n", t_fd_apply)

print("  H_basis + δa × dH/da_AD  (AD update) ........... ")
t_ad_apply = @elapsed H_ad_exact_da = H_eq .+ δa_exact .* dH_da_ad
@printf("[%.2e s]\n", t_ad_apply)

# Also compute with IFT δa variants
H_fd_ift_da  = H_eq .+ δa_ift  .* dH_da_fd
H_ad_ift_da  = H_eq .+ δa_ift  .* dH_da_ad
H_ad_ift2_da = H_eq .+ δa_ift2 .* dH_da_ad

# ── 5. Contract with θ_1 → elastic tensor C (GPa) ────────────────────────────
# Helper: contract 6×6×n_params tensor with θ → 6×6 matrix
function contract_hessian(H, θ)
    dropdims(sum(H .* reshape(θ, 1, 1, :); dims=3); dims=3) .* eV_to_GPa
end

C_naive     = contract_hessian(H_naive,        θ_1)
C_fd_exact  = contract_hessian(H_fd_exact_da,  θ_1)
C_ad_exact  = contract_hessian(H_ad_exact_da,  θ_1)
C_fd_ift    = contract_hessian(H_fd_ift_da,    θ_1)
C_ad_ift    = contract_hessian(H_ad_ift_da,    θ_1)
C_ad_ift2   = contract_hessian(H_ad_ift2_da,   θ_1)

# ── 6. Accuracy report ────────────────────────────────────────────────────────
println("\n── ACCURACY: Hessian basis (6×6×n_params) ──────────────────────────")
rel_err(A, B) = norm(A - B) / (norm(B) + 1e-30)

@printf("  FD update (exact δa)         vs naive:  rel err = %.3e\n",
        rel_err(H_fd_exact_da, H_naive))
@printf("  AD update (exact δa)         vs naive:  rel err = %.3e\n",
        rel_err(H_ad_exact_da, H_naive))
@printf("  AD update (IFT 1st order δa) vs naive:  rel err = %.3e\n",
        rel_err(H_ad_ift_da,   H_naive))
@printf("  AD update (IFT 2nd order δa) vs naive:  rel err = %.3e\n",
        rel_err(H_ad_ift2_da,  H_naive))
@printf("  FD update (IFT 1st order δa) vs naive:  rel err = %.3e\n",
        rel_err(H_fd_ift_da,   H_naive))

println("\n── ACCURACY: Elastic tensor C (GPa) ────────────────────────────────")
@printf("  FD update (exact δa)         vs naive:  ‖ΔC‖ = %.3e GPa\n",
        norm(C_fd_exact  - C_naive))
@printf("  AD update (exact δa)         vs naive:  ‖ΔC‖ = %.3e GPa\n",
        norm(C_ad_exact  - C_naive))
@printf("  AD update (IFT 1st order δa) vs naive:  ‖ΔC‖ = %.3e GPa\n",
        norm(C_ad_ift    - C_naive))
@printf("  AD update (IFT 2nd order δa) vs naive:  ‖ΔC‖ = %.3e GPa\n",
        norm(C_ad_ift2   - C_naive))
@printf("  FD update (IFT 1st order δa) vs naive:  ‖ΔC‖ = %.3e GPa\n",
        norm(C_fd_ift    - C_naive))

# C11 = [1,1], C12 = [1,2], C44 = [4,4]
fmt_row(C) = @sprintf("[C11=%.1f, C12=%.1f, C44=%.1f] GPa",
                      C[1,1], C[1,2], C[4,4])
println()
println("  C11, C12, C44 (naive):                   ", fmt_row(C_naive))
println("  C11, C12, C44 (AD, exact δa):            ", fmt_row(C_ad_exact))
println("  C11, C12, C44 (AD, IFT 1st order δa):   ", fmt_row(C_ad_ift))
println("  C11, C12, C44 (AD, IFT 2nd order δa):   ", fmt_row(C_ad_ift2))
println("  C11, C12, C44 (FD, exact δa):            ", fmt_row(C_fd_exact))
println("  C11, C12, C44 (FD, IFT 1st order δa):   ", fmt_row(C_fd_ift))

# ── 7. Timing summary ─────────────────────────────────────────────────────────
println("\n── TIMING SUMMARY ──────────────────────────────────────────────────")
println("  Precomputation (shared across all ensemble members):")
@printf("    relax_lattice_constant (nominal):   %.2f s\n", t_relax_eq)
@printf("    elastic_hessian_basis (H_eq):       %.2f s\n", t_H_eq)
@printf("    dH/da FD derivative:                %.2f s\n", t_fd)
@printf("    dH/da AD (build closure):           %.2f s\n", t_ad_build)
@printf("    dH/da AD (eval only):               %.2f s\n", t_ad_eval)
@printf("    dH/da AD (total):                   %.2f s\n", t_ad)
@printf("    b′, b″, b‴ for IFT:                 %.2f s\n", t_ift_pre)
println()
println("  Per-member cost:")
@printf("    Geometry opt (exact a_1):           %.2f s\n", t_relax_1)
@printf("    Naive H rebuild at a_1:             %.2f s\n", t_naive)
@printf("    IFT δa 1st order (1 dot product):   %.2e s\n", t_ift_apply)
@printf("    IFT δa 2nd order (2 dot products):  %.2e s\n", t_ift2_apply)
@printf("    Linear H update (matrix add):       %.2e s\n", t_ad_apply)
println()
t_naive_total  = t_relax_1 + t_naive
t_linear_total = t_ift2_apply + t_ad_apply
speedup = t_naive_total / (t_linear_total + 1e-30)
@printf("  Speedup per member (naive vs 2nd-order IFT + linear H): %.0fx\n", speedup)
