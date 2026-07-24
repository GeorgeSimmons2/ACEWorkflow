# unclipped_pops_born_check_Al_16_4_6A_3.jl
#
# Question: does an UNCLIPPED POPS hypercube on the high-data, order-3
# Al_16_4_6A_3 model (684 params, ~15k design rows) produce samples with
# negative / Born-violating elastic constants?
#
# Recomputing the strain Hessian per sample is too expensive at this size, so
# each sample is checked with the 2nd-order IFT lattice update (the
# born_stability_forest scheme):
#
#   δa  = δa₁ + δa₂
#   δa₁ = −(b′·δθ)/K
#   δa₂ = −((b″·δθ)·δa₁ + ½(θ̄·b‴)·δa₁²)/K ,   K = θ̄·b″
#   H(a_eq+δa) ≈ H(a_eq) + δa·(dH/da)
#
# so the per-sample cost is a handful of length-684 dot products.  All
# expensive objects (POPS corrections, H_basis, dH/da, b′/b″/b‴) are built
# once.  dH/da uses the finite-difference variant (2 extra basis builds) —
# the triple-nested AD version scales poorly at 684 parameters.
#
# Outputs (to models/Al_16_4_6A_3/results/):
#   unclipped_pops_born_margins.png   — histograms of C11−C12, C11+2C12, C44
#   unclipped_pops_born_check.csv     — per-sample C11, C12, C44, δa, flags
#
# Run:  julia --project scripts/uq/unclipped_pops_born_check_Al_16_4_6A_3.jl

using LinearAlgebra, DelimitedFiles, Statistics, Printf, Random
using ForwardDiff, Unitful, CairoMakie
using ACEpotentials, ACEWorkflow

Random.seed!(1234)

element   = :Al
N_samples = 10_000

# ── Load model ────────────────────────────────────────────────────────────────
result     = load_model(element, 16, 4, 6, 3)
model      = result.model
lin_params = result.lin_params
n_params   = length(lin_params)
println("Model $(result.name): $n_params parameters, $(length(result.Y)) design rows")

Ap = Diagonal(result.W) * result.A / result.P
Yw = result.W .* result.Y

# ── POPS corrections → UNCLIPPED hypercube ───────────────────────────────────
println("Computing POPS corrections …")
pops_corr = corrections(Ap, Yw, result.P)
println("  $(size(pops_corr, 1)) pointwise corrections")

hyp_eig, hyp_bound = hypercube(pops_corr)          # percentile_clipping = 0 → unclipped
println("  Hypercube: $(size(hyp_eig, 2)) active directions")

# ── One-off elastic ingredients at the mean model ────────────────────────────
println("Relaxing mean model …")
a_eq = ACEWorkflow.relax_lattice_constant(model, element)
@printf("  a_eq = %.6f Å\n", a_eq)

sys0 = ACEWorkflow.Elasticity.reference_system(element; a=a_eq)
L0   = ustrip.(ACEWorkflow.Elasticity.lattice_matrix(sys0.cell.cell_vectors))
V0   = abs(det(L0))
eV_to_GPa = 160.2176621 / V0

println("Building strain-Hessian basis (6×6×$n_params) …")
H_basis = elastic_hessian_basis(model; element=element, a=a_eq)

println("Building dH/da (central FD, 2 extra basis builds) …")
dH_da_fn = ACEWorkflow.Elasticity.strain_hessian_lattice_constant_derivative(model, element; a=a_eq)
dH_da    = dH_da_fn(a_eq)

println("Building b′, b″, b‴ …")
function lattice_basis(a_val)
    sys = ACEWorkflow.Elasticity.reference_system(element; a=a_val)
    ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(sys, model))
end
b_prime        = ForwardDiff.derivative(lattice_basis, a_eq)
b_double_prime = ForwardDiff.derivative(a -> ForwardDiff.derivative(lattice_basis, a), a_eq)
b_triple_prime = ForwardDiff.derivative(
                     a -> ForwardDiff.derivative(
                         a2 -> ForwardDiff.derivative(lattice_basis, a2), a), a_eq)
K               = dot(lin_params, b_double_prime)
scalar_b_triple = dot(lin_params, b_triple_prime)
@printf("  K = θ̄·b″ = %.5g eV/Å²\n", K)

# Rows of the flattened operators for the three cubic constants
# (column-major 6×6: entry (i,j) → flat index i + 6(j−1))
M1 = reshape(H_basis, 36, n_params)
M2 = reshape(dH_da,   36, n_params)
c11_0, c11_a = M1[1, :],  M2[1, :]     # (1,1)
c12_0, c12_a = M1[7, :],  M2[7, :]     # (1,2)
c44_0, c44_a = M1[22, :], M2[22, :]    # (4,4)

C11_mean = dot(c11_0, lin_params) * eV_to_GPa
C12_mean = dot(c12_0, lin_params) * eV_to_GPa
C44_mean = dot(c44_0, lin_params) * eV_to_GPa
@printf("Mean model:  C11 = %.2f  C12 = %.2f  C44 = %.2f  GPa\n",
        C11_mean, C12_mean, C44_mean)

# ── Sample the unclipped hypercube and check each sample ─────────────────────
println("\nSampling $N_samples members from the unclipped hypercube …")
deltas, _ = sample_hypercube(hyp_eig, hyp_bound, zeros(n_params);
                             number_of_committee_members=N_samples)

C11 = Vector{Float64}(undef, N_samples)
C12 = Vector{Float64}(undef, N_samples)
C44 = Vector{Float64}(undef, N_samples)
δa_v = Vector{Float64}(undef, N_samples)
curv_ok = Vector{Bool}(undef, N_samples)     # member's own E(a) curvature θ·b″ > 0

for s in 1:N_samples
    δθ = view(deltas, :, s)
    θ  = lin_params .+ δθ

    δa1 = -dot(b_prime, δθ) / K
    δa2 = -(dot(b_double_prime, δθ) * δa1 + 0.5 * scalar_b_triple * δa1^2) / K
    δa  = δa1 + δa2
    δa_v[s] = δa

    C11[s] = (dot(c11_0, θ) + δa * dot(c11_a, θ)) * eV_to_GPa
    C12[s] = (dot(c12_0, θ) + δa * dot(c12_a, θ)) * eV_to_GPa
    C44[s] = (dot(c44_0, θ) + δa * dot(c44_a, θ)) * eV_to_GPa
    curv_ok[s] = dot(θ, b_double_prime) > 0

    s % 1000 == 0 && print("\r  $s / $N_samples …")
end
println("\r  done.              ")

# ── Born criteria and summary ────────────────────────────────────────────────
shear  = C11 .- C12          # (i)   C11 − C12   > 0
bulkm  = C11 .+ 2 .* C12     # (ii)  C11 + 2C12  > 0
ok_i   = shear .> 0
ok_ii  = bulkm .> 0
ok_iii = C44   .> 0
stable = ok_i .& ok_ii .& ok_iii .& curv_ok

println(repeat('─', 64))
@printf("  Born-stable samples          : %5d / %d  (%.2f%%)\n",
        count(stable), N_samples, 100count(stable)/N_samples)
@printf("  violate C11−C12 > 0          : %5d  (%.2f%%)\n", count(.!ok_i),   100count(.!ok_i)/N_samples)
@printf("  violate C11+2C12 > 0         : %5d  (%.2f%%)\n", count(.!ok_ii),  100count(.!ok_ii)/N_samples)
@printf("  violate C44 > 0              : %5d  (%.2f%%)\n", count(.!ok_iii), 100count(.!ok_iii)/N_samples)
@printf("  violate θ·b″ > 0 (E(a) max)  : %5d  (%.2f%%)\n", count(.!curv_ok), 100count(.!curv_ok)/N_samples)
println(repeat('─', 64))
@printf("  C11 ∈ [%.2f, %.2f] GPa   (mean %.2f)\n", minimum(C11), maximum(C11), mean(C11))
@printf("  C12 ∈ [%.2f, %.2f] GPa   (mean %.2f)\n", minimum(C12), maximum(C12), mean(C12))
@printf("  C44 ∈ [%.2f, %.2f] GPa   (mean %.2f)\n", minimum(C44), maximum(C44), mean(C44))
@printf("  δa  ∈ [%+.4f, %+.4f] Å\n", minimum(δa_v), maximum(δa_v))
println(repeat('─', 64))

# ── Save per-sample data ─────────────────────────────────────────────────────
open("$(result.dir)/results/unclipped_pops_born_check.csv", "w") do io
    println(io, "# Unclipped POPS hypercube on $(result.name); IFT-updated Born check; N=$N_samples")
    println(io, "C11_GPa,C12_GPa,C44_GPa,da_Ang,curv_ok,stable")
    writedlm(io, hcat(C11, C12, C44, δa_v, Int.(curv_ok), Int.(stable)), ',')
end

# ── Histograms of the three Born margins ─────────────────────────────────────
blue = RGBAf(0.15, 0.40, 0.75, 0.85)
fig  = Figure(size=(1000, 340))
panels = [("C11 − C12 (GPa)", shear,  C11_mean - C12_mean,  count(.!ok_i)),
          ("C11 + 2C12 (GPa)", bulkm, C11_mean + 2C12_mean, count(.!ok_ii)),
          ("C44 (GPa)",        C44,   C44_mean,             count(.!ok_iii))]
for (p, (lab, data, meanval, nviol)) in enumerate(panels)
    ax = Axis(fig[1, p];
              xlabel = lab,
              ylabel = p == 1 ? "samples" : "",
              title  = @sprintf("%.1f%% < 0", 100nviol / N_samples))
    hist!(ax, data; bins=60, color=blue)
    vlines!(ax, [0.0];     color=:black, linestyle=:dash, linewidth=1.2)
    vlines!(ax, [meanval]; color=RGBAf(0.85, 0.45, 0.05, 0.9), linewidth=1.8)
end
Label(fig[0, :], "Unclipped POPS hypercube — Born margins ($(result.name), N=$N_samples; orange = mean model)";
      fontsize=13)
save("$(result.dir)/results/unclipped_pops_born_margins.png", fig)
display(fig)
println("Saved: unclipped_pops_born_margins.png, unclipped_pops_born_check.csv")
