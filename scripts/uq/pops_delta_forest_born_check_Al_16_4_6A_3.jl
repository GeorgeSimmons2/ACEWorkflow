# pops_delta_forest_born_check_Al_16_4_6A_3.jl
#
# Born-stability of the POPS DELTA FOREST for Al_16_4_6A_3: each member is
# θ = lin_params + δθ_i where δθ_i is one raw pointwise correction — no
# hypercube, no sampling.  Companion to unclipped_pops_born_check_Al_16_4_6A_3.jl
# (which checks 10k draws from the unclipped hypercube built over this same
# forest); comparing the two separates "the corrections are unstable" from
# "the box construction manufactures unstable combinations".
#
# Same 2nd-order IFT lattice update as the hypercube check; per-member cost is
# a handful of length-684 dot products.
#
# Outputs (to models/Al_16_4_6A_3/results/):
#   pops_delta_forest_born_margins.png
#   pops_delta_forest_born_check.csv
#
# Run:  julia --project scripts/uq/pops_delta_forest_born_check_Al_16_4_6A_3.jl

using LinearAlgebra, DelimitedFiles, Statistics, Printf
using ForwardDiff, Unitful, CairoMakie
using ACEpotentials, ACEWorkflow

element = :Al

# ── Load model ────────────────────────────────────────────────────────────────
result     = load_model(element, 20, 4, 6, 2; dataset_name="subset_50_percent")
model      = result.model
lin_params = result.lin_params
n_params   = length(lin_params)
println("Model $(result.name): $n_params parameters, $(length(result.Y)) design rows")

Ap = Diagonal(result.W) * result.A / result.P
Yw = result.W .* result.Y

# ── POPS delta forest ────────────────────────────────────────────────────────
println("Computing POPS corrections (delta forest) …")
pops_corr = corrections(Ap, Yw, result.P; leverage_percentile=0.0)       # rows = δθ_i
N_forest  = size(pops_corr, 1)
println("  $N_forest forest members")

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

println("Building dH/da (central FD) …")
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

M1 = reshape(H_basis, 36, n_params)
M2 = reshape(dH_da,   36, n_params)
c11_0, c11_a = M1[1, :],  M2[1, :]
c12_0, c12_a = M1[7, :],  M2[7, :]
c44_0, c44_a = M1[22, :], M2[22, :]

C11_mean = dot(c11_0, lin_params) * eV_to_GPa
C12_mean = dot(c12_0, lin_params) * eV_to_GPa
C44_mean = dot(c44_0, lin_params) * eV_to_GPa
@printf("Mean model:  C11 = %.2f  C12 = %.2f  C44 = %.2f  GPa\n",
        C11_mean, C12_mean, C44_mean)

# ── Check every forest member ────────────────────────────────────────────────
println("\nChecking $N_forest forest members …")
C11 = Vector{Float64}(undef, N_forest)
C12 = Vector{Float64}(undef, N_forest)
C44 = Vector{Float64}(undef, N_forest)
δa_v    = Vector{Float64}(undef, N_forest)
δθ_norm = Vector{Float64}(undef, N_forest)
curv_ok = Vector{Bool}(undef, N_forest)

for s in 1:N_forest
    δθ = vec(pops_corr[s, :])
    θ  = lin_params .+ δθ
    δθ_norm[s] = norm(δθ)

    δa1 = -dot(b_prime, δθ) / K
    δa2 = -(dot(b_double_prime, δθ) * δa1 + 0.5 * scalar_b_triple * δa1^2) / K
    δa  = δa1 + δa2
    δa_v[s] = δa

    C11[s] = (dot(c11_0, θ) + δa * dot(c11_a, θ)) * eV_to_GPa
    C12[s] = (dot(c12_0, θ) + δa * dot(c12_a, θ)) * eV_to_GPa
    C44[s] = (dot(c44_0, θ) + δa * dot(c44_a, θ)) * eV_to_GPa
    curv_ok[s] = dot(θ, b_double_prime) > 0

    s % 1000 == 0 && print("\r  $s / $N_forest …")
end
println("\r  done.              ")

# ── Born criteria and summary ────────────────────────────────────────────────
shear  = C11 .- C12
bulkm  = C11 .+ 2 .* C12
ok_i   = shear .> 0
ok_ii  = bulkm .> 0
ok_iii = C44   .> 0
stable = ok_i .& ok_ii .& ok_iii .& curv_ok

println(repeat('─', 64))
@printf("  Born-stable forest members   : %5d / %d  (%.2f%%)\n",
        count(stable), N_forest, 100count(stable)/N_forest)
@printf("  violate C11−C12 > 0          : %5d  (%.2f%%)\n", count(.!ok_i),   100count(.!ok_i)/N_forest)
@printf("  violate C11+2C12 > 0         : %5d  (%.2f%%)\n", count(.!ok_ii),  100count(.!ok_ii)/N_forest)
@printf("  violate C44 > 0              : %5d  (%.2f%%)\n", count(.!ok_iii), 100count(.!ok_iii)/N_forest)
@printf("  violate θ·b″ > 0 (E(a) max)  : %5d  (%.2f%%)\n", count(.!curv_ok), 100count(.!curv_ok)/N_forest)
println(repeat('─', 64))
@printf("  C11 ∈ [%.2f, %.2f] GPa   (mean %.2f)\n", minimum(C11), maximum(C11), mean(C11))
@printf("  C12 ∈ [%.2f, %.2f] GPa   (mean %.2f)\n", minimum(C12), maximum(C12), mean(C12))
@printf("  C44 ∈ [%.2f, %.2f] GPa   (mean %.2f)\n", minimum(C44), maximum(C44), mean(C44))
@printf("  δa  ∈ [%+.4f, %+.4f] Å\n", minimum(δa_v), maximum(δa_v))
@printf("  ‖δθ‖ ∈ [%.3g, %.3g]  (median %.3g)\n",
        minimum(δθ_norm), maximum(δθ_norm), median(δθ_norm))
println(repeat('─', 64))

# ── Save per-member data ─────────────────────────────────────────────────────
open("$(result.dir)/results/pops_delta_forest_born_check.csv", "w") do io
    println(io, "# POPS delta forest on $(result.name); IFT-updated Born check; N=$N_forest")
    println(io, "C11_GPa,C12_GPa,C44_GPa,da_Ang,dtheta_norm,curv_ok,stable")
    writedlm(io, hcat(C11, C12, C44, δa_v, δθ_norm, Int.(curv_ok), Int.(stable)), ',')
end

# ── Histograms (same layout/colors as the hypercube figure) ──────────────────
blue = RGBAf(0.15, 0.40, 0.75, 0.85)
fig  = Figure(size=(1000, 340))
panels = [("C11 − C12 (GPa)", shear,  C11_mean - C12_mean,  count(.!ok_i)),
          ("C11 + 2C12 (GPa)", bulkm, C11_mean + 2C12_mean, count(.!ok_ii)),
          ("C44 (GPa)",        C44,   C44_mean,             count(.!ok_iii))]
for (p, (lab, data, meanval, nviol)) in enumerate(panels)
    ax = Axis(fig[1, p];
              xlabel = lab,
              ylabel = p == 1 ? "forest members" : "",
              title  = @sprintf("%.2f%% < 0", 100nviol / N_forest))
    hist!(ax, data; bins=60, color=blue)
    vlines!(ax, [0.0];     color=:black, linestyle=:dash, linewidth=1.2)
    vlines!(ax, [meanval]; color=RGBAf(0.85, 0.45, 0.05, 0.9), linewidth=1.8)
end
Label(fig[0, :], "POPS delta forest — Born margins ($(result.name), N=$N_forest; orange = mean model)";
      fontsize=13)
save("$(result.dir)/results/pops_delta_forest_born_margins.png", fig)
display(fig)
println("Saved: pops_delta_forest_born_margins.png, pops_delta_forest_born_check.csv")
