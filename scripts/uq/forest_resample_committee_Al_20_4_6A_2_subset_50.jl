# forest_resample_committee_Al_20_4_6A_2_subset_50.jl
#
# Committee = actual POPS forest members (no hypercube box, no rejection).
# Each forest member is a real, data-supported, physically-scaled model, so the
# high-dimensional box-corner pathology simply does not arise.
#
# We take the 5 highest-LEVERAGE and 5 highest-RESIDUAL training observations —
# the points that move the fit the most and the points the fit explains the
# worst — and use their POPS corrections as the 10 committee members:
#
#     leverage_i = diag(Ap C⁻¹ Apᵀ)_i        C = Ap'Ap + λ P'P   (θ̃-space)
#     residual_i = Yw_i − (Ap θ̃)_i
#     δθ_i       = P⁻¹ C⁻¹ Apᵢᵀ · (residual_i / leverage_i)      (one `corrections` row)
#     member_i   = lin_params + δθ_i
#
# These are the extreme forest members — if the committee's job is to bracket
# parameter uncertainty, they are the widest honest brackets available from the
# data itself.  Reported: leverage, |residual|, ‖δθ‖, Born constants, and full
# phonon bands (phonon_committee) with imaginary-mode counts.
#
# Outputs (models/Al_20_4_6A_2_subset_50_percent/results/):
#   forest_resample_committee_10.csv          (one member per row)
#   forest_resample_members.csv               (per-member leverage/resid/Born/min-freq)
#   forest_resample_phonon_committee_*         (band plots + freq CSVs)
#   forest_resample_bands.png                  (mean + 10 members, coloured by stability)
#
# Run:  julia --project [-t N] scripts/uq/forest_resample_committee_Al_20_4_6A_2_subset_50.jl

using LinearAlgebra, DelimitedFiles, Statistics, Printf
using Unitful, ForwardDiff, CairoMakie
using ACEpotentials, ACEWorkflow
import ACEWorkflow: phonon_committee

element      = :Al
n_lev        = 5     # highest-leverage observations
n_res        = 5     # highest-|residual| observations
N_cell_bands = 3

# ── Model ────────────────────────────────────────────────────────────────────
result     = load_model(element, 20, 4, 6, 2; dataset_name="subset_50_percent")
model      = result.model
lin_params = result.lin_params
n_params   = length(lin_params)
P          = result.P
println("Model $(result.name): $n_params parameters, $(length(result.Y)) design rows")

Ap = Diagonal(result.W) * result.A / P
Yw = result.W .* result.Y
λ  = 1.0 / size(Ap, 1)

# ── Leverage, residual, and the selected forest corrections ──────────────────
println("Factorising C = Ap'Ap + λ P'P and solving for leverage / residuals …")
C    = Symmetric(Ap' * Ap .+ λ .* (P' * P))
Cf   = cholesky(C)
AtX  = Cf \ Matrix(Ap')                       # C⁻¹ Apᵀ  (n_params × n_rows)
θ̃    = Cf \ (Ap' * Yw)                        # θ̃-space mean solution
leverage = vec(sum(Ap' .* AtX; dims=1))       # diag(Ap C⁻¹ Apᵀ)
residual = Yw .- Ap * θ̃                        # weighted residual per observation
@printf("  leverage ∈ [%.2e, %.2e]   |residual| ∈ [%.2e, %.2e]\n",
        minimum(leverage), maximum(leverage), minimum(abs.(residual)), maximum(abs.(residual)))

lev_idx = sortperm(leverage; rev=true)[1:n_lev]
res_idx = Int[]
for i in sortperm(abs.(residual); rev=true)
    i in lev_idx && continue                  # keep the 10 members distinct
    push!(res_idx, i); length(res_idx) == n_res && break
end
selected = vcat(lev_idx, res_idx)
groups   = vcat(fill("leverage", n_lev), fill("residual", n_res))

# δθ_i = P⁻¹ C⁻¹ Apᵢᵀ (residual_i / leverage_i)  — one row of `corrections`
committee = [lin_params .+ (P \ (AtX[:, i] .* (residual[i] / leverage[i]))) for i in selected]
writedlm("$(result.dir)/results/forest_resample_committee_10.csv", reduce(hcat, committee)', ',')

# ── Born diagnostics at a_eq ─────────────────────────────────────────────────
println("Relaxing mean model & building strain-Hessian basis …")
a_eq = ACEWorkflow.relax_lattice_constant(model, element)
sys0 = ACEWorkflow.Elasticity.reference_system(element; a=a_eq)
L0   = ustrip.(ACEWorkflow.Elasticity.lattice_matrix(sys0.cell.cell_vectors))
eV_to_GPa = 160.2176621 / abs(det(L0))
H_basis = elastic_hessian_basis(model; element=element, a=a_eq)
c11_0 = reshape(H_basis, 36, n_params)[1, :]
c12_0 = reshape(H_basis, 36, n_params)[7, :]
c44_0 = reshape(H_basis, 36, n_params)[22, :]
born(θ) = (dot(c11_0, θ)*eV_to_GPa, dot(c12_0, θ)*eV_to_GPa, dot(c44_0, θ)*eV_to_GPa)

println("\n── Selected forest members ─────────────────────────────────")
@printf("  %-9s %6s %10s %8s   %7s %7s %7s  Born\n",
        "group", "obs", "leverage", "‖δθ‖", "C11", "C12", "C44")
for (k, i) in enumerate(selected)
    θ = committee[k]
    C11, C12, C44 = born(θ)
    ok = (C11 - C12 > 0) && (C11 + 2C12 > 0) && (C44 > 0)
    @printf("  %-9s %6d %10.2e %8.3f   %7.1f %7.1f %7.1f  %s\n",
            groups[k], i, leverage[i], norm(θ .- lin_params), C11, C12, C44, ok ? "✓" : "✗")
end

# ── Phonon bands (mean + 10 forest members) ──────────────────────────────────
println("\n── Phonon bands of the forest-member committee ─────────────")
ACEpotentials.Models.set_linear_parameters!(model, lin_params)   # member 0 = nominal mean
x_vals, all_freqs, x_ticks, labels =
    phonon_committee(model, committee, result, element; N_cell=N_cell_bands, file_prefix="forest_resample_")

min_f  = [minimum(all_freqs[i + 1]) for i in 1:length(committee)]
n_imag = count(<(-0.05), min_f)
println("\n  member min band frequencies (THz):")
for (k, i) in enumerate(selected)
    @printf("    %-9s obs %6d: ‖δθ‖=%.3f  min ω=%+8.4f %s\n",
            groups[k], i, norm(committee[k] .- lin_params), min_f[k], min_f[k] < -0.05 ? "✗" : "✓")
end
@printf("  → %d / %d forest members phonon-UNSTABLE  (mean model min ω = %+.4f THz)\n",
        n_imag, length(committee), minimum(all_freqs[1]))

# ── Per-member CSV ───────────────────────────────────────────────────────────
open("$(result.dir)/results/forest_resample_members.csv", "w") do io
    println(io, "group,obs,leverage,abs_residual,dtheta_norm,C11_GPa,C12_GPa,C44_GPa,min_freq_THz")
    for (k, i) in enumerate(selected)
        C11, C12, C44 = born(committee[k])
        @printf(io, "%s,%d,%.6e,%.6e,%.5f,%.3f,%.3f,%.3f,%.4f\n",
                groups[k], i, leverage[i], abs(residual[i]), norm(committee[k] .- lin_params),
                C11, C12, C44, min_f[k])
    end
end

# ── Figure: mean (blue) + members (grey stable / red unstable) ───────────────
fig = Figure(size=(880, 520))
ax  = Axis(fig[1, 1]; xlabel="Wave vector", ylabel="Frequency (THz)",
           title="$(result.name) — POPS forest-member committee (5 leverage + 5 residual)",
           xticks=(x_ticks, labels), xgridvisible=false)
for k in 1:length(committee)
    col = min_f[k] < -0.05 ? RGBAf(0.80, 0.15, 0.15, 0.45) : RGBAf(0.5, 0.5, 0.5, 0.40)
    for b in 1:size(all_freqs[k+1], 1)
        lines!(ax, x_vals, all_freqs[k+1][b, :]; color=col, linewidth=1.0)
    end
end
for b in 1:size(all_freqs[1], 1)
    lines!(ax, x_vals, all_freqs[1][b, :]; color=RGBAf(0.0, 0.3, 0.7, 0.95), linewidth=2.0)
end
hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=0.8)
vlines!(ax, x_ticks; color=(:black, 0.3), linewidth=0.8)
Label(fig[0, :], "$n_imag/$(length(committee)) forest members phonon-unstable"; fontsize=13)
save("$(result.dir)/results/forest_resample_bands.png", fig)
display(fig)

println("\nSaved to $(result.dir)/results/:")
println("  forest_resample_committee_10.csv, forest_resample_members.csv")
println("  forest_resample_bands.png, forest_resample_phonon_committee_*")
