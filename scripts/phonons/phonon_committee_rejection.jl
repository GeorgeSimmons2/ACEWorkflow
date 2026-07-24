# phonon_committee_rejection.jl
#
# Phonon band structures for the rejection-sampled constrained-POPS committee
# produced by scripts/elasticity/elastic_constraints_rejection.jl.
#
# Uses the packaged ACEWorkflow.Phonons machinery (phonon_committee) instead of
# the local copies living in the older phonon scripts.
#
# Outputs (to models/Al_14_4_6A_2_subset_20_percent/results/):
#   rejection_sampled_phonon_committee_THz_4x4x4.png
#   rejection_sampled_phonon_committee_eV_4x4x4.png
#   rejection_sampled_phonon_committee_x_vals_4x4x4.csv
#   rejection_sampled_phonon_committee_freqs_THz_4x4x4.csv
#     (rows 1:3 = constrained mean, rows 3i+1:3i+3 = member i)
#   rejection_sampled_gruneisen_comparison.png / .csv
#     (γ̄(T) for constrained mean + committee vs unconstrained mean + naive POPS)
#
# Run:  julia --project -t auto scripts/phonons/phonon_committee_rejection.jl

using LinearAlgebra, DelimitedFiles, Statistics, Printf
using AtomsBuilder, Unitful, CairoMakie
using ACEpotentials, ACEWorkflow

element = :Al

# ── Model + constrained mean ─────────────────────────────────────────────────
result = load_model(element, 14, 4, 6, 2; dataset_name="subset_20_percent")
model  = result.model

# The committee's mean is the constrained ridge solution, not lin_params.
# Use the in-session solution if elastic_constraints_rejection.jl was just run,
# otherwise fall back to the saved constrained model.
constrained_params = if @isdefined(constrained_ridge_teta)
    constrained_ridge_teta
else
    con_model, _ = ACEpotentials.load_model("$(result.dir)/constrained_model.json")
    vec(vcat(con_model.ps[1], con_model.ps[2]))
end

# ── Committee (one member per row of the CSV) ────────────────────────────────
committee_mat = readdlm("$(result.dir)/rejection_sampled_constrained_pops_samples.csv", ',')
committee     = [committee_mat[i, :] for i in 1:size(committee_mat, 1)]
println("Loaded $(length(committee)) rejection-sampled committee members")

# ── Phonon bands: constrained mean + committee ───────────────────────────────
# phonon_committee plots whatever parameters `model` currently holds as the
# mean, so install the constrained solution first.  It restores
# result.lin_params when it finishes, hence the second set_linear_parameters!.
ACEpotentials.Models.set_linear_parameters!(model, constrained_params)

x_vals, all_freqs, x_ticks, labels = phonon_committee(
    model, committee, result, element;
    N_per_seg   = 30,
    N_cell      = 4,          # 4×4×4 conventional cells ≈ 16.2 Å ≥ 2 × rcut
    file_prefix = "rejection_sampled_",
)

ACEpotentials.Models.set_linear_parameters!(model, constrained_params)

# ── Dynamical-stability summary ──────────────────────────────────────────────
n_imag = [count(f .< 0) for f in all_freqs]
println("\nImaginary modes — constrained mean: ", n_imag[1])
println("Members with no imaginary modes anywhere on the path: ",
        count(==(0), n_imag[2:end]), " / ", length(committee))

# ═════════════════════════════════════════════════════════════════════════════
#  Grüneisen parameter γ̄(T)
#
#  γ_qs = −(V/ω_qs) ∂ω_qs/∂V, central finite difference over ±δ lattice
#  scaling (±3δ in volume), Cv-weighted BZ average:
#      γ̄(T) = Σ_qs Cv(ω_qs, T) γ_qs / Σ_qs Cv(ω_qs, T)
#
#  Deliberate differences from the older path-based version
#  (gruneisen_phonon_bands_ace.jl):
#    - Uniform shifted q-mesh over the primitive BZ (equal weights), not the
#      high-symmetry path: the path over-weights symmetry lines and double
#      counts Γ, biasing the thermodynamic average.  The half-step shift also
#      avoids q = Γ entirely, so no ω → 0 acoustic zeros enter the average.
#    - Frequencies at each q are sorted ascending in every volume, so the
#      finite difference always pairs mode s with mode s.  (Branch tracking
#      seeded at the degenerate Γ point is arbitrary and can pair different
#      physical branches across volumes.)
# ═════════════════════════════════════════════════════════════════════════════

const ħ  = 1.054571817e-34   # J·s
const kB = 1.380649e-23      # J/K

"""
    bz_mesh_qcart(L, n) -> Vector{Vector{Float64}}

Uniform n×n×n Monkhorst–Pack-style mesh over the primitive BZ of lattice `L`
(columns = lattice vectors, Å), shifted by half a step so Γ is excluded.
Returns Cartesian q-vectors (Å⁻¹); all points carry equal weight.
"""
function bz_mesh_qcart(L, n)
    B  = 2π * inv(transpose(L))
    qs = Vector{Vector{Float64}}()
    for i in 0:n-1, j in 0:n-1, k in 0:n-1
        f = ([i, j, k] .+ 0.5) ./ n .- 0.5
        push!(qs, B * f)
    end
    return qs
end

"""
    mesh_frequencies(model, element, a; N_cell=4, n_mesh=8) -> 3Np × Nq Matrix [THz]

Phonon frequencies on the shifted BZ mesh at lattice constant `a`, sorted
ascending at each q (consistent mode pairing across volumes).
"""
function mesh_frequencies(model, element, a; N_cell=4, n_mesh=8)
    sys_prim  = bulk(element; a=a*u"Å")
    sys_super = bulk(element; a=a*u"Å", cubic=true) * (N_cell, N_cell, N_cell)
    fc = precompute_force_constants(sys_prim, sys_super, model)
    qs = bz_mesh_qcart(fc.L, n_mesh)
    freqs = Matrix{Float64}(undef, 3*length(sys_prim), length(qs))
    for (iq, q) in enumerate(qs)
        ω2 = eigvals(Hermitian(dynamical_matrix_from_fc(fc, q)))   # ascending
        freqs[:, iq] = sign.(ω2) .* sqrt.(abs.(ω2)) .* FREQ_THz
    end
    return freqs
end

"""
    gruneisen_gamma_T(model, element, θ; δ=0.01, N_cell=4, n_mesh=8,
                      T_range=50:50:800, freq_cutoff_THz=0.05)

γ̄(T) for parameter vector `θ` at its own relaxed lattice constant.
Returns `(γT, a_eq, n_dropped)` where `n_dropped` counts modes excluded by the
`ω₀ > freq_cutoff_THz` mask (soft/imaginary modes; 0 for a stable member since
the shifted mesh contains no acoustic zeros).
"""
function gruneisen_gamma_T(model, element, θ; δ=0.01, N_cell=4, n_mesh=8,
                           T_range=50:50:800, freq_cutoff_THz=0.05)
    ACEpotentials.Models.set_linear_parameters!(model, θ)
    a_eq = ACEWorkflow.relax_lattice_constant(model, element)

    to_rad_s = 2π * 1e12
    ωm = mesh_frequencies(model, element, (1-δ)*a_eq; N_cell, n_mesh) .* to_rad_s
    ω0 = mesh_frequencies(model, element,        a_eq; N_cell, n_mesh) .* to_rad_s
    ωp = mesh_frequencies(model, element, (1+δ)*a_eq; N_cell, n_mesh) .* to_rad_s

    V0 = a_eq^3;  Vm = ((1-δ)*a_eq)^3;  Vp = ((1+δ)*a_eq)^3
    γ  = @. -(V0 / ω0) * (ωp - ωm) / (Vp - Vm)

    valid     = ω0 .> freq_cutoff_THz * to_rad_s
    n_dropped = length(ω0) - count(valid)

    γT = map(T_range) do T
        x  = ħ .* ω0[valid] ./ (kB * T)
        Cv = @. kB * x^2 * exp(x) / (exp(x) - 1)^2
        sum(Cv .* γ[valid]) / sum(Cv)
    end
    return γT, a_eq, n_dropped
end

"""
    committee_gruneisen(model, members, element; label="", kwargs...)

γ̄(T) for every member; failed members (unstable relaxation etc.) give a NaN
column and are reported, not fatal — naive POPS members can be unphysical.
Returns a `nT × N` matrix.
"""
function committee_gruneisen(model, members, element; label="", T_range=50:50:800, kwargs...)
    γ_mat = fill(NaN, length(T_range), length(members))
    for (i, θ) in enumerate(members)
        try
            γT, a_i, n_drop = gruneisen_gamma_T(model, element, θ; T_range, kwargs...)
            γ_mat[:, i] = γT
            @printf("  [%s member %d] a = %.5f Å, γ(300K-ish) = %.3f%s\n",
                    label, i, a_i, γT[min(6, length(γT))],
                    n_drop > 0 ? "  ($n_drop soft modes dropped)" : "")
        catch err
            println("  [$label member $i] FAILED: ", sprint(showerror, err))
        end
    end
    return γ_mat
end

# ── Compute all four cases ───────────────────────────────────────────────────
T_range = 50:50:800
δ_grun  = 0.01
n_mesh  = 8
N_cell_grun = 4

println("\n=== Grüneisen: constrained mean ===")
γ_con_mean, a_con, _ = gruneisen_gamma_T(model, element, constrained_params;
                                         δ=δ_grun, N_cell=N_cell_grun, n_mesh, T_range)

println("\n=== Grüneisen: unconstrained mean (lin_params) ===")
γ_unc_mean, a_unc, _ = gruneisen_gamma_T(model, element, result.lin_params;
                                         δ=δ_grun, N_cell=N_cell_grun, n_mesh, T_range)

println("\n=== Grüneisen: rejection-sampled constrained committee ===")
γ_con_committee = committee_gruneisen(model, committee, element;
                                      label="constrained", T_range,
                                      δ=δ_grun, N_cell=N_cell_grun, n_mesh)

# Naive unconstrained POPS committee: standard corrections → unclipped
# hypercube → sample_hypercube about lin_params, no physical constraints.
println("\n=== Grüneisen: naive unconstrained POPS committee ===")
Ap = Diagonal(result.W) * result.A / result.P
Yw = result.W .* result.Y
naive_corr             = corrections(Ap, Yw, result.P)
naive_eig, naive_bound = hypercube(naive_corr)
naive_mat, _ = sample_hypercube(naive_eig, naive_bound, result.lin_params;
                                number_of_committee_members=length(committee))
naive_committee = [naive_mat[:, i] for i in 1:size(naive_mat, 2)]
γ_naive_committee = committee_gruneisen(model, naive_committee, element;
                                        label="naive", T_range,
                                        δ=δ_grun, N_cell=N_cell_grun, n_mesh)

# Model still holds the last member's parameters — restore the constrained mean.
ACEpotentials.Models.set_linear_parameters!(model, constrained_params)

# ── Ensemble statistics (NaN-skipping over failed members) ───────────────────
row_stats(mat) = begin
    μ = similar(mat[:, 1]); σ = similar(μ)
    for t in 1:size(mat, 1)
        v = filter(!isnan, mat[t, :])
        μ[t] = isempty(v) ? NaN : mean(v)
        σ[t] = length(v) < 2 ? NaN : std(v)
    end
    μ, σ
end
γ_con_μ,   γ_con_σ   = row_stats(γ_con_committee)
γ_naive_μ, γ_naive_σ = row_stats(γ_naive_committee)

n_con_ok   = count(!isnan, γ_con_committee[1, :])
n_naive_ok = count(!isnan, γ_naive_committee[1, :])
println("\nUsable members — constrained: $n_con_ok / $(length(committee)), ",
        "naive: $n_naive_ok / $(length(naive_committee))")

# ── Plot ─────────────────────────────────────────────────────────────────────
T = collect(T_range)
blue   = RGBAf(0.15, 0.40, 0.75, 1.0)
orange = RGBAf(0.85, 0.45, 0.05, 1.0)

fig = Figure(size=(650, 450))
ax  = Axis(fig[1, 1];
           xlabel = "Temperature (K)",
           ylabel = "Grüneisen parameter γ̄",
           title  = "γ̄(T): constrained (rejection-sampled) vs naive POPS",
           xgridvisible = false)

n_naive_ok >= 2 && band!(ax, T, γ_naive_μ .- γ_naive_σ, γ_naive_μ .+ γ_naive_σ;
                         color=RGBAf(0.85, 0.45, 0.05, 0.20),
                         label="naive POPS ±1σ ($n_naive_ok members)")
n_con_ok   >= 2 && band!(ax, T, γ_con_μ .- γ_con_σ, γ_con_μ .+ γ_con_σ;
                         color=RGBAf(0.15, 0.40, 0.75, 0.20),
                         label="constrained committee ±1σ ($n_con_ok members)")
lines!(ax, T, γ_unc_mean; color=orange, linestyle=:dash, linewidth=2.5,
       label="unconstrained mean")
lines!(ax, T, γ_con_mean; color=blue, linewidth=2.5,
       label="constrained mean")
axislegend(ax; position=:rb, framevisible=false)

save("$(result.dir)/results/rejection_sampled_gruneisen_comparison.png", fig)
display(fig)

# ── Save data ────────────────────────────────────────────────────────────────
open("$(result.dir)/results/rejection_sampled_gruneisen_comparison.csv", "w") do io
    println(io, "# gamma_bar(T): shifted $(n_mesh)^3 BZ mesh, delta=$(δ_grun), $(N_cell_grun)^3 supercell")
    println(io, "# constrained committee: $n_con_ok/$(length(committee)) usable; naive POPS: $n_naive_ok/$(length(naive_committee)) usable")
    println(io, "T_K,gamma_con_mean,gamma_con_comm_mean,gamma_con_comm_std,gamma_unc_mean,gamma_naive_comm_mean,gamma_naive_comm_std")
    writedlm(io, hcat(T, γ_con_mean, γ_con_μ, γ_con_σ, γ_unc_mean, γ_naive_μ, γ_naive_σ), ',')
end
println("Saved: rejection_sampled_gruneisen_comparison.{png,csv}")
