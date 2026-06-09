using ACEWorkflow, Random, ExtXYZ, CairoMakie, Statistics
import AtomsCalculators: forces, potential_energy#, @committee
Random.seed!(1234)
# ── Load model (adjust path as needed) ──────────────────────────────────────
result = load_model(:Al, 20, 4, 6, 3)
model  = result.model

N_largest_leverage = 40

# ── POPS ─────────────────────────────────────────────────────────────────────
# X = Diagonal(result.W) * result.A / result.P
# Yw = result.W .* result.Y
# Gamma = result.P
# lambda = 1.0 / size(X,1)
# C      = (Gamma' * Gamma .* lambda .+ X' * X)
# A      = C \ X'
# leverage = diag(X * A)    
# pops_corrections = corrections(X, Yw, Gamma; leverage_percentile=0.0)
# # hypercube_eigenvectors, hypercube_bounds = hypercube(pops_corrections)
# # pops_samples, dθ = sample_hypercube(hypercube_eigenvectors, hypercube_bounds, result.lin_params; number_of_committee_members=50)
# pops_samples = [vec(pops_corrections[i,:]) .+ result.lin_params for i=1:size(pops_samples, 1)]
# N_largest_leverage_inds = partialsortperm(leverage, length(Yw)-N_largest_leverage:length(Yw))
ACEpotentials.Models.set_committee!(model, pops_samples)
# corrections_maxima = maximum(pops_corrections, dims=1)' .+ result.lin_params
# corrections_minima = minimum(pops_corrections, dims=1)' .+ result.lin_params

# lb = vec(corrections_minima)
# ub = vec(corrections_maxima)
# outside_naive_ind = [(i, findall((s .< lb) .| (s .> ub))) for (i, s) in enumerate(pops_samples) if !all(lb .< s .< ub)]
testing_configs = ExtXYZ.load("../ace_archive/high_entropy_pops/manual_df_test_Al.xyz")
testing_configs = testing_configs[1:20:end]

predicted_test_energies = []
test_energies = []
predicted_test_forces = []
test_forces = []
predicted_test_energies_uncertainties = ([], [])
predicted_test_forces_uncertainties = ([], [])

for config in testing_configs
    E, co_E = @committee potential_energy(config, model)
    append!(predicted_test_energies, ustrip(E))
    append!(predicted_test_energies_uncertainties[1], minimum(ustrip.(co_E)))
    append!(predicted_test_energies_uncertainties[2], maximum(ustrip.(co_E)))
    append!(test_energies, ustrip(config[:dft_energy]))
    
    if haskey(config[1], :dft_forces)
        config_forces = [at[:dft_forces] for at in config]
        F, co_F = @committee forces(config, model)
        append!(test_forces, reduce(vcat, ustrip.(config_forces)))
        append!(predicted_test_forces, reduce(vcat, ustrip.(F)))
        for i in eachindex(co_F[1])
            fi = reduce(hcat, ustrip(co_F[k][i]) for k in eachindex(co_F))  # 3×n_committee matrix
            append!(predicted_test_forces_uncertainties[1], vec(minimum(fi; dims=2)))
            append!(predicted_test_forces_uncertainties[2], vec(maximum(fi; dims=2)))
        end
    end
end

# ── Parity plot: Energies ─────────────────────────────────────────────────────
let
    e_true = Float64.(test_energies)
    e_pred = Float64.(predicted_test_energies)
    e_lo   = Float64.(predicted_test_energies_uncertainties[1])
    e_hi   = Float64.(predicted_test_energies_uncertainties[2])

    e_rmse = sqrt(mean((e_pred .- e_true).^2))

    fig = Figure(size = (600, 600))
    ax  = Axis(fig[1, 1];
        xlabel = "DFT energy (eV)",
        ylabel = "ACE energy (eV)",
        title  = "Energy parity plot  (RMSE = $(round(e_rmse, sigdigits=3)) eV)",
    )

    errorbars!(ax, e_true, e_pred, e_pred .- e_lo, e_hi .- e_pred;
               whiskerwidth = 6, color = (:steelblue, 0.5))
    scatter!(ax, e_true, e_pred; color = :steelblue, markersize = 8)

    elims = extrema([e_true; e_pred])
    lines!(ax, collect(elims), collect(elims); color = :black, linestyle = :dash)

    save("$(result.dir)/results/leverage_weighted_delta_forest_sampling_energy_parity.png", fig)
end

# ── Parity plot: Forces ───────────────────────────────────────────────────────
if !isempty(test_forces)
    let
        f_true = Float64.(test_forces)
        f_pred = Float64.(predicted_test_forces)
        f_lo   = Float64.(predicted_test_forces_uncertainties[1])
        f_hi   = Float64.(predicted_test_forces_uncertainties[2])

        f_rmse = sqrt(mean((f_pred .- f_true).^2))

        fig = Figure(size = (600, 600))
        ax  = Axis(fig[1, 1];
            xlabel = "DFT force component (eV/Å)",
            ylabel = "ACE force component (eV/Å)",
            title  = "Force parity plot  (RMSE = $(round(f_rmse, sigdigits=3)) eV/Å)",
        )

        errorbars!(ax, f_true, f_pred, f_pred .- f_lo, f_hi .- f_pred;
                   whiskerwidth = 6, color = (:tomato, 0.4))
        scatter!(ax, f_true, f_pred; color = :tomato, markersize = 5)

        flims = extrema([f_true; f_pred])
        lines!(ax, collect(flims), collect(flims); color = :black, linestyle = :dash)

        save("$(result.dir)/results/leverage_weighted_delta_forest_sampling_force_parity.png", fig)
    end
end
