using ACEWorkflow, Random, ExtXYZ, CairoMakie, Statistics, Unitful, DelimitedFiles, ACEpotentials
import AtomsCalculators: forces, potential_energy
Random.seed!(1234)

# ── Load model ────────────────────────────────────────────────────────────────
result = load_model(:Al, 12, 4, 6, 2)
model, _ = ACEpotentials.load_model("$(result.dir)/constrained_model.json")

testing_configs = ExtXYZ.load("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/data/Al/manual_df_test_Al.xyz")
testing_configs = testing_configs[1:10:end]

predicted_test_energies = []
test_energies           = []
predicted_test_forces   = []
test_forces             = []

for config in testing_configs
    E = potential_energy(config, model)
    n = length(config)
    append!(predicted_test_energies, ustrip(E) / n)
    append!(test_energies, ustrip(config[:dft_energy]) / n)

    if haskey(config[1], :dft_forces)
        config_forces = [at[:dft_forces] for at in config]
        F = forces(config, model)
        append!(test_forces, reduce(vcat, ustrip.(config_forces)))
        append!(predicted_test_forces, reduce(vcat, ustrip.(F)))
    end
end

# ── Parity plot: Energies ─────────────────────────────────────────────────────
let
    e_true = Float64.(test_energies)
    e_pred = Float64.(predicted_test_energies)
    e_rmse = sqrt(mean((e_pred .- e_true).^2))

    fig = Figure(size = (600, 600))
    ax  = Axis(fig[1, 1];
        xlabel = "DFT energy (eV/atom)",
        ylabel = "ACE energy (eV/atom)",
        title  = "Energy parity plot  (RMSE = $(round(e_rmse, sigdigits=3)) eV/atom)",
    )

    scatter!(ax, e_true, e_pred; color = :steelblue, markersize = 8)

    elims = extrema([e_true; e_pred])
    lines!(ax, collect(elims), collect(elims); color = :black, linestyle = :dash)

    save("$(result.dir)/results/constrained_energy_parity.png", fig)
end

# ── Parity plot: Forces ───────────────────────────────────────────────────────
if !isempty(test_forces)
    let
        f_true = Float64.(test_forces)
        f_pred = Float64.(predicted_test_forces)
        f_rmse = sqrt(mean((f_pred .- f_true).^2))

        fig = Figure(size = (600, 600))
        ax  = Axis(fig[1, 1];
            xlabel = "DFT force component (eV/Å)",
            ylabel = "ACE force component (eV/Å)",
            title  = "Force parity plot  (RMSE = $(round(f_rmse, sigdigits=3)) eV/Å)",
        )

        scatter!(ax, f_true, f_pred; color = :tomato, markersize = 5)

        flims = extrema([f_true; f_pred])
        lines!(ax, collect(flims), collect(flims); color = :black, linestyle = :dash)

        save("$(result.dir)/results/constrained_force_parity.png", fig)
    end
end
