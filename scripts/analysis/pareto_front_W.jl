using ACEWorkflow, Random, ExtXYZ, CairoMakie, Statistics, Unitful, ACEpotentials
import AtomsCalculators: forces, potential_energy
Random.seed!(1234)

# ── Load model ────────────────────────────────────────────────────────────────
result = load_model(:W, 20, 4, 6, 3)
model  = result.model

constrained_model = deepcopy(model)
ACEpotentials.Models.set_linear_parameters!(constrained_model, vec(readdlm("$(result.dir)/repulsive_constrained_params.csv", ',')))

testing_configs = ExtXYZ.load("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/data/W/df_W_test.extxyz")
testing_configs = testing_configs[1:10:end]

predicted_test_energies             = []
constrained_predicted_test_energies = []
test_energies                       = []
energy_time                         = []
constrained_energy_time             = []
predicted_test_forces               = []
constrained_predicted_test_forces   = []
test_forces                         = []
force_time                          = []
constrained_force_time              = []

for config in testing_configs
    t_E = @elapsed begin
        E = potential_energy(config, model)
    end
    t_E_con = @elapsed begin
        E_con = potential_energy(config, constrained_model)
    end
    n = length(config)
    append!(predicted_test_energies, ustrip(E) / n)
    append!(constrained_predicted_test_energies, ustrip(E_con) / n)
    append!(test_energies, ustrip(config[:energy]) / n)
    append!(energy_time, t_E / n)
    append!(constrained_energy_time, t_E_con / n)

    if haskey(config[1], :forces)
        config_forces = [at[:forces] for at in config]
        t_F = @elapsed begin 
            F = forces(config, model)
        end
        append!(predicted_test_forces, reduce(vcat, ustrip.(F)))
        append!(test_forces, reduce(vcat, ustrip.(config_forces)))
        t_F_con = @elapsed begin
            F_con = forces(config, constrained_model)
        end
        append!(constrained_predicted_test_forces, reduce(vcat, ustrip.(F_con)))
        append!(force_time, t_F / n)
        append!(constrained_force_time, t_F_con / n)
    end
end

predicted_test_energies             = []
constrained_predicted_test_energies = []
test_energies                       = []
energy_time                         = []
constrained_energy_time             = []
predicted_test_forces               = []
constrained_predicted_test_forces   = []
test_forces                         = []
force_time                          = []
constrained_force_time              = []

for config in testing_configs
    t_E = @elapsed begin
        E = potential_energy(config, model)
    end
    t_E_con = @elapsed begin
        E_con = potential_energy(config, constrained_model)
    end
    n = length(config)
    append!(predicted_test_energies, ustrip(E) / n)
    append!(constrained_predicted_test_energies, ustrip(E_con) / n)
    append!(test_energies, ustrip(config[:dft_energy]) / n)
    append!(energy_time, t_E / n)
    append!(constrained_energy_time, t_E_con / n)

    if haskey(config[1], :dft_forces)
        config_forces = [at[:dft_forces] for at in config]
        t_F = @elapsed begin 
            F = forces(config, model)
        end
        append!(predicted_test_forces, reduce(vcat, ustrip.(F)))
        append!(test_forces, reduce(vcat, ustrip.(config_forces)))
        t_F_con = @elapsed begin
            F_con = forces(config, constrained_model)
        end
        append!(constrained_predicted_test_forces, reduce(vcat, ustrip.(F_con)))
        append!(force_time, t_F / n)
        append!(constrained_force_time, t_F_con / n)
    end
end

let
    e_true = Float64.(test_energies)
    e_pred = Float64.(predicted_test_energies)
    e_con_pred = Float64.(constrained_predicted_test_energies)
    e_rmse = sqrt(mean((e_pred .- e_true).^2))
    e_con_rmse = sqrt(mean((e_con_pred .- e_true).^2))
    f_true = Float64.(test_forces)
    f_pred = Float64.(predicted_test_forces)
    f_con_pred = Float64.(constrained_predicted_test_forces)
    f_rmse = sqrt(mean((f_pred .- f_true).^2))
    f_con_rmse = sqrt(mean((f_con_pred .- f_true).^2))
    append!(e_rmses, e_rmse)
    append!(e_con_rmses, e_con_rmse)
    append!(f_rmses, f_rmse)
    append!(f_con_rmses, f_con_rmse)    
    append!(e_time_average, mean(energy_time))
    append!(e_con_time_average, mean(constrained_energy_time))
    append!(f_time_average, mean(force_time))
    append!(f_con_time_average, mean(constrained_force_time))
end
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

    save("$(result.dir)/results/energy_parity.png", fig)
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

        save("$(result.dir)/results/force_parity.png", fig)
    end
end
