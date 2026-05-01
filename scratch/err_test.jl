using Pkg; Pkg.add("CairoMakie")
using DelimitedFiles, CairoMakie, ExtXYZ, Unitful
using ACEpotentials
test = ExtXYZ.load("high_entropy_pops/manual_df_test_Al.xyz")
num_atoms = []
test_energies = []
for (i, at) in enumerate(test)
  push!(num_atoms, length(at))
  push!(test_energies, ustrip(at[:dft_energy]))
end

predictions = readdlm("./big_ACE/test_errors_18.234966384203847__17_4.csv", ',')
predictions = predictions[:,1]
predictions = predictions[2:2:end]
predictions = predictions ./ num_atoms
test_energies = test_energies ./ num_atoms

using CairoMakie, Statistics

# --- RMSE ---
function rmse(pred, truth)
    return sqrt(mean((pred .- truth).^2))
end

err = rmse(predictions, test_energies)
println("RMSE = $err")

# --- Plot ---
f = Figure(resolution = (600, 600))
ax = Axis(f[1, 1],
    xlabel = "DFT Test Energies per Atom / eV",
    ylabel = "Predicted Energies per Atom / eV",
    title = "Predicted vs True Energies\nRMSE = $(round(err, digits=4)) / eV"
)

scatter!(ax, test_energies, predictions)

# Optional: identity line
lines!(ax,
    [0, maximum(test_energies)],
    [0, maximum(test_energies)],
    color = :red,
    linestyle = :dash,
    label = "Identity"
)
axislegend(ax)

save("prediction_vs_truth.png", f)




