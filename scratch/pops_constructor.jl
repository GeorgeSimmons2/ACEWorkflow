using ACEpotentials, ExtXYZ, DelimitedFiles, POPSRegression

training_configs = ExtXYZ.load("manual_df_train_Al.xyz")

model = ACEpotentials.load_model("high_entropy_POPS/model.json")
A = readdlm("high_entropy_POPS/A.csv", ',')
Y = readdlm("high_entropy_POPS/Y.csv", ',')
P = readdlm("high_entropy_POPS/P.csv", ',')
W = readdlm("high_entropy_POPS/W.csv", ',')
Y = Y[:,1]
W = W[:,1]
Ap= Diagonal(W) * A / P
Y = W .* Y
lin_params = P \ (Ap \ Y)
ACEpotentials.Models.set_linear_parameters!(model, lin_params)
pointwise_corrections = corrections(Ap, Y, P; leverage_percentile=0.0, lambda=1/size(Ap,1))
hypercube_eigs, hypercube_bounds = hypercube(pointwise_corrections)
committee, dΘ = sample_hypercube(hypercube_eigs, hypercube_bounds, lin_params)
co_ps_vec = [committee[:,i] for i = 1:size(committee,2)]
ACEpotentials.Models.set_committee!(model, co_ps_vec)

test = ExtXYZ.load("high_entropy_pops/manual_df_test_Al.xyz")
test_E = []
test_E_predictions = []
test_co_E_predictions = []
ev_val = counter = 0  
co_E_range = []
for (i, at) in enumerate(test)
    try
        energy = at[:dft_energy]
    
        push!(test_E, energy)
        E, co_E= @committee AtomsCalculators.potential_energy(at, model)
        push!(test_E_predictions, ustrip(E) - energy)
        push!(test_co_E_predictions, ustrip.(co_E) .- ustrip(E))
        push!(co_E_range, abs(maximum(ustrip.(co_E)) - minimum(ustrip.(co_E))))
        if (ustrip(E) > minimum(ustrip.(co_E)) && ustrip(E) < maximum(ustrip.(co_E)))
            counter += 1
        else
            ev_val +=  1
            counter += 1
        end
    catch
    end
    
end
ev_val /= counter
ev_val *= 100

using CairoMakie

# Create figure and axis
fig = CairoMakie.Figure()
ax = CairoMakie.Axis(fig[1, 1],
    xlabel = "Test errors / eV",
    ylabel = "Uncertainty estimates / eV"
)

# Data
xdata = abs.(test_E_predictions)
ydata = co_E_range ./ 2

# Scatter plot
CairoMakie.scatter!(ax, xdata, ydata)

# Add identity line
max_val = maximum([maximum(xdata), maximum(ydata)])
CairoMakie.lines!(ax, [0, max_val], [0, max_val],
    linestyle = :dash,
    label = "Identity"
)

# Add EV label (invisible dummy line)
CairoMakie.lines!(ax, [NaN], [NaN],
    label = "EV = $(ev_val)%",
    color = :white
)

# Axis limits and legend
padding = 0.05 * max_val
CairoMakie.xlims!(ax, 0, max_val + padding)
CairoMakie.ylims!(ax, 0, max_val + padding)
CairoMakie.axislegend(ax, position = :rb)

# Save to file
CairoMakie.save("./high_entropy_pops/test_error_corner_plot_2.5.png", fig)

# Display figure
fig
