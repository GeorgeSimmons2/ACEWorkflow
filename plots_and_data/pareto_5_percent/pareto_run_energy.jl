using DelimitedFiles
e_rmses=[]
e_con_rmses=[]
f_rmses=[]
f_con_rmses=[]
e_time_average=[]
e_con_time_average=[]
f_time_average=[]
f_con_time_average=[]

for i=12:20
    global current_i=i; include("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/scripts/analysis/pareto_front.jl")
end

writedlm("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/plots_and_data/pareto_5_percent/f_con_rmses.csv", f_con_rmses, ',')
writedlm("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/plots_and_data/pareto_5_percent/e_con_rmses.csv", e_con_rmses, ',')
writedlm("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/plots_and_data/pareto_5_percent/e_rmses.csv", e_rmses, ',')
writedlm("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/plots_and_data/pareto_5_percent/f_rmses.csv", f_rmses, ',')
writedlm("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/plots_and_data/pareto_5_percent/e_time_average.csv", e_time_average, ',')
writedlm("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/plots_and_data/pareto_5_percent/e_con_time_average.csv", e_con_time_average, ',')
writedlm("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/plots_and_data/pareto_5_percent/f_con_time_average.csv", f_con_time_average, ',')
writedlm("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/plots_and_data/pareto_5_percent/f_time_average.csv", f_time_average, ',')

using CairoMakie

fig = Figure(size = (425, 263), fontsize = 12)

ax = Axis(fig[1,1],
    xlabel = "RMSE (eV/atom)",
    ylabel = "Time (seconds/atom)"
)

degrees = 12:20
cmap = cgrad(:viridis, length(degrees), categorical=true)

# Plot points
for (j, d) in enumerate(degrees)

    # Unconstrained
    scatter!(ax,
        [e_rmses[j]], [e_time_average[j]],
        marker = :x,
        color = d,
        colormap = cmap,
        colorrange = (12,20),
        markersize = 16
    )

    # Constrained
    scatter!(ax,
        [e_con_rmses[j]], [e_con_time_average[j]],
        marker = :circle,
        color = d,
        colormap = cmap,
        colorrange = (12,20),
        markersize = 14
    )

end

# Marker legend
axislegend(ax,
    [
        MarkerElement(marker=:x, color=:black),
        MarkerElement(marker=:circle, color=:black)
    ],
    [
        "Unconstrained",
        "Constrained"
    ],
    position=:rt
)

# Discrete colorbar
Colorbar(fig[1,2],
    colormap = cmap,
    limits = (12,20),
    ticks = (12:20, string.(12:20)),
    label = "Total degree"
)

save("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/plots_and_data/pareto_5_percent/energy_pareto_front.png", fig)


using CairoMakie

fig = Figure(size = (425, 263), fontsize = 12)

ax = Axis(fig[1,1],
    xlabel = "RMSE (eV/Å)",
    ylabel = "Time (seconds/atom)"
)

degrees = 12:20
cmap = cgrad(:viridis, length(degrees), categorical=true)

# Plot points
for (j, d) in enumerate(degrees)

    # Unconstrained
    scatter!(ax,
        [f_rmses[j]], [f_time_average[j]],
        marker = :x,
        color = d,
        colormap = cmap,
        colorrange = (12,20),
        markersize = 16
    )

    # Constrained
    scatter!(ax,
        [f_con_rmses[j]], [f_con_time_average[j]],
        marker = :circle,
        color = d,
        colormap = cmap,
        colorrange = (12,20),
        markersize = 14
    )

end

# Marker legend
axislegend(ax,
    [
        MarkerElement(marker=:x, color=:black),
        MarkerElement(marker=:circle, color=:black)
    ],
    [
        "Unconstrained",
        "Constrained"
    ],
    position=:rt
)

# Discrete colorbar
Colorbar(fig[1,2],
    colormap = cmap,
    limits = (12,20),
    ticks = (12:20, string.(12:20)),
    label = "Total degree"
)

save("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/plots_and_data/pareto_5_percent/force_pareto_front.png", fig)

