using CairoMakie, DelimitedFiles


f_con_rmses          = vec(readdlm("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/plots_and_data/pareto_5_percent/f_con_rmses.csv", ','))
e_con_rmses          = vec(readdlm("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/plots_and_data/pareto_5_percent/e_con_rmses.csv", ','))
e_rmses              = vec(readdlm("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/plots_and_data/pareto_5_percent/e_rmses.csv", ','))
f_rmses              = vec(readdlm("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/plots_and_data/pareto_5_percent/f_rmses.csv", ','))
e_time_average       = vec(readdlm("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/plots_and_data/pareto_5_percent/e_time_average.csv", ','))
e_con_time_average   = vec(readdlm("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/plots_and_data/pareto_5_percent/e_con_time_average.csv", ','))
f_con_time_average   = vec(readdlm("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/plots_and_data/pareto_5_percent/f_con_time_average.csv", ','))
f_time_average       = vec(readdlm("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/plots_and_data/pareto_5_percent/f_time_average.csv", ','))

f_con_rmses_3        = vec(readdlm("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/plots_and_data/pareto_5_percent_v_3/f_con_rmses.csv", ','))
e_con_rmses_3        = vec(readdlm("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/plots_and_data/pareto_5_percent_v_3/e_con_rmses.csv", ','))
e_rmses_3            = vec(readdlm("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/plots_and_data/pareto_5_percent_v_3/e_rmses.csv", ','))
f_rmses_3            = vec(readdlm("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/plots_and_data/pareto_5_percent_v_3/f_rmses.csv", ','))
e_time_average_3     = vec(readdlm("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/plots_and_data/pareto_5_percent_v_3/e_time_average.csv", ','))
e_con_time_average_3 = vec(readdlm("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/plots_and_data/pareto_5_percent_v_3/e_con_time_average.csv", ','))
f_con_time_average_3 = vec(readdlm("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/plots_and_data/pareto_5_percent_v_3/f_con_time_average.csv", ','))
f_time_average_3     = vec(readdlm("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/plots_and_data/pareto_5_percent_v_3/f_time_average.csv", ','))

fig = Figure(size = (425, 263), fontsize = 12)

ax = Axis(fig[1,1],
    xlabel = "RMSE (eV/atom)",
    ylabel = "Time (seconds/atom)",
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
        markersize = 16
    )

    # Unconstrained 4 body
    scatter!(ax,
        [e_rmses_3[j]], [e_time_average_3[j]],
        marker = :star6,
        color = d,
        colormap = cmap,
        colorrange = (12,20),
        markersize = 16
    )

    # Constrained 4 body
    scatter!(ax,
        [e_con_rmses_3[j]], [e_con_time_average_3[j]],
        marker = :diamond,
        color = d,
        colormap = cmap,
        colorrange = (12,20),
        markersize = 16
    )

end

# Marker legend
axislegend(ax,
    [
        MarkerElement(marker=:x, color=:black),
        MarkerElement(marker=:circle, color=:black),
        MarkerElement(marker=:star6, color=:black),
        MarkerElement(marker=:diamond, color=:black)
    ],
    [
        "Unconstrained 3-body",
        "Constrained 3-body",
        "Unconstrained 4-body",
        "Constrained 4-body"
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

save("plots_and_data/energy_pareto_front.png", fig)