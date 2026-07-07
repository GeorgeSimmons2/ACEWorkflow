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

writedlm("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/plots_and_data/pareto_50_percent/f_con_rmses.csv", f_con_rmses, ',')
writedlm("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/plots_and_data/pareto_50_percent/e_con_rmses.csv", e_con_rmses, ',')
writedlm("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/plots_and_data/pareto_50_percent/e_rmses.csv", e_rmses, ',')
writedlm("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/plots_and_data/pareto_50_percent/f_rmses.csv", f_rmses, ',')
writedlm("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/plots_and_data/pareto_50_percent/e_time_average.csv", e_time_average, ',')
writedlm("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/plots_and_data/pareto_50_percent/e_con_time_average.csv", e_con_time_average, ',')
writedlm("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/plots_and_data/pareto_50_percent/f_con_time_average.csv", f_con_time_average, ',')
writedlm("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/plots_and_data/pareto_50_percent/f_time_average.csv", f_time_average, ',')

using CairoMakie
using CairoMakie

fig = Figure(
    size = (900, 1000),   # taller for stacked panels
    fontsize = 18         # base font size for everything
)

# -----------------------------
# TOP: EOS curves
# -----------------------------
ax1 = Axis(fig[1, 1];
    title  = "Bulk energy vs lattice constant — repulsive core",
    xlabel = "Lattice constant (Å)",
    ylabel = "Energy per cell (eV)",
    titlesize = 22,
    xlabelsize = 20,
    ylabelsize = 20,
    xticklabelsize = 16,
    yticklabelsize = 16
)

lines!(ax1, full_lattice_constants, constrained_bulk_energies_add_zbl;
    color=:steelblue, linewidth=3,
    label="ACE constrained + ZBL"
)

lines!(ax1, full_lattice_constants, unconstrained_bulk_energies_add_zbl;
    color=:orange, linewidth=3, linestyle=:dash,
    label="ACE unconstrained + ZBL"
)

lines!(ax1, full_lattice_constants, zbl_energies;
    color=:red, linewidth=2, linestyle=:dot,
    label="ZBL"
)

hlines!(ax1, [0.0]; color=(:black, 0.4), linewidth=1, linestyle=:dash)

vlines!(ax1, [lattice_constants[end]];
    color=(:black, 0.4), linewidth=1,
    linestyle=:dash
)

axislegend(ax1;
    position = :rt,
    fontsize = 16,
    framevisible = false
)

ylims!(ax1, e_lo, e_hi)


# -----------------------------
# BOTTOM: histogram
# -----------------------------
ax2 = Axis(fig[2, 1];
    xlabel = "Pair distance (Å)",
    ylabel = "Frequency",
    title  = "Pair distance distribution (all configs)",
    titlesize = 22,
    xlabelsize = 20,
    ylabelsize = 20,
    xticklabelsize = 16,
    yticklabelsize = 16
)

all_pairs = vcat(all_dists...)

edges = 0:0.05:maximum(full_lattice_constants)
h = fit(Histogram, all_pairs, edges)

centers = 0.5 .* (h.edges[1][1:end-1] .+ h.edges[1][2:end])

barplot!(ax2, centers, h.weights;
    color = (:gray, 0.7)
)

linkxaxes!(ax1, ax2)

# spacing between panels (IMPORTANT for papers)
rowgap!(fig.layout, 20)

# -----------------------------
# SAVE (IMPORTANT: use vector format for papers)
# -----------------------------
save("$(result.dir)/results/eos_with_pair_hist.pdf", fig)
save("$(result.dir)/results/eos_with_pair_hist.png", fig; px_per_unit = 2)