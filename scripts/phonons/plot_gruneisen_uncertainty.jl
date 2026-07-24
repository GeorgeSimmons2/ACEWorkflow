# plot_gruneisen_uncertainty.jl
#
# Plots the Grüneisen parameter vs. temperature, with ensemble uncertainty,
# from the `results::Dict{Int,NamedTuple}` produced by gruneisen_param_sweep
# in gruneisen_phonon_bands_ace.jl (each entry has .T and .gamma for one
# parameter sample, all sharing the same T grid).
#
# Usage (results already in scope from gruneisen_param_sweep):
#   include("plot_gruneisen_uncertainty.jl")
#
# Or starting fresh from a saved summary:
#   using Serialization
#   results = deserialize("gruneisen_param_sweep/summary_Al.jls")
#   include("plot_gruneisen_uncertainty.jl")

using Statistics, CairoMakie

T      = first(values(results)).T
gammas = reduce(hcat, [r.gamma for r in values(results)])   # n_T x n_samples

γ_mean = vec(mean(gammas; dims=2))
γ_std  = vec(std(gammas; dims=2))

fig = Figure(size=(600, 420))
ax  = Axis(fig[1, 1];
           xlabel = "Temperature (K)",
           ylabel = "Grüneisen parameter γ",
           title  = "Grüneisen parameter vs. temperature (ensemble mean ± 1σ)")

band!(ax, T, γ_mean .- γ_std, γ_mean .+ γ_std; color=RGBAf(0.15, 0.4, 0.75, 0.25))
scatterlines!(ax, T, γ_mean; color=RGBAf(0.15, 0.4, 0.75, 0.95), linewidth=2.5, markersize=8)

save("$(result.dir)/results/gruneisen_uncertainty.png", fig)
display(fig)
println("Saved: $(result.dir)/results/gruneisen_uncertainty.png")
