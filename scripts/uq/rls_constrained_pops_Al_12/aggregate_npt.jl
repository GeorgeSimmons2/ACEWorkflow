# aggregate_npt.jl — a(T), FCC survival and α for RLS vs constrained RLS (Al_12_4_6A_2).
#   julia --project scripts/uq/rls_constrained_pops_Al_12/aggregate_npt.jl
# α = OLS slope / intercept over FCC points (convention of thermal_expansion_vs_experiment/),
# reported with the relaxed 0 K point (0–600 K) and MD points only (200–600 K).
using DelimitedFiles, Statistics, Printf, CairoMakie

D = joinpath(@__DIR__, "npt")
W = readdlm(joinpath(@__DIR__, "..", "..", "..", "thermal_expansion_vs_experiment", "wilson_1941_aluminium.csv"),
            ','; header=true, comments=true, comment_char='#'); cw = vec(string.(W[2]))
Tw = Float64.(W[1][:, findfirst(==("T_C"), cw)]) .+ 273.15; aw = Float64.(W[1][:, findfirst(==("a_kX_obs"), cw)]) .* 1.00202
linα(T, a) = length(T) < 2 ? NaN : (s = cov(T, a)/var(T); 1e5 * s / (mean(a) - s*mean(T)))

fig = Figure(size=(380, 304)); ax = Axis(fig[1,1]; xlabel="Temperature (K)", ylabel="Lattice constant a (Å)",
                                         xgridvisible=false, ygridvisible=false, xticks=0:100:600)
for (m, lab, col, mk) in (("rls", "RLS", RGBf(0.80,0.15,0.15), :rect),
                          ("constrained_2pct", "constrained ±2%", RGBf(0.90,0.62,0.0), :utriangle),
                          ("constrained", "constrained ±10%", RGBf(0.0,0.447,0.698), :circle))
    isfile("$D/$m/a0.csv") || continue
    a0 = readdlm("$D/$m/a0.csv", ','; skipstart=1)
    rows = [readdlm(f, ','; skipstart=1) for f in sort(filter(isfile, ["$D/$m/T$(t)K/summary_row.csv" for t in (200, 400, 600)]))]
    T = vcat(0.0, [r[1] for r in rows]); a = vcat(a0[1], [r[2] for r in rows]); σ = vcat(0.0, [r[3] for r in rows])
    fcc = vcat(true, [string(r[8]) == "true" for r in rows]); ω = vcat(a0[2], [r[5] for r in rows])
    @printf("%-16s", lab); for i in eachindex(T); @printf("  %3.0f K: a=%.5f ω=%+.2f %s", T[i], a[i], ω[i], fcc[i] ? "FCC" : "LEFT FCC"); end
    @printf("\n%-16s α (10⁻⁵/K), FCC points: 0–600 K %.2f   200–600 K %.2f  |  all points: 0–600 K %.2f\n", "",
            linα(T[fcc], a[fcc]), linα(T[fcc .& (T .> 0)], a[fcc .& (T .> 0)]), linα(T, a))
    lines!(ax, T, a; color=(col, 0.5)); errorbars!(ax, T, a, σ; color=col, whiskerwidth=5)
    scatter!(ax, T[fcc], a[fcc]; color=col, marker=mk, markersize=8, label=lab)
    any(.!fcc) && scatter!(ax, T[.!fcc], a[.!fcc]; color=col, marker=:xcross, markersize=10)
end
k = Tw .<= 650; scatter!(ax, Tw[k], aw[k]; color=:black, marker=:diamond, markersize=7, label="experiment (Wilson 1941)")
axislegend(ax; position=:lt, framevisible=false, labelsize=9)
save(joinpath(@__DIR__, "npt_aT.pdf"), fig); save(joinpath(@__DIR__, "npt_aT.png"), fig; px_per_unit=4)
