# aggregate.jl — a(T) and α for the two Al_16 members, from every T<T>K/summary_row.csv
# under npt_al16/runs*/.  Safe on partial results.
#
#   julia --project npt_al16/aggregate.jl
#
# α uses the same convention as thermal_expansion_vs_experiment/: ordinary least squares of
# a(T) over the FCC points in a window, α = slope / fitted intercept.  Two windows are
# reported because they answer different questions:
#   0–300 K    includes the relaxed T = 0 point — the figure's convention
#   100–300 K  MD points only — insensitive to a jump between 0 K and the first MD point,
#              which is exactly what an unstable member does
# A member whose two windows disagree badly is not expanding linearly, and a single α for
# it is not a thermal-expansion coefficient.

using DelimitedFiles, Statistics, Printf, CairoMakie

ROOT   = @__DIR__
OUT    = get(ENV, "OUT", joinpath(ROOT, "aT_al16"))
FIGW   = parse(Float64, get(ENV, "FIGW", "380"))
WILSON = joinpath(ROOT, "..", "thermal_expansion_vs_experiment", "wilson_1941_aluminium.csv")
MEMBERS = [("unconstrained_softest_m02", "unconstrained (softest)", RGBf(0.80,0.15,0.15), :rect),
           ("constrained_softest_m04",   "constrained (softest)",   RGBf(0.0,0.447,0.698), :circle)]

function collect_member(tag)
    a0 = nothing; rows = []
    for base in filter(d -> startswith(basename(d), "runs"), readdir(ROOT; join=true))
        d = joinpath(base, tag); isdir(d) || continue
        f0 = joinpath(d, "a0.csv"); isfile(f0) && (a0 = readdlm(f0, ','; skipstart=1)[1, 1])
        for td in filter(x -> occursin(r"^T\d+K$", x), readdir(d))
            f = joinpath(d, td, "summary_row.csv"); isfile(f) || continue
            r = readdlm(f, ','; skipstart=1)
            push!(rows, (T=Float64(r[1]), a=Float64(r[2]), σ=Float64(r[3]), ω=Float64(r[5]),
                         coord=Float64(r[6]), fcc=(string(r[8]) == "true")))
        end
    end
    sort!(rows; by = r -> r.T)
    (; a0, rows)
end

function linfit(x, y)
    x̄, ȳ = mean(x), mean(y)
    s = sum((x .- x̄) .* (y .- ȳ)) / sum((x .- x̄) .^ 2)
    ȳ - s * x̄, s
end
alpha(T, a) = length(T) >= 2 ? (c = linfit(T, a); (α = c[2]/c[1], c0 = c[1], s = c[2])) : nothing

data = Dict(tag => collect_member(tag) for (tag, _) in MEMBERS)

println("member                       T (K)   a (Å)       ±σ        Δa/a₀    min ω (THz)  ⟨coord⟩  FCC")
for (tag, label, _, _) in MEMBERS
    d = data[tag]
    @printf("%-26s     0   %.6f   —          —        (0 K)          12.00    ✓\n", label, d.a0)
    for r in d.rows
        @printf("%-26s   %3.0f   %.6f   %.6f   %+.3f%%   %+8.3f      %5.2f    %s\n",
                "", r.T, r.a, r.σ, 100(r.a - d.a0)/d.a0, r.ω, r.coord, r.fcc ? "✓" : "✗ LEFT FCC")
    end
end

println("\nα (×10⁻⁵ K⁻¹), FCC points only")
fits = Dict()
for (tag, label, _, _) in MEMBERS
    d = data[tag]; ok = [r for r in d.rows if r.fcc]
    T_all = vcat(0.0, [r.T for r in ok]); a_all = vcat(d.a0, [r.a for r in ok])
    f03 = alpha(T_all, a_all)
    f13 = alpha([r.T for r in ok if r.T >= 100], [r.a for r in ok if r.T >= 100])
    fits[tag] = f03
    steps = diff(a_all) ./ diff(T_all) ./ d.a0 .* 1e5
    @printf("  %-26s 0–300 K: %5.2f    100–300 K: %5.2f    per-interval: %s\n",
            label, f03 === nothing ? NaN : 1e5f03.α, f13 === nothing ? NaN : 1e5f13.α,
            join([@sprintf("%.2f", s) for s in steps], ", "))
end

# ── figure ───────────────────────────────────────────────────────────────────
# Wilson 1941: T in °C and a in kX units, converted exactly as the paper figure does
# (thermal_expansion_vs_experiment/plot_thermal_expansion_vs_experiment.jl, KX_TO_ANG).
KX_TO_ANG = parse(Float64, get(ENV, "KX_TO_ANG", "1.00202"))
W  = readdlm(WILSON, ','; header=true, comments=true, comment_char='#')
cw = vec(string.(W[2]))
Tw = Float64.(W[1][:, findfirst(==("T_C"), cw)]) .+ 273.15
aw = Float64.(W[1][:, findfirst(==("a_kX_obs"), cw)]) .* KX_TO_ANG
keepw = Tw .<= 350

set_theme!(fontsize = 11)
fig = Figure(size = (FIGW, 0.8FIGW), figure_padding = (4, 10, 4, 4))
ax = Axis(fig[1, 1]; xlabel = "Temperature (K)", ylabel = "Lattice constant a (Å)",
          xgridvisible = false, ygridvisible = false, xtickalign = 1, ytickalign = 1,
          xticks = 0:100:300)
for (tag, label, col, mk) in MEMBERS
    d = data[tag]; T = vcat(0.0, [r.T for r in d.rows]); a = vcat(d.a0, [r.a for r in d.rows])
    σ = vcat(0.0, [r.σ for r in d.rows])
    lines!(ax, T, a; color = (col, 0.5), linewidth = 1.0)
    errorbars!(ax, T, a, σ; color = (col, 0.7), whiskerwidth = 5)
    scatter!(ax, T, a; color = col, marker = mk, markersize = 8, label = label)
end
any(keepw) && scatter!(ax, Tw[keepw], aw[keepw]; color = :black, marker = :diamond,
                       markersize = 8, label = "experiment (Wilson 1941)")
f = fits["constrained_softest_m04"]
f === nothing || text!(ax, 0.03, 0.97; space = :relative, align = (:left, :top), fontsize = 10,
                       color = RGBf(0.0,0.447,0.698),
                       text = @sprintf("constrained α = %.1f × 10⁻⁵ K⁻¹", 1e5f.α))
axislegend(ax; position = :rb, framevisible = false, labelsize = 9, patchsize = (10, 8), rowgap = 0)
save("$OUT.pdf", fig); save("$OUT.png", fig; px_per_unit = 4)
println("\nfigure → $OUT.{pdf,png}")
