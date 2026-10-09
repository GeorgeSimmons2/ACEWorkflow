# aggregate.jl — α over every member of the constrained committee, not one member.
#
#   julia --project constrained_optimization/npt_spread/aggregate.jl
#
# Safe to run while the array is still going: members without a summary are listed and
# skipped, so a partial run gives an early estimate.
#
# α is computed EXACTLY as thermal_expansion_vs_experiment/ does for the figure: ordinary
# least squares of a(T) over the common window 0–700 K (the T = 0 relaxed point included,
# 900 K excluded), α = slope / fitted intercept.
#
# Env:  SPREAD_DIR  (default models/Al_12_4_6A_2_/results/repro_npt_spread)
#       OUT         figure stem (default constrained_optimization/npt_spread/alpha_spread)
#       FIGW        display width in pt (default 260 — a half-width slot)

using DelimitedFiles, Statistics, Printf, CairoMakie

ROOT   = normpath(joinpath(@__DIR__, "..", ".."))
SPREAD = get(ENV, "SPREAD_DIR", joinpath(ROOT, "models/Al_12_4_6A_2_/results/repro_npt_spread"))
PUB    = joinpath(ROOT, "models/Al_12_4_6A_2_/results/npt_multivolume_softest/thermal_expansion_summary.csv")
OUT    = get(ENV, "OUT", joinpath(@__DIR__, "alpha_spread"))
FIGW   = parse(Float64, get(ENV, "FIGW", "260"))
WIN    = (0.0, 700.0)
PUBLISHED_ROW = 18                                   # theta_used.csv == committee row 18
α_EXP  = 2.669e-5                                    # Wilson 1941, same window (figure printout)
α_PUB  = 2.670e-5                                    # the figure's constrained α

function read_summary(p)
    lines = readlines(p)
    meta  = Dict{String,String}()
    for l in filter(startswith("#"), lines), kv in split(strip(l, ['#', ' ']), "  "; keepempty=false)
        if occursin('=', kv)
            k, v = split(kv, '='; limit=2)
            meta[strip(k)] = strip(v)
        end
    end
    body   = filter(!startswith("#"), lines)
    header = split(body[1], ',')
    rows   = [split(l, ',') for l in body[2:end] if !isempty(strip(l))]
    col(n) = [r[findfirst(==(n), header)] for r in rows]
    T   = parse.(Float64, col("T_K")); a = parse.(Float64, col("a_Ang"))
    fcc = "still_fcc" in header ? (col("still_fcc") .== "true") : trues(length(T))
    (; T, a, fcc, meta)
end

function linfit(x, y)
    x̄, ȳ = mean(x), mean(y)
    s = sum((x .- x̄) .* (y .- ȳ)) / sum((x .- x̄) .^ 2)
    ȳ - s * x̄, s
end
function alpha(s)
    keep = (WIN[1] .<= s.T .<= WIN[2]) .& (s.T .!= 900.0)
    a0, slope = linfit(s.T[keep], s.a[keep])
    (α = slope / a0, fcc_in_window = all(s.fcc[keep]))
end
worst_minω(s) = haskey(s.meta, "minomega_at_constrained_volumes_THz") ?
    minimum(parse.(Float64, split(s.meta["minomega_at_constrained_volumes_THz"]))) : NaN

# ── collect ──────────────────────────────────────────────────────────────────
res, missing_members = [], Int[]
for k in 1:30
    p = joinpath(SPREAD, @sprintf("member_%02d", k), "thermal_expansion_summary.csv")
    isfile(p) || (push!(missing_members, k); continue)
    s = read_summary(p); f = alpha(s)
    push!(res, (k = k, α = f.α, fcc = f.fcc_in_window, fcc900 = s.fcc[end], ω = worst_minω(s), s = s))
end
@printf("members with a summary: %d/30%s\n", length(res),
        isempty(missing_members) ? "" : "   (pending: $(join(missing_members, ' ')))")
isempty(res) && exit(0)

println("\nmember   α (1e-5/K)   worst min ω over volumes (THz)   FCC ≤700 K   FCC at 900 K")
for r in sort(res; by = r -> r.α)
    @printf("  %2d%s     %6.3f          %+.4f                     %-5s        %s\n",
            r.k, r.k == PUBLISHED_ROW ? "*" : " ", 1e5r.α, r.ω, r.fcc, r.fcc900)
end
println("  * = the published member")

# ── control: member 18 must reproduce the published run exactly ──────────────
i18 = findfirst(r -> r.k == PUBLISHED_ROW, res)
if i18 !== nothing
    p = read_summary(PUB); m = res[i18].s
    d = maximum(abs.(p.a .- m.a))
    @printf("\nCONTROL member %d vs published summary: max |Δa| = %.3e Å  %s\n", PUBLISHED_ROW, d,
            d == 0 ? "✓ identical — MD is deterministic given θ and MD_SEED" :
                     "✗ NOT identical — the determinism assumption behind this spread is broken")
end

# ── the distribution ─────────────────────────────────────────────────────────
good = [r for r in res if r.fcc]                     # members still FCC wherever α is fitted
for (label, set) in (("all members", res), ("FCC at every fitted point", good))
    isempty(set) && continue
    αs = [r.α for r in set]
    q = quantile(αs, [0.1, 0.25, 0.5, 0.75, 0.9])
    @printf("\n%s (n=%d)\n", label, length(set))
    @printf("  α mean %.3f ± %.3f (std) ×10⁻⁵/K,  std. error of mean %.3f\n",
            1e5mean(αs), 1e5std(αs), 1e5std(αs)/sqrt(length(αs)))
    @printf("  median %.3f   IQR [%.3f, %.3f]   10–90%% [%.3f, %.3f]   range [%.3f, %.3f]\n",
            1e5q[3], 1e5q[2], 1e5q[4], 1e5q[1], 1e5q[5], 1e5minimum(αs), 1e5maximum(αs))
    @printf("  experiment %.3f sits at the %.0f%% point; published member %.3f at the %.0f%% point\n",
            1e5α_EXP, 100mean(αs .<= α_EXP), 1e5α_PUB, 100mean(αs .<= α_PUB))
end
if length(res) >= 4
    ω = [r.ω for r in res]; α = [r.α for r in res]
    @printf("\ncorrelation(worst min ω, α) = %+.2f   — is the 'softest' selection biased?\n", cor(ω, α))
end
println("members transformed out of FCC by 700 K: ",
        join([r.k for r in res if !r.fcc], ' ') |> x -> isempty(x) ? "none" : x)

# ── figure: α per member, experiment and the published member marked ─────────
set_theme!(fontsize = 11)
fig = Figure(size = (FIGW, 0.8FIGW), figure_padding = (4, 8, 4, 4))
ax  = Axis(fig[1, 1]; xlabel = "α (10⁻⁵ K⁻¹)", ylabel = "committee members",
           xgridvisible = false, ygridvisible = false, xtickalign = 1, ytickalign = 1)
αs  = [1e5r.α for r in res]
hist!(ax, αs; bins = max(6, round(Int, sqrt(length(αs)) + 2)), color = (RGBf(0.0, 0.447, 0.698), 0.55),
      strokecolor = :white, strokewidth = 0.5)
bad = [1e5r.α for r in res if !r.fcc]
isempty(bad) || scatter!(ax, bad, fill(0.25, length(bad)); color = :crimson, marker = :xcross, markersize = 7,
                         label = "left FCC ≤ 700 K")
vlines!(ax, [1e5α_EXP]; color = :black, linewidth = 1.4, label = "experiment")
vlines!(ax, [1e5α_PUB]; color = RGBf(0.0, 0.447, 0.698), linestyle = :dash, linewidth = 1.4,
        label = "published member")
ylims!(ax, low = 0)
axislegend(ax; position = :rt, framevisible = false, labelsize = 9, patchsize = (12, 8), rowgap = 0)
save("$OUT.pdf", fig); save("$OUT.png", fig; px_per_unit = 4)
println("\nfigure → $OUT.{pdf,png}")
writedlm("$OUT.csv", vcat(["member" "alpha_1perK" "worst_minomega_THz" "fcc_le_700K" "fcc_900K"],
                          hcat([r.k for r in res], [r.α for r in res], [r.ω for r in res],
                               [r.fcc for r in res], [r.fcc900 for r in res])), ',')
println("table  → $OUT.csv")
