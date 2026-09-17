# replot_thermal_expansion_aT_Al_12_4_6A_2.jl
#
# Replot the two a(T) figures at MLST quality from the summary CSVs the NPT runs
# already wrote.  NO molecular dynamics is repeated: this reads
# thermal_expansion_summary.csv and rewrites thermal_expansion_aT.{pdf,png} in place,
# so the curves are the published numbers, not a re-run of the trajectories.
#
#   constrained   results/npt_multivolume_softest/                  (multi-volume line)
#   unconstrained results/npt_thermal_expansion_naive_worst_member/
#
# Scatter with error bars, one linear least-squares fit, α quoted top left.  Nothing
# else on the axes.
#
# ── WHICH POINTS ARE FITTED ─────────────────────────────────────────────────
# `fit_exclude` in the `runs` tuple below lists temperatures that are PLOTTED but not
# FITTED.  The constrained run's 900 K point is excluded: still_fcc=false and
# mean_coord 9.12 against 12 for FCC, so the structure has transformed and including it
# would turn α into a number about a solid-solid transition.  The fit line is drawn
# only across the fitted temperature range, so which points it covers is visible from
# the figure without any annotation.
#
# α is computed from THIS fit (slope / intercept), so the quoted number always matches
# the line that is drawn.  Where the producing run recorded its own alpha_1perK it is
# printed to stdout for comparison, not to the figure.
#
# ── COLUMNS ─────────────────────────────────────────────────────────────────
# Parsed BY NAME from the header line, not by position: the two files do not have the
# same columns (only the constrained one carries mean_coord / median_nn_Ang /
# still_fcc), so positional indexing would silently mis-plot one of them.
#
# FIGURE SIZING.  These two are meant to sit SIDE BY SIDE, so the default FIGW is 260 pt
# — half of the MLST \linewidth — and the panel is near-square rather than short and
# wide.  Fonts stay at 13/12/11 pt, which at 260 pt wide is large relative to the canvas:
# that is the point.  Building at 540 and letting LaTeX shrink each to 0.5\linewidth is
# what made the text tiny.  Pass FIGW=540 for a single full-width plot, and do NOT add a
# width= factor to \includegraphics either way — use \includegraphics[]{} at natural
# size in a 0.5\linewidth minipage/subfigure.
#
# Run:  julia --project scripts/uq/replot_thermal_expansion_aT_Al_12_4_6A_2.jl

using DelimitedFiles, Printf, Statistics, CairoMakie

FIGW = parse(Float64, get(ENV, "FIGW", "260"))    # half of \linewidth: side-by-side
BLU  = RGBf(0.0, 0.447, 0.698)
RED  = RGBf(0.80, 0.15, 0.15)
TITLE, LAB, TICK = 13, 12, 11

MODELDIR = abspath(joinpath(@__DIR__, "..", "..", "models", "Al_12_4_6A_2_"))
RESDIR   = "$MODELDIR/results"

"""
    read_summary(path)

Return (cols::Dict{String,Vector{Float64}}, flags::Dict{String,Vector{Bool}}, meta::Dict).
`meta` holds the `key=value` pairs from the leading `#` comment lines, which is where
the producing scripts recorded alpha_1perK and fcc_points.
"""
function read_summary(path)
    isfile(path) || error("missing $path — run the NPT driver for this member first")
    lines = readlines(path)
    ncomment = count(l -> startswith(l, "#"), lines)
    meta = Dict{String,String}()
    for l in lines[1:ncomment], m in eachmatch(r"([A-Za-z_0-9]+)=([^\s]+)", l)
        meta[m.captures[1]] = m.captures[2]
    end
    header = split(strip(lines[ncomment + 1]), ',')
    raw = readdlm(path, ','; skipstart = ncomment + 1)
    cols  = Dict{String,Vector{Float64}}()
    flags = Dict{String,Vector{Bool}}()
    for (j, name) in enumerate(header)
        col = raw[:, j]
        if any(x -> x isa AbstractString, col)
            s = lowercase.(strip.(string.(col)))
            all(x -> x in ("true", "false"), s) && (flags[name] = s .== "true")
        else
            cols[name] = Float64.(col)
        end
    end
    return cols, flags, meta
end

"ordinary least squares y = c + s·x; returns (intercept, slope)"
function linfit(x, y)
    x̄ = mean(x); ȳ = mean(y)
    s = sum((x .- x̄) .* (y .- ȳ)) / sum((x .- x̄) .^ 2)
    return ȳ - s * x̄, s
end

function plot_aT(cols, flags, meta, ttl, col, fit_exclude, stem)
    T = cols["T_K"]; a = cols["a_Ang"]
    σ = haskey(cols, "a_std_Ang") ? cols["a_std_Ang"] : zeros(length(a))

    keep = [!(t in fit_exclude) for t in T]
    a0, slope = linfit(T[keep], a[keep])
    α = slope / a0                                   # (1/a) da/dT at the T = 0 intercept

    fig = Figure(size=(FIGW, 0.92FIGW), figure_padding=(4, 8, 2, 4))
    ax = Axis(fig[1, 1]; xlabel="Temperature (K)", ylabel="Lattice constant a (Å)",
              title=ttl, titlesize=TITLE, titlecolor=col,
              xlabelsize=LAB, ylabelsize=LAB, xticklabelsize=TICK, yticklabelsize=TICK,
              xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1,
              xticksize=3, yticksize=3, xticks=0:300:900)

    # fit line spans only the fitted range, so the excluded point is visibly outside it
    Tf = [minimum(T[keep]), maximum(T[keep])]
    lines!(ax, Tf, a0 .+ slope .* Tf; color=(col, 0.85), linewidth=1.8)
    any(σ .> 0) && errorbars!(ax, T, a, σ; whiskerwidth=5, linewidth=1.0, color=(col, 0.6))
    scatter!(ax, T, a; color=col, markersize=9)

    text!(ax, 0.03, 0.97; text=@sprintf("α = %.2f × 10⁻⁵ K⁻¹", 1e5α), space=:relative,
          align=(:left, :top), fontsize=LAB, color=col)

    save("$stem.pdf", fig); save("$stem.png", fig; px_per_unit=4)
    @printf("  α = %.4e /K from the fit over %d of %d points → %s.{pdf,png}\n",
            α, count(keep), length(T), stem)
    return α
end

runs = (
 (dir = "$RESDIR/npt_multivolume_softest",
  ttl = "Constrained (multi-volume)", col = BLU,
  # 900 K: still_fcc = false, mean_coord 9.12 — transformed, so plotted but not fitted
  fit_exclude = [900.0]),
 (dir = "$RESDIR/npt_thermal_expansion_naive_worst_member",
  ttl = "Unconstrained (worst member)", col = RED,
  fit_exclude = Float64[]),
)

println("── replotting a(T) at MLST sizing ($(Int(FIGW)) pt wide) ──")
for r in runs
    cols, flags, meta = read_summary("$(r.dir)/thermal_expansion_summary.csv")
    @printf("\n%s\n  %d temperatures, a ∈ [%.5f, %.5f] Å\n", r.ttl,
            length(cols["T_K"]), minimum(cols["a_Ang"]), maximum(cols["a_Ang"]))
    if haskey(flags, "still_fcc")
        bad = cols["T_K"][.!flags["still_fcc"]]
        @printf("  FCC at %d/%d points%s\n", count(flags["still_fcc"]),
                length(flags["still_fcc"]),
                isempty(bad) ? "" : "; not FCC at " * join(Int.(bad), ", ") * " K")
    else
        println("  no FCC check recorded in this summary")
    end
    isempty(r.fit_exclude) || @printf("  excluded from the fit: %s K\n",
                                      join(Int.(r.fit_exclude), ", "))
    haskey(meta, "alpha_1perK") &&
        @printf("  run recorded α = %s /K (for comparison; the figure quotes the fit)\n",
                meta["alpha_1perK"])
    plot_aT(cols, flags, meta, r.ttl, r.col, r.fit_exclude, "$(r.dir)/thermal_expansion_aT")
end
println("\nfilenames unchanged — existing \\includegraphics keep resolving")
