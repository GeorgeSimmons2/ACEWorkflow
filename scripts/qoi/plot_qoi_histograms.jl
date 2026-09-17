# plot_qoi_histograms.jl
#
# Replot the histogram QoIs — Grüneisen, surface energy, vacancy formation — from the
# .jls each run wrote.  REPLOT ONLY: restyling costs seconds instead of repeating
# hundreds of relaxations.
#
# ── WHAT CHANGED FROM THE IN-RUN FIGURES ────────────────────────────────────
#   * a real LEGEND says what the dashed line is (that ensemble's own central model),
#     instead of leaving an unexplained line on the axes
#   * no "mean … / sd …" block and no "n/m unstable" corner text — those are numbers,
#     they belong in the caption and the CSV, not scrawled over the data
#   * no axis clipping by default, so nothing is silently hidden and no "off scale"
#     annotation is needed.  XCLIP=1 restores it, and then the count of hidden members
#     goes in the legend rather than floating in a corner.
#
# Run from the repository root:
#   QOI=gruneisen          julia --project scripts/qoi/plot_qoi_histograms.jl
#   QOI=surface_energy     SURFACE=111 julia --project scripts/qoi/plot_qoi_histograms.jl
#   QOI=vacancy_formation  julia --project scripts/qoi/plot_qoi_histograms.jl
#
#   SRC     directory holding the .jls (defaults per QoI, Al_12; set it for Al_16/Al_20)
#   NBINS   default 10        XCLIP  default 0        SHARED  default 0 (shared x-axis)
#   FIGW    default 540

using Serialization, Statistics, Printf, CairoMakie

QOI     = get(ENV, "QOI", "surface_energy")
SURFACE = get(ENV, "SURFACE", "001")
NBINS   = parse(Int,     get(ENV, "NBINS", "10"))
XCLIP   = get(ENV, "XCLIP", "0") != "0"
XCLIP_IQR = parse(Float64, get(ENV, "XCLIP_IQR", "3"))
SHARED  = get(ENV, "SHARED", "0") != "0"
FIGW    = parse(Float64, get(ENV, "FIGW", "540"))
const J = 16.0218                                  # eV/Å² → J/m²

RED = RGBf(0.80, 0.15, 0.15); BLU = RGBf(0.0, 0.447, 0.698)
TITLE, LAB, TICK = 13, 12, 11

# ── per-QoI adapters: where the file is, how to get values and central models out ──
# The three runs store different field names; everything else below is common.
ADAPTERS = Dict(
 "gruneisen" => (
   dir   = "models/Al_12_4_6A_2_/results/gruneisen",
   files = ["gruneisen.jls"],
   vals  = (d, t) -> d.out[t].γ,
   centre= (d, t) -> d.centre[d.out[t].centre].γT,
   xlab  = "Grüneisen parameter γ",
   stem  = "gruneisen"),
 "surface_energy" => (
   dir   = "models/Al_12_4_6A_2_/results/surface_energy",
   files = ["surface_energy_$(SURFACE).jls", "surface_energy.jls"],
   vals  = (d, t) -> d.out[t].γ .* J,
   centre= (d, t) -> d.centre_γ[d.out[t].centre] * J,
   xlab  = "Surface energy (J/m²)",
   stem  = "surface_energy_$(SURFACE)"),
 "vacancy_formation" => (
   dir   = "models/Al_12_4_6A_2_/results/vacancy_formation",
   files = ["vacancy_formation.jls"],
   vals  = (d, t) -> d.out[t].Ef,
   centre= (d, t) -> d.centre_Ef[d.out[t].centre],
   xlab  = "Vacancy formation energy (eV)",
   stem  = "vacancy_formation"),
)
haskey(ADAPTERS, QOI) || error("QOI must be one of " * join(sort(collect(keys(ADAPTERS))), ", "))
A   = ADAPTERS[QOI]
SRC = get(ENV, "SRC", A.dir)

hit = findfirst(f -> isfile(joinpath(SRC, f)), A.files)
hit === nothing && error("no run found in $SRC; tried " * join(A.files, ", ") *
                         "\nRun the QoI script first, or set SRC.")
d = deserialize(joinpath(SRC, A.files[hit]))
@printf("replotting %s   (%s)\n", joinpath(SRC, A.files[hit]), QOI)

# less-constrained ensemble on the left, so the figure reads failure → fix
tags = collect(keys(d.out))
pref = ["unconstrained", "naive"]
i1   = findfirst(t -> t in pref, tags)
ordered = i1 === nothing ? sort(tags) : [tags[i1], only(filter(!=(tags[i1]), tags))]
cols = Dict(ordered[1] => RED, ordered[2] => BLU)

vals = Dict(t => A.vals(d, t) for t in ordered)
cen  = Dict(t => A.centre(d, t) for t in ordered)
for t in ordered
    g = vals[t]
    @printf("  %-14s n=%2d  mean %+.4f  sd %.4f  range [%+.4f, %+.4f]  central model %+.4f\n",
            t, length(g), mean(g), std(g), minimum(g), maximum(g), cen[t])
end
@printf("  spread ratio = %.4f\n", std(vals[ordered[2]]) / std(vals[ordered[1]]))

function limits(g)
    XCLIP || return (minimum(g), maximum(g), 0)
    q1, q3 = quantile(g, 0.25), quantile(g, 0.75); iqr = q3 - q1
    iqr == 0 && return (minimum(g), maximum(g), 0)
    lo = max(q1 - XCLIP_IQR*iqr, minimum(g)); hi = min(q3 + XCLIP_IQR*iqr, maximum(g))
    return (lo, hi, count(<(lo), g) + count(>(hi), g))
end

allv = vcat((vals[t] for t in ordered)...)
glo, ghi, _ = limits(allv)

fig = Figure(size=(FIGW, 0.46FIGW), figure_padding=(6, 10, 4, 6))
# a function, not a bare loop: at top level a `for` opens a soft scope, so `n_off += off`
# would create a fresh local and reading it afterwards throws UndefVarError
function draw_panels!(fig, ordered, vals, cen, cols)
n_off = 0
for (c, t) in enumerate(ordered)
    g = vals[t]
    lo, hi, off = SHARED ? (glo, ghi, count(<(glo), g) + count(>(ghi), g)) : limits(g)
    n_off += off
    ax = Axis(fig[2, c]; xlabel=A.xlab, ylabel = c == 1 ? "count" : "",
              title=t, titlesize=TITLE, titlecolor=cols[t],
              xlabelsize=LAB, ylabelsize=LAB, xticklabelsize=TICK, yticklabelsize=TICK,
              xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1,
              xticksize=3, yticksize=3)
    hist!(ax, filter(x -> lo <= x <= hi, g); bins=range(lo, hi; length=NBINS+1),
          color=(cols[t], 0.65), strokecolor=:white, strokewidth=0.6)
    lo <= cen[t] <= hi && vlines!(ax, [cen[t]]; color=cols[t], linestyle=:dash, linewidth=1.6)
    xlims!(ax, lo, hi)
end
return n_off
end
n_off = draw_panels!(fig, ordered, vals, cen, cols)

# ONE legend, spanning both panels, saying what the dashed line is.  This is the whole
# point: a dashed line with no key is decoration, not information.
entries = [LineElement(color=:gray30, linestyle=:dash, linewidth=2)]
labels  = ["central model of each ensemble"]
if XCLIP && n_off > 0
    push!(entries, MarkerElement(color=:transparent, marker=:rect, markersize=1))
    push!(labels, "$n_off member(s) outside the plotted range")
end
Legend(fig[1, 1:2], entries, labels; orientation=:horizontal, tellwidth=false,
       framevisible=false, labelsize=TICK, padding=(2,2,0,0))
rowgap!(fig.layout, 1, 2); colgap!(fig.layout, 22)

stem = "$SRC/$(A.stem)_hist"
save("$stem.pdf", fig); save("$stem.png", fig; px_per_unit=4)
@printf("\nfigure → %s.{pdf,png}\n", stem)
