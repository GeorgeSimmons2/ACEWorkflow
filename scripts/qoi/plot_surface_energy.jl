# plot_surface_energy.jl
#
# Replot the surface-energy histograms from a finished run.  REPLOT ONLY — reads the
# .jls that surface_energy_vacuum.jl wrote and redraws, so the figure can be restyled
# without repeating ~100 minimisations.
#
# Works for any model, surface and pair of ensembles: the tags, the central-model
# values and the source CSVs all come out of the .jls.
#
#   SRC      results directory holding the .jls  (default: the Al_12 surface_energy dir)
#   SURFACE  001 or 111                          (default 001)
#   NBINS    default 10
#   XCLIP    1 to clip the axis to a robust range, 0 for the full range
#
# Run:  julia --project scripts/qoi/plot_surface_energy.jl
#       SURFACE=111 julia --project scripts/qoi/plot_surface_energy.jl
#       SRC=models/Al_20_4_6A_3_/results/pinned_hypercube_rejection/surface_energy \
#         julia --project scripts/qoi/plot_surface_energy.jl

using Serialization, Statistics, Printf, DelimitedFiles, CairoMakie

SURFACE   = get(ENV, "SURFACE", "001")
SRC       = get(ENV, "SRC", "models/Al_12_4_6A_2_/results/surface_energy")
NBINS     = parse(Int,     get(ENV, "NBINS", "10"))
XCLIP     = get(ENV, "XCLIP", "1") != "0"
XCLIP_IQR = parse(Float64, get(ENV, "XCLIP_IQR", "3"))
FIGW      = parse(Float64, get(ENV, "FIGW", "540"))
C         = 16.0218                                  # eV/Å² → J/m²

BLU = RGBf(0.0, 0.447, 0.698); RED = RGBf(0.80, 0.15, 0.15)
TITLE, LAB, TICK, SMALL = 13, 12, 11, 10

# the very first 001 run predates the surface-suffixed filenames
cand = ["$SRC/surface_energy_$(SURFACE).jls", "$SRC/surface_energy.jls"]
f = findfirst(isfile, cand)
f === nothing && error("no run found: tried\n  " * join(cand, "\n  ") *
                       "\nRun scripts/qoi/surface_energy_vacuum.jl first, or set SRC.")
d = deserialize(cand[f])
@printf("replotting %s   (%s, surface %s)\n", cand[f], d.element, get(d, :SURFACE, SURFACE))

# Dicts are unordered, so fix the order explicitly: the LESS constrained ensemble goes
# left, so the figure reads failure → fix.  Falls back to sorted for unknown tags.
tags = collect(keys(d.out))
pref = ["unconstrained", "naive"]
first_tag = findfirst(t -> t in pref, tags)
ordered = first_tag === nothing ? sort(tags) :
          [tags[first_tag], only(filter(!=(tags[first_tag]), tags))]
cols = Dict(ordered[1] => RED, ordered[2] => BLU)

for t in ordered
    g = d.out[t].γ .* C
    @printf("  %-14s n = %2d, mean %+.4f, sd %.4f, range [%+.4f, %+.4f] J/m², γ≤0: %d\n",
            t, length(g), mean(g), std(g), minimum(g), maximum(g), count(<=(0), g))
end
@printf("  spread ratio = %.4f\n", std(d.out[ordered[2]].γ) / std(d.out[ordered[1]].γ))

function clip(g)
    XCLIP || return (minimum(g), maximum(g), 0)
    q1, q3 = quantile(g, 0.25), quantile(g, 0.75); iqr = q3 - q1
    iqr == 0 && return (minimum(g), maximum(g), 0)
    lo = max(q1 - XCLIP_IQR*iqr, minimum(g)); hi = min(q3 + XCLIP_IQR*iqr, maximum(g))
    return (lo, hi, count(<(lo), g) + count(>(hi), g))
end

fig = Figure(size=(FIGW, 0.42FIGW), figure_padding=(6, 10, 4, 6))
for (c, t) in enumerate(ordered)
    g = d.out[t].γ .* C
    lo, hi, off = clip(g)
    ax = Axis(fig[1, c]; xlabel="Surface energy (J/m²)", ylabel = c == 1 ? "count" : "",
              title="$t  (n = $(length(g)))", titlesize=TITLE, titlecolor=cols[t],
              xlabelsize=LAB, ylabelsize=LAB, xticklabelsize=TICK, yticklabelsize=TICK,
              xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1,
              xticksize=3, yticksize=3)
    hist!(ax, filter(x -> lo <= x <= hi, g); bins=range(lo, hi; length=NBINS+1),
          color=(cols[t], 0.65), strokecolor=:white, strokewidth=0.6)
    # central model of THIS ensemble, dashed
    ck = d.out[t].centre
    haskey(d.centre_γ, ck) && (lo <= d.centre_γ[ck]*C <= hi) &&
        vlines!(ax, [d.centre_γ[ck]*C]; color=(cols[t], 0.9), linestyle=:dash, linewidth=1.4)
    lo <= 0 <= hi && vlines!(ax, [0.0]; color=:black, linewidth=1.0)
    xlims!(ax, lo, hi)
    # The only annotation kept.  Without it a clipped axis silently hides members, which
    # matters here because the tails are the result.  Set XCLIP=0 to remove both.
    off == 0 || text!(ax, 0.96, 0.96; text="$off off scale", space=:relative,
                      align=(:right, :top), fontsize=SMALL, color=:gray45)
end
colgap!(fig.layout, 22)
stem = "$SRC/surface_energy_$(SURFACE)_hist"
save("$stem.pdf", fig); save("$stem.png", fig; px_per_unit=4)
@printf("\nside-by-side → %s.{pdf,png}\n", stem)

for t in ordered
    g = d.out[t].γ .* C
    lo, hi, off = clip(g)
    f1 = Figure(size=(0.5FIGW, 0.46FIGW), figure_padding=(4, 8, 2, 4))
    ax = Axis(f1[1, 1]; xlabel="Surface energy (J/m²)", ylabel="count",
              title="$t  (n = $(length(g)))", titlesize=TITLE, titlecolor=cols[t],
              xlabelsize=LAB, ylabelsize=LAB, xticklabelsize=TICK, yticklabelsize=TICK,
              xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1,
              xticksize=3, yticksize=3)
    hist!(ax, filter(x -> lo <= x <= hi, g); bins=range(lo, hi; length=NBINS+1),
          color=(cols[t], 0.65), strokecolor=:white, strokewidth=0.6)
    ck = d.out[t].centre
    haskey(d.centre_γ, ck) && (lo <= d.centre_γ[ck]*C <= hi) &&
        vlines!(ax, [d.centre_γ[ck]*C]; color=(cols[t], 0.9), linestyle=:dash, linewidth=1.4)
    lo <= 0 <= hi && vlines!(ax, [0.0]; color=:black, linewidth=1.0)
    xlims!(ax, lo, hi)
    off == 0 || text!(ax, 0.96, 0.96; text="$off off scale", space=:relative,
                      align=(:right, :top), fontsize=SMALL, color=:gray45)
    save("$SRC/surface_energy_$(SURFACE)_hist_$(t).pdf", f1)
    save("$SRC/surface_energy_$(SURFACE)_hist_$(t).png", f1; px_per_unit=4)
    @printf("single       → %s/surface_energy_%s_hist_%s.{pdf,png}\n", SRC, SURFACE, t)
end
