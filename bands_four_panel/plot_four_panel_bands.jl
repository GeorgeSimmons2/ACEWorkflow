# plot_four_panel_bands.jl
#
# One 2×2 figure: unconstrained POPS (left) vs constrained POPS (right), for the
# smaller model (top row) and the larger one (bottom row).  Replaces the three separate
# figures it is assembled from:
#
#   models/Al_12_4_6A_2_/results/naive_vs_constrained/bands_naive_shared_axis.pdf   → (1,1)
#   models/Al_12_4_6A_2_/results/naive_vs_constrained/bands_constrained_shared_axis.pdf → (1,2)
#   models/Al_16_4_6A_3_/results/pinned_ensembles/bands_two_ensembles_clean.pdf     → (2,1),(2,2)
#
# The mean model is blue; EVERY ensemble member is grey, whether or not it is
# dynamically stable.  Colouring the unstable ones crimson pre-labels the conclusion —
# the point of the figure is that you can see which members dip below ω = 0 without
# being told, and that the right column has none.  The counts are in the printout and
# belong in the caption.
#
# ── PURE REPLOT ─────────────────────────────────────────────────────────────
# Reads serialised band curves only; no Hessians, no relaxations, no sampling.  The
# Al_16 curves come straight from its run.  The Al_12 run saved only summary
# statistics, so its curves have to be cached once first:
#
#   julia --project -t 40 bands_four_panel/build_bands_cache_Al_12.jl
#
# ── THE TWO ROWS DO NOT SHARE A FREQUENCY AXIS ──────────────────────────────
# Within a row they must: unconstrained vs constrained is the comparison the figure
# makes, and independent axes would autoscale away exactly the width difference being
# claimed.  Across rows they must not: Al_12's unconstrained members reach ≈ −23 THz
# against Al_16's ≈ −7, so one global axis squeezes the whole Al_16 row into the middle
# third of its panels and hides its structure.  SHARE_Y=1 forces a global axis anyway.
#
# ── THE X-AXIS IS A REMAPPED PATH COORDINATE ────────────────────────────────
# The two models relax to different lattice constants, so their reciprocal lattices —
# and hence the Cartesian lengths of Γ→X, X→U, … — differ by ~0.3%.  Drawn raw, the
# high-symmetry ticks would not line up between rows, which in a column-aligned figure
# reads as an error.  Each row's path coordinate is therefore mapped piecewise-linearly
# onto the mean of the two tick vectors: exact at every high-symmetry point, monotone
# in between, and the maximum displacement is printed so it can be audited.  The
# distortion is far below a line width; if it ever is not, the printout says so.
#
# Within a row, every member is drawn against that row's REFERENCE path (the mean
# model's).  On the unconstrained side each member has its own lattice constant and so
# its own reciprocal lattice, but N_per_seg is identical, so the paths correspond
# point-for-point between the same pair of high-symmetry points.  This is the same
# convention the source scripts used; the assertion below catches any mismatch.
#
# Run:  julia --project bands_four_panel/plot_four_panel_bands.jl
#   SRC12/SRC16  input .jls        OUT  output stem (default bands_four_panel/bands_four_panel)
#   FIGW  540 (points, = the width the figure is DISPLAYED at in the paper)
#   ASPECT 0.78    SHARE_Y 0    LETTERS 0    UNSTABLE  from the runs

using Serialization, Statistics, Printf, CairoMakie

ROOT    = normpath(joinpath(@__DIR__, ".."))
SRC12   = get(ENV, "SRC12", "$ROOT/models/Al_12_4_6A_2_/results/naive_vs_constrained/bands_four_panel_Al_12.jls")
SRC16   = get(ENV, "SRC16", "$ROOT/models/Al_16_4_6A_3_/results/pinned_ensembles/bands_two_ensembles.jls")
OUT     = get(ENV, "OUT", joinpath(@__DIR__, "bands_four_panel"))
FIGW    = parse(Float64, get(ENV, "FIGW", "540"))
ASPECT  = parse(Float64, get(ENV, "ASPECT", "0.78"))
SHARE_Y = get(ENV, "SHARE_Y", "0") != "0"
LETTERS = get(ENV, "LETTERS", "0") != "0"

BLU  = RGBf(0.0, 0.447, 0.698)
GRY  = RGBAf(0.45, 0.45, 0.45, 0.38)      # every member, stable or not
TITLE, LAB, TICK = 13, 12, 11

isfile(SRC12) || error("""
    missing $SRC12
    The Al_12 run saved only summary statistics, not the band curves.  Build the cache:
        julia --project -t 40 bands_four_panel/build_bands_cache_Al_12.jl""")
isfile(SRC16) || error("missing $SRC16 — run scripts/qoi/bands_two_ensembles_Al_16_4_6A_3.jl, or set SRC16")

d12 = deserialize(SRC12); d16 = deserialize(SRC16)

# `cen` (per-panel central model) exists only in the Al_12 cache: its two panels are
# centred on different models — lin_params at its own geometry on the left, θ_mean on
# the right.  Al_16 pins both ensembles to one mean model, so mean_b serves both.
centre(d, k) = haskey(d, :cen) ? d.cen[k] : d.mean_b

# named for the models, not "small/large": "Al_12" is the polynomial degree, so a
# subscript would misread as a 12-atom cluster.  The parameter count is the size claim.
rows = [(d = d12, label = get(ENV, "ROW1", "Al_12_4_6A_2_  (91 params)")),
        (d = d16, label = get(ENV, "ROW2", "Al_16_4_6A_3_  (684 params)"))]

for (i, r) in enumerate(rows)
    r.d.tags == ["unconstrained", "constrained"] ||
        error("row $i tags are $(r.d.tags); the figure assumes [unconstrained, constrained]")
end
lbl = rows[1].d.mean_b.labels
all(r -> r.d.mean_b.labels == lbl, rows) ||
    error("the two rows use different band paths: $(rows[1].d.mean_b.labels) vs $(rows[2].d.mean_b.labels)")

UNSTABLE = haskey(ENV, "UNSTABLE") ? parse(Float64, ENV["UNSTABLE"]) : rows[1].d.UNSTABLE
for (i, r) in enumerate(rows), (k, tag) in enumerate(r.d.tags)
    w = [b.minω for b in r.d.res[k]]
    @printf("row %d %-14s %2d members, %2d soft (< %.2f THz), min ω ∈ [%+.4f, %+.4f]\n",
            i, tag, length(w), count(<(UNSTABLE), w), UNSTABLE, minimum(w), maximum(w))
end

# ── x: map each row's path coordinate onto the mean tick vector ─────────────
tickmats = [r.d.mean_b.x_ticks for r in rows]
xt = mean(tickmats)
function remap(x, from, to)
    y = similar(x)
    for (i, xi) in enumerate(x)
        j = clamp(searchsortedlast(from, xi), 1, length(from) - 1)
        t = (xi - from[j]) / (from[j+1] - from[j])
        y[i] = to[j] + t * (to[j+1] - to[j])
    end
    return y
end
xr = [remap(r.d.mean_b.x_vals, r.d.mean_b.x_ticks, xt) for r in rows]
@printf("path coordinate: rows differ by ≤ %.4f (%.3f%% of the path) before remapping\n",
        maximum(abs.(tickmats[1] .- tickmats[2])),
        100 * maximum(abs.(tickmats[1] .- tickmats[2])) / xt[end])

# ── y: shared within a row, independent between rows unless SHARE_Y ────────
ylims = [r.d.ylim for r in rows]
if SHARE_Y
    g = (minimum(y[1] for y in ylims), maximum(y[2] for y in ylims))
    ylims = [g, g]
    @printf("shared frequency axis over all four panels: [%.2f, %.2f] THz\n", g...)
else
    for (i, y) in enumerate(ylims)
        @printf("row %d frequency axis: [%.2f, %.2f] THz\n", i, y...)
    end
end

# ── figure ──────────────────────────────────────────────────────────────────
fig = Figure(size = (FIGW, ASPECT * FIGW), figure_padding = (6, 10, 4, 6))
letters = ["(a)" "(b)"; "(c)" "(d)"]

for i in 1:2, k in 1:2
    r = rows[i]; bs = r.d.res[k]; Np = r.d.mean_b.Np
    bottom = i == 2
    ax = Axis(fig[i + 1, k];
              xlabel = bottom ? "Wave vector" : "",
              ylabel = k == 1 ? "Frequency (THz)" : "",
              title  = i == 1 ? r.d.tags[k] : "", titlesize = TITLE,
              xlabelsize = LAB, ylabelsize = LAB,
              xticklabelsize = TICK, yticklabelsize = TICK,
              xticks = (xt, lbl), xgridvisible = false, ygridvisible = false,
              xtickalign = 1, ytickalign = 1, xticksize = 4, yticksize = 4)
    size(bs[1].F, 2) == length(xr[i]) || error(
        "row $i $(r.d.tags[k]): members have $(size(bs[1].F,2)) q-points but the row's path " *
        "has $(length(xr[i])) — the ensembles were not computed on the same N_per_seg")
    for b in bs, br in 1:3Np
        lines!(ax, xr[i], b.F[br, :]; color = GRY, linewidth = 0.8)
    end
    c = centre(r.d, k)
    for br in 1:3Np; lines!(ax, xr[i], c.F[br, :]; color = BLU, linewidth = 1.8); end
    hlines!(ax, [0.0]; color = :black, linestyle = :dash, linewidth = 1.0)
    vlines!(ax, xt; color = (:black, 0.22), linewidth = 0.7)
    xlims!(ax, first(xt), last(xt)); ylims!(ax, ylims[i]...)
    k == 1 || hideydecorations!(ax; grid = false, ticks = false, minorticks = false)
    bottom  || hidexdecorations!(ax; grid = false, ticks = false, minorticks = false)
    LETTERS && text!(ax, 0.025, 0.97; text = letters[i, k], space = :relative,
                     align = (:left, :top), font = :bold, fontsize = TITLE)
end

# row labels down the left-hand side — with the column titles these identify every
# panel, so no per-panel title text is needed
for i in 1:2
    Label(fig[i + 1, 0], rows[i].label; rotation = π/2, fontsize = LAB, tellheight = false)
end

# ONE legend across the top.  Every line on the axes has an entry.
Legend(fig[1, 1:2],
       [LineElement(color = BLU, linewidth = 2.2),
        LineElement(color = RGBf(0.45, 0.45, 0.45), linewidth = 2.2),
        LineElement(color = :black, linestyle = :dash, linewidth = 1.6)],
       ["mean model", "ensemble member", "ω = 0"];
       orientation = :horizontal, tellwidth = false, framevisible = false,
       labelsize = TICK, padding = (2, 2, 0, 0), colgap = 16)

rowgap!(fig.layout, 1, 2); rowgap!(fig.layout, 2, 8)
colgap!(fig.layout, 1, 4); colgap!(fig.layout, 2, 16)

mkpath(dirname(OUT))
save("$OUT.pdf", fig); save("$OUT.png", fig; px_per_unit = 4)
@printf("\nfigure → %s.{pdf,png}\n", OUT)
