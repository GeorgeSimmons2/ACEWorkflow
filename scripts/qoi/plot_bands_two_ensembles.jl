# plot_bands_two_ensembles.jl
#
# Replot the two-ensemble phonon band figure from the .jls that
# scripts/qoi/bands_two_ensembles_Al_16_4_6A_3.jl wrote.  REPLOT ONLY — no relaxations,
# no Hessians.
#
# ── WHAT CHANGED FROM THE IN-RUN FIGURE ─────────────────────────────────────
#   * a real LEGEND identifies every line: the mean model, an ensemble member, an
#     unstable member, and the ω = 0 axis.  Previously the dashed line and the two
#     greys were unexplained.
#   * no "n/m unstable" corner text — that is a number for the caption and the CSV
#   * panel titles are just "unconstrained" / "constrained"
#
# Both panels keep the SHARED frequency axis from the original: the comparison is how
# much wider one ensemble is, and independent axes would autoscale that away.
#
# Run:  julia --project scripts/qoi/plot_bands_two_ensembles.jl
#   SRC   directory holding bands_two_ensembles.jls
#   FIGW  default 540      UNSTABLE  default from the run

using Serialization, Statistics, Printf, CairoMakie

SRC  = get(ENV, "SRC", "models/Al_16_4_6A_3_/results/pinned_ensembles")
FIGW = parse(Float64, get(ENV, "FIGW", "540"))

f = "$SRC/bands_two_ensembles.jls"
isfile(f) || error("missing $f — run scripts/qoi/bands_two_ensembles_Al_16_4_6A_3.jl first, or set SRC")
d = deserialize(f)
UNSTABLE = haskey(ENV, "UNSTABLE") ? parse(Float64, ENV["UNSTABLE"]) : d.UNSTABLE
@printf("replotting %s\n", f)

BLU = RGBf(0.0, 0.447, 0.698)
GRY = RGBAf(0.45, 0.45, 0.45, 0.40)
RED = RGBAf(0.80, 0.15, 0.15, 0.80)
TITLE, LAB, TICK = 13, 12, 11

mean_b = d.mean_b
xref = mean_b.x_vals; xt = mean_b.x_ticks; lbl = mean_b.labels; Np = mean_b.Np
ylim = d.ylim
for (k, tag) in enumerate(d.tags)
    w = [b.minω for b in d.res[k]]
    @printf("  %-14s %d members, %d soft, min ω ∈ [%+.4f, %+.4f]\n",
            tag, length(w), count(<(UNSTABLE), w), minimum(w), maximum(w))
end
@printf("  shared frequency axis [%.2f, %.2f] THz\n", ylim...)

fig = Figure(size=(FIGW, 0.46FIGW), figure_padding=(6, 10, 4, 6))
for (k, tag) in enumerate(d.tags)
    bs   = d.res[k]
    soft = [b.minω < UNSTABLE for b in bs]
    ax = Axis(fig[2, k]; xlabel="Wave vector",
              ylabel = k == 1 ? "Frequency (THz)" : "",
              title=tag, titlesize=TITLE,          # just the tag, nothing else
              xlabelsize=LAB, ylabelsize=LAB, xticklabelsize=TICK, yticklabelsize=TICK,
              xticks=(xt, lbl), xgridvisible=false, ygridvisible=false,
              xtickalign=1, ytickalign=1, xticksize=4, yticksize=4)
    # stable members first so the unstable ones are not buried under them
    for i in findall(.!soft), br in 1:3Np
        lines!(ax, xref, bs[i].F[br, :]; color=GRY, linewidth=0.7)
    end
    for i in findall(soft), br in 1:3Np
        lines!(ax, xref, bs[i].F[br, :]; color=RED, linewidth=1.1)
    end
    for br in 1:3Np; lines!(ax, xref, mean_b.F[br, :]; color=BLU, linewidth=1.8); end
    hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=1.0)
    vlines!(ax, xt; color=(:black, 0.22), linewidth=0.7)
    xlims!(ax, first(xref), last(xref)); ylims!(ax, ylim...)
    k == 1 || hideydecorations!(ax; grid=false, ticks=false, minorticks=false)
end

# ONE legend across the top.  Every line on the axes has an entry — an unlabelled
# dashed line is decoration, not information.
Legend(fig[1, 1:2],
       [LineElement(color=BLU, linewidth=2.2),
        LineElement(color=RGBf(0.45,0.45,0.45), linewidth=2.2),
        LineElement(color=RGBf(0.80,0.15,0.15), linewidth=2.2),
        LineElement(color=:black, linestyle=:dash, linewidth=1.6)],
       ["mean model", "ensemble member", "dynamically unstable", "ω = 0"];
       orientation=:horizontal, tellwidth=false, framevisible=false,
       labelsize=TICK, padding=(2,2,0,0), colgap=14)
rowgap!(fig.layout, 1, 2); colgap!(fig.layout, 20)

stem = "$SRC/bands_two_ensembles_clean"
save("$stem.pdf", fig); save("$stem.png", fig; px_per_unit=4)
@printf("\nfigure → %s.{pdf,png}\n", stem)
