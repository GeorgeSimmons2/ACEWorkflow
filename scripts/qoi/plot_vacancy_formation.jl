# plot_vacancy_formation.jl
#
# Side-by-side vacancy formation energy histograms, unconstrained vs constrained.
# REPLOT ONLY — reads the CSVs vacancy_formation.jl wrote, runs no minimisation.
#
# ── WHY TWO INDEPENDENT X-AXES ──────────────────────────────────────────────
# The two ensembles differ in spread by a factor of ~40 (sd 64.4 eV against 1.51 eV).
# On one shared axis the constrained distribution collapses to a single spike and the
# figure says nothing except "one is wider", which the numbers already say better.
# Each panel therefore gets its own x-range, and the sd is printed IN the panel so the
# comparison is quantitative rather than visual.  The ratio belongs in the caption.
#
# ── WHY THE X-RANGE IS CLIPPED BY DEFAULT ───────────────────────────────────
# The unconstrained ensemble has a few members at −264 eV — structures that collapsed
# during relaxation.  Ten bins across that range puts every other member in one bar.
# The axis is therefore clipped to a robust range (median ± XCLIP_IQR × IQR) and the
# number of members falling outside is printed in the panel, so nothing is hidden: the
# reader is told exactly how many are off scale and in which direction.
# Set XCLIP=0 for the untruncated range.
#
# Run:  julia --project scripts/qoi/plot_vacancy_formation.jl
#   NBINS=10   XCLIP=1   XCLIP_IQR=3   FIGW=540   MODELDIR=models/Al_12_4_6A_2_

using DelimitedFiles, Statistics, Printf, Serialization, CairoMakie

NBINS     = parse(Int, get(ENV, "NBINS", "10"))
XCLIP     = get(ENV, "XCLIP", "1") != "0"
XCLIP_IQR = parse(Float64, get(ENV, "XCLIP_IQR", "3"))
FIGW      = parse(Float64, get(ENV, "FIGW", "540"))
MODELDIR  = get(ENV, "MODELDIR", "models/Al_12_4_6A_2_")
SRC       = get(ENV, "SRC", "$MODELDIR/results/vacancy_formation")

BLU = RGBf(0.0, 0.447, 0.698); RED = RGBf(0.80, 0.15, 0.15)
TITLE, LAB, TICK, SMALL = 13, 12, 11, 10

# column 2 of the CSV is E_f from the full per-member relaxation
function read_Ef(tag)
    f = "$SRC/vacancy_formation_$(tag).csv"
    isfile(f) || error("missing $f — run scripts/qoi/vacancy_formation.jl first")
    return Float64.(readdlm(f, ',')[:, 2])
end

mean_Ef = let f = "$SRC/vacancy_formation.jls"
    isfile(f) ? deserialize(f).mean_Ef : nothing
end

panels = [(tag = "unconstrained", label = "Unconstrained POPS", col = RED),
          (tag = "constrained",   label = "Constrained POPS",   col = BLU)]
data = Dict(p.tag => read_Ef(p.tag) for p in panels)

println("── vacancy formation histograms ──")
for p in panels
    E = data[p.tag]
    @printf("%-14s n = %d, mean %+.4f, sd %.4f, range [%+.4f, %+.4f] eV\n",
            p.tag, length(E), mean(E), std(E), minimum(E), maximum(E))
end
@printf("spread ratio σ_constrained / σ_unconstrained = %.3f\n",
        std(data["constrained"]) / std(data["unconstrained"]))
mean_Ef === nothing || @printf("mean model E_f = %+.4f eV\n", mean_Ef)

"robust axis limits; returns (lo, hi, n_below, n_above)"
function clip_range(E)
    XCLIP || return (minimum(E), maximum(E), 0, 0)
    q1, q3 = quantile(E, 0.25), quantile(E, 0.75)
    iqr = q3 - q1
    iqr == 0 && return (minimum(E), maximum(E), 0, 0)
    lo, hi = q1 - XCLIP_IQR*iqr, q3 + XCLIP_IQR*iqr
    lo = max(lo, minimum(E)); hi = min(hi, maximum(E))
    return (lo, hi, count(<(lo), E), count(>(hi), E))
end

function panel!(gp, p; showy=true)
    E = data[p.tag]
    lo, hi, nlo, nhi = clip_range(E)
    edges = range(lo, hi; length=NBINS+1)
    ax = Axis(gp; xlabel="Vacancy formation energy (eV)",
              ylabel = showy ? "count" : "",
              title="$(p.label)  (n = $(length(E)))", titlesize=TITLE, titlecolor=p.col,
              xlabelsize=LAB, ylabelsize=LAB, xticklabelsize=TICK, yticklabelsize=TICK,
              xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1,
              xticksize=3, yticksize=3)
    inb = filter(x -> lo <= x <= hi, E)
    hist!(ax, inb; bins=edges, color=(p.col, 0.65), strokecolor=:white, strokewidth=0.6)
    mean_Ef === nothing || (lo <= mean_Ef <= hi &&
        vlines!(ax, [mean_Ef]; color=:black, linestyle=:dash, linewidth=1.2))
    xlims!(ax, lo, hi)

    text!(ax, 0.04, 0.96; text=@sprintf("mean %+.2f\nsd %.2f", mean(E), std(E)),
          space=:relative, align=(:left, :top), fontsize=SMALL, color=p.col)
    off = nlo + nhi
    off == 0 || text!(ax, 0.96, 0.96;
                      text=@sprintf("%d off scale\n(%d below, %d above)", off, nlo, nhi),
                      space=:relative, align=(:right, :top), fontsize=SMALL, color=:gray40)
    return ax
end

fig = Figure(size=(FIGW, 0.42FIGW), figure_padding=(6, 10, 4, 6))
panel!(fig[1, 1], panels[1])
panel!(fig[1, 2], panels[2]; showy=false)
colgap!(fig.layout, 22)
save("$SRC/vacancy_formation_hist.pdf", fig)
save("$SRC/vacancy_formation_hist.png", fig; px_per_unit=4)
@printf("\nside-by-side → %s/vacancy_formation_hist.{pdf,png}\n", SRC)

# each panel alone, for independent placement at half width
for p in panels
    f1 = Figure(size=(0.5FIGW, 0.46FIGW), figure_padding=(4, 8, 2, 4))
    panel!(f1[1, 1], p)
    save("$SRC/vacancy_formation_hist_$(p.tag).pdf", f1)
    save("$SRC/vacancy_formation_hist_$(p.tag).png", f1; px_per_unit=4)
    @printf("single       → %s/vacancy_formation_hist_%s.{pdf,png}\n", SRC, p.tag)
end
