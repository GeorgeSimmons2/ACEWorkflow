# plot_parity_simple.jl
#
# A single energy parity panel from plain REPL arrays — for when you have
#
#     true_Es :: Vector          DFT energy per atom
#     Es      :: Vector          point-model prediction per atom
#     co_Es   :: Vector{Vector}  one committee vector per configuration
#
# and want the figure without assembling a full committee_predictions NamedTuple
# (which also wants forces).  Units are stripped defensively, so Unitful Quantities
# straight out of `potential_energy` work as-is.
#
# Conventions match the tidied QoI plots: MLST fonts at final display width, a legend
# that names the dashed line, and no floating "mean …" or corner annotations — RMSE and
# coverage live in the title.
#
# Use:
#   include("scripts/uq/plot_parity_simple.jl")
#   parity_simple(true_Es, Es, co_Es; path="parity.pdf")
#
# Included with those three names already in the session, it plots them automatically.

using CairoMakie, Statistics, Printf, Unitful

_strip(x::Number)      = Float64(ustrip(x))
_strip(x::AbstractArray) = Float64.(ustrip.(x))

"""
    parity_simple(true_E, pred_E, committee_E; path, label, FIGW)

`committee_E[i]` is the vector of committee predictions for configuration `i`.
Error bars span the committee min/max; the point is the central model.
"""
function parity_simple(true_E, pred_E, committee_E;
                       path::AbstractString = "parity_energy.pdf",
                       label::AbstractString = "",
                       xlabel::AbstractString = "DFT energy (eV/atom)",
                       ylabel::AbstractString = "ACE energy (eV/atom)",
                       FIGW::Real = 380)
    t  = _strip(true_E); p = _strip(pred_E)
    co = [_strip(c) for c in committee_E]
    length(t) == length(p) == length(co) ||
        error("lengths differ: $(length(t)) truth, $(length(p)) predictions, $(length(co)) committees")
    lo = [minimum(c) for c in co]; hi = [maximum(c) for c in co]

    rmse = sqrt(mean((p .- t).^2))
    cov  = 100*(1 - mean((t .< lo) .| (t .> hi)))
    BLU  = RGBf(0.0, 0.447, 0.698)
    TITLE, LAB, TICK = 13, 12, 11

    fig = Figure(size=(FIGW, 1.05FIGW), figure_padding=(6, 10, 4, 6))
    ax = Axis(fig[2, 1]; xlabel, ylabel,
              title = @sprintf("RMSE %.3g meV/atom   |   coverage %.1f%%", 1000rmse, cov),
              titlesize=TITLE, xlabelsize=LAB, ylabelsize=LAB,
              xticklabelsize=TICK, yticklabelsize=TICK,
              xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1, aspect=1)
    l = extrema(vcat(t, p, lo, hi))
    lines!(ax, collect(l), collect(l); color=:black, linestyle=:dash, linewidth=1.2)
    errorbars!(ax, t, p, p .- lo, hi .- p; whiskerwidth=4, linewidth=0.7, color=(BLU, 0.35))
    scatter!(ax, t, p; color=(BLU, 0.9), markersize=5)
    xlims!(ax, l...); ylims!(ax, l...)

    Legend(fig[1, 1],
           [LineElement(color=:black, linestyle=:dash, linewidth=2),
            MarkerElement(color=BLU, marker=:circle, markersize=8),
            LineElement(color=(BLU, 0.5), linewidth=6)],
           ["perfect agreement", "central model", "committee range"];
           orientation=:horizontal, tellwidth=false, framevisible=false,
           labelsize=TICK, padding=(2,2,0,0), colgap=12)
    isempty(label) || Label(fig[0, 1], label; fontsize=TICK, padding=(0,0,0,2))
    rowgap!(fig.layout, 1, 2)

    stem = replace(String(path), r"\.(pdf|png)$" => "")
    save("$stem.pdf", fig); save("$stem.png", fig; px_per_unit=4)
    @printf("RMSE %.4g meV/atom, coverage %.2f%%, %d configs → %s.{pdf,png}\n",
            1000rmse, cov, length(t), stem)
    return fig
end

# convenience: if the three arrays are already in the session, just plot them
if (@isdefined true_Es) && (@isdefined Es) && (@isdefined co_Es)
    parity_simple(true_Es, Es, co_Es;
                  path = get(ENV, "PARITY_OUT", "parity_energy"),
                  label = get(ENV, "PARITY_LABEL", ""))
end
