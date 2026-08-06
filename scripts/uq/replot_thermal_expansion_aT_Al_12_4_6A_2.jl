# replot_thermal_expansion_aT_Al_12_4_6A_2.jl
#
# Replot the two a(T) figures at MLST quality from the summary CSVs the NPT runs
# already wrote.  NO molecular dynamics is repeated: this reads
# thermal_expansion_summary.csv and rewrites thermal_expansion_aT.{pdf,png} in place,
# so the curves are the published numbers, not a re-run of the trajectories.
#
#   constrained  results/npt_multivolume_softest/                  (multi-volume line)
#   unconstrained results/npt_thermal_expansion_naive_worst_member/
#
# ── WHY α IS NOT QUOTED FOR BOTH ────────────────────────────────────────────
# A thermal-expansion coefficient is only meaningful if the crystal is still FCC at
# every temperature it was fitted over.  The unconstrained worst member is not: its
# min ω is around −11 THz throughout and its 300 K RDF shows it has left FCC, so a
# straight line through its a(T) would be a number about a solid-solid transformation,
# not about thermal expansion.  α is therefore printed only where the producing run
# recorded one AND the FCC check passed, and any non-FCC point is drawn as an open
# marker so the reader can see which temperatures are trustworthy.
#
# The constrained run records fcc_points=3/4 in its header — one temperature already
# fails.  That is shown rather than smoothed over.
#
# ── COLUMNS ─────────────────────────────────────────────────────────────────
# Parsed BY NAME from the header line, not by position: the two files do not have the
# same columns (only the constrained one carries mean_coord / median_nn_Ang /
# still_fcc), so positional indexing would silently mis-plot one of them.
#
# FIGURE SIZING.  FIGW is the width the figure will be DISPLAYED at, in points; fonts
# stay at 13/12/11 pt.  540 assumes a full \linewidth slot — if the two go side by
# side, rebuild with FIGW=260 rather than scaling in LaTeX.
#
# Run:  julia --project scripts/uq/replot_thermal_expansion_aT_Al_12_4_6A_2.jl

using DelimitedFiles, Printf, Statistics, CairoMakie

FIGW = parse(Float64, get(ENV, "FIGW", "540"))
BLU  = RGBf(0.0, 0.447, 0.698)
RED  = RGBf(0.80, 0.15, 0.15)
GRY  = RGBf(0.45, 0.45, 0.45)
TITLE, LAB, TICK, SMALL = 13, 12, 11, 10

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
    for l in lines[1:ncomment]
        for m in eachmatch(r"([A-Za-z_0-9]+)=([^\s]+)", l)
            meta[m.captures[1]] = m.captures[2]
        end
    end
    header = split(strip(lines[ncomment + 1]), ',')
    raw = readdlm(path, ','; skipstart = ncomment + 1)
    cols  = Dict{String,Vector{Float64}}()
    flags = Dict{String,Vector{Bool}}()
    for (j, name) in enumerate(header)
        col = raw[:, j]
        if eltype(col) <: AbstractString || any(x -> x isa AbstractString, col)
            s = lowercase.(strip.(string.(col)))
            all(x -> x in ("true", "false"), s) && (flags[name] = s .== "true")
        else
            cols[name] = Float64.(col)
        end
    end
    return cols, flags, meta
end

"one a(T) figure; `fcc` may be nothing when the run did not record an FCC check"
function plot_aT(cols, flags, meta, ttl, col, stem)
    T = cols["T_K"]; a = cols["a_Ang"]
    σ = haskey(cols, "a_std_Ang") ? cols["a_std_Ang"] : zeros(length(a))
    fcc = get(flags, "still_fcc", nothing)

    fig = Figure(size=(FIGW, 0.52FIGW), figure_padding=(6, 10, 4, 6))
    ax = Axis(fig[1, 1]; xlabel="Temperature (K)", ylabel="Lattice constant a (Å)",
              title=ttl, titlesize=TITLE, titlecolor=col,
              xlabelsize=LAB, ylabelsize=LAB, xticklabelsize=TICK, yticklabelsize=TICK,
              xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1,
              xticksize=4, yticksize=4)

    lines!(ax, T, a; color=(col, 0.55), linewidth=1.4)
    any(σ .> 0) && errorbars!(ax, T, a, σ; whiskerwidth=6, linewidth=1.0, color=(col, 0.8))

    # open markers where the structure is no longer FCC — those points do not belong to
    # a thermal-expansion curve even though the box still has a well-defined volume
    if fcc === nothing
        scatter!(ax, T, a; color=col, markersize=9)
    else
        ok = findall(fcc); bad = findall(.!fcc)
        isempty(ok)  || scatter!(ax, T[ok],  a[ok];  color=col, markersize=9)
        isempty(bad) || scatter!(ax, T[bad], a[bad]; color=:white, strokecolor=col,
                                 strokewidth=1.4, markersize=9)
        isempty(bad) || text!(ax, 0.03, 0.03; text="open marker: no longer FCC",
                              space=:relative, align=(:left, :bottom),
                              fontsize=SMALL, color=GRY)
    end

    # α only where the producing run recorded one and every plotted point is still FCC
    α_ok = haskey(meta, "alpha_1perK") && (fcc === nothing ? false : all(fcc))
    note = if α_ok
        α = parse(Float64, meta["alpha_1perK"])
        @sprintf("α = %.2f × 10⁻⁵ K⁻¹", 1e5α)
    elseif haskey(meta, "alpha_1perK")
        @sprintf("α = %.2f × 10⁻⁵ K⁻¹ (%s FCC — treat with care)",
                 1e5*parse(Float64, meta["alpha_1perK"]), get(meta, "fcc_points", "not all"))
    else
        "α not quoted: structure is not FCC"
    end
    text!(ax, 0.97, 0.05; text=note, space=:relative, align=(:right, :bottom),
          fontsize=SMALL, color=GRY)

    save("$stem.pdf", fig); save("$stem.png", fig; px_per_unit=4)
    @printf("  %-13s → %s.{pdf,png}\n", "figure", stem)
    return T, a, fcc
end

runs = (
 (dir = "$RESDIR/npt_multivolume_softest",
  ttl = "Constrained (multi-volume)", col = BLU,  tag = "constrained"),
 (dir = "$RESDIR/npt_thermal_expansion_naive_worst_member",
  ttl = "Unconstrained (worst member)", col = RED, tag = "unconstrained"),
)

println("── replotting a(T) at MLST sizing ($(Int(FIGW)) pt wide) ──")
for r in runs
    cols, flags, meta = read_summary("$(r.dir)/thermal_expansion_summary.csv")
    @printf("\n%s\n  %d temperatures, a ∈ [%.5f, %.5f] Å%s\n", r.ttl,
            length(cols["T_K"]), minimum(cols["a_Ang"]), maximum(cols["a_Ang"]),
            haskey(flags, "still_fcc") ? @sprintf(", FCC at %d/%d",
                count(flags["still_fcc"]), length(flags["still_fcc"])) : ", no FCC check recorded")
    haskey(meta, "alpha_1perK") && @printf("  recorded α = %s /K\n", meta["alpha_1perK"])
    haskey(meta, "fcc_points")  && @printf("  recorded fcc_points = %s\n", meta["fcc_points"])
    plot_aT(cols, flags, meta, r.ttl, r.col, "$(r.dir)/thermal_expansion_aT")
end
println("\nfilenames unchanged — existing \\includegraphics keep resolving")
