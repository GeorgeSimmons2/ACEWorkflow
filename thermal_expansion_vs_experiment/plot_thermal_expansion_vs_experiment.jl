# plot_thermal_expansion_vs_experiment.jl
#
# a(T) for both NPT runs on ONE set of axes, against Wilson's 1941 x-ray measurement.
# Replaces the two separate figures:
#
#   models/Al_12_4_6A_2_/results/npt_thermal_expansion_naive_worst_member/thermal_expansion_aT.png
#   models/Al_12_4_6A_2_/results/npt_multivolume_softest/thermal_expansion_aT.png
#
#   red    unconstrained POPS, worst member
#   blue   constrained POPS, softest member (multi-volume)
#   black  experiment — A. J. C. Wilson, Proc. Phys. Soc. 53 (1941) 235
#
# NO molecular dynamics is repeated.  Both model series are read from the
# thermal_expansion_summary.csv files the NPT runs already wrote, so the points are
# the published numbers.
#
# ── THE EXPERIMENTAL DATA NEEDS A UNIT CONVERSION, AND IT MATTERS ───────────
# Wilson's table is in kX (Siegbahn x-ray) units, though the paper writes "A."
# throughout — the rename to kX came in 1947, precisely because of this confusion.
# Plotted raw, the experimental curve would sit ~0.008 Å low and the models would look
# systematically over-expanded by an amount that is pure unit error.  See the header of
# wilson_1941_aluminium.csv for the two independent confirmations that these are kX.
#
# The conversion factor itself is not perfectly settled — values between 1.00202
# (the Siegbahn-scale figure) and ~1.00208 (Bearden's redetermination) are in use.  The
# spread is 2×10⁻⁴ Å, which is an order of magnitude below the smallest feature in this
# figure and comparable to Wilson's own quoted error, so the choice does not affect any
# conclusion drawn here.  The script prints the converted 25 °C spacing against the
# accepted 4.0495 Å so the conversion can be checked at a glance, and KX_TO_ANG
# overrides it.
#
# ── WHICH POINTS ARE FITTED, AND OVER WHAT RANGE ────────────────────────────
# α for aluminium is NOT constant — Wilson measures it rising from 22 to 37 ×10⁻⁶ K⁻¹
# across his range — so a single linear α is a range average, and two αs are comparable
# only if they came from the same temperature window.  Putting three series on one axes
# makes that the central issue, so all three windows are always computed and printed:
#
#   native   each series' own published convention.  These are the numbers the two
#            figures this replaces quote, kept so they stay checkable.
#   common   FIT_MIN–FIT_MAX (default 0–700 K) for everything.  THE DEFAULT: it uses
#            every physically valid model point, and each fit line then spans nearly all
#            of its own data rather than a short central segment.  Caveat, stated rather
#            than hidden: the models' T = 0 point is a STATIC lattice constant with no
#            thermal motion at all, and the experiment has no counterpart to it, so the
#            low end of the window is not strictly like-for-like.
#   overlap  273–700 K, the range all three series actually cover — the strictly
#            like-for-like comparison.  Independently, it reproduces the α the
#            constrained NPT run recorded for itself (2.631e-5) to four figures.
#
# The choice does not change the conclusion.  common gives 6.1 / 2.7 / 2.7 ×10⁻⁵ K⁻¹ for
# unconstrained / constrained / experiment; overlap gives 3.7 / 2.6 / 2.7.  Either way
# the constrained member lands on the measurement and the unconstrained one does not —
# and no straight line describes the unconstrained series at all, since its a(T) is not
# even monotonic (4.167 Å at 300 K, 4.139 Å at 500 K).
#
# Independent of the window, the constrained run's 900 K point is PLOTTED but never
# FITTED: still_fcc=false and mean coordination 9.12 against 12 for FCC, so the structure
# has transformed and fitting it would turn α into a number about a solid-solid
# transition.  Each fit line is drawn only across its own fitted range, so which points
# it covers is visible without annotation.
#
# ── SIZING ──────────────────────────────────────────────────────────────────
# FIGW is in POINTS and must equal the width the figure is DISPLAYED at in the paper.
# Building at 540 and letting LaTeX shrink it is what made the text tiny before.  The
# default 380 pt suits a ~0.7\linewidth placement; use \includegraphics[]{} at natural
# size, with no width= factor.
#
# Run:  julia --project thermal_expansion_vs_experiment/plot_thermal_expansion_vs_experiment.jl
#   FIGW 380   ASPECT 0.82   WIN overlap|common|native   FIT_MIN 0   FIT_MAX 700
#   KX_TO_ANG 1.00202
#   RESDIR  models/Al_12_4_6A_2_/results   WILSON  the transcribed table
#   OUT  thermal_expansion_vs_experiment/thermal_expansion_aT_vs_experiment

using DelimitedFiles, Printf, Statistics, CairoMakie

ROOT      = normpath(joinpath(@__DIR__, ".."))
# RESDIR is overridable so the figure can be built against a checkout that holds the
# model outputs when the script itself lives somewhere else (e.g. a git worktree)
RESDIR    = get(ENV, "RESDIR", joinpath(ROOT, "models", "Al_12_4_6A_2_", "results"))
WILSON    = get(ENV, "WILSON", joinpath(@__DIR__, "wilson_1941_aluminium.csv"))
OUT       = get(ENV, "OUT", joinpath(@__DIR__, "thermal_expansion_aT_vs_experiment"))
FIGW      = parse(Float64, get(ENV, "FIGW", "380"))
ASPECT    = parse(Float64, get(ENV, "ASPECT", "0.82"))
WIN       = get(ENV, "WIN", "common")
FIT_MIN   = parse(Float64, get(ENV, "FIT_MIN", "0"))
FIT_MAX   = parse(Float64, get(ENV, "FIT_MAX", "700"))
OVERLAP   = (273.15, 700.0)   # the window all three series actually share
KX_TO_ANG = parse(Float64, get(ENV, "KX_TO_ANG", "1.00202"))
WIN in ("native", "common", "overlap") ||
    error("WIN must be native, common or overlap, got $WIN")

BLU = RGBf(0.0, 0.447, 0.698)
RED = RGBf(0.80, 0.15, 0.15)
BLK = RGBf(0.0, 0.0, 0.0)
LAB, TICK = 12, 11
A_REF_25C = 4.0495          # accepted lattice constant of Al at 25 °C, Å — conversion check

# ── readers ─────────────────────────────────────────────────────────────────
"""
    read_summary(path)

Return (cols, flags, meta) from an NPT thermal_expansion_summary.csv.  Columns are
parsed BY NAME from the header, not by position: the two runs do not carry the same
columns (only the constrained one has mean_coord / still_fcc), so positional indexing
would silently mis-plot one of them.  `meta` holds the key=value pairs from the leading
`#` lines, where the runs recorded alpha_1perK.
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
        # readdlm parses bare true/false into Bool, so test for that FIRST — otherwise
        # still_fcc silently becomes a 1.0/0.0 float column and the FCC check is lost
        if all(x -> x isa Bool, col)
            flags[name] = Bool.(col)
        elseif any(x -> x isa AbstractString, col)
            s = lowercase.(strip.(string.(col)))
            all(x -> x in ("true", "false"), s) && (flags[name] = s .== "true")
        else
            cols[name] = Float64.(col)
        end
    end
    return cols, flags, meta
end

function read_wilson(path)
    isfile(path) || error("missing $path — the transcribed Wilson 1941 table")
    lines  = readlines(path)
    ncom   = count(l -> startswith(l, "#"), lines)
    header = split(strip(lines[ncom + 1]), ',')
    raw    = readdlm(path, ','; skipstart = ncom + 1)
    col(n) = Float64.(raw[:, findfirst(==(n), header)])
    T_C  = col("T_C")
    a_kX = col("a_kX_obs")
    return (T_K = T_C .+ 273.15, a = a_kX .* KX_TO_ANG, a_kX = a_kX, T_C = T_C,
            α_eq2 = col("alpha_1e6_eq2"), α_pref = col("alpha_1e6_preferred"))
end

"ordinary least squares y = c + s·x; returns (intercept, slope)"
function linfit(x, y)
    x̄ = mean(x); ȳ = mean(y)
    s = sum((x .- x̄) .* (y .- ȳ)) / sum((x .- x̄) .^ 2)
    return ȳ - s * x̄, s
end

"α = (1/a₀)·da/dT from a straight-line fit to the kept points, plus the fitted range"
function fit_alpha(T, a, keep)
    a0, slope = linfit(T[keep], a[keep])
    return (α = slope / a0, a0 = a0, slope = slope,
            Tlo = minimum(T[keep]), Thi = maximum(T[keep]), n = count(keep))
end

# ── assemble the three series ───────────────────────────────────────────────
un_cols, un_flags, un_meta = read_summary("$RESDIR/npt_thermal_expansion_naive_worst_member/thermal_expansion_summary.csv")
co_cols, co_flags, co_meta = read_summary("$RESDIR/npt_multivolume_softest/thermal_expansion_summary.csv")
w = read_wilson(WILSON)

@printf("kX → Å conversion: ×%.5f\n", KX_TO_ANG)
@printf("  Wilson's 25 °C spacing %.5f kX → %.5f Å   (accepted %.4f Å, Δ = %+.5f Å)\n",
        w.a_kX[2], w.a[2], A_REF_25C, w.a[2] - A_REF_25C)
abs(w.a[2] - A_REF_25C) < 5e-3 ||
    @warn "converted 25 °C spacing is $(w.a[2]) Å against an accepted $A_REF_25C Å — check KX_TO_ANG"
@printf("  Wilson's own α rises %.1f → %.1f ×10⁻⁶ K⁻¹ over 0–650 °C: α is NOT constant,\n",
        w.α_pref[1], w.α_pref[end])
println("  so a single linear α is a range average and only comparable on a common window.")

# `native` reproduces the fit convention of the two figures this replaces, so the α
# values printed on the figure match the ones already in the paper.
series = [
 (name = "unconstrained", col = RED,  marker = :rect,    T = un_cols["T_K"], a = un_cols["a_Ang"],
  σ = get(un_cols, "a_std_Ang", zeros(length(un_cols["T_K"]))),
  exclude = Float64[], flags = un_flags, meta = un_meta),
 (name = "constrained",   col = BLU,  marker = :circle,  T = co_cols["T_K"], a = co_cols["a_Ang"],
  σ = get(co_cols, "a_std_Ang", zeros(length(co_cols["T_K"]))),
  # 900 K: still_fcc = false, mean coordination 9.12 — transformed, plotted but not fitted
  exclude = [900.0], flags = co_flags, meta = co_meta),
 (name = "experiment (Wilson 1941)",  col = BLK, marker = :diamond, T = w.T_K, a = w.a,
  # systematic and random errors each ~1e-4 Å ⇒ ~1.4e-4 Å combined; invisible at this scale
  σ = fill(1.4e-4, length(w.T_K)), exclude = Float64[],
  flags = Dict{String,Vector{Bool}}(), meta = Dict{String,String}()),
]

keep_native(s)  = [!(t in s.exclude) for t in s.T]
keep_common(s)  = [!(t in s.exclude) && FIT_MIN <= t <= FIT_MAX for t in s.T]
keep_overlap(s) = [!(t in s.exclude) && OVERLAP[1] <= t <= OVERLAP[2] for t in s.T]

# Three windows, always all three printed, because α for aluminium is temperature
# dependent and a single number only means something alongside the range it came from:
#   native   each series' own published convention — what the two figures this replaces
#            quote, so those numbers stay checkable
#   common   FIT_MIN–FIT_MAX for everything; the models' T = 0 point is a STATIC lattice
#            constant with no thermal motion, and the experiment has no counterpart to
#            it, so this window still is not quite like-for-like
#   overlap  273–700 K, the range all three series actually cover — the only genuinely
#            like-for-like comparison, and the one to quote when comparing to experiment
function report(series)
    fits = []
    for s in series
        f(keep) = count(keep) >= 2 ? fit_alpha(s.T, s.a, keep) : nothing
        fn = f(keep_native(s)); fc = f(keep_common(s)); fo = f(keep_overlap(s))
        push!(fits, (native = fn, common = fc, overlap = fo))
        @printf("\n%s\n  %d points, a ∈ [%.5f, %.5f] Å over %.0f–%.0f K\n",
                s.name, length(s.T), minimum(s.a), maximum(s.a), minimum(s.T), maximum(s.T))
        if haskey(s.flags, "still_fcc")
            bad = s.T[.!s.flags["still_fcc"]]
            @printf("  FCC at %d/%d points%s\n", count(s.flags["still_fcc"]),
                    length(s.flags["still_fcc"]),
                    isempty(bad) ? "" : "; NOT FCC at " * join(Int.(bad), ", ") * " K")
        end
        isempty(s.exclude) || @printf("  excluded from the fit: %s K\n",
                                      join(Int.(s.exclude), ", "))
        for (tag, fit) in (("native", fn), ("common", fc), ("overlap", fo))
            fit === nothing ?
                @printf("  %-8s window: fewer than 2 points, not fitted\n", tag) :
                @printf("  %-8s window %.0f–%.0f K (%d pts): α = %.3e /K\n",
                        tag, fit.Tlo, fit.Thi, fit.n, fit.α)
        end
        haskey(s.meta, "alpha_1perK") &&
            @printf("  the producing run recorded α = %s /K\n", s.meta["alpha_1perK"])
    end
    return fits
end

println("\n── a(T): two NPT members against Wilson 1941 ──")
fits = report(series)
function pick(f, win)
    g = win == "native" ? f.native : win == "common" ? f.common : f.overlap
    return g === nothing ? f.native : g       # fall back rather than fail on a short series
end
chosen = [pick(f, WIN) for f in fits]
@printf("\nfigure quotes the %s window: %s\n", WIN,
        join([@sprintf("%.3e", c.α) for c in chosen], ", "))

# ── figure ──────────────────────────────────────────────────────────────────
fig = Figure(size = (FIGW, ASPECT * FIGW), figure_padding = (4, 10, 2, 4))
ax = Axis(fig[2, 1]; xlabel = "Temperature (K)", ylabel = "Lattice constant a (Å)",
          xlabelsize = LAB, ylabelsize = LAB,
          xticklabelsize = TICK, yticklabelsize = TICK,
          xgridvisible = false, ygridvisible = false,
          xtickalign = 1, ytickalign = 1, xticksize = 3, yticksize = 3,
          xticks = 0:300:900)

function draw!(ax, series, chosen)
    handles = []; labels = String[]
    for (s, f) in zip(series, chosen)
        # the fit line spans only the fitted range, so an excluded point sits visibly
        # outside it and needs no annotation
        Tf = [f.Tlo, f.Thi]
        lines!(ax, Tf, f.a0 .+ f.slope .* Tf; color = (s.col, 0.85), linewidth = 1.6)
        any(s.σ .> 0) && errorbars!(ax, s.T, s.a, s.σ;
                                    whiskerwidth = 5, linewidth = 1.0, color = (s.col, 0.6))
        h = scatter!(ax, s.T, s.a; color = s.col, marker = s.marker, markersize = 8)
        push!(handles, h)
        push!(labels, s.name)
    end
    # α top left, one coloured line per series, in the only corner the data leaves empty.
    # Kept out of the legend so the legend stays narrow enough to sit across the top.
    for (i, (s, f)) in enumerate(zip(series, chosen))
        text!(ax, 0.03, 0.97 - 0.085 * (i - 1);
              text = @sprintf("α = %.1f × 10⁻⁵ K⁻¹", 1e5 * f.α),
              space = :relative, align = (:left, :top), fontsize = TICK, color = s.col)
    end
    return handles, labels
end
handles, labels = draw!(ax, series, chosen)

# identity across the top, OUTSIDE the axis: the data fills both lower corners and the
# upper left carries α, so there is no in-axis position a legend can occupy without
# covering something
Legend(fig[1, 1], handles, labels; orientation = :horizontal, tellwidth = false,
       framevisible = false, labelsize = TICK, patchsize = (12, 10),
       padding = (2, 2, 0, 0), colgap = 12)
rowgap!(fig.layout, 1, 2)

mkpath(dirname(OUT))
save("$OUT.pdf", fig); save("$OUT.png", fig; px_per_unit = 4)
@printf("\nfigure → %s.{pdf,png}\n", OUT)
