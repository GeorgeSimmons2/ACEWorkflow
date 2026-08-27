# ─────────────────────────────────────────────────────────────────────────────
# COPY of scripts/uq/fcc_compare_figure_Al_12_4_6A_2.jl, placed here because it is the
# structural companion to the a(T) figure: it reads the SAME two NPT runs and answers
# the question a(T) alone cannot — whether the lattice was still FCC when a(T) was
# measured.  Differences from the original are marked `# [REPRO]`:
#   * include path (this directory is one level from the repo root, not two)
#   * DIR_CON / DIR_NAI overridable, so it can be pointed at a pipeline rerun
#   * θ for the constrained arm read from the RUN's own theta_used.csv rather than the
#     committee — byte-identical (verified, max |Δ| = 0.0) but drops the dependency on
#     a committee directory, matching how everything else here now works
# No physics or plotting is changed.
# ─────────────────────────────────────────────────────────────────────────────
# fcc_compare_figure_Al_12_4_6A_2.jl
#
# MERGED version of fcc_stability_figure_Al_12_4_6A_2.jl (constrained + rejection,
# survives) and fcc_instability_figure_Al_12_4_6A_2.jl (naive worst, transforms).
# Both originals are left untouched.
#
# Layout, one figure:
#   LEFT  (full height) — phonon bands of BOTH models overlaid at their own a(T)
#   RIGHT top           — RDF of the UNCONSTRAINED worst member   (the failure)
#   RIGHT bottom        — RDF of the CONSTRAINED member           (the fix)
#
# The coordination panel is dropped.
#
# TERMINOLOGY.  "naive POPS" is called UNCONSTRAINED throughout the figure text, against
# CONSTRAINED.  Directory and file names still say `naive` — those are on disk and are
# not renamed here.  The output filename is likewise unchanged so existing
# \includegraphics keep resolving.
#
# READING ORDER.  Unconstrained sits on top: the reader meets the failure (a structure
# that has left FCC) before the fix directly beneath it, and the two RDFs share an
# x-axis so the shell structure lines up vertically.
#
# FIGURE SIZING.  Makie's `size` is in POINTS for vector output, so this is built at its
# final display width with 11-13 pt text and needs no rescaling in Overleaf.  Do NOT add
# a width= factor to \includegraphics; set FIGW instead.
#
# WHY OVERLAY THE PHONONS.  The two originals used independent y-axes (0→10 THz
# and −11→+5 THz), which visually understates the difference. On a shared axis the
# naive member's imaginary branches sit below the line while the constrained bands
# stay entirely above it, and the ω<0 shading is then meaningful rather than decorative.
#
# DATA REUSE.  The RDFs are read from the rdf_<T>K.csv written by the two original
# scripts, so the histograms are byte-identical to the published panels — no
# re-processing of the trajectories (~13M pair evaluations each). The band
# structures ARE recomputed, but undotted_Hbasis() hits its cache at both lattice
# constants (undotted_Hbasis_4x4x4_a4.07786.jls and …_a4.16663.jls), so no Hessian
# is rebuilt.
#
# FAST MODEL LOAD.  load_model() reads the 259 MB A.csv, which is pure overhead
# here: bandpath_Dk only touches result.dir and result.lin_params. We therefore
# build a minimal stand-in NamedTuple from the JSON model plus lin_params.csv and
# assert the parameter count matches. Set ARGS[2]="full" to force the real loader.
#
# Run:  julia --project -t 8 scripts/uq/fcc_compare_figure_Al_12_4_6A_2.jl [T] [full]

include(joinpath(@__DIR__, "..", "scripts", "bandpath_phonon_uq", "lib.jl"))  # [REPRO] path from repo root

element    = :Al
T_K        = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 300
use_full   = length(ARGS) >= 2 && ARGS[2] == "full"
N_cell_fc  = 4
N_per_seg  = [20, 20, 20, 20, 60]

MODELDIR = abspath(get(ENV, "MODELDIR", joinpath(@__DIR__, "..", "models", "Al_12_4_6A_2_")))  # [REPRO]
RESDIR   = "$MODELDIR/results"
# [REPRO] overridable so this can be built from a pipeline rerun; same variable names
# as the a(T) plotter uses, so one pair of settings drives both figures
DIR_CON  = get(ENV, "DIR_CON",   "$RESDIR/npt_multivolume_softest")
DIR_NAI  = get(ENV, "DIR_UNCON", "$RESDIR/npt_thermal_expansion_naive_worst_member")

# ── model ────────────────────────────────────────────────────────────────────
if use_full
    result = load_model(element, 12, 4, 6, 2; dataset_name="")
    model  = result.model
else
    # Mirrors src/Models/Models.jl:88-95 exactly, minus the A/Y/P/W reads.
    # set_linear_parameters! throws on a length mismatch, which is the check.
    model, _ = ACEpotentials.load_model("$MODELDIR/Al_12_4_6A_2.json")
    lin      = vec(readdlm("$MODELDIR/lin_params.csv", ','))
    ACEpotentials.Models.set_linear_parameters!(model, lin)
    result = (; dir = MODELDIR, lin_params = lin)     # only fields bandpath_Dk uses
    @printf("fast load: %d parameters (skipped A.csv)\n", length(lin))
end
flush(stdout)

# ── per-run inputs: θ, a(T), RDF ─────────────────────────────────────────────
"read a(T) from a thermal_expansion_summary.csv with an unknown number of # lines"
function read_aT(path, T_K)
    lines = readlines(path)
    ncomment = count(l -> startswith(l, "#"), lines)
    s = readdlm(path, ','; skipstart = ncomment + 1)
    r = findfirst(==(Float64(T_K)), Float64.(s[:, 1]))
    r === nothing && error("no T = $T_K row in $path")
    return Float64(s[r, 2])
end

read_rdf(path) = (d = readdlm(path, ','; skipstart=1); (Float64.(d[:,1]), Float64.(d[:,2])))

# [REPRO] prefer the run's own saved vector over the committee's copy.  Verified
# byte-identical for the published run (max |Δ| = 0.0 over all 91 coefficients), but
# reading it from the run directory means this figure does not depend on a committee
# and follows DIR_CON automatically when pointed at a rerun.
θ_con = let run_local = "$DIR_CON/theta_used.csv",
            committee = "$RESDIR/bandpath_undotted_multivolume/theta_npt_softest.csv"
    src = isfile(run_local) ? run_local : committee
    isfile(src) || error("no θ for the constrained arm: tried $run_local and $committee")
    @printf("constrained θ ← %s\n", src)
    vec(readdlm(src, ','))
end
θ_nai = vec(readdlm("$DIR_NAI/theta_naive_worst.csv", ','))
for (nm, θ) in (("constrained", θ_con), ("unconstrained", θ_nai))
    length(θ) == length(result.lin_params) ||
        error("$nm θ has $(length(θ)) entries, model has $(length(result.lin_params))")
end

a_con = read_aT("$DIR_CON/thermal_expansion_summary.csv", T_K)
a_nai = read_aT("$DIR_NAI/thermal_expansion_summary.csv", T_K)
@printf("a(%d K): constrained %.5f Å | unconstrained %.5f Å\n", T_K, a_con, a_nai); flush(stdout)

r_con, g_con = read_rdf("$DIR_CON/rdf_$(T_K)K.csv")
r_nai, g_nai = read_rdf("$DIR_NAI/rdf_$(T_K)K.csv")
@printf("RDF: %d bins (constrained), %d bins (unconstrained)\n", length(r_con), length(r_nai)); flush(stdout)

# ── phonons at each model's own a(T) ─────────────────────────────────────────
println("── constrained bands ──"); flush(stdout)
bp_con = bandpath_Dk(result, model, element, a_con, N_cell_fc; N_per_seg=N_per_seg)
F_con  = bands(θ_con, bp_con)
mo_con = min_freq_stable(θ_con, bp_con)

println("── unconstrained bands ──"); flush(stdout)
bp_nai = bandpath_Dk(result, model, element, a_nai, N_cell_fc; N_per_seg=N_per_seg)
F_nai  = bands(θ_nai, bp_nai)
mo_nai = min_freq_stable(θ_nai, bp_nai)

@printf("min non-acoustic ω: constrained %+.3f THz | unconstrained %+.3f THz\n", mo_con, mo_nai)
flush(stdout)

# The two paths are generated at different lattice constants, so their x-grids are
# scaled differently (|q| ∝ 1/a). Plot each against its own, but they share ticks
# since the labels are the same high-symmetry points.
x_con, x_nai = bp_con.x_vals, bp_nai.x_vals

# ── figure ───────────────────────────────────────────────────────────────────
# FIGW is the width this figure will be DISPLAYED at, in points.  Text is fixed at
# 11-13 pt so it matches MLST body text at that width; the canvas scales, the fonts
# do not.  540 pt ≈ \linewidth in the MLST template.
FIGW = parse(Float64, get(ENV, "FIGW", "540"))
BLU = RGBf(0.0, 0.447, 0.698)
RED = RGBf(0.80, 0.15, 0.15)
ORA = RGBf(0.835, 0.369, 0.0)
TITLE, SUB, LAB, TICK, SMALL = 13, 12, 12, 11, 10

fig = Figure(size=(FIGW, 0.66FIGW), figure_padding=(6, 10, 4, 6))

# ---- LEFT: both band structures, shared axis --------------------------------
ax1 = Axis(fig[1:2, 1]; xlabel="Wave vector", ylabel="Frequency (THz)",
           title="Phonon dispersion at a($(T_K) K)", titlesize=TITLE,
           xlabelsize=LAB, ylabelsize=LAB, xticklabelsize=TICK, yticklabelsize=TICK,
           xticks=(bp_con.x_ticks, bp_con.labels),
           xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1)

lo = min(minimum(F_con), minimum(F_nai))
hi = max(maximum(F_con), maximum(F_nai))
pad = 0.06 * (hi - lo)

# shade the unstable half-plane — now genuinely occupied
band!(ax1, [first(x_con), last(x_con)], [lo - pad, lo - pad], [0.0, 0.0];
      color=(RED, 0.07))

for b in 1:3bp_nai.Np
    lines!(ax1, x_nai, F_nai[b, :]; color=(RED, 0.95), linewidth=1.5)
end
for b in 1:3bp_con.Np
    lines!(ax1, x_con, F_con[b, :]; color=BLU, linewidth=1.6)
end
hlines!(ax1, [0.0]; color=:black, linestyle=:dash, linewidth=0.9)
vlines!(ax1, bp_con.x_ticks; color=(:black, 0.22), linewidth=0.6)
xlims!(ax1, first(x_con), last(x_con))
ylims!(ax1, lo - pad, hi + pad)

# Labels kept short: at TICK pt on a ~0.54·FIGW panel a longer string overruns the
# axis.  The qualifiers ("+ rejection", "worst member") live in the caption.
axislegend(ax1,
    [LineElement(color=RED, linewidth=2.2), LineElement(color=BLU, linewidth=2.2)],
    ["unconstrained:  min ω = $(round(mo_nai; digits=2)) THz",
     "constrained:  min ω = $(round(mo_con; digits=3)) THz"],
    position=:rb, framevisible=true, labelsize=TICK, patchsize=(18, 2),
    padding=(5, 5, 3, 3), rowgap=1)
text!(ax1, 0.015, 0.985; text="(a)", space=:relative, align=(:left, :top),
      font=:bold, fontsize=TITLE)

# ---- RIGHT: the two RDFs, unconstrained ABOVE constrained -------------------
# Reading order is the argument: the failure first, the fix directly beneath it, on a
# shared x-axis so the shell structure lines up vertically.
"ideal FCC shells r_n = a√(n/2) inside the plotted range"
shells(a, rmax) = filter(<(rmax), a .* sqrt.((1:16) ./ 2))

rmax = min(maximum(r_con), maximum(r_nai))   # common x-range so the two compare

ax2 = Axis(fig[1, 2]; ylabel="g(r)",
           title="Unconstrained — a = $(round(a_nai; digits=3)) Å",
           titlesize=SUB, titlecolor=RED,
           xlabelsize=LAB, ylabelsize=LAB, xticklabelsize=TICK, yticklabelsize=TICK,
           xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1)
barplot!(ax2, r_nai, g_nai; width=step(range(r_nai[1], r_nai[end]; length=length(r_nai))),
         color=(RED, 0.70), gap=0, strokewidth=0)
vlines!(ax2, shells(a_nai, rmax); color=(ORA, 0.85), linestyle=:dash, linewidth=1.0)
xlims!(ax2, 0, rmax)
hidexdecorations!(ax2; grid=false, ticks=false, minorticks=false)
text!(ax2, 0.025, 0.95; text="(b)", space=:relative, align=(:left, :top),
      font=:bold, fontsize=TITLE)
text!(ax2, 0.985, 0.93; text="dashed: ideal FCC shells",
      space=:relative, align=(:right, :top), fontsize=SMALL, color=ORA)

ax3 = Axis(fig[2, 2]; xlabel="r (Å)", ylabel="g(r)",
           title="Constrained — a = $(round(a_con; digits=3)) Å",
           titlesize=SUB, titlecolor=BLU,
           xlabelsize=LAB, ylabelsize=LAB, xticklabelsize=TICK, yticklabelsize=TICK,
           xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1)
barplot!(ax3, r_con, g_con; width=step(range(r_con[1], r_con[end]; length=length(r_con))),
         color=(BLU, 0.75), gap=0, strokewidth=0)
vlines!(ax3, shells(a_con, rmax); color=(ORA, 0.85), linestyle=:dash, linewidth=1.0)
xlims!(ax3, 0, rmax)
text!(ax3, 0.025, 0.95; text="(c)", space=:relative, align=(:left, :top),
      font=:bold, fontsize=TITLE)

linkxaxes!(ax2, ax3)
colsize!(fig.layout, 1, Relative(0.54))
colgap!(fig.layout, 14)
rowgap!(fig.layout, 6)

# Filename unchanged: the figure is a restyle, not a new result, and an existing
# \includegraphics{fcc_compare_constrained_vs_naive_300K} keeps resolving.
# [REPRO] defaults to the published stem so existing \includegraphics keep resolving —
# which also means a plain rerun OVERWRITES that file.  Set OUT to leave it alone.
out = get(ENV, "OUT", "$RESDIR/fcc_compare_constrained_vs_naive_$(T_K)K")
save("$out.pdf", fig)
save("$out.png", fig; px_per_unit=4)
@printf("\nfigure → %s.{pdf,png}  (%.0f × %.0f pt)\n", out, FIGW, 0.66FIGW)
