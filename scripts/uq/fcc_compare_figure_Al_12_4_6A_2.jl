# fcc_compare_figure_Al_12_4_6A_2.jl
#
# MERGED version of fcc_stability_figure_Al_12_4_6A_2.jl (constrained + rejection,
# survives) and fcc_instability_figure_Al_12_4_6A_2.jl (naive worst, transforms).
# Both originals are left untouched.
#
# Layout, one figure:
#   LEFT  (full height) — phonon bands of BOTH models overlaid at their own a(T)
#   RIGHT top           — RDF of the constrained member
#   RIGHT bottom        — RDF of the naive worst member
#
# The coordination panel is dropped.
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

include(joinpath(@__DIR__, "..", "bandpath_phonon_uq", "lib.jl"))

element    = :Al
T_K        = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 300
use_full   = length(ARGS) >= 2 && ARGS[2] == "full"
N_cell_fc  = 4
N_per_seg  = [20, 20, 20, 20, 60]

MODELDIR = abspath(joinpath(@__DIR__, "..", "..", "models", "Al_12_4_6A_2_"))
RESDIR   = "$MODELDIR/results"
DIR_CON  = "$RESDIR/npt_multivolume_softest"
DIR_NAI  = "$RESDIR/npt_thermal_expansion_naive_worst_member"

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

θ_con = vec(readdlm("$RESDIR/bandpath_undotted_multivolume/theta_npt_softest.csv", ','))
θ_nai = vec(readdlm("$DIR_NAI/theta_naive_worst.csv", ','))
for (nm, θ) in (("constrained", θ_con), ("naive", θ_nai))
    length(θ) == length(result.lin_params) ||
        error("$nm θ has $(length(θ)) entries, model has $(length(result.lin_params))")
end

a_con = read_aT("$DIR_CON/thermal_expansion_summary.csv", T_K)
a_nai = read_aT("$DIR_NAI/thermal_expansion_summary.csv", T_K)
@printf("a(%d K): constrained %.5f Å | naive %.5f Å\n", T_K, a_con, a_nai); flush(stdout)

r_con, g_con = read_rdf("$DIR_CON/rdf_$(T_K)K.csv")
r_nai, g_nai = read_rdf("$DIR_NAI/rdf_$(T_K)K.csv")
@printf("RDF: %d bins (constrained), %d bins (naive)\n", length(r_con), length(r_nai)); flush(stdout)

# ── phonons at each model's own a(T) ─────────────────────────────────────────
println("── constrained bands ──"); flush(stdout)
bp_con = bandpath_Dk(result, model, element, a_con, N_cell_fc; N_per_seg=N_per_seg)
F_con  = bands(θ_con, bp_con)
mo_con = min_freq_stable(θ_con, bp_con)

println("── naive bands ──"); flush(stdout)
bp_nai = bandpath_Dk(result, model, element, a_nai, N_cell_fc; N_per_seg=N_per_seg)
F_nai  = bands(θ_nai, bp_nai)
mo_nai = min_freq_stable(θ_nai, bp_nai)

@printf("min non-acoustic ω: constrained %+.3f THz | naive %+.3f THz\n", mo_con, mo_nai)
flush(stdout)

# The two paths are generated at different lattice constants, so their x-grids are
# scaled differently (|q| ∝ 1/a). Plot each against its own, but they share ticks
# since the labels are the same high-symmetry points.
x_con, x_nai = bp_con.x_vals, bp_nai.x_vals

# ── figure ───────────────────────────────────────────────────────────────────
BLU = RGBf(0.0, 0.447, 0.698)
RED = RGBf(0.80, 0.15, 0.15)
ORA = RGBf(0.835, 0.369, 0.0)

fig = Figure(size=(900, 470), figure_padding=(8, 12, 6, 8))

# ---- LEFT: both band structures, shared axis --------------------------------
ax1 = Axis(fig[1:2, 1]; xlabel="Wave vector", ylabel="Frequency (THz)",
           title="Phonon dispersion at a($(T_K) K)", titlesize=12,
           xlabelsize=11, ylabelsize=11, xticklabelsize=10, yticklabelsize=10,
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

axislegend(ax1,
    [LineElement(color=BLU, linewidth=2.2), LineElement(color=RED, linewidth=2.2)],
    ["constrained + rejection:  min ω = $(round(mo_con; digits=3)) THz",
     "naive POPS (worst):  min ω = $(round(mo_nai; digits=2)) THz"],
    position=:rb, framevisible=true, labelsize=9, patchsize=(20, 2),
    padding=(6, 6, 4, 4), rowgap=1)
text!(ax1, 0.015, 0.985; text="(a)", space=:relative, align=(:left, :top),
      font=:bold, fontsize=13)

# ---- RIGHT: the two RDFs ----------------------------------------------------
"ideal FCC shells r_n = a√(n/2) inside the plotted range"
shells(a, rmax) = filter(<(rmax), a .* sqrt.((1:16) ./ 2))

rmax = min(maximum(r_con), maximum(r_nai))   # common x-range so the two compare

ax2 = Axis(fig[1, 2]; ylabel="g(r)",
           title="Constrained + rejection — a = $(round(a_con; digits=3)) Å",
           titlesize=10, titlecolor=BLU,
           xlabelsize=11, ylabelsize=11, xticklabelsize=9, yticklabelsize=9,
           xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1)
barplot!(ax2, r_con, g_con; width=step(range(r_con[1], r_con[end]; length=length(r_con))),
         color=(BLU, 0.75), gap=0, strokewidth=0)
vlines!(ax2, shells(a_con, rmax); color=(ORA, 0.85), linestyle=:dash, linewidth=1.0)
xlims!(ax2, 0, rmax)
hidexdecorations!(ax2; grid=false, ticks=false, minorticks=false)
text!(ax2, 0.025, 0.95; text="(b)", space=:relative, align=(:left, :top),
      font=:bold, fontsize=13)
text!(ax2, 0.985, 0.93; text="dashed: ideal FCC shells",
      space=:relative, align=(:right, :top), fontsize=8, color=ORA)

ax3 = Axis(fig[2, 2]; xlabel="r (Å)", ylabel="g(r)",
           title="Naive POPS, worst member — a = $(round(a_nai; digits=3)) Å",
           titlesize=10, titlecolor=RED,
           xlabelsize=11, ylabelsize=11, xticklabelsize=9, yticklabelsize=9,
           xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1)
barplot!(ax3, r_nai, g_nai; width=step(range(r_nai[1], r_nai[end]; length=length(r_nai))),
         color=(RED, 0.70), gap=0, strokewidth=0)
vlines!(ax3, shells(a_nai, rmax); color=(ORA, 0.85), linestyle=:dash, linewidth=1.0)
xlims!(ax3, 0, rmax)
text!(ax3, 0.025, 0.95; text="(c)", space=:relative, align=(:left, :top),
      font=:bold, fontsize=13)

linkxaxes!(ax2, ax3)
colsize!(fig.layout, 1, Relative(0.54))
colgap!(fig.layout, 14)
rowgap!(fig.layout, 8)

out = "$RESDIR/fcc_compare_constrained_vs_naive_$(T_K)K"
save("$out.pdf", fig)
save("$out.png", fig; px_per_unit=4)
@printf("\nfigure → %s.{pdf,png}\n", out)
