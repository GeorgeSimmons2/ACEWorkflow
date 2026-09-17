# bands_two_ensembles_Al_16_4_6A_3.jl
#
# STAGE 2 of three.  Phonon bands of the two ensembles from
# scripts/uq/pinned_rejection_ensembles_Al_16_4_6A_3.jl, side by side on ONE shared
# frequency axis.
#
# Every member gets an INDEPENDENT geometry optimisation and its OWN native Hessian.
# Stage 1 screened proposals with a single undotted Hessian at a_eq, which is exact
# only because the members are pinned there — but that is the screen, not the result.
# Rebuilding natively at each member's own relaxed lattice constant means the figure
# does not inherit the screening assumption, and the two can be compared:
# stage 1's min ω is reported alongside this one as a CONTROL.  They should agree
# closely for the constrained ensemble; a gap means the pin drifted.
#
# ── WHY BOTH PANELS SHARE A FREQUENCY AXIS ──────────────────────────────────
# The comparison is how much WIDER the unconstrained ensemble is.  Independent axes
# would autoscale each to its own spread and destroy exactly that.
#
# ── PARALLELISM ─────────────────────────────────────────────────────────────
# One force-constant build per member, each task with its OWN deep copy of the model —
# set_linear_parameters! mutates, so a shared model is a data race giving quietly wrong
# bands rather than a crash.  Chunked @spawn, not @threads with threadid().
#
# Run:  julia --project -t 40 scripts/qoi/bands_two_ensembles_Al_16_4_6A_3.jl
#   N_CELL=4  N_PER_SEG=20  UNSTABLE=-0.05  QOI_THREADS=<n>  FIGW=540

include(joinpath(@__DIR__, "..", "bandpath_phonon_uq", "lib.jl"))
using Serialization

element     = :Al
N_CELL      = parse(Int,     get(ENV, "N_CELL", "4"))
# MUST match stage 1's sampling, or the control below compares different q-grids and
# reports a difference that has nothing to do with the Hessians.
N_PER_SEG   = haskey(ENV, "N_PER_SEG") ? parse(Int, ENV["N_PER_SEG"]) :
                                         [20, 20, 20, 20, 60]
UNSTABLE    = parse(Float64, get(ENV, "UNSTABLE", "-0.05"))
qΓtol       = 5e-2
QOI_THREADS = parse(Int,     get(ENV, "QOI_THREADS", string(Threads.nthreads())))
FIGW        = parse(Float64, get(ENV, "FIGW", "540"))

MODELDIR = "models/Al_16_4_6A_3_"
SRC      = get(ENV, "SRC", "$MODELDIR/results/pinned_ensembles")
outdir   = get(ENV, "OUTDIR", SRC)

BLU = RGBf(0.0,0.447,0.698); GRY = RGBAf(0.45,0.45,0.45,0.35); RED = RGBAf(0.80,0.15,0.15,0.75)
TITLE, LAB, TICK = 13, 12, 11

# ── model (fast load: nothing here needs the 1.9 GB A.csv) ──────────────────
model, _ = ACEpotentials.load_model("$MODELDIR/Al_16_4_6A_3.json")
lin_params = vec(readdlm("$MODELDIR/lin_params.csv", ','))
ACEpotentials.Models.set_linear_parameters!(model, lin_params)
n_params = length(lin_params)
@printf("fast load: %d parameters (A.csv skipped)\n", n_params)

function read_ens(tag)
    f = "$SRC/ensemble_$(tag).csv"
    isfile(f) || error("missing $f — run scripts/uq/pinned_rejection_ensembles_Al_16_4_6A_3.jl first")
    M = readdlm(f, ',')
    size(M, 2) == n_params || error("$f is $(size(M,2)) wide, model has $n_params")
    return [collect(Float64, M[i, :]) for i in 1:size(M, 1)]
end
ens = [(tag = "unconstrained", label = "unconstrained (pinned, no predicate)", col = RED,
        mem = read_ens("unconstrained")),
       (tag = "constrained",   label = "constrained (phonon-positive)",        col = BLU,
        mem = read_ens("constrained"))]
for e in ens; @printf("%-14s %d members\n", e.tag, length(e.mem)); end
structure = AtomsBuilder.Chemistry.symmetry(element)
flush(stdout)

# ── one member: relax, build its own Hessian, band structure ────────────────
# `m` is a per-task model copy.  Never pass the shared one.
function bands_native(m, θ)
    ACEpotentials.Models.set_linear_parameters!(m, θ)
    a = ACEWorkflow.relax_lattice_constant(m, element)
    sp, ss = bulk_prim_super(element; a=a, N_cell=N_CELL)
    fc = precompute_force_constants(sp, ss, m)
    ql, xv, xt, lb, _ = _band_path(structure, fc.L; N_per_seg=N_PER_SEG)
    Np = fc.Np; qn = norm.(ql)
    F = Matrix{Float64}(undef, 3Np, length(ql))
    for (iq, q) in enumerate(ql)
        ev = eigvals(Hermitian(dynamical_matrix_from_fc(fc, q)))
        F[:, iq] = sign.(ev) .* sqrt.(abs.(ev)) .* FREQ_THz
    end
    return (; F, x_vals=xv, x_ticks=xt, labels=lb, Np, a,
              minω = minimum(F[:, qn .> qΓtol]))
end

function bands_many(θs; label="")
    n = length(θs); out = Vector{Any}(undef, n)
    nt = clamp(QOI_THREADS, 1, min(Threads.nthreads(), n))
    @printf("[%s] %d independent relaxations + Hessians on %d tasks …\n", label, n, nt)
    flush(stdout)
    t = @elapsed @sync for k in 1:nt
        Threads.@spawn begin
            m = deepcopy(model)
            for i in k:nt:n; out[i] = bands_native(m, θs[i]); end
        end
    end
    @printf("[%s] %.1f min (%.1f s/member)\n", label, t/60, t/n); flush(stdout)
    return out
end

mean_b = bands_native(deepcopy(model), lin_params)
@printf("\nmean model: a = %.5f Å, min ω = %+.4f THz\n", mean_b.a, mean_b.minω)
res = [bands_many(e.mem; label=e.tag) for e in ens]

# ── control: does the native rebuild agree with stage 1's undotted screen? ──
JLS = "$SRC/pinned_ensembles.jls"
if isfile(JLS)
    d = deserialize(JLS)
    for (k, e) in enumerate(ens)
        scr = k == 1 ? d.wu : d.wc
        nat = [b.minω for b in res[k]]
        @printf("CONTROL [%-13s] native vs stage-1 undotted screen: max |Δ| = %.3e THz\n",
                e.tag, maximum(abs.(nat .- scr)))
    end
    println("  (should be ~1e-3 THz or below: members are pinned to a_eq to microangstroms,")
    println("   so the screen's single Hessian is the right operator.  A gap of ~0.1 THz")
    println("   means the two stages sampled DIFFERENT q-points, not that the pin drifted —")
    println("   check N_PER_SEG matches stage 1.)")
end

# ── statistics ──────────────────────────────────────────────────────────────
println("\n══ PHONONS, NATIVE PER-MEMBER HESSIANS ══════════════════════════")
for (k, e) in enumerate(ens)
    w = [b.minω for b in res[k]]; a = [b.a for b in res[k]]
    @printf("%-14s min ω ∈ [%+.4f, %+.4f], median %+.4f | %d/%d soft | a ∈ [%.5f, %.5f]\n",
            e.tag, minimum(w), maximum(w), median(w),
            count(<(UNSTABLE), w), length(w), minimum(a), maximum(a))
end
let wu = [b.minω for b in res[1]], wc = [b.minω for b in res[2]]
    @printf("\nspread ratio σ_constrained/σ_unconstrained = %.4f\n", std(wc)/std(wu))
    @printf("soft members: %d → %d out of %d\n",
            count(<(UNSTABLE), wu), count(<(UNSTABLE), wc), length(wu))
end
flush(stdout)

# ── figure: one shared frequency axis ───────────────────────────────────────
allF = vcat([b.F for r in res for b in r], [mean_b.F])
lo = minimum(minimum.(allF)); hi = maximum(maximum.(allF)); pad = 0.06*(hi-lo)
ylim = (min(lo-pad, -0.5), hi+pad)
xref = mean_b.x_vals; xt = mean_b.x_ticks; lbl = mean_b.labels; Np = mean_b.Np
@printf("\nshared frequency axis: [%.2f, %.2f] THz\n", ylim...)

function panel!(gp, bs, ttl; showy=true)
    soft = [b.minω < UNSTABLE for b in bs]
    ax = Axis(gp; xlabel="Wave vector", ylabel = showy ? "Frequency (THz)" : "",
              title=ttl, titlesize=TITLE, xlabelsize=LAB, ylabelsize=LAB,
              xticklabelsize=TICK, yticklabelsize=TICK, xticks=(xt, lbl),
              xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1,
              xticksize=4, yticksize=4)
    band!(ax, [first(xref), last(xref)], [ylim[1], ylim[1]], [0.0, 0.0];
          color=(RGBf(0.8,0.15,0.15), 0.06))
    for (i, b) in enumerate(bs), br in 1:3Np
        lines!(ax, xref, b.F[br, :]; color = soft[i] ? RED : GRY,
               linewidth = soft[i] ? 1.0 : 0.7)
    end
    for br in 1:3Np; lines!(ax, xref, mean_b.F[br, :]; color=BLU, linewidth=1.8); end
    hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=1.0)
    vlines!(ax, xt; color=(:black,0.22), linewidth=0.7)
    xlims!(ax, first(xref), last(xref)); ylims!(ax, ylim...)
    showy || hideydecorations!(ax; grid=false, ticks=false, minorticks=false)
    text!(ax, 0.97, 0.03; text="$(count(soft))/$(length(soft)) unstable", space=:relative,
          align=(:right,:bottom), fontsize=TICK-1, color=:gray30)
end

fig = Figure(size=(FIGW, 0.42FIGW), figure_padding=(6, 10, 4, 6))
panel!(fig[1,1], res[1], ens[1].label)
panel!(fig[1,2], res[2], ens[2].label; showy=false)
colgap!(fig.layout, 20)
save("$outdir/bands_two_ensembles.pdf", fig)
save("$outdir/bands_two_ensembles.png", fig; px_per_unit=4)

for (k, e) in enumerate(ens)
    writedlm("$outdir/bands_$(e.tag).csv",
             hcat(1:length(res[k]), [b.a for b in res[k]], [b.minω for b in res[k]]), ',')
end
serialize("$outdir/bands_two_ensembles.jls",
          (; res, mean_b, ylim, N_CELL, N_PER_SEG, UNSTABLE,
             tags = [e.tag for e in ens]))
println("\nfigure → $outdir/bands_two_ensembles.{pdf,png}")
println("data   → $outdir/bands_{unconstrained,constrained}.csv  (member, relaxed a, min ω)")
