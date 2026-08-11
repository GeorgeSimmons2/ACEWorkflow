# qha_bands_expanded_Al_16_4_6A_3.jl
#
# Show the failure that qha_finiteT_Al_16_4_6A_3.jl counts: naive POPS members that are
# dynamically stable at their own 0 K equilibrium and develop IMAGINARY phonon modes
# once the crystal is expanded to its finite-temperature volume.
#
# Two panels on one shared frequency axis:
#     (a)  every member at its own a_static          — the volume everything is
#                                                      constrained and validated at
#     (b)  every member at EXPAND × its own a_static — the volume it actually occupies
#                                                      once heated
#
# Members whose minimum frequency goes below `unstable_tol` in a panel are drawn in red
# there, so the figure reads: all fine cold, several branches dip below zero hot.
#
# ── WHY A RELATIVE EXPANSION, NOT ONE SHARED LATTICE CONSTANT ───────────────
# Members differ in a_static, so a single shared lattice constant would put some under
# compression and others under tension and confound the comparison.  EXPAND is applied
# to each member's OWN equilibrium, which is the well-defined statement "expand this
# member's crystal by X%".  It is also defined for the members that have no QHA minimum
# at all — precisely the ones of interest, which a plot at "their a(600 K)" could not
# include, since they have none.
#
# The default 1.023 is the mean model's own QHA expansion to 600 K, so panel (b) is the
# thermal volume for the mean model and the equivalent relative expansion for everyone
# else.  Set EXPAND to sweep.
#
# ── BAND PATH, NOT THE MP GRID ──────────────────────────────────────────────
# The free-energy work uses a Monkhorst-Pack grid because it is a thermodynamic average.
# Here the point is to SEE the instability, so the high-symmetry path is right: it shows
# which branch and which part of the zone goes soft.  The two are consistent — a mode
# imaginary on the path is imaginary on the grid — but the path can miss an instability
# that sits off it, so the counts here can be a slight underestimate of the grid counts.
#
# Members come from the CSV the QHA run saved.  If it is absent they are regenerated
# with the same seed and written out, which costs a read of the 1.9 GB A.csv.
#
# Run:  julia --project -t <ncores> scripts/qoi/qha_bands_expanded_Al_16_4_6A_3.jl
#   EXPAND=1.023  N_CELL=4  N_PER_SEG=20  UNSTABLE=-0.05  QOI_THREADS=<n>  FIGW=540

include(joinpath(@__DIR__, "..", "bandpath_phonon_uq", "lib.jl"))
using Random, Serialization
Random.seed!(1234)

element     = :Al
EXPAND      = parse(Float64, get(ENV, "EXPAND", "1.023"))
N_CELL      = parse(Int,     get(ENV, "N_CELL", "4"))
N_PER_SEG   = parse(Int,     get(ENV, "N_PER_SEG", "20"))
UNSTABLE    = parse(Float64, get(ENV, "UNSTABLE", "-0.05"))
LEV_PCT     = parse(Float64, get(ENV, "LEV_PCT", "0.5"))
N_MEMBERS   = parse(Int,     get(ENV, "N_MEMBERS", "20"))
QOI_THREADS = parse(Int,     get(ENV, "QOI_THREADS", string(Threads.nthreads())))
FIGW        = parse(Float64, get(ENV, "FIGW", "540"))

MODELDIR = "models/Al_16_4_6A_3_"
outdir   = get(ENV, "OUTDIR", "$MODELDIR/results/qha_finiteT"); mkpath(outdir)
CSV      = "$outdir/samples_naive.csv"

BLU = RGBf(0.0,0.447,0.698); GRY = RGBAf(0.45,0.45,0.45,0.35); RED = RGBAf(0.80,0.15,0.15,0.75)
TITLE, LAB, TICK = 13, 12, 11

# ── model + the SAME committee the QHA run used ─────────────────────────────
if isfile(CSV)
    M = readdlm(CSV, ',')
    model, _ = ACEpotentials.load_model("$MODELDIR/Al_16_4_6A_3.json")
    lin_params = vec(readdlm("$MODELDIR/lin_params.csv", ','))
    ACEpotentials.Models.set_linear_parameters!(model, lin_params)
    size(M, 2) == length(lin_params) || error("$CSV is $(size(M,2)) wide, model has $(length(lin_params))")
    members = [collect(Float64, M[i, :]) for i in 1:size(M, 1)]
    @printf("model loaded (A.csv skipped); %d members ← %s\n", length(members), basename(CSV))
else
    @info "no saved committee; regenerating with seed 1234 (reads the 1.9 GB A.csv)"
    result = load_model(element, 16, 4, 6, 3; dataset_name="")
    model = result.model; lin_params = result.lin_params
    Ap = Diagonal(result.W) * result.A / result.P
    deltas = corrections(Ap, result.W .* result.Y, result.P; leverage_percentile=LEV_PCT)
    heig, hb = hypercube(deltas)
    mat, _ = sample_hypercube(heig, hb, lin_params; number_of_committee_members=N_MEMBERS)
    members = [mat[:, i] for i in 1:size(mat, 2)]
    writedlm(CSV, mat', ','); @printf("committee saved → %s\n", CSV)
    Ap = nothing; deltas = nothing; GC.gc()
end
structure = AtomsBuilder.Chemistry.symmetry(element)
@printf("expansion factor %.4f applied to each model's OWN a_static\n", EXPAND); flush(stdout)

# ── bands for one parameter vector at one relative expansion ────────────────
# `m` is a per-task model copy.  Never pass the shared one.
function bands_at(m, θ, factor)
    ACEpotentials.Models.set_linear_parameters!(m, θ)
    a0 = ACEWorkflow.relax_lattice_constant(m, element)
    a  = factor * a0
    sp, ss = bulk_prim_super(element; a=a, N_cell=N_CELL)
    fc = precompute_force_constants(sp, ss, m)
    ql, xv, xt, lb, _ = _band_path(structure, fc.L; N_per_seg=N_PER_SEG)
    Np = fc.Np; qn = norm.(ql)
    F = Matrix{Float64}(undef, 3Np, length(ql))
    for (iq, q) in enumerate(ql)
        ev = eigvals(Hermitian(dynamical_matrix_from_fc(fc, q)))
        F[:, iq] = sign.(ev) .* sqrt.(abs.(ev)) .* FREQ_THz
    end
    keep = qn .> 5e-2
    iw = argmin(vec(F[:, keep]))
    return (; F, x_vals=xv, x_ticks=xt, labels=lb, Np, a0, a,
              minω = minimum(F[:, keep]),
              iq_min = findall(keep)[fld1(iw, 3Np)])
end

function bands_many(θs, factor; label="")
    n = length(θs); out = Vector{Any}(undef, n)
    nt = clamp(QOI_THREADS, 1, min(Threads.nthreads(), n))
    @printf("[%s] %d models on %d tasks …\n", label, n, nt); flush(stdout)
    t = @elapsed @sync for k in 1:nt
        Threads.@spawn begin
            m = deepcopy(model)
            for i in k:nt:n; out[i] = bands_at(m, θs[i], factor); end
        end
    end
    @printf("[%s] %.1f min\n", label, t/60); flush(stdout)
    return out
end

cold = bands_many(members, 1.0;    label="cold")
hot  = bands_many(members, EXPAND; label="hot")
mc   = bands_at(deepcopy(model), lin_params, 1.0)
mh   = bands_at(deepcopy(model), lin_params, EXPAND)

wc = [b.minω for b in cold]; wh = [b.minω for b in hot]
sc = wc .>= UNSTABLE;        sh = wh .>= UNSTABLE
@printf("\nmean model: a %.5f → %.5f Å, min ω %+.4f → %+.4f THz\n",
        mc.a, mh.a, mc.minω, mh.minω)
@printf("committee : stable cold %d/%d, stable hot %d/%d\n",
        count(sc), length(sc), count(sh), length(sh))
@printf("            STABLE COLD → SOFT HOT: %d   (soft at both: %d)\n",
        count(sc .& .!sh), count(.!sc .& .!sh))
@printf("min ω: cold median %+.4f, hot median %+.4f, median shift %+.4f THz\n",
        median(wc), median(wh), median(wh .- wc))

# where in the zone does the instability appear?
seglab = ["Γ→X","X→U","U→L","L→Γ","Γ→K"]
segof(b, iq) = clamp(searchsortedlast(b.x_ticks, b.x_vals[iq]), 1, length(seglab))
newly = findall(sc .& .!sh)
if !isempty(newly)
    println("\nmembers that lose stability on expansion, and where:")
    for i in newly
        @printf("  member %2d: min ω %+.4f → %+.4f THz, soft on %s\n",
                i, wc[i], wh[i], seglab[segof(hot[i], hot[i].iq_min)])
    end
end
flush(stdout)

# ── figure: shared frequency axis so the two panels are comparable ──────────
allF = vcat([b.F for b in cold], [b.F for b in hot], [mc.F, mh.F])
lo = minimum(minimum.(allF)); hi = maximum(maximum.(allF)); pad = 0.06*(hi-lo)
ylim = (min(lo-pad, -0.5), hi+pad)
xref = mc.x_vals; xt = mc.x_ticks; lbl = mc.labels; Np = mc.Np
@printf("\nshared frequency axis: [%.2f, %.2f] THz\n", ylim...)

function panel!(gp, bs, soft, central, ttl; showy=true)
    ax = Axis(gp; xlabel="Wave vector", ylabel = showy ? "Frequency (THz)" : "",
              title=ttl, titlesize=TITLE, xlabelsize=LAB, ylabelsize=LAB,
              xticklabelsize=TICK, yticklabelsize=TICK, xticks=(xt, lbl),
              xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1,
              xticksize=4, yticksize=4)
    band!(ax, [first(xref), last(xref)], [ylim[1], ylim[1]], [0.0, 0.0];
          color=(RGBf(0.8,0.15,0.15), 0.06))          # shade the unstable half-plane
    for (i, b) in enumerate(bs), br in 1:3Np
        lines!(ax, xref, b.F[br, :]; color = soft[i] ? RED : GRY,
               linewidth = soft[i] ? 1.0 : 0.7)
    end
    for br in 1:3Np; lines!(ax, xref, central.F[br, :]; color=BLU, linewidth=1.8); end
    hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=1.0)
    vlines!(ax, xt; color=(:black,0.22), linewidth=0.7)
    xlims!(ax, first(xref), last(xref)); ylims!(ax, ylim...)
    showy || hideydecorations!(ax; grid=false, ticks=false, minorticks=false)
    text!(ax, 0.97, 0.03; text="$(count(soft))/$(length(soft)) unstable", space=:relative,
          align=(:right,:bottom), fontsize=TICK-1, color=:gray30)
    return ax
end

fig = Figure(size=(FIGW, 0.42FIGW), figure_padding=(6, 10, 4, 6))
panel!(fig[1,1], cold, .!sc, mc, "at a₀ (0 K equilibrium)")
panel!(fig[1,2], hot,  .!sh, mh,
       "expanded ×$(round(EXPAND;digits=3))  —  a($(Int(600)) K)"; showy=false)
colgap!(fig.layout, 20)
tag = replace(@sprintf("%.3f", EXPAND), "." => "p")
save("$outdir/qha_bands_expanded_$(tag).pdf", fig)
save("$outdir/qha_bands_expanded_$(tag).png", fig; px_per_unit=4)

writedlm("$outdir/qha_bands_expanded_$(tag).csv",
         hcat(1:length(wc), [b.a0 for b in cold], [b.a for b in hot], wc, wh), ',')
serialize("$outdir/qha_bands_expanded_$(tag).jls",
          (; cold, hot, mc, mh, wc, wh, sc, sh, EXPAND, N_CELL, N_PER_SEG, UNSTABLE, ylim))
println("\nfigure → $outdir/qha_bands_expanded_$(tag).{pdf,png}")
println("data   → $outdir/qha_bands_expanded_$(tag).csv  (member, a₀, a_expanded, min ω cold, min ω hot)")
