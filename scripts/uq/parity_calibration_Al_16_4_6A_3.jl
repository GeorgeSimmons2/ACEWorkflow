# parity_calibration_Al_16_4_6A_3.jl
#
# STAGE 3 of three.  Test-set parity and calibration for the two ensembles from
# scripts/uq/pinned_rejection_ensembles_Al_16_4_6A_3.jl — independent figures per
# ensemble, four in total.
#
# Plotting comes from scripts/uq/lib_parity_calibration.jl, shared with the Al_12 and
# Al_20 figures, so none of them can drift apart in styling.  MLST sizing throughout:
# built at final display width with 13/12/11 pt text, so Overleaf does no rescaling.
#
# ── ENERGIES ARE PER ATOM ───────────────────────────────────────────────────
# `per_atom=true`, so energies are eV/atom and the RMSE is quoted in meV/atom.  This is
# not cosmetic: the Al test set spans 2 to 25 atoms per configuration, with 2-atom
# frames the most common, so a TOTAL-energy RMSE is dominated by the largest cells and
# the parity spread is partly just cell-size spread.  Forces are already intensive and
# are never rescaled.  The July figures in results/bandpath_undotted/ were total-energy
# and are not comparable to these.
#
# ── WHAT TO READ ────────────────────────────────────────────────────────────
# Both ensembles are lin_params + δ from the SAME box with the same seed, differing only
# in the phonon predicate, so the point prediction is identical and the two are paired.
#   RMSE      should barely move — same mean model on both sides.
#   COVERAGE  is the number.  Rejection removes members, so the envelope narrows and
#             coverage can only fall; how far it falls is what the physics costs.
# Coverage is committee min/max envelope containment, NOT a calibrated interval — read
# it as a relative measure between the two columns.
#
# ── FAST MODEL LOAD ─────────────────────────────────────────────────────────
# committee_predictions only touches the model object, so the 1.9 GB A.csv is skipped.
# Pass "full" as the second argument to force the real loader.
#
# Run:  julia --project -t 8 scripts/uq/parity_calibration_Al_16_4_6A_3.jl [stride] [full]
#   stride 20 = every 20th test configuration (default).  1 for the whole set.
#   FIGW=540

include(joinpath(@__DIR__, "..", "bandpath_phonon_uq", "lib.jl"))
include(joinpath(@__DIR__, "lib_parity_calibration.jl"))

element  = :Al
stride   = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20
use_full = length(ARGS) >= 2 && ARGS[2] == "full"
FIGW     = parse(Float64, get(ENV, "FIGW", "540"))

MODELDIR = "models/Al_16_4_6A_3_"
SRC      = get(ENV, "SRC", "$MODELDIR/results/pinned_ensembles")
outdir   = get(ENV, "OUTDIR", "$SRC/parity_calibration"); mkpath(outdir)
TEST     = abspath(joinpath(@__DIR__, "..", "..", "data", "Al", "manual_df_test_Al.xyz"))

if use_full
    result = load_model(element, 16, 4, 6, 3; dataset_name="")
    model, lin_params = result.model, result.lin_params
else
    model, _ = ACEpotentials.load_model("$MODELDIR/Al_16_4_6A_3.json")
    lin_params = vec(readdlm("$MODELDIR/lin_params.csv", ','))
    ACEpotentials.Models.set_linear_parameters!(model, lin_params)
    @printf("fast load: %d parameters (A.csv skipped)\n", length(lin_params))
end
isfile(TEST) || error("test set not found: $TEST")
n_params = length(lin_params)

function read_ens(tag)
    f = "$SRC/ensemble_$(tag).csv"
    isfile(f) || error("""
        missing $f
        Run scripts/uq/pinned_rejection_ensembles_Al_16_4_6A_3.jl first.""")
    M = readdlm(f, ',')
    size(M, 2) == n_params || error("$f is $(size(M,2)) wide, model has $n_params")
    return [collect(Float64, M[i, :]) for i in 1:size(M, 1)]
end

ensembles = [("unconstrained", "unconstrained — pinned, no predicate",  read_ens("unconstrained")),
             ("constrained",   "constrained — phonon-positive",          read_ens("constrained"))]
for (tag, _, mem) in ensembles
    @printf("%-14s %d members × %d params\n", tag, length(mem), n_params)
end
flush(stdout)

# ── predictions ─────────────────────────────────────────────────────────────
# point_params = lin_params for BOTH: the ensembles are lin_params + δ, so the mean
# model is the common point prediction and the deviation histograms are comparable.
@printf("\n── test-set predictions (stride %d, per atom) ──\n", stride); flush(stdout)
preds = Pair{String,Any}[]
for (tag, ttl, mem) in ensembles
    t = @elapsed pr = committee_predictions_fast(model, mem, TEST; stride=stride,
                                            point_params=lin_params, per_atom=true)
    @printf("  %-14s %d configs, %d force components  [%.1f s]\n",
            tag, pr.n, length(pr.tF), t)
    push!(preds, tag => (ttl=ttl, pr=pr)); flush(stdout)
end

# ── predictions: energies AND forces, threaded over configurations ──────────
# The model is linear in the parameters, so one call to energy_forces_virial_basis per
# configuration gives the whole design row and every member is a matvec.  No
# set_committee!, no @committee — cost is O(n_configs) basis evaluations plus BLAS,
# independent of committee size.
#
# Threaded over configurations, each task with its OWN model copy: basis evaluation
# uses the model's internal scratch buffers, so a shared model is a data race.  Results
# are collected per configuration and concatenated IN ORDER.
test_configs = ExtXYZ.load(TEST)[1:20:end]
con_com = readdlm("models/Al_16_4_6A_3_/results/pinned_ensembles/ensemble_constrained.csv", ',')
con_com = [con_com[i,:] for i=1:size(con_com, 1)]
Θcom = reduce(hcat, con_com)                      # n_params × n_members
@printf("%d configurations, %d committee members, %d threads\n",
        length(test_configs), length(con_com), Threads.nthreads()); flush(stdout)

ncfg = length(test_configs)
slots = Vector{Any}(undef, ncfg)
nt = clamp(Threads.nthreads(), 1, ncfg)
tpred = @elapsed @sync for k in 1:nt
    Threads.@spawn begin
        m = nt == 1 ? model : deepcopy(model)
        for i in k:nt:ncfg
            cfg = test_configs[i]
            nat = length(cfg)
            b   = ACEpotentials.Models.energy_forces_virial_basis(cfg, m)

            BE  = ustrip.(u"eV", b.energy)                    # n_params
            eco = (transpose(BE) * Θcom)[:] ./ nat            # per atom, n_members
            pe  = dot(BE, lin_params) / nat
            te  = ustrip(cfg[:dft_energy]) / nat

            bf  = b.forces                                    # n_atoms × n_params of SVector{3}
            na, npar = size(bf)
            BF  = Matrix{Float64}(undef, 3na, npar)           # atom-major, matching the DFT layout
            @inbounds for kk in 1:npar, ii in 1:na
                v = ustrip.(bf[ii, kk])
                BF[3ii-2, kk] = v[1]; BF[3ii-1, kk] = v[2]; BF[3ii, kk] = v[3]
            end
            Fco = BF * Θcom                                   # 3na × n_members
            pf  = BF * lin_params
            tf  = reduce(vcat, ustrip.([at[:dft_forces] for at in cfg]))

            slots[i] = (; pe, te, eco, pf, tf, Fco)
        end
    end
end
@printf("predictions in %.2f s (%.1f ms/config)\n", tpred, 1000tpred/ncfg); flush(stdout)

# energies, one entry per configuration
true_Es = Float64[o.te for o in slots]
Es      = Float64[o.pe for o in slots]
co_Es   = [o.eco for o in slots]

# forces, flattened over configurations × atoms × components
true_Fs = reduce(vcat, (o.tf for o in slots))
Fs      = reduce(vcat, (o.pf for o in slots))
co_Fs   = reduce(vcat, (o.Fco for o in slots))    # (Σ 3n_atoms) × n_members

# ── parity and calibration, MLST styling ────────────────────────────────────
# Built at final display width with 13/12/11 pt text, so Overleaf does no rescaling.
BLU = RGBf(0.0, 0.447, 0.698); ORN = RGBf(0.835, 0.369, 0.0); AMB = RGBf(0.902, 0.624, 0.0)
TITLE, LAB, TICK = 13, 12, 11

loE = Float64[minimum(c) for c in co_Es]
hiE = Float64[maximum(c) for c in co_Es]
loF = vec(minimum(co_Fs; dims=2))
hiF = vec(maximum(co_Fs; dims=2))
rmseE = sqrt(mean((Es .- true_Es).^2))
rmseF = sqrt(mean((Fs .- true_Fs).^2))
covE  = 100 * mean((true_Es .>= loE) .& (true_Es .<= hiE))
covF  = 100 * mean((true_Fs .>= loF) .& (true_Fs .<= hiF))

function parity_panel!(gp, t, p, lo, hi, xl, yl, col, ttl)
    ax = Axis(gp; xlabel=xl, ylabel=yl, title=ttl, titlesize=TITLE-1,
              xlabelsize=LAB, ylabelsize=LAB, xticklabelsize=TICK, yticklabelsize=TICK,
              xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1, aspect=1)
    l = extrema(vcat(t, p, lo, hi))
    lines!(ax, collect(l), collect(l); color=:black, linestyle=:dash, linewidth=1.2)
    errorbars!(ax, t, p, p .- lo, hi .- p; whiskerwidth=4, linewidth=0.7, color=(col, 0.35))
    scatter!(ax, t, p; color=(col, 0.9), markersize=4)
    xlims!(ax, l...); ylims!(ax, l...)
end

fig = Figure(size=(FIGW, 0.62FIGW), figure_padding=(6, 10, 4, 6))
parity_panel!(fig[2,1], true_Es, Es, loE, hiE, "DFT energy (eV/atom)", "ACE energy (eV/atom)",
              BLU, @sprintf("RMSE %.4f eV/atom  |  coverage %.1f%%", rmseE, covE))
parity_panel!(fig[2,2], true_Fs, Fs, loF, hiF, "DFT force (eV/Å)", "ACE force (eV/Å)",
              ORN, @sprintf("RMSE %.4f eV/Å  |  coverage %.1f%%", rmseF, covF))
Legend(fig[1, 1:2],
       [LineElement(color=:black, linestyle=:dash, linewidth=2),
        MarkerElement(color=:gray30, marker=:circle, markersize=8),
        LineElement(color=(:gray30, 0.5), linewidth=6)],
       ["perfect agreement", "central model", "committee range"];
       orientation=:horizontal, tellwidth=false, framevisible=false,
       labelsize=TICK, padding=(2,2,0,0), colgap=12)
colgap!(fig.layout, 20); rowgap!(fig.layout, 1, 2)
save("$outdir/parity_constrained.pdf", fig)
save("$outdir/parity_constrained.png", fig; px_per_unit=4)

# calibration: test error against the committee's own spread, both scaled by the MAE so
# they share an axis.  Well calibrated ⇒ the two overlap; a narrower amber curve means
# the ensemble understates its own error.
function calib_panel!(gp, err, dev, ttl, showy)
    mae = mean(abs.(err)); ne = err ./ mae; sp = dev ./ mae
    lim = maximum(abs.(vcat(ne, sp))); ed = range(-lim, lim; length=61)
    d1 = (h = fit(Histogram, ne, ed).weights; max.(h ./ (sum(h)*step(ed)), 1e-3))
    d2 = (h = fit(Histogram, sp, ed).weights; max.(h ./ (sum(h)*step(ed)), 1e-3))
    ax = Axis(gp; xlabel="deviation from central model / MAE",
              ylabel = showy ? "density" : "", yscale=log10, title=ttl, titlesize=TITLE-1,
              xlabelsize=LAB, ylabelsize=LAB, xticklabelsize=TICK, yticklabelsize=TICK,
              xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1)
    stairs!(ax, ed[1:end-1], d1; step=:post, color=:black, linewidth=1.8)
    stairs!(ax, ed[1:end-1], d2; step=:post, color=AMB, linewidth=1.8)
    ylims!(ax, 1e-3, maximum(vcat(d1, d2))*3)
end

devE = reduce(vcat, [co_Es[i] .- Es[i] for i in eachindex(Es)])
devF = vec(co_Fs .- Fs)
fig2 = Figure(size=(FIGW, 0.58FIGW), figure_padding=(6, 10, 4, 6))
calib_panel!(fig2[2,1], true_Es .- Es, devE, @sprintf("energy — coverage %.1f%%", covE), true)
calib_panel!(fig2[2,2], true_Fs .- Fs, devF, @sprintf("force — coverage %.1f%%", covF), false)
Legend(fig2[1, 1:2],
       [LineElement(color=:black, linewidth=2), LineElement(color=AMB, linewidth=2)],
       ["test error (DFT − central model)", "committee spread (member − central model)"];
       orientation=:horizontal, tellwidth=false, framevisible=false,
       labelsize=TICK, padding=(2,2,0,0), colgap=12)
colgap!(fig2.layout, 20); rowgap!(fig2.layout, 1, 2)
save("$outdir/calibration_constrained.pdf", fig2)
save("$outdir/calibration_constrained.png", fig2; px_per_unit=4)

# ── save the predictions so the figures can be redrawn without recomputing ──
serialize("$outdir/predictions_constrained.jls",
          (; true_Es, Es, co_Es, true_Fs, Fs, co_Fs,
             loE, hiE, loF, hiF, rmseE, rmseF, covE, covF,
             n_configs = ncfg, n_members = length(con_com),
             stride = 20, per_atom = true, secs = tpred))
@printf("\nenergy: RMSE %.4f eV/atom, coverage %.1f%%\n", rmseE, covE)
@printf("force : RMSE %.4f eV/Å,    coverage %.1f%%\n", rmseF, covF)
@printf("  → %s/parity_constrained.{pdf,png}\n", outdir)
@printf("  → %s/calibration_constrained.{pdf,png}\n", outdir)
@printf("  → %s/predictions_constrained.jls  (replot without recomputing)\n", outdir)

# ── phonon bands of the SAME ensemble ───────────────────────────────────────
# Uses con_com, the identical member list the test errors were computed from, so the
# two figures cannot describe different ensembles.
#
# Valid on the PREMADE undotted band path because every member is a_eq-pinned: the
# equilibrium is the same for all of them, so Σ_k θ_k D_k(q) is the exact operator for
# each.  ~10^4 members × 141 q-points is a few seconds of 3×3 eigendecompositions.
#
# At this ensemble size individual lines are meaningless (30,000 of them), so the
# figure shows ENVELOPES per branch: the full min/max across members, and the 5–95%
# band inside it.  The gap between the two is where the outliers live.
BPCACHE = "$SRC/bandpath_4x4x4_aref.jls"
if isfile(BPCACHE)
    bp = deserialize(BPCACHE)
    nq = length(bp.Bq); nb = 3bp.Np; nm = length(con_com)
    @printf("\nphonons: %d members × %d q-points on the cached %s\n",
            nm, nq, basename(BPCACHE)); flush(stdout)

    Fall = Array{Float64}(undef, nm, nb, nq)
    tph = @elapsed Threads.@threads for i in 1:nm
        θ = con_com[i]
        for iq in 1:nq
            ev = eigvals(Hermitian(reshape(bp.Bq[iq]*θ, nb, nb)))
            @inbounds for b in 1:nb
                Fall[i, b, iq] = sign(ev[b]) * sqrt(abs(ev[b])) * FREQ_THz
            end
        end
    end
    Fmean = bands(lin_params, bp)
    keepq = findall(bp.qnorm .>= 5e-2)
    minω  = [minimum(@view Fall[i, :, keepq]) for i in 1:nm]
    @printf("  %.1f s; min ω ∈ [%+.4f, %+.4f] THz, median %+.4f, %d/%d below -0.05\n",
            tph, minimum(minω), maximum(minω), median(minω),
            count(<(-0.05), minω), nm); flush(stdout)

    figp = Figure(size=(FIGW, 0.46FIGW), figure_padding=(6, 10, 4, 6))
    axp = Axis(figp[2, 1]; xlabel="Wave vector", ylabel="Frequency (THz)",
               title=@sprintf("constrained ensemble, %d members", nm), titlesize=TITLE,
               xlabelsize=LAB, ylabelsize=LAB, xticklabelsize=TICK, yticklabelsize=TICK,
               xticks=(bp.x_ticks, bp.labels), xgridvisible=false, ygridvisible=false,
               xtickalign=1, ytickalign=1, xticksize=4, yticksize=4)
    for b in 1:nb
        lo_b  = [minimum(@view Fall[:, b, iq]) for iq in 1:nq]
        hi_b  = [maximum(@view Fall[:, b, iq]) for iq in 1:nq]
        q05_b = [quantile(@view(Fall[:, b, iq]), 0.05) for iq in 1:nq]
        q95_b = [quantile(@view(Fall[:, b, iq]), 0.95) for iq in 1:nq]
        band!(axp, bp.x_vals, lo_b, hi_b;   color=(BLU, 0.15))
        band!(axp, bp.x_vals, q05_b, q95_b; color=(BLU, 0.35))
    end
    for b in 1:nb; lines!(axp, bp.x_vals, Fmean[b, :]; color=BLU, linewidth=1.8); end
    hlines!(axp, [0.0]; color=:black, linestyle=:dash, linewidth=1.0)
    vlines!(axp, bp.x_ticks; color=(:black, 0.22), linewidth=0.7)
    xlims!(axp, first(bp.x_vals), last(bp.x_vals))
    Legend(figp[1, 1],
           [LineElement(color=BLU, linewidth=2.2),
            PolyElement(color=(BLU, 0.35)), PolyElement(color=(BLU, 0.15)),
            LineElement(color=:black, linestyle=:dash, linewidth=1.6)],
           ["mean model", "5–95% of members", "full range", "ω = 0"];
           orientation=:horizontal, tellwidth=false, framevisible=false,
           labelsize=TICK, padding=(2,2,0,0), colgap=12)
    rowgap!(figp.layout, 1, 2)
    save("$outdir/bands_constrained.pdf", figp)
    save("$outdir/bands_constrained.png", figp; px_per_unit=4)
    serialize("$outdir/bands_constrained.jls",
              (; Fall, Fmean, minω, x_vals=bp.x_vals, x_ticks=bp.x_ticks,
                 labels=bp.labels, qnorm=bp.qnorm, a_ref=bp.a_ref, n_members=nm))
    @printf("  → %s/bands_constrained.{pdf,png}\n", outdir)
    @printf("  → %s/bands_constrained.jls  (Fall is %d×%d×%d, for replotting)\n",
            outdir, nm, nb, nq)
else
    @warn "no band-path cache at $BPCACHE — skipping the phonon figure"
end

# ── summary ─────────────────────────────────────────────────────────────────
open("$outdir/parity_calibration_summary.csv", "w") do io
    parity_calibration_header(io)
    for (tag, e) in preds; parity_calibration_row(io, tag, e.pr); end
end
println("\n══ SUMMARY (energies per atom) ═══════════════════════════════════")
@printf("%-14s %14s %14s %10s %10s\n",
        "ensemble", "E RMSE meV/at", "F RMSE eV/Å", "E cov %", "F cov %")
for (tag, e) in preds
    @printf("%-14s %14.4g %14.4g %10.1f %10.1f\n", tag,
            1000*pc_rmse(e.pr.pE, e.pr.tE), pc_rmse(e.pr.pF, e.pr.tF),
            pc_cover(e.pr.tE, e.pr.loE, e.pr.hiE), pc_cover(e.pr.tF, e.pr.loF, e.pr.hiF))
end
let u = preds[1].second.pr, c = preds[2].second.pr
    @printf("\ncoverage change from the phonon predicate: energy %+.1f pp, force %+.1f pp\n",
            pc_cover(c.tE, c.loE, c.hiE) - pc_cover(u.tE, u.loE, u.hiE),
            pc_cover(c.tF, c.loF, c.hiF) - pc_cover(u.tF, u.loF, u.hiF))
    @printf("RMSE change: energy %+.4g meV/atom, force %+.4g eV/Å\n",
            1000*(pc_rmse(c.pE, c.tE) - pc_rmse(u.pE, u.tE)),
            pc_rmse(c.pF, c.tF) - pc_rmse(u.pF, u.tF))
    println("  (negative coverage change is expected — rejection removes members and")
    println("   narrows the envelope; RMSE should barely move, same mean model both sides)")
end
serialize("$outdir/parity_calibration_predictions.jls",
          (; preds = Dict(tag => e.pr for (tag, e) in preds), stride, n_params,
             per_atom = true))
println("\nfigures → $outdir/{parity,calibration}_{unconstrained,constrained}.{pdf,png}")
println("summary → $outdir/parity_calibration_summary.csv")
