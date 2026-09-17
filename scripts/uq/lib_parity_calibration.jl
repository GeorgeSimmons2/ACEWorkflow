# lib_parity_calibration.jl
#
# Paper-quality parity and calibration figures, extracted verbatim from
# hypercube_full_cloud_bands_Al_12_4_6A_2.jl so there is ONE copy of the plotting
# machinery rather than one per model.  Any script with a `pr` from
# committee_predictions can produce identical figures.
#
# Assumes the caller has already included scripts/bandpath_phonon_uq/lib.jl (for
# CairoMakie, Printf, Statistics and StatsBase's fit/Histogram).
#
# Each call writes ONE figure with two panels, (a) energy and (b) force:
#     parity_figure(pr, "hypercube from 73411 deltas", "fullcloud", outdir)
#     calibration_figure(pr, "hypercube from 73411 deltas", "fullcloud", outdir)
# → $outdir/parity_fullcloud.{pdf,png}, $outdir/calibration_fullcloud.{pdf,png}
#
# SIZING.  FIGW is the width the figure will be DISPLAYED at, in points.  Fonts stay
# at 13/12/11 pt whatever FIGW is — that is the entire point of building at final
# size.  540 pt assumes the figure spans \linewidth on its own; for a half-width slot
# pass FIGW=260 rather than scaling in LaTeX, and do NOT add a width= factor to
# \includegraphics either way.

PC_BLU = RGBf(0.0, 0.447, 0.698)
PC_ORN = RGBf(0.835, 0.369, 0.0)
PC_AMB = RGBf(0.902, 0.624, 0.0)
PC_TITLE, PC_LAB, PC_TICK = 13, 12, 11

# Older cached `pr` NamedTuples (and any caller that did not ask for it) have no
# per_atom field; those were total-energy runs.
pc_per_atom(pr) = hasproperty(pr, :per_atom) ? pr.per_atom : false

"RMSE with its unit, in meV/atom for per-atom energies where the eV number is unreadable"
function pc_rmse_label(pr, which::Symbol)
    if which === :F
        return "$(round(pc_rmse(pr.pF, pr.tF), sigdigits=3)) eV/Å"
    elseif pc_per_atom(pr)
        return "$(round(1000*pc_rmse(pr.pE, pr.tE), sigdigits=3)) meV/atom"
    else
        return "$(round(pc_rmse(pr.pE, pr.tE), sigdigits=3)) eV"
    end
end

pc_cover(t, lo, hi) = 100*(1 - mean((t .< lo) .| (t .> hi)))
pc_rmse(p, t)       = sqrt(mean((p .- t).^2))

function pc_parity!(ax, t, p, lo, hi, col)
    l = extrema([t; p])
    lines!(ax, collect(l), collect(l); color=:black, linestyle=:dash, linewidth=1.0)
    errorbars!(ax, t, p, p .- lo, hi .- p; whiskerwidth=4, linewidth=0.7, color=(col, 0.35))
    scatter!(ax, t, p; color=(col, 0.9), markersize=4)
end

# test-error distribution vs the ensemble's own predictive spread, both normalised by
# the MAE so the two are on one scale.  Floor at 1e-3 keeps empty bins on a log axis.
function pc_calib!(ax, t, p, spread)
    err = t .- p; mae = mean(abs.(err)); ne = err ./ mae; sp = spread ./ mae
    lim = maximum(abs.(vcat(ne, sp))); ed = range(-lim, lim; length=61)
    d1 = (h = fit(Histogram, ne, ed).weights; max.(h ./ (sum(h)*step(ed)), 1e-3))
    d2 = (h = fit(Histogram, sp, ed).weights; max.(h ./ (sum(h)*step(ed)), 1e-3))
    l1 = stairs!(ax, ed[1:end-1], d1; step=:post, color=:black, linewidth=1.8)
    l2 = stairs!(ax, ed[1:end-1], d2; step=:post, color=PC_AMB, linewidth=1.8)
    ylims!(ax, 1e-3, maximum(vcat(d1, d2))*3)
    return l1, l2
end

"""
    parity_figure(pr, ttl, tag, outdir; FIGW=540.0)

DFT vs ACE with committee error bars: (a) energy, (b) force.  `pr` is a
committee_predictions result.  Writes `\$outdir/parity_\$(tag).{pdf,png}`.
"""
function parity_figure(pr, ttl, tag, outdir; FIGW=540.0)
    pa = pc_per_atom(pr)
    eax = pa ? "energy (eV/atom)" : "energy (eV)"
    fig = Figure(size=(FIGW, 0.60FIGW), figure_padding=(6, 10, 4, 6))
    Label(fig[1, 1:2], ttl; fontsize=PC_TICK, padding=(0, 0, 0, 2))
    for (c, (t, p, lo, hi, xl, yl, col, rms, letter)) in enumerate((
            (pr.tE, pr.pE, pr.loE, pr.hiE, "DFT $eax",
             "ACE $eax", PC_BLU, pc_rmse_label(pr, :E), "(a)"),
            (pr.tF, pr.pF, pr.loF, pr.hiF, "DFT force (eV/Å)",
             "ACE force (eV/Å)", PC_ORN, pc_rmse_label(pr, :F), "(b)")))
        ax = Axis(fig[2, c]; xlabel=xl, ylabel=yl,
                  title="RMSE $rms  |  cov $(round(pc_cover(t, lo, hi), digits=1))%",
                  titlesize=PC_TITLE-1, xlabelsize=PC_LAB, ylabelsize=PC_LAB,
                  xticklabelsize=PC_TICK, yticklabelsize=PC_TICK, xgridvisible=false,
                  ygridvisible=false, xtickalign=1, ytickalign=1, aspect=1)
        pc_parity!(ax, t, p, lo, hi, col)
        text!(ax, 0.03, 0.97; text=letter, space=:relative, align=(:left, :top),
              font=:bold, fontsize=PC_TITLE)
    end
    colgap!(fig.layout, 20); rowgap!(fig.layout, 1, 2)
    save(joinpath(outdir, "parity_$(tag).pdf"), fig)
    save(joinpath(outdir, "parity_$(tag).png"), fig; px_per_unit=4)
    @printf("parity  [%-12s] → %s/parity_%s.{pdf,png}\n", tag, outdir, tag)
    return fig
end

"""
    calibration_figure(pr, ttl, tag, outdir; FIGW=540.0)

Test error vs ensemble spread, log density: (a) energy, (b) force.  Writes
`\$outdir/calibration_\$(tag).{pdf,png}`.
"""
function calibration_figure(pr, ttl, tag, outdir; FIGW=540.0)
    fig = Figure(size=(FIGW, 0.58FIGW), figure_padding=(6, 10, 4, 6))
    Label(fig[1, 1:2], ttl; fontsize=PC_TICK, padding=(0, 0, 0, 2))
    # LineElement rather than plot handles: assignments inside the loop below do not
    # escape its scope, which is what broke the first version of this figure.
    Legend(fig[2, 1:2],
           [LineElement(color=:black, linewidth=2),
            LineElement(color=PC_AMB, linewidth=2)],
           ["test error (truth − model)", "ensemble spread (member − model)"];
           orientation=:horizontal, tellwidth=false, framevisible=true,
           labelsize=PC_TICK, padding=(4,4,2,2))
    for (c, (dev, tt, pp, lo, hi, nm, letter)) in enumerate((
            (pr.dE, pr.tE, pr.pE, pr.loE, pr.hiE, "energy", "(a)"),
            (pr.dF, pr.tF, pr.pF, pr.loF, pr.hiF, "force",  "(b)")))
        ax = Axis(fig[3, c]; xlabel="deviation from point model / MAE",
                  ylabel = c == 1 ? "density" : "", yscale=log10,
                  title="$nm — coverage $(round(pc_cover(tt, lo, hi), digits=1))%",
                  titlesize=PC_TITLE-1, xlabelsize=PC_LAB, ylabelsize=PC_LAB,
                  xticklabelsize=PC_TICK, yticklabelsize=PC_TICK, xgridvisible=false,
                  ygridvisible=false, xtickalign=1, ytickalign=1)
        pc_calib!(ax, tt, pp, dev)
        text!(ax, 0.03, 0.97; text=letter, space=:relative, align=(:left, :top),
              font=:bold, fontsize=PC_TITLE)
    end
    colgap!(fig.layout, 20)
    rowgap!(fig.layout, 1, 2); rowgap!(fig.layout, 2, 6)
    save(joinpath(outdir, "calibration_$(tag).pdf"), fig)
    save(joinpath(outdir, "calibration_$(tag).png"), fig; px_per_unit=4)
    @printf("calib   [%-12s] → %s/calibration_%s.{pdf,png}\n", tag, outdir, tag)
    return fig
end

"one CSV row per committee: RMSE, coverage and bias for energy and force"
function parity_calibration_row(io, tag, pr)
    @printf(io, "%s,%d,%s,%.6g,%.6g,%.4f,%.4f,%.6g,%.6g\n", tag, pr.n,
            pc_per_atom(pr) ? "eV_per_atom" : "eV_total",
            pc_rmse(pr.pE, pr.tE), pc_rmse(pr.pF, pr.tF),
            pc_cover(pr.tE, pr.loE, pr.hiE), pc_cover(pr.tF, pr.loF, pr.hiF),
            mean(pr.pE .- pr.tE), mean(pr.pF .- pr.tF))
end
parity_calibration_header(io) =
    println(io, "committee,n_configs,energy_units,E_rmse,F_rmse_eV_per_A,E_coverage_pct,F_coverage_pct,E_bias,F_bias_eV_per_A")

# ═════════════════════════════════════════════════════════════════════════════
#  Fast committee predictions: build the design row ONCE per configuration, then
#  every member is a matvec.
# ═════════════════════════════════════════════════════════════════════════════
# committee_predictions in scripts/bandpath_phonon_uq/lib.jl calls @committee, which
# re-evaluates the potential for EVERY member of EVERY configuration — O(n_cfg × n_mem)
# full ACE evaluations.  But the model is LINEAR in the parameters:
#
#     E(cfg; θ) = B_E(cfg) · θ ,     F(cfg; θ) = B_F(cfg) · θ
#
# so one call to energy_forces_virial_basis per configuration gives the whole design
# row, and the committee collapses to two matrix products.  Cost becomes O(n_cfg) basis
# evaluations plus BLAS, independent of committee size — for 500 members that is two
# orders of magnitude.
#
# Returns the SAME NamedTuple as committee_predictions, so parity_figure and
# calibration_figure take it unchanged.  Verified against the slow path to ~1e-12.
function committee_predictions_fast(model, committee, test_xyz; stride=10,
                                    point_params, per_atom::Bool=false,
                                    nthreads::Int=Threads.nthreads())
    Θ  = reduce(hcat, committee)                     # n_params × n_members
    tc = ExtXYZ.load(test_xyz)[1:stride:end]
    n  = length(tc)

    # PARALLEL OVER CONFIGURATIONS.  After the linearisation the cost is dominated by
    # energy_forces_virial_basis, one call per configuration and independent between
    # them; the two matmuls are far too small for BLAS threading to help (684 × n_mem
    # against 3·n_atoms × 684).  Each task gets its OWN model copy: basis evaluation
    # uses the model's internal scratch buffers, so a shared model is a data race.
    # Results are collected per configuration and concatenated IN ORDER, so the output
    # is identical to the serial version rather than merely equivalent.
    out = Vector{Any}(undef, n)
    nt  = clamp(nthreads, 1, min(Threads.nthreads(), n))
    @sync for k in 1:nt
        Threads.@spawn begin
            m = nt == 1 ? model : deepcopy(model)
            for i in k:nt:n
                cfg = tc[i]
                nat = per_atom ? length(cfg) : 1
                b   = ACEpotentials.Models.energy_forces_virial_basis(cfg, m)

                BE  = ustrip.(u"eV", b.energy)               # n_params
                eco = (transpose(BE) * Θ)[:] ./ nat          # n_members
                pe  = dot(BE, point_params) / nat
                te  = ustrip(cfg[:dft_energy]) / nat

                if haskey(cfg[1], :dft_forces)
                    bf = b.forces                            # n_atoms × n_params of SVector{3}
                    na, npar = size(bf)
                    BF = Matrix{Float64}(undef, 3na, npar)   # atom-major, matching the DFT layout
                    @inbounds for kk in 1:npar, ii in 1:na
                        v = ustrip.(bf[ii, kk])
                        BF[3ii-2, kk] = v[1]; BF[3ii-1, kk] = v[2]; BF[3ii, kk] = v[3]
                    end
                    Fco = BF * Θ                             # 3na × n_members
                    pf  = BF * point_params
                    tf  = reduce(vcat, ustrip.([at[:dft_forces] for at in cfg]))
                    out[i] = (; pe, te, elo=minimum(eco), ehi=maximum(eco), ed=eco .- pe,
                                pf, tf, flo=vec(minimum(Fco; dims=2)),
                                fhi=vec(maximum(Fco; dims=2)), fd=vec(Fco .- pf))
                else
                    out[i] = (; pe, te, elo=minimum(eco), ehi=maximum(eco), ed=eco .- pe,
                                pf=Float64[], tf=Float64[], flo=Float64[],
                                fhi=Float64[], fd=Float64[])
                end
            end
        end
    end

    pE = [o.pe for o in out]; tE = [o.te for o in out]
    loE = [o.elo for o in out]; hiE = [o.ehi for o in out]
    dE = reduce(vcat, (o.ed for o in out))
    pF = reduce(vcat, (o.pf for o in out)); tF = reduce(vcat, (o.tf for o in out))
    loF = reduce(vcat, (o.flo for o in out)); hiF = reduce(vcat, (o.fhi for o in out))
    dF = reduce(vcat, (o.fd for o in out))
    return (; pE, tE, loE, hiE, pF, tF, loF, hiF, dE, dF, n, per_atom)
end
