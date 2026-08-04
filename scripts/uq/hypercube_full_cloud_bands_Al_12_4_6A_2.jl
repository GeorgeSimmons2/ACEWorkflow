# hypercube_full_cloud_bands_Al_12_4_6A_2.jl
#
# Take the cutting-plane-constrained FULL top-50%-leverage cloud (73,411 stable
# members from cutting_plane_full_cloud_Al_12_4_6A_2.jl), build the POPS hypercube
# around it, rejection-sample, and plot the resulting phonon bands.
#
# The legacy pipeline built its hypercube from a 30-member stratified forest. With
# 73k constrained members the covariance is enormously better resolved, so the
# retained eigendirections and their bounds are the interesting comparison -- both
# are reported, and panel (b) repeats the whole procedure on the 30-member
# committee for a like-for-like picture.
#
# FIGURE SIZING. Makie's `size` is in POINTS for vector output, so a figure built
# at 900 pt and dropped in at width=\linewidth (~510 pt) has its text scaled by
# 0.57 -- 11 pt labels land at ~6 pt. Everything here is built at its final
# display width with 11-13 pt fonts, so Overleaf does no rescaling and in-figure
# text matches MLST body text. Do NOT add a width= scale factor in \includegraphics.
#
# Run: julia --project -t 8 scripts/uq/hypercube_full_cloud_bands_Al_12_4_6A_2.jl [n_members]

include(joinpath(@__DIR__, "..", "bandpath_phonon_uq", "lib.jl"))
using SparseArrays, LinearAlgebra, Serialization, Random
Random.seed!(1234)

element, dataset = :Al, ""
N_cell_fc, N_per_seg = 4, [20, 20, 20, 20, 60]
cut_margin_THz, qΓtol = 0.15, 5e-2
n_members  = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 30
max_attempts = 5_000_000

result = load_model(element, 12, 4, 6, 2; dataset_name=dataset)
model  = result.model; lin_params = result.lin_params; n_params = length(lin_params)
P = result.P
RES    = "$(result.dir)/results"
outdir = "$RES/cutting_plane_full_cloud"; mkpath(outdir)

# ── geometry, Born rows, band path (same as everywhere else) ─────────────────
a_eq = ACEWorkflow.relax_lattice_constant(model, element)
sys0 = ACEWorkflow.Elasticity.reference_system(element; a=a_eq)
L0   = ustrip.(ACEWorkflow.Elasticity.lattice_matrix(sys0.cell.cell_vectors))
eV_to_GPa = 160.2176621/abs(det(L0))
H_el = elastic_hessian_basis(model; element=element, a=a_eq)
c11_0 = reshape(H_el,36,n_params)[1,:]; c12_0 = reshape(H_el,36,n_params)[7,:]; c44_0 = reshape(H_el,36,n_params)[22,:]
lattice_basis(a) = ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(ACEWorkflow.Elasticity.reference_system(element; a=a), model))
b_prime        = ForwardDiff.derivative(lattice_basis, a_eq)
b_double_prime = ForwardDiff.derivative(a -> ForwardDiff.derivative(lattice_basis, a), a_eq)
born_rows  = vcat(c44_0', (c11_0.-c12_0)', (c11_0.+2 .*c12_0)', b_double_prime')
born_lower = [0.1, 1.0, 0.1, 1e-9]

bp = bandpath_Dk(result, model, element, a_eq, N_cell_fc; N_per_seg=N_per_seg)
ω2_cut = (cut_margin_THz / FREQ_THz)^2
@printf("a_eq = %.5f Å; band path %d q-points\n", a_eq, length(bp.Bq)); flush(stdout)

# ── the two clouds ───────────────────────────────────────────────────────────
θ_mean = vec(readdlm("$RES/bandpath_undotted_ncell4_densek/theta_mean.csv", ','))
full   = deserialize("$outdir/committee_stable.jls")
Θ_full = full.Θ                                            # 91 × N, stable only
Θ_30   = Matrix(readdlm("$RES/bandpath_undotted_ncell4_densek/committee_repaired.csv", ',')')
@printf("full cloud %d members | legacy forest %d members\n", size(Θ_full,2), size(Θ_30,2))
flush(stdout)

function build_and_sample(Θ, label)
    deltas = Matrix(Θ' .- θ_mean')                          # N × K
    heig, hb = hypercube(deltas)
    widths = hb[2,:] .- hb[1,:]
    @printf("\n[%s] hypercube: %d directions retained, width mean %.4g max %.4g\n",
            label, size(heig,2), mean(widths), maximum(widths)); flush(stdout)
    K_ref = dot(θ_mean, b_double_prime)
    nck = Ref(0); nborn = Ref(0); naeq = Ref(0)
    pred = θ -> begin
        nck[] += 1
        all(born_lower .<= born_rows*θ) || return false
        nborn[] += 1
        abs(dot(b_prime, θ .- θ_mean)/K_ref) <= 0.1 || return false
        naeq[] += 1
        min_freq_stable(θ, bp; qΓtol=qΓtol) >= cut_margin_THz - 1e-6
    end
    t = @elapsed (mat, _) = rejection_sample_hypercube(heig, hb, θ_mean, pred;
                              number_of_committee_members=n_members,
                              max_attempts=max_attempts)
    @printf("[%s] funnel: %d drawn → %d Born → %d a_eq → %d accepted (%.3f%%)  [%.1f s]\n",
            label, nck[], nborn[], naeq[], n_members, 100n_members/max(nck[],1), t)
    members = [mat[:, i] for i in 1:size(mat,2)]
    mf = [min_freq_stable(θ, bp; qΓtol=qΓtol) for θ in members]
    @printf("[%s] sampled min ω ∈ [%.4f, %.4f] THz\n", label, minimum(mf), maximum(mf))
    flush(stdout)
    return members, heig, hb, 100n_members/max(nck[],1)
end

mem_full, heig_f, hb_f, acc_f = build_and_sample(Θ_full, "full cloud")
mem_30,   heig_3, hb_3, acc_3 = build_and_sample(Θ_30,   "30-member")

# ── why is the sampled full-cloud committee so much more dispersed? ──────────
# Two separable causes:
#   (i)  RANK. 30 points span at most 30 directions in 91-dim parameter space, so
#        the forest hypercube is rank-deficient and cannot move at all in the
#        remaining directions. The full cloud resolves many more.
#   (ii) BOX LOOSENESS. hypercube() takes percentile_clipping = 0, so the bounds
#        are the EXTREMA of the projections. Extrema grow with sample size, and a
#        box in d dimensions puts essentially all of its volume in corners that no
#        actual POPS member occupies. Sampling the box therefore explores far
#        outside the cloud it was built from.
# Compare the ACTUAL constrained members against the hypercube-SAMPLED ones: if the
# real members are tight and the sampled ones wild, (ii) dominates.
function band_span(members)
    hi = Float64[]; lo = Float64[]
    for θ in members
        F = bands(θ, bp); push!(hi, maximum(F)); push!(lo, minimum(F))
    end
    return lo, hi
end
rng_idx  = randperm(size(Θ_full,2))[1:400]
lo_a, hi_a = band_span([Θ_full[:, j] for j in rng_idx])       # actual constrained members
lo_s, hi_s = band_span(mem_full)                              # hypercube-sampled
lo_3, hi_3 = band_span([Θ_30[:, j] for j in 1:size(Θ_30,2)])
println("\n── dispersion: actual members vs hypercube draws ──")
@printf("  actual full-cloud members (n=400) : ω_max ∈ [%.2f, %.2f] THz, median %.2f\n",
        minimum(hi_a), maximum(hi_a), median(hi_a))
@printf("  hypercube draws from that cloud   : ω_max ∈ [%.2f, %.2f] THz, median %.2f\n",
        minimum(hi_s), maximum(hi_s), median(hi_s))
@printf("  actual forest members (n=%d)       : ω_max ∈ [%.2f, %.2f] THz, median %.2f\n",
        length(hi_3), minimum(hi_3), maximum(hi_3), median(hi_3))
@printf("  → sampled/actual median ω_max ratio = %.2f\n", median(hi_s)/median(hi_a))
flush(stdout)

# ── figure: MLST-sized text ──────────────────────────────────────────────────
# Built at 540 pt wide = the width it will be displayed at, so no rescaling.
BLU = RGBf(0.0, 0.447, 0.698); GRY = RGBAf(0.45, 0.45, 0.45, 0.35)
TITLE, LAB, TICK = 13, 12, 11

F_all = vcat([bands(θ, bp) for θ in mem_full], [bands(θ, bp) for θ in mem_30])
lo = minimum(minimum.(F_all)); hi = maximum(maximum.(F_all))
pad = 0.06*(hi-lo); ylim = (min(lo-pad, -0.4), hi+pad)

fig = Figure(size=(540, 350), figure_padding=(6, 10, 4, 6))
# Both panels are constrain → hypercube → rejection-sample. They differ ONLY in
# which set of constrained deltas the hypercube was built from, so the titles say
# that rather than naming the panels after their source cloud.
Label(fig[1, 1:2],
      "constrained POPS deltas → hypercube → rejection sample (30 drawn)";
      fontsize=TICK, padding=(0, 0, 0, 2))
for (col, (members, ttl, acc)) in enumerate((
        (mem_full, "hypercube from $(size(Θ_full,2)) deltas", acc_f),
        (mem_30,   "hypercube from $(size(Θ_30,2)) deltas",   acc_3)))
    ax = Axis(fig[2, col]; xlabel="Wave vector",
              ylabel = col == 1 ? "Frequency (THz)" : "",
              title=ttl, titlesize=TITLE, xlabelsize=LAB, ylabelsize=LAB,
              xticklabelsize=TICK, yticklabelsize=TICK,
              xticks=(bp.x_ticks, bp.labels), xgridvisible=false, ygridvisible=false,
              xtickalign=1, ytickalign=1, xticksize=4, yticksize=4)
    for θ in members
        Fθ = bands(θ, bp)
        for b in 1:3bp.Np; lines!(ax, bp.x_vals, Fθ[b,:]; color=GRY, linewidth=0.8); end
    end
    Fm = bands(θ_mean, bp)
    for b in 1:3bp.Np; lines!(ax, bp.x_vals, Fm[b,:]; color=BLU, linewidth=1.8); end
    hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=1.0)
    vlines!(ax, bp.x_ticks; color=(:black, 0.22), linewidth=0.7)
    xlims!(ax, first(bp.x_vals), last(bp.x_vals)); ylims!(ax, ylim...)
    col == 2 && hideydecorations!(ax; grid=false, ticks=false, minorticks=false)
    text!(ax, 0.03, 0.97; text=(col==1 ? "(a)" : "(b)"), space=:relative,
          align=(:left,:top), font=:bold, fontsize=TITLE)
    # top-right: bottom-right collides with the Γ tick label and the zero line
    text!(ax, 0.97, 0.97; text="accept $(round(acc; digits=2))%", space=:relative,
          align=(:right,:top), fontsize=TICK-1, color=:gray30)
end
colgap!(fig.layout, 20)      # 8 let panel (a)'s "K" collide with panel (b)'s "Γ"
save("$outdir/bands_hypercube_full_cloud.pdf", fig)
save("$outdir/bands_hypercube_full_cloud.png", fig; px_per_unit=4)
@printf("\nfigure → %s/bands_hypercube_full_cloud.{pdf,png}\n", outdir)

writedlm("$outdir/committee_rejection_full_cloud.csv", reduce(hcat, mem_full)', ',')

# ── test-set parity + calibration ────────────────────────────────────────────
# lib.jl's parity_plot/calibration_hist hardcode 10-11 pt fonts on a 330-380 pt
# canvas, i.e. sized for a half-width panel. Re-done here at full display width
# with MLST-sized text; the statistics are computed exactly as lib.jl does.
TEST   = joinpath(@__DIR__, "..", "..", "data", "Al", "manual_df_test_Al.xyz")
stride = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 20
@printf("\n── test-set predictions (stride %d) ──\n", stride); flush(stdout)
pr_f = committee_predictions(model, mem_full, TEST; stride=stride, point_params=θ_mean)
pr_3 = committee_predictions(model, mem_30,   TEST; stride=stride, point_params=θ_mean)
@printf("  %d test configurations\n", pr_f.n); flush(stdout)

cover(t, lo, hi) = 100*(1 - mean((t .< lo) .| (t .> hi)))
rmse(p, t)       = sqrt(mean((p .- t).^2))

function parity!(ax, t, p, lo, hi, col)
    l = extrema([t; p])
    lines!(ax, collect(l), collect(l); color=:black, linestyle=:dash, linewidth=1.0)
    errorbars!(ax, t, p, p .- lo, hi .- p; whiskerwidth=4, linewidth=0.7, color=(col, 0.35))
    scatter!(ax, t, p; color=(col, 0.9), markersize=4)
end

fig2 = Figure(size=(540, 540), figure_padding=(6, 10, 4, 6))
rowlab = ["hypercube from\n73k deltas", "hypercube from\n30 deltas"]
for (r, pr) in enumerate((pr_f, pr_3))
    axE = Axis(fig2[r, 1]; xlabel="DFT energy (eV)", ylabel="$(rowlab[r])\nACE energy (eV)",
               title="RMSE $(round(rmse(pr.pE, pr.tE), sigdigits=3)) eV  |  cov $(round(cover(pr.tE, pr.loE, pr.hiE), digits=1))%",
               titlesize=TITLE-1, xlabelsize=LAB, ylabelsize=LAB,
               xticklabelsize=TICK, yticklabelsize=TICK, xgridvisible=false,
               ygridvisible=false, xtickalign=1, ytickalign=1, aspect=1)
    parity!(axE, pr.tE, pr.pE, pr.loE, pr.hiE, BLU)
    axF = Axis(fig2[r, 2]; xlabel="DFT force (eV/Å)", ylabel="ACE force (eV/Å)",
               title="RMSE $(round(rmse(pr.pF, pr.tF), sigdigits=3)) eV/Å  |  cov $(round(cover(pr.tF, pr.loF, pr.hiF), digits=1))%",
               titlesize=TITLE-1, xlabelsize=LAB, ylabelsize=LAB,
               xticklabelsize=TICK, yticklabelsize=TICK, xgridvisible=false,
               ygridvisible=false, xtickalign=1, ytickalign=1, aspect=1)
    parity!(axF, pr.tF, pr.pF, pr.loF, pr.hiF, RGBf(0.835,0.369,0.0))
end
save("$outdir/parity_full_cloud.pdf", fig2); save("$outdir/parity_full_cloud.png", fig2; px_per_unit=4)
@printf("parity → %s/parity_full_cloud.{pdf,png}\n", outdir); flush(stdout)

# calibration: test-error distribution vs the ensemble's own predictive spread
function calib!(ax, t, p, spread)
    err = t .- p; mae = mean(abs.(err)); ne = err ./ mae; sp = spread ./ mae
    lim = maximum(abs.(vcat(ne, sp))); ed = range(-lim, lim; length=61)
    d1 = (h = fit(Histogram, ne, ed).weights; max.(h ./ (sum(h)*step(ed)), 1e-3))
    d2 = (h = fit(Histogram, sp, ed).weights; max.(h ./ (sum(h)*step(ed)), 1e-3))
    l1 = stairs!(ax, ed[1:end-1], d1; step=:post, color=:black, linewidth=1.8)
    l2 = stairs!(ax, ed[1:end-1], d2; step=:post, color=RGBf(0.902,0.624,0.0), linewidth=1.8)
    ylims!(ax, 1e-3, maximum(vcat(d1, d2))*3)
    return l1, l2
end

fig3 = Figure(size=(540, 460), figure_padding=(6, 10, 4, 6))
for (r, pr) in enumerate((pr_f, pr_3))
    for (c, (dev, tt, pp, lo, hi, nm)) in enumerate((
            (pr.dE, pr.tE, pr.pE, pr.loE, pr.hiE, "energy"),
            (pr.dF, pr.tF, pr.pF, pr.loF, pr.hiF, "force")))
        ax = Axis(fig3[r+1, c]; xlabel="deviation from point model / MAE",
                  ylabel = c == 1 ? "$(rowlab[r])\ndensity" : "", yscale=log10,
                  title="$nm — coverage $(round(cover(tt, lo, hi), digits=1))%",
                  titlesize=TITLE-1, xlabelsize=LAB, ylabelsize=LAB,
                  xticklabelsize=TICK, yticklabelsize=TICK, xgridvisible=false,
                  ygridvisible=false, xtickalign=1, ytickalign=1)
        calib!(ax, tt, pp, dev)
    end
end
# LineElement rather than plot handles: assignments inside the loop above do not
# escape its scope at top level, which is what broke the first version.
Legend(fig3[1, 1:2],
       [LineElement(color=:black, linewidth=2),
        LineElement(color=RGBf(0.902,0.624,0.0), linewidth=2)],
       ["test error (truth − model)", "ensemble spread (member − model)"];
       orientation=:horizontal, tellwidth=false, framevisible=true,
       labelsize=TICK, padding=(4,4,2,2))
save("$outdir/calibration_full_cloud.pdf", fig3); save("$outdir/calibration_full_cloud.png", fig3; px_per_unit=4)
@printf("calibration → %s/calibration_full_cloud.{pdf,png}\n", outdir)

println("\n══ COVERAGE ═══════════════════════════════════════════════════")
@printf("%-14s %10s %10s %12s %12s\n", "cloud", "E cov %", "F cov %", "E RMSE eV", "F RMSE eV/Å")
for (nm, pr) in (("full cloud", pr_f), ("forest (30)", pr_3))
    @printf("%-14s %10.1f %10.1f %12.4g %12.4g\n", nm,
            cover(pr.tE, pr.loE, pr.hiE), cover(pr.tF, pr.loF, pr.hiF),
            rmse(pr.pE, pr.tE), rmse(pr.pF, pr.tF))
end

open("$outdir/hypercube_summary.csv", "w") do io
    println(io, "cloud,n_members,n_directions,width_mean,width_max,acceptance_pct,E_coverage_pct,F_coverage_pct,E_rmse_eV,F_rmse_eV_per_A")
    for (nm, Θ, he, hb, ac, pr) in (("full_cloud", Θ_full, heig_f, hb_f, acc_f, pr_f),
                                    ("forest_30",  Θ_30,   heig_3, hb_3, acc_3, pr_3))
        @printf(io, "%s,%d,%d,%.6g,%.6g,%.4f,%.3f,%.3f,%.6g,%.6g\n", nm, size(Θ,2), size(he,2),
                mean(hb[2,:].-hb[1,:]), maximum(hb[2,:].-hb[1,:]), ac,
                cover(pr.tE, pr.loE, pr.hiE), cover(pr.tF, pr.loF, pr.hiF),
                rmse(pr.pE, pr.tE), rmse(pr.pF, pr.tF))
    end
end
println("summary → $outdir/hypercube_summary.csv")
