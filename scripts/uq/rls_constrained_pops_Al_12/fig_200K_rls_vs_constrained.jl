# fig_200K_rls_vs_constrained.jl — RLS vs constrained RLS after 200 K NPT (Al_12_4_6A_2).
#
# Layout and styling follow thermal_expansion_vs_experiment/fcc_compare_constrained_vs_naive.jl
# (left untouched — it reads a different run format):
#   LEFT  (full height)  phonon bands of both models, each at its own a(200 K), shared axis
#   RIGHT top            time-averaged RDF, RLS             (red)
#   RIGHT bottom         time-averaged RDF, constrained RLS (blue)
# Phonons: 4×4×4 force constants, dense Γ→K (N_per_seg 20,20,20,20,60), as the paper figure.
#
# Inputs (written by the pipeline in README.md):
#   npt/<m>/theta_used.csv          θ the MD ran with
#   npt/<m>/T200K/summary_row.csv   a(200 K) = mean ∛V/4 over production frames (step ≥ 10 000)
#   npt/<m>/rdf_200K.csv            make_rdf_coordination.jl, same production frames
# Outputs → fig_200K/: the figure, bands_<m>.csv, rdf_<m>.csv, metadata.csv
#
# Run:  julia --project -t 8 scripts/uq/rls_constrained_pops_Al_12/fig_200K_rls_vs_constrained.jl
#   env: NPT (default <this dir>/npt)  OUTDIR (default <this dir>/fig_200K)  FIGW (540)

include(joinpath(@__DIR__, "..", "..", "bandpath_phonon_uq", "lib.jl"))

T_K       = 200
N_per_seg = [20, 20, 20, 20, 60]
NPT       = get(ENV, "NPT", joinpath(@__DIR__, "npt"))
outdir    = get(ENV, "OUTDIR", joinpath(@__DIR__, "fig_200K")); mkpath(outdir)
MODELDIR  = abspath(joinpath(@__DIR__, "..", "..", "..", "models", "Al_12_4_6A_2_"))

# fast load (JSON + lin_params, no design matrix) — bandpath_Dk only uses dir and lin_params
model, _ = ACEpotentials.load_model("$MODELDIR/Al_12_4_6A_2.json")
lin      = vec(readdlm("$MODELDIR/lin_params.csv", ','))
result   = (; dir = MODELDIR, lin_params = lin)

M = map((("rls", "RLS"), ("constrained", "constrained RLS"))) do (m, lab)
    d  = "$NPT/$m"
    θ  = vec(readdlm("$d/theta_used.csv", ','))
    s  = readdlm("$d/T$(T_K)K/summary_row.csv", ','; header=true)
    row = Dict(zip(vec(string.(s[2])), vec(s[1])))
    a  = Float64(row["a_Ang"])
    rg = readdlm("$d/rdf_$(T_K)K.csv", ','; skipstart=1)
    bp = bandpath_Dk(result, model, :Al, a, 4; N_per_seg=N_per_seg)
    (; m, lab, θ, a, a_std=Float64(row["a_std_Ang"]), coord=Float64(row["mean_coord"]),
       fcc=string(row["still_fcc"]) == "true", a0=readdlm("$d/a0.csv", ','; skipstart=1)[1],
       r=Float64.(rg[:,1]), g=Float64.(rg[:,2]), bp, F=bands(θ, bp), mω=min_freq_stable(θ, bp))
end
rls, con = M

# Both paths are the same reduced-q path; |q| ∝ 1/a only stretches x uniformly, so RLS is
# drawn on the constrained model's x (exact: every segment scales by the same factor).
xc = con.bp.x_vals
xplot(x) = x.bp.x_vals .* (last(xc) / last(x.bp.x_vals))

# ── plot data ────────────────────────────────────────────────────────────────
for x in M
    writedlm("$outdir/bands_$(x.m).csv", vcat(["x_plot" "branch1_THz" "branch2_THz" "branch3_THz"], hcat(xplot(x), x.F')), ',')
    writedlm("$outdir/rdf_$(x.m).csv", vcat(["r_Ang" "g_r"], hcat(x.r, x.g)), ',')
end
writedlm("$outdir/band_ticks.csv", vcat(["label" "x_plot"], hcat(con.bp.labels, con.bp.x_ticks)), ',')
open("$outdir/metadata.csv", "w") do io
    println(io, "# NPT 4x4x4 (256 atoms), 0 Pa, T = $T_K K, Langevin + MC barostat, dt 1 fs, 10k equil + 20k prod, seed 1234 (both)")
    println(io, "# a_200K = mean cube-root(V)/4 over production frames; RDF over the same frames (200 bins, per-frame box)")
    println(io, "# phonons: 4x4x4 force constants at each model's own a_200K, N_per_seg 20,20,20,20,60; min_omega excludes |q| < 0.05")
    println(io, "# bands x_plot: the constrained model's path coordinate; RLS's (|q| scales as 1/a) rescaled onto it, exact since all segments scale equally")
    println(io, "# RLS left FCC at 200 K: its a_200K is cube-root(V)/4 of a transformed cell, not an FCC lattice constant")
    println(io, "model,a0_Ang,a_200K_Ang,a_200K_std_Ang,expansion_pct,mean_coord,still_fcc,min_omega_THz,theta_file")
    for x in M
        @printf(io, "%s,%.6f,%.6f,%.6f,%.3f,%.4f,%s,%.4f,%s\n", x.lab, x.a0, x.a, x.a_std, 100(x.a - x.a0)/x.a0,
                x.coord, x.fcc, x.mω, relpath("$NPT/$(x.m)/theta_used.csv", outdir))
    end
end
for x in M; @printf("%-16s a(200 K) = %.5f Å  ⟨coord⟩ = %.2f  min ω = %+.3f THz\n", x.lab, x.a, x.coord, x.mω); end

# ── figure (styling as the paper figure) ─────────────────────────────────────
FIGW = parse(Float64, get(ENV, "FIGW", "540"))
BLU, RED, ORA = RGBf(0.0, 0.447, 0.698), RGBf(0.80, 0.15, 0.15), RGBf(0.835, 0.369, 0.0)
TITLE, SUB, LAB, TICK, SMALL = 13, 12, 12, 11, 10
fig = Figure(size=(FIGW, 0.66FIGW), figure_padding=(6, 10, 4, 6))

ax1 = Axis(fig[1:2, 1]; xlabel="Wave vector", ylabel="Frequency (THz)", title="Phonon dispersion at a($(T_K) K)",
           titlesize=TITLE, xlabelsize=LAB, ylabelsize=LAB, xticklabelsize=TICK, yticklabelsize=TICK,
           xticks=(con.bp.x_ticks, con.bp.labels), xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1)
lo, hi = extrema(vcat(vec(rls.F), vec(con.F))); pad = 0.06(hi - lo)
band!(ax1, [first(xc), last(xc)], [lo - pad, lo - pad], [0.0, 0.0]; color=(RED, 0.07))
for b in axes(rls.F, 1); lines!(ax1, xplot(rls), rls.F[b, :]; color=(RED, 0.95), linewidth=1.5); end
for b in axes(con.F, 1); lines!(ax1, xc, con.F[b, :]; color=BLU, linewidth=1.6); end
hlines!(ax1, [0.0]; color=:black, linestyle=:dash, linewidth=0.9)
vlines!(ax1, con.bp.x_ticks; color=(:black, 0.22), linewidth=0.6)
xlims!(ax1, first(xc), last(xc)); ylims!(ax1, lo - pad, hi + 4pad)   # headroom for the legend
axislegend(ax1, [LineElement(color=RED, linewidth=2.2), LineElement(color=BLU, linewidth=2.2)],
           [@sprintf("RLS"), @sprintf("constrained RLS")];
           position=:rt, framevisible=true, labelsize=TICK, patchsize=(18, 2), padding=(5, 5, 3, 3), rowgap=1)
text!(ax1, 0.015, 0.985; text="(a)", space=:relative, align=(:left, :top), font=:bold, fontsize=TITLE)

shells(a, rmax) = filter(<(rmax), a .* sqrt.((1:16) ./ 2))     # ideal FCC shells r_n = a√(n/2)
rmax = min(maximum(rls.r), maximum(con.r))
axs = map(enumerate(((rls, RED, 0.70, "RLS", "(b)"), (con, BLU, 0.75, "Constrained RLS", "(c)")))) do (i, (x, col, α, ttl, tag))
    ax = Axis(fig[i, 2]; ylabel="g(r)", xlabel=i == 2 ? "r (Å)" : "", title=ttl, titlesize=SUB, titlecolor=col,
              xlabelsize=LAB, ylabelsize=LAB, xticklabelsize=TICK, yticklabelsize=TICK,
              xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1)
    barplot!(ax, x.r, x.g; width=step(range(x.r[1], x.r[end]; length=length(x.r))), color=(col, α), gap=0, strokewidth=0)
    vlines!(ax, shells(x.a, rmax); color=(ORA, 0.85), linestyle=:dash, linewidth=1.0)
    xlims!(ax, 0, rmax)
    i == 1 && (hidexdecorations!(ax; grid=false, ticks=false, minorticks=false);
               text!(ax, 0.985, 0.93; space=:relative, align=(:right, :top), fontsize=SMALL, color=ORA))
    text!(ax, 0.025, 0.95; text=tag, space=:relative, align=(:left, :top), font=:bold, fontsize=TITLE)
    ax
end
linkxaxes!(axs...)
colsize!(fig.layout, 1, Relative(0.54)); colgap!(fig.layout, 14); rowgap!(fig.layout, 6)

out = "$outdir/fig_200K_rls_vs_constrained"
save("$out.pdf", fig); save("$out.png", fig; px_per_unit=4)
println("figure + plot data → $outdir")
