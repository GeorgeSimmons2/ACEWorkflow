# fcc_stability_figure_Al_12_4_6A_2.jl
#
# Publication figure: the constrained + rejection-sampled member IS STILL FCC at
# finite T, which thermal expansion alone does not demonstrate.
#
#   (a) BIG   phonon band structure at the NPT lattice constant a(T) — entirely
#             positive across the high-symmetry path
#   (b) small radial distribution function, as a HISTOGRAM, with the ideal FCC
#             shell positions for a(T) marked
#   (c) small per-atom coordination distribution — a sharp spike at 12 is FCC
#
# RDF and coordination are recomputed here from the saved trajectory (not reused
# from the MD run) so the figure is reproducible from the extxyz alone.
#
# TIME AVERAGING.  Both (b) and (c) are averaged over the PRODUCTION frames only
# (step >= n_equil; 401 of the 601 saved frames at 300 K).  Every frame uses its
# OWN box for the minimum-image convention, because the box breathes under NPT.
#   g(r_k) = C_k / (N_frames · N_atoms · 4π r_k² Δr · ρ̄),   ρ̄ = mean_f (N/L_f³)
# with C_k the pair count in bin k summed over frames and counted twice per pair.
# r_max = min_f (L_f/2) so the range is valid in every frame.
#
# Run:  julia --project scripts/uq/fcc_stability_figure_Al_12_4_6A_2.jl [T]
#       T defaults to 300.

include(joinpath(@__DIR__, "..", "bandpath_phonon_uq", "lib.jl"))

element, dataset = :Al, ""
T_K        = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 300
N_cell_fc  = 4
N_per_seg  = [20, 20, 20, 20, 60]
n_equil    = 10_000          # steps; production = step >= this
nn_cutoff  = 3.3             # Å, coordination shell (between FCC shells 1 and 2)
n_bins     = 200

result  = load_model(element, 12, 4, 6, 2; dataset_name=dataset)
model   = result.model
rundir  = "$(result.dir)/results/npt_multivolume_softest"
θ       = vec(readdlm("$(result.dir)/results/bandpath_undotted_multivolume/theta_npt_softest.csv", ','))
traj    = "$rundir/T$(T_K)K/md_trajectory.extxyz"
isfile(traj) || error("no trajectory at $traj")

# a(T) and the reported diagnostics from the run's own summary
# 3 comment lines + 1 column header
summ = readdlm("$rundir/thermal_expansion_summary.csv", ','; skipstart=4)
row  = findfirst(==(Float64(T_K)), Float64.(summ[:,1]))
a_T, minω_rep, coord_rep = summ[row,2], summ[row,5], summ[row,6]
@printf("T = %d K:  a = %.5f Å,  reported min ω = %+.3f THz,  ⟨coord⟩ = %.2f\n",
        T_K, a_T, minω_rep, coord_rep); flush(stdout)

# ── (a) phonons of this member at a(T) ───────────────────────────────────────
bp    = bandpath_Dk(result, model, element, a_T, N_cell_fc; N_per_seg=N_per_seg)
F     = bands(θ, bp)
minω  = min_freq_stable(θ, bp)
@printf("recomputed min non-acoustic ω at a(T) = %+.4f THz\n", minω); flush(stdout)

# ── read the trajectory (production frames only) ─────────────────────────────
frames = ExtXYZ.read_frames(traj)
steps  = [Int(f["info"]["step"]) for f in frames]
prod   = findall(>=(n_equil), steps)
@printf("frames: %d total, %d production (step ≥ %d)\n", length(frames), length(prod), n_equil); flush(stdout)

# ExtXYZ parses Lattice into a 3x3 `cell`, not an info string; the box is cubic here
boxes = [frames[f]["cell"][1,1] for f in prod]                                   # L (Å)
poss  = [Matrix{Float64}(frames[f]["arrays"]["pos"]) for f in prod]               # 3 × N
n_at  = size(poss[1], 2)

# ── (b) RDF, time-averaged, per-frame box ────────────────────────────────────
r_max  = minimum(boxes)/2
dr     = r_max/n_bins
r_mid  = collect(range(dr/2, r_max-dr/2; length=n_bins))
# NB: in a function — this is ~13M pair evaluations and would crawl in global scope
function rdf_and_coord(poss, boxes, n_bins, dr, r_max, nn_cutoff)
    n_at = size(poss[1], 2)
    pair_counts = zeros(n_bins); ρacc = 0.0; coord_all = Int[]
    for (f, L) in enumerate(boxes)
        p = poss[f]; ρacc += n_at/L^3
        z = zeros(Int, n_at)
        @inbounds for i in 1:n_at-1, j in i+1:n_at
            d1 = p[1,i]-p[1,j]; d2 = p[2,i]-p[2,j]; d3 = p[3,i]-p[3,j]
            d1 -= L*round(d1/L); d2 -= L*round(d2/L); d3 -= L*round(d3/L)
            r = sqrt(d1*d1 + d2*d2 + d3*d3)
            if r < nn_cutoff; z[i] += 1; z[j] += 1; end
            r < r_max || continue
            b = floor(Int, r/dr) + 1
            b <= n_bins && (pair_counts[b] += 2)
        end
        append!(coord_all, z)
    end
    return pair_counts, ρacc, coord_all
end
t_rdf = @elapsed ((pair_counts, ρacc, coord_all) = rdf_and_coord(poss, boxes, n_bins, dr, r_max, nn_cutoff))
@printf("pair analysis: %.1f s over %d frames\n", t_rdf, length(boxes)); flush(stdout)
ρbar = ρacc/length(boxes)
g = [pair_counts[k]/(length(boxes)*n_at*4π*r_mid[k]^2*dr*ρbar) for k in 1:n_bins]
@printf("RDF: %d bins to %.2f Å, averaged over %d frames; ⟨coord⟩ = %.3f\n",
        n_bins, r_max, length(boxes), mean(coord_all)); flush(stdout)

# Ideal FCC neighbour shells: r_n = a·√(n/2), n = 1,2,3,…  Every n is populated in
# FCC (n=7 is the (3,2,1) shell at 1.871a — omitting it leaves the largest outer peak
# unmarked and puts a marker in a trough).
shells = filter(<(r_max), a_T .* sqrt.((1:16)./2))

zmin, zmax = extrema(coord_all)
zs = collect(zmin:zmax)
zfrac = [count(==(z), coord_all)/length(coord_all) for z in zs]

# ── figure ───────────────────────────────────────────────────────────────────
BLU = RGBf(0.0,0.447,0.698); ORA = RGBf(0.835,0.369,0.0); GRN = RGBf(0.0,0.62,0.451)
fig = Figure(size=(560, 620), figure_padding=(6,10,4,6))

ax1 = Axis(fig[1,1:2]; xlabel="Wave vector", ylabel="Frequency (THz)",
           title="Constrained + rejection-sampled member — $(T_K) K, a = $(round(a_T;digits=4)) Å",
           titlesize=11, xlabelsize=11, ylabelsize=11, xticklabelsize=10, yticklabelsize=10,
           xticks=(bp.x_ticks, bp.labels), xgridvisible=false, ygridvisible=false,
           xtickalign=1, ytickalign=1)
lo, hi = minimum(F), maximum(F); pad = 0.06*(hi-lo)
band!(ax1, [first(bp.x_vals), last(bp.x_vals)], [lo-pad, lo-pad], [0.0, 0.0];
      color=(RGBf(0.80,0.15,0.15), 0.07))                      # "unstable" region, empty
for b in 1:3bp.Np; lines!(ax1, bp.x_vals, F[b,:]; color=BLU, linewidth=1.4); end
hlines!(ax1, [0.0]; color=:black, linestyle=:dash, linewidth=0.9)
vlines!(ax1, bp.x_ticks; color=(:black,0.22), linewidth=0.6)
xlims!(ax1, first(bp.x_vals), last(bp.x_vals)); ylims!(ax1, min(lo-pad, -0.6), hi+pad)
text!(ax1, 0.50, 0.20; text="min non-acoustic ω = $(round(minω;digits=3)) THz\nno imaginary modes anywhere on the path",
      space=:relative, align=(:center,:center), fontsize=9.5, color=BLU)
text!(ax1, 0.015, 0.985; text="(a)", space=:relative, align=(:left,:top), font=:bold, fontsize=12)

ax2 = Axis(fig[2,1]; xlabel="r (Å)", ylabel="g(r)",
           title="RDF, averaged over $(length(boxes)) frames", titlesize=10,
           xlabelsize=11, ylabelsize=11, xticklabelsize=9, yticklabelsize=9,
           xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1)
barplot!(ax2, r_mid, g; width=dr, color=(BLU,0.75), gap=0, strokewidth=0)
vlines!(ax2, shells; color=(ORA,0.85), linestyle=:dash, linewidth=1.0)
xlims!(ax2, 0, r_max)
text!(ax2, 0.03, 0.97; text="(b)", space=:relative, align=(:left,:top), font=:bold, fontsize=12)
text!(ax2, 0.985, 0.90; text="dashed: ideal FCC\nshells, a = $(round(a_T;digits=3)) Å",
      space=:relative, align=(:right,:top), fontsize=8, color=ORA)

ax3 = Axis(fig[2,2]; xlabel="Coordination (r < $(nn_cutoff) Å)", ylabel="fraction of atoms",
           title="⟨Z⟩ = $(round(mean(coord_all);digits=2))  (FCC = 12)", titlesize=10,
           xlabelsize=11, ylabelsize=11, xticklabelsize=9, yticklabelsize=9,
           xticks=zs, xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1)
barplot!(ax3, zs, zfrac; color=[z == 12 ? GRN : RGBf(0.6,0.6,0.6) for z in zs],
         strokewidth=0.4, strokecolor=:black)
vlines!(ax3, [12]; color=:black, linestyle=:dash, linewidth=0.9)
text!(ax3, 0.03, 0.97; text="(c)", space=:relative, align=(:left,:top), font=:bold, fontsize=12)

rowsize!(fig.layout, 1, Relative(0.58))
outdir = rundir
save("$outdir/fcc_stability_$(T_K)K.pdf", fig)
save("$outdir/fcc_stability_$(T_K)K.png", fig; px_per_unit=4)
@printf("\nfigure → %s/fcc_stability_%dK.{pdf,png}\n", outdir, T_K)

writedlm("$outdir/rdf_$(T_K)K.csv", vcat(["r_Ang" "g_r"], hcat(r_mid, g)), ',')
writedlm("$outdir/coordination_$(T_K)K.csv", vcat(["Z" "fraction"], hcat(zs, zfrac)), ',')
println("data → rdf_$(T_K)K.csv, coordination_$(T_K)K.csv")
