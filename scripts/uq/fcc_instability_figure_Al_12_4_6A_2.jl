# fcc_instability_figure_Al_12_4_6A_2.jl
#
# The counterpart to fcc_stability_figure_Al_12_4_6A_2.jl: the same three-panel
# analysis applied to a member that FAILS, so the two figures can sit side by side.
#
#   (a) BIG   phonon band structure at the NPT lattice constant a(T)
#   (b) small RDF histogram with the ideal FCC shell positions for a(T) marked
#   (c) small per-atom coordination distribution (FCC = a spike at 12)
#
# Defaults to the naive-worst member (which left FCC at every temperature), but the
# script is generic: point it at any NPT run directory and it adapts.
#
#   argv[1] = run directory, absolute or relative to models/<model>/results/
#             default: npt_thermal_expansion_naive_worst_member
#   argv[2] = temperature (default 300)
#
# It auto-detects:
#   • how many leading '#' comment lines the summary CSV has (1 for the older NPT
#     script, 3 for the multi-volume one) and whether the coordination columns exist
#   • which theta file the run used — theta_used.csv, theta_naive_worst.csv or
#     theta_npt_softest.csv, in that order of preference
#   • the sign of min omega, and colours/labels the verdict accordingly
#
# RDF and coordination are recomputed from the saved trajectory, time-averaged over
# PRODUCTION frames only (step >= n_equil), each frame using its OWN box because the
# barostat changes the cell:
#   g(r_k) = C_k / (N_frames · N_atoms · 4π r_k² Δr · ρ̄),  ρ̄ = mean_f (N/L_f³)
# r_max = min_f(L_f/2) so the range is valid in every frame.
#
# Run:  julia --project scripts/uq/fcc_instability_figure_Al_12_4_6A_2.jl [rundir] [T]

include(joinpath(@__DIR__, "..", "bandpath_phonon_uq", "lib.jl"))

element, dataset = :Al, ""
runarg    = length(ARGS) >= 1 ? ARGS[1] : "npt_thermal_expansion_naive_worst_member"
T_K       = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 300
N_cell_fc = 4
N_per_seg = [20, 20, 20, 20, 60]
n_equil   = 10_000
nn_cutoff = 3.3
n_bins    = 200

result = load_model(element, 12, 4, 6, 2; dataset_name=dataset)
model  = result.model
rundir = isdir(runarg) ? runarg : "$(result.dir)/results/$runarg"
isdir(rundir) || error("no run directory at $rundir")
traj = "$rundir/T$(T_K)K/md_trajectory.extxyz"
isfile(traj) || error("no trajectory at $traj")
@printf("run : %s\nT   : %d K\n", rundir, T_K); flush(stdout)

# ── which θ did this run use?  prefer a copy inside the run dir ──────────────
θpaths = ["$rundir/theta_used.csv", "$rundir/theta_naive_worst.csv",
          "$rundir/theta_npt_softest.csv",
          "$(result.dir)/results/bandpath_undotted_multivolume/theta_npt_softest.csv"]
θfile = findfirst(isfile, θpaths)
θfile === nothing && error("no theta file found; looked in:\n  " * join(θpaths, "\n  "))
θ = vec(readdlm(θpaths[θfile], ','))
length(θ) == length(result.lin_params) ||
    error("theta has $(length(θ)) entries, model has $(length(result.lin_params))")
@printf("θ   : %s\n", θpaths[θfile]); flush(stdout)

# ── summary CSV: comment lines and column count both vary between NPT scripts ─
sumfile = "$rundir/thermal_expansion_summary.csv"
lines   = readlines(sumfile)
n_cmt   = count(l -> startswith(strip(l), "#"), lines)
hdr     = split(strip(lines[n_cmt+1]), ',')
summ    = readdlm(sumfile, ','; skipstart=n_cmt+1)
row     = findfirst(==(Float64(T_K)), Float64.(summ[:, 1]))
row === nothing && error("no T = $T_K row in $sumfile")
colof(name) = findfirst(==(name), hdr)
a_T   = summ[row, colof("a_Ang")]
minω_rep = summ[row, colof("min_omega_THz")]
coord_rep = colof("mean_coord") === nothing ? NaN : summ[row, colof("mean_coord")]
@printf("summary: %d comment lines, %d columns | a(T) = %.5f Å, reported min ω = %+.3f THz%s\n",
        n_cmt, length(hdr), a_T, minω_rep,
        isnan(coord_rep) ? "  (no coordination column)" : @sprintf(", ⟨Z⟩ = %.2f", coord_rep))
flush(stdout)

# ── (a) phonons at a(T): FRESH NATIVE HESSIAN with these parameters ──────────
# Force constants built directly from the model with theta loaded, at the lattice
# constant this run actually settled at — no reuse of the cached per-basis (undotted)
# Hessian.  The two are mathematically identical (the undotted basis is theta-independent
# and exact at the geometry it was built at), but computing fresh removes the cache as a
# variable; they are cross-checked below.
qGtol = 5e-2
function native_bands(theta, a)
    ACEpotentials.Models.set_linear_parameters!(model, theta)
    sp, ss = bulk_prim_super(element; a=a, N_cell=N_cell_fc)
    fc = precompute_force_constants(sp, ss, model)
    structure = AtomsBuilder.Chemistry.symmetry(element)
    ql, x_vals, x_ticks, labels, _ = _band_path(structure, fc.L; N_per_seg=N_per_seg)
    qn = norm.(ql)
    Fm = Matrix{Float64}(undef, 3*fc.Np, length(ql)); m = Inf
    for (iq, q) in enumerate(ql)
        ev = eigvals(Hermitian(dynamical_matrix_from_fc(fc, q)))
        Fm[:, iq] = sign.(ev) .* sqrt.(abs.(ev)) .* FREQ_THz
        qn[iq] < qGtol && continue
        m = min(m, minimum(Fm[:, iq]))
    end
    return (; F=Fm, x_vals, x_ticks, labels, Np=fc.Np, minomega=m)
end
@printf("building a fresh native Hessian at a = %.5f A (%d^3 supercell) ...\n", a_T, N_cell_fc); flush(stdout)
t_nat = @elapsed nat = native_bands(θ, a_T)
F = nat.F; minω = nat.minomega
stable = minω >= 0
@printf("  done in %.1f min | min non-acoustic omega = %+.4f THz  ->  %s\n",
        t_nat/60, minω, stable ? "STABLE" : "UNSTABLE (imaginary modes)"); flush(stdout)

# cross-check against the cached undotted-basis contraction at the same geometry
bp = bandpath_Dk(result, model, element, a_T, N_cell_fc; N_per_seg=N_per_seg)
minω_und = min_freq_stable(θ, bp)
@printf("  cross-check (cached undotted basis): %+.4f THz | d(min omega) = %.2e | max|dF| = %.2e\n",
        minω_und, abs(minω_und - minω), maximum(abs.(bands(θ, bp) .- F))); flush(stdout)
ACEpotentials.Models.set_linear_parameters!(model, θ)

# ── trajectory, production frames only ───────────────────────────────────────
frames = ExtXYZ.read_frames(traj)
steps  = [Int(f["info"]["step"]) for f in frames]
prod   = findall(>=(n_equil), steps)
@printf("frames: %d total, %d production (step ≥ %d)\n", length(frames), length(prod), n_equil); flush(stdout)
boxes = [frames[f]["cell"][1,1] for f in prod]                # cubic box side, Å
poss  = [Matrix{Float64}(frames[f]["arrays"]["pos"]) for f in prod]
n_at  = size(poss[1], 2)

r_max = minimum(boxes)/2
dr    = r_max/n_bins
r_mid = collect(range(dr/2, r_max-dr/2; length=n_bins))

# in a function: ~13M pair evaluations, which would crawl in global scope
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
ρbar = ρacc/length(boxes)
g = [pair_counts[k]/(length(boxes)*n_at*4π*r_mid[k]^2*dr*ρbar) for k in 1:n_bins]
Zmean = mean(coord_all)
@printf("pair analysis %.1f s | RDF %d bins to %.2f Å over %d frames | ⟨Z⟩ = %.3f\n",
        t_rdf, n_bins, r_max, length(boxes), Zmean); flush(stdout)

# Ideal FCC shells: r_n = a·√(n/2), n = 1,2,3,… (every n is populated in FCC)
shells = filter(<(r_max), a_T .* sqrt.((1:16)./2))
zs = collect(minimum(coord_all):maximum(coord_all))
zfrac = [count(==(z), coord_all)/length(coord_all) for z in zs]
fcc_like = Zmean >= 11.0                       # same 12 − 1 criterion as the NPT script

# ── figure ───────────────────────────────────────────────────────────────────
BLU = RGBf(0.0,0.447,0.698); ORA = RGBf(0.835,0.369,0.0)
GRN = RGBf(0.0,0.62,0.451);  CRIM = RGBf(0.80,0.15,0.15)
ACC  = stable ? BLU : CRIM                      # band colour follows the verdict
fig = Figure(size=(560, 620), figure_padding=(6,10,4,6))

verdict = fcc_like ? "still FCC" : "LEFT FCC"
ax1 = Axis(fig[1,1:2]; xlabel="Wave vector", ylabel="Frequency (THz)",
           title="$(basename(rundir)) — $(T_K) K, a = $(round(a_T;digits=4)) Å  ($verdict)",
           titlesize=11, xlabelsize=11, ylabelsize=11, xticklabelsize=10, yticklabelsize=10,
           xticks=(nat.x_ticks, nat.labels), xgridvisible=false, ygridvisible=false,
           xtickalign=1, ytickalign=1)
lo, hi = minimum(F), maximum(F); pad = 0.06*(hi-lo)
ylo = min(lo-pad, -0.6)
band!(ax1, [first(nat.x_vals), last(nat.x_vals)], [ylo, ylo], [0.0, 0.0]; color=(CRIM, 0.07))
for b in 1:3nat.Np; lines!(ax1, nat.x_vals, F[b,:]; color=ACC, linewidth=1.4); end
hlines!(ax1, [0.0]; color=:black, linestyle=:dash, linewidth=0.9)
vlines!(ax1, nat.x_ticks; color=(:black,0.22), linewidth=0.6)
xlims!(ax1, first(nat.x_vals), last(nat.x_vals)); ylims!(ax1, ylo, hi+pad)
msg = stable ? "min non-acoustic ω = $(round(minω;digits=3)) THz\nno imaginary modes anywhere on the path" :
               "min non-acoustic ω = $(round(minω;digits=2)) THz\nimaginary modes across much of the zone"
text!(ax1, 0.50, stable ? 0.20 : 0.93; text=msg, space=:relative,
      align=(:center,:center), fontsize=9.5, color=ACC)
text!(ax1, 0.015, 0.985; text="(a)", space=:relative, align=(:left,:top), font=:bold, fontsize=12)

ax2 = Axis(fig[2,1]; xlabel="r (Å)", ylabel="g(r)",
           title="RDF, averaged over $(length(boxes)) frames", titlesize=10,
           xlabelsize=11, ylabelsize=11, xticklabelsize=9, yticklabelsize=9,
           xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1)
barplot!(ax2, r_mid, g; width=dr, color=(ACC,0.75), gap=0, strokewidth=0)
vlines!(ax2, shells; color=(ORA,0.85), linestyle=:dash, linewidth=1.0)
xlims!(ax2, 0, r_max)
text!(ax2, 0.03, 0.97; text="(b)", space=:relative, align=(:left,:top), font=:bold, fontsize=12)
text!(ax2, 0.985, 0.90; text="dashed: ideal FCC\nshells, a = $(round(a_T;digits=3)) Å",
      space=:relative, align=(:right,:top), fontsize=8, color=ORA)

ax3 = Axis(fig[2,2]; xlabel="Coordination (r < $(nn_cutoff) Å)", ylabel="fraction of atoms",
           title="⟨Z⟩ = $(round(Zmean;digits=2))  (FCC = 12)", titlesize=10,
           xlabelsize=11, ylabelsize=11, xticklabelsize=9, yticklabelsize=9,
           xticks=zs, xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1)
barplot!(ax3, zs, zfrac; color=[z == 12 ? GRN : RGBf(0.6,0.6,0.6) for z in zs],
         strokewidth=0.4, strokecolor=:black)
vlines!(ax3, [12]; color=:black, linestyle=:dash, linewidth=0.9)
text!(ax3, 0.03, 0.97; text="(c)", space=:relative, align=(:left,:top), font=:bold, fontsize=12)

rowsize!(fig.layout, 1, Relative(0.58))
stem = "$rundir/fcc_structure_$(T_K)K"
save("$stem.pdf", fig); save("$stem.png", fig; px_per_unit=4)
@printf("\nfigure → %s.{pdf,png}\n", stem)
writedlm("$rundir/rdf_$(T_K)K.csv", vcat(["r_Ang" "g_r"], hcat(r_mid, g)), ',')
writedlm("$rundir/coordination_$(T_K)K.csv", vcat(["Z" "fraction"], hcat(zs, zfrac)), ',')
@printf("verdict: min ω = %+.3f THz, ⟨Z⟩ = %.2f → %s\n", minω, Zmean, verdict)
