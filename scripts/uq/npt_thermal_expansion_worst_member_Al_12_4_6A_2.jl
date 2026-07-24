# npt_thermal_expansion_worst_member_Al_12_4_6A_2.jl
#
# "Final boss" of the unstable phonons in Al_12_4_6A_2_ (full dataset):  take the
# single NAIVE POPS committee member with the most-negative band-path phonon and
# show that pure thermal expansion (NPT at 0 Pa) lifts its soft mode from
# NEGATIVE back to POSITIVE — the harmonic instability is healed anharmonically.
#
# Pipeline
#   1. Reproduce the naive committee EXACTLY as bandpath_committee_undotted_Al_12_4_6A_2.jl
#      (same load_model, seed 1234, leverage/residual/random selection) and pick
#      θ_worst = argmin min_freq_stable — this ties to that script's bands_naive.png.
#   2. NPT at 0 Pa (Langevin + MonteCarloBarostat) at 300/500/700/900 K on a 4×4×4
#      FCC supercell (256 atoms), started from θ_worst's OWN 0 K relaxed a₀.  Average
#      the volume over the production window → a(T), thermal expansion Δa/a₀, and α.
#   3. The money plot: rebuild the undotted band path at a₀ and at each a(T) and
#      recompute θ_worst's phonons.  min ω goes from < 0 at a₀ to > 0 at a(T).
#   4. Full "W-machinery" MD diagnostics per T (RDF, MSD, T/E/V convergence, cluster
#      / overlap check) + extXYZ trajectory, all publication quality, saved under
#      models/Al_12_4_6A_2_/results/npt_thermal_expansion_worst_member/.
#
# All phonon/undotted machinery + Γ-correct band plotting lives in
# ../bandpath_phonon_uq/lib.jl.  NPT MD mirrors ../repulsive_core/md_example.jl but
# fixes the per-frame box (the box VOLUME changes under NPT, so RDF/MSD must use each
# frame's own box, not a single fixed one).
#
# Run (cluster):  sbatch scripts/uq/run_npt_thermal_expansion.slurm
# Run (local):    julia --project -t <N> scripts/uq/npt_thermal_expansion_worst_member_Al_12_4_6A_2.jl

include(joinpath(@__DIR__, "..", "bandpath_phonon_uq", "lib.jl"))
using Molly, Random
using AtomsBuilder: bulk
Random.seed!(1234)

# ── config: model / forest (identical to the committee script) ───────────────
element          = :Al
dataset          = ""          # "" → full-dataset model (Al_12_4_6A_2_)
N_cell_fc        = 3           # 3×3×3 for the undotted band-path Hessian
N_per_seg        = 20
n_lev, n_res, n_rand = 5, 10, 15
scan_full_forest = false       # true → also argmin over the WHOLE forest (slow) as a sanity check

# ── config: NPT thermal-expansion sweep ──────────────────────────────────────
supercell     = (4, 4, 4)              # 256-atom FCC supercell
temperatures_K = [300.0, 500.0, 700.0, 900.0]
pressure      = 0.0u"GPa"              # 0 Pa → PURE thermal expansion
dt            = 1.0u"fs"
friction      = 0.01u"fs^-1"
n_equil       = 10_000                 # equilibration steps (discarded from averages)
n_prod        = 20_000                 # production steps (volume averaged here)
log_every     = 50

# ── load model, assemble the POPS forest exactly as the committee script ─────
result = load_model(element, 12, 4, 6, 2; dataset_name=dataset)
model  = result.model; lin_params = result.lin_params; n_params = length(lin_params)
P = result.P; Ap = Diagonal(result.W)*result.A/P; Yw = result.W.*result.Y; λ = 1.0/size(Ap,1)
outdir = "$(result.dir)/results/npt_thermal_expansion_worst_member"; mkpath(outdir)
@printf("Model %s: %d params, %d threads.  Outputs → %s\n", result.name, n_params, Threads.nthreads(), outdir)

# reference (mean-model) geometry — used ONLY to rank naive members by the same
# metric the committee script uses (so θ_worst matches its bands_naive.png)
a_mean = ACEWorkflow.relax_lattice_constant(model, element)
@printf("Mean-model a = %.5f Å  (naive-ranking geometry)\n", a_mean)
bp_mean = bandpath_Dk(result, model, element, a_mean, N_cell_fc; N_per_seg=N_per_seg)

# leverage/residual forest (Stage-1 machinery of the committee script)
C = Symmetric(Ap'*Ap .+ λ.*(P'*P)); Cf = cholesky(C)
AtX = Cf\Matrix(Ap'); θ̃ = Cf\(Ap'*Yw)
leverage = vec(sum(Ap'.*AtX; dims=1)); residual = Yw .- Ap*θ̃
forest_member(i) = lin_params .+ (P \ (AtX[:, i] .* (residual[i]/leverage[i])))

lev_idx = sortperm(leverage; rev=true)[1:n_lev]
res_idx = Int[]; for i in sortperm(abs.(residual); rev=true); i in lev_idx && continue; push!(res_idx,i); length(res_idx)==n_res && break; end
taken = Set(vcat(lev_idx,res_idx)); rand_idx = Int[]
while length(rand_idx) < n_rand; i = rand(1:length(Yw)); (i in taken) && continue; push!(rand_idx,i); push!(taken,i); end
selected = vcat(lev_idx, res_idx, rand_idx)
naive    = [forest_member(i) for i in selected]
minf_naive = [min_freq_stable(θ, bp_mean) for θ in naive]

kworst  = argmin(minf_naive); i_worst = selected[kworst]; θ_worst = naive[kworst]
@printf("\n── Worst naive member ───────────────────────────────────────\n")
@printf("  committee slot %d (obs %d): min ω = %.3f THz at a_mean\n", kworst, i_worst, minf_naive[kworst])
@printf("  naive committee min ω ∈ [%.3f, %.3f] THz\n", minimum(minf_naive), maximum(minf_naive))

if scan_full_forest
    println("  scanning the full forest for a more-negative member …")
    mf = fill(Inf, length(Yw))
    Threads.@threads for i in 1:length(Yw)
        mf[i] = min_freq_stable(forest_member(i), bp_mean)
    end
    gi = argmin(mf)
    @printf("  full-forest worst: obs %d, min ω = %.3f THz%s\n", gi, mf[gi],
            gi == i_worst ? "  (same as committee pick)" : "  ← MORE negative than the committee pick!")
end

writedlm("$outdir/worst_member_theta.csv", θ_worst, ',')

# ── θ_worst's own 0 K equilibrium + its (soft) phonons there ─────────────────
ACEpotentials.Models.set_linear_parameters!(model, θ_worst)
a0 = try
    a = ACEWorkflow.relax_lattice_constant(model, element)
    (0.9a_mean < a < 1.1a_mean) ? a : (@warn "θ_worst relaxed a=$a Å is implausible; using a_mean"; a_mean)
catch e
    @warn "relax_lattice_constant failed for θ_worst ($e); using a_mean"; a_mean
end
@printf("\nθ_worst 0 K relaxed a₀ = %.5f Å  (thermal-expansion baseline)\n", a0)

bp_a0    = bandpath_Dk(result, model, element, a0, N_cell_fc; N_per_seg=N_per_seg)
minω_a0  = min_freq_stable(θ_worst, bp_a0)
bands_a0 = bands(θ_worst, bp_a0)
@printf("θ_worst 0 K phonons at a₀: min ω = %.3f THz  (the 'most negative' start)\n", minω_a0)

# ── publication plot helper ──────────────────────────────────────────────────
_savepub(fig, stem) = (save("$stem.pdf", fig); save("$stem.png", fig; px_per_unit=4))

# ── per-T MD analysis with PER-FRAME box (volume changes under NPT) ──────────
function analyze_md(sys_md, dir, T_K; log_every, equil_frames, N_super)
    mkpath(dir)
    coords_hist = sys_md.loggers.coords.history
    vol_hist    = ustrip.(u"Å^3", sys_md.loggers.volume.history)   # force Å³ regardless of Molly's internal unit
    temps_hist  = ustrip.(sys_md.loggers.temp.history)
    ener_hist   = ustrip.(sys_md.loggers.energy.history)
    n_frames    = length(coords_hist); n_atoms = length(coords_hist[1])
    side(f)     = cbrt(vol_hist[f])                           # cubic box side (Å) at frame f
    prod        = (equil_frames+1):n_frames                   # production frames only
    t_axis      = (0:n_frames-1) .* (log_every * ustrip(u"fs", dt))   # fs

    # lattice constant from the production volume average
    a_prod   = cbrt.(vol_hist[prod]) ./ N_super
    a_T      = mean(a_prod); a_T_std = std(a_prod)

    # ── save extXYZ trajectory (per-frame lattice) ──────────────────────────
    species = [string(Molly.atomic_symbol(sys_md, i)) for i in 1:n_atoms]
    frames  = Dict{String,Any}[]
    for (f, fc) in enumerate(coords_hist)
        L = side(f)
        push!(frames, Dict{String,Any}(
            "N_atoms" => n_atoms,
            "info"    => Dict{String,Any}(
                "Lattice"     => "$L 0.0 0.0 0.0 $L 0.0 0.0 0.0 $L",
                "Properties"  => "species:S:1:pos:R:3",
                "energy"      => ener_hist[f], "temperature" => temps_hist[f],
                "step"        => (f-1)*log_every),
            "arrays"  => Dict{String,Any}(
                "species" => species,
                "pos"     => reduce(hcat, [ustrip.(u"Å", c) for c in fc]))))
    end
    ExtXYZ.write_frames("$dir/md_trajectory.extxyz", frames)

    # ── RDF (production frames, per-frame minimum image) ────────────────────
    r_max = minimum(side.(prod))/2; n_bins = 200; dr = r_max/n_bins
    r_mids = collect(range(dr/2, r_max-dr/2; length=n_bins)); rdf_counts = zeros(n_bins); ρacc = 0.0
    for f in prod
        L = side(f); pos = [ustrip.(u"Å", c) for c in coords_hist[f]]; ρacc += n_atoms/L^3
        for i in 1:n_atoms, j in i+1:n_atoms
            d = pos[i] .- pos[j]; d = d .- L .* round.(d ./ L); r = norm(d)
            r < r_max || continue; b = floor(Int, r/dr)+1; b <= n_bins && (rdf_counts[b] += 2)
        end
    end
    ρ̄ = ρacc/length(prod)
    rdf = [rdf_counts[k]/(length(prod)*n_atoms*4π*r_mids[k]^2*dr*ρ̄) for k in 1:n_bins]

    # ── MSD relative to first production frame (per-frame minimum image) ─────
    ref = [ustrip.(u"Å", c) for c in coords_hist[first(prod)]]
    msd = Float64[]
    for f in prod
        L = side(f); pos = [ustrip.(u"Å", c) for c in coords_hist[f]]; s = 0.0
        for i in 1:n_atoms
            d = pos[i] .- ref[i]; d = d .- L .* round.(d ./ L); s += sum(abs2, d)
        end
        push!(msd, s/n_atoms)
    end
    t_prod = (0:length(prod)-1) .* (log_every * ustrip(u"fs", dt))

    # ── cluster / overlap diagnostics (production frames) ───────────────────
    nn_cutoff = 3.3; cluster_cutoff = 2.2
    min_pair = Float64[]; max_coord = Int[]; big_cluster = Int[]
    for f in prod
        L = side(f); pos = [ustrip.(u"Å", c) for c in coords_hist[f]]; n = length(pos)
        mind = Inf; coord = zeros(Int, n); adj = [Int[] for _ in 1:n]
        for i in 1:n, j in i+1:n
            d = pos[i] .- pos[j]; d = d .- L .* round.(d ./ L); r = norm(d)
            r < mind && (mind = r)
            r < nn_cutoff && (coord[i]+=1; coord[j]+=1)
            r < cluster_cutoff && (push!(adj[i], j); push!(adj[j], i))
        end
        visited = falses(n); maxc = 0
        for s0 in 1:n
            visited[s0] && continue; q = [s0]; visited[s0] = true; cs = 0
            while !isempty(q)
                v = popfirst!(q); cs += 1
                for nb in adj[v]; visited[nb] && continue; visited[nb]=true; push!(q, nb); end
            end
            cs > maxc && (maxc = cs)
        end
        push!(min_pair, mind); push!(max_coord, maximum(coord)); push!(big_cluster, maxc)
    end

    # ── plots (PDF + hi-DPI PNG) ────────────────────────────────────────────
    fr = Figure(size=(560,340)); axr = Axis(fr[1,1]; title="RDF — Al θ_worst NPT $(round(Int,T_K)) K",
        xlabel="r (Å)", ylabel="g(r)", xgridvisible=false, ygridvisible=false)
    lines!(axr, r_mids, rdf; color=RGBf(0.0,0.447,0.698))
    vlines!(axr, [2.0]; color=(:red,0.6), linestyle=:dash); _savepub(fr, "$dir/md_rdf")

    fm = Figure(size=(560,340)); axm = Axis(fm[1,1]; title="MSD — Al θ_worst NPT $(round(Int,T_K)) K",
        xlabel="Time (fs)", ylabel="MSD (Å²)", xgridvisible=false, ygridvisible=false)
    lines!(axm, t_prod, msd; color=RGBf(0.835,0.369,0.0)); _savepub(fm, "$dir/md_msd")

    fc2 = Figure(size=(600,620))
    axT = Axis(fc2[1,1]; title="Temperature", xlabel="Time (fs)", ylabel="T (K)")
    axE = Axis(fc2[2,1]; title="Potential energy", xlabel="Time (fs)", ylabel="E (eV)")
    axV = Axis(fc2[3,1]; title="Volume", xlabel="Time (fs)", ylabel="V (Å³)")
    lines!(axT, t_axis, temps_hist; color=RGBf(0.0,0.447,0.698)); hlines!(axT, [T_K]; color=:black, linestyle=:dash, linewidth=0.8)
    lines!(axE, t_axis, ener_hist;  color=RGBf(0.835,0.369,0.0))
    lines!(axV, t_axis, vol_hist;   color=RGBf(0.0,0.62,0.451))
    vlines!(axV, [equil_frames*log_every*ustrip(u"fs",dt)]; color=(:black,0.4), linestyle=:dash, linewidth=0.8)
    _savepub(fc2, "$dir/md_convergence")

    fcl = Figure(size=(620,640))
    a1 = Axis(fcl[1,1]; title="Min pair distance", xlabel="Time (fs)", ylabel="min r (Å)")
    lines!(a1, t_prod, min_pair; color=RGBf(0.0,0.447,0.698)); hlines!(a1, [cluster_cutoff]; color=:red, linestyle=:dash, linewidth=0.8)
    a2 = Axis(fcl[2,1]; title="Max coordination (cutoff $nn_cutoff Å)", xlabel="Time (fs)", ylabel="max coord.")
    lines!(a2, t_prod, Float64.(max_coord); color=RGBf(0.835,0.369,0.0))
    a3 = Axis(fcl[3,1]; title="Largest cluster (cutoff $cluster_cutoff Å)", xlabel="Time (fs)", ylabel="atoms")
    lines!(a3, t_prod, Float64.(big_cluster); color=RGBf(0.902,0.624,0.0)); hlines!(a3, [2.0]; color=:black, linestyle=:dash, linewidth=0.8)
    _savepub(fcl, "$dir/md_cluster_analysis")

    return (a_T=a_T, a_T_std=a_T_std, mean_T=mean(temps_hist[prod]),
            min_pair=minimum(min_pair), max_cluster=maximum(big_cluster))
end

# ── NPT sweep ────────────────────────────────────────────────────────────────
equil_frames = div(n_equil, log_every)
N_super      = supercell[1]
a_of_T = Float64[]; a_of_T_std = Float64[]; minω_of_T = Float64[]; bands_hi = nothing; a_hi = a0
println("\n── NPT sweep (0 Pa) from a₀ = $(round(a0;digits=5)) Å ─────────────")
for (ti, T_K) in enumerate(temperatures_K)
    global bands_hi, a_hi
    T = T_K * u"K"
    ACEpotentials.Models.set_linear_parameters!(model, θ_worst)   # bandpath_Dk resets model → re-set each T
    sys    = bulk(element, a=a0*u"Å", cubic=true) * supercell
    sys_md = Molly.System(sys; force_units=u"eV/Å", energy_units=u"eV")
    sys_md = Molly.System(sys_md;
        general_inters = (model,),
        velocities = Molly.random_velocities(sys_md, T),
        loggers = (temp   = Molly.TemperatureLogger(log_every),
                   coords = Molly.CoordinatesLogger(log_every),
                   volume = Molly.VolumeLogger(log_every),
                   energy = Molly.PotentialEnergyLogger(typeof(1.0u"eV"), log_every)))
    sim = Molly.Langevin(dt=dt, temperature=T, friction=friction,
                         coupling=Molly.MonteCarloBarostat(pressure, T, sys_md.boundary))
    @printf("  T = %4.0f K: %d equil + %d prod steps …\n", T_K, n_equil, n_prod)
    el = @elapsed Molly.simulate!(sys_md, sim, n_equil + n_prod)

    dir = "$outdir/T$(round(Int,T_K))K"
    ana = analyze_md(sys_md, dir, T_K; log_every=log_every, equil_frames=equil_frames, N_super=N_super)

    # phonons at the thermally-expanded lattice
    bp_aT = bandpath_Dk(result, model, element, ana.a_T, N_cell_fc; N_per_seg=N_per_seg)
    mω    = min_freq_stable(θ_worst, bp_aT)
    push!(a_of_T, ana.a_T); push!(a_of_T_std, ana.a_T_std); push!(minω_of_T, mω)
    if ti == length(temperatures_K); bands_hi = bands(θ_worst, bp_aT); a_hi = ana.a_T; end
    @printf("    a(%.0f K) = %.5f ± %.5f Å  (Δa/a₀ = %+.2f%%),  ⟨T⟩ = %.0f K,  min ω = %+.3f THz  [%.1f min]\n",
            T_K, ana.a_T, ana.a_T_std, 100*(ana.a_T-a0)/a0, ana.mean_T, mω, el/60)
end

# ── money plot 1: soft → healed band structure (a₀ vs highest-T a) ───────────
let ylo = min(minimum(bands_a0), minimum(bands_hi)), yhi = max(maximum(bands_a0), maximum(bands_hi))
    pad = 0.05*(yhi-ylo)
    fig = Figure(size=(400,320), figure_padding=(6,10,4,6))
    ax  = Axis(fig[1,1]; xlabel="Wave vector", ylabel="Frequency (THz)",
               title="Al_12_4_6A_2_ worst member — thermal healing of the soft mode",
               titlesize=10, xlabelsize=11, ylabelsize=11, xticklabelsize=10, yticklabelsize=10,
               xticks=(bp_a0.x_ticks, bp_a0.labels), xgridvisible=false, ygridvisible=false,
               xtickalign=1, ytickalign=1)
    for b in 1:3bp_a0.Np; lines!(ax, bp_a0.x_vals, bands_a0[b,:]; color=RGBAf(0.80,0.15,0.15,0.7), linewidth=1.0); end
    for b in 1:3bp_a0.Np; lines!(ax, bp_a0.x_vals, bands_hi[b,:]; color=RGBf(0.0,0.447,0.698), linewidth=1.2); end
    hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=0.8)
    vlines!(ax, bp_a0.x_ticks; color=(:black,0.22), linewidth=0.6)
    xlims!(ax, first(bp_a0.x_vals), last(bp_a0.x_vals)); ylims!(ax, ylo-pad, yhi+pad)
    elem = [LineElement(color=RGBAf(0.80,0.15,0.15,0.7)), LineElement(color=RGBf(0.0,0.447,0.698))]
    Legend(fig[1,1], elem, ["a₀ = $(round(a0;digits=4)) Å  (0 K, min ω $(round(minω_a0;digits=2)))",
                            "a = $(round(a_hi;digits=4)) Å  ($(round(Int,temperatures_K[end])) K, min ω $(round(minω_of_T[end];digits=2)))"];
           tellwidth=false, tellheight=false, halign=:left, valign=:bottom, margin=(8,8,8,8),
           framevisible=true, labelsize=8, patchsize=(16,10))
    _savepub(fig, "$outdir/bands_healing_soft_to_positive")
end

# ── money plot 2: min ω vs lattice constant (negative → positive crossing) ──
let
    a_pts = vcat(a0, a_of_T); ω_pts = vcat(minω_a0, minω_of_T); Tlab = vcat(0.0, temperatures_K)
    fig = Figure(size=(430,330), figure_padding=(6,10,4,6))
    ax  = Axis(fig[1,1]; xlabel="Lattice constant a (Å)", ylabel="min non-acoustic ω (THz)",
               title="Soft mode vs lattice constant — thermal expansion heals it",
               titlesize=10, xlabelsize=11, ylabelsize=11, xticklabelsize=10, yticklabelsize=10,
               xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1)
    hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=0.8)
    lines!(ax, a_pts, ω_pts; color=(RGBf(0.0,0.447,0.698),0.5), linewidth=1.0)
    scatter!(ax, a_pts, ω_pts; color=RGBf(0.0,0.447,0.698), markersize=9)
    for (a, ω, T) in zip(a_pts, ω_pts, Tlab)
        text!(ax, a, ω; text=(T==0 ? "0 K" : "$(round(Int,T)) K"), fontsize=9,
              align=(:left,:bottom), offset=(4,2))
    end
    _savepub(fig, "$outdir/minomega_vs_lattice")
end

# ── thermal-expansion curve a(T) + linear expansion coefficient α ────────────
let
    T = temperatures_K; a = a_of_T
    X = hcat(ones(length(T)), T); coef = X \ a            # a ≈ β₀ + β₁·T
    dadT = coef[2]; α = dadT / a0                          # 1/K, referenced to a₀
    fig = Figure(size=(430,330), figure_padding=(6,10,4,6))
    ax  = Axis(fig[1,1]; xlabel="Temperature (K)", ylabel="Lattice constant a (Å)",
               title="Thermal expansion (NPT, 0 Pa) — α = $(round(α*1e6;digits=2)) ×10⁻⁶ K⁻¹",
               titlesize=10, xlabelsize=11, ylabelsize=11, xticklabelsize=10, yticklabelsize=10,
               xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1)
    Tfine = range(0, maximum(T); length=50)
    lines!(ax, Tfine, coef[1] .+ coef[2].*Tfine; color=(:black,0.5), linestyle=:dash, linewidth=1.0)
    errorbars!(ax, T, a, a_of_T_std; whiskerwidth=6, color=RGBf(0.0,0.62,0.451))
    scatter!(ax, T, a; color=RGBf(0.0,0.62,0.451), markersize=9)
    scatter!(ax, [0.0], [a0]; color=:black, marker=:diamond, markersize=10)
    text!(ax, 0.0, a0; text="a₀ (0 K)", fontsize=9, align=(:left,:top), offset=(4,-2))
    _savepub(fig, "$outdir/thermal_expansion_aT")
    @printf("\nLinear expansion coefficient α = %.3g K⁻¹ (%.2f ×10⁻⁶ K⁻¹)\n", α, α*1e6)
end

# ── summary table ────────────────────────────────────────────────────────────
open("$outdir/thermal_expansion_summary.csv", "w") do io
    println(io, "T_K,a_Ang,a_std_Ang,delta_a_over_a0_pct,min_omega_THz")
    @printf(io, "0,%.6f,0,0,%.4f\n", a0, minω_a0)
    for (T, a, s, ω) in zip(temperatures_K, a_of_T, a_of_T_std, minω_of_T)
        @printf(io, "%.0f,%.6f,%.6f,%.4f,%.4f\n", T, a, s, 100*(a-a0)/a0, ω)
    end
end

ACEpotentials.Models.set_linear_parameters!(model, lin_params)
println("\n══ RESULT ══════════════════════════════════════════════════")
@printf("  θ_worst (obs %d): min ω %+.3f THz at a₀=%.4f Å  →  %+.3f THz at a(%.0f K)=%.4f Å\n",
        i_worst, minω_a0, a0, minω_of_T[end], temperatures_K[end], a_of_T[end])
@printf("  soft mode healed by thermal expansion: %s\n", (minω_a0 < 0 && minω_of_T[end] > 0) ? "YES ✓" : "check outputs")
println("  All outputs → $outdir/")
