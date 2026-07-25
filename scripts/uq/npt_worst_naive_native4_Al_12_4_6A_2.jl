# npt_worst_naive_native4_Al_12_4_6A_2.jl
#
# NPT thermal-expansion run on a genuinely-unstable NAIVE POPS hypercube sample
# (produced + validated by check_worst_naive_native4_Al_12_4_6A_2.jl, which saved its
# θ after confirming a negative band ω on a NATIVE relaxed 4×4×4 Hessian).
#
# Every phonon check here is NATIVE (precompute_force_constants) at N_cell=4 — no
# undotted cache, no contaminated 3×3×3.  Story: the member is harmonically unstable
# at its own 0 K a₀; watch whether NPT thermal expansion (0 Pa, 300–900 K) lifts the
# soft mode toward positive as the lattice grows.
#
# Prerequisite:  run check_worst_naive_native4_Al_12_4_6A_2.jl first (writes the θ CSV).
# Run:           julia --project -t <N> scripts/uq/npt_worst_naive_native4_Al_12_4_6A_2.jl

include(joinpath(@__DIR__, "..", "bandpath_phonon_uq", "lib.jl"))
using Molly, Random
using AtomsBuilder: bulk
Random.seed!(1234)

element = :Al; N_cell_ph = 4; N_per_seg = 40; qΓtol = 5e-2
supercell     = (4, 4, 4)              # 256-atom MD cell
temperatures_K = [300.0, 500.0, 700.0, 900.0]
pressure      = 0.0u"GPa"
dt            = 1.0u"fs"
friction      = 0.01u"fs^-1"
n_equil       = 10_000
n_prod        = 20_000
log_every     = 50

result = load_model(element, 12, 4, 6, 2; dataset_name="")
model  = result.model; lin_params = result.lin_params
outdir = "$(result.dir)/results/npt_worst_naive_native4"; mkpath(outdir)
structure = AtomsBuilder.Chemistry.symmetry(element)

θfile = "$outdir/theta_worst_naive_native4.csv"
isfile(θfile) || error("Missing $θfile — run check_worst_naive_native4_Al_12_4_6A_2.jl first.")
θ_naive = vec(readdlm(θfile, ','))
@printf("Loaded worst-naive θ (%d params), %d threads.  Outputs → %s\n", length(θ_naive), Threads.nthreads(), outdir)

# native phonon bands for θ at lattice constant a (own Hessian, 4×4×4, from scratch)
function native_bands_at(θ, a)
    ACEpotentials.Models.set_linear_parameters!(model, θ)
    sp, ss = bulk_prim_super(element; a=a, N_cell=N_cell_ph)
    fc = precompute_force_constants(sp, ss, model)
    ql, xv, xt, lb, _ = _band_path(structure, fc.L; N_per_seg=N_per_seg); qn = norm.(ql); Np = fc.Np
    F = Matrix{Float64}(undef, 3Np, length(ql))
    for (iq, q) in enumerate(ql)
        ev = eigvals(Hermitian(dynamical_matrix_from_fc(fc, q))); F[:, iq] = sign.(ev).*sqrt.(abs.(ev)).*FREQ_THz
    end
    (F=F, x_vals=xv, x_ticks=xt, labels=lb, Np=Np, min_stable=minimum(F[:, qn .> qΓtol]))
end

_savepub(fig, stem) = (save("$stem.pdf", fig); save("$stem.png", fig; px_per_unit=4))

ACEpotentials.Models.set_linear_parameters!(model, θ_naive)
a0 = ACEWorkflow.relax_lattice_constant(model, element)
nb0 = native_bands_at(θ_naive, a0)
@printf("worst-naive a₀ = %.5f Å;  native 4×4×4 min ω at a₀ = %+.3f THz\n", a0, nb0.min_stable)

# ── per-T MD analysis with PER-FRAME box (volume changes under NPT) ──────────
function analyze_md(sys_md, dir, T_K; log_every, equil_frames, N_super)
    mkpath(dir)
    coords_hist = sys_md.loggers.coords.history
    vol_hist    = ustrip.(u"Å^3", sys_md.loggers.volume.history)
    temps_hist  = ustrip.(sys_md.loggers.temp.history)
    ener_hist   = ustrip.(sys_md.loggers.energy.history)
    n_frames    = length(coords_hist); n_atoms = length(coords_hist[1])
    side(f)     = cbrt(vol_hist[f])
    prod        = (equil_frames+1):n_frames
    t_axis      = (0:n_frames-1) .* (log_every * ustrip(u"fs", dt))
    a_prod      = cbrt.(vol_hist[prod]) ./ N_super
    a_T = mean(a_prod); a_T_std = std(a_prod)

    species = [string(Molly.atomic_symbol(sys_md, i)) for i in 1:n_atoms]
    frames  = Dict{String,Any}[]
    for (f, fc) in enumerate(coords_hist)
        L = side(f)
        push!(frames, Dict{String,Any}(
            "N_atoms" => n_atoms,
            "info"    => Dict{String,Any}("Lattice"=>"$L 0.0 0.0 0.0 $L 0.0 0.0 0.0 $L",
                "Properties"=>"species:S:1:pos:R:3", "energy"=>ener_hist[f], "temperature"=>temps_hist[f], "step"=>(f-1)*log_every),
            "arrays"  => Dict{String,Any}("species"=>species, "pos"=>reduce(hcat, [ustrip.(u"Å", c) for c in fc]))))
    end
    ExtXYZ.write_frames("$dir/md_trajectory.extxyz", frames)

    r_max = minimum(side.(prod))/2; n_bins = 200; dr = r_max/n_bins
    r_mids = collect(range(dr/2, r_max-dr/2; length=n_bins)); rdf_counts = zeros(n_bins); ρacc = 0.0
    for f in prod
        L = side(f); pos = [ustrip.(u"Å", c) for c in coords_hist[f]]; ρacc += n_atoms/L^3
        for i in 1:n_atoms, j in i+1:n_atoms
            d = pos[i] .- pos[j]; d = d .- L .* round.(d ./ L); r = norm(d)
            r < r_max || continue; b = floor(Int, r/dr)+1; b <= n_bins && (rdf_counts[b] += 2)
        end
    end
    ρbar = ρacc/length(prod)
    rdf = [rdf_counts[k]/(length(prod)*n_atoms*4π*r_mids[k]^2*dr*ρbar) for k in 1:n_bins]

    ref = [ustrip.(u"Å", c) for c in coords_hist[first(prod)]]; msd = Float64[]
    for f in prod
        L = side(f); pos = [ustrip.(u"Å", c) for c in coords_hist[f]]; s = 0.0
        for i in 1:n_atoms; d = pos[i] .- ref[i]; d = d .- L .* round.(d ./ L); s += sum(abs2, d); end
        push!(msd, s/n_atoms)
    end
    t_prod = (0:length(prod)-1) .* (log_every * ustrip(u"fs", dt))

    nn_cutoff = 3.3; cluster_cutoff = 2.2
    min_pair = Float64[]; max_coord = Int[]; big_cluster = Int[]
    for f in prod
        L = side(f); pos = [ustrip.(u"Å", c) for c in coords_hist[f]]; n = length(pos)
        mind = Inf; coord = zeros(Int, n); adj = [Int[] for _ in 1:n]
        for i in 1:n, j in i+1:n
            d = pos[i] .- pos[j]; d = d .- L .* round.(d ./ L); r = norm(d)
            r < mind && (mind = r); r < nn_cutoff && (coord[i]+=1; coord[j]+=1)
            r < cluster_cutoff && (push!(adj[i], j); push!(adj[j], i))
        end
        visited = falses(n); maxc = 0
        for s0 in 1:n
            visited[s0] && continue; q = [s0]; visited[s0] = true; cs = 0
            while !isempty(q); v = popfirst!(q); cs += 1; for nb in adj[v]; visited[nb] && continue; visited[nb]=true; push!(q, nb); end; end
            cs > maxc && (maxc = cs)
        end
        push!(min_pair, mind); push!(max_coord, maximum(coord)); push!(big_cluster, maxc)
    end

    fr = Figure(size=(560,340)); axr = Axis(fr[1,1]; title="RDF — naive NPT $(round(Int,T_K)) K", xlabel="r (Å)", ylabel="g(r)", xgridvisible=false, ygridvisible=false)
    lines!(axr, r_mids, rdf; color=RGBf(0.0,0.447,0.698)); vlines!(axr, [2.0]; color=(:red,0.6), linestyle=:dash); _savepub(fr, "$dir/md_rdf")
    fm = Figure(size=(560,340)); axm = Axis(fm[1,1]; title="MSD — naive NPT $(round(Int,T_K)) K", xlabel="Time (fs)", ylabel="MSD (Å²)", xgridvisible=false, ygridvisible=false)
    lines!(axm, t_prod, msd; color=RGBf(0.835,0.369,0.0)); _savepub(fm, "$dir/md_msd")
    fc2 = Figure(size=(600,620))
    axT = Axis(fc2[1,1]; title="Temperature", xlabel="Time (fs)", ylabel="T (K)")
    axE = Axis(fc2[2,1]; title="Potential energy", xlabel="Time (fs)", ylabel="E (eV)")
    axV = Axis(fc2[3,1]; title="Volume", xlabel="Time (fs)", ylabel="V (Å³)")
    lines!(axT, t_axis, temps_hist; color=RGBf(0.0,0.447,0.698)); hlines!(axT, [T_K]; color=:black, linestyle=:dash, linewidth=0.8)
    lines!(axE, t_axis, ener_hist; color=RGBf(0.835,0.369,0.0)); lines!(axV, t_axis, vol_hist; color=RGBf(0.0,0.62,0.451))
    vlines!(axV, [equil_frames*log_every*ustrip(u"fs",dt)]; color=(:black,0.4), linestyle=:dash, linewidth=0.8); _savepub(fc2, "$dir/md_convergence")
    fcl = Figure(size=(620,640))
    a1 = Axis(fcl[1,1]; title="Min pair distance", xlabel="Time (fs)", ylabel="min r (Å)"); lines!(a1, t_prod, min_pair; color=RGBf(0.0,0.447,0.698)); hlines!(a1, [cluster_cutoff]; color=:red, linestyle=:dash, linewidth=0.8)
    a2 = Axis(fcl[2,1]; title="Max coordination (cutoff $nn_cutoff Å)", xlabel="Time (fs)", ylabel="max coord."); lines!(a2, t_prod, Float64.(max_coord); color=RGBf(0.835,0.369,0.0))
    a3 = Axis(fcl[3,1]; title="Largest cluster (cutoff $cluster_cutoff Å)", xlabel="Time (fs)", ylabel="atoms"); lines!(a3, t_prod, Float64.(big_cluster); color=RGBf(0.902,0.624,0.0)); hlines!(a3, [2.0]; color=:black, linestyle=:dash, linewidth=0.8)
    _savepub(fcl, "$dir/md_cluster_analysis")

    return (a_T=a_T, a_T_std=a_T_std, mean_T=mean(temps_hist[prod]), min_pair=minimum(min_pair), max_cluster=maximum(big_cluster))
end

# ── NPT sweep ────────────────────────────────────────────────────────────────
equil_frames = div(n_equil, log_every); N_super = supercell[1]
a_of_T = Float64[]; a_of_T_std = Float64[]; minω_of_T = Float64[]; bands_hi = nothing; a_hi = a0
println("\n── NPT sweep (0 Pa) on worst-naive from a₀ = $(round(a0;digits=5)) Å ─────────────")
for (ti, T_K) in enumerate(temperatures_K)
    global bands_hi, a_hi
    T = T_K * u"K"
    ACEpotentials.Models.set_linear_parameters!(model, θ_naive)
    sys    = bulk(element, a=a0*u"Å", cubic=true) * supercell
    sys_md = Molly.System(sys; force_units=u"eV/Å", energy_units=u"eV")
    sys_md = Molly.System(sys_md;
        general_inters = (model,),
        velocities = Molly.random_velocities(sys_md, T),
        loggers = (temp=Molly.TemperatureLogger(log_every), coords=Molly.CoordinatesLogger(log_every),
                   volume=Molly.VolumeLogger(log_every), energy=Molly.PotentialEnergyLogger(typeof(1.0u"eV"), log_every)))
    sim = Molly.Langevin(dt=dt, temperature=T, friction=friction,
                         coupling=Molly.MonteCarloBarostat(pressure, T, sys_md.boundary))
    @printf("  T = %4.0f K: %d equil + %d prod steps …\n", T_K, n_equil, n_prod)
    el = @elapsed Molly.simulate!(sys_md, sim, n_equil + n_prod)
    dir = "$outdir/T$(round(Int,T_K))K"
    ana = analyze_md(sys_md, dir, T_K; log_every=log_every, equil_frames=equil_frames, N_super=N_super)
    nbT = native_bands_at(θ_naive, ana.a_T)       # native 4×4×4 phonons at the expanded lattice
    push!(a_of_T, ana.a_T); push!(a_of_T_std, ana.a_T_std); push!(minω_of_T, nbT.min_stable)
    if ti == length(temperatures_K); bands_hi = nbT.F; a_hi = ana.a_T; end
    @printf("    a(%.0f K) = %.5f ± %.5f Å  (Δa/a₀ = %+.2f%%),  ⟨T⟩ = %.0f K,  native min ω = %+.3f THz  [%.1f min]\n",
            T_K, ana.a_T, ana.a_T_std, 100*(ana.a_T-a0)/a0, ana.mean_T, nbT.min_stable, el/60)
end

# ── money plot: native bands, a₀ (soft) vs highest-T a (maybe healed) ────────
let ylo = min(minimum(nb0.F), minimum(bands_hi)), yhi = max(maximum(nb0.F), maximum(bands_hi)); pad = 0.05*(yhi-ylo)
    fig = Figure(size=(400,320), figure_padding=(6,10,4,6))
    ax  = Axis(fig[1,1]; xlabel="Wave vector", ylabel="Frequency (THz)",
               title="worst naive (native 4×4×4) — a₀ vs thermally-expanded a(T)",
               titlesize=10, xlabelsize=11, ylabelsize=11, xticklabelsize=10, yticklabelsize=10,
               xticks=(nb0.x_ticks, nb0.labels), xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1)
    for b in 1:3nb0.Np; lines!(ax, nb0.x_vals, nb0.F[b,:]; color=RGBAf(0.80,0.15,0.15,0.7), linewidth=1.0); end
    for b in 1:3nb0.Np; lines!(ax, nb0.x_vals, bands_hi[b,:]; color=RGBf(0.0,0.447,0.698), linewidth=1.2); end
    hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=0.8); vlines!(ax, nb0.x_ticks; color=(:black,0.22), linewidth=0.6)
    xlims!(ax, first(nb0.x_vals), last(nb0.x_vals)); ylims!(ax, ylo-pad, yhi+pad)
    elem = [LineElement(color=RGBAf(0.80,0.15,0.15,0.7)), LineElement(color=RGBf(0.0,0.447,0.698))]
    Legend(fig[1,1], elem, ["a₀ = $(round(a0;digits=4)) Å  (0 K, min ω $(round(nb0.min_stable;digits=2)))",
                            "a = $(round(a_hi;digits=4)) Å  ($(round(Int,temperatures_K[end])) K, min ω $(round(minω_of_T[end];digits=2)))"];
           tellwidth=false, tellheight=false, halign=:left, valign=:bottom, margin=(8,8,8,8), framevisible=true, labelsize=8, patchsize=(16,10))
    _savepub(fig, "$outdir/bands_worst_naive_a0_vs_aT_native4")
end

# ── min ω vs lattice constant (native) ──────────────────────────────────────
let a_pts = vcat(a0, a_of_T), ω_pts = vcat(nb0.min_stable, minω_of_T), Tlab = vcat(0.0, temperatures_K)
    fig = Figure(size=(430,330), figure_padding=(6,10,4,6))
    ax  = Axis(fig[1,1]; xlabel="Lattice constant a (Å)", ylabel="min non-acoustic ω (THz)",
               title="worst naive — soft mode vs lattice (native 4×4×4)",
               titlesize=10, xlabelsize=11, ylabelsize=11, xticklabelsize=10, yticklabelsize=10, xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1)
    hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=0.8)
    lines!(ax, a_pts, ω_pts; color=(RGBf(0.0,0.447,0.698),0.5), linewidth=1.0)
    scatter!(ax, a_pts, ω_pts; color=RGBf(0.0,0.447,0.698), markersize=9)
    for (a, ω, T) in zip(a_pts, ω_pts, Tlab); text!(ax, a, ω; text=(T==0 ? "0 K" : "$(round(Int,T)) K"), fontsize=9, align=(:left,:bottom), offset=(4,2)); end
    _savepub(fig, "$outdir/minomega_vs_lattice_native4")
end

# ── thermal expansion a(T) + α ──────────────────────────────────────────────
let T = temperatures_K, a = a_of_T
    X = hcat(ones(length(T)), T); coef = X \ a; α = coef[2]/a0
    fig = Figure(size=(430,330), figure_padding=(6,10,4,6))
    ax  = Axis(fig[1,1]; xlabel="Temperature (K)", ylabel="Lattice constant a (Å)",
               title="Thermal expansion (NPT, 0 Pa) — α = $(round(α*1e6;digits=2)) ×10⁻⁶ K⁻¹",
               titlesize=10, xlabelsize=11, ylabelsize=11, xticklabelsize=10, yticklabelsize=10, xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1)
    Tfine = range(0, maximum(T); length=50); lines!(ax, Tfine, coef[1] .+ coef[2].*Tfine; color=(:black,0.5), linestyle=:dash, linewidth=1.0)
    errorbars!(ax, T, a, a_of_T_std; whiskerwidth=6, color=RGBf(0.0,0.62,0.451)); scatter!(ax, T, a; color=RGBf(0.0,0.62,0.451), markersize=9)
    scatter!(ax, [0.0], [a0]; color=:black, marker=:diamond, markersize=10); text!(ax, 0.0, a0; text="a₀ (0 K)", fontsize=9, align=(:left,:top), offset=(4,-2))
    _savepub(fig, "$outdir/thermal_expansion_aT_native4")
    @printf("\nLinear expansion coefficient α = %.3g K⁻¹ (%.2f ×10⁻⁶ K⁻¹)\n", α, α*1e6)
end

open("$outdir/thermal_expansion_summary_native4.csv", "w") do io
    println(io, "# native 4x4x4 phonons; worst-naive hypercube sample")
    println(io, "T_K,a_Ang,a_std_Ang,delta_a_over_a0_pct,min_omega_THz")
    @printf(io, "0,%.6f,0,0,%.4f\n", a0, nb0.min_stable)
    for (T, a, s, ω) in zip(temperatures_K, a_of_T, a_of_T_std, minω_of_T); @printf(io, "%.0f,%.6f,%.6f,%.4f,%.4f\n", T, a, s, 100*(a-a0)/a0, ω); end
end

ACEpotentials.Models.set_linear_parameters!(model, lin_params)
println("\n══ RESULT ══════════════════════════════════════════════════")
@printf("  worst naive (native 4×4×4): min ω %+.3f THz at a₀=%.4f Å  →  %+.3f THz at a(%.0f K)=%.4f Å\n",
        nb0.min_stable, a0, minω_of_T[end], temperatures_K[end], a_of_T[end])
@printf("  thermal healing: %s\n", (nb0.min_stable < 0 && minω_of_T[end] > 0) ? "YES ✓" : "not fully — see minomega_vs_lattice")
println("  All outputs → $outdir/")
