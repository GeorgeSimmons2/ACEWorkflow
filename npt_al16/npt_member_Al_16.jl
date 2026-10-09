# ─────────────────────────────────────────────────────────────────────────────
# npt_member_Al_16.jl — NPT thermal expansion of ONE Al_16_4_6A_3_ ensemble member.
#
# Adapted from npt_trajectories/npt_constrained_softest.jl (Al_12), which is left untouched.
# The MD protocol and the per-temperature analysis (`analyze_md`, copied verbatim) are
# unchanged: 4×4×4 FCC, 256 atoms, 0 Pa, Langevin + MonteCarloBarostat, dt 1 fs, friction
# 0.01 fs⁻¹, 10k equilibration + 20k production, logged every 50, per-frame box.
#
# What changed for Al_16, and why:
#
#   * MODEL.  Al_16_4_6A_3_, loaded from its JSON + lin_params only — not via load_model,
#     which also reads the 1.9 GB design matrix this script never uses.
#
#   * PHONONS.  The Al_12 driver evaluates min ω through the undotted per-basis Hessian at
#     every lattice constant it visits.  For 684 parameters that basis is 3.2 GB per
#     lattice constant, and the NPT visits a new one at every temperature.  This uses a
#     NATIVE Hessian of the single member instead (same as qoi/bands_two_ensembles), which
#     is the exact operator for one θ and costs seconds.  min ω excludes q within qΓtol of
#     Γ, like min_freq_stable, so the acoustic branches going to zero do not count.
#
#   * NO MULTI-VOLUME COMMITTEE BLOCK.  That block reported min ω at the volumes the Al_12
#     multi-volume committee was constrained at.  The Al_16 ensembles are pinned at a_eq
#     only, so there is no constrained volume range to report.
#
#   * RE-SEEDED PER TEMPERATURE (MD_SEED + T).  The Al_12 driver seeds once before the
#     sweep, so its 300 K trajectory depends on which temperatures ran before it in the
#     same job.  Here each temperature's trajectory depends only on (θ, T, MD_SEED), so a
#     300 K run on its own and a 100/200/300 K sweep give the same 300 K trajectory, and
#     temperatures can be added later without invalidating earlier ones.
#
#   * PER-TEMPERATURE OUTPUT.  Each T writes T<T>K/summary_row.csv, so temperatures from
#     separate jobs combine; aggregate.jl fits α across whatever exists.
#
# Env:  MODELDIR (default models/Al_16_4_6A_3_)  THETA_FILE (required)  TAG  OUTDIR  TEMPS ("300" or "100,200,300")  MD_SEED (1234)
#       N_EQUIL (10000)  N_PROD (20000)
# Run:  sbatch npt_al16/run_npt_member.slurm   (see that file for the two members)

include(joinpath(@__DIR__, "..", "scripts", "bandpath_phonon_uq", "lib.jl"))
using Molly, Random
using AtomsBuilder: bulk
import AtomsBuilder

@async while true; flush(stdout); flush(stderr); sleep(5); end   # SLURM block-buffers stdout

element        = :Al
MODELDIR       = normpath(get(ENV, "MODELDIR", joinpath(@__DIR__, "..", "models", "Al_16_4_6A_3_")))  # any models/<name>_ dir
MODELNAME      = basename(rstrip(MODELDIR, '/'))
N_CELL         = 4
N_PER_SEG      = 20                   # the published Al_16 bands grid
qΓtol          = 5e-2
structure      = AtomsBuilder.Chemistry.symmetry(element)

THETA_FILE     = get(ENV, "THETA_FILE", "")
isempty(THETA_FILE) && error("set THETA_FILE to the member's parameter vector")
TAG            = get(ENV, "TAG", splitext(basename(THETA_FILE))[1])
outdir         = get(ENV, "OUTDIR", joinpath(@__DIR__, "runs", TAG)); mkpath(outdir)
temperatures_K = parse.(Float64, split(get(ENV, "TEMPS", "300"), ','))
MD_SEED        = parse(Int, get(ENV, "MD_SEED", "1234"))
# SEED_MODE=same : every temperature (and, if the caller passes the same MD_SEED, every
#                  member) starts from the identical random stream — common random numbers,
#                  so member-to-member differences reflect θ rather than trajectory noise.
# SEED_MODE=per_T: seed MD_SEED+T (the original single-member runs).
SEED_MODE      = get(ENV, "SEED_MODE", "per_T")
SEED_MODE in ("same", "per_T") || error("SEED_MODE must be same or per_T")

supercell      = (4, 4, 4)
pressure       = 0.0u"GPa"
dt             = 1.0u"fs"
friction       = 0.01u"fs^-1"
n_equil        = parse(Int, get(ENV, "N_EQUIL", "10000"))
n_prod         = parse(Int, get(ENV, "N_PROD",  "20000"))
log_every      = 50
fcc_coord_tol  = 1.0
nn_cutoff      = 3.3
cluster_cutoff = 2.2

model, _ = ACEpotentials.load_model("$MODELDIR/$(rstrip(MODELNAME, '_')).json")
θ = vec(readdlm(THETA_FILE, ','))
n_params = length(vec(readdlm("$MODELDIR/lin_params.csv", ',')))
length(θ) == n_params || error("θ has $(length(θ)) entries, model has $n_params")
writedlm("$outdir/theta_used.csv", θ, ',')
@printf("%s (%d params), member %s, %d threads\n  θ ← %s\n  out → %s\n  T = %s K, seed %d (%s)\n",
        MODELNAME, n_params, TAG, Threads.nthreads(), THETA_FILE, outdir, join(Int.(temperatures_K), ", "), MD_SEED, SEED_MODE)

"min non-acoustic ω of θ at lattice constant a, native 4×4×4 Hessian, Γ excluded"
function native_minω(m, θ, a)
    ACEpotentials.Models.set_linear_parameters!(m, θ)
    sp, ss = bulk_prim_super(element; a=a, N_cell=N_CELL)
    fc = precompute_force_constants(sp, ss, m)
    ql, _ = _band_path(structure, fc.L; N_per_seg=N_PER_SEG)
    w = Inf
    for q in ql
        norm(q) > qΓtol || continue
        ev = eigvals(Hermitian(dynamical_matrix_from_fc(fc, q)))
        w = min(w, minimum(sign.(ev) .* sqrt.(abs.(ev)) .* FREQ_THz))
    end
    return w
end

ACEpotentials.Models.set_linear_parameters!(model, θ)
a0 = ACEWorkflow.relax_lattice_constant(model, element)
minω_a0 = native_minω(model, θ, a0)
@printf("0 K: relaxed a₀ = %.6f Å, min ω = %+.4f THz\n", a0, minω_a0)

_savepub(fig, stem) = (save("$stem.pdf", fig); save("$stem.png", fig; px_per_unit=4))

# ── per-T MD analysis — VERBATIM from npt_trajectories/npt_constrained_softest.jl ──
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

    a_prod   = cbrt.(vol_hist[prod]) ./ N_super
    a_T      = mean(a_prod); a_T_std = std(a_prod)

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

    min_pair = Float64[]; max_coord = Int[]; mean_coord = Float64[]
    med_nn   = Float64[]; big_cluster = Int[]
    for f in prod
        L = side(f); pos = [ustrip.(u"Å", c) for c in coords_hist[f]]; n = length(pos)
        mind = Inf; coord = zeros(Int, n); nn = fill(Inf, n); adj = [Int[] for _ in 1:n]
        for i in 1:n, j in i+1:n
            d = pos[i] .- pos[j]; d = d .- L .* round.(d ./ L); r = norm(d)
            r < mind && (mind = r)
            r < nn[i] && (nn[i] = r); r < nn[j] && (nn[j] = r)
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
        push!(min_pair, mind); push!(max_coord, maximum(coord))
        push!(mean_coord, mean(coord)); push!(med_nn, median(nn)); push!(big_cluster, maxc)
    end
    coord_prod = mean(mean_coord); nn_prod = median(med_nn)
    still_fcc  = coord_prod >= 12.0 - fcc_coord_tol

    fr = Figure(size=(560,340)); axr = Axis(fr[1,1]; title="RDF — Al NPT $(round(Int,T_K)) K",
        xlabel="r (Å)", ylabel="g(r)", xgridvisible=false, ygridvisible=false)
    lines!(axr, r_mids, rdf; color=RGBf(0.0,0.447,0.698))
    vlines!(axr, [2.0]; color=(:red,0.6), linestyle=:dash); _savepub(fr, "$dir/md_rdf")

    fm = Figure(size=(560,340)); axm = Axis(fm[1,1]; title="MSD — Al NPT $(round(Int,T_K)) K",
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

    ffc = Figure(size=(600,440))
    b1 = Axis(ffc[1,1]; title="Mean coordination (cutoff $nn_cutoff Å) — FCC = 12",
              xlabel="Time (fs)", ylabel="⟨coord⟩")
    lines!(b1, t_prod, mean_coord; color=RGBf(0.0,0.447,0.698))
    hlines!(b1, [12.0]; color=:black, linestyle=:dash, linewidth=0.8)
    hlines!(b1, [12.0-fcc_coord_tol]; color=(:red,0.6), linestyle=:dot, linewidth=0.8)
    b2 = Axis(ffc[2,1]; title="Median nearest-neighbour distance — FCC = a/√2",
              xlabel="Time (fs)", ylabel="median NN (Å)")
    lines!(b2, t_prod, med_nn; color=RGBf(0.835,0.369,0.0))
    hlines!(b2, [mean(cbrt.(vol_hist[prod])./N_super)/sqrt(2)]; color=:black, linestyle=:dash, linewidth=0.8)
    _savepub(ffc, "$dir/md_fcc_survival")

    fcl = Figure(size=(620,640))
    a1 = Axis(fcl[1,1]; title="Min pair distance", xlabel="Time (fs)", ylabel="min r (Å)")
    lines!(a1, t_prod, min_pair; color=RGBf(0.0,0.447,0.698)); hlines!(a1, [cluster_cutoff]; color=:red, linestyle=:dash, linewidth=0.8)
    a2 = Axis(fcl[2,1]; title="Max coordination (cutoff $nn_cutoff Å)", xlabel="Time (fs)", ylabel="max coord.")
    lines!(a2, t_prod, Float64.(max_coord); color=RGBf(0.835,0.369,0.0))
    a3 = Axis(fcl[3,1]; title="Largest cluster (cutoff $cluster_cutoff Å)", xlabel="Time (fs)", ylabel="atoms")
    lines!(a3, t_prod, Float64.(big_cluster); color=RGBf(0.902,0.624,0.0)); hlines!(a3, [2.0]; color=:black, linestyle=:dash, linewidth=0.8)
    _savepub(fcl, "$dir/md_cluster_analysis")

    return (a_T=a_T, a_T_std=a_T_std, mean_T=mean(temps_hist[prod]),
            min_pair=minimum(min_pair), max_cluster=maximum(big_cluster),
            mean_coord=coord_prod, med_nn=nn_prod, still_fcc=still_fcc)
end

# ── NPT, one temperature at a time ───────────────────────────────────────────
equil_frames = div(n_equil, log_every)
N_super      = supercell[1]
open("$outdir/a0.csv", "w") do io
    println(io, "a0_Ang,min_omega_a0_THz"); @printf(io, "%.6f,%.4f\n", a0, minω_a0)
end

for T_K in temperatures_K
    Random.seed!(SEED_MODE == "same" ? MD_SEED : MD_SEED + round(Int, T_K))
    T = T_K * u"K"
    ACEpotentials.Models.set_linear_parameters!(model, θ)
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
    @printf("\n  T = %4.0f K: %d equil + %d prod steps …\n", T_K, n_equil, n_prod)
    el = @elapsed Molly.simulate!(sys_md, sim, n_equil + n_prod)

    dir = "$outdir/T$(round(Int,T_K))K"
    ana = analyze_md(sys_md, dir, T_K; log_every=log_every, equil_frames=equil_frames, N_super=N_super)
    mω  = native_minω(model, θ, ana.a_T)
    @printf("    a(%.0f K) = %.5f ± %.5f Å  (Δa/a₀ = %+.3f%%),  ⟨T⟩ = %.0f K,  min ω(a_T) = %+.3f THz  [%.1f min MD]\n",
            T_K, ana.a_T, ana.a_T_std, 100*(ana.a_T-a0)/a0, ana.mean_T, mω, el/60)
    @printf("    structure: ⟨coord⟩ = %.2f (FCC 12), median NN = %.3f Å  →  %s\n",
            ana.mean_coord, ana.med_nn,
            ana.still_fcc ? "STILL FCC ✓" : "*** LEFT FCC — a(T) is NOT thermal expansion ***")
    open("$dir/summary_row.csv", "w") do io
        println(io, "T_K,a_Ang,a_std_Ang,mean_T_K,min_omega_THz,mean_coord,median_nn_Ang,still_fcc,md_minutes")
        @printf(io, "%.0f,%.6f,%.6f,%.2f,%.4f,%.4f,%.4f,%s,%.1f\n",
                T_K, ana.a_T, ana.a_T_std, ana.mean_T, mω, ana.mean_coord, ana.med_nn, ana.still_fcc, el/60)
    end
end
println("\ndone → $outdir")
