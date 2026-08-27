# ─────────────────────────────────────────────────────────────────────────────
# PINNED REPRODUCTION COPY.  Do not edit to run a different study — copy it again.
#
# This is scripts/uq/npt_multivolume_member_Al_12_4_6A_2.jl
# copied verbatim, then pinned back to the configuration that produced the published
# trajectories behind thermal_expansion_vs_experiment/.  The working copy in scripts/uq
# has since been repurposed for other studies, so running THAT file today reproduces a
# different member into a different directory.  Every difference from it is listed in
# ../npt_trajectories/README.md and marked `# [REPRO]` below.
#
# The parameter vector is verified against the one the published run saved, so if the
# upstream committee ever changes this script fails loudly instead of quietly running
# a different member.
#
# Outputs go to a FRESH directory by default — rerunning must not overwrite the
# published trajectories the figure is built from.
# ─────────────────────────────────────────────────────────────────────────────
# npt_multivolume_member_Al_12_4_6A_2.jl
#
# NPT thermal expansion (0 Pa) on a member of the MULTI-VOLUME constrained committee
# (bandpath_undotted_multivolume) — the follow-up test of whether constraining phonon
# stability over a_eq → 1.1·a_eq keeps FCC intact under the barostat.
#
# WHY THIS EXISTS.  The single-volume run measured "6.3% thermal expansion" on a member
# that had actually LEFT FCC: coordination 12 → 9.5, NN 2.86 → 2.58 Å, ΔE ≈ −0.18 eV/atom.
# Nothing in that script noticed.  So this version adds an explicit FCC-SURVIVAL
# DIAGNOSTIC — mean coordination and median nearest-neighbour distance per frame — and
# reports, per temperature, whether the lattice was still FCC when a(T) was measured.
# An a(T) taken from a transformed cell is not thermal expansion, and is flagged as such.
#
# Selection: `npt_member` picks from results/bandpath_undotted_multivolume/
#   :softest → theta_npt_softest.csv (worst-case member: smallest min ω over volumes)
#   :median  → theta_npt_median.csv  (typical member — the fairer "does it work" test)
#   ::Int    → that row of committee_rejection.csv
#
# Everything else mirrors npt_thermal_expansion_worst_member_Al_12_4_6A_2.jl (legacy
# untouched): Langevin + MonteCarloBarostat, per-frame box in RDF/MSD, same plots.
#
# Run:  sbatch scripts/uq/run_npt_multivolume.slurm
#       julia --project -t <N> scripts/uq/npt_multivolume_member_Al_12_4_6A_2.jl

include(joinpath(@__DIR__, "..", "scripts", "bandpath_phonon_uq", "lib.jl"))  # [REPRO] path from repo root
using Molly, Random
using AtomsBuilder: bulk
Random.seed!(1234)

# Julia BLOCK-BUFFERS stdout when it is not a TTY, so under SLURM this job writes
# nothing to its .log until the buffer fills or the process exits — a ~4.5 h run
# looks dead the whole time.  Background flusher plus explicit flushes at each
# milestone below.
@async while true; flush(stdout); flush(stderr); sleep(5); end

element        = :Al
dataset        = ""
N_cell_fc      = 4
N_per_seg      = [20, 20, 20, 20, 60]
vol_scales     = collect(1.00:0.02:1.10)      # must match the committee run
committee_subdir = "bandpath_undotted_multivolume"
npt_member     = :softest                       # [REPRO] published run used :softest; working copy has drifted to :cheap_rej

supercell      = (4, 4, 4)
temperatures_K = [300.0, 500.0, 700.0, 900.0]
pressure       = 0.0u"GPa"
dt             = 1.0u"fs"
friction       = 0.01u"fs^-1"
n_equil        = 10_000
n_prod         = 20_000
log_every      = 50

# FCC-survival thresholds: perfect FCC has coord 12 within 3.3 Å.  Thermal disorder
# knocks this around by <0.5; a transformation drops it to ~9.  1.0 is a safe line.
fcc_coord_tol  = 1.0
nn_cutoff      = 3.3
cluster_cutoff = 2.2

result = load_model(element, 12, 4, 6, 2; dataset_name=dataset)
model  = result.model; lin_params = result.lin_params; n_params = length(lin_params)
# [REPRO] the published run read the multi-volume committee named in `committee_subdir`
# above; the working copy now points at results/aeq_cheap_vs_expensive instead.
# COMMITTEE_DIR lets stage 2 consume the committee stage 1 just built.
committee_dir = get(ENV, "COMMITTEE_DIR", "$(result.dir)/results/$committee_subdir")
tag    = npt_member isa Integer ? "rejection$(npt_member)" : string(npt_member)
# [REPRO] fresh directory by default so a rerun cannot overwrite the published
# trajectories in results/npt_multivolume_softest/.  Set OUTDIR to compare in place.
outdir = get(ENV, "OUTDIR", "$(result.dir)/results/repro_npt_multivolume_softest"); mkpath(outdir)
@printf("Model %s: %d params, %d threads.  Outputs → %s\n", result.name, n_params, Threads.nthreads(), outdir)
flush(stdout)

# ── load the chosen member ──────────────────────────────────────────────────
# [REPRO] THETA_FILE runs a SAVED parameter vector directly, with no committee at all.
# This is the reproduction path: committee members are not reproducible run to run (see
# README.md), so the only way to rerun the published trajectory is to rerun the published
# theta.  Default is the vector the published run saved beside its own outputs.
THETA_FILE = get(ENV, "THETA_FILE",
                 "$(result.dir)/results/npt_multivolume_softest/theta_used.csv")

θ_npt = if !isempty(THETA_FILE)
    isfile(THETA_FILE) || error("THETA_FILE=$THETA_FILE does not exist")
    @printf("θ from saved vector: %s  (no committee needed)\n", THETA_FILE)
    vec(readdlm(THETA_FILE, ','))
elseif npt_member === :softest
    vec(readdlm("$committee_dir/theta_npt_softest.csv", ','))
elseif npt_member === :median
    vec(readdlm("$committee_dir/theta_npt_median.csv", ','))
elseif npt_member === :cheap_rej
    vec(readdlm("$committee_dir/committee_A_cheap_phononreject.csv", ',')[15,:])
elseif npt_member isa Integer
    collect(Float64, readdlm("$committee_dir/committee_rejection.csv", ',')[npt_member, :])
else
    error("npt_member must be :softest, :median, or an Int")
end
length(θ_npt) == n_params || error("θ has $(length(θ_npt)) entries, expected $n_params")
@printf("NPT member: %s  (from %s)\n", tag,
        isempty(THETA_FILE) ? committee_dir : "saved θ")

# [REPRO] the published run copied its own parameter vector next to its outputs.  If the
# upstream committee has been regenerated, the member this script selects today is no
# longer the member behind the figure — fail loudly rather than silently run a different
# one.  See results/npt_multivolume_softest/PROVENANCE.md: theta_used.csv is byte
# identical to row 18 of the multi-volume rejection committee.
let ref = get(ENV, "THETA_REF", "$(result.dir)/results/npt_multivolume_softest/theta_used.csv")
    if !isempty(THETA_FILE) && abspath(THETA_FILE) == abspath(ref)
        println("REPRO CHECK: trivially satisfied — θ was loaded from the reference file itself")
    elseif ref == "none"
        println("REPRO CHECK: skipped (THETA_REF=none) — running a freshly generated member")
    elseif isfile(ref)
        θ_ref = vec(readdlm(ref, ','))
        d = maximum(abs.(θ_npt .- θ_ref))
        @printf("REPRO CHECK: max |θ − θ_published| = %.3e\n", d)
        d < 1e-10 || error("""
            selected member differs from the published one by $d.
            The committee in $committee_dir has changed, so this run would produce a
            DIFFERENT trajectory from the one the figure uses.  To run the published
            member regardless, point npt_member at the saved vector directly.""")
    else
        @warn "no reference θ at $ref — cannot verify this is the published member"
    end
end
# [REPRO] always record the vector actually used, so a later rerun has a reference
writedlm("$outdir/theta_used.csv", θ_npt, ',')
flush(stdout)

# ── stability of this member across the constrained volume range ────────────
a_mean = ACEWorkflow.relax_lattice_constant(model, element)
a_list = a_mean .* vol_scales
@printf("a_mean = %.5f Å.  Checking the member at the %d constrained volumes …\n", a_mean, length(a_list))
flush(stdout)
bps_vol = [bandpath_Dk(result, model, element, a, N_cell_fc; N_per_seg=N_per_seg) for a in a_list]
minω_constrained_vols = [min_freq_stable(θ_npt, bp) for bp in bps_vol]
for (v, a) in enumerate(a_list)
    @printf("  a = %.5f Å (%.0f%%): min ω = %+.3f THz\n", a, 100*vol_scales[v], minω_constrained_vols[v])
end
@printf("  worst over constrained range: %+.3f THz\n", minimum(minω_constrained_vols))
flush(stdout)

# ── its own 0 K equilibrium + phonons there ─────────────────────────────────
ACEpotentials.Models.set_linear_parameters!(model, θ_npt)
a0 = try
    a = ACEWorkflow.relax_lattice_constant(model, element)
    (0.9a_mean < a < 1.1a_mean) ? a : (@warn "relaxed a=$a Å is implausible; using a_mean"; a_mean)
catch e
    @warn "relax_lattice_constant failed ($e); using a_mean"; a_mean
end
@printf("NPT member 0 K relaxed a₀ = %.5f Å\n", a0)
flush(stdout)
bp_a0    = bandpath_Dk(result, model, element, a0, N_cell_fc; N_per_seg=N_per_seg)
minω_a0  = min_freq_stable(θ_npt, bp_a0)
bands_a0 = bands(θ_npt, bp_a0)
@printf("NPT member 0 K phonons at a₀: min ω = %+.3f THz\n", minω_a0)
flush(stdout)

_savepub(fig, stem) = (save("$stem.pdf", fig); save("$stem.png", fig; px_per_unit=4))

# ── per-T MD analysis with PER-FRAME box + FCC-SURVIVAL diagnostic ──────────
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

    # ── structure diagnostics: min pair, MAX and MEAN coordination, MEDIAN NN ──
    # mean coordination and median NN are the FCC-survival signal; the old script
    # logged only max coordination, which stays at 12 even after the lattice has gone.
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

    # FCC-survival panel — mean coordination and median NN vs time, with FCC references
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

# ── NPT sweep ────────────────────────────────────────────────────────────────
equil_frames = div(n_equil, log_every)
N_super      = supercell[1]
a_of_T = Float64[]; a_of_T_std = Float64[]; minω_of_T = Float64[]
coord_of_T = Float64[]; nn_of_T = Float64[]; fcc_of_T = Bool[]
bands_hi = nothing; a_hi = a0
# [REPRO] Re-seed immediately before the sweep.  Everything above this line consumes a
# different number of random draws depending on how θ was obtained — the saved-θ path
# skips the forest re-derivation, which itself calls rand() — so without this the
# Langevin/barostat stream would depend on the selection route.  Re-seeding here makes
# the MD depend only on MD_SEED, so two `reproduce` runs give identical trajectories.
# It does NOT bit-match the original published run, whose stream started from a
# different point; agreement there is statistical, within the quoted fluctuation width.
Random.seed!(parse(Int, get(ENV, "MD_SEED", "1234")))
println("\n── NPT sweep (0 Pa) on $tag from a₀ = $(round(a0;digits=5)) Å ─────────────")
for (ti, T_K) in enumerate(temperatures_K)
    global bands_hi, a_hi
    T = T_K * u"K"
    ACEpotentials.Models.set_linear_parameters!(model, θ_npt)   # bandpath_Dk resets model → re-set each T
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
    flush(stdout)
    el = @elapsed Molly.simulate!(sys_md, sim, n_equil + n_prod)

    dir = "$outdir/T$(round(Int,T_K))K"
    ana = analyze_md(sys_md, dir, T_K; log_every=log_every, equil_frames=equil_frames, N_super=N_super)

    bp_aT = bandpath_Dk(result, model, element, ana.a_T, N_cell_fc; N_per_seg=N_per_seg)
    mω    = min_freq_stable(θ_npt, bp_aT)
    push!(a_of_T, ana.a_T); push!(a_of_T_std, ana.a_T_std); push!(minω_of_T, mω)
    push!(coord_of_T, ana.mean_coord); push!(nn_of_T, ana.med_nn); push!(fcc_of_T, ana.still_fcc)
    if ti == length(temperatures_K); bands_hi = bands(θ_npt, bp_aT); a_hi = ana.a_T; end
    @printf("    a(%.0f K) = %.5f ± %.5f Å  (Δa/a₀ = %+.2f%%),  ⟨T⟩ = %.0f K,  min ω = %+.3f THz  [%.1f min]\n",
            T_K, ana.a_T, ana.a_T_std, 100*(ana.a_T-a0)/a0, ana.mean_T, mω, el/60)
    flush(stdout)
    @printf("    structure: ⟨coord⟩ = %.2f (FCC 12), median NN = %.3f Å  →  %s\n",
            ana.mean_coord, ana.med_nn,
            ana.still_fcc ? "STILL FCC ✓" : "*** LEFT FCC — a(T) is NOT thermal expansion ***")
    flush(stdout)
end

n_fcc = count(fcc_of_T)
@printf("\n  FCC survived at %d / %d temperatures\n", n_fcc, length(temperatures_K))
flush(stdout)

# ── money plot B: phonons at a₀ vs highest-T a ──────────────────────────────
let ylo = min(minimum(bands_a0), minimum(bands_hi)), yhi = max(maximum(bands_a0), maximum(bands_hi))
    pad = 0.05*(yhi-ylo)
    fig = Figure(size=(400,320), figure_padding=(6,10,4,6))
    ax  = Axis(fig[1,1]; xlabel="Wave vector", ylabel="Frequency (THz)",
               title="Multi-volume-constrained member ($tag) — a₀ vs a(T)",
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
    _savepub(fig, "$outdir/bands_npt_member_a0_vs_aT")
end

# ── money plot C: min ω vs lattice constant, with the CONSTRAINED range shaded ──
let
    a_pts = vcat(a0, a_of_T); ω_pts = vcat(minω_a0, minω_of_T); Tlab = vcat(0.0, temperatures_K)
    fig = Figure(size=(430,330), figure_padding=(6,10,4,6))
    ax  = Axis(fig[1,1]; xlabel="Lattice constant a (Å)", ylabel="min non-acoustic ω (THz)",
               title="Soft mode vs lattice constant — constrained range shaded",
               titlesize=10, xlabelsize=11, ylabelsize=11, xticklabelsize=10, yticklabelsize=10,
               xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1)
    vspan!(ax, first(a_list), last(a_list); color=(RGBf(0.0,0.62,0.451), 0.12))
    hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=0.8)
    lines!(ax, a_list, minω_constrained_vols; color=(RGBf(0.0,0.62,0.451),0.9), linewidth=1.4)
    scatter!(ax, a_list, minω_constrained_vols; color=RGBf(0.0,0.62,0.451), markersize=7, marker=:rect)
    lines!(ax, a_pts, ω_pts; color=(RGBf(0.0,0.447,0.698),0.5), linewidth=1.0)
    scatter!(ax, a_pts, ω_pts; color=RGBf(0.0,0.447,0.698), markersize=9)
    for (a, ω, T) in zip(a_pts, ω_pts, Tlab)
        text!(ax, a, ω; text=(T==0 ? "0 K" : "$(round(Int,T)) K"), fontsize=9, align=(:left,:bottom), offset=(4,2))
    end
    elem = [LineElement(color=RGBf(0.0,0.62,0.451)), LineElement(color=RGBf(0.0,0.447,0.698))]
    Legend(fig[1,1], elem, ["constrained volumes (0 K)", "NPT a(T)"];
           tellwidth=false, tellheight=false, halign=:right, valign=:bottom, margin=(8,8,8,8),
           framevisible=true, labelsize=8, patchsize=(16,10))
    _savepub(fig, "$outdir/minomega_vs_lattice")
end

# ── thermal expansion a(T) + α — FCC points only ────────────────────────────
# α is fitted ONLY over temperatures where the lattice was still FCC.  Fitting across a
# transformed cell is what made the single-volume run's α look plausible for the wrong
# reason, so points that left FCC are plotted but excluded from the fit.
fit_coef = n_fcc >= 2 ?
    hcat(ones(n_fcc), temperatures_K[fcc_of_T]) \ a_of_T[fcc_of_T] : [NaN, NaN]
α = fit_coef[2] / a0
let
    fig = Figure(size=(430,330), figure_padding=(6,10,4,6))
    ttl = n_fcc >= 2 ?
        "Thermal expansion (NPT, 0 Pa) — α = $(round(α*1e6;digits=2)) ×10⁻⁶ K⁻¹ ($n_fcc FCC pts)" :
        "NPT (0 Pa) — LATTICE LEFT FCC; α not fitted"
    ax = Axis(fig[1,1]; xlabel="Temperature (K)", ylabel="Lattice constant a (Å)", title=ttl,
              titlesize=10, xlabelsize=11, ylabelsize=11, xticklabelsize=10, yticklabelsize=10,
              xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1)
    if n_fcc >= 2
        Tfine = range(0, maximum(temperatures_K); length=50)
        lines!(ax, Tfine, fit_coef[1] .+ fit_coef[2].*Tfine; color=(:black,0.5), linestyle=:dash, linewidth=1.0)
    end
    errorbars!(ax, temperatures_K, a_of_T, a_of_T_std; whiskerwidth=6, color=RGBf(0.0,0.62,0.451))
    # FCC points filled, transformed points hollow crimson — never silently mixed
    for (T, a, ok) in zip(temperatures_K, a_of_T, fcc_of_T)
        scatter!(ax, [T], [a]; color=ok ? RGBf(0.0,0.62,0.451) : RGBAf(0.80,0.15,0.15,0.9),
                 marker=ok ? :circle : :xcross, markersize=10)
    end
    scatter!(ax, [0.0], [a0]; color=:black, marker=:diamond, markersize=10)
    text!(ax, 0.0, a0; text="a₀ (0 K)", fontsize=9, align=(:left,:top), offset=(4,-2))
    elem = [MarkerElement(color=RGBf(0.0,0.62,0.451), marker=:circle),
            MarkerElement(color=RGBAf(0.80,0.15,0.15,0.9), marker=:xcross)]
    Legend(fig[1,1], elem, ["still FCC", "left FCC"]; tellwidth=false, tellheight=false,
           halign=:right, valign=:bottom, margin=(8,8,8,8), framevisible=true,
           labelsize=8, patchsize=(14,10))
    _savepub(fig, "$outdir/thermal_expansion_aT")
end
isnan(α) ? println("\nα NOT fitted — fewer than 2 temperatures stayed FCC.") :
           @printf("\nLinear expansion coefficient α = %.3g K⁻¹ (%.2f ×10⁻⁶ K⁻¹), fitted on %d FCC points\n", α, α*1e6, n_fcc)

# ── summary table ────────────────────────────────────────────────────────────
s_scales = join(round.(vol_scales; digits=2), " ")
s_minω   = join(round.(minω_constrained_vols; digits=4), " ")
s_alpha  = isnan(α) ? "NA_left_FCC" : string(round(α, sigdigits=4))
open("$outdir/thermal_expansion_summary.csv", "w") do io
    println(io, "# npt_member=$tag  committee=$committee_subdir  vol_scales=$s_scales")
    println(io, "# minomega_at_constrained_volumes_THz=$s_minω")
    println(io, "# alpha_1perK=$s_alpha  fcc_points=$n_fcc/$(length(temperatures_K))")
    println(io, "T_K,a_Ang,a_std_Ang,delta_a_over_a0_pct,min_omega_THz,mean_coord,median_nn_Ang,still_fcc")
    @printf(io, "0,%.6f,0,0,%.4f,12.00,%.4f,true\n", a0, minω_a0, a0/sqrt(2))
    for (T, a, s, ω, c, nn, ok) in zip(temperatures_K, a_of_T, a_of_T_std, minω_of_T, coord_of_T, nn_of_T, fcc_of_T)
        @printf(io, "%.0f,%.6f,%.6f,%.4f,%.4f,%.4f,%.4f,%s\n", T, a, s, 100*(a-a0)/a0, ω, c, nn, ok)
    end
end

ACEpotentials.Models.set_linear_parameters!(model, lin_params)
println("\n══ RESULT ══════════════════════════════════════════════════")
@printf("  member %s: min ω over constrained volumes %+.3f THz (worst), at a₀ %+.3f THz\n",
        tag, minimum(minω_constrained_vols), minω_a0)
@printf("  FCC survived %d/%d temperatures; α = %s\n", n_fcc, length(temperatures_K),
        isnan(α) ? "not fitted (left FCC)" : @sprintf("%.2f ×10⁻⁶ K⁻¹", α*1e6))
println("  All outputs → $outdir/")
