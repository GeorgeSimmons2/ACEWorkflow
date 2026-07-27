# npt_ensemble_member_Al_12_4_6A_2.jl
#
# ONE trajectory of a 6-member NPT ENSEMBLE.  Run as a SLURM array (0..5) so all six
# integrate in parallel; each task writes its own directory and metadata.
#
#   index 0   : the CONSTRAINED MEAN — a_eq pinned (b′·θ=0, b″·θ>0) AND phonon-positive
#               at all six volumes a_eq·{1.00,…,1.10}.  Source:
#               results/bandpath_undotted_multivolume/theta_mean.csv
#   index 1-5 : five members drawn deterministically (subsample_seed) from
#               results/aeq_cheap_vs_expensive/committee_A_cheap_phononreject.csv
#               — the cheap a_eq-QP cloud with phonon rejection sampling.
#
# WHY THE SAME MD SEED FOR EVERY MEMBER.  Velocities and barostat moves are seeded
# identically per temperature across all six runs, so any spread in a(T), g(r) or the
# phonons is attributable to the PARAMETER VECTORS — the POPS uncertainty — and not to
# MD noise.  That is what makes the ensemble spread interpretable as a committee
# uncertainty band rather than a mix of two noise sources.
#
# Every run records what it used: the θ itself, its SHA-256, its source file and row,
# its min ω at all six constrained volumes, its own relaxed a₀, and the full MD
# settings.  See metadata.csv / theta_used.csv in each output directory, and the
# manifest written by index 0.
#
# Run:  sbatch scripts/uq/run_npt_ensemble.slurm            (array 0-5, parallel)
#       julia --project -t 8 scripts/uq/npt_ensemble_member_Al_12_4_6A_2.jl 3   (single)

include(joinpath(@__DIR__, "..", "bandpath_phonon_uq", "lib.jl"))
using Molly, Random, SHA
using AtomsBuilder: bulk
@async while true; flush(stdout); flush(stderr); sleep(5); end

# ── which ensemble member is this task? ──────────────────────────────────────
idx = if !isempty(ARGS)
    parse(Int, ARGS[1])
elseif haskey(ENV, "SLURM_ARRAY_TASK_ID")
    parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
else
    error("give an ensemble index 0..5 as ARGS[1] or via SLURM_ARRAY_TASK_ID")
end
0 <= idx <= 5 || error("ensemble index must be 0..5, got $idx")

element, dataset = :Al, ""
N_cell_fc      = 4
N_per_seg      = [20, 20, 20, 20, 60]
vol_scales     = collect(1.00:0.02:1.10)
subsample_seed = 20260727        # picks WHICH 5 committee members — recorded below
md_seed        = 1234            # SAME for every member; per-T offset applied below
n_members      = 5

supercell      = (4, 4, 4)
temperatures_K = [300.0, 500.0, 700.0, 900.0]
pressure       = 0.0u"GPa"
dt             = 1.0u"fs"
friction       = 0.01u"fs^-1"
n_equil        = 10_000
n_prod         = 20_000
log_every      = 50
fcc_coord_tol  = 1.0
nn_cutoff      = 3.3
cluster_cutoff = 2.2

result = load_model(element, 12, 4, 6, 2; dataset_name=dataset)
model  = result.model; lin_params = result.lin_params; n_params = length(lin_params)
RES    = "$(result.dir)/results"
MEAN_CSV = "$RES/bandpath_undotted_multivolume/theta_mean.csv"
COMM_CSV = "$RES/aeq_cheap_vs_expensive/committee_A_cheap_phononreject.csv"
ens_root = "$RES/npt_ensemble"; mkpath(ens_root)

# ── deterministic subsample of 5 committee members ───────────────────────────
comm = readdlm(COMM_CSV, ',')
size(comm, 2) == n_params || error("committee has $(size(comm,2)) cols, model has $n_params")
picked = sort(randperm(MersenneTwister(subsample_seed), size(comm,1))[1:n_members])

if idx == 0
    role   = "constrained_mean"
    θ      = vec(readdlm(MEAN_CSV, ','))
    src    = MEAN_CSV; srcrow = 0
else
    row    = picked[idx]
    role   = "member_$(row)"
    θ      = collect(Float64, comm[row, :])
    src    = COMM_CSV; srcrow = row
end
length(θ) == n_params || error("θ has $(length(θ)) entries, expected $n_params")
outdir = "$ens_root/$role"; mkpath(outdir)
θsha = bytes2hex(sha256(join(string.(θ), ",")))
@printf("ensemble index %d → role %s\n  source %s row %d\n  sha256(θ) %s\n  out → %s\n",
        idx, role, src, srcrow, θsha[1:16], outdir); flush(stdout)
@printf("  subsample_seed %d → committee rows %s\n", subsample_seed, string(picked)); flush(stdout)

# ── stability of this vector across the constrained volumes ─────────────────
ACEpotentials.Models.set_linear_parameters!(model, lin_params)
a_mean = ACEWorkflow.relax_lattice_constant(model, element)
a_list = a_mean .* vol_scales
bps    = [bandpath_Dk(result, model, element, a, N_cell_fc; N_per_seg=N_per_seg) for a in a_list]
minω_vols = [min_freq_stable(θ, bp) for bp in bps]
@printf("\n  min ω at the six constrained volumes: %s THz\n", string(round.(minω_vols; digits=4)))
@printf("  worst over range: %+.4f THz  %s\n", minimum(minω_vols),
        minimum(minω_vols) > 0 ? "(phonon-stable everywhere)" : "*** NEGATIVE ***"); flush(stdout)

# ── own 0 K equilibrium and phonons there ───────────────────────────────────
ACEpotentials.Models.set_linear_parameters!(model, θ)
a0 = try
    a = ACEWorkflow.relax_lattice_constant(model, element)
    (0.9a_mean < a < 1.1a_mean) ? a : (@warn "implausible a=$a; using a_mean"; a_mean)
catch e; @warn "relax failed ($e); using a_mean"; a_mean end
bp_a0    = bandpath_Dk(result, model, element, a0, N_cell_fc; N_per_seg=N_per_seg)
minω_a0  = min_freq_stable(θ, bp_a0)
bands_a0 = bands(θ, bp_a0)
@printf("  own a₀ = %.5f Å,  min ω there = %+.4f THz\n\n", a0, minω_a0); flush(stdout)

_savepub(fig, stem) = (save("$stem.pdf", fig); save("$stem.png", fig; px_per_unit=4))

# ── per-T analysis with per-frame box + FCC-survival diagnostic ─────────────
function analyze_md(sys_md, dir, T_K; log_every, equil_frames, N_super)
    mkpath(dir)
    ch = sys_md.loggers.coords.history
    vh = ustrip.(u"Å^3", sys_md.loggers.volume.history)
    th = ustrip.(sys_md.loggers.temp.history); eh = ustrip.(sys_md.loggers.energy.history)
    nf = length(ch); na = length(ch[1]); side(f) = cbrt(vh[f])
    prod = (equil_frames+1):nf
    a_prod = cbrt.(vh[prod]) ./ N_super
    a_T, a_T_std = mean(a_prod), std(a_prod)

    species = [string(Molly.atomic_symbol(sys_md, i)) for i in 1:na]
    frames = Dict{String,Any}[]
    for (f, fc) in enumerate(ch)
        L = side(f)
        push!(frames, Dict{String,Any}("N_atoms"=>na,
            "info"=>Dict{String,Any}("Lattice"=>"$L 0.0 0.0 0.0 $L 0.0 0.0 0.0 $L",
                "Properties"=>"species:S:1:pos:R:3", "energy"=>eh[f],
                "temperature"=>th[f], "step"=>(f-1)*log_every),
            "arrays"=>Dict{String,Any}("species"=>species,
                "pos"=>reduce(hcat, [ustrip.(u"Å", c) for c in fc]))))
    end
    ExtXYZ.write_frames("$dir/md_trajectory.extxyz", frames)

    r_max = minimum(side.(prod))/2; nb = 200; dr = r_max/nb
    rmid = collect(range(dr/2, r_max-dr/2; length=nb)); cnt = zeros(nb); ρacc = 0.0
    mean_coord = Float64[]; med_nn = Float64[]; min_pair = Float64[]
    for f in prod
        L = side(f); p = [ustrip.(u"Å", c) for c in ch[f]]; ρacc += na/L^3
        z = zeros(Int, na); nn = fill(Inf, na); mind = Inf
        for i in 1:na-1, j in i+1:na
            d = p[i] .- p[j]; d = d .- L .* round.(d ./ L); r = norm(d)
            r < mind && (mind = r)
            r < nn[i] && (nn[i] = r); r < nn[j] && (nn[j] = r)
            r < nn_cutoff && (z[i]+=1; z[j]+=1)
            r < r_max || continue
            b = floor(Int, r/dr)+1; b <= nb && (cnt[b] += 2)
        end
        push!(mean_coord, mean(z)); push!(med_nn, median(nn)); push!(min_pair, mind)
    end
    ρbar = ρacc/length(prod)
    rdf = [cnt[k]/(length(prod)*na*4π*rmid[k]^2*dr*ρbar) for k in 1:nb]
    Z = mean(mean_coord); nnv = median(med_nn); still_fcc = Z >= 12.0 - fcc_coord_tol

    ref = [ustrip.(u"Å", c) for c in ch[first(prod)]]; msd = Float64[]
    for f in prod
        L = side(f); p = [ustrip.(u"Å", c) for c in ch[f]]; s = 0.0
        for i in 1:na; d = p[i] .- ref[i]; d = d .- L .* round.(d ./ L); s += sum(abs2, d); end
        push!(msd, s/na)
    end
    t_prod = (0:length(prod)-1) .* (log_every * ustrip(u"fs", dt))

    writedlm("$dir/rdf.csv", vcat(["r_Ang" "g_r"], hcat(rmid, rdf)), ',')
    writedlm("$dir/msd.csv", vcat(["t_fs" "msd_A2"], hcat(collect(t_prod), msd)), ',')
    f1 = Figure(size=(560,340)); a1 = Axis(f1[1,1]; title="RDF — $role, $(round(Int,T_K)) K",
        xlabel="r (Å)", ylabel="g(r)", xgridvisible=false, ygridvisible=false)
    barplot!(a1, rmid, rdf; width=dr, color=(RGBf(0.0,0.447,0.698),0.75), gap=0, strokewidth=0)
    _savepub(f1, "$dir/md_rdf")
    f2 = Figure(size=(600,620))
    for (r,(ttl,yy,cc)) in enumerate((("Temperature",th,RGBf(0.0,0.447,0.698)),
                                      ("Potential energy",eh,RGBf(0.835,0.369,0.0)),
                                      ("Volume",vh,RGBf(0.0,0.62,0.451))))
        ax = Axis(f2[r,1]; title=ttl, xlabel="Time (fs)")
        lines!(ax, (0:nf-1).*(log_every*ustrip(u"fs",dt)), yy; color=cc)
    end
    _savepub(f2, "$dir/md_convergence")
    return (; a_T, a_T_std, mean_T=mean(th[prod]), Z, nnv, still_fcc, min_pair=minimum(min_pair))
end

# ── NPT sweep ────────────────────────────────────────────────────────────────
equil_frames = div(n_equil, log_every); N_super = supercell[1]
aT=Float64[]; aTs=Float64[]; wT=Float64[]; ZT=Float64[]; nnT=Float64[]; fccT=Bool[]
for T_K in temperatures_K
    T = T_K*u"K"
    # IDENTICAL seed across ensemble members at each temperature
    Random.seed!(md_seed + round(Int, T_K))
    ACEpotentials.Models.set_linear_parameters!(model, θ)
    sys = bulk(element, a=a0*u"Å", cubic=true) * supercell
    sys_md = Molly.System(sys; force_units=u"eV/Å", energy_units=u"eV")
    sys_md = Molly.System(sys_md; general_inters=(model,),
        velocities = Molly.random_velocities(sys_md, T),
        loggers = (temp=Molly.TemperatureLogger(log_every),
                   coords=Molly.CoordinatesLogger(log_every),
                   volume=Molly.VolumeLogger(log_every),
                   energy=Molly.PotentialEnergyLogger(typeof(1.0u"eV"), log_every)))
    sim = Molly.Langevin(dt=dt, temperature=T, friction=friction,
                         coupling=Molly.MonteCarloBarostat(pressure, T, sys_md.boundary))
    @printf("  T = %4.0f K (seed %d): %d + %d steps …\n", T_K, md_seed+round(Int,T_K), n_equil, n_prod); flush(stdout)
    el = @elapsed Molly.simulate!(sys_md, sim, n_equil + n_prod)
    ana = analyze_md(sys_md, "$outdir/T$(round(Int,T_K))K", T_K;
                     log_every=log_every, equil_frames=equil_frames, N_super=N_super)
    bpT = bandpath_Dk(result, model, element, ana.a_T, N_cell_fc; N_per_seg=N_per_seg)
    mω  = min_freq_stable(θ, bpT)
    push!(aT,ana.a_T); push!(aTs,ana.a_T_std); push!(wT,mω)
    push!(ZT,ana.Z); push!(nnT,ana.nnv); push!(fccT,ana.still_fcc)
    @printf("    a = %.5f ± %.5f Å | ⟨T⟩ = %.0f K | min ω = %+.3f | ⟨Z⟩ = %.2f → %s  [%.1f min]\n",
            ana.a_T, ana.a_T_std, ana.mean_T, mω, ana.Z,
            ana.still_fcc ? "FCC" : "LEFT FCC", el/60); flush(stdout)
end

nf = count(fccT)
α = nf >= 2 ? (hcat(ones(nf), temperatures_K[fccT]) \ aT[fccT])[2] / a0 : NaN
@printf("\n  FCC at %d/%d temperatures | α = %s\n", nf, length(temperatures_K),
        isnan(α) ? "not fitted" : @sprintf("%.2f ×10⁻⁶ K⁻¹", α*1e6)); flush(stdout)

# ── metadata: everything needed to reproduce or audit this trajectory ────────
writedlm("$outdir/theta_used.csv", θ, ',')
# precompute anything needing quotes — they cannot be escaped inside $() in a string
dt_fs      = ustrip(u"fs", dt)
fric_invfs = ustrip(u"fs^-1", friction)
s_scales   = join(round.(vol_scales; digits=2), " ")
s_minw     = join(round.(minω_vols; digits=4), " ")
s_picked   = join(picked, " ")
s_temps    = join(round.(Int, temperatures_K), " ")
s_alpha    = isnan(α) ? "NA" : string(α)
s_cell     = "$(supercell[1])x$(supercell[2])x$(supercell[3])"
open("$outdir/metadata.csv", "w") do io
    println(io, "key,value")
    println(io, "role,$role")
    println(io, "ensemble_index,$idx")
    println(io, "source_file,$src")
    println(io, "source_row,$srcrow  # 0 = not a committee row (the constrained mean)")
    println(io, "theta_sha256,$θsha")
    println(io, "n_params,$n_params")
    println(io, "subsample_seed,$subsample_seed")
    println(io, "subsample_rows,$s_picked")
    println(io, "md_seed_base,$md_seed  # actual per-T seed = base + round(T)")
    println(io, "a_mean_reference_Ang,$a_mean")
    println(io, "a0_own_relaxed_Ang,$a0")
    println(io, "minomega_at_a0_THz,$minω_a0")
    println(io, "vol_scales,$s_scales")
    println(io, "minomega_at_volumes_THz,$s_minw")
    println(io, "worst_minomega_over_volumes_THz,$(minimum(minω_vols))")
    println(io, "N_cell_fc,$N_cell_fc")
    println(io, "supercell,$s_cell")
    println(io, "pressure,$pressure")
    println(io, "dt_fs,$dt_fs")
    println(io, "friction_invfs,$fric_invfs")
    println(io, "n_equil,$n_equil")
    println(io, "n_prod,$n_prod")
    println(io, "log_every,$log_every")
    println(io, "fcc_coord_tol,$fcc_coord_tol")
    println(io, "nn_cutoff_Ang,$nn_cutoff")
    println(io, "alpha_1perK,$s_alpha")
    println(io, "fcc_points,$nf/$(length(temperatures_K))")
end
open("$outdir/thermal_expansion_summary.csv", "w") do io
    println(io, "# role=$role  source=$src  row=$srcrow  sha=$(θsha[1:16])")
    println(io, "T_K,a_Ang,a_std_Ang,delta_a_over_a0_pct,min_omega_THz,mean_coord,median_nn_Ang,still_fcc")
    @printf(io, "0,%.6f,0,0,%.4f,12.00,%.4f,true\n", a0, minω_a0, a0/sqrt(2))
    for (T,a,s,w,z,nn,ok) in zip(temperatures_K, aT, aTs, wT, ZT, nnT, fccT)
        @printf(io, "%.0f,%.6f,%.6f,%.4f,%.4f,%.4f,%.4f,%s\n", T,a,s,100*(a-a0)/a0,w,z,nn,ok)
    end
end

# index 0 also writes the ensemble manifest
if idx == 0
    open("$ens_root/MANIFEST.csv", "w") do io
        println(io, "# NPT ensemble — one trajectory set per parameter vector, identical MD seeds")
        println(io, "index,role,source_file,source_row")
        println(io, "0,constrained_mean,$MEAN_CSV,0")
        for (i,r) in enumerate(picked); println(io, "$i,member_$r,$COMM_CSV,$r"); end
        println(io, "# subsample_seed=$subsample_seed  md_seed_base=$md_seed")
        println(io, "# temperatures=$s_temps  supercell=$(supercell[1])^3")
    end
    println("wrote $ens_root/MANIFEST.csv")
end
println("\ndone: $outdir")
