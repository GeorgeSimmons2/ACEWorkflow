# npt_naive_member_Al_20_4_6A_4.jl
#
# ONE NPT temperature sweep for one naive POPS hypercube sample of
# Al_20_4_6A_4 (5476 basis functions).  Run as a SLURM array 0..4 so all five
# integrate in parallel.  Members come from
#   results/pops_naive_committee/committee_naive.csv   (one member per ROW)
# drawn by scripts/uq/pops_naive_committee_Al_20_4_6A_4.jl.
#
# ARGS[1] = member index, 0-based (0..4)
#
# DIFFERENCES FROM npt_ensemble_member_Al_12_4_6A_2.jl, and why:
#
#  1. The model is loaded from its JSON, NOT via load_model. load_model reads the
#     15 GB A.csv into a 6.4 GB dense matrix, which MD never touches. The JSON
#     carries the basis and lets us set θ directly.
#
#  2. NO band-path phonons. The Al_12 driver evaluates min ω at each a(T) via the
#     undotted per-basis Hessian; for 5476 parameters at 4x4x4 that array is
#     768^2 * 5476 * 8 B = 26 GB per volume, and phonons were not asked for here.
#     Structure is diagnosed from the trajectory instead (see 3).
#
#  3. `still_fcc` in the summary uses the driver's legacy mean-coordination test
#     inside a FIXED 3.3 A cutoff. That cutoff does not scale with the lattice
#     constant and mis-scores expanded cells -- it is kept only for continuity
#     with the Al_12 runs. THE AUTHORITATIVE VERDICT comes from
#     scripts/uq/fcc_order_ensemble_Al_12_4_6A_2.py (Steinhardt q6bar + a
#     diffusion-based melting test), which reads the trajectories this script
#     writes and works unmodified on this output directory.
#
# Every member uses IDENTICAL MD seeds at each temperature, so differences
# between members are differences in θ and nothing else. Each temperature starts
# from a fresh perfect FCC lattice, so transformation is a stochastic nucleation
# event within the run, not a thermodynamic threshold -- do not read "FCC at
# N/4 temperatures" as a stability boundary.
#
# OUTPUTS -> models/Al_20_4_6A_4/results/npt_naive/member_<k>/
#   T{300,500,700,900}K/{md_trajectory.extxyz,rdf.csv,msd.csv,*.pdf,*.png}
#   thermal_expansion_summary.csv, theta_used.csv, metadata.csv
#
# Run:  julia --project -t 8 scripts/uq/npt_naive_member_Al_20_4_6A_4.jl 0

include(joinpath(@__DIR__, "..", "bandpath_phonon_uq", "lib.jl"))
using Molly, Random, SHA
using AtomsBuilder: bulk

@async while true; flush(stdout); sleep(5); end     # Julia block-buffers to files

element        = :Al
temperatures_K = [300.0, 500.0, 700.0, 900.0]
pressure       = 0.0u"GPa"
dt             = 1.0u"fs"
friction       = 0.01u"fs^-1"
log_every      = 50

# This model is ~11x more expensive per step than Al_12_4_6A_2_ (5476 vs 91 basis
# functions): measured 1789 ms/step at 4x4x4 and 843 ms/step at 3x3x3 on 8 threads.
# At the Al_12 protocol (4x4x4, 10k+20k) that is ~60 h for four temperatures,
# above the 48 h partition limit — so cell size and run length are tunable here
# rather than hard-coded. Defaults match the Al_12 study for comparability;
# override from the slurm file when the walltime demands it.
N_super        = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
n_equil        = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 10_000
n_prod         = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 20_000
supercell      = (N_super, N_super, N_super)
fcc_coord_tol  = 1.0
nn_cutoff      = 3.3          # legacy, see header note 3
md_seed        = 1234

idx = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 0
role = "member_$idx"

MODEL_DIR = abspath(joinpath(@__DIR__, "..", "..", "models", "Al_20_4_6A_4"))
COMM_CSV  = "$MODEL_DIR/results/pops_naive_committee/committee_naive.csv"
isfile(COMM_CSV) || error("committee not found: $COMM_CSV\n" *
                          "run scripts/uq/pops_naive_committee_Al_20_4_6A_4.jl first")

comm = readdlm(COMM_CSV, ',')
0 <= idx < size(comm, 1) || error("member index $idx outside 0..$(size(comm,1)-1)")
θ = vec(comm[idx+1, :])                      # rows are members; ARGS is 0-based

model, _ = ACEpotentials.load_model("$MODEL_DIR/Al_20_4_6A_4.json")
ACEpotentials.Models.set_linear_parameters!(model, θ)

outdir = "$MODEL_DIR/results/npt_naive/$role"; mkpath(outdir)
writedlm("$outdir/theta_used.csv", θ', ',')

a0 = ACEWorkflow.relax_lattice_constant(model, element)
@printf("── %s ── %d params | own a₀ = %.5f Å | %d threads\n",
        role, length(θ), a0, Threads.nthreads())
@printf("   source: %s row %d\n", COMM_CSV, idx)
@printf("   out   : %s\n", outdir)
@printf("   MD    : %d^3 = %d atoms, %d equil + %d prod steps, T = %s K\n\n",
        N_super, 4*N_super^3, n_equil, n_prod, join(round.(Int, temperatures_K), " "))
flush(stdout)

_savepub(fig, stem) = (save("$stem.pdf", fig); save("$stem.png", fig; px_per_unit=4))

# ── per-T analysis (identical to the Al_12 driver, minus the phonon call) ────
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
equil_frames = div(n_equil, log_every)
aT=Float64[]; aTs=Float64[]; ZT=Float64[]; nnT=Float64[]; fccT=Bool[]
for T_K in temperatures_K
    T = T_K*u"K"
    Random.seed!(md_seed + round(Int, T_K))       # identical across members
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
    @printf("  T = %4.0f K (seed %d): %d + %d steps …\n",
            T_K, md_seed+round(Int,T_K), n_equil, n_prod); flush(stdout)
    el = @elapsed Molly.simulate!(sys_md, sim, n_equil + n_prod)
    ana = analyze_md(sys_md, "$outdir/T$(round(Int,T_K))K", T_K;
                     log_every=log_every, equil_frames=equil_frames, N_super=N_super)
    push!(aT,ana.a_T); push!(aTs,ana.a_T_std)
    push!(ZT,ana.Z); push!(nnT,ana.nnv); push!(fccT,ana.still_fcc)
    @printf("    a = %.5f ± %.5f Å | ⟨T⟩ = %.0f K | ⟨Z⟩ = %.2f → %s  [%.1f min]\n",
            ana.a_T, ana.a_T_std, ana.mean_T, ana.Z,
            ana.still_fcc ? "FCC" : "LEFT FCC", el/60); flush(stdout)
end

nf = count(fccT)
α = nf >= 2 ? (hcat(ones(nf), temperatures_K[fccT]) \ aT[fccT])[2] / a0 : NaN
@printf("\n  FCC at %d/%d temperatures (legacy coordination test) | α = %s\n",
        nf, length(temperatures_K),
        isnan(α) ? "not fitted" : @sprintf("%.2f ×10⁻⁶ K⁻¹", α*1e6))
@printf("  → authoritative verdict: python3 scripts/uq/fcc_order_ensemble_Al_12_4_6A_2.py %s\n",
        "$MODEL_DIR/results/npt_naive"); flush(stdout)

open("$outdir/thermal_expansion_summary.csv", "w") do io
    println(io, "# role=$role  source=$COMM_CSV  row=$idx  " *
                "sha=$(bytes2hex(sha256(string(θ)))[1:16])")
    println(io, "# still_fcc uses the legacy fixed $(nn_cutoff) Å coordination cutoff — " *
                "prefer the q6bar verdict from fcc_order_ensemble_Al_12_4_6A_2.py")
    println(io, "T_K,a_Ang,a_std_Ang,delta_a_over_a0_pct,mean_coord,median_nn_Ang,still_fcc")
    @printf(io, "0,%.6f,0,0,12.00,%.4f,true\n", a0, a0/sqrt(2))
    for (T,a,s,z,nn,ok) in zip(temperatures_K, aT, aTs, ZT, nnT, fccT)
        @printf(io, "%.0f,%.6f,%.6f,%.4f,%.4f,%.4f,%s\n",
                T, a, s, 100*(a-a0)/a0, z, nn, ok)
    end
end

s_temps = join(round.(Int, temperatures_K), " ")   # built outside the interpolation:
                                                   # escaped quotes inside $() do not parse
open("$outdir/metadata.csv", "w") do io
    println(io, "key,value")
    println(io, "role,$role"); println(io, "model,Al_20_4_6A_4")
    println(io, "n_params,$(length(θ))")
    println(io, "source_file,$COMM_CSV"); println(io, "source_row,$idx")
    println(io, "theta_sha256,$(bytes2hex(sha256(string(θ))))")
    println(io, "a0_Ang,$a0")
    println(io, "supercell,$(N_super)^3")
    println(io, "n_steps_per_T,$(n_equil + n_prod)")
    println(io, "temperatures_K,$s_temps")
    println(io, "md_seed_base,$md_seed")
    println(io, "n_equil,$n_equil"); println(io, "n_prod,$n_prod")
    println(io, "constraints,none (naive POPS hypercube)")
end

@printf("\ndone: %s\n", outdir)
