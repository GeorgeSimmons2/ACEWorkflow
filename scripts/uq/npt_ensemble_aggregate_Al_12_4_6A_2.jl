# npt_ensemble_aggregate_Al_12_4_6A_2.jl
#
# Combine the 6 NPT ensemble trajectories into committee-level results with POPS
# uncertainty bands.  Run after the array job in run_npt_ensemble.slurm completes.
#
# Because every member integrated with IDENTICAL MD seeds, the spread across members
# is attributable to the parameter vectors, not to MD noise — so these bands are a
# committee uncertainty, not a mixture of two noise sources.
#
# Produces
#   ensemble_thermal_expansion.pdf/png  a(T) per member + mean ± spread, α per member
#   ensemble_rdf.pdf/png                g(r) envelope across members at each T
#   ensemble_phonons.pdf/png            min ω vs volume, and vs a(T), across members
#   ensemble_summary.csv                per-member α, FCC survival, worst min ω
#   ensemble_alpha.csv                  α mean, std, min, max over members
#
# Run:  julia --project scripts/uq/npt_ensemble_aggregate_Al_12_4_6A_2.jl [ensemble_dir]
#         ensemble_dir defaults to npt_ensemble (constrained); pass npt_ensemble_naive
#         for the unconstrained control.

include(joinpath(@__DIR__, "..", "bandpath_phonon_uq", "lib.jl"))

result   = load_model(:Al, 12, 4, 6, 2; dataset_name="")
# argv[1] = which ensemble directory under results/ (default the constrained one)
ens_name = isempty(ARGS) ? "npt_ensemble" : ARGS[1]
ens_root = "$(result.dir)/results/$ens_name"
isdir(ens_root) || error("no ensemble at $ens_root — run the array job first")
roles = sort(filter(d -> isdir(joinpath(ens_root, d)) &&
                         isfile(joinpath(ens_root, d, "thermal_expansion_summary.csv")),
                    readdir(ens_root)))
isempty(roles) && error("no completed member directories in $ens_root")
@printf("aggregating %d members: %s\n\n", length(roles), join(roles, ", "))

# ── read each member's summary ───────────────────────────────────────────────
mem = Dict{String,Any}()
for r in roles
    f = joinpath(ens_root, r, "thermal_expansion_summary.csv")
    lines = readlines(f); ncmt = count(l -> startswith(strip(l), "#"), lines)
    d = readdlm(f, ','; skipstart=ncmt+1)
    T = Float64.(d[:,1]); a = Float64.(d[:,2]); w = Float64.(d[:,5])
    Z = Float64.(d[:,6]); fcc = [strip(string(x)) == "true" for x in d[:,8]]
    a0 = a[1]
    keep = (T .> 0) .& fcc
    α = count(keep) >= 2 ? (hcat(ones(count(keep)), T[keep]) \ a[keep])[2]/a0 : NaN
    mem[r] = (; T, a, w, Z, fcc, a0, α, nfcc=count(keep))
    @printf("  %-20s a₀ = %.5f  α = %8s  FCC at %d/%d  worst ω = %+.3f\n",
            r, a0, isnan(α) ? "NA" : @sprintf("%.2f", α*1e6), count(keep), count(T .> 0), minimum(w))
end

Ts = sort(unique(vcat([m.T[m.T .> 0] for m in values(mem)]...)))
BLU = RGBf(0.0,0.447,0.698); GRN = RGBf(0.0,0.62,0.451); CRIM = RGBf(0.80,0.15,0.15)
GREY = RGBAf(0.45,0.45,0.45,0.55)
ismean(r) = r in ("constrained_mean", "rls_model")   # index-0 reference of either ensemble

# ── (1) thermal expansion with committee band ────────────────────────────────
let fig = Figure(size=(520,380), figure_padding=(6,10,4,6))
    ax = Axis(fig[1,1]; xlabel="Temperature (K)", ylabel="Lattice constant a (Å)",
              title="NPT ensemble — a(T) across the constrained committee",
              titlesize=11, xlabelsize=11, ylabelsize=11, xgridvisible=false,
              ygridvisible=false, xtickalign=1, ytickalign=1)
    for r in roles
        m = mem[r]; sel = m.T .> 0
        col = ismean(r) ? BLU : GREY
        lines!(ax, m.T[sel], m.a[sel]; color=col, linewidth=ismean(r) ? 2.2 : 1.0)
        scatter!(ax, m.T[sel .& m.fcc], m.a[sel .& m.fcc]; color=col, markersize=ismean(r) ? 10 : 6)
        any(sel .& .!m.fcc) && scatter!(ax, m.T[sel .& .!m.fcc], m.a[sel .& .!m.fcc];
                                        color=CRIM, marker=:xcross, markersize=9)
    end
    # band over members that stayed FCC at each T
    lo = Float64[]; hi = Float64[]; Tb = Float64[]
    for T in Ts
        v = [mem[r].a[findfirst(==(T), mem[r].T)] for r in roles
             if (i = findfirst(==(T), mem[r].T)) !== nothing && mem[r].fcc[i]]
        length(v) >= 2 && (push!(Tb,T); push!(lo,minimum(v)); push!(hi,maximum(v)))
    end
    length(Tb) >= 2 && band!(ax, Tb, lo, hi; color=(GRN, 0.18))
    elem = [LineElement(color=BLU), LineElement(color=GREY),
            MarkerElement(color=CRIM, marker=:xcross), PolyElement(color=(GRN,0.18))]
    Legend(fig[1,1], elem, ["constrained mean","committee members","left FCC","committee range"];
           tellwidth=false, tellheight=false, halign=:left, valign=:top,
           margin=(8,8,8,8), framevisible=true, labelsize=8, patchsize=(14,10))
    save("$ens_root/ensemble_thermal_expansion.pdf", fig)
    save("$ens_root/ensemble_thermal_expansion.png", fig; px_per_unit=4)
end

# ── (2) RDF envelope at each temperature ─────────────────────────────────────
let
    Tshow = [T for T in Ts if all(isfile(joinpath(ens_root,r,"T$(round(Int,T))K","rdf.csv")) for r in roles)]
    if !isempty(Tshow)
        fig = Figure(size=(560, 180*length(Tshow)+40), figure_padding=(6,10,4,6))
        for (k,T) in enumerate(Tshow)
            ax = Axis(fig[k,1]; xlabel = k == length(Tshow) ? "r (Å)" : "", ylabel="g(r)",
                      title="$(round(Int,T)) K", titlesize=10, xgridvisible=false,
                      ygridvisible=false, xtickalign=1, ytickalign=1)
            G = Vector{Vector{Float64}}(); rr = Float64[]
            for r in roles
                d = readdlm(joinpath(ens_root,r,"T$(round(Int,T))K","rdf.csv"), ','; skipstart=1)
                rr = Float64.(d[:,1]); push!(G, Float64.(d[:,2]))
            end
            n = minimum(length.(G)); Gm = reduce(hcat, [g[1:n] for g in G])
            band!(ax, rr[1:n], vec(minimum(Gm;dims=2)), vec(maximum(Gm;dims=2)); color=(GRN,0.30))
            im = findfirst(ismean, roles)
            im !== nothing && lines!(ax, rr[1:n], G[im][1:n]; color=BLU, linewidth=1.2)
            xlims!(ax, 0, maximum(rr[1:n]))
        end
        save("$ens_root/ensemble_rdf.pdf", fig); save("$ens_root/ensemble_rdf.png", fig; px_per_unit=4)
    end
end

# ── (3) phonons: min ω vs volume (0 K) and vs a(T) ───────────────────────────
let fig = Figure(size=(560,320), figure_padding=(6,10,4,6))
    ax1 = Axis(fig[1,1]; xlabel="a / a_eq", ylabel="min non-acoustic ω (THz)",
               title="0 K stability across the constrained volumes", titlesize=10,
               xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1)
    vs = collect(1.00:0.02:1.10)
    for r in roles
        f = joinpath(ens_root, r, "metadata.csv"); isfile(f) || continue
        kv = Dict(split(l, ',', limit=2)[1] => split(l, ',', limit=2)[2]
                  for l in readlines(f) if occursin(',', l))
        haskey(kv, "minomega_at_volumes_THz") || continue
        w = parse.(Float64, split(strip(kv["minomega_at_volumes_THz"])))
        lines!(ax1, vs, w; color = ismean(r) ? BLU : GREY, linewidth = ismean(r) ? 2.2 : 1.0)
    end
    hlines!(ax1, [0.0]; color=:black, linestyle=:dash, linewidth=0.8)
    ax2 = Axis(fig[1,2]; xlabel="Temperature (K)", ylabel="min ω at a(T) (THz)",
               title="stability at the NPT lattice constant", titlesize=10,
               xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1)
    for r in roles
        m = mem[r]; sel = m.T .> 0
        lines!(ax2, m.T[sel], m.w[sel]; color = ismean(r) ? BLU : GREY, linewidth = ismean(r) ? 2.2 : 1.0)
    end
    hlines!(ax2, [0.0]; color=:black, linestyle=:dash, linewidth=0.8)
    save("$ens_root/ensemble_phonons.pdf", fig); save("$ens_root/ensemble_phonons.png", fig; px_per_unit=4)
end

# ── summary tables ───────────────────────────────────────────────────────────
open("$ens_root/ensemble_summary.csv", "w") do io
    println(io, "role,a0_Ang,alpha_1e6_perK,n_fcc,n_T,worst_minomega_THz,source_file,source_row,theta_sha256_16")
    for r in roles
        m = mem[r]; f = joinpath(ens_root, r, "metadata.csv")
        kv = isfile(f) ? Dict(split(l, ',', limit=2)[1] => strip(split(l, ',', limit=2)[2])
                              for l in readlines(f) if occursin(',', l)) : Dict{String,Any}()
        @printf(io, "%s,%.6f,%s,%d,%d,%.4f,%s,%s,%s\n", r, m.a0,
                isnan(m.α) ? "NA" : @sprintf("%.4f", m.α*1e6), m.nfcc, count(m.T .> 0),
                minimum(m.w), get(kv,"source_file","?"), get(kv,"source_row","?"),
                first(get(kv,"theta_sha256","?"), 16))
    end
end
αs = [mem[r].α for r in roles if !isnan(mem[r].α)]
open("$ens_root/ensemble_alpha.csv", "w") do io
    println(io, "quantity,value")
    println(io, "n_members_with_alpha,$(length(αs))")
    if !isempty(αs)
        @printf(io, "alpha_mean_1e6perK,%.4f\n", mean(αs)*1e6)
        @printf(io, "alpha_std_1e6perK,%.4f\n",  (length(αs) > 1 ? std(αs) : 0.0)*1e6)
        @printf(io, "alpha_min_1e6perK,%.4f\n",  minimum(αs)*1e6)
        @printf(io, "alpha_max_1e6perK,%.4f\n",  maximum(αs)*1e6)
    end
end
if !isempty(αs)
    @printf("\n  α over %d members: %.2f ± %.2f ×10⁻⁶ K⁻¹  (range %.2f – %.2f)\n",
            length(αs), mean(αs)*1e6, (length(αs)>1 ? std(αs) : 0.0)*1e6,
            minimum(αs)*1e6, maximum(αs)*1e6)
    @printf("  experiment: 23.1 at 300 K rising to ~26 by 700 K\n")
end
println("\noutputs → $ens_root/")
