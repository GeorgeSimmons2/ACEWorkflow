# vacancy_formation.jl
#
# Vacancy formation energy: UNCONSTRAINED POPS vs CONSTRAINED POPS, with the full
# geometry minimisation redone for every member of both committees.
#
# ── THE COMPARISON ──────────────────────────────────────────────────────────
# Same two ensembles as the phonon comparison in
# scripts/uq/naive_vs_constrained_fullcloud_Al_12_4_6A_2.jl, so the vacancy result and
# the band result describe the SAME committees and can be quoted side by side:
#
#   unconstrained  results/naive_vs_constrained/samples_naive.csv
#                  top-50%-leverage POPS cloud → hypercube → sample, no predicate
#   constrained    results/cutting_plane_full_cloud/committee_rejection_full_cloud.csv
#                  the same cloud after cutting-plane constraints, then rejection
#
# The headline number is the ratio of spreads: does constraining the parameter cloud
# tighten a quantity nobody constrained directly?  E_f is not in any predicate — not in
# the Born rows, not in the a_eq pin, not in the phonon cut — so any narrowing is the
# physics prior propagating into an unconstrained observable, which is the paper's
# claim.  Any WIDENING would be evidence against it and is worth knowing too.
#
# ── FULL MINIMISATION, NOT FROZEN GEOMETRY ──────────────────────────────────
# The previous version relaxed the bulk cell and the vacancy supercell ONCE with the
# mean model and evaluated every member as dot(E_def_design, θ) — the member's energy
# functional contracted against the MEAN model's relaxed geometries.  That is exact
# only if every member relaxes like the mean model, which is what a committee does not
# do: members differ in equilibrium lattice constant and in how far the neighbours pull
# in around the vacancy.  E_f is a difference of two large energies, so a geometry error
# does not cancel between the terms.
#
# Here each member relaxes its OWN bulk cell (variable cell) and its OWN vacancy
# supercell (fixed cell, relaxed positions).  The frozen-geometry estimate is still
# computed as a CONTROL — it costs nothing once the mean geometries exist — and the
# spread of (full − frozen) is what the extra cost bought.  If it is negligible the old
# shortcut was safe; report it either way rather than assuming.
#
# ── PARALLELISM ─────────────────────────────────────────────────────────────
# Two minimisations per member.  Each task gets its OWN deep copy of the model:
# set_linear_parameters! mutates, so a shared model would be a data race producing
# quietly wrong energies rather than a crash.  Chunked @spawn, not @threads with
# threadid(), which is unsound under task migration.  Minimisation of a pathological
# member can fail; every member is wrapped, failures are counted and excluded rather
# than entering the histogram as garbage.
#
# Run:  julia --project -t <ncores> scripts/qoi/vacancy_formation.jl [unconstrained.csv constrained.csv]
#   ELEMENT=Al   N_SUPER=4   QOI_THREADS=<n>   MODELDIR=models/Al_12_4_6A_2_   FIGW=540

using ACEWorkflow, ACEpotentials, AtomsBuilder, GeometryOptimization
using LinearAlgebra, Statistics, DelimitedFiles, Printf, Serialization, Unitful
using ACEpotentials: potential_energy
import ACEpotentials.Models: potential_energy_basis
using CairoMakie

element     = Symbol(get(ENV, "ELEMENT", "Al"))   # the original fragment said :W;
                                                  # every model under models/ is Al
N_SUPER     = parse(Int, get(ENV, "N_SUPER", "4"))
QOI_THREADS = parse(Int, get(ENV, "QOI_THREADS", string(Threads.nthreads())))
FIGW        = parse(Float64, get(ENV, "FIGW", "540"))
MODELDIR    = get(ENV, "MODELDIR", "models/Al_12_4_6A_2_")
RES         = "$MODELDIR/results"
outdir      = get(ENV, "OUTDIR", "$RES/vacancy_formation"); mkpath(outdir)

BLU = RGBf(0.0, 0.447, 0.698); RED = RGBf(0.80, 0.15, 0.15); ORN = RGBf(0.835, 0.369, 0.0)

# EACH ENSEMBLE HAS ITS OWN CENTRAL MODEL, and it matters.  samples_naive.csv was drawn
# around lin_params; committee_rejection_full_cloud.csv was drawn around theta_mean, the
# phonon-repaired CONSTRAINED mean.  Using one model as "the" reference for both would
# put the constrained panel's reference line, and its whole frozen-geometry column, at a
# model its members are not centred on.
THETA_MEAN = "$RES/bandpath_undotted_ncell4_densek/theta_mean.csv"
ensembles = [
 (tag = "unconstrained", label = "unconstrained POPS", col = RED, centre = :lin_params,
  csv = length(ARGS) >= 1 ? ARGS[1] : "$RES/naive_vs_constrained/samples_naive.csv"),
 (tag = "constrained",   label = "constrained POPS",   col = BLU, centre = :theta_mean,
  csv = length(ARGS) >= 2 ? ARGS[2] : "$RES/cutting_plane_full_cloud/committee_rejection_full_cloud.csv"),
]

# ── model ───────────────────────────────────────────────────────────────────
if !@isdefined(model) || model === nothing
    name = basename(rstrip(MODELDIR, '/'))
    json = joinpath(MODELDIR, "$(rstrip(name, '_')).json")
    isfile(json) || error("no model in the session and $json not found; set MODELDIR")
    global model, _ = ACEpotentials.load_model(json)
    global lin_params = vec(readdlm(joinpath(MODELDIR, "lin_params.csv"), ','))
    ACEpotentials.Models.set_linear_parameters!(model, lin_params)
    @printf("loaded %s: %d parameters\n", name, length(lin_params))
else
    @isdefined(lin_params) || error("`model` is in the session but `lin_params` is not")
    println("using the model already in the session")
end
n_params = length(lin_params)

function read_committee(path)
    isfile(path) || error("""
        missing $path
        The defaults expect scripts/uq/naive_vs_constrained_fullcloud_Al_12_4_6A_2.jl and
        scripts/uq/hypercube_full_cloud_bands_Al_12_4_6A_2.jl to have been run, or pass
        two CSV paths as arguments.""")
    M = readdlm(path, ',')
    size(M, 2) == n_params || error("$path is $(size(M,2)) wide, model has $n_params parameters")
    return [collect(Float64, M[i, :]) for i in 1:size(M, 1)]
end
for e in ensembles
    @printf("%-14s %d members  ← %s\n", e.tag, length(read_committee(e.csv)), basename(e.csv))
end
flush(stdout)

# ── one member: relax bulk, relax vacancy supercell, form E_f ───────────────
# `m` is a per-task model copy.  Never pass the shared one.
function vacancy_formation(m, θ)
    ACEpotentials.Models.set_linear_parameters!(m, θ)
    sys_bulk = bulk(element; cubic=true)
    rb   = minimize_energy!(sys_bulk, m; variablecell=true)
    bopt = rb.system
    sys_vac  = deleteat!(deepcopy(bopt) * (N_SUPER, N_SUPER, N_SUPER), 1)
    Eb   = ustrip(u"eV", potential_energy(bopt, m))
    # UNRELAXED reference: vacancy cell straight from the relaxed bulk, positions not
    # moved.  E_f_unrelaxed − E_f is the relaxation energy, which for an Al vacancy is
    # tens of meV.  If it comes out at electronvolts the minimisation is not relaxing a
    # vacancy, it is finding a different structure entirely.
    Ev_un = ustrip(u"eV", potential_energy(sys_vac, m))
    nv    = length(sys_vac)
    Ef_un = Ev_un - nv*Eb/length(bopt)
    rv   = minimize_energy!(sys_vac, m; variablecell=false)
    vopt = rv.system
    Ev = ustrip(u"eV", potential_energy(vopt, m))
    conv(r) = hasproperty(r, :converged) ? r.converged : missing
    vol = abs(det(ustrip.(reduce(hcat, collect(bopt.cell.cell_vectors)))))
    return (; Ef = Ev - length(vopt)*Eb/length(bopt), Ef_unrelaxed = Ef_un,
              E_bulk_per_atom = Eb/length(bopt),
              vol_per_atom = vol/length(bopt),
              n_bulk = length(bopt), n_vac = length(vopt),
              conv_bulk = conv(rb), conv_vac = conv(rv), bopt, vopt)
end

function vacancy_formation_many(θs; label="")
    n = length(θs)
    out  = Vector{Any}(undef, n)
    errs = Vector{Union{Nothing,String}}(nothing, n)
    nt = clamp(QOI_THREADS, 1, min(Threads.nthreads(), n))
    done = Threads.Atomic{Int}(0)
    @printf("\n[%s] %d members × 2 minimisations, %d tasks, %d deep …\n",
            label, n, nt, cld(n, nt)); flush(stdout)
    t = @elapsed @sync for k in 1:nt
        Threads.@spawn begin
            m = deepcopy(model)                       # private to this task
            for i in k:nt:n
                try
                    out[i] = vacancy_formation(m, θs[i])
                catch e
                    errs[i] = sprint(showerror, e); out[i] = nothing
                end
                d = Threads.atomic_add!(done, 1) + 1
                d % max(1, n ÷ 10) == 0 && (@printf("  [%s] %d/%d\n", label, d, n); flush(stdout))
            end
        end
    end
    @printf("[%s] done in %.1f min (%.1f s/member)\n", label, t/60, t/n); flush(stdout)
    return out, errs
end

# ── one central model PER ENSEMBLE; each defines its own frozen-geometry control ──
centres = Dict{Symbol,Vector{Float64}}(:lin_params => lin_params)
if any(e -> e.centre === :theta_mean, ensembles)
    isfile(THETA_MEAN) || error("missing $THETA_MEAN — the constrained committee is centred on it")
    centres[:theta_mean] = vec(readdlm(THETA_MEAN, ','))
    length(centres[:theta_mean]) == n_params || error("$THETA_MEAN has the wrong width")
end

println("\n── central models ──"); flush(stdout)
centre_res = Dict{Symbol,Any}(); design = Dict{Symbol,Vector{Float64}}()
for (k, θc) in centres
    r = vacancy_formation(deepcopy(model), θc)
    centre_res[k] = r
    design[k] = ustrip.(u"eV", potential_energy_basis(r.vopt, model)) .-
        (length(r.vopt) .* ustrip.(u"eV", potential_energy_basis(r.bopt, model)) ./ length(r.bopt))
    @printf("  %-11s E_f = %+.4f eV   unrelaxed %+.4f eV   relaxation %+.4f eV\n",
            k, r.Ef, r.Ef_unrelaxed, r.Ef - r.Ef_unrelaxed)
    @printf("  %-11s bulk %d atoms at %.5f eV/atom, vacancy cell %d atoms, V/atom %.4f Å³\n",
            "", r.n_bulk, r.E_bulk_per_atom, r.n_vac, r.vol_per_atom)
    @printf("  %-11s frozen functional reproduces E_f to %.2e eV\n",
            "", abs(dot(design[k], θc) - r.Ef))
end
println("  NOTE  an Al vacancy relaxes by tens of meV.  A relaxation energy of order eV")
println("        means the minimiser is not relaxing a vacancy but finding another")
println("        structure — read that column before quoting any absolute E_f.")
mean_res = centre_res[:lin_params]

# ── both committees ─────────────────────────────────────────────────────────
out = Dict{String,Any}()
for e in ensembles
    mem = read_committee(e.csv)
    res, errs = vacancy_formation_many(mem; label=e.tag)
    ok  = findall(!isnothing, res); bad = findall(isnothing, res)
    if !isempty(bad)
        @printf("[%s] %d/%d FAILED to minimise, excluded:\n", e.tag, length(bad), length(res))
        for i in bad[1:min(3, end)]; @printf("    member %d: %s\n", i, first(split(errs[i], '\n'))); end
    end
    isempty(ok) && error("[$(e.tag)] no member minimised successfully")
    out[e.tag] = (; ok, bad, centre = e.centre, centre_Ef = centre_res[e.centre].Ef,
                  Ef     = [res[i].Ef for i in ok],
                  frozen = [dot(design[e.centre], mem[i]) for i in ok],
                  vpa    = [res[i].vol_per_atom for i in ok],
                  n      = length(mem))
end

# ── the comparison ──────────────────────────────────────────────────────────
println("\n══ VACANCY FORMATION ENERGY ══════════════════════════════════════")
for e in ensembles
    @printf("%-14s central model (%s): E_f = %+.4f eV\n",
            e.tag, e.centre, centre_res[e.centre].Ef)
end
println()
@printf("%-14s %7s %11s %11s %11s %11s\n", "ensemble", "n", "mean eV", "sd eV", "min eV", "max eV")
for e in ensembles
    r = out[e.tag]
    @printf("%-14s %7d %11.4f %11.4f %11.4f %11.4f\n", e.tag, length(r.ok),
            mean(r.Ef), std(r.Ef), minimum(r.Ef), maximum(r.Ef))
end
let u = out["unconstrained"], c = out["constrained"]
    @printf("\nspread ratio  σ_constrained / σ_unconstrained = %.3f\n", std(c.Ef)/std(u.Ef))
    @printf("range  ratio  (max−min) constrained / unconstrained = %.3f\n",
            (maximum(c.Ef)-minimum(c.Ef)) / (maximum(u.Ef)-minimum(u.Ef)))
    println("  E_f is in NO predicate — not the Born rows, not the a_eq pin, not the phonon")
    println("  cut — so a ratio below 1 is the physics prior propagating into an")
    println("  unconstrained observable.  A ratio above 1 would be evidence against that.")
end

println("\n── what the full relaxation changed (full − frozen) ──")
for e in ensembles
    r = out[e.tag]; d = r.Ef .- r.frozen
    @printf("%-14s Δ ∈ [%+.4f, %+.4f] eV, mean %+.4f, |Δ|max %.4f  = %.0f%% of that ensemble's σ\n",
            e.tag, minimum(d), maximum(d), mean(d), maximum(abs.(d)),
            100*maximum(abs.(d))/max(std(r.Ef), eps()))
end
println("  small ⇒ the frozen-geometry shortcut was safe; comparable to σ ⇒ the previous")
println("  version's error bars were not the committee's")

println("\n── relaxed volume per atom (Å³), mean model $(round(mean_res.vol_per_atom; digits=4)) ──")
for e in ensembles
    r = out[e.tag]
    @printf("%-14s [%.4f, %.4f]\n", e.tag, minimum(r.vpa), maximum(r.vpa))
end
flush(stdout)

# ── figure: both ensembles on one axis ──────────────────────────────────────
TITLE, LAB, TICK = 13, 12, 11
allEf = vcat((out[e.tag].Ef for e in ensembles)...)
edges = range(minimum(allEf), maximum(allEf); length=41)      # shared bins

fig = Figure(size=(FIGW, 0.42FIGW), figure_padding=(6, 10, 4, 6))
ax = Axis(fig[1, 1]; xlabel="Vacancy formation energy (eV)", ylabel="density",
          title="Full relaxation, every member", titlesize=TITLE,
          xlabelsize=LAB, ylabelsize=LAB, xticklabelsize=TICK, yticklabelsize=TICK,
          xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1)
for e in ensembles
    hist!(ax, out[e.tag].Ef; bins=edges, normalization=:pdf, color=(e.col, 0.55),
          strokecolor=:white, strokewidth=0.4, label=e.label)
end
for e in ensembles
    vlines!(ax, [centre_res[e.centre].Ef]; color=(e.col, 0.9), linestyle=:dash, linewidth=1.2)
end
axislegend(ax; position=:rt, labelsize=TICK, framevisible=false)

ax2 = Axis(fig[1, 2]; xlabel="E_f  full − frozen (eV)", ylabel="density",
           title="What relaxing each member changed", titlesize=TITLE,
           xlabelsize=LAB, ylabelsize=LAB, xticklabelsize=TICK, yticklabelsize=TICK,
           xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1)
alld = vcat((out[e.tag].Ef .- out[e.tag].frozen for e in ensembles)...)
edges2 = range(minimum(alld), maximum(alld); length=41)
for e in ensembles
    hist!(ax2, out[e.tag].Ef .- out[e.tag].frozen; bins=edges2, normalization=:pdf,
          color=(e.col, 0.55), strokecolor=:white, strokewidth=0.4)
end
vlines!(ax2, [0.0]; color=:black, linestyle=:dash, linewidth=1.0)
colgap!(fig.layout, 20)
save("$outdir/vacancy_formation.pdf", fig)
save("$outdir/vacancy_formation.png", fig; px_per_unit=4)

for e in ensembles
    r = out[e.tag]
    writedlm("$outdir/vacancy_formation_$(e.tag).csv",
             hcat(r.ok, r.Ef, r.frozen, r.Ef .- r.frozen, r.vpa), ',')
end
serialize("$outdir/vacancy_formation.jls",
          (; out, mean_Ef = mean_res.Ef, mean_vpa = mean_res.vol_per_atom,
             centre_Ef = Dict(k => r.Ef for (k, r) in centre_res),
             centre_Ef_unrelaxed = Dict(k => r.Ef_unrelaxed for (k, r) in centre_res),
             design, element, N_SUPER,
             sources = Dict(e.tag => e.csv for e in ensembles)))
println("\nfigure → $outdir/vacancy_formation.{pdf,png}")
println("data   → $outdir/vacancy_formation_{unconstrained,constrained}.csv")
println("         columns: member, E_f full, E_f frozen, Δ, V/atom")
