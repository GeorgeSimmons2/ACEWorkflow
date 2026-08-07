# vacancy_formation.jl
#
# Vacancy formation energy across a POPS committee, with the FULL geometry
# minimisation redone for every member.
#
# ── WHAT CHANGED AND WHY ────────────────────────────────────────────────────
# The previous version relaxed the bulk cell and the vacancy supercell ONCE, with the
# mean model, and then evaluated every committee member as dot(E_def_design, θ) — the
# member's energy functional contracted against the MEAN model's relaxed geometries.
# That is a frozen-geometry estimate.  It is exact only if every member relaxes to the
# same structure as the mean model, which is precisely what a committee does not do:
# members differ in their equilibrium lattice constant and in how far the atoms around
# the vacancy pull in.  Because E_f is a difference of two large energies, a small
# geometry error does not cancel between the two terms.
#
# Here each member relaxes its OWN bulk cell (variable cell) and its OWN vacancy
# supercell (fixed cell, relaxed positions), and E_f is formed from its own energies.
#
# The frozen-geometry estimate is still computed, for free, as a CONTROL: the spread of
# (full − frozen) is how much the relaxation actually mattered, and is the number that
# justifies the extra cost.  If it is negligible the cheap route was fine after all;
# report it either way rather than assuming.
#
# ── PARALLELISM ─────────────────────────────────────────────────────────────
# Two minimisations per member.  Each task gets its OWN deep copy of the model:
# set_linear_parameters! mutates, so a shared model would be a data race producing
# quietly wrong energies rather than a crash.  Chunked @spawn, not @threads with
# threadid(), which is unsound under task migration.
#
# Minimisation of a pathological member can fail or hit the iteration cap.  Every
# member is wrapped, failures are counted and excluded from the statistics rather than
# silently entering the histogram as a garbage number.
#
# Run:  julia --project -t <ncores> scripts/qoi/vacancy_formation.jl [committee.csv]
#   ELEMENT=Al        element to build (bulk(element; cubic=true))
#   N_SUPER=4         supercell repeat for the vacancy cell
#   QOI_THREADS       concurrent minimisations (default: all Julia threads)
#   The committee CSV has one FULL coefficient vector per row, e.g. the samples_*.csv
#   or committee_*.csv written by the UQ scripts.  If a `committee` variable is already
#   in the session it is used instead and the argument is ignored.

using ACEWorkflow, ACEpotentials, AtomsBuilder, GeometryOptimization
using LinearAlgebra, Statistics, DelimitedFiles, Printf, Serialization, Unitful
using ACEpotentials: potential_energy
import ACEpotentials.Models: potential_energy_basis
using CairoMakie

element      = Symbol(get(ENV, "ELEMENT", "Al"))   # the original fragment said :W;
                                                   # every model in models/ is Al
N_SUPER      = parse(Int, get(ENV, "N_SUPER", "4"))
QOI_THREADS  = parse(Int, get(ENV, "QOI_THREADS", string(Threads.nthreads())))
FIGW         = parse(Float64, get(ENV, "FIGW", "540"))

# ── model + committee ───────────────────────────────────────────────────────
if !@isdefined(model) || model === nothing
    MODELDIR = get(ENV, "MODELDIR", "models/Al_12_4_6A_2_")
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

if !@isdefined(committee) || committee === nothing
    length(ARGS) >= 1 || error("give a committee CSV (one full coefficient vector per row)")
    M = readdlm(ARGS[1], ',')
    size(M, 2) == n_params ||
        error("$(ARGS[1]) is $(size(M,2)) wide, model has $n_params parameters")
    global committee = [collect(Float64, M[i, :]) for i in 1:size(M, 1)]
    @printf("committee: %d members from %s\n", length(committee), ARGS[1])
else
    committee isa AbstractMatrix &&
        (global committee = [collect(Float64, committee[i, :]) for i in 1:size(committee,1)])
    @printf("committee: %d members from the session\n", length(committee))
end
outdir = get(ENV, "OUTDIR", "results/vacancy_formation"); mkpath(outdir)
flush(stdout)

# ── one member: relax bulk, relax vacancy supercell, form E_f ───────────────
# `m` is a per-task model copy.  Never pass the shared one.
function vacancy_formation(m, θ)
    ACEpotentials.Models.set_linear_parameters!(m, θ)

    sys_bulk = bulk(element; cubic=true)
    rb   = minimize_energy!(sys_bulk, m; variablecell=true)
    bopt = rb.system

    sys_vac = deleteat!(deepcopy(bopt) * (N_SUPER, N_SUPER, N_SUPER), 1)
    rv   = minimize_energy!(sys_vac, m; variablecell=false)
    vopt = rv.system

    Eb = ustrip(u"eV", potential_energy(bopt, m))
    Ev = ustrip(u"eV", potential_energy(vopt, m))
    Ef = Ev - length(vopt) * Eb / length(bopt)

    conv(r) = hasproperty(r, :converged) ? r.converged : missing
    vol = abs(det(ustrip.(reduce(hcat, collect(bopt.cell.cell_vectors)))))
    return (; Ef, vol_per_atom = vol / length(bopt), n_bulk = length(bopt),
              n_vac = length(vopt), conv_bulk = conv(rb), conv_vac = conv(rv))
end

function vacancy_formation_many(θs)
    n = length(θs)
    out  = Vector{Any}(undef, n)
    errs = Vector{Union{Nothing,String}}(nothing, n)
    nt = clamp(QOI_THREADS, 1, min(Threads.nthreads(), n))
    done = Threads.Atomic{Int}(0)
    @printf("\n── %d members × 2 minimisations, %d tasks, %d deep ──\n", n, nt, cld(n, nt))
    flush(stdout)
    t = @elapsed @sync for k in 1:nt
        Threads.@spawn begin
            m = deepcopy(model)                       # private to this task
            for i in k:nt:n
                try
                    out[i] = vacancy_formation(m, θs[i])
                catch e
                    errs[i] = sprint(showerror, e)
                    out[i]  = nothing
                end
                d = Threads.atomic_add!(done, 1) + 1
                d % max(1, n ÷ 10) == 0 && (@printf("  %d/%d\n", d, n); flush(stdout))
            end
        end
    end
    @printf("  done in %.1f min (%.1f s/member)\n", t/60, t/n); flush(stdout)
    return out, errs
end

# ── mean model first: its geometries define the frozen-geometry control ─────
println("\n── mean model ──"); flush(stdout)
mean_res = vacancy_formation(deepcopy(model), lin_params)
@printf("  E_f = %.4f eV   (bulk %d atoms, vacancy cell %d atoms, V/atom %.4f Å³)\n",
        mean_res.Ef, mean_res.n_bulk, mean_res.n_vac, mean_res.vol_per_atom)

# rebuild the mean model's relaxed structures once to get the basis-resolved functional
let m = deepcopy(model)
    ACEpotentials.Models.set_linear_parameters!(m, lin_params)
    sb = bulk(element; cubic=true)
    bopt = minimize_energy!(sb, m; variablecell=true).system
    sv = deleteat!(deepcopy(bopt) * (N_SUPER, N_SUPER, N_SUPER), 1)
    vopt = minimize_energy!(sv, m; variablecell=false).system
    global E_def_design =
        ustrip.(u"eV", potential_energy_basis(vopt, m)) .-
        (length(vopt) .* ustrip.(u"eV", potential_energy_basis(bopt, m)) ./ length(bopt))
end

# ── the committee ───────────────────────────────────────────────────────────
res, errs = vacancy_formation_many(committee)
ok  = findall(!isnothing, res)
bad = findall(isnothing, res)
if !isempty(bad)
    @printf("\n%d/%d members FAILED to minimise and are excluded:\n", length(bad), length(res))
    for i in bad[1:min(5, end)]; @printf("  member %d: %s\n", i, first(split(errs[i], '\n'))); end
    length(bad) > 5 && @printf("  … and %d more\n", length(bad) - 5)
end
isempty(ok) && error("no member minimised successfully")

Ef_full   = [res[i].Ef for i in ok]
Ef_frozen = [dot(E_def_design, committee[i]) for i in ok]   # mean model's geometries
vpa       = [res[i].vol_per_atom for i in ok]
Ef_mean_frozen = dot(E_def_design, lin_params)

nconv = count(i -> res[i].conv_bulk === true && res[i].conv_vac === true, ok)
@printf("\nconverged (both minimisations): %s\n",
        any(i -> res[i].conv_bulk === missing, ok) ?
        "not reported by this GeometryOptimization version" :
        @sprintf("%d/%d", nconv, length(ok)))

@printf("\nmean model      : E_f = %.4f eV (full relaxation), %.4f eV (its own frozen functional — should match)\n",
        mean_res.Ef, Ef_mean_frozen)
@printf("committee, FULL : E_f ∈ [%.4f, %.4f] eV, mean %.4f ± %.4f\n",
        minimum(Ef_full), maximum(Ef_full), mean(Ef_full), std(Ef_full))
@printf("committee, FROZEN geometry: E_f ∈ [%.4f, %.4f] eV, mean %.4f ± %.4f\n",
        minimum(Ef_frozen), maximum(Ef_frozen), mean(Ef_frozen), std(Ef_frozen))

d = Ef_full .- Ef_frozen
@printf("\nRELAXATION MATTERS BY: full − frozen ∈ [%+.4f, %+.4f] eV, mean %+.4f, |Δ| max %.4f\n",
        minimum(d), maximum(d), mean(d), maximum(abs.(d)))
@printf("  that is %.1f%% of the committee spread (σ_full = %.4f eV)\n",
        100*maximum(abs.(d))/max(std(Ef_full), eps()), std(Ef_full))
println("  → if this is small the frozen-geometry shortcut was safe; if it is comparable")
println("    to σ_full, the previous version's error bars were not the committee's.")
@printf("\nrelaxed volume per atom across the committee: [%.4f, %.4f] Å³ (mean model %.4f)\n",
        minimum(vpa), maximum(vpa), mean_res.vol_per_atom)
flush(stdout)

# ── figure ──────────────────────────────────────────────────────────────────
BLU = RGBf(0.0, 0.447, 0.698); ORN = RGBf(0.835, 0.369, 0.0); RED = RGBf(0.80, 0.15, 0.15)
TITLE, LAB, TICK = 13, 12, 11

fig = Figure(size=(FIGW, 0.42FIGW), figure_padding=(6, 10, 4, 6))
ax = Axis(fig[1, 1]; xlabel="Vacancy formation energy (eV)", ylabel="density",
          title="Full relaxation per member", titlesize=TITLE,
          xlabelsize=LAB, ylabelsize=LAB, xticklabelsize=TICK, yticklabelsize=TICK,
          xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1)
hist!(ax, Ef_full; bins=40, normalization=:pdf, color=(BLU, 0.55),
      strokecolor=:white, strokewidth=0.4)
vlines!(ax, [mean_res.Ef]; color=RED, linewidth=1.8)

ax2 = Axis(fig[1, 2]; xlabel="E_f full − frozen (eV)", ylabel="density",
           title="What the relaxation changed", titlesize=TITLE,
           xlabelsize=LAB, ylabelsize=LAB, xticklabelsize=TICK, yticklabelsize=TICK,
           xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1)
hist!(ax2, d; bins=40, normalization=:pdf, color=(ORN, 0.6),
      strokecolor=:white, strokewidth=0.4)
vlines!(ax2, [0.0]; color=:black, linestyle=:dash, linewidth=1.0)
colgap!(fig.layout, 20)
save("$outdir/vacancy_formation.pdf", fig)
save("$outdir/vacancy_formation.png", fig; px_per_unit=4)

writedlm("$outdir/vacancy_formation.csv",
         hcat(ok, Ef_full, Ef_frozen, d, vpa), ',')
serialize("$outdir/vacancy_formation.jls",
          (; Ef_full, Ef_frozen, delta=d, vol_per_atom=vpa, ok, failed=bad,
             mean_Ef=mean_res.Ef, mean_frozen=Ef_mean_frozen, E_def_design,
             element, N_SUPER, n_members=length(committee)))
println("\nfigure → $outdir/vacancy_formation.{pdf,png}")
println("data   → $outdir/vacancy_formation.csv  (member, E_f full, E_f frozen, Δ, V/atom)")
