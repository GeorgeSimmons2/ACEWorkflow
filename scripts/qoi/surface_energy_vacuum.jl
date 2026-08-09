# surface_energy_vacuum.jl
#
# Surface energy across a POPS committee, unconstrained vs constrained, by the
# add-vacuum route:
#
#     take the relaxed bulk supercell, stretch ONE cell vector by `vacuum` so the
#     periodic images separate along that normal, relax positions at fixed cell, then
#
#         E_slab = 2 A γ + E_bulk      ⇒      γ = (E_slab − E_bulk) / (2A)
#
#     with A the area of the cell face and the 2 because periodicity gives two free
#     surfaces.  For a cubic conventional FCC cell stretched along z this is the (001)
#     surface.
#
# ── WHY THIS IS BETTER CONDITIONED THAN THE VACANCY CALCULATION ─────────────
# Both terms have the SAME atom count and the same atoms, so the reference cancels
# exactly.  E_f for a vacancy needs E_vac − (N−1)·E_coh, where the per-atom cohesive
# energy is multiplied by 255 and any error in it is amplified accordingly.  Here
# E_bulk enters once, undivided.  γ should come out POSITIVE — that is the sanity
# check the vacancy number failed.
#
# Reference: Al(001) is about 0.9–1.0 J/m² in DFT, (111) about 0.7.
#
# ── THE TWO ENSEMBLES ───────────────────────────────────────────────────────
# Same committees as the phonon and vacancy comparisons, each with its OWN central
# model, so all three results describe the same ensembles:
#   unconstrained  results/naive_vs_constrained/samples_naive.csv        (← lin_params)
#   constrained    results/cutting_plane_full_cloud/committee_rejection_full_cloud.csv
#                                                                        (← theta_mean)
# γ is in no predicate, so a narrower constrained spread is the prior reaching an
# observable nobody constrained.  A wider one would be evidence against that.
#
# ── GEOMETRY CHECKS, DONE NOT ASSUMED ───────────────────────────────────────
# The two surfaces must not see each other, through the vacuum OR through the slab:
#   vacuum         > cutoff        else the surfaces interact across the gap
#   slab thickness > 2 × cutoff    else they interact through the material
# Both are checked against the model's actual cutoff and reported.
#
# ── PARALLELISM ─────────────────────────────────────────────────────────────
# Two minimisations per member, each task with its OWN deep copy of the model —
# set_linear_parameters! mutates.  Chunked @spawn, not @threads with threadid().
# Failures are caught per member, counted, and excluded rather than entering the
# histogram as garbage.
#
# Run:  julia --project -t <ncores> scripts/qoi/surface_energy_vacuum.jl [unconstrained.csv constrained.csv]
#   ELEMENT=Al  SURFACE=001|111  N_SUPER=4  VACUUM=12.0  NORMAL=3  QOI_THREADS=<n>
#   MODELDIR=models/Al_12_4_6A_2_  FIGW=540

using ACEWorkflow, ACEpotentials, AtomsBase, AtomsBuilder, GeometryOptimization
using LinearAlgebra, Statistics, DelimitedFiles, Printf, Serialization, Unitful
using ACEpotentials: potential_energy
import AtomsCalculatorsUtilities.SitePotentials: cutoff_radius
using CairoMakie

element     = Symbol(get(ENV, "ELEMENT", "Al"))
N_SUPER     = parse(Int,     get(ENV, "N_SUPER", "4"))
VACUUM      = parse(Float64, get(ENV, "VACUUM",  "12.0"))   # Å added along the normal
NORMAL      = parse(Int,     get(ENV, "NORMAL",  "3"))      # which cell vector to stretch
SURFACE     = get(ENV, "SURFACE", "001")                    # "001" or "111"
SURFACE in ("001", "111") || error("SURFACE must be \"001\" or \"111\", got $SURFACE")
QOI_THREADS = parse(Int,     get(ENV, "QOI_THREADS", string(Threads.nthreads())))
FIGW        = parse(Float64, get(ENV, "FIGW", "540"))
MODELDIR    = get(ENV, "MODELDIR", "models/Al_12_4_6A_2_")
RES         = "$MODELDIR/results"
outdir      = get(ENV, "OUTDIR", "$RES/surface_energy"); mkpath(outdir)
EV_PER_A2_TO_J_PER_M2 = 16.0218

BLU = RGBf(0.0, 0.447, 0.698); RED = RGBf(0.80, 0.15, 0.15)

THETA_MEAN = "$RES/bandpath_undotted_ncell4_densek/theta_mean.csv"
ensembles = [
 (tag = "unconstrained", label = "unconstrained POPS", col = RED, centre = :lin_params,
  csv = length(ARGS) >= 1 ? ARGS[1] : "$RES/naive_vs_constrained/samples_naive.csv"),
 (tag = "constrained",   label = "constrained POPS",   col = BLU, centre = :theta_mean,
  csv = length(ARGS) >= 2 ? ARGS[2] : "$RES/cutting_plane_full_cloud/committee_rejection_full_cloud.csv"),
]

# ── model ───────────────────────────────────────────────────────────────────
if !@isdefined(model) || model === nothing
    nm   = basename(rstrip(MODELDIR, '/'))
    json = joinpath(MODELDIR, "$(rstrip(nm, '_')).json")
    isfile(json) || error("no model in the session and $json not found; set MODELDIR")
    global model, _ = ACEpotentials.load_model(json)
    global lin_params = vec(readdlm(joinpath(MODELDIR, "lin_params.csv"), ','))
    ACEpotentials.Models.set_linear_parameters!(model, lin_params)
    @printf("loaded %s: %d parameters\n", nm, length(lin_params))
else
    @isdefined(lin_params) || error("`model` is in the session but `lin_params` is not")
    println("using the model already in the session")
end
n_params = length(lin_params)
rcut = try ustrip(cutoff_radius(model)) catch; 6.0 end
@printf("model cutoff %.2f Å; vacuum %.2f Å along cell vector %d\n", rcut, VACUUM, NORMAL)
VACUUM > rcut || @warn "vacuum ($VACUUM Å) is not larger than the cutoff ($rcut Å) — the two surfaces interact ACROSS THE GAP and γ is meaningless"

function read_committee(path)
    isfile(path) || error("missing $path — run the UQ scripts first, or pass two CSV paths")
    M = readdlm(path, ',')
    size(M, 2) == n_params || error("$path is $(size(M,2)) wide, model has $n_params parameters")
    return [collect(Float64, M[i, :]) for i in 1:size(M, 1)]
end
for e in ensembles
    @printf("%-14s %d members  ← %s\n", e.tag, length(read_committee(e.csv)), basename(e.csv))
end
flush(stdout)

# ── oriented unit cell ──────────────────────────────────────────────────────
# (001) is the conventional cubic cell, whose third vector is already the surface
# normal.  (111) needs a cell whose third vector lies along [111] — for a cubic lattice
# that direction IS the (111) normal, so `add_vacuum` then opens the right gap.
#
#     a1 = a/2 [1,-1, 0]      in-plane FCC lattice vectors
#     a2 = a/2 [0, 1,-1]
#     a3 = a   [1, 1, 1]      three (111) layers, spacing |a3|/3 = a/sqrt(3)
#
# det = 3a³/4 against a primitive volume of a³/4, so the cell holds exactly 3 atoms —
# the ABC stacking.  Rather than hard-coding their positions (easy to get subtly wrong),
# the FCC lattice points inside the cell are found by enumeration and the count is
# asserted.  Face area |a1 x a2| = sqrt(3)a²/4, the standard (111) area per surface atom.
function oriented_cell(a)
    SURFACE == "001" && return nothing        # caller uses the relaxed cubic cell as-is
    a1 = (a/2) .* [ 1.0, -1.0,  0.0]
    a2 = (a/2) .* [ 0.0,  1.0, -1.0]
    a3 =  a    .* [ 1.0,  1.0,  1.0]
    C    = hcat(a1, a2, a3)
    Cinv = inv(C)
    f = [(a/2) .* [1.0,1.0,0.0], (a/2) .* [0.0,1.0,1.0], (a/2) .* [1.0,0.0,1.0]]
    pts = Vector{Vector{Float64}}()
    for i in -4:4, j in -4:4, k in -4:4
        p = i .* f[1] .+ j .* f[2] .+ k .* f[3]
        s = Cinv * p
        all(x -> -1e-8 <= x < 1 - 1e-8, s) && push!(pts, p)
    end
    length(pts) == 3 || error("(111) cell enumeration found $(length(pts)) atoms, expected 3")
    atoms = [AtomsBase.Atom(element, p .* u"Å") for p in pts]
    return periodic_system(atoms, [a1, a2, a3] .* u"Å")
end

# ── stretch one cell vector, keep the atoms where they are ──────────────────
# Single-element system, so every atom is rebuilt as `element`; positions are copied in
# Cartesian coordinates so the slab is a genuine cleave, not a rescale of the contents.
function add_vacuum(sys, vac_A, axis)
    cv = collect(AtomsBase.cell(sys).cell_vectors)
    n̂  = cv[axis] ./ norm(cv[axis])
    cv[axis] = cv[axis] .+ (vac_A * u"Å") .* n̂
    atoms = [AtomsBase.Atom(element, AtomsBase.position(sys, i)) for i in 1:length(sys)]
    return periodic_system(atoms, cv)
end

# NOT a parenthesised one-liner: `i, j = ...` inside `( ; )` parses as a named-tuple
# element and fails.  Plain function body.
function face_area(sys, axis)                                                 # Å²
    cv   = collect(AtomsBase.cell(sys).cell_vectors)
    i, j = filter(!=(axis), 1:3)
    return norm(cross(ustrip.(cv[i]), ustrip.(cv[j])))
end

# ── one member ──────────────────────────────────────────────────────────────
# `m` is a per-task model copy.  Never pass the shared one.
function surface_energy(m, θ)
    ACEpotentials.Models.set_linear_parameters!(m, θ)

    # relax the cubic cell to fix the lattice constant, then orient.  The (001) path is
    # unchanged: it uses that relaxed cubic cell directly.
    bopt = minimize_energy!(bulk(element; cubic=true), m; variablecell=true).system
    a_rel = norm(ustrip.(collect(AtomsBase.cell(bopt).cell_vectors)[1]))
    cell1 = SURFACE == "001" ? bopt : oriented_cell(a_rel)
    bulk_s = deepcopy(cell1) * (N_SUPER, N_SUPER, N_SUPER)
    E_bulk = ustrip(u"eV", potential_energy(bulk_s, m))

    A     = face_area(bulk_s, NORMAL)                       # Å², same for slab and bulk
    thick = norm(ustrip.(collect(AtomsBase.cell(bulk_s).cell_vectors)[NORMAL]))

    slab0 = add_vacuum(bulk_s, VACUUM, NORMAL)
    # unrelaxed: the cleaved surface before the atoms move.  Relaxation must LOWER it.
    γ_un  = (ustrip(u"eV", potential_energy(slab0, m)) - E_bulk) / (2A)

    rs   = minimize_energy!(slab0, m; variablecell=false)
    Eslb = ustrip(u"eV", potential_energy(rs.system, m))
    γ    = (Eslb - E_bulk) / (2A)

    conv(r) = hasproperty(r, :converged) ? r.converged : missing
    return (; γ, γ_un, A, thick, E_bulk, n_atoms = length(bulk_s),
              a = a_rel,
              conv_slab = conv(rs))
end

function surface_energy_many(θs; label="")
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
                    out[i] = surface_energy(m, θs[i])
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

# ── central models ──────────────────────────────────────────────────────────
centres = Dict{Symbol,Vector{Float64}}(:lin_params => lin_params)
if any(e -> e.centre === :theta_mean, ensembles)
    isfile(THETA_MEAN) || error("missing $THETA_MEAN")
    centres[:theta_mean] = vec(readdlm(THETA_MEAN, ','))
end
println("\n── central models ──"); flush(stdout)
centre_res = Dict{Symbol,Any}()
for (k, θc) in centres
    r = surface_energy(deepcopy(model), θc); centre_res[k] = r
    @printf("  %-11s γ = %+.4f eV/Å² = %+.3f J/m²   (unrelaxed %+.3f J/m², relaxation %+.3f)\n",
            k, r.γ, r.γ*EV_PER_A2_TO_J_PER_M2, r.γ_un*EV_PER_A2_TO_J_PER_M2,
            (r.γ - r.γ_un)*EV_PER_A2_TO_J_PER_M2)
    @printf("  %-11s %d atoms, a = %.4f Å, face area %.2f Å², slab thickness %.2f Å\n",
            "", r.n_atoms, r.a, r.A, r.thick)
    r.thick > 2rcut || @warn "slab thickness $(round(r.thick;digits=2)) Å ≤ 2×cutoff — the two surfaces interact THROUGH THE SLAB; raise N_SUPER"
    r.γ > 0 || @warn "γ ≤ 0 for $k — a surface that lowers the energy is unphysical"
end
@printf("  reference for Al(%s): %s J/m² in DFT\n", SURFACE,
        SURFACE == "001" ? "0.9-1.0" : "~0.7")
SURFACE == "111" && println("  (111) is the close-packed face and should come out BELOW (001)")
flush(stdout)

# ── committees ──────────────────────────────────────────────────────────────
out = Dict{String,Any}()
for e in ensembles
    mem = read_committee(e.csv)
    res, errs = surface_energy_many(mem; label=e.tag)
    ok = findall(!isnothing, res); bad = findall(isnothing, res)
    if !isempty(bad)
        @printf("[%s] %d/%d FAILED, excluded:\n", e.tag, length(bad), length(res))
        for i in bad[1:min(3,end)]; @printf("    member %d: %s\n", i, first(split(errs[i], '\n'))); end
    end
    isempty(ok) && error("[$(e.tag)] no member succeeded")
    out[e.tag] = (; ok, bad, centre = e.centre,
                  γ    = [res[i].γ    for i in ok],
                  γ_un = [res[i].γ_un for i in ok],
                  A    = [res[i].A    for i in ok])
end

# ── results ─────────────────────────────────────────────────────────────────
J(x) = x .* EV_PER_A2_TO_J_PER_M2
println("\n══ SURFACE ENERGY  $(element)($(SURFACE))  ═══════════════════════════════")
for e in ensembles
    @printf("%-14s central model (%s): γ = %+.3f J/m²\n",
            e.tag, e.centre, centre_res[e.centre].γ * EV_PER_A2_TO_J_PER_M2)
end
println()
@printf("%-14s %6s %11s %10s %10s %10s %8s\n",
        "ensemble", "n", "mean J/m²", "sd", "min", "max", "γ≤0")
for e in ensembles
    g = J(out[e.tag].γ)
    @printf("%-14s %6d %11.4f %10.4f %10.4f %10.4f %8d\n",
            e.tag, length(g), mean(g), std(g), minimum(g), maximum(g), count(<=(0), g))
end
let u = out["unconstrained"], c = out["constrained"]
    @printf("\nspread ratio  σ_constrained / σ_unconstrained = %.3f\n", std(c.γ)/std(u.γ))
    @printf("range  ratio  = %.3f\n",
            (maximum(c.γ)-minimum(c.γ)) / (maximum(u.γ)-minimum(u.γ)))
    println("  γ is in NO predicate, so a ratio below 1 is the prior reaching an")
    println("  unconstrained observable.  Above 1 would be evidence against it.")
end
println("\n── relaxation of the cleaved surface (γ − γ_unrelaxed, J/m²) ──")
for e in ensembles
    d = J(out[e.tag].γ .- out[e.tag].γ_un)
    @printf("%-14s [%+.4f, %+.4f], mean %+.4f   %s\n", e.tag, minimum(d), maximum(d), mean(d),
            all(<=(1e-9), d) ? "(all ≤ 0, as relaxation must be)" : "← SOME POSITIVE: minimiser went uphill")
end
flush(stdout)

# ── figure ──────────────────────────────────────────────────────────────────
TITLE, LAB, TICK, SMALL = 13, 12, 11, 10
NBINS = parse(Int, get(ENV, "NBINS", "10"))
fig = Figure(size=(FIGW, 0.42FIGW), figure_padding=(6, 10, 4, 6))
for (c, e) in enumerate(ensembles)
    g = J(out[e.tag].γ)
    q1, q3 = quantile(g, 0.25), quantile(g, 0.75); iqr = q3 - q1
    lo, hi = iqr == 0 ? (minimum(g), maximum(g)) :
             (max(q1 - 3iqr, minimum(g)), min(q3 + 3iqr, maximum(g)))
    off = count(<(lo), g) + count(>(hi), g)
    ax = Axis(fig[1, c]; xlabel="Surface energy (J/m²)", ylabel = c == 1 ? "count" : "",
              title="$(e.label)  (n = $(length(g)))", titlesize=TITLE, titlecolor=e.col,
              xlabelsize=LAB, ylabelsize=LAB, xticklabelsize=TICK, yticklabelsize=TICK,
              xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1)
    hist!(ax, filter(x -> lo <= x <= hi, g); bins=range(lo, hi; length=NBINS+1),
          color=(e.col, 0.65), strokecolor=:white, strokewidth=0.6)
    vlines!(ax, [centre_res[e.centre].γ * EV_PER_A2_TO_J_PER_M2];
            color=(e.col, 0.9), linestyle=:dash, linewidth=1.2)
    lo <= 0 <= hi && vlines!(ax, [0.0]; color=:black, linewidth=1.0)
    xlims!(ax, lo, hi)
    text!(ax, 0.04, 0.96; text=@sprintf("mean %.3f\nsd %.3f", mean(g), std(g)),
          space=:relative, align=(:left,:top), fontsize=SMALL, color=e.col)
    off == 0 || text!(ax, 0.96, 0.96; text="$off off scale", space=:relative,
                      align=(:right,:top), fontsize=SMALL, color=:gray40)
end
colgap!(fig.layout, 22)
save("$outdir/surface_energy_$(SURFACE).pdf", fig)
save("$outdir/surface_energy_$(SURFACE).png", fig; px_per_unit=4)

for e in ensembles
    r = out[e.tag]
    writedlm("$outdir/surface_energy_$(SURFACE)_$(e.tag).csv",
             hcat(r.ok, J(r.γ), J(r.γ_un), r.A), ',')
end
serialize("$outdir/surface_energy_$(SURFACE).jls",
          (; out, centre_γ = Dict(k => r.γ for (k, r) in centre_res),
             centre_γ_un = Dict(k => r.γ_un for (k, r) in centre_res),
             element, SURFACE, N_SUPER, VACUUM, NORMAL, rcut,
             sources = Dict(e.tag => e.csv for e in ensembles)))
println("\nfigure → $outdir/surface_energy_$(SURFACE).{pdf,png}")
println("data   → $outdir/surface_energy_$(SURFACE)_{unconstrained,constrained}.csv")
println("         columns: member, γ (J/m²), γ unrelaxed (J/m²), face area (Å²)")
