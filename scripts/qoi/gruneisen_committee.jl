# gruneisen_committee.jl
#
# Grüneisen parameter across the unconstrained and constrained POPS committees, with
# every member relaxed to its OWN equilibrium and its own phonons rebuilt at three
# volumes about it.
#
#   mode:           γ_qν = −∂ln ω_qν / ∂ln V
#   thermodynamic:  γ(T) = Σ C_v(ω_qν,T) γ_qν / Σ C_v(ω_qν,T)
#
# with C_v the Einstein heat capacity of each mode.  Reference: Al is γ ≈ 2.1–2.2.
#
# ── WHY THIS ONE MATTERS ────────────────────────────────────────────────────
# γ is the bridge between the phonon constraints and the thermal expansion result:
# α = γ C_v / (3 B V).  The predicates constrain phonon STABILITY (min ω ≥ cut), never
# the volume derivative of ω, so γ is again an observable nobody constrained — but
# unlike the surface energy it is built from exactly the quantity the prior does act on.
# If constraining stabilises the spectrum without pinning its volume dependence, γ is
# where that shows up.
#
# ── GEOMETRY, PER MEMBER ────────────────────────────────────────────────────
# γ is a derivative about the member's OWN equilibrium.  Evaluating every member on one
# shared set of volumes centred on the mean model's a_eq would differentiate at the
# wrong reference AND under residual stress, which is the same mistake the frozen-
# geometry vacancy estimate made.  So each member is relaxed first, then three native
# force-constant builds at a·(1−δ), a, a·(1+δ).
#
# ── A PROPER q-GRID, NOT A BAND PATH ────────────────────────────────────────
# scripts/phonons/gruneisen_phonon_bands_ace.jl averages over a high-symmetry band path.
# A path is a 1-D cut through the zone and is not a Brillouin-zone sample, so the C_v
# weighting is over the wrong measure.  Here the average is over a Γ-centred NQ³
# Monkhorst-Pack grid with Γ itself dropped (the acoustic branches vanish there).
# Modes are also dropped where ω is below OMEGA_MIN at ANY of the three volumes: γ is a
# log derivative, so a mode approaching zero makes it diverge, and an imaginary mode
# makes it meaningless.  The number dropped is reported per member.
#
# ── PARALLELISM ─────────────────────────────────────────────────────────────
# Three force-constant builds per member, each task with its OWN deep copy of the model
# — set_linear_parameters! mutates.  Chunked @spawn, not @threads with threadid().
# Failures are caught per member, counted and excluded.
#
# Run:  julia --project -t <ncores> scripts/qoi/gruneisen_committee.jl [unconstrained.csv constrained.csv]
#   ELEMENT=Al  N_CELL=4  NQ=8  DELTA=0.01  T_K=300  OMEGA_MIN=0.1
#   QOI_THREADS=<n>  MODELDIR=models/Al_12_4_6A_2_  FIGW=540  NBINS=10

include(joinpath(@__DIR__, "..", "bandpath_phonon_uq", "lib.jl"))
using Serialization

element     = Symbol(get(ENV, "ELEMENT", "Al"))
N_CELL      = parse(Int,     get(ENV, "N_CELL", "4"))
NQ          = parse(Int,     get(ENV, "NQ", "8"))        # NQ³ Monkhorst-Pack grid
DELTA       = parse(Float64, get(ENV, "DELTA", "0.01"))  # ±1% in the lattice constant
T_K         = parse(Float64, get(ENV, "T_K", "300"))
OMEGA_MIN   = parse(Float64, get(ENV, "OMEGA_MIN", "0.1"))   # THz
QOI_THREADS = parse(Int,     get(ENV, "QOI_THREADS", string(Threads.nthreads())))
FIGW        = parse(Float64, get(ENV, "FIGW", "540"))
NBINS       = parse(Int,     get(ENV, "NBINS", "10"))
MODELDIR    = get(ENV, "MODELDIR", "models/Al_12_4_6A_2_")
RES         = "$MODELDIR/results"
outdir      = get(ENV, "OUTDIR", "$RES/gruneisen"); mkpath(outdir)

const ħ  = 1.054571817e-34
const kB = 1.380649e-23
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
@printf("NQ = %d (%d q-points, Γ dropped), δ = %.3f, T = %.0f K, ω_min = %.2f THz\n",
        NQ, NQ^3 - 1, DELTA, T_K, OMEGA_MIN)

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

# ── Γ-centred Monkhorst-Pack grid, Γ removed ────────────────────────────────
# B = 2π L^{-T}, matching src/Phonons/phonon_bands.jl.  Fractional coordinates are the
# same at every volume, so the grid corresponds point-for-point across the three builds.
function mp_grid(L, n)
    B = 2π * inv(transpose(L))
    qs = Vector{Vector{Float64}}()
    for i in 0:n-1, j in 0:n-1, k in 0:n-1
        (i == 0 && j == 0 && k == 0) && continue
        push!(qs, B * [i/n, j/n, k/n])
    end
    return qs
end

"frequencies (THz) on the grid for the CURRENT parameters of `m` at lattice constant a"
function grid_freqs(m, a)
    sp, ss = bulk_prim_super(element; a=a, N_cell=N_CELL)
    fc = precompute_force_constants(sp, ss, m)
    qs = mp_grid(fc.L, NQ)
    Np = fc.Np
    F  = Matrix{Float64}(undef, 3Np, length(qs))
    for (iq, q) in enumerate(qs)
        ev = eigvals(Hermitian(dynamical_matrix_from_fc(fc, q)))
        F[:, iq] = sign.(ev) .* sqrt.(abs.(ev)) .* FREQ_THz
    end
    return F
end

# ── one member ──────────────────────────────────────────────────────────────
# `m` is a per-task model copy.  Never pass the shared one.
function gruneisen(m, θ)
    ACEpotentials.Models.set_linear_parameters!(m, θ)
    a0 = ACEWorkflow.relax_lattice_constant(m, element)
    am, ap = a0*(1-DELTA), a0*(1+DELTA)

    F0 = grid_freqs(m, a0); Fm = grid_freqs(m, am); Fp = grid_freqs(m, ap)
    size(F0) == size(Fm) == size(Fp) || error("grid mismatch across volumes")

    # a mode counts only if it is real and above ω_min at ALL THREE volumes: γ is a log
    # derivative, so a vanishing or imaginary ω makes it divergent or meaningless
    good = (F0 .> OMEGA_MIN) .& (Fm .> OMEGA_MIN) .& (Fp .> OMEGA_MIN)
    n_drop = count(.!good)

    dlnV = 3*(log(ap) - log(am))                       # V ∝ a³
    γ_mode = .-(log.(abs.(Fp)) .- log.(abs.(Fm))) ./ dlnV

    ω  = F0 .* 2π .* 1e12                              # rad/s
    x  = ħ .* ω ./ (kB * T_K)
    ex = exp.(x)
    Cv = kB .* x.^2 .* ex ./ (ex .- 1).^2

    γT = sum(Cv[good] .* γ_mode[good]) / sum(Cv[good])
    return (; γT, a0, n_drop, n_modes = length(F0),
              n_imag = count(F0 .<= 0),
              γ_mode_median = median(γ_mode[good]),
              minω = minimum(F0))
end

function gruneisen_many(θs; label="")
    n = length(θs)
    out  = Vector{Any}(undef, n)
    errs = Vector{Union{Nothing,String}}(nothing, n)
    nt = clamp(QOI_THREADS, 1, min(Threads.nthreads(), n))
    done = Threads.Atomic{Int}(0)
    @printf("\n[%s] %d members × 3 force-constant builds, %d tasks, %d deep …\n",
            label, n, nt, cld(n, nt)); flush(stdout)
    t = @elapsed @sync for k in 1:nt
        Threads.@spawn begin
            m = deepcopy(model)                       # private to this task
            for i in k:nt:n
                try
                    out[i] = gruneisen(m, θs[i])
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
    r = gruneisen(deepcopy(model), θc); centre_res[k] = r
    @printf("  %-11s γ(%.0f K) = %+.4f   a = %.5f Å, min ω %+.3f THz, %d/%d modes dropped\n",
            k, T_K, r.γT, r.a0, r.minω, r.n_drop, r.n_modes)
end
println("  reference: Al γ ≈ 2.1–2.2")
flush(stdout)

# ── committees ──────────────────────────────────────────────────────────────
out = Dict{String,Any}()
for e in ensembles
    mem = read_committee(e.csv)
    res, errs = gruneisen_many(mem; label=e.tag)
    ok = findall(!isnothing, res); bad = findall(isnothing, res)
    if !isempty(bad)
        @printf("[%s] %d/%d FAILED, excluded:\n", e.tag, length(bad), length(res))
        for i in bad[1:min(3,end)]; @printf("    member %d: %s\n", i, first(split(errs[i], '\n'))); end
    end
    isempty(ok) && error("[$(e.tag)] no member succeeded")
    out[e.tag] = (; ok, bad, centre = e.centre,
                  γ      = [res[i].γT     for i in ok],
                  a0     = [res[i].a0     for i in ok],
                  n_drop = [res[i].n_drop for i in ok],
                  n_imag = [res[i].n_imag for i in ok],
                  minω   = [res[i].minω   for i in ok])
end

# ── results ─────────────────────────────────────────────────────────────────
println("\n══ GRÜNEISEN PARAMETER at $(Int(T_K)) K ══════════════════════════════")
for e in ensembles
    @printf("%-14s central model (%s): γ = %+.4f\n",
            e.tag, e.centre, centre_res[e.centre].γT)
end
println()
@printf("%-14s %6s %10s %10s %10s %10s\n", "ensemble", "n", "mean", "sd", "min", "max")
for e in ensembles
    g = out[e.tag].γ
    @printf("%-14s %6d %10.4f %10.4f %10.4f %10.4f\n",
            e.tag, length(g), mean(g), std(g), minimum(g), maximum(g))
end
let u = out["unconstrained"], c = out["constrained"]
    @printf("\nspread ratio  σ_constrained / σ_unconstrained = %.3f\n", std(c.γ)/std(u.γ))
    @printf("range  ratio  = %.3f\n",
            (maximum(c.γ)-minimum(c.γ)) / (maximum(u.γ)-minimum(u.γ)))
end

println("\n── how much of the spectrum had to be discarded ──")
println("  (modes below ω_min or imaginary at any of the three volumes — γ is a log")
println("   derivative, so those are divergent or meaningless, not merely noisy)")
for e in ensembles
    r = out[e.tag]; f = 100 .* r.n_drop ./ centre_res[r.centre].n_modes
    @printf("%-14s dropped %.1f%% median, up to %.1f%%;  %d/%d members with imaginary modes\n",
            e.tag, median(f), maximum(f), count(>(0), r.n_imag), length(r.n_imag))
end
println("\n── relaxed lattice constant ──")
for e in ensembles
    r = out[e.tag]
    @printf("%-14s a ∈ [%.5f, %.5f] Å\n", e.tag, minimum(r.a0), maximum(r.a0))
end
flush(stdout)

# ── figure ──────────────────────────────────────────────────────────────────
TITLE, LAB, TICK, SMALL = 13, 12, 11, 10
fig = Figure(size=(FIGW, 0.42FIGW), figure_padding=(6, 10, 4, 6))
for (c, e) in enumerate(ensembles)
    g = out[e.tag].γ
    q1, q3 = quantile(g, 0.25), quantile(g, 0.75); iqr = q3 - q1
    lo, hi = iqr == 0 ? (minimum(g), maximum(g)) :
             (max(q1 - 3iqr, minimum(g)), min(q3 + 3iqr, maximum(g)))
    off = count(<(lo), g) + count(>(hi), g)
    ax = Axis(fig[1, c]; xlabel="Grüneisen parameter γ($(Int(T_K)) K)",
              ylabel = c == 1 ? "count" : "",
              title="$(e.label)  (n = $(length(g)))", titlesize=TITLE, titlecolor=e.col,
              xlabelsize=LAB, ylabelsize=LAB, xticklabelsize=TICK, yticklabelsize=TICK,
              xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1)
    hist!(ax, filter(x -> lo <= x <= hi, g); bins=range(lo, hi; length=NBINS+1),
          color=(e.col, 0.65), strokecolor=:white, strokewidth=0.6)
    vlines!(ax, [centre_res[e.centre].γT]; color=(e.col, 0.9), linestyle=:dash, linewidth=1.2)
    lo <= 2.15 <= hi && vlines!(ax, [2.15]; color=:black, linewidth=1.0)   # Al reference
    xlims!(ax, lo, hi)
    text!(ax, 0.04, 0.96; text=@sprintf("mean %.3f\nsd %.3f", mean(g), std(g)),
          space=:relative, align=(:left,:top), fontsize=SMALL, color=e.col)
    off == 0 || text!(ax, 0.96, 0.96; text="$off off scale", space=:relative,
                      align=(:right,:top), fontsize=SMALL, color=:gray40)
end
colgap!(fig.layout, 22)
save("$outdir/gruneisen.pdf", fig)
save("$outdir/gruneisen.png", fig; px_per_unit=4)

for e in ensembles
    r = out[e.tag]
    writedlm("$outdir/gruneisen_$(e.tag).csv", hcat(r.ok, r.γ, r.a0, r.n_drop, r.minω), ',')
end
serialize("$outdir/gruneisen.jls",
          (; out, centre = Dict(k => (; r.γT, r.a0, r.n_drop, r.n_modes, r.minω)
                                for (k, r) in centre_res),
             element, N_CELL, NQ, DELTA, T_K, OMEGA_MIN,
             sources = Dict(e.tag => e.csv for e in ensembles)))
println("\nfigure → $outdir/gruneisen.{pdf,png}")
println("data   → $outdir/gruneisen_{unconstrained,constrained}.csv")
println("         columns: member, γ, relaxed a, modes dropped, min ω")
