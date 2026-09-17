# qha_finiteT_Al_16_4_6A_3.jl
#
# Quasi-harmonic phonons at finite temperature for the mean model and a NAIVE POPS
# committee of Al_16_4_6A_3.  The question is not "is the model stable at its 0 K
# equilibrium" — that is what everything else in this repo already asks — but
#
#     does it stay stable once thermal expansion moves the lattice out to a(600 K)?
#
# ── WHY THIS IS THE INTERESTING TEST ────────────────────────────────────────
# The a_eq-only constraint programme imposes phonon stability at ONE volume.  A member
# can satisfy that and still go soft a few percent out, which is what the multi-volume
# work found under NPT: members leave FCC once the cell is free to expand.  QHA gets at
# the same thing without any MD — expand the lattice to the volume that minimises the
# free energy at 600 K and re-examine the spectrum.  Members that are stable at a(0)
# and unstable at a(600) are the failure mode the single-volume constraint misses.
#
# ── THE QHA ITSELF ──────────────────────────────────────────────────────────
# Helmholtz free energy per primitive cell, on a Gamma-centred MP grid:
#
#     F(a,T) = E_static(a) + (1/N_q) Σ_qν [ ħω_qν/2 + kT ln(1 − exp(−ħω_qν/kT)) ]
#
# minimised over a.  The first term is the zero-point energy, so a(0 K) from QHA is
# already slightly larger than the static minimum — that is physics, not an error.
#
# Gamma is dropped from the grid: the acoustic branches vanish there and the log term
# diverges.  Three modes out of 3·N_q is a vanishing fraction and the standard omission.
#
# F is UNDEFINED where any mode is imaginary — ln of a negative argument.  Such a member
# has no quasi-harmonic free energy at that volume, and is reported rather than being
# given a fabricated number.  For the naive committee that is expected to be common, and
# the count is itself a result.
#
# ── COST ────────────────────────────────────────────────────────────────────
# N_A force-constant builds per member on a 256-atom cell, plus one more at the
# minimising volume.  Default 5 volumes x (1 + N_MEMBERS) models.  A.csv (1.9 GB) is
# read ONCE, only to build the naive committee; the per-member work never touches it.
#
# Run:  julia --project -t <ncores> scripts/qoi/qha_finiteT_Al_16_4_6A_3.jl
#   T_K=600  N_MEMBERS=20  N_A=5  A_LO=0.995  A_HI=1.035  N_CELL=4  NQ=8
#   QOI_THREADS=<n>  FIGW=540

include(joinpath(@__DIR__, "..", "bandpath_phonon_uq", "lib.jl"))
using Random, Serialization
Random.seed!(1234)

element     = :Al
T_K         = parse(Float64, get(ENV, "T_K", "600"))
N_MEMBERS   = parse(Int,     get(ENV, "N_MEMBERS", "20"))
N_A         = parse(Int,     get(ENV, "N_A", "5"))        # lattice constants in the scan
A_LO        = parse(Float64, get(ENV, "A_LO", "0.995"))   # relative to the static minimum
A_HI        = parse(Float64, get(ENV, "A_HI", "1.035"))
N_CELL      = parse(Int,     get(ENV, "N_CELL", "4"))
NQ          = parse(Int,     get(ENV, "NQ", "8"))
LEV_PCT     = parse(Float64, get(ENV, "LEV_PCT", "0.5"))
UNSTABLE    = parse(Float64, get(ENV, "UNSTABLE", "-0.05"))   # THz
QOI_THREADS = parse(Int,     get(ENV, "QOI_THREADS", string(Threads.nthreads())))
FIGW        = parse(Float64, get(ENV, "FIGW", "540"))

MODELDIR = "models/Al_16_4_6A_3_"
outdir   = get(ENV, "OUTDIR", "$MODELDIR/results/qha_finiteT"); mkpath(outdir)

const ħ  = 1.054571817e-34
const kB = 1.380649e-23
const J_PER_EV = 1.602176634e-19

BLU = RGBf(0.0, 0.447, 0.698); RED = RGBf(0.80, 0.15, 0.15); GRY = RGBf(0.45,0.45,0.45)

result     = load_model(element, 16, 4, 6, 3; dataset_name="")
model      = result.model
lin_params = result.lin_params
n_params   = length(lin_params)
@printf("Model %s: %d params, %d threads.  T = %.0f K, %d volumes in [%.3f, %.3f]·a_static\n",
        result.name, n_params, Threads.nthreads(), T_K, N_A, A_LO, A_HI); flush(stdout)

# ── naive POPS committee (none exists for this model, so build it) ──────────
Ap = Diagonal(result.W) * result.A / result.P
Yw = result.W .* result.Y
println("building the naive POPS cloud (top $(Int(100LEV_PCT))% leverage, no constraints) …")
flush(stdout)
deltas = corrections(Ap, Yw, result.P; leverage_percentile=LEV_PCT)
@printf("  cloud: %d deltas × %d params\n", size(deltas)...)
heig, hb = hypercube(deltas)
@printf("  hypercube: %d directions retained\n", size(heig, 2)); flush(stdout)
mat, _ = sample_hypercube(heig, hb, lin_params; number_of_committee_members=N_MEMBERS)
members = [mat[:, i] for i in 1:size(mat, 2)]
# Persist it: regenerating means re-reading the 1.9 GB A.csv, and any downstream figure
# must use the SAME members, not a fresh draw that happens to share a seed.
writedlm("$outdir/samples_naive.csv", mat', ',')
@printf("  committee saved → %s/samples_naive.csv\n", outdir)
Ap = nothing; deltas = nothing; GC.gc()      # release before the phonon work

structure = AtomsBuilder.Chemistry.symmetry(element)
function mp_grid(L, n)
    B = 2π * inv(transpose(L))
    [B * [i/n, j/n, k/n] for i in 0:n-1, j in 0:n-1, k in 0:n-1
     if !(i == 0 && j == 0 && k == 0)]
end

"static energy per primitive cell and the phonon frequencies (THz) on the MP grid"
function static_and_freqs(m, a)
    sp, ss = bulk_prim_super(element; a=a, N_cell=N_CELL)
    E = ustrip(u"eV", ACEpotentials.potential_energy(sp, m)) / length(sp)  # per cell (Np=1)
    fc = precompute_force_constants(sp, ss, m)
    qs = mp_grid(fc.L, NQ)
    F = Matrix{Float64}(undef, 3fc.Np, length(qs))
    for (iq, q) in enumerate(qs)
        ev = eigvals(Hermitian(dynamical_matrix_from_fc(fc, q)))
        F[:, iq] = sign.(ev) .* sqrt.(abs.(ev)) .* FREQ_THz
    end
    return E, F
end

"quasi-harmonic free energy per primitive cell (eV); NaN if any mode is imaginary"
function F_qha(E_static, Fthz, T)
    any(Fthz .<= 0) && return NaN
    ω = Fthz .* 2π .* 1e12
    zp = sum(0.5 .* ħ .* ω) / length(ω) / J_PER_EV
    th = T <= 0 ? 0.0 :
         sum(kB*T .* log.(1 .- exp.(-ħ .* ω ./ (kB*T)))) / length(ω) / J_PER_EV
    return E_static + 3*(zp + th)      # 3 branches averaged above, so restore the count
end

# ── one model: scan volumes, minimise F(a,T), re-examine the spectrum there ──
function qha(m, θ)
    ACEpotentials.Models.set_linear_parameters!(m, θ)
    a_static = ACEWorkflow.relax_lattice_constant(m, element)
    as = collect(range(A_LO*a_static, A_HI*a_static; length=N_A))
    Es = Float64[]; Fs = Vector{Matrix{Float64}}()
    for a in as
        E, Fq = static_and_freqs(m, a); push!(Es, E); push!(Fs, Fq)
    end
    minω = [minimum(F) for F in Fs]
    Fh   = [F_qha(Es[i], Fs[i], T_K) for i in eachindex(as)]

    ok = findall(!isnan, Fh)
    a_T = NaN; minω_T = NaN
    if length(ok) >= 3
        # parabola through the three lowest-F points; QHA minima are shallow and smooth
        p = sortperm(Fh[ok])[1:3]; idx = ok[p]
        A = hcat(as[idx].^2, as[idx], ones(3))
        c = A \ Fh[idx]
        if c[1] > 0
            a_T = -c[2] / (2c[1])
            (minimum(as) <= a_T <= maximum(as)) || (a_T = NaN)
        end
    end
    if !isnan(a_T)
        _, F_at = static_and_freqs(m, a_T)
        minω_T = minimum(F_at)
    end
    return (; a_static, as, Es, minω, Fh, a_T, minω_T,
              minω_static = minω[argmin(abs.(as .- a_static))],
              n_bad_vol = count(isnan, Fh))
end

function qha_many(θs; label="")
    n = length(θs); out = Vector{Any}(undef, n)
    errs = Vector{Union{Nothing,String}}(nothing, n)
    nt = clamp(QOI_THREADS, 1, min(Threads.nthreads(), n))
    done = Threads.Atomic{Int}(0)
    @printf("\n[%s] %d models × %d volumes, %d tasks …\n", label, n, N_A, nt); flush(stdout)
    t = @elapsed @sync for k in 1:nt
        Threads.@spawn begin
            m = deepcopy(model)                      # private: set_linear_parameters! mutates
            for i in k:nt:n
                try
                    out[i] = qha(m, θs[i])
                catch e
                    errs[i] = sprint(showerror, e); out[i] = nothing
                end
                d = Threads.atomic_add!(done, 1) + 1
                d % max(1, n ÷ 10) == 0 && (@printf("  [%s] %d/%d\n", label, d, n); flush(stdout))
            end
        end
    end
    @printf("[%s] done in %.1f min (%.1f s/model)\n", label, t/60, t/n); flush(stdout)
    return out, errs
end

# ── mean model ──────────────────────────────────────────────────────────────
println("\n── mean model ──"); flush(stdout)
mr = qha(deepcopy(model), lin_params)
@printf("  a_static = %.5f Å\n", mr.a_static)
for i in eachindex(mr.as)
    @printf("    a = %.5f  E = %+.6f eV  min ω = %+.4f THz  F(%.0fK) = %s\n",
            mr.as[i], mr.Es[i], mr.minω[i], T_K,
            isnan(mr.Fh[i]) ? "undefined (imaginary modes)" : @sprintf("%+.6f eV", mr.Fh[i]))
end
if isnan(mr.a_T)
    @warn "QHA minimum not bracketed for the mean model — widen [A_LO, A_HI] or check stability"
else
    @printf("  a(%.0f K) = %.5f Å  → linear expansion %.3f%%,  min ω there = %+.4f THz  %s\n",
            T_K, mr.a_T, 100*(mr.a_T/mr.a_static - 1), mr.minω_T,
            mr.minω_T < UNSTABLE ? "← UNSTABLE at temperature" : "stable")
end
flush(stdout)

# ── committee ───────────────────────────────────────────────────────────────
res, errs = qha_many(members; label="naive")
ok = findall(!isnothing, res); bad = findall(isnothing, res)
isempty(bad) || @printf("\n%d/%d members errored and are excluded\n", length(bad), length(res))
isempty(ok) && error("no member completed")

a0   = [res[i].a_static  for i in ok]
aT   = [res[i].a_T       for i in ok]
w0   = [res[i].minω_static for i in ok]
wT   = [res[i].minω_T    for i in ok]
nbad = [res[i].n_bad_vol for i in ok]
have = findall(!isnan, aT)

println("\n══ QHA AT $(Int(T_K)) K — Al_16_4_6A_3, naive POPS ═══════════════════")
@printf("members with a well-defined QHA minimum: %d/%d\n", length(have), length(ok))
@printf("  (the rest have imaginary modes somewhere in the volume scan, so F is undefined;\n")
@printf("   median %d of %d volumes bad, max %d)\n", median(nbad), N_A, maximum(nbad))
if !isempty(have)
    ex = 100 .* (aT[have] ./ a0[have] .- 1)
    @printf("\nlinear expansion a(%.0fK)/a(0) − 1: median %.3f%%, range [%.3f%%, %.3f%%]\n",
            T_K, median(ex), minimum(ex), maximum(ex))
    # Members WITHOUT a QHA minimum must be counted too.  F is undefined for them
    # because a mode went imaginary somewhere in the volume scan — if they were stable
    # cold, that IS a failure on expansion, just one that leaves no a(T) to report.
    # Counting only the members that kept a minimum silently drops exactly the cases
    # this script exists to find.
    nomin  = findall(isnan, aT)
    s0     = w0[have] .>= UNSTABLE
    sT     = wT[have] .>= UNSTABLE
    lost   = count(s0 .& .!sT)                       # explicit: soft at a(T)
    n_cold_nomin = count(>=(UNSTABLE), w0[nomin])    # stable cold, went imaginary on the way
    tot_cold = count(s0) + n_cold_nomin
    fail     = lost + n_cold_nomin
    @printf("\nstability, ALL %d members:\n", length(ok))
    @printf("  stable at a(0)                        : %d  (%d keep a QHA minimum, %d lose it)\n",
            tot_cold, count(s0), n_cold_nomin)
    @printf("  of those, still stable at a(%.0f K)      : %d\n", T_K, count(sT))
    @printf("  FAIL ON HEATING                       : %d / %d  (%.0f%%)\n",
            fail, tot_cold, 100fail/max(tot_cold,1))
    @printf("      soft at a(T)                      : %d\n", lost)
    @printf("      imaginary during the scan, no a(T): %d\n", n_cold_nomin)
    @printf("  soft already at a(0)                  : %d\n",
            count(.!s0) + count(<(UNSTABLE), w0[nomin]))
    println("  ↑ the second failure route leaves no a(T) to report, so counting only")
    println("    members that kept a QHA minimum undercounts the effect badly")
    @printf("\nmean model: min ω %+.4f → %+.4f THz on heating\n", mr.minω_static, mr.minω_T)
end
flush(stdout)

# ── figure ──────────────────────────────────────────────────────────────────
TITLE, LAB, TICK = 13, 12, 11
fig = Figure(size=(FIGW, 0.44FIGW), figure_padding=(6, 10, 4, 6))
ax = Axis(fig[1, 1]; xlabel="min ω at a(0) (THz)", ylabel="min ω at a($(Int(T_K)) K) (THz)",
          title="Cold vs hot stability", titlesize=TITLE,
          xlabelsize=LAB, ylabelsize=LAB, xticklabelsize=TICK, yticklabelsize=TICK,
          xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1)
if !isempty(have)
    lo = min(minimum(w0[have]), minimum(wT[have])) - 0.3
    hi = max(maximum(w0[have]), maximum(wT[have])) + 0.3
    lines!(ax, [lo, hi], [lo, hi]; color=(:black, 0.4), linestyle=:dash, linewidth=1.0)
    hlines!(ax, [UNSTABLE]; color=(RED, 0.6), linewidth=1.0)
    vlines!(ax, [UNSTABLE]; color=(RED, 0.6), linewidth=1.0)
    lost = (w0[have] .>= UNSTABLE) .& (wT[have] .< UNSTABLE)
    scatter!(ax, w0[have][.!lost], wT[have][.!lost]; color=(GRY, 0.8), markersize=9)
    scatter!(ax, w0[have][lost],   wT[have][lost];   color=RED, markersize=10)
    scatter!(ax, [mr.minω_static], [mr.minω_T]; color=BLU, markersize=13, marker=:diamond)
    xlims!(ax, lo, hi); ylims!(ax, lo, hi)
end
ax2 = Axis(fig[1, 2]; xlabel="linear expansion to $(Int(T_K)) K (%)", ylabel="count",
           title="QHA thermal expansion", titlesize=TITLE,
           xlabelsize=LAB, ylabelsize=LAB, xticklabelsize=TICK, yticklabelsize=TICK,
           xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1)
if !isempty(have)
    hist!(ax2, 100 .* (aT[have] ./ a0[have] .- 1); bins=12,
          color=(BLU, 0.6), strokecolor=:white, strokewidth=0.6)
    isnan(mr.a_T) || vlines!(ax2, [100*(mr.a_T/mr.a_static - 1)];
                             color=BLU, linestyle=:dash, linewidth=1.6)
end
colgap!(fig.layout, 22)
save("$outdir/qha_$(Int(T_K))K.pdf", fig)
save("$outdir/qha_$(Int(T_K))K.png", fig; px_per_unit=4)

writedlm("$outdir/qha_$(Int(T_K))K.csv", hcat(ok, a0, aT, w0, wT, nbad), ',')
serialize("$outdir/qha_$(Int(T_K))K.jls",
          (; mean = mr, res, ok, bad, a0, aT, w0, wT, nbad, members,
             T_K, N_A, A_LO, A_HI, N_CELL, NQ, N_MEMBERS, UNSTABLE, seed = 1234))
println("\nfigure → $outdir/qha_$(Int(T_K))K.{pdf,png}")
println("data   → $outdir/qha_$(Int(T_K))K.csv  (member, a0, a(T), min ω cold, min ω hot, bad volumes)")
