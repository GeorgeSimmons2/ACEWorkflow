# stratified_pops_Al_16_4_6A_3.jl
#
# Stratified sampling of the PINNED POPS corrections for Al_16_4_6A_3, stratified by the
# size of the correction itself:
#
#   1. pin every correction to a_eq  (b′·δ = 0, closed form)
#   2. ‖δ‖₂ for each, in θ-space
#   3. sort, split into N_BINS strata, draw N_PER_BIN from each
#   4. phonons on the PREMADE undotted Hessian — valid precisely because they are pinned
#   5. parity and calibration, pooled and per stratum
#
# ── WHY THIS IS A DIFFERENT OBJECT FROM THE HYPERCUBE COMMITTEES ────────────
# Members here are individual corrections, θ_i = lin_params + δ_i — the same objects the
# July "naive committee" plotted, not draws from a box fitted around them.  That
# distinction already explained one puzzle: raw corrections are mild, hypercube samples
# reach into the box corners and are wild.  Stratifying by ‖δ‖ makes the transition
# explicit: if instability and miscalibration are driven by correction magnitude, the
# strata will show it monotonically.
#
# ── WHY THE PREMADE HESSIAN IS LEGITIMATE HERE ──────────────────────────────
# Every correction is pinned, so every member's equilibrium is a_eq and Σ_k θ_k D_k(q)
# is the EXACT operator for all of them.  One cached band path serves all
# N_BINS × N_PER_BIN members at ~ms each.  Without the pin this shortcut would be
# evaluating members at someone else's geometry.  The pin is checked, not assumed.
#
# ── BINNING ─────────────────────────────────────────────────────────────────
# "ordered by norm then split into bins uniformly" is ambiguous, so both are available:
#   BINNING=quantile  (default) equal COUNT per stratum — deciles of the sorted list
#   BINNING=width               equal WIDTH in ‖δ‖
# Quantile is the default because it puts equal sampling weight on each stratum; with
# equal width the top strata may hold only a handful of corrections and cannot supply
# N_PER_BIN without replacement.  Either way, a stratum with fewer than N_PER_BIN
# members is taken whole and the shortfall reported rather than silently padded.
#
# Run:  julia --project -t 40 scripts/uq/stratified_pops_Al_16_4_6A_3.jl
#   N_BINS=10  N_PER_BIN=50  BINNING=quantile  STRIDE=40  LEV_PCT=0.5
#   PIN_TOL=1e-6  UNSTABLE=-0.05  FIGW=540

include(joinpath(@__DIR__, "..", "bandpath_phonon_uq", "lib.jl"))
include(joinpath(@__DIR__, "lib_parity_calibration.jl"))
using Random, Serialization
Random.seed!(1234)

element    = :Al
N_BINS     = parse(Int,     get(ENV, "N_BINS", "10"))
N_PER_BIN  = parse(Int,     get(ENV, "N_PER_BIN", "50"))
BINNING    = get(ENV, "BINNING", "quantile")
STRIDE     = parse(Int,     get(ENV, "STRIDE", "40"))
LEV_PCT    = parse(Float64, get(ENV, "LEV_PCT", "0.5"))
PIN_TOL    = parse(Float64, get(ENV, "PIN_TOL", "1e-6"))
UNSTABLE   = parse(Float64, get(ENV, "UNSTABLE", "-0.05"))
qΓtol      = 5e-2
FIGW       = parse(Float64, get(ENV, "FIGW", "540"))
BINNING in ("quantile", "width") || error("BINNING must be quantile or width")

MODELDIR = "models/Al_16_4_6A_3_"
RES      = "$MODELDIR/results"
outdir   = get(ENV, "OUTDIR", "$RES/stratified_pops"); mkpath(outdir)
BPCACHE  = get(ENV, "BPCACHE", "$RES/pinned_ensembles/bandpath_4x4x4_aref.jls")
TEST     = abspath(joinpath(@__DIR__, "..", "..", "data", "Al", "manual_df_test_Al.xyz"))

result = load_model(element, 16, 4, 6, 3; dataset_name="")
model, lin_params = result.model, result.lin_params
n_params = length(lin_params)
P, W, Y  = result.P, result.W, result.Y
Ap = Diagonal(W) * result.A / P
Yw = W .* Y
N  = size(Ap, 1)
@printf("Model %s: %d params, %d observations, %d threads\n",
        result.name, n_params, N, Threads.nthreads()); flush(stdout)

# ── pin every correction (closed form, exactly as the ensembles script) ─────
a_eq = ACEWorkflow.relax_lattice_constant(model, element)
lattice_basis(a) = ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(
                       ACEWorkflow.Elasticity.reference_system(element; a=a), model))
b_prime = ForwardDiff.derivative(lattice_basis, a_eq)
λ  = 1.0 / N
C  = Symmetric(P'*P .* λ .+ Ap'*Ap); Cf = cholesky(C)
Ainv = Cf \ Matrix(Ap')
x̃    = Cf \ (Ap' * Yw)
lev  = vec(sum(Ap' .* Ainv; dims=1))
g = transpose(P) \ b_prime
u = Cf \ g; gu = dot(g, u); v = Ap * u
x̃_c   = x̃ .- u .* (dot(g, x̃)/gu)
lev_c = lev .- (v .^ 2) ./ gu
scale = (Yw .- Ap * x̃_c) ./ lev_c
PCx = transpose(Ainv .* transpose(scale))
PCx .-= (scale .* v ./ gu) * transpose(u)
Ainv = nothing; GC.gc()
@printf("PIN CHECK (forest): max |b′·δ| = %.3e over %d corrections\n",
        maximum(abs.(PCx * g)), N); flush(stdout)

keep = lev_c .>= quantile(lev_c, 1 - LEV_PCT)
@printf("cloud: %d of %d corrections (top %.0f%% constrained leverage)\n",
        count(keep), N, 100LEV_PCT)
idx_all = findall(keep)
Δθ = PCx[keep, :] / transpose(P)          # θ-space corrections, one per row
PCx = nothing; GC.gc()

# ── ‖δ‖₂ in θ-SPACE: the perturbation to the physical coefficients, which is what
#    "size of the correction" should mean.  x-space norms are preconditioner artefacts.
nrm = vec(sqrt.(sum(abs2, Δθ; dims=2)))
@printf("‖δ‖₂: min %.4e, median %.4e, max %.4e, ratio max/min %.3g\n",
        minimum(nrm), median(nrm), maximum(nrm), maximum(nrm)/minimum(nrm)); flush(stdout)

# ── strata ──────────────────────────────────────────────────────────────────
edges = BINNING == "quantile" ?
        [quantile(nrm, b/N_BINS) for b in 0:N_BINS] :
        collect(range(minimum(nrm), maximum(nrm); length=N_BINS+1))
edges[1] -= 1e-12; edges[end] += 1e-12
bin_of = [searchsortedlast(edges, x) for x in nrm]
bin_of = clamp.(bin_of, 1, N_BINS)

# a function, not a bare loop: at top level a `for` opens a soft scope, so `shortfall +=`
# would create a fresh local and reading it afterwards throws UndefVarError
function draw_strata(bin_of, edges)
    sel_local = Int[]; sel_bin = Int[]; shortfall = 0
    for b in 1:N_BINS
        pool = findall(==(b), bin_of)
        take = min(N_PER_BIN, length(pool))
        take < N_PER_BIN && (shortfall += N_PER_BIN - take)
        chosen = length(pool) <= take ? pool : pool[randperm(length(pool))[1:take]]
        append!(sel_local, chosen); append!(sel_bin, fill(b, take))
        @printf("  stratum %2d: ‖δ‖ ∈ [%.4e, %.4e], %7d available, %d taken\n",
                b, edges[b], edges[b+1], length(pool), take)
    end
    return sel_local, sel_bin, shortfall
end
sel_local, sel_bin, shortfall = draw_strata(bin_of, edges)
shortfall == 0 || @warn "$shortfall members short of the requested $(N_BINS*N_PER_BIN)"
n_sel = length(sel_local)
members = [lin_params .+ Δθ[i, :] for i in sel_local]
sel_norm = nrm[sel_local]
sel_obs  = idx_all[sel_local]
@printf("\n%d members selected across %d strata\n", n_sel, N_BINS); flush(stdout)

da = [-dot(b_prime, θ .- lin_params) /
       dot(ForwardDiff.derivative(a -> ForwardDiff.derivative(lattice_basis, a), a_eq), θ)
      for θ in members[1:min(50, n_sel)]]
@printf("PIN CHECK (members): max |Δa| = %.3e Å over a sample of %d (tol %.0e)\n",
        maximum(abs.(da)), length(da), PIN_TOL)
maximum(abs.(da)) < PIN_TOL ||
    error("members are not pinned — the premade Hessian is the wrong operator for them")

# ── phonons on the PREMADE undotted band path ───────────────────────────────
isfile(BPCACHE) || error("""
    missing $BPCACHE
    Run scripts/uq/pinned_rejection_ensembles_Al_16_4_6A_3.jl first — it builds and
    caches the undotted band path this study reuses.""")
bp = deserialize(BPCACHE)
abs(bp.a_ref - a_eq) < 1e-8 ||
    @warn "cached band path is at a = $(bp.a_ref) but a_eq = $a_eq"
keep_q = findall(bp.qnorm .>= qΓtol)
@printf("band path: %d q-points (%d after the near-Γ cut) at a = %.6f Å\n",
        length(bp.Bq), length(keep_q), bp.a_ref); flush(stdout)

minω = Vector{Float64}(undef, n_sel)
t = @elapsed Threads.@threads for i in 1:n_sel
    θ = members[i]; mw = Inf
    for iq in keep_q
        ev = eigvals(Hermitian(reshape(bp.Bq[iq]*θ, 3bp.Np, 3bp.Np)))
        mw = min(mw, minimum(sign.(ev).*sqrt.(abs.(ev)).*FREQ_THz))
    end
    minω[i] = mw
end
minω_mean = let mw = Inf
    for iq in keep_q
        ev = eigvals(Hermitian(reshape(bp.Bq[iq]*lin_params, 3bp.Np, 3bp.Np)))
        mw = min(mw, minimum(sign.(ev).*sqrt.(abs.(ev)).*FREQ_THz))
    end; mw
end
@printf("phonons: %d members in %.1f s (%.2f ms each); mean model min ω = %+.4f THz\n",
        n_sel, t, 1000t/n_sel, minω_mean); flush(stdout)

# ── parity and calibration: pooled, then per stratum ────────────────────────
@printf("\n── test-set predictions (stride %d, per atom) ──\n", STRIDE); flush(stdout)
t = @elapsed pr_all = committee_predictions_fast(model, members, TEST; stride=STRIDE,
                                            point_params=lin_params, per_atom=true)
@printf("pooled: %d configs  [%.1f s]\n", pr_all.n, t); flush(stdout)
parity_figure(pr_all, "stratified POPS corrections — all $n_sel members",
              "pooled", outdir; FIGW=FIGW)
calibration_figure(pr_all, "stratified POPS corrections — all $n_sel members",
                   "pooled", outdir; FIGW=FIGW)

pr_bin = Vector{Any}(undef, N_BINS)
for b in 1:N_BINS
    mb = members[findall(==(b), sel_bin)]
    pr_bin[b] = committee_predictions_fast(model, mb, TEST; stride=STRIDE,
                                      point_params=lin_params, per_atom=true)
    @printf("  stratum %2d: E cov %.1f%%, F cov %.1f%%\n", b,
            pc_cover(pr_bin[b].tE, pr_bin[b].loE, pr_bin[b].hiE),
            pc_cover(pr_bin[b].tF, pr_bin[b].loF, pr_bin[b].hiF)); flush(stdout)
end

# ── summary ─────────────────────────────────────────────────────────────────
println("\n══ STRATA ════════════════════════════════════════════════════════")
@printf("%4s %12s %12s %10s %10s %9s %9s\n",
        "bin", "median ‖δ‖", "median min ω", "min min ω", "% unstable", "E cov %", "F cov %")
rows = []
for b in 1:N_BINS
    m = findall(==(b), sel_bin)
    ec = pc_cover(pr_bin[b].tE, pr_bin[b].loE, pr_bin[b].hiE)
    fc = pc_cover(pr_bin[b].tF, pr_bin[b].loF, pr_bin[b].hiF)
    pu = 100count(<(UNSTABLE), minω[m]) / length(m)
    @printf("%4d %12.4e %12.4f %10.4f %10.1f %9.1f %9.1f\n",
            b, median(sel_norm[m]), median(minω[m]), minimum(minω[m]), pu, ec, fc)
    push!(rows, (b, median(sel_norm[m]), median(minω[m]), minimum(minω[m]), pu, ec, fc))
end
@printf("\npooled: %d/%d unstable (%.1f%%), E cov %.1f%%, F cov %.1f%%, E RMSE %.4g meV/atom\n",
        count(<(UNSTABLE), minω), n_sel, 100count(<(UNSTABLE), minω)/n_sel,
        pc_cover(pr_all.tE, pr_all.loE, pr_all.hiE),
        pc_cover(pr_all.tF, pr_all.loF, pr_all.hiF),
        1000*pc_rmse(pr_all.pE, pr_all.tE))
println("\nThe question this answers: does correction MAGNITUDE predict instability and")
println("miscalibration?  A monotone trend down the table says the sampler only needs to")
println("respect ‖δ‖; a flat one says magnitude is the wrong coordinate and the")
println("instability lives in specific directions instead.")
flush(stdout)

# ── figures ─────────────────────────────────────────────────────────────────
BLU = RGBf(0.0,0.447,0.698); RED = RGBf(0.80,0.15,0.15)
TITLE, LAB, TICK = 13, 12, 11
cmap = cgrad(:viridis, N_BINS; categorical=true)

fig = Figure(size=(FIGW, 0.44FIGW), figure_padding=(6, 10, 4, 6))
ax1 = Axis(fig[1,1]; xlabel="‖δ‖₂ of the correction", ylabel="min ω (THz)",
           title="Stability vs correction size", titlesize=TITLE, xscale=log10,
           xlabelsize=LAB, ylabelsize=LAB, xticklabelsize=TICK, yticklabelsize=TICK,
           xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1)
for b in 1:N_BINS
    m = findall(==(b), sel_bin)
    scatter!(ax1, sel_norm[m], minω[m]; color=cmap[b], markersize=5)
end
hlines!(ax1, [UNSTABLE]; color=RED, linestyle=:dash, linewidth=1.4)
hlines!(ax1, [minω_mean]; color=BLU, linewidth=1.6)

ax2 = Axis(fig[1,2]; xlabel="stratum (increasing ‖δ‖)", ylabel="coverage (%)",
           title="Calibration vs correction size", titlesize=TITLE,
           xlabelsize=LAB, ylabelsize=LAB, xticklabelsize=TICK, yticklabelsize=TICK,
           xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1)
scatterlines!(ax2, 1:N_BINS, [r[6] for r in rows]; color=BLU, markersize=8, label="energy")
scatterlines!(ax2, 1:N_BINS, [r[7] for r in rows]; color=RED, markersize=8, label="force")
axislegend(ax2; position=:rb, labelsize=TICK, framevisible=false)
colgap!(fig.layout, 22)
save("$outdir/stratified_trends.pdf", fig)
save("$outdir/stratified_trends.png", fig; px_per_unit=4)

writedlm("$outdir/stratified_members.csv",
         hcat(sel_obs, sel_bin, sel_norm, minω), ',')
open("$outdir/stratified_summary.csv", "w") do io
    println(io, "bin,median_norm,median_minomega,min_minomega,pct_unstable,E_cov_pct,F_cov_pct")
    for r in rows; @printf(io, "%d,%.6e,%.6f,%.6f,%.3f,%.4f,%.4f\n", r...); end
end
serialize("$outdir/stratified_pops.jls",
          (; sel_obs, sel_bin, sel_norm, minω, minω_mean, edges, rows,
             pr_all, pr_bin, N_BINS, N_PER_BIN, BINNING, STRIDE, LEV_PCT,
             a_eq, n_params, seed=1234))
println("\nfigures → $outdir/{parity,calibration}_pooled.{pdf,png}, stratified_trends.{pdf,png}")
println("data    → $outdir/stratified_{members,summary}.csv")
