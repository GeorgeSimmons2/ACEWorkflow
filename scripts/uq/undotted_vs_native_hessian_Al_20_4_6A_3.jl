# undotted_vs_native_hessian_Al_20_4_6A_3.jl
#
# WHY: unstable_recheck_native_Al_20_4_6A_3.jl found that the undotted contraction
# Sum_k theta_k H_k and the native hessian() disagree by up to 0.20 THz in min omega
# at the SAME geometry (3x3x3, a = 4.01962 A) — where they should be identical,
# since E is linear in theta.  lib.jl claims validation to 1e-13.
#
# The discrepancy is not uniform.  It tracks cell-size sensitivity exactly:
#
#     member 34  Delta = 0.0003 THz   min omega identical at 3^3 and 4^3
#     member 26  Delta = 0.2015 THz   min omega changes a lot with cell size
#
# HYPOTHESIS: this is not a bug in the undotted code.  The 3x3x3 cubic cell is
# 12.06 A, so its box half-width is 6.03 A against a 6 A cutoff.  Pairs sitting
# within 0.03 A of the half-box boundary are exactly where an atom starts to see
# its own periodic image, and the two implementations resolve that boundary
# slightly differently.  If so, the undotted machinery is sound and only the
# 3x3x3 CACHE is unusable — a 4x4x4 cache (8.04 A clearance) would be clean.
#
# ALTERNATIVE: a genuine bug in undotted_hessian() (accumulation, PBC, threading),
# in which case the error is spread over all pair distances, not concentrated at
# the boundary, and no cache at any size can be trusted.
#
# THE TEST distinguishes them without any phonon machinery in the way: contract
# the cached basis Hessian and compare it element-by-element against the native
# Hessian at the same parameters, then resolve the error by pair distance.
#
#     error concentrated near r = 6 A  ->  cutoff marginality, rebuild at 4x4x4
#     error spread across all r        ->  real bug, fix before trusting anything
#
# Cost: one native Hessian build per theta (~1 min) plus a contraction. The 1.5 GB
# cache is loaded once and reused.
#
# Run:  julia --project -t 8 scripts/uq/undotted_vs_native_hessian_Al_20_4_6A_3.jl

using ACEWorkflow        # load_model, bulk_prim_super, Elasticity.lattice_matrix
using ACEpotentials      # Models.set_linear_parameters!
import AtomsCalculatorsUtilities.SitePotentials: hessian   # NOT re-exported by ACEWorkflow
using Unitful            # ustrip
using LinearAlgebra, Statistics, Serialization, DelimitedFiles, Printf
using CairoMakie

element = :Al
N_cell  = 3
# a_ref is taken from the cache itself, below — NOT from the filename.  undotted_Hbasis
# names its file with round(a; digits=5) but builds at the unrounded a, so using the
# filename value puts the native build at a slightly different geometry.  With
# dH/da ~ 1e4 /Å even a 3e-6 Å offset moves H by ~0.03, which is enough to fake a
# machinery discrepancy.

result = load_model(element, 20, 4, 6, 3; dataset_name="")
model  = result.model
lin_params = result.lin_params
RES    = "$(result.dir)/results"
outdir = "$RES/different_filter_recheck"; mkpath(outdir)

samples = Matrix(readdlm("$RES/different_filter/samples.csv", ',')')     # p × N

# member 34 agreed to 3e-4, member 26 disagreed by 0.20 — the contrast is the point
probes = [("mean", lin_params), ("member 26", samples[:, 26]), ("member 34", samples[:, 34])]

Hcache = deserialize("$RES/undotted_Hbasis_3x3x3_a4.01962.jls")
Hb     = Hcache.H_basis
a_ref  = Hcache.a_eq                     # exact stored geometry, not the rounded filename
@printf("cache: H_basis %s, N_cell = %d\n", string(size(Hb)), Hcache.N_cell)
@printf("a_ref = %.17g  (filename says 4.01962; differ by %.3e Å)\n", a_ref, abs(a_ref - 4.01962))

_, sys_super = bulk_prim_super(element; a=a_ref, N_cell=N_cell)
Nat = length(sys_super); N3 = 3Nat
@printf("supercell: %d atoms, N3 = %d, box = %.3f A, half-box = %.3f A vs cutoff 6 A\n",
        Nat, N3, N_cell*a_ref, N_cell*a_ref/2)
flush(stdout)

# ── pair distances under the minimum-image convention, for binning the error ──
L = ustrip.(ACEWorkflow.Elasticity.lattice_matrix(sys_super.cell.cell_vectors))
Linv = inv(L)
pos = [ustrip.(sys_super[i].position) for i in 1:Nat]
rij = zeros(Nat, Nat)
for i in 1:Nat, j in 1:Nat
    d = pos[j] .- pos[i]
    f = Linv * d; f = f .- round.(f)
    rij[i, j] = norm(L * f)
end
@printf("pair distances: max = %.3f A (half-box %.3f)\n", maximum(rij), N_cell*a_ref/2)

Hflat = reshape(Hb, N3*N3, size(Hb, 3))     # view-free reshape for the contraction

results = NamedTuple[]
for (tag, θ) in probes
    H_und = reshape(Hflat * θ, N3, N3)                      # Sum_k theta_k H_k
    ACEpotentials.Models.set_linear_parameters!(model, θ)
    t = @elapsed H_nat = ustrip.(hessian(sys_super, model))  # native, same geometry
    D = abs.(H_und .- H_nat)
    scale = maximum(abs.(H_nat))
    @printf("\n[%s] native build %.1f min | max|H_nat| = %.4g\n", tag, t/60, scale)
    @printf("  max abs diff  = %.4e  (%.3e relative)\n", maximum(D), maximum(D)/scale)
    @printf("  mean abs diff = %.4e\n", mean(D))

    # 3x3 block error per atom pair, binned by distance
    blk = [maximum(@view D[3i-2:3i, 3j-2:3j]) for i in 1:Nat, j in 1:Nat]
    edges = 0.0:0.5:(maximum(rij)+0.5)
    binmax = zeros(length(edges)-1); binmean = zeros(length(edges)-1); bincount = zeros(Int, length(edges)-1)
    for i in 1:Nat, j in 1:Nat
        b = searchsortedlast(edges, rij[i,j]); (b < 1 || b > length(binmax)) && continue
        binmax[b] = max(binmax[b], blk[i,j]); binmean[b] += blk[i,j]; bincount[b] += 1
    end
    binmean ./= max.(bincount, 1)
    println("  error vs pair distance (max |Δ| in each 0.5 A shell):")
    for b in eachindex(binmax)
        bincount[b] == 0 && continue
        mark = (edges[b] >= 5.5 && edges[b] < 6.5) ? "  <-- cutoff / half-box shell" : ""
        @printf("    r ∈ [%.1f, %.1f)  max %.3e  mean %.3e  (n=%d)%s\n",
                edges[b], edges[b+1], binmax[b], binmean[b], bincount[b], mark)
    end

    # how much of the total error lives in the marginal shell?
    near = (rij .>= 5.5) .& (rij .< 6.5)
    frac = sum(blk[near]) / max(sum(blk), eps())
    @printf("  fraction of summed block error at 5.5 <= r < 6.5 A: %.1f%%\n", 100frac)
    push!(results, (; tag, maxD=maximum(D), rel=maximum(D)/scale, frac, binmax, binmean, edges))
    flush(stdout)
end

println("\n══ verdict ═══════════════════════════════════════════════════════")
for r in results
    v = r.rel < 1e-8                    ? "machineries agree — discrepancy is NOT in the Hessian" :
        r.frac > 0.5                    ? "error concentrated at the half-box shell — cutoff marginality, rebuild the cache at N_cell=4" :
                                          "error spread across all distances — genuine bug in undotted_hessian(), do not trust any cache"
    @printf("  %-10s rel %.2e, %.0f%% of error at r~6 A  ->  %s\n", r.tag, r.rel, 100r.frac, v)
end

# ── figure ───────────────────────────────────────────────────────────────────
TITLE, LAB, TICK = 13, 12, 11
fig = Figure(size=(540, 320), figure_padding=(6, 10, 4, 6))
ax = Axis(fig[1,1]; xlabel="pair distance r (Å)", ylabel="max |H_und − H_nat| in shell",
          yscale=log10, title="undotted vs native Hessian, 3×3×3, a = $a_ref Å",
          titlesize=TITLE, xlabelsize=LAB, ylabelsize=LAB,
          xticklabelsize=TICK, yticklabelsize=TICK,
          xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1)
cols = [RGBf(0.0,0.447,0.698), RGBf(0.835,0.369,0.0), RGBf(0.0,0.62,0.451)]
for (k, r) in enumerate(results)
    x = [(r.edges[b]+r.edges[b+1])/2 for b in eachindex(r.binmax)]
    y = max.(r.binmax, 1e-18)
    lines!(ax, x, y; color=cols[k], linewidth=1.6, label=r.tag)
    scatter!(ax, x, y; color=cols[k], markersize=6)
end
vlines!(ax, [6.0]; color=:black, linestyle=:dash, linewidth=1.2)
vlines!(ax, [N_cell*a_ref/2]; color=(:red, 0.6), linestyle=:dot, linewidth=1.2)
text!(ax, 6.05, 0.5; text="cutoff 6 Å / half-box $(round(N_cell*a_ref/2; digits=2)) Å",
      space=:relative, fontsize=TICK-1, color=:gray30)
axislegend(ax; position=:lt, labelsize=TICK, framevisible=true)
save("$outdir/undotted_vs_native.pdf", fig)
save("$outdir/undotted_vs_native.png", fig; px_per_unit=4)
println("\nfigure → $outdir/undotted_vs_native.{pdf,png}")
