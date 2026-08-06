# unstable_recheck_native_Al_20_4_6A_3.jl
#
# Re-test the dynamically-unstable members found by different_filter.jl, removing
# the two approximations that run made, one at a time.  Nothing here reuses the
# undotted machinery — every Hessian is built NATIVELY at that member's own
# parameters, so this is an independent check rather than a re-evaluation.
#
# What different_filter.jl assumed, and why each is worth removing:
#
#   (1) N_cell = 3.  The cubic 3x3x3 cell is 3*4.01962 = 12.06 A, so its box
#       half-width is 6.03 A against a 6 A cutoff.  An atom very nearly interacts
#       with its own periodic image, which contaminates zone-boundary force
#       constants — precisely where these soft modes live.  4x4x4 gives 8.04 A of
#       clearance.  (Same reason the Al_12 committee moved to N_cell=4.)
#
#   (2) Every member evaluated at the MEAN model's a_eq = 4.01962 A.  The samples
#       are not a_eq-pinned (no b'.theta = 0 row), and different_filter.jl reported
#       implied shifts up to 0.0153 A = 0.38%.  A phonon spectrum at the wrong
#       volume is under residual stress, and negative frequencies are exactly what
#       residual tensile stress produces.  So each member is also relaxed to its
#       OWN equilibrium and re-tested there.
#
# Four numbers per member, so the two effects separate cleanly:
#
#   undotted 3x3x3 @ a_mean   what different_filter.jl reported
#   native   3x3x3 @ a_mean   controls for the undotted machinery (must agree)
#   native   4x4x4 @ a_mean   isolates the supercell-size effect
#   native   4x4x4 @ a_own    the physically meaningful answer
#
# An instability that survives all four is real.  One that vanishes at 4x4x4 was a
# periodic-image artefact; one that vanishes at a_own was residual stress.
#
# Run:  julia --project -t 8 scripts/uq/unstable_recheck_native_Al_20_4_6A_3.jl
#
#   NOTE: run this WITH THREADS (-t 8 or more).  different_filter.jl ran on 1
#   thread; the native Hessian build on a 256-atom cell is the whole cost here.

using ACEWorkflow        # load_model, bulk_prim_super, precompute_force_constants,
                         # dynamical_matrix_from_fc, fcc_band_path, relax_lattice_constant, FREQ_THz
using ACEpotentials      # Models.set_linear_parameters!
using AtomsBuilder       # Chemistry.symmetry
using LinearAlgebra      # eigvals, Hermitian, norm
using Statistics         # median
using DelimitedFiles     # readdlm, writedlm
using Printf             # @printf
using CairoMakie         # Figure, Axis, lines!, save

element      = :Al
N_per_seg    = [20, 20, 20, 20, 60]   # identical path to different_filter.jl
unstable_tol = -0.05                  # THz — the threshold that flagged them
qΓtol        = 5e-2                   # skip near-Γ (acoustic → 0)
N_small, N_big = 3, 4

result   = load_model(element, 20, 4, 6, 3; dataset_name="")
model    = result.model
lin_params = result.lin_params
RES      = "$(result.dir)/results"
indir    = "$RES/different_filter"
outdir   = "$RES/different_filter_recheck"; mkpath(outdir)

# ── the samples different_filter.jl actually used ────────────────────────────
# Read them from disk rather than re-drawing: sample_hypercube() is unseeded, so a
# fresh draw would be a DIFFERENT committee and the indices would not line up.
isfile("$indir/samples.csv") || error("$indir/samples.csv not found — run different_filter.jl first")
samples = Matrix(readdlm("$indir/samples.csv", ',')')          # p × N (file is N × p)
minf_undotted = vec(readdlm("$indir/min_freq.csv", ','))
size(samples, 1) == length(lin_params) ||
    error("samples.csv has $(size(samples,1)) params, model has $(length(lin_params))")
length(minf_undotted) == size(samples, 2) ||
    error("min_freq.csv has $(length(minf_undotted)) rows, samples.csv has $(size(samples,2)) columns")

bad = findall(<(unstable_tol), minf_undotted)
@printf("%d / %d members flagged unstable by different_filter.jl: %s\n",
        length(bad), length(minf_undotted), string(bad))
isempty(bad) && error("no member is below $unstable_tol THz — nothing to re-check")   # not exit(): that would kill an including REPL
flush(stdout)

# ── native band structure: build THIS member's Hessian from scratch ──────────
band_path_for(sym, L; N_per_seg) =
    sym in (:fcc, :diamond) ? fcc_band_path(L; N_per_seg) :
    sym == :bcc             ? bcc_band_path(L; N_per_seg) :
    sym == :hcp             ? hcp_band_path(L; N_per_seg) :
    error("unknown structure $sym")

function native_bands(model, θ, element, a, N_cell; N_per_seg=N_per_seg, qΓtol=qΓtol)
    ACEpotentials.Models.set_linear_parameters!(model, θ)
    sys_prim, sys_super = bulk_prim_super(element; a=a, N_cell=N_cell)
    fc = precompute_force_constants(sys_prim, sys_super, model)   # ← native hessian(), not Σ_k θ_k H_k
    q_list, x_vals, x_ticks, labels, _ =
        band_path_for(AtomsBuilder.Chemistry.symmetry(element), fc.L; N_per_seg=N_per_seg)
    F  = Matrix{Float64}(undef, 3fc.Np, length(q_list))
    qn = norm.(q_list)
    for (iq, q) in enumerate(q_list)
        ev = eigvals(Hermitian(dynamical_matrix_from_fc(fc, q)))
        F[:, iq] = sign.(ev) .* sqrt.(abs.(ev)) .* FREQ_THz
    end
    keep  = qn .>= qΓtol
    Fk    = F[:, keep]
    minω  = minimum(Fk)
    iq_min = findall(keep)[argmin(vec(minimum(Fk; dims=1)))]      # which q carries it
    return (; F, x_vals, x_ticks, labels, Np=fc.Np, minω, iq_min,
              q_min = q_list[iq_min], x_min = x_vals[iq_min])
end

# ── run the four treatments for each flagged member (+ the mean as a control) ─
rows   = NamedTuple[]
curves = Dict{Int,Any}()      # member index → (big_own, big_mean) band structures for plotting

todo = vcat(bad, 0)           # 0 = mean model, as a reference row
for i in todo
    θ     = i == 0 ? lin_params : samples[:, i]
    tag   = i == 0 ? "mean" : "member $i"
    und   = i == 0 ? NaN     : minf_undotted[i]

    ACEpotentials.Models.set_linear_parameters!(model, θ)
    a_own = relax_lattice_constant(model, element)

    t = @elapsed begin
        s_mean = native_bands(model, θ, element, 4.01962, N_small)
        b_mean = native_bands(model, θ, element, 4.01962, N_big)
        b_own  = native_bands(model, θ, element, a_own,   N_big)
    end

    push!(rows, (; i, tag, a_own, und,
                   nat3 = s_mean.minω, nat4 = b_mean.minω, nat4_own = b_own.minω,
                   q_lab = b_own.x_min))
    curves[i] = (; b_own, b_mean)

    @printf("%-10s a_own=%.5f Å (Δ%+.4f) | undotted 3³ %+.4f | native 3³ %+.4f | native 4³ %+.4f | native 4³ @a_own %+.4f  [%.1f min]\n",
            tag, a_own, a_own - 4.01962, und, s_mean.minω, b_mean.minω, b_own.minω, t/60)
    flush(stdout)
end

# ── verdict per member ───────────────────────────────────────────────────────
println("\n══ verdict ═══════════════════════════════════════════════════════")
for r in rows
    r.i == 0 && continue
    verdict = r.nat4_own < unstable_tol  ? "REAL — unstable at 4×4×4 and at its own a_eq" :
              r.nat4     < unstable_tol  ? "residual stress — stable once relaxed to a_own" :
              r.nat3     < unstable_tol  ? "periodic-image artefact — gone at 4×4×4" :
                                           "undotted/native mismatch — investigate"
    @printf("  member %-3d  %s\n", r.i, verdict)
end
n_real = count(r -> r.i != 0 && r.nat4_own < unstable_tol, rows)
@printf("\n%d / %d flagged members survive both corrections.\n", n_real, length(bad))

# machinery control: undotted vs native at the SAME geometry must agree closely
disc = [abs(r.und - r.nat3) for r in rows if r.i != 0]
@printf("undotted-vs-native at 3×3×3 @a_mean: max |Δ| = %.2e THz %s\n",
        maximum(disc), maximum(disc) < 1e-3 ? "(machinery agrees)" :
        "← LARGE: the undotted contraction and the native Hessian disagree, investigate before trusting either")
flush(stdout)

# ── figure ───────────────────────────────────────────────────────────────────
BLU = RGBf(0.0, 0.447, 0.698); RED = RGBAf(0.80, 0.15, 0.15, 0.75); GRY = RGBAf(0.45, 0.45, 0.45, 0.55)
TITLE, LAB, TICK = 13, 12, 11
ref = curves[bad[1]].b_own

fig = Figure(size=(540, 350), figure_padding=(6, 10, 4, 6))
ax = Axis(fig[1, 1]; xlabel="Wave vector", ylabel="Frequency (THz)",
          title="native Hessian, 4×4×4, each member at its own a_eq",
          titlesize=TITLE, xlabelsize=LAB, ylabelsize=LAB,
          xticklabelsize=TICK, yticklabelsize=TICK,
          xticks=(ref.x_ticks, ref.labels), xgridvisible=false, ygridvisible=false,
          xtickalign=1, ytickalign=1, xticksize=4, yticksize=4)
for i in bad
    c = curves[i].b_own
    col = c.minω < unstable_tol ? RED : GRY
    for b in 1:3c.Np; lines!(ax, c.x_vals, c.F[b, :]; color=col, linewidth=1.0); end
end
mc = curves[0].b_own
for b in 1:3mc.Np; lines!(ax, mc.x_vals, mc.F[b, :]; color=BLU, linewidth=1.8); end
hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=1.0)
vlines!(ax, ref.x_ticks; color=(:black, 0.22), linewidth=0.7)
xlims!(ax, first(ref.x_vals), last(ref.x_vals))
text!(ax, 0.97, 0.97; text="$n_real/$(length(bad)) still unstable", space=:relative,
      align=(:right, :top), fontsize=TICK-1, color=:gray30)

# min ω under each treatment, so the two corrections are visible separately
ax2 = Axis(fig[1, 2]; ylabel="min ω (THz)", title="per treatment",
           titlesize=TITLE, xlabelsize=LAB, ylabelsize=LAB,
           xticklabelsize=TICK, yticklabelsize=TICK,
           xticks=(1:4, ["und\n3³", "nat\n3³", "nat\n4³", "nat 4³\n@a_own"]),
           xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1)
for r in rows
    r.i == 0 && continue
    ys = [r.und, r.nat3, r.nat4, r.nat4_own]
    lines!(ax2, 1:4, ys; color=(r.nat4_own < unstable_tol ? RED : GRY), linewidth=1.2)
    scatter!(ax2, 1:4, ys; color=(r.nat4_own < unstable_tol ? RED : GRY), markersize=7)
end
hlines!(ax2, [0.0]; color=:black, linestyle=:dash, linewidth=1.0)
colsize!(fig.layout, 1, Relative(0.64)); colgap!(fig.layout, 20)
save("$outdir/unstable_recheck.pdf", fig)
save("$outdir/unstable_recheck.png", fig; px_per_unit=4)

open("$outdir/recheck_table.csv", "w") do io
    println(io, "member,a_own,undotted_3x3x3,native_3x3x3,native_4x4x4,native_4x4x4_own_aeq")
    for r in rows
        @printf(io, "%d,%.6f,%.6f,%.6f,%.6f,%.6f\n", r.i, r.a_own, r.und, r.nat3, r.nat4, r.nat4_own)
    end
end
println("\nfigure → $outdir/unstable_recheck.{pdf,png}")
println("table  → $outdir/recheck_table.csv")
