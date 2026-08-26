# build_bands_cache_Al_12.jl
#
# Rebuild the Al_12_4_6A_2_ phonon band CURVES and cache them, so that the
# four-panel figure is a pure replot.
#
# ── WHY THIS SCRIPT HAS TO EXIST ────────────────────────────────────────────
# The Al_16 study serialised its band curves (bands_two_ensembles.jls holds every
# member's F, 3 × n_q), so its two panels can be redrawn in a second.  The Al_12 study
# did NOT: naive_vs_constrained.jls holds only summary statistics (min ω per member,
# relaxed a, the shared y-limits).  The curves behind
# bands_naive_shared_axis.pdf / bands_constrained_shared_axis.pdf exist only inside
# those PDFs, and a PDF cannot be restyled into a new panel.
#
# So the Al_12 row has to be recomputed once.  Recomputed, NOT re-sampled: both
# ensembles are read back verbatim from the CSVs the original run wrote, and each
# naive member is rebuilt at the lattice constant that run recorded for it.  Nothing
# here depends on an RNG seed, so the curves are the ones already published, not a
# fresh draw that happens to look similar.  The check at the end proves it: every
# member's recomputed min ω is compared against the saved value.
#
#   naive members       samples_naive.csv            (30 × 91, one member per row)
#   their geometries    min_freq_naive.csv column 2  (each member's own relaxed a)
#   constrained members committee_rejection_full_cloud.csv  (first 30 rows)
#   constrained centre  bandpath_undotted_ncell4_densek/theta_mean.csv
#
# ── THE TWO EVALUATORS ARE NOT INTERCHANGEABLE ─────────────────────────────
# Same split as naive_vs_constrained_fullcloud_Al_12_4_6A_2.jl, for the same reason:
#
#   CONSTRAINED  prebuilt undotted per-basis Hessian at a_eq.  These members have
#                b′·θ = 0 imposed, so a_eq IS their equilibrium and Σ_k θ_k D_k(q) is
#                the exact operator for all 30.  The pin residual is re-checked below.
#   NAIVE        native Hessian per member at its OWN lattice constant.  Nothing pins
#                these (a spans 3.98–4.30 Å), and evaluating them at a_eq would fold
#                residual stress into the phonons — a soft mode read off that would be
#                unattributable between bad parameters and wrong volume.
#
# Cost is dominated by the 31 native 4×4×4 (256-atom) Hessians on the naive side.
# Give it threads; each task holds its own model copy because native evaluation calls
# set_linear_parameters!, and a shared model would be a silent data race producing
# wrong bands rather than a crash.
#
# Run (once):
#   julia --project -t 40 bands_four_panel/build_bands_cache_Al_12.jl
#
#   OUT      output file (default models/Al_12_4_6A_2_/results/naive_vs_constrained/
#                                  bands_four_panel_Al_12.jls)
#   RELAX=1  re-relax each naive member instead of reusing the saved a
#   HESS_THREADS  cap on concurrent native builds (default: all Julia threads)

include(joinpath(@__DIR__, "..", "scripts", "bandpath_phonon_uq", "lib.jl"))

element, dataset = :Al, ""
N_CELL      = 4                        # clean cell: half-box 8.04 Å vs 6 Å cutoff
N_PER_SEG   = [20, 20, 20, 20, 60]     # identical to the original run — dense Γ→K
qΓtol       = 5e-2
UNSTABLE    = -0.05                    # kept only so the cache records the run's convention
N_MEMBERS   = 30
RELAX       = get(ENV, "RELAX", "0") != "0"
HESS_THREADS = parse(Int, get(ENV, "HESS_THREADS", string(Threads.nthreads())))

result = load_model(element, 12, 4, 6, 2; dataset_name=dataset)
model  = result.model; lin_params = result.lin_params; n_params = length(lin_params)
RES    = "$(result.dir)/results"
SRC_N  = "$RES/naive_vs_constrained"                  # the original run's outputs
SRC_C  = "$RES/cutting_plane_full_cloud"              # the constrained committee
OUT    = get(ENV, "OUT", "$SRC_N/bands_four_panel_Al_12.jls")
structure = AtomsBuilder.Chemistry.symmetry(element)
@printf("Model %s: %d params, %d Julia threads (%d for native Hessians)\n",
        result.name, n_params, Threads.nthreads(), HESS_THREADS); flush(stdout)

# ── geometry, and the reproducibility check against the original run ────────
ACEpotentials.Models.set_linear_parameters!(model, lin_params)
a_eq = ACEWorkflow.relax_lattice_constant(model, element)
prev = deserialize("$SRC_N/naive_vs_constrained.jls")
@printf("a_eq = %.7f Å   (original run: %.7f Å, Δ = %.2e)\n", a_eq, prev.a_eq, a_eq - prev.a_eq)
abs(a_eq - prev.a_eq) < 1e-6 ||
    error("a_eq differs from the original run by $(a_eq - prev.a_eq) Å — the model on disk " *
          "is not the one that produced the published figure; the cache would not be a replot")
prev.N_cell_fc == N_CELL || error("original run used N_cell = $(prev.N_cell_fc), this uses $N_CELL")

# ── the CONSTRAINED evaluator: one prebuilt undotted band path at a_eq ──────
t_bp = @elapsed bp = bandpath_Dk(result, model, element, a_eq, N_CELL; N_per_seg=N_PER_SEG)
@printf("undotted band path: %d q-points, %d branches  [%.1f min]\n",
        length(bp.Bq), 3bp.Np, t_bp/60); flush(stdout)

lattice_basis(a) = ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(
                       ACEWorkflow.Elasticity.reference_system(element; a=a), model))
b_prime        = ForwardDiff.derivative(lattice_basis, a_eq)
b_double_prime = ForwardDiff.derivative(a -> ForwardDiff.derivative(lattice_basis, a), a_eq)

function undotted_bands(θ)
    F = bands(θ, bp)
    return (F=F, x_vals=bp.x_vals, x_ticks=bp.x_ticks, labels=bp.labels, Np=bp.Np,
            minω = minimum(F[:, bp.qnorm .> qΓtol]), a = a_eq)
end

# ── the NAIVE evaluator: native Hessian at the member's own lattice constant ─
# `m` is a per-task model copy — never the shared one, it gets mutated.
function native_bands_one(m, θ, a)
    ACEpotentials.Models.set_linear_parameters!(m, θ)
    sp, ss = bulk_prim_super(element; a=a, N_cell=N_CELL)
    fc = precompute_force_constants(sp, ss, m)
    ql, xv, xt, lb, _ = _band_path(structure, fc.L; N_per_seg=N_PER_SEG)
    Np = fc.Np; qn = norm.(ql)
    F = Matrix{Float64}(undef, 3Np, length(ql))
    for (iq, q) in enumerate(ql)
        ev = eigvals(Hermitian(dynamical_matrix_from_fc(fc, q)))
        F[:, iq] = sign.(ev) .* sqrt.(abs.(ev)) .* FREQ_THz
    end
    return (F=F, x_vals=xv, x_ticks=xt, labels=lb, Np=Np,
            minω = minimum(F[:, qn .> qΓtol]), a=a)
end

function relax_or_reuse(m, θ, a_saved)
    RELAX || return a_saved
    ACEpotentials.Models.set_linear_parameters!(m, θ)
    try
        aa = ACEWorkflow.relax_lattice_constant(m, element)
        (0.9a_eq < aa < 1.1a_eq) ? aa : (@warn "relaxed a = $aa Å implausible; using saved"; a_saved)
    catch e
        @warn "relax failed ($e); using saved a"; a_saved
    end
end

function native_bands_many(θs, as; label="")
    n = length(θs)
    out = Vector{Any}(undef, n)
    nt  = clamp(HESS_THREADS, 1, min(Threads.nthreads(), n))
    done = Threads.Atomic{Int}(0)
    @printf("  [%s] %d native builds on %d tasks, %d deep …\n", label, n, nt, cld(n, nt))
    flush(stdout)
    t = @elapsed @sync for k in 1:nt
        Threads.@spawn begin
            m = deepcopy(model)                       # private to this task
            for i in k:nt:n
                out[i] = native_bands_one(m, θs[i], relax_or_reuse(m, θs[i], as[i]))
                d = Threads.atomic_add!(done, 1) + 1
                d % max(1, n ÷ 10) == 0 && (@printf("    %d/%d\n", d, n); flush(stdout))
            end
        end
    end
    @printf("  [%s] done in %.1f min (%.1f s/member)\n", label, t/60, t/n); flush(stdout)
    return Vector{typeof(out[1])}(out)
end

# ── read both ensembles back verbatim ───────────────────────────────────────
mat_n = readdlm("$SRC_N/samples_naive.csv", ',')
size(mat_n) == (N_MEMBERS, n_params) ||
    error("samples_naive.csv is $(size(mat_n)), expected ($N_MEMBERS, $n_params)")
mem_n = [collect(Float64, mat_n[i, :]) for i in 1:N_MEMBERS]
mf_n  = readdlm("$SRC_N/min_freq_naive.csv", ',')          # col 1 = min ω, col 2 = relaxed a
a_n   = collect(Float64, mf_n[:, 2])

REJ = "$SRC_C/committee_rejection_full_cloud.csv"
rej = readdlm(REJ, ',')
size(rej, 2) == n_params || error("$REJ is $(size(rej,2)) wide, model has $n_params params")
mem_c  = [collect(Float64, rej[i, :]) for i in 1:min(N_MEMBERS, size(rej, 1))]
θ_mean = vec(readdlm("$RES/bandpath_undotted_ncell4_densek/theta_mean.csv", ','))
@printf("ensembles: %d naive (a ∈ [%.5f, %.5f] Å), %d constrained\n",
        length(mem_n), minimum(a_n), maximum(a_n), length(mem_c)); flush(stdout)

# Is the prebuilt a_eq Hessian actually the right operator for the constrained members?
# Δa ≈ −b′·θ / b″·θ is the Newton step from a_eq to the member's own equilibrium.
# If that is not ~0 the constrained panel is being drawn at the wrong geometry.
Δa_c = [-dot(b_prime, θ)/dot(b_double_prime, θ) for θ in mem_c]
@printf("PIN CHECK: |Δa| ≤ %.3e Å (%.5f%% of a_eq)  %s\n",
        maximum(abs.(Δa_c)), 100*maximum(abs.(Δa_c))/a_eq,
        maximum(abs.(Δa_c))/a_eq < 1e-3 ? "← pinned; prebuilt a_eq Hessian is exact" :
                                          "← NOT pinned; these need native rebuilds too")

# ── evaluate ────────────────────────────────────────────────────────────────
println("\n── constrained: prebuilt undotted path ──"); flush(stdout)
res_c = [undotted_bands(θ) for θ in mem_c]
cen_c = undotted_bands(θ_mean)

println("\n── naive: native Hessians, one per member geometry ──"); flush(stdout)
res_n = native_bands_many(mem_n, a_n; label="naive")
cen_n = native_bands_one(deepcopy(model), lin_params, a_eq)   # the mean model at its own a_eq
ACEpotentials.Models.set_linear_parameters!(model, lin_params) # restore the shared model

# ── does this reproduce the published figure? ───────────────────────────────
# Not a formality: if the members, geometries or q-path had drifted, min ω would move.
dn = maximum(abs.([b.minω for b in res_n] .- prev.minf_n))
dc = maximum(abs.([b.minω for b in res_c] .- prev.minf_c))
@printf("\nREPRODUCTION CHECK vs naive_vs_constrained.jls\n")
@printf("  naive       max |Δ min ω| = %.3e THz\n", dn)
@printf("  constrained max |Δ min ω| = %.3e THz\n", dc)
@printf("  centres: naive %+.4f (saved %+.4f), constrained %+.4f (saved %+.4f)\n",
        cen_n.minω, prev.cen_n_min, cen_c.minω, prev.cen_c_min)
if max(dn, dc) > 1e-6
    @warn """recomputed min ω differs from the original run by up to $(max(dn,dc)) THz.
             The curves are NOT the published ones — check the model on disk and the CSVs
             before using this cache in the paper figure."""
else
    println("  ✓ identical to the published run")
end

# ── shared y-limits over BOTH Al_12 panels (the row's own comparison) ───────
allF = vcat([b.F for b in res_n], [b.F for b in res_c], [cen_n.F, cen_c.F])
ylo = minimum(minimum.(allF)); yhi = maximum(maximum.(allF)); pad = 0.06*(yhi-ylo)
ylim = (min(ylo-pad, -0.4), yhi+pad)
@printf("\nrow frequency axis: [%.2f, %.2f] THz   (original run: [%.2f, %.2f])\n",
        ylim..., prev.ylim...)

# Same field names as the Al_16 bands_two_ensembles.jls so the plotter is shared, plus
# `cen`: unlike Al_16, the two Al_12 panels have DIFFERENT central models (the naive
# panel is centred on lin_params at its own geometry, the constrained one on θ_mean).
serialize(OUT, (; res = [res_n, res_c], cen = [cen_n, cen_c], mean_b = cen_c,
                  tags = ["unconstrained", "constrained"], ylim,
                  N_CELL, N_PER_SEG, UNSTABLE, n_params, a_eq,
                  model_name = result.name))
@printf("\n══ DONE ══  cache → %s\n", OUT)
println("Now:  julia --project bands_four_panel/plot_four_panel_bands.jl")
