# naive_vs_constrained_fullcloud_Al_12_4_6A_2.jl
#
# Unconstrained POPS vs constrained POPS, side by side on ONE shared frequency axis.
# Each ensemble is evaluated at the geometry its members actually hold:
#
#   CONSTRAINED  prebuilt undotted per-basis Hessian at a_eq, 4×4×4.  These members
#                have b′·θ = 0 imposed, so a_eq IS their equilibrium and Σ_k θ_k D_k(q)
#                is the exact operator for every one of them — one build serves the
#                whole ensemble.  The pin residual is checked below, so this is a
#                verified assumption rather than an asserted one.
#
#   NAIVE        native Hessian per member, each relaxed to its OWN lattice constant,
#                4×4×4.  Nothing pins these: each has its own equilibrium, and
#                evaluating them at the mean model's a_eq would fold residual stress
#                into the phonons.  A soft mode read off that would be unattributable
#                between "bad parameters" and "wrong volume", which is precisely the
#                claim the figure makes.
#
# Both routes compute the same physical quantity — the member's phonons at its own
# equilibrium — and on the constrained side they agree to ~1e-14 THz (verified against
# a full 4×4×4 cache: rows 1.5e-16, D_k(q) 3.0e-16).  So this is not a method confound;
# it is using the cheap exact route where it is exact and paying for the build only
# where it is needed.
#
# ── WHY THIS IS A NEW SCRIPT AND NOT TWO EXISTING PDFs PASTED TOGETHER ───────
# results/bands_individual_hessian/bands_naive_individual_hessian.pdf next to
# results/cutting_plane_full_cloud/bands_fullcloud.pdf compares four things at once,
# only one of which is the constraint:
#
#            bands_naive_individual_hessian      bands_fullcloud
#   members  10                                  30
#   cell     3×3×3 (half-box 6.03 Å vs a 6 Å     4×4×4 (half-box 8.04 Å — clean)
#            cutoff — periodic images contaminate)
#   q-path   N_per_seg = 20 uniform              [20,20,20,20,60], dense Γ→K
#   y-limits its own                             its own
#
# and each script computes its own y-limits, so they cannot be matched by cropping.
# Here: same top-50%-leverage delta population, same member count, same clean 4×4×4
# cell, same q-path, ONE shared y-limit over both.
#
# ── PARALLELISM ─────────────────────────────────────────────────────────────
# n_members + 1 native builds on a 256-atom cell, chunked over tasks.  Each task gets
# its OWN deep copy of the model: native evaluation calls set_linear_parameters!,
# which mutates, so a shared model would be a silent data race producing wrong bands
# rather than a crash.  Chunked @spawn, not @threads + threadid(), which is unsound
# under task migration.
#
# Run:  julia --project -t <ncores> scripts/uq/naive_vs_constrained_fullcloud_Al_12_4_6A_2.jl [n_members]
#   HESS_THREADS caps concurrent native builds (default: all Julia threads).
#   Needs committee_rejection_full_cloud.csv from
#   scripts/uq/hypercube_full_cloud_bands_Al_12_4_6A_2.jl — the constrained panel reuses
#   those exact members, so it is the same ensemble as bands_fullcloud, not a re-draw.

include(joinpath(@__DIR__, "..", "bandpath_phonon_uq", "lib.jl"))
using Random, Serialization
Random.seed!(1234)

element, dataset  = :Al, ""
N_cell_fc         = 4                          # clean cell: half-box 8.04 Å vs 6 Å cutoff
N_per_seg         = [20, 20, 20, 20, 60]       # identical to the constrained study
lev_percentile    = 0.5                        # the cloud the constrained study started from
qΓtol             = 5e-2
unstable_tol      = -0.05
n_members         = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 30
FIGW              = parse(Float64, get(ENV, "FIGW", "540"))
HESS_THREADS      = parse(Int, get(ENV, "HESS_THREADS", string(Threads.nthreads())))

result = load_model(element, 12, 4, 6, 2; dataset_name=dataset)
model  = result.model; lin_params = result.lin_params; n_params = length(lin_params)
P = result.P
Ap = Diagonal(result.W) * result.A / P
Yw = result.W .* result.Y
RES    = "$(result.dir)/results"
SRC    = "$RES/cutting_plane_full_cloud"
outdir = "$RES/naive_vs_constrained"; mkpath(outdir)
structure = AtomsBuilder.Chemistry.symmetry(element)
@printf("Model %s: %d params, %d observations, %d Julia threads (%d for native Hessians)\n",
        result.name, n_params, size(Ap,1), Threads.nthreads(), HESS_THREADS); flush(stdout)

# ── geometry + the prebuilt undotted band path (the CONSTRAINED evaluator) ───
ACEpotentials.Models.set_linear_parameters!(model, lin_params)
a_eq = ACEWorkflow.relax_lattice_constant(model, element)
@printf("a_eq = %.5f Å; %d×%d×%d cell, half-box %.2f Å vs 6 Å cutoff\n",
        a_eq, N_cell_fc, N_cell_fc, N_cell_fc, N_cell_fc*a_eq/2); flush(stdout)
t_bp = @elapsed bp = bandpath_Dk(result, model, element, a_eq, N_cell_fc; N_per_seg=N_per_seg)
@printf("undotted band path: %d q-points, %d branches  [%.1f min]\n",
        length(bp.Bq), 3bp.Np, t_bp/60); flush(stdout)

lattice_basis(a) = ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(
                       ACEWorkflow.Elasticity.reference_system(element; a=a), model))
b_prime        = ForwardDiff.derivative(lattice_basis, a_eq)
b_double_prime = ForwardDiff.derivative(a -> ForwardDiff.derivative(lattice_basis, a), a_eq)

# ── evaluators, both returning the same shape so the plotter is shared ───────
function undotted_bands(θ)
    F = bands(θ, bp)
    return (F=F, x_vals=bp.x_vals, x_ticks=bp.x_ticks, labels=bp.labels, Np=bp.Np,
            min_stable=minimum(F[:, bp.qnorm .> qΓtol]), a=a_eq)
end

# `m` is a per-task model copy — never pass the shared one, it gets mutated.
function native_bands_one(m, θ, a_ref)
    ACEpotentials.Models.set_linear_parameters!(m, θ)
    a = try
        aa = ACEWorkflow.relax_lattice_constant(m, element)
        (0.9a_ref < aa < 1.1a_ref) ? aa :
            (@warn "relaxed a = $aa Å implausible; falling back to a_ref"; a_ref)
    catch e
        @warn "relax failed ($e); falling back to a_ref"; a_ref
    end
    sp, ss = bulk_prim_super(element; a=a, N_cell=N_cell_fc)
    fc = precompute_force_constants(sp, ss, m)
    ql, xv, xt, lb, _ = _band_path(structure, fc.L; N_per_seg=N_per_seg)
    Np = fc.Np; qn = norm.(ql)
    F = Matrix{Float64}(undef, 3Np, length(ql))
    for (iq, q) in enumerate(ql)
        ev = eigvals(Hermitian(dynamical_matrix_from_fc(fc, q)))
        F[:, iq] = sign.(ev) .* sqrt.(abs.(ev)) .* FREQ_THz
    end
    return (F=F, x_vals=xv, x_ticks=xt, labels=lb, Np=Np,
            min_stable=minimum(F[:, qn .> qΓtol]), a=a)
end

function native_bands_many(θs, a_ref; label="")
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
                out[i] = native_bands_one(m, θs[i], a_ref)
                d = Threads.atomic_add!(done, 1) + 1
                d % max(1, n ÷ 10) == 0 && (@printf("    %d/%d\n", d, n); flush(stdout))
            end
        end
    end
    @printf("  [%s] done in %.1f min (%.1f s/member)\n", label, t/60, t/n); flush(stdout)
    return Vector{typeof(out[1])}(out)
end

# ── ensemble A: NAIVE.  Same top-50%-leverage cloud, no constraints at all ───
# `corrections` is the library's own POPS routine; leverage_percentile = 0.5 selects
# exactly the population that cutting_plane_full_cloud_Al_12_4_6A_2.jl then constrained.
println("\n── naive: unconstrained POPS cloud → hypercube → sample ──"); flush(stdout)
naive_deltas = corrections(Ap, Yw, P; leverage_percentile=lev_percentile)
@printf("  cloud: %d unconstrained deltas, %d params\n", size(naive_deltas)...)
eig_n, bd_n = hypercube(naive_deltas)
@printf("  hypercube: %d directions retained\n", size(eig_n, 2)); flush(stdout)
mat_n, _ = sample_hypercube(eig_n, bd_n, lin_params; number_of_committee_members=n_members)
mem_n = [mat_n[:, i] for i in 1:size(mat_n, 2)]

# ── ensemble B: CONSTRAINED.  The exact members behind bands_fullcloud ───────
REJ = "$SRC/committee_rejection_full_cloud.csv"
isfile(REJ) || error("""
    missing $REJ
    Run scripts/uq/hypercube_full_cloud_bands_Al_12_4_6A_2.jl first — this script reuses
    its committee so the constrained panel is the SAME ensemble as bands_fullcloud
    rather than an independent re-draw.""")
rej = readdlm(REJ, ',')
size(rej, 2) == n_params || error("$REJ is $(size(rej,2)) wide, model has $n_params params")
mem_c  = [collect(Float64, rej[i, :]) for i in 1:min(n_members, size(rej,1))]
θ_mean = vec(readdlm("$RES/bandpath_undotted_ncell4_densek/theta_mean.csv", ','))
@printf("\n── constrained: %d members reused from %s ──\n", length(mem_c), basename(REJ))
length(mem_c) == n_members ||
    @warn "constrained committee has only $(length(mem_c)) members; naive has $n_members — panels are not member-matched"

# ── IS THE PREBUILT a_eq HESSIAN ACTUALLY VALID FOR THESE MEMBERS? ──────────
# The whole justification for not rebuilding on the constrained side is b′·θ = 0.
# Δa ≈ −b′·θ / b″·θ is the Newton step from a_eq to the member's own equilibrium; if
# that is not ~0 the prebuilt operator is being used at the wrong geometry and the
# constrained panel is wrong.  Checked, not assumed.
Δa_c = [-dot(b_prime, θ)/dot(b_double_prime, θ) for θ in mem_c]
@printf("\nPIN CHECK (constrained): |Δa| ≤ %.3e Å (%.4f%% of a_eq)  %s\n",
        maximum(abs.(Δa_c)), 100*maximum(abs.(Δa_c))/a_eq,
        maximum(abs.(Δa_c))/a_eq < 1e-3 ? "← pinned; prebuilt a_eq Hessian is exact" :
                                          "← NOT pinned; rebuild these natively too")
maximum(abs.(Δa_c))/a_eq < 1e-3 ||
    @warn "constrained members are not at a_eq — the prebuilt Hessian is the wrong operator for them"
flush(stdout)

# ── evaluate ────────────────────────────────────────────────────────────────
println("\n── native Hessians for the naive ensemble ──"); flush(stdout)
nat_n = native_bands_many(mem_n, a_eq; label="naive")
cen_n = native_bands_one(deepcopy(model), lin_params, a_eq)
ACEpotentials.Models.set_linear_parameters!(model, lin_params)   # restore the shared model

nat_c = [undotted_bands(θ) for θ in mem_c]
cen_c = undotted_bands(θ_mean)

minf_n = [m.min_stable for m in nat_n]; a_n = [m.a for m in nat_n]
minf_c = [m.min_stable for m in nat_c]

@printf("\nmean model (lin_params)  : min ω = %+.4f THz at its own a = %.5f Å\n",
        cen_n.min_stable, cen_n.a)
@printf("constrained mean (θ_mean): min ω = %+.4f THz at a_eq\n", cen_c.min_stable)
@printf("NAIVE       : min ω ∈ [%+.4f, %+.4f], median %+.4f — %d/%d unstable (< %.2f THz)\n",
        minimum(minf_n), maximum(minf_n), median(minf_n),
        count(<(unstable_tol), minf_n), length(minf_n), unstable_tol)
@printf("CONSTRAINED : min ω ∈ [%+.4f, %+.4f], median %+.4f — %d/%d unstable\n",
        minimum(minf_c), maximum(minf_c), median(minf_c),
        count(<(unstable_tol), minf_c), length(minf_c))

@printf("\nrelaxed lattice constant, naive ensemble (a_eq = %.5f Å):\n", a_eq)
@printf("  a ∈ [%.5f, %.5f] Å, spread %.4f Å (%.2f%% of a_eq)\n",
        minimum(a_n), maximum(a_n), maximum(a_n)-minimum(a_n),
        100*(maximum(a_n)-minimum(a_n))/a_eq)
println("  → this spread is why the naive side is rebuilt natively; the constrained")
println("    side sits at a_eq by construction, so one prebuilt Hessian covers it.")
flush(stdout)

# ── figures: one shared y-axis over BOTH ensembles ──────────────────────────
# Naive members each have their own reciprocal lattice, so their q-points differ in
# Cartesian terms; N_per_seg is identical so the paths correspond point-for-point and
# everything is drawn against the a_eq path coordinate.
BLU = RGBf(0.0, 0.447, 0.698); GRY = RGBAf(0.45,0.45,0.45,0.30); RED = RGBAf(0.80,0.15,0.15,0.45)
TITLE, LAB, TICK = 13, 12, 11
xref = bp.x_vals; xticks = bp.x_ticks; labels = bp.labels; Np = bp.Np
length(cen_n.x_vals) == length(xref) ||
    error("native path has $(length(cen_n.x_vals)) q-points, undotted has $(length(xref)) — N_per_seg mismatch")

allF = vcat([m.F for m in nat_n], [m.F for m in nat_c], [cen_n.F, cen_c.F])
ylo = minimum(minimum.(allF)); yhi = maximum(maximum.(allF)); pad = 0.06*(yhi-ylo)
ylim = (min(ylo-pad, -0.4), yhi+pad)
@printf("\nshared frequency axis: [%.2f, %.2f] THz\n", ylim...)

function panel!(gp, members, central, ttl, letter; showy=true)
    ax = Axis(gp; xlabel="Wave vector", ylabel = showy ? "Frequency (THz)" : "",
              title=ttl, titlesize=TITLE, xlabelsize=LAB, ylabelsize=LAB,
              xticklabelsize=TICK, yticklabelsize=TICK,
              xticks=(xticks, labels), xgridvisible=false, ygridvisible=false,
              xtickalign=1, ytickalign=1, xticksize=4, yticksize=4)
    for m in members
        col = m.min_stable < unstable_tol ? RED : GRY
        for b in 1:3Np; lines!(ax, xref, m.F[b,:]; color=col, linewidth=0.8); end
    end
    for b in 1:3Np; lines!(ax, xref, central.F[b,:]; color=BLU, linewidth=1.8); end
    hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=1.0)
    vlines!(ax, xticks; color=(:black,0.22), linewidth=0.7)
    xlims!(ax, first(xref), last(xref)); ylims!(ax, ylim...)
    showy || hideydecorations!(ax; grid=false, ticks=false, minorticks=false)
    isempty(letter) || text!(ax, 0.03, 0.97; text=letter, space=:relative,
                             align=(:left,:top), font=:bold, fontsize=TITLE)
    return ax
end

nu = count(<(unstable_tol), minf_n); cu = count(<(unstable_tol), minf_c)
t_n = "unconstrained POPS — $nu/$(length(minf_n)) unstable"
t_c = "constrained POPS — $cu/$(length(minf_c)) unstable"

fig = Figure(size=(FIGW, 0.40FIGW), figure_padding=(6, 10, 4, 6))
panel!(fig[1,1], nat_n, cen_n, t_n, "(a)")
panel!(fig[1,2], nat_c, cen_c, t_c, "(b)"; showy=false)
colgap!(fig.layout, 20)
save("$outdir/bands_naive_vs_constrained.pdf", fig)
save("$outdir/bands_naive_vs_constrained.png", fig; px_per_unit=4)
println("side-by-side → $outdir/bands_naive_vs_constrained.{pdf,png}")

# also as separate files, on the SAME y-limits, for independent placement
for (members, central, ttl, tag) in ((nat_n, cen_n, t_n, "naive"),
                                     (nat_c, cen_c, t_c, "constrained"))
    f1 = Figure(size=(FIGW, 0.62FIGW), figure_padding=(6, 10, 4, 6))
    panel!(f1[1,1], members, central, ttl, "")
    save("$outdir/bands_$(tag)_shared_axis.pdf", f1)
    save("$outdir/bands_$(tag)_shared_axis.png", f1; px_per_unit=4)
    @printf("single       → %s/bands_%s_shared_axis.{pdf,png}\n", outdir, tag)
end

writedlm("$outdir/min_freq_naive.csv",       hcat(minf_n, a_n), ',')
writedlm("$outdir/min_freq_constrained.csv", hcat(minf_c, Δa_c), ',')
writedlm("$outdir/samples_naive.csv", mat_n', ',')
serialize("$outdir/naive_vs_constrained.jls",
          (; minf_n, minf_c, a_n, Δa_c, ylim, a_eq, N_cell_fc, n_members,
             cen_n_min=cen_n.min_stable, cen_c_min=cen_c.min_stable,
             unstable_tol, lev_percentile, seed=1234))
println("data         → $outdir/min_freq_naive.csv (min ω, relaxed a)")
println("               $outdir/min_freq_constrained.csv (min ω, implied Δa)")
