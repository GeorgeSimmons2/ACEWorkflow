# bands_individual_hessian_naive_vs_constrained_Al_12_4_6A_2.jl
#
# Apples-to-apples phonon comparison for Al_12_4_6A_2_ (full dataset), with every
# member's phonons computed from its OWN native Hessian (precompute_force_constants
# at the member's own relaxed geometry) — the fully honest calculation, no shared
# fixed-geometry undotted sum.  Two plots:
#
#   • NAIVE hypercube:   box fitted to the naive delta forest, sampled (NO predicate)
#                        around the LSQ point lin_params.  10 samples + lin_params.
#   • CONSTRAINED:       replot of bands_constrained — 10 members from the saved
#                        rejection committee (committee_rejection.csv) + θ_mean.
#
# 10 ensemble members + 1 central model per plot = 11 Hessians each = 22 total,
# all on a 3×3×3 supercell.  (The rejection SAMPLING itself was done with the fast
# precomputed-basis Hessian in the committee run; here we only re-evaluate the
# already-saved members with their own native Hessian.)
#
# Formatting matches results/bandpath_phonon_uq/bands_naive.pdf (same figsize/style
# as lib's plot_committee_bands), and BOTH plots share one y-limit so the max/min
# frequencies line up.
#
# Committee CSVs are read from results/bandpath_undotted/ (the only saved ones — see
# the NPT-study header for the CSV-vs-PDF provenance note).
#
# Run:  julia --project -t <N> scripts/uq/bands_individual_hessian_naive_vs_constrained_Al_12_4_6A_2.jl

include(joinpath(@__DIR__, "..", "bandpath_phonon_uq", "lib.jl"))
using Random
Random.seed!(1234)

element          = :Al
dataset          = ""
N_cell_fc        = 3
N_per_seg        = 20
n_ensemble       = 10          # members per plot (10 + central = 11 Hessians per plot → 22 total)
qΓtol            = 5e-2        # skip near-Γ when judging stability (acoustic → 0)
unstable_tol     = -0.05       # THz; below → colour the member crimson (as in plot_committee_bands)
committee_subdir = "bandpath_undotted"

result = load_model(element, 12, 4, 6, 2; dataset_name=dataset)
model  = result.model; lin_params = result.lin_params; n_params = length(lin_params)
P = result.P; Ap = Diagonal(result.W)*result.A/P; Yw = result.W.*result.Y; λ = 1.0/size(Ap,1)
committee_dir = "$(result.dir)/results/$committee_subdir"
outdir = "$(result.dir)/results/bands_individual_hessian"; mkpath(outdir)
structure = AtomsBuilder.Chemistry.symmetry(element)
@printf("Model %s: %d params, %d threads.  Outputs → %s\n", result.name, n_params, Threads.nthreads(), outdir)

# reference lattice constant (for the shared x-axis)
ACEpotentials.Models.set_linear_parameters!(model, lin_params)
a_ref = ACEWorkflow.relax_lattice_constant(model, element)
@printf("Reference a (lin_params) = %.5f Å\n", a_ref)

# ── native per-member bands: relax to the member's own geometry, native Hessian ─
function native_bands(θ)
    ACEpotentials.Models.set_linear_parameters!(model, θ)
    a = try
        aa = ACEWorkflow.relax_lattice_constant(model, element)
        (0.9a_ref < aa < 1.1a_ref) ? aa : (@warn "relaxed a=$aa Å implausible; using a_ref"; a_ref)
    catch e
        @warn "relax failed ($e); using a_ref"; a_ref
    end
    sp, ss = bulk_prim_super(element; a=a, N_cell=N_cell_fc)
    fc = precompute_force_constants(sp, ss, model)
    ql, xv, xt, lb, _ = _band_path(structure, fc.L; N_per_seg=N_per_seg)
    Np = fc.Np; qn = norm.(ql)
    F = Matrix{Float64}(undef, 3Np, length(ql))
    for (iq, q) in enumerate(ql)
        ev = eigvals(Hermitian(dynamical_matrix_from_fc(fc, q)))
        F[:, iq] = sign.(ev) .* sqrt.(abs.(ev)) .* FREQ_THz
    end
    mstab = minimum(F[:, qn .> qΓtol])
    return (F=F, x_vals=xv, x_ticks=xt, labels=lb, Np=Np, min_stable=mstab, a=a)
end

# ── ensembles ────────────────────────────────────────────────────────────────
# naive delta forest (Stage-1 machinery of the committee script) → naive hypercube
C = Symmetric(Ap'*Ap .+ λ.*(P'*P)); Cf = cholesky(C)
AtX = Cf\Matrix(Ap'); θ̃ = Cf\(Ap'*Yw)
leverage = vec(sum(Ap'.*AtX; dims=1)); residual = Yw .- Ap*θ̃
forest_member(i) = lin_params .+ (P \ (AtX[:, i] .* (residual[i]/leverage[i])))
lev_idx = sortperm(leverage; rev=true)[1:5]
res_idx = Int[]; for i in sortperm(abs.(residual); rev=true); i in lev_idx && continue; push!(res_idx,i); length(res_idx)==10 && break; end
taken = Set(vcat(lev_idx,res_idx)); rand_idx = Int[]
while length(rand_idx) < 15; i = rand(1:length(Yw)); (i in taken) && continue; push!(rand_idx,i); push!(taken,i); end
selected = vcat(lev_idx, res_idx, rand_idx)
naive_forest = [forest_member(i) for i in selected]
naive_deltas = reduce(hcat, naive_forest)' .- lin_params'
hyp_eig, hyp_bound = hypercube(Matrix(naive_deltas))
smat, _ = rejection_sample_hypercube(hyp_eig, hyp_bound, lin_params, θ->true;
                                     number_of_committee_members=n_ensemble, max_attempts=1_000_000)
naive_samples = [smat[:, i] for i in 1:size(smat, 2)]

# constrained rejection committee (first n_ensemble) + constrained mean
rej     = readdlm("$committee_dir/committee_rejection.csv", ',')
@assert size(rej,2) == n_params "committee CSV width ≠ n_params"
con_samples = [collect(Float64, rej[i, :]) for i in 1:min(n_ensemble, size(rej,1))]
θ_mean  = vec(readdlm("$committee_dir/theta_mean.csv", ','))

# ── compute all 22 native Hessians / band structures ────────────────────────
println("\nComputing native Hessians (22 × 3×3×3) …")
t = @elapsed begin
    naive_members = [native_bands(θ) for θ in naive_samples]
    naive_central = native_bands(lin_params)
    con_members   = [native_bands(θ) for θ in con_samples]
    con_central   = native_bands(θ_mean)
end
@printf("  done in %.1f min\n", t/60)
@printf("  naive samples    : min ω ∈ [%+.3f, %+.3f] THz — %d/%d unstable\n",
        minimum(m.min_stable for m in naive_members), maximum(m.min_stable for m in naive_members),
        count(m -> m.min_stable < unstable_tol, naive_members), length(naive_members))
@printf("  constrained (rej): min ω ∈ [%+.3f, %+.3f] THz — %d/%d unstable\n",
        minimum(m.min_stable for m in con_members), maximum(m.min_stable for m in con_members),
        count(m -> m.min_stable < unstable_tol, con_members), length(con_members))

# ── shared x-axis (reference) + shared y-limits ─────────────────────────────
xref = naive_central.x_vals; xticks = naive_central.x_ticks; labels = naive_central.labels; Np = naive_central.Np
allF = vcat([m.F for m in naive_members], [m.F for m in con_members], [naive_central.F, con_central.F])
ylo  = minimum(minimum.(allF)); yhi = maximum(maximum.(allF)); pad = 0.05*(yhi-ylo)
ylims = (ylo-pad, yhi+pad)
@printf("Shared frequency axis: [%.2f, %.2f] THz\n", ylims...)

# ── plot (exact plot_committee_bands styling; PDF + hi-DPI PNG) ──────────────
function plot_native(members, central, title, stem)
    fig = Figure(size=(340,300), figure_padding=(6,10,4,6))
    ax = Axis(fig[1,1]; xlabel="Wave vector", ylabel="Frequency (THz)", title=title,
              titlesize=11, xlabelsize=11, ylabelsize=11, xticklabelsize=10, yticklabelsize=10,
              xticks=(xticks, labels), xgridvisible=false, ygridvisible=false,
              xtickalign=1, ytickalign=1, xticksize=4, yticksize=4)
    for m in members
        col = m.min_stable < unstable_tol ? RGBAf(0.80,0.15,0.15,0.45) : RGBAf(0.45,0.45,0.45,0.30)
        for b in 1:3Np; lines!(ax, xref, m.F[b, :]; color=col, linewidth=0.7); end
    end
    for b in 1:3Np; lines!(ax, xref, central.F[b, :]; color=RGBf(0.0,0.447,0.698), linewidth=1.6); end
    hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=0.8)
    vlines!(ax, xticks; color=(:black,0.22), linewidth=0.6)
    xlims!(ax, first(xref), last(xref)); ylims!(ax, ylims...)
    save("$stem.pdf", fig); save("$stem.png", fig; px_per_unit=4)
    @printf("  saved %s.{pdf,png}\n", basename(stem))
end

plot_native(naive_members, naive_central, "$(result.name) — naive hypercube (individual Hessians)", "$outdir/bands_naive_individual_hessian")
plot_native(con_members,   con_central,   "$(result.name) — constrained rejection (individual Hessians)", "$outdir/bands_constrained_individual_hessian")

ACEpotentials.Models.set_linear_parameters!(model, lin_params)
println("\n══ DONE ══  22 native Hessians, shared y-axis $(round.(ylims;digits=2)) THz → $outdir/")
