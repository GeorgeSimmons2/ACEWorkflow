# bands_one_from_each_Al_12_4_6A_2.jl
#
# One member from each hypercube, individual (native, cache-free) Hessians, overlaid:
#   • naive hypercube (built from the naive POPS delta forest, NO predicate) — the
#     MOST UNSTABLE sample  → crimson, dips below 0.
#   • rejection-sampled constrained hypercube — the SOFTEST member (fairest: worst
#     case of the constrained committee) → blue, stays positive.
#
# Members are ranked with the (now cache-consistent, 1e-13-validated) undotted check
# for speed; the two chosen members' final band structures are computed from scratch
# with precompute_force_constants at each member's own relaxed geometry.
#
# Run:  julia --project -t <N> scripts/uq/bands_one_from_each_Al_12_4_6A_2.jl

include(joinpath(@__DIR__, "..", "bandpath_phonon_uq", "lib.jl"))
using Random
Random.seed!(1234)

element = :Al; N_cell = 3; N_per_seg = 20; qΓtol = 5e-2; n_naive_samples = 60
result = load_model(element, 12, 4, 6, 2; dataset_name="")
model  = result.model; lin_params = result.lin_params; n_params = length(lin_params)
P = result.P; Ap = Diagonal(result.W)*result.A/P; Yw = result.W.*result.Y; λ = 1.0/size(Ap,1)
cdir   = "$(result.dir)/results/bandpath_undotted"
outdir = "$(result.dir)/results/bands_individual_hessian"; mkpath(outdir)
structure = AtomsBuilder.Chemistry.symmetry(element)

ACEpotentials.Models.set_linear_parameters!(model, lin_params)
a_ref = ACEWorkflow.relax_lattice_constant(model, element)
bp = bandpath_Dk(result, model, element, a_ref, N_cell; N_per_seg=N_per_seg)   # fast undotted ranking

# native individual-Hessian bands at the member's own relaxed geometry (guarded)
function native_bands(θ)
    ACEpotentials.Models.set_linear_parameters!(model, θ)
    a = try
        aa = ACEWorkflow.relax_lattice_constant(model, element)
        (0.9a_ref < aa < 1.1a_ref) ? aa : a_ref
    catch; a_ref end
    sp, ss = bulk_prim_super(element; a=a, N_cell=N_cell)
    fc = precompute_force_constants(sp, ss, model)
    ql, xv, xt, lb, _ = _band_path(structure, fc.L; N_per_seg=N_per_seg); qn = norm.(ql); Np = fc.Np
    F = Matrix{Float64}(undef, 3Np, length(ql))
    for (iq, q) in enumerate(ql)
        ev = eigvals(Hermitian(dynamical_matrix_from_fc(fc, q))); F[:, iq] = sign.(ev).*sqrt.(abs.(ev)).*FREQ_THz
    end
    (F=F, x_vals=xv, x_ticks=xt, labels=lb, Np=Np, min_stable=minimum(F[:, qn .> qΓtol]), a=a)
end

# ── naive hypercube from the naive POPS delta forest ─────────────────────────
C = Symmetric(Ap'*Ap .+ λ.*(P'*P)); Cf = cholesky(C); AtX = Cf\Matrix(Ap'); θ̃ = Cf\(Ap'*Yw)
leverage = vec(sum(Ap'.*AtX; dims=1)); residual = Yw .- Ap*θ̃
forest_member(i) = lin_params .+ (P \ (AtX[:, i] .* (residual[i]/leverage[i])))
lev_idx = sortperm(leverage; rev=true)[1:5]
res_idx = Int[]; for i in sortperm(abs.(residual); rev=true); i in lev_idx && continue; push!(res_idx,i); length(res_idx)==10 && break; end
taken = Set(vcat(lev_idx,res_idx)); rand_idx = Int[]
while length(rand_idx) < 15; i = rand(1:length(Yw)); (i in taken) && continue; push!(rand_idx,i); push!(taken,i); end
naive_forest = [forest_member(i) for i in vcat(lev_idx,res_idx,rand_idx)]
naive_deltas = reduce(hcat, naive_forest)' .- lin_params'
hyp_eig, hyp_bound = hypercube(Matrix(naive_deltas))
smat, _ = rejection_sample_hypercube(hyp_eig, hyp_bound, lin_params, θ->true;
                                     number_of_committee_members=n_naive_samples, max_attempts=1_000_000)
naive_samples = [smat[:, i] for i in 1:size(smat, 2)]
θ_naive = naive_samples[argmin(min_freq_stable(θ, bp) for θ in naive_samples)]

# ── rejection committee: softest member ─────────────────────────────────────
rej = [collect(Float64, r) for r in eachrow(readdlm("$cdir/committee_rejection.csv", ','))]
θ_rej = rej[argmin(min_freq_stable(θ, bp) for θ in rej)]

nb = native_bands(θ_naive); rb = native_bands(θ_rej)
@printf("naive-hypercube pick : native min ω = %+.3f THz (a=%.4f)\n", nb.min_stable, nb.a)
@printf("rejection    pick    : native min ω = %+.3f THz (a=%.4f)\n", rb.min_stable, rb.a)
writedlm("$outdir/one_from_each_theta_naive.csv", θ_naive, ',')
writedlm("$outdir/one_from_each_theta_rejection.csv", θ_rej, ',')

# ── overlay plot (bands_naive.pdf styling) ──────────────────────────────────
xref = nb.x_vals; Np = nb.Np
ylo = min(minimum(nb.F), minimum(rb.F)); yhi = max(maximum(nb.F), maximum(rb.F)); pad = 0.05*(yhi-ylo)
fig = Figure(size=(380,320), figure_padding=(6,10,4,6))
ax = Axis(fig[1,1]; xlabel="Wave vector", ylabel="Frequency (THz)",
          title="$(result.name) — naive hypercube vs rejection (individual Hessians)",
          titlesize=10, xlabelsize=11, ylabelsize=11, xticklabelsize=10, yticklabelsize=10,
          xticks=(nb.x_ticks, nb.labels), xgridvisible=false, ygridvisible=false,
          xtickalign=1, ytickalign=1, xticksize=4, yticksize=4)
for b in 1:3Np; lines!(ax, xref, nb.F[b, :]; color=RGBAf(0.80,0.15,0.15,0.9), linewidth=1.3); end
for b in 1:3Np; lines!(ax, xref, rb.F[b, :]; color=RGBf(0.0,0.447,0.698), linewidth=1.3); end
hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=0.8)
vlines!(ax, nb.x_ticks; color=(:black,0.22), linewidth=0.6)
xlims!(ax, first(xref), last(xref)); ylims!(ax, ylo-pad, yhi+pad)
elem = [LineElement(color=RGBAf(0.80,0.15,0.15,0.9)), LineElement(color=RGBf(0.0,0.447,0.698))]
Legend(fig[1,1], elem, ["naive hypercube  (min ω $(round(nb.min_stable;digits=2)) THz)",
                        "rejection sampled  (min ω $(round(rb.min_stable;digits=2)) THz)"];
       tellwidth=false, tellheight=false, halign=:left, valign=:bottom, margin=(8,8,8,8),
       framevisible=true, labelsize=8, patchsize=(16,10))
save("$outdir/bands_one_from_each.pdf", fig); save("$outdir/bands_one_from_each.png", fig; px_per_unit=4)
println("Saved → $outdir/bands_one_from_each.{pdf,png}")
