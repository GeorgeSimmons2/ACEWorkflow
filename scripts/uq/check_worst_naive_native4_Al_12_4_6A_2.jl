# check_worst_naive_native4_Al_12_4_6A_2.jl
#
# Find a genuinely-unstable NAIVE POPS HYPERCUBE SAMPLE, validated with a NATIVE
# Hessian on its OWN relaxed 4×4×4 supercell (the undotted band-path check runs at a
# fixed reference geometry, which is invalid for unpinned naive vectors, and the 3×3×3
# cell is too small for the 6 Å cutoff — both fixed here).
#
# Draw samples from the naive hypercube (box fitted to the naive delta forest, no
# predicate), check each with precompute_force_constants on its own relaxed 4×4×4 cell,
# and SAVE THE FIRST sample that comes back unstable (min band ω < -0.05).  That θ is
# then loaded by the NPT run.
#
# Uniform N_per_seg=40 (works against main's scalar fcc_band_path).
#
# Run:  julia --project -t <N> scripts/uq/check_worst_naive_native4_Al_12_4_6A_2.jl

include(joinpath(@__DIR__, "..", "bandpath_phonon_uq", "lib.jl"))
using Random
Random.seed!(1234)

element = :Al; dataset = ""; qΓtol = 5e-2
N_val = 4; N_per_seg = 40; unstable_tol = -0.05
n_draw = 40                                   # hypercube samples to draw; we stop at the first bad one
result = load_model(element, 12, 4, 6, 2; dataset_name=dataset)
model  = result.model; lin_params = result.lin_params; n_params = length(lin_params)
P = result.P; Ap = Diagonal(result.W)*result.A/P; Yw = result.W.*result.Y; λ = 1.0/size(Ap,1)
outdir = "$(result.dir)/results/npt_worst_naive_native4"; mkpath(outdir)
structure = AtomsBuilder.Chemistry.symmetry(element)

ACEpotentials.Models.set_linear_parameters!(model, lin_params)
a_ref = ACEWorkflow.relax_lattice_constant(model, element)
@printf("a_ref = %.5f Å.  Drawing naive hypercube samples; validating natively at N_cell=%d.\n", a_ref, N_val)

# ── naive hypercube: box fitted to the naive delta forest (seed-1234 draw) ────
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
                                     number_of_committee_members=n_draw, max_attempts=1_000_000)
samples = [smat[:, i] for i in 1:size(smat, 2)]

# native min band ω at the sample's own relaxed geometry, 4×4×4, from scratch
function native_minω_relaxed(θ)
    ACEpotentials.Models.set_linear_parameters!(model, θ)
    a = try
        aa = ACEWorkflow.relax_lattice_constant(model, element)
        (0.9a_ref < aa < 1.1a_ref) ? aa : a_ref
    catch; a_ref end
    sp, ss = bulk_prim_super(element; a=a, N_cell=N_val)
    fc = precompute_force_constants(sp, ss, model)
    ql, _, _, _, _ = _band_path(structure, fc.L; N_per_seg=N_per_seg); qn = norm.(ql)
    m = Inf
    for (iq, q) in enumerate(ql)
        qn[iq] < qΓtol && continue
        ev = eigvals(Hermitian(dynamical_matrix_from_fc(fc, q)))
        m = min(m, minimum(sign.(ev).*sqrt.(abs.(ev)).*FREQ_THz))
    end
    (minω=m, a=a)
end

println("\n── scanning naive hypercube samples for the first unstable one (native, relaxed, 4×4×4) ──")
chosen = nothing
for s in 1:length(samples)
    global chosen
    r = native_minω_relaxed(samples[s])
    genuine = r.minω < unstable_tol
    @printf("  sample %2d: native N4 relaxed min ω = %+7.3f THz (a=%.4f)%s\n",
            s, r.minω, r.a, genuine ? "  ← UNSTABLE, taking this one" : "")
    if genuine
        chosen = (θ=samples[s], idx=s, minω=r.minω, a=r.a)
        break
    end
end

if chosen === nothing
    error("None of the $n_draw naive hypercube samples was unstable at native relaxed 4×4×4 — raise n_draw.")
end
@printf("\n✓ FIRST bad naive hypercube sample: draw #%d\n", chosen.idx)
@printf("  native relaxed 4×4×4 min ω = %+.3f THz,  own a₀ = %.5f Å\n", chosen.minω, chosen.a)
writedlm("$outdir/theta_worst_naive_native4.csv", chosen.θ, ',')
writedlm("$outdir/worst_naive_native4_info.csv",
         ["sample_idx" "native_N4_minomega_THz" "a0_Ang"; chosen.idx chosen.minω chosen.a], ',')
println("  saved θ → $outdir/theta_worst_naive_native4.csv  (load this in the NPT run)")
ACEpotentials.Models.set_linear_parameters!(model, lin_params)
