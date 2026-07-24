# bz_stable_hypercube_committee_Al_20_4_6A_2_subset_50.jl
#
# POPS hypercube committee with NO negative phonon eigenmodes — constrained
# directly, not via proxies (Born rows, rattled orderings, curvature rows).
#
# The trick (from bz_stability_rejection_sampling.jl): the supercell Hessian
# is linear in the ACE parameters, H(θ) = Σ_k θ_k H_k, and the spectrum of the
# mass-weighted PBC Hessian of an N×N×N supercell is exactly the union of
# ω²(q) over every q commensurate with it (zone folding).  So ONE symmetric
# partial eigendecomposition checks a whole q-mesh:
#
#     D(θ) = reshape(D_flat * θ, N3, N3);   eigvals(Symmetric(D), 1:4)
#
# ~ms per proposal after a ONE-OFF precompute of n_params supercell Hessians
# (serialized — subsequent runs load it in seconds).  Rejection sampling the
# unclipped POPS hypercube against this predicate gives a committee whose
# every member is phonon-stable on the full folded mesh BY CONSTRUCTION.
#
# Predicate ordering (cheap → expensive):
#   1. Born rows (q→0 limit: C44, C11−C12, C11+2C12, b″) — 4 dot products
#   2. |δa_IFT| ≤ ift_da_max — draws that relax far from the reference
#      geometry can't be certified there, so they are discarded
#   3. folded-mesh eigencheck with a THz margin (absorbs the small per-member
#      relaxed-geometry shift the margin exists for)
# Accepted members are then VERIFIED with full phonon bands at each member's
# own exactly-relaxed lattice constant (phonon_committee from src/Phonons).
#
# Outputs (to models/Al_20_4_6A_2_subset_50_percent/results/):
#   bz_hessian_basis_<N>x<N>x<N>.jls            (one-off, reused across runs)
#   bz_stable_hypercube_committee_10.csv        (one member per row)
#   bz_stable_hypercube_phonon_committee_*      (plots + freq CSVs)
#
# Run:  julia --project [-t N] scripts/uq/bz_stable_hypercube_committee_Al_20_4_6A_2_subset_50.jl

using LinearAlgebra, DelimitedFiles, Statistics, Printf, Random, Serialization
using StaticArrays, Unitful, ForwardDiff
using ACEpotentials, ACEWorkflow
using AtomsBuilder
using AtomsCalculatorsUtilities.SitePotentials: hessian
import ACEWorkflow: phonon_committee

Random.seed!(1234)

element      = :Al
N_cell       = 3      # mesh supercell → certifies the 4·N³-point commensurate q-mesh
f_margin_THz = 0.2    # required ω at every non-acoustic mesh mode (absorbs δa shift)
ift_da_max   = 0.1    # Å — reject draws whose minimum moves beyond certification range
N_committee  = 10

# ── Model + forest (reuse in-session objects when present) ───────────────────
if !(@isdefined(result)) || result.name != "Al_20_4_6A_2"
    result = load_model(element, 20, 4, 6, 2; dataset_name="subset_50_percent")
end
model      = result.model
lin_params = result.lin_params
n_params   = length(lin_params)
P          = result.P
println("Model $(result.name): $n_params parameters")

if !(@isdefined(pops_corr)) || size(pops_corr, 2) != n_params
    println("Computing POPS corrections (delta forest) …")
    Ap = Diagonal(result.W) * result.A / P
    Yw = result.W .* result.Y
    pops_corr = corrections(Ap, Yw, P; leverage_percentile=0.0)
end
println("Forest: $(size(pops_corr, 1)) members")

# ── Reference geometry + cheap pre-filter rows ───────────────────────────────
println("Relaxing mean model …")
a_eq = ACEWorkflow.relax_lattice_constant(model, element)
@printf("  a_eq = %.6f Å\n", a_eq)

println("Building strain-Hessian basis for the Born pre-filter …")
H_basis = elastic_hessian_basis(model; element=element, a=a_eq)
constraint_1 = H_basis[1, 1, :]
constraint_2 = H_basis[1, 2, :]
constraint_4 = H_basis[4, 4, :]

println("Building b′, b″ …")
function lattice_basis(a_val)
    sys = ACEWorkflow.Elasticity.reference_system(element; a=a_val)
    ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(sys, model))
end
b_prime        = ForwardDiff.derivative(lattice_basis, a_eq)
b_double_prime = ForwardDiff.derivative(a -> ForwardDiff.derivative(lattice_basis, a), a_eq)

born_rows  = vcat(constraint_4',                        # C44        ≥ 0.1
                  (constraint_1 .- constraint_2)',      # C11 − C12  ≥ 1.0
                  (constraint_1 .+ 2 .* constraint_2)', # C11 + 2C12 ≥ 0.1
                  b_double_prime')                      # b″·θ       ≥ 1e-9
born_lower = [0.1, 1.0, 0.1, 1e-9]

# ── Mass-weighted supercell Hessian basis (one-off, serialized) ──────────────
sys_super = bulk(element; a=a_eq*u"Å", cubic=true) * (N_cell, N_cell, N_cell)
N3        = 3 * length(sys_super)
m_amu     = ustrip(sys_super[1].mass)
@printf("Mesh supercell: %d atoms → %d modes (= %d commensurate q-points × 3 branches)\n",
        length(sys_super), N3, 4 * N_cell^3)

basis_file = "$(result.dir)/results/bz_hessian_basis_$(N_cell)x$(N_cell)x$(N_cell).jls"
D_flat = if isfile(basis_file)
    cached = deserialize(basis_file)
    abs(cached.a_ref - a_eq) < 1e-4 ||
        error("cached basis built at a=$(cached.a_ref) ≠ current a_eq=$a_eq — delete $basis_file to rebuild")
    println("Loaded cached Hessian basis: $basis_file")
    cached.D_flat
else
    println("Building supercell Hessian basis ($n_params basis functions — one-off, serialized) …")
    Df  = zeros(N3 * N3, n_params)
    e_k = zeros(n_params)
    t0  = time()
    for k in 1:n_params
        fill!(e_k, 0.0); e_k[k] = 1.0
        ACEpotentials.Models.set_linear_parameters!(model, e_k)
        Df[:, k] = vec(ustrip.(hessian(sys_super, model))) ./ m_amu
        if k % 10 == 0
            eta = (time() - t0) / k * (n_params - k) / 60
            @printf("\r  %d / %d  (ETA %.0f min) …", k, n_params, eta)
        end
    end
    println("\r  done in $(round((time() - t0)/60; digits=1)) min.            ")
    ACEpotentials.Models.set_linear_parameters!(model, lin_params)
    serialize(basis_file, (D_flat=Df, a_ref=a_eq, N_cell=N_cell))
    println("Serialized → $basis_file")
    Df
end

# ── Sanity: the mean model must clear the margin ─────────────────────────────
ω2_mean = eigvals(Symmetric(reshape(D_flat * lin_params, N3, N3)))
f_mean  = sign.(ω2_mean) .* sqrt.(abs.(ω2_mean)) .* FREQ_THz
@printf("Mean model on mesh:  ω ∈ [%.4f, %.4f] THz  (acoustic zeros: %.1e %.1e %.1e)\n",
        f_mean[4], f_mean[end], f_mean[1], f_mean[2], f_mean[3])
f_mean[4] > f_margin_THz ||
    error("mean model violates the margin ($(f_mean[4]) < $f_margin_THz THz) — acceptance would be ~0")

# ── Feasibility predicate: Born rows → δa trust → folded-mesh eigencheck ─────
ω2_margin = (f_margin_THz / FREQ_THz)^2
ω2_scale  = maximum(abs.(ω2_mean))
K_ref     = dot(lin_params, b_double_prime)

n_checked = Ref(0); n_pass_born = Ref(0); n_pass_da = Ref(0)
function is_feasible(θ)
    n_checked[] += 1
    all(born_lower .<= born_rows * θ) || return false
    n_pass_born[] += 1
    δθ = θ .- lin_params
    abs(-dot(b_prime, δθ) / K_ref) <= ift_da_max || return false
    n_pass_da[] += 1
    D  = Symmetric(reshape(D_flat * θ, N3, N3))
    ω2 = eigvals(D, 1:4)                      # 3 acoustic zeros + lowest real mode
    return ω2[1] > -1e-8 * ω2_scale && ω2[4] >= ω2_margin
end

# ── Rejection-sample the unclipped POPS hypercube ────────────────────────────
println("Rejection sampling the unclipped hypercube against the folded-mesh PSD predicate …")
hyp_eig, hyp_bound = hypercube(pops_corr)     # unclipped, raw POPS corrections
committee_mat, _ = rejection_sample_hypercube(hyp_eig, hyp_bound, lin_params,
                                              is_feasible;
                                              number_of_committee_members=N_committee,
                                              max_attempts=2_000_000)
@printf("Predicate funnel: %d checked → %d past Born → %d past |δa| → %d accepted\n",
        n_checked[], n_pass_born[], n_pass_da[], N_committee)

writedlm("$(result.dir)/results/bz_stable_hypercube_committee_$(N_committee).csv", committee_mat', ',')
committee = [committee_mat[:, i] for i in 1:size(committee_mat, 2)]

# ── Report the mesh margin of the accepted ensemble ──────────────────────────
println("\n── Accepted committee: folded-mesh margins ─────────────────")
for (k, θk) in enumerate(committee)
    ω2k = eigvals(Symmetric(reshape(D_flat * θk, N3, N3)), 1:4)
    @printf("  member %2d:  lowest non-acoustic ω = %.4f THz\n",
            k, sign(ω2k[4]) * sqrt(abs(ω2k[4])) * FREQ_THz)
end

# ── VERIFY: full phonon bands at each member's own relaxed geometry ──────────
println("\n── Full-band verification (each member exactly relaxed) ────")
ACEpotentials.Models.set_linear_parameters!(model, lin_params)
_, bands, _, _ = phonon_committee(model, committee, result, element;
                                  N_cell=N_cell, file_prefix="bz_stable_hypercube_")
println()
n_unstable = 0
for i in 1:length(committee)
    fmin = minimum(bands[i + 1])
    bad  = fmin < -0.05
    global n_unstable += bad
    @printf("  member %2d:  min band frequency = %+8.4f THz  %s\n", i, fmin, bad ? "✗ UNSTABLE" : "✓")
end
@printf("\nCommittee members with imaginary modes: %d / %d\n", n_unstable, length(committee))
n_unstable == 0 && println("All members phonon-stable — the folded-mesh constraint held on the full path.")

println("\nSaved:")
println("  $basis_file")
println("  $(result.dir)/results/bz_stable_hypercube_committee_$(N_committee).csv")
println("  $(result.dir)/results/bz_stable_hypercube_phonon_committee_* (plots + freq CSVs)")
