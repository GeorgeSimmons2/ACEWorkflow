# bz_stability_rejection_sampling.jl
#
# Rejection sampling of the constrained-POPS hypercube conditioned on phonon
# stability ACROSS the Brillouin zone — no percentile clipping.
#
# The trick (linear-in-θ): the supercell Hessian is linear in the ACE
# parameters, H(θ) = Σ_k θ_k H_k, and the spectrum of the mass-weighted PBC
# Hessian of an N×N×N supercell is exactly the union of ω²(q) over every q
# commensurate with that supercell (zone folding).  So ONE symmetric
# eigendecomposition checks a whole q-mesh, and the per-proposal cost is
#     D(θ) = reshape(D_flat * θ, 3N, 3N);   eigvals(Symmetric(D), 1:4)
# i.e. a GEMV + a partial eigendecomposition (~ms), after a one-off precompute
# of n_params analytic supercell Hessians (serialized for reuse).
#
# Notes:
#   - The folded check is EXACT at the commensurate q for the periodic
#     supercell — no 2×rcut requirement, because the force constants are never
#     unfolded.  N_cell controls the mesh density (4·N_cell³ primitive-BZ
#     q-points for FCC).
#   - The basis is built at the constrained-mean geometry a_ref.  Committee
#     members relax to slightly different a; the positive frequency margin
#     f_margin_THz absorbs that.  Members passing with a thin margin should be
#     re-verified with full bands (phonon_committee_rejection.jl).
#   - The lowest 3 eigenvalues are the acoustic Γ zeros (rigid translations);
#     the margin is applied from the 4th eigenvalue up.
#
# Run AFTER elastic_constraints_rejection.jl in the same session to reuse
# constrained_pops_delta and the elastic inequality rows; standalone it falls
# back to the saved committee CSV as (coarser) proposal and checks BZ
# stability only.

using LinearAlgebra, DelimitedFiles, Statistics, Printf, Serialization
using AtomsBuilder, Unitful
using AtomsCalculatorsUtilities.SitePotentials: hessian
using ACEpotentials, ACEWorkflow

element      = :Al
N_cell       = 3      # supercell → checks the 4·N_cell³-point commensurate q-mesh
f_margin_THz = 0.1    # required ω at every checked q ≠ Γ (absorbs δa geometry error)
N_committee  = 50

# ── Model + constrained mean ─────────────────────────────────────────────────
result   = load_model(element, 14, 4, 6, 2; dataset_name="subset_20_percent")
model    = result.model
n_params = length(result.lin_params)

constrained_params = if @isdefined(constrained_ridge_teta)
    constrained_ridge_teta
else
    con_model, _ = ACEpotentials.load_model("$(result.dir)/constrained_model.json")
    vec(vcat(con_model.ps[1], con_model.ps[2]))
end

ACEpotentials.Models.set_linear_parameters!(model, constrained_params)
a_ref     = ACEWorkflow.relax_lattice_constant(model, element)
sys_super = bulk(element; a=a_ref*u"Å", cubic=true) * (N_cell, N_cell, N_cell)
N3        = 3 * length(sys_super)
m_amu     = ustrip(sys_super[1].mass)
@printf("a_ref = %.6f Å,  supercell: %d atoms → %d modes (= %d q-points × 3 branches)\n",
        a_ref, length(sys_super), N3, 4 * N_cell^3)

# ── Mass-weighted Hessian basis: D_flat[:, k] = vec(H_k) / m  (eV/Å²/amu) ────
basis_file = "$(result.dir)/results/bz_hessian_basis_$(N_cell)x$(N_cell)x$(N_cell).jls"
D_flat = if isfile(basis_file)
    cached = deserialize(basis_file)
    abs(cached.a_ref - a_ref) < 1e-4 ||
        error("cached basis built at a=$(cached.a_ref) ≠ current a_ref=$a_ref — delete $basis_file to rebuild")
    println("Loaded cached Hessian basis: $basis_file")
    cached.D_flat
else
    println("Building supercell Hessian basis ($n_params basis functions) …")
    Df  = zeros(N3 * N3, n_params)
    e_k = zeros(n_params)
    for k in 1:n_params
        fill!(e_k, 0.0); e_k[k] = 1.0
        ACEpotentials.Models.set_linear_parameters!(model, e_k)
        Df[:, k] = vec(ustrip.(hessian(sys_super, model))) ./ m_amu
        k % 10 == 0 && print("\r  $k / $n_params …")
    end
    println("\r  done.                    ")
    ACEpotentials.Models.set_linear_parameters!(model, constrained_params)
    serialize(basis_file, (D_flat=Df, a_ref=a_ref, N_cell=N_cell))
    println("Serialized → $basis_file")
    Df
end

# ── Sanity check: contracted basis reproduces the mean-model spectrum ────────
ω2_mean = eigvals(Symmetric(reshape(D_flat * constrained_params, N3, N3)))
f_mean  = sign.(ω2_mean) .* sqrt.(abs.(ω2_mean)) .* FREQ_THz
@printf("Constrained mean on mesh:  ω ∈ [%.4f, %.4f] THz  (acoustic zeros: %.2e, %.2e, %.2e)\n",
        f_mean[4], f_mean[end], f_mean[1], f_mean[2], f_mean[3])
f_mean[4] > f_margin_THz ||
    @warn "Constrained mean itself violates the margin — acceptance will be ~0. Lower f_margin_THz."

# ── Feasibility predicate: cheap linear rows first, then BZ stability ────────
ω2_margin = (f_margin_THz / FREQ_THz)^2
ω2_scale  = maximum(abs.(ω2_mean))

function bz_stable(c)
    D  = Symmetric(reshape(D_flat * c, N3, N3))
    ω2 = eigvals(D, 1:4)                     # 4 smallest suffice
    return ω2[1] > -1e-8 * ω2_scale && ω2[4] >= ω2_margin
end

have_linear = @isdefined(ineq_constraint_matrix)
is_feasible = if have_linear
    println("Elastic inequality rows found in session — combining with BZ check.")
    c -> begin
        Ac = ineq_constraint_matrix * c
        (all(ineq_bounds[1] .<= Ac) && all(Ac .<= ineq_bounds[2])) || return false
        bz_stable(c)
    end
else
    println("No elastic inequality rows in session — BZ stability check only.")
    bz_stable
end

# ── Proposal: unclipped hypercube over the constrained-POPS cloud ────────────
proposal_cloud = if @isdefined(constrained_pops_delta)
    println("Using in-session constrained_pops_delta as proposal cloud.")
    constrained_pops_delta
else
    f = "$(result.dir)/constrained_pops_samples.csv"
    @warn "constrained_pops_delta not in session — building proposal from $f (coarser)."
    m = readdlm(f, ',')
    size(m, 1) == n_params ? Matrix(m') : m      # orient rows = points
end

pops_eig, pops_bound = hypercube(proposal_cloud)   # percentile_clipping = 0 → unclipped

# ── Rejection sampling ───────────────────────────────────────────────────────
committee_mat, _ = rejection_sample_hypercube(pops_eig, pops_bound, zeros(n_params),
                                              is_feasible;
                                              number_of_committee_members=N_committee)
committee = [committee_mat[:, i] for i in 1:size(committee_mat, 2)]

# ── Report the stability margin of the accepted ensemble ─────────────────────
f_min = [ (ω2 = eigvals(Symmetric(reshape(D_flat * c, N3, N3)), 1:4);
           sign(ω2[4]) * sqrt(abs(ω2[4])) * FREQ_THz) for c in committee ]
@printf("Accepted committee: min non-acoustic ω on mesh ∈ [%.4f, %.4f] THz (margin %.2f)\n",
        minimum(f_min), maximum(f_min), f_margin_THz)

out_file = "$(result.dir)/bz_stable_rejection_sampled_pops_samples.csv"
writedlm(out_file, committee, ',')
println("Saved committee (one member per row) → $out_file")
println("Verify with full bands: point phonon_committee_rejection.jl at this CSV.")
