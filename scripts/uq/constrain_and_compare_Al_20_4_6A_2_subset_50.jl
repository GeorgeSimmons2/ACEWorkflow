# constrain_and_compare_Al_20_4_6A_2_subset_50.jl   (v2 — exact QP solves)
#
# Full before/after pipeline for Al_20_4_6A_2 (subset_50_percent), whose POPS
# delta forest contains genuinely Born-unstable members (dominantly C44 < 0):
#
#   1. UNCONSTRAINED baseline: full delta forest + N_check draws from the
#      unclipped hypercube — IFT Born check, elastic ranges.
#   2. Constrained ridge fit — solved EXACTLY by active-set enumeration.
#   3. Selectively constrained POPS: closed-form pointwise solutions checked
#      against the inequality rows; violators repaired EXACTLY (same solver).
#   4. Unclipped hypercube over the repaired forest, CENTERED on the cloud
#      mean — no rejection.
#   5. Rejection sampling against the inequality rows (the honest committee).
#   6. Summary table + overlaid Born-margin histograms + CSVs.
#
# v2 changes (why OSQP was removed):
#   With λ = 1/n ≈ 1.3e-5 the QP Hessian has near-flat directions of
#   curvature ~λ·min(P²); OSQP at eps 1e-4–5e-4 "converges" while the
#   solution wanders by up to eps/λ ~ O(10) along them.  Those wandering
#   solutions barely move the constraint values but blow up the hypercube
#   bounds (the v1 run produced committee members with ‖θ‖ ≈ 2–4× ‖θ̄‖ and
#   min margins of −1000 GPa).  With only 1 equality + 4 inequalities the
#   KKT system can instead be solved exactly: enumerate the ≤2⁴ active sets
#   against a prefactorized Cholesky of the QP Hessian.  No tolerances.
#   Sampling is also now centered: the hypercube is built on deltas about the
#   cloud mean and samples are reconstructed about that mean, so no component
#   of the mean is lost to the eigen-mask.
#
# Outputs (to models/Al_20_4_6A_2_subset_50_percent/results/):
#   constrain_compare_born_margins.png
#   constrain_compare_summary.csv
#   constrained_ridge_params.csv
#   rejection_sampled_committee.csv        (N_committee members, one per row)
#
# Run:  julia --project scripts/uq/constrain_and_compare_Al_20_4_6A_2_subset_50.jl

using LinearAlgebra, DelimitedFiles, Statistics, Printf, Random
using ForwardDiff, Unitful, CairoMakie
using ACEpotentials, ACEWorkflow

Random.seed!(1234)

element     = :Al
N_check     = 10_000   # draws per ensemble for the Born statistics
N_committee = 50       # committee members saved for downstream use

# ── Load model ────────────────────────────────────────────────────────────────
result     = load_model(element, 20, 4, 6, 2; dataset_name="subset_50_percent")
model      = result.model
lin_params = result.lin_params
n_params   = length(lin_params)
P          = result.P
println("Model $(result.name): $n_params parameters, $(length(result.Y)) design rows")

Ap = Diagonal(result.W) * result.A / P
Yw = result.W .* result.Y

# ── θ-independent elastic / IFT ingredients ──────────────────────────────────
println("Relaxing mean model …")
a_eq = ACEWorkflow.relax_lattice_constant(model, element)
@printf("  a_eq = %.6f Å\n", a_eq)

sys0 = ACEWorkflow.Elasticity.reference_system(element; a=a_eq)
L0   = ustrip.(ACEWorkflow.Elasticity.lattice_matrix(sys0.cell.cell_vectors))
V0   = abs(det(L0))
eV_to_GPa = 160.2176621 / V0

println("Building strain-Hessian basis (6×6×$n_params) …")
H_basis = elastic_hessian_basis(model; element=element, a=a_eq)

println("Building dH/da (central FD) …")
dH_da = ACEWorkflow.Elasticity.strain_hessian_lattice_constant_derivative(model, element; a=a_eq)(a_eq)

println("Building b′, b″, b‴ …")
function lattice_basis(a_val)
    sys = ACEWorkflow.Elasticity.reference_system(element; a=a_val)
    ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(sys, model))
end
b_prime        = ForwardDiff.derivative(lattice_basis, a_eq)
b_double_prime = ForwardDiff.derivative(a -> ForwardDiff.derivative(lattice_basis, a), a_eq)
b_triple_prime = ForwardDiff.derivative(
                     a -> ForwardDiff.derivative(
                         a2 -> ForwardDiff.derivative(lattice_basis, a2), a), a_eq)

M1 = reshape(H_basis, 36, n_params)
M2 = reshape(dH_da,   36, n_params)
c11_0, c11_a = M1[1, :],  M2[1, :]
c12_0, c12_a = M1[7, :],  M2[7, :]
c44_0, c44_a = M1[22, :], M2[22, :]

@printf("Mean model:  C11 = %.2f  C12 = %.2f  C44 = %.2f  GPa\n",
        dot(c11_0, lin_params) * eV_to_GPa,
        dot(c12_0, lin_params) * eV_to_GPa,
        dot(c44_0, lin_params) * eV_to_GPa)

# ── IFT Born check for a matrix of members (columns) around θ_ref ────────────
# Members whose IFT lattice shift exceeds ift_da_max are outside the validity
# of the 2nd-order expansion; they are counted and reported, not trusted.
function born_stats(label, members::AbstractMatrix, θ_ref; ift_da_max=0.1)
    K  = dot(θ_ref, b_double_prime)
    s3 = dot(θ_ref, b_triple_prime)
    N  = size(members, 2)
    C11 = Vector{Float64}(undef, N); C12 = similar(C11); C44 = similar(C11)
    δa_all = Vector{Float64}(undef, N)
    for s in 1:N
        θ   = view(members, :, s)
        δθ  = θ .- θ_ref
        δa1 = -dot(b_prime, δθ) / K
        δa  = δa1 - (dot(b_double_prime, δθ) * δa1 + 0.5 * s3 * δa1^2) / K
        δa_all[s] = δa
        C11[s] = (dot(c11_0, θ) + δa * dot(c11_a, θ)) * eV_to_GPa
        C12[s] = (dot(c12_0, θ) + δa * dot(c12_a, θ)) * eV_to_GPa
        C44[s] = (dot(c44_0, θ) + δa * dot(c44_a, θ)) * eV_to_GPa
    end
    shear  = C11 .- C12
    bulkm  = C11 .+ 2 .* C12
    stable = (shear .> 0) .& (bulkm .> 0) .& (C44 .> 0)
    n_oor  = count(abs.(δa_all) .> ift_da_max)
    dnorm  = [norm(view(members, :, s) .- θ_ref) for s in 1:N]
    @printf("%-34s N=%6d  stable %6.2f%%  | shear<0 %5.2f%%  bulk<0 %5.2f%%  C44<0 %5.2f%%\n",
            label, N, 100count(stable)/N,
            100count(shear .<= 0)/N, 100count(bulkm .<= 0)/N, 100count(C44 .<= 0)/N)
    @printf("%34s C11∈[%.1f,%.1f]  C12∈[%.1f,%.1f]  C44∈[%.1f,%.1f] GPa  max|δa|=%.4f Å%s\n", "",
            minimum(C11), maximum(C11), minimum(C12), maximum(C12),
            minimum(C44), maximum(C44), maximum(abs.(δa_all)),
            n_oor > 0 ? "  (⚠ $n_oor beyond IFT range)" : "")
    @printf("%34s ‖δθ‖: median=%.3g  max=%.3g\n", "", median(dnorm), maximum(dnorm))
    return (; label, N, C11, C12, C44, shear, bulkm, stable, δa=δa_all, dnorm)
end

# ═════════════════════════════════════════════════════════════════════════════
#  Stage 1 — UNCONSTRAINED baseline
# ═════════════════════════════════════════════════════════════════════════════
println("\n── Stage 1: unconstrained POPS ─────────────────────────────")
pops_corr = corrections(Ap, Yw, P; leverage_percentile=0.0)
println("  $(size(pops_corr, 1)) forest members")

stats_forest_unc = born_stats("A0 unconstrained forest",
                              lin_params .+ permutedims(pops_corr), lin_params)

hyp_eig_u, hyp_bound_u = hypercube(pops_corr)                     # unclipped
samples_u, _ = sample_hypercube(hyp_eig_u, hyp_bound_u, lin_params;
                                number_of_committee_members=N_check)
stats_hyp_unc = born_stats("A  unconstrained hypercube", samples_u, lin_params)

# ═════════════════════════════════════════════════════════════════════════════
#  Exact constrained-ridge solver (active-set enumeration, no tolerances)
#
#  min ½xᵀHx + bᵀx  s.t.  E x = e,  R x ≥ d     (x = θ̃ = Γc)
#
#  For ≤ a dozen inequality rows, enumerate active sets: for each subset S,
#  solve the equality-KKT system via the prefactorized Cholesky of H and keep
#  the (unique, by convexity) solution with λ_S ≥ 0 and inactive rows
#  feasible.  Cost per candidate: one small (n_E+|S|)² solve.
# ═════════════════════════════════════════════════════════════════════════════
function exact_constrained_ridge(Hfact, x0, HE, E, e, HR, R, d; tol=1e-8)
    nE, nR = size(E, 1), size(R, 1)
    nR <= 12 || error("too many inequality rows for enumeration")
    for S in 0:(2^nR - 1)
        act = [j for j in 1:nR if (S >> (j - 1)) & 1 == 1]
        G   = vcat(E, R[act, :])
        g   = vcat(e, d[act])
        HG  = hcat(HE, HR[:, act])                   # H⁻¹Gᵀ
        ν   = (G * HG) \ (g .- G * x0)
        x   = x0 .+ HG * ν
        all(ν[nE+1:end] .>= -tol) || continue        # multiplier signs
        inact = setdiff(1:nR, act)
        (isempty(inact) || all(R[inact, :] * x .>= d[inact] .- tol)) || continue
        return x
    end
    error("no KKT-consistent active set found (degenerate constraint geometry)")
end

# Shared QP pieces (θ̃-space)
Hqp   = Ap' * Ap .+ (1.0 / size(Ap, 1)) .* (P' * P)
Hfact = cholesky(Symmetric(Hqp))
x0    = Hfact \ (Ap' * Yw)                # unconstrained ridge minimiser (θ̃)
@printf("  sanity: ‖Γ·lin_params − x₀‖ = %.3e\n", norm(P * lin_params .- x0))

# ═════════════════════════════════════════════════════════════════════════════
#  Stage 2 — constrained ridge fit (exact)
# ═════════════════════════════════════════════════════════════════════════════
println("\n── Stage 2: constrained ridge (exact active-set) ───────────")
constraint_1 = H_basis[1, 1, :];  C11_pin = dot(constraint_1, lin_params)
constraint_2 = H_basis[1, 2, :]
constraint_4 = H_basis[4, 4, :]

ineq_constraint_matrix = vcat(constraint_4',                        # C44        ≥ 0.1
                              (constraint_1 .- constraint_2)',      # C11 − C12  ≥ 1.0
                              (constraint_1 .+ 2 .* constraint_2)', # C11 + 2C12 ≥ 0.1
                              b_double_prime')                      # b″·c       ≥ 1e-9
ineq_lower  = [0.1, 1.0, 0.1, 1e-9]
ineq_bounds = (ineq_lower, fill(Inf, 4))

E_mean = Matrix(vcat(constraint_1', b_prime') / P)   # equalities (θ̃-space)
e_mean = [C11_pin, 0.0]
R_til  = Matrix(ineq_constraint_matrix / P)          # inequalities (θ̃-space)

HE_mean = Hfact \ Matrix(E_mean')
HR_til  = Hfact \ Matrix(R_til')

x_con = exact_constrained_ridge(Hfact, x0, HE_mean, E_mean, e_mean, HR_til, R_til, ineq_lower)
constrained_teta = P \ x_con
writedlm("$(result.dir)/results/constrained_ridge_params.csv", constrained_teta, ',')
@printf("  ‖constrained − lin_params‖ = %.4e  (≈0 when the mean is already feasible)\n",
        norm(constrained_teta .- lin_params))
@printf("Constrained: C11 = %.2f  C12 = %.2f  C44 = %.2f  GPa\n",
        dot(c11_0, constrained_teta) * eV_to_GPa,
        dot(c12_0, constrained_teta) * eV_to_GPa,
        dot(c44_0, constrained_teta) * eV_to_GPa)

# ═════════════════════════════════════════════════════════════════════════════
#  Stage 3 — selectively constrained POPS: exact repair of violators only
# ═════════════════════════════════════════════════════════════════════════════
println("\n── Stage 3: selectively constrained POPS ───────────────────")
# Closed-form pointwise solutions that satisfy the inequality rows are the
# exact constrained solutions (inactive constraints).  Violators are repaired
# by the exact solver with the data pin Ap[i,:]·θ̃ = Yw[i] as the equality.
n_rows   = length(Yw)
con_pops = Matrix{Float64}(undef, n_rows, n_params)
repaired = falses(n_rows)
for i in 1:n_rows
    θ_i = lin_params .+ vec(pops_corr[i, :])
    Aθ  = ineq_constraint_matrix * θ_i
    if all(ineq_lower .<= Aθ)
        con_pops[i, :] = θ_i                     # inactive constraints → exact
    else
        repaired[i] = true
        a  = Ap[i, :]
        ha = Hfact \ a
        x  = exact_constrained_ridge(Hfact, x0, reshape(ha, :, 1), Matrix(a'), [Yw[i]],
                                     HR_til, R_til, ineq_lower)
        con_pops[i, :] = P \ x
    end
    i % 2000 == 0 && print("\r  $i / $n_rows  (repaired: $(count(repaired))) …")
end
println("\r  done.  exact repairs: $(count(repaired)) / $n_rows " *
        "($(round(100count(repaired)/n_rows; digits=2))%)                ")

# Verification: every forest member must now satisfy the inequality rows.
V = con_pops * ineq_constraint_matrix'            # n_rows × 4
n_bad = count(any(V[i, :] .< ineq_lower .- 1e-6) for i in 1:n_rows)
println("  post-repair feasibility violations: $n_bad / $n_rows  (must be 0)")
n_bad == 0 || error("repair stage left infeasible members — investigate before sampling")

θ_center = vec(mean(con_pops; dims=1))            # cloud centre for stages 4–6
stats_forest_con = born_stats("B  constrained forest",
                              permutedims(con_pops), θ_center)

# ═════════════════════════════════════════════════════════════════════════════
#  Stage 4 — hypercube over the repaired forest, centered on the cloud mean
# ═════════════════════════════════════════════════════════════════════════════
println("\n── Stage 4: constrained hypercube (centered), no rejection ─")
con_deltas = con_pops .- θ_center'                # centered fluctuation cloud
hyp_eig_c, hyp_bound_c = hypercube(con_deltas)    # unclipped
samples_c, _ = sample_hypercube(hyp_eig_c, hyp_bound_c, θ_center;
                                number_of_committee_members=N_check)
stats_hyp_con = born_stats("C  constrained hypercube", samples_c, θ_center)

# ═════════════════════════════════════════════════════════════════════════════
#  Stage 5 — box + rejection (kept for comparison; box draws are still huge)
# ═════════════════════════════════════════════════════════════════════════════
println("\n── Stage 5: box + rejection ────────────────────────────────")
samples_r, _ = rejection_sample_hypercube(hyp_eig_c, hyp_bound_c, θ_center,
                                          ineq_constraint_matrix, ineq_bounds;
                                          number_of_committee_members=N_check,
                                          max_attempts=5_000_000)
stats_rej = born_stats("D  box + rejection", samples_r, θ_center)

committee = [samples_r[:, i] for i in 1:N_committee]
writedlm("$(result.dir)/results/rejection_sampled_committee.csv", committee, ',')

# ═════════════════════════════════════════════════════════════════════════════
#  Stage 5b — covariance-matched Gaussian proposal (the honest committee)
#
#  The box's per-direction bounds are min/max projections over 74k members:
#  a uniform box draw is near-extreme in ~all 300 directions at once, giving
#  ‖δθ‖ ~ 60× the median correction (measured), which no 4-row rejection can
#  fix.  The Gaussian proposal matches the cloud's covariance, so draws have
#  the same scale and correlation structure as real corrections.
# ═════════════════════════════════════════════════════════════════════════════
# println("\n── Stage 5b: Gaussian proposal ± rejection ─────────────────")
# gauss_eig, gauss_sig = gaussian_proposal(con_deltas)
# samples_g, _ = sample_gaussian(gauss_eig, gauss_sig, θ_center;
#                                number_of_committee_members=N_check)
# stats_gauss = born_stats("E  Gaussian, no rejection", samples_g, θ_center)

# samples_gr, _ = rejection_sample_gaussian(gauss_eig, gauss_sig, θ_center,
#                                           ineq_constraint_matrix, ineq_bounds;
#                                           number_of_committee_members=N_check,
#                                           max_attempts=5_000_000)
# stats_gauss_rej = born_stats("F  Gaussian + rejection", samples_gr, θ_center)

# committee = [samples_gr[:, i] for i in 1:N_committee]
# writedlm("$(result.dir)/results/rejection_sampled_committee.csv", committee, ',')

# ═════════════════════════════════════════════════════════════════════════════
#  Stage 6 — summary + figure
# ═════════════════════════════════════════════════════════════════════════════
all_stats = [stats_forest_unc, stats_hyp_unc, stats_forest_con, stats_hyp_con,
             stats_rej]

println("\n══ Summary ══════════════════════════════════════════════════")
for st in all_stats
    @printf("  %-34s stable %6.2f%%   min(C11−C12)=%8.2f   min(C44)=%8.2f GPa\n",
            st.label, 100count(st.stable)/st.N, minimum(st.shear), minimum(st.C44))
end

open("$(result.dir)/results/constrain_compare_summary.csv", "w") do io
    println(io, "# $(result.name); IFT Born check; N_check=$N_check; exact active-set QP; centered proposals")
    println(io, "ensemble,N,stable_frac,shear_viol_frac,bulk_viol_frac,C44_viol_frac,min_shear_GPa,min_C44_GPa,max_abs_da_Ang,median_dnorm")
    for st in all_stats
        @printf(io, "%s,%d,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%.5f,%.5f\n",
                st.label, st.N, count(st.stable)/st.N,
                count(st.shear .<= 0)/st.N, count(st.bulkm .<= 0)/st.N,
                count(st.C44 .<= 0)/st.N, minimum(st.shear), minimum(st.C44),
                maximum(abs.(st.δa)), median(st.dnorm))
    end
end

series = [(stats_hyp_unc,   RGBAf(0.00, 0.447, 0.698, 1.0)),   # blue
          (stats_hyp_con,   RGBAf(0.902, 0.624, 0.000, 1.0)),  # orange
          (stats_rej,       RGBAf(0.00, 0.620, 0.451, 1.0))]   # green

fig = Figure(size=(1150, 360))
panels = [("C11 − C12 (GPa)", st -> st.shear),
          ("C11 + 2C12 (GPa)", st -> st.bulkm),
          ("C44 (GPa)",        st -> st.C44)]
for (p, (lab, getter)) in enumerate(panels)
    ax = Axis(fig[1, p]; xlabel=lab, ylabel=p == 1 ? "samples" : "")
    for (st, col) in series
        stephist!(ax, getter(st); bins=80, color=col, linewidth=1.8)
    end
    vlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=1.2)
end
Legend(fig[1, 4],
       [LineElement(color=c, linewidth=2.5) for (_, c) in series],
       [@sprintf("%s  (%.1f%% stable)", st.label, 100count(st.stable)/st.N)
        for (st, _) in series])
Label(fig[0, :], "Born margins — $(result.name): unconstrained box vs constrained box vs rejection";
      fontsize=13)
save("$(result.dir)/results/constrain_compare_born_margins.png", fig)
display(fig)
println("\nSaved: constrain_compare_born_margins.png, constrain_compare_summary.csv,")
println("       constrained_ridge_params.csv, rejection_sampled_committee.csv")
