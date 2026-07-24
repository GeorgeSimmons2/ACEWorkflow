# hypercube_vs_rejection_born_phonons_Al_20_4_6A_2_subset_50.jl
#
# Naive POPS hypercube vs rejection-sampled hypercube on the SAME unclipped
# proposal, built over the RAW POPS corrections of Al_20_4_6A_2 (subset_50):
#
#   1. Verify the raw delta forest is Born-physical (it should be mostly —
#      the pathology is the box proposal, not the forest).
#   1b. Any forest members violating the fixed-geometry Born rows (a few
#      C44 < 0 outliers) are repaired EXACTLY — active-set enumeration with
#      their data pin, as in constrain_and_compare — so the proposal cloud
#      is fully Born-physical before any box is built.
#   2. Draw N_check naive box samples and N_check rejection-sampled box
#      samples (rejected against the fixed-geometry Born inequality rows),
#      both from the SAME unclipped hypercube over the repaired forest.
#   3. Born criteria for both via the IFT lattice-shift correction, so every
#      sample's elastic constants are evaluated at ITS OWN minimum structure.
#      Samples with |δa| > IFT_DA_MAX are beyond the 2nd-order expansion's
#      validity: the rejection sampler DISCARDS such draws and redraws (its
#      committee is fully checkable), while the naive box keeps them (it is
#      the baseline being measured) and its stable fraction is reported both
#      overall and over the checkable subset.
#   4. Phonon bands for the first N_phonon members of each ensemble using the
#      src/Phonons machinery (phonon_committee): each member's lattice
#      constant is relaxed EXACTLY before its bands are computed, and the
#      exact relaxed a is printed against the IFT estimate as a cross-check.
#
# Companion to constrain_and_compare_Al_20_4_6A_2_subset_50.jl; reuses an
# in-session `pops_corr` when present so the forest isn't recomputed.
#
# Outputs (to models/Al_20_4_6A_2_subset_50_percent/results/):
#   hyp_vs_rejection_born_margins.png
#   hyp_vs_rejection_summary.csv
#   naive_hypercube_committee_10.csv / rejection_hypercube_committee_10.csv
#   naive_hypercube_phonon_committee_*     (plots + freq CSVs, via src/Phonons)
#   rejection_hypercube_phonon_committee_*
#   hyp_vs_rejection_phonon_bands_<N>x<N>x<N>.png
#   hyp_vs_rejection_phonon_members.csv
#
# Run:  julia --project [-t N] scripts/uq/hypercube_vs_rejection_born_phonons_Al_20_4_6A_2_subset_50.jl

using LinearAlgebra, DelimitedFiles, Statistics, Printf, Random
using ForwardDiff, Unitful, CairoMakie
using ACEpotentials, ACEWorkflow
import ACEWorkflow: phonon_committee

Random.seed!(1234)

element  = :Al
N_check  = 10_000   # draws per ensemble for the Born statistics
N_phonon = 10       # members per ensemble pushed through phonon bands
N_cell   = 3        # phonon supercell (N_cell³ conventional cells)
IFT_DA_MAX = 0.1    # Å — samples with |δa_IFT| beyond this cannot be checked
                    #     accurately; the rejection sampler discards & redraws them

# ── Model + forest (reuse in-session objects when present) ───────────────────
if !(@isdefined(result)) || result.name != "Al_20_4_6A_3"
    # full-dataset model lives in models/Al_20_4_6A_2_ (trailing underscore);
    # dataset_name="" resolves there — plain load_model(…) throws
    result = load_model(element, 20, 4, 6, 3)
end
model      = result.model
lin_params = result.lin_params
n_params   = length(lin_params)
P          = result.P
println("Model $(result.name): $n_params parameters")

Ap = Diagonal(result.W) * result.A / P
Yw = result.W .* result.Y

if !(@isdefined(pops_corr)) || size(pops_corr, 2) != n_params
    println("Computing POPS corrections (delta forest) …")
    pops_corr = corrections(Ap, Yw, P; leverage_percentile=0.0)
else
    println("Reusing in-session pops_corr.")
end
println("Forest: $(size(pops_corr, 1)) members")
size(pops_corr, 1) == length(Yw) ||
    error("forest rows ($(size(pops_corr,1))) ≠ design rows ($(length(Yw))) — the repair stage needs the 1:1 mapping")

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

# ── IFT Born check for a matrix of members (columns) around θ_ref ────────────
# Identical to constrain_and_compare: elastic constants are evaluated at each
# member's own IFT-shifted minimum, C(θ) = C₀·θ + δa·(dC/da)·θ.
function born_stats(label, members::AbstractMatrix, θ_ref; ift_da_max=IFT_DA_MAX)
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
    checkable = abs.(δa_all) .<= ift_da_max
    n_oor  = count(.!checkable)
    dnorm  = [norm(view(members, :, s) .- θ_ref) for s in 1:N]
    @printf("%-26s N=%6d  stable %6.2f%%  | shear<0 %5.2f%%  bulk<0 %5.2f%%  C44<0 %5.2f%%\n",
            label, N, 100count(stable)/N,
            100count(shear .<= 0)/N, 100count(bulkm .<= 0)/N, 100count(C44 .<= 0)/N)
    @printf("%26s C11∈[%.1f,%.1f]  C12∈[%.1f,%.1f]  C44∈[%.1f,%.1f] GPa  max|δa|=%.4f Å%s\n", "",
            minimum(C11), maximum(C11), minimum(C12), maximum(C12),
            minimum(C44), maximum(C44), maximum(abs.(δa_all)),
            n_oor > 0 ? "  (⚠ $n_oor beyond IFT range)" : "")
    @printf("%26s ‖δθ‖: median=%.3g  max=%.3g\n", "", median(dnorm), maximum(dnorm))
    n_chk = count(checkable)
    n_chk < N && @printf("%26s checkable (|δa| ≤ %.2f Å): %d / %d, of which stable %.2f%%\n", "",
                         ift_da_max, n_chk, N,
                         n_chk == 0 ? NaN : 100count(stable .& checkable) / n_chk)
    return (; label, N, C11, C12, C44, shear, bulkm, stable, checkable, δa=δa_all, dnorm)
end

# Fixed-geometry Born inequality rows (linear in θ) — used to repair forest
# violators in Stage 1b AND as the rejection test in Stage 2
constraint_1 = H_basis[1, 1, :]
constraint_2 = H_basis[1, 2, :]
constraint_4 = H_basis[4, 4, :]
ineq_constraint_matrix = vcat(constraint_4',                        # C44        ≥ 0.1
                              (constraint_1 .- constraint_2)',      # C11 − C12  ≥ 1.0
                              (constraint_1 .+ 2 .* constraint_2)', # C11 + 2C12 ≥ 0.1
                              b_double_prime')                      # b″·θ       ≥ 1e-9
ineq_lower  = [0.1, 1.0, 0.1, 1e-9]
ineq_bounds = (ineq_lower, fill(Inf, 4))

# ═════════════════════════════════════════════════════════════════════════════
#  Stage 1 — raw delta forest (expected mostly Born-physical)
# ═════════════════════════════════════════════════════════════════════════════
println("\n── Stage 1: raw POPS delta forest ──────────────────────────")
stats_forest = born_stats("forest", lin_params .+ permutedims(pops_corr), lin_params)

# ═════════════════════════════════════════════════════════════════════════════
#  Stage 1b — exact repair of Born-violating forest members
#
#  Members failing the inequality rows are re-solved with their data pin:
#     min ½xᵀHx + bᵀx  s.t.  Ap[i,:]·x = Yw[i],  R x ≥ d    (x = θ̃ = Pθ)
#  by active-set enumeration against a prefactorized Cholesky (same exact
#  solver as constrain_and_compare — no OSQP tolerances).  Feasible members
#  are already the exact constrained solutions (inactive constraints) and
#  pass through untouched.
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

println("\n── Stage 1b: exact repair of Born-violating members ────────")
Hqp   = Ap' * Ap .+ (1.0 / size(Ap, 1)) .* (P' * P)
Hfact = cholesky(Symmetric(Hqp))
x0    = Hfact \ (Ap' * Yw)                # unconstrained ridge minimiser (θ̃)
@printf("  sanity: ‖P·lin_params − x₀‖ = %.3e\n", norm(P * lin_params .- x0))
R_til  = Matrix(ineq_constraint_matrix / P)
HR_til = Hfact \ Matrix(R_til')

con_pops = Matrix{Float64}(undef, size(pops_corr, 1), n_params)
repaired = falses(size(pops_corr, 1))
for i in 1:size(pops_corr, 1)
    θ_i = lin_params .+ vec(pops_corr[i, :])
    if all(ineq_lower .<= ineq_constraint_matrix * θ_i)
        con_pops[i, :] = θ_i                 # already feasible → untouched
    else
        repaired[i] = true
        a  = Ap[i, :]
        ha = Hfact \ a
        x  = exact_constrained_ridge(Hfact, x0, reshape(ha, :, 1), Matrix(a'), [Yw[i]],
                                     HR_til, R_til, ineq_lower)
        con_pops[i, :] = P \ x
    end
    i % 5000 == 0 && print("\r  $i / $(size(pops_corr, 1))  (repaired: $(count(repaired))) …")
end
println("\r  done.  exact repairs: $(count(repaired)) / $(size(pops_corr, 1))                    ")

V_chk = con_pops * ineq_constraint_matrix'
n_bad = count(any(V_chk[i, :] .< ineq_lower .- 1e-6) for i in 1:size(con_pops, 1))
println("  post-repair feasibility violations: $n_bad (must be 0)")
n_bad == 0 || error("repair stage left infeasible members — investigate before sampling")

stats_forest_rep = born_stats("repaired forest", permutedims(con_pops), lin_params)

# ═════════════════════════════════════════════════════════════════════════════
#  Stage 2 — naive box vs rejection box over the REPAIRED forest cloud
# ═════════════════════════════════════════════════════════════════════════════
println("\n── Stage 2: naive vs rejection hypercube ───────────────────")
repaired_corr = con_pops .- lin_params'           # corrections about lin_params
hyp_eig, hyp_bound = hypercube(repaired_corr)     # unclipped

samples_naive, _ = sample_hypercube(hyp_eig, hyp_bound, lin_params;
                                    number_of_committee_members=N_check)
stats_naive = born_stats("naive hypercube", samples_naive, lin_params)

# Rejection predicate: fixed-geometry Born rows AND an IFT-trust test.
# Samples whose lattice shift |δa| exceeds IFT_DA_MAX cannot be accurately
# checked against the constraints, so they are chucked out and replaced by
# fresh draws (cheap linear rows first, δa test second).
K_ref  = dot(lin_params, b_double_prime)
s3_ref = dot(lin_params, b_triple_prime)
function ift_da(δθ)
    δa1 = -dot(b_prime, δθ) / K_ref
    return δa1 - (dot(b_double_prime, δθ) * δa1 + 0.5 * s3_ref * δa1^2) / K_ref
end
is_feasible_checkable = θ -> begin
    Aθ = ineq_constraint_matrix * θ
    all(ineq_lower .<= Aθ) || return false
    abs(ift_da(θ .- lin_params)) <= IFT_DA_MAX
end

samples_rej, _ = rejection_sample_hypercube(hyp_eig, hyp_bound, lin_params,
                                            is_feasible_checkable;
                                            number_of_committee_members=N_check,
                                            max_attempts=5_000_000)
stats_rej = born_stats("rejection hypercube", samples_rej, lin_params)

# ── Born-margin histograms + summary CSV ─────────────────────────────────────
series = [(stats_forest,     RGBAf(0.20, 0.20, 0.20, 1.0)),    # dark grey
          (stats_forest_rep, RGBAf(0.902, 0.624, 0.000, 1.0)), # orange
          (stats_naive,      RGBAf(0.00, 0.447, 0.698, 1.0)),  # blue
          (stats_rej,        RGBAf(0.00, 0.620, 0.451, 1.0))]  # green

fig = Figure(size=(1150, 360))
panels = [("C11 − C12 (GPa)", st -> st.shear),
          ("C11 + 2C12 (GPa)", st -> st.bulkm),
          ("C44 (GPa)",        st -> st.C44)]
for (p, (lab, getter)) in enumerate(panels)
    ax = Axis(fig[1, p]; xlabel=lab, ylabel=p == 1 ? "density" : "")
    for (st, col) in series
        stephist!(ax, getter(st); bins=80, color=col, linewidth=1.8,
                  normalization=:pdf)
    end
    vlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=1.2)
end
Legend(fig[1, 4],
       [LineElement(color=c, linewidth=2.5) for (_, c) in series],
       [@sprintf("%s  (%.1f%% stable)", st.label, 100count(st.stable)/st.N)
        for (st, _) in series])
Label(fig[0, :], "Born margins (IFT-corrected) — $(result.name): raw/repaired forest vs naive box vs rejection";
      fontsize=13)
save("$(result.dir)/results/hyp_vs_rejection_born_margins.png", fig)
display(fig)

open("$(result.dir)/results/hyp_vs_rejection_summary.csv", "w") do io
    println(io, "# $(result.name); IFT Born check; N_check=$N_check; same unclipped hypercube proposal")
    println(io, "ensemble,N,stable_frac,checkable_frac,stable_frac_checkable,shear_viol_frac,bulk_viol_frac,C44_viol_frac,min_shear_GPa,min_C44_GPa,max_abs_da_Ang,median_dnorm")
    for (st, _) in series
        n_chk = count(st.checkable)
        @printf(io, "%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%.5f,%.5f\n",
                st.label, st.N, count(st.stable)/st.N,
                n_chk/st.N, n_chk == 0 ? NaN : count(st.stable .& st.checkable)/n_chk,
                count(st.shear .<= 0)/st.N, count(st.bulkm .<= 0)/st.N,
                count(st.C44 .<= 0)/st.N, minimum(st.shear), minimum(st.C44),
                maximum(abs.(st.δa)), median(st.dnorm))
    end
end

# ═════════════════════════════════════════════════════════════════════════════
#  Stage 3 — phonons: first N_phonon members of each ensemble
# ═════════════════════════════════════════════════════════════════════════════
committee_naive = [samples_naive[:, i] for i in 1:N_phonon]
committee_rej   = [samples_rej[:, i]   for i in 1:N_phonon]
writedlm("$(result.dir)/results/naive_hypercube_committee_10.csv",     committee_naive, ',')
writedlm("$(result.dir)/results/rejection_hypercube_committee_10.csv", committee_rej,   ',')

# Exact relaxed lattice constant vs the IFT estimate for every phonon member —
# the "definitely at the correct minimum" cross-check.  The phonon machinery
# itself relaxes each member exactly, so bands are at the true minimum either
# way; this quantifies how far the IFT estimate used in the 10k-sample Born
# statistics is from that truth.
model_chk = deepcopy(model)
function exact_minimum_check(label, committee, stats)
    println("\n── $label: exact relaxed a vs IFT estimate ──────────────")
    a_exact = fill(NaN, length(committee))
    for (i, θ) in enumerate(committee)
        ACEpotentials.Models.set_linear_parameters!(model_chk, θ)
        a_ift = a_eq + stats.δa[i]
        try
            a_exact[i] = ACEWorkflow.relax_lattice_constant(model_chk, element)
            @printf("  %2d: a_IFT = %.5f Å   a_exact = %.5f Å   |Δ| = %.2e Å%s\n",
                    i, a_ift, a_exact[i], abs(a_ift - a_exact[i]),
                    abs(stats.δa[i]) > 0.1 ? "  (⚠ beyond IFT range)" : "")
        catch err
            @warn "member $i: lattice relaxation failed — no minimum found?" err
        end
    end
    return a_exact
end

a_exact_naive = exact_minimum_check("naive hypercube",     committee_naive, stats_naive)
a_exact_rej   = exact_minimum_check("rejection hypercube", committee_rej,   stats_rej)

println("\n── Phonon bands: naive hypercube committee ─────────────────")
x_vals, bands_naive, x_ticks, labels = phonon_committee(model, committee_naive, result, element;
                                                        N_cell=N_cell, file_prefix="naive_hypercube_")

println("\n── Phonon bands: rejection hypercube committee ─────────────")
_, bands_rej, _, _ = phonon_committee(model, committee_rej, result, element;
                                      N_cell=N_cell, file_prefix="rejection_hypercube_")

# ── Per-member table: Born margins (IFT), exact minimum, phonon stability ────
open("$(result.dir)/results/hyp_vs_rejection_phonon_members.csv", "w") do io
    println(io, "ensemble,member,shear_GPa,bulk_GPa,C44_GPa,da_IFT_Ang,a_exact_Ang,min_freq_THz,n_imag_modes")
    for (label, stats, a_ex, bands) in (("naive",     stats_naive, a_exact_naive, bands_naive),
                                        ("rejection", stats_rej,   a_exact_rej,   bands_rej))
        for i in 1:N_phonon
            f = bands[i + 1]
            @printf(io, "%s,%d,%.4f,%.4f,%.4f,%.5f,%.5f,%.4f,%d\n",
                    label, i, stats.shear[i], stats.bulkm[i], stats.C44[i],
                    stats.δa[i], a_ex[i], minimum(f), count(f .< -0.05))
        end
    end
end

println("\n── Phonon stability summary ────────────────────────────────")
for (label, bands) in (("naive hypercube", bands_naive), ("rejection hypercube", bands_rej))
    n_unstable = count(any(bands[i + 1] .< -0.05) for i in 1:N_phonon)
    @printf("  %-22s %d / %d members with imaginary modes (< −0.05 THz)\n",
            label, n_unstable, N_phonon)
end

# ── Overlay figure: naive (grey) vs rejection (green) vs mean (blue) ─────────
fig_ph = Figure(size=(900, 520))
ax_ph  = Axis(fig_ph[1, 1];
              xlabel       = "Wave vector",
              ylabel       = "Frequency (THz)",
              title        = "Al phonon bands — naive vs rejection-sampled hypercube ($N_phonon members each)",
              xticks       = (x_ticks, labels),
              xgridvisible = false)
for freqs in bands_naive[2:end], b in 1:size(freqs, 1)
    lines!(ax_ph, x_vals, freqs[b, :]; color=RGBAf(0.5, 0.5, 0.5, 0.35), linewidth=1.0)
end
for freqs in bands_rej[2:end], b in 1:size(freqs, 1)
    lines!(ax_ph, x_vals, freqs[b, :]; color=RGBAf(0.00, 0.620, 0.451, 0.5), linewidth=1.0)
end
for b in 1:size(bands_naive[1], 1)
    lines!(ax_ph, x_vals, bands_naive[1][b, :]; color=RGBAf(0.00, 0.447, 0.698, 0.95), linewidth=2.0)
end
hlines!(ax_ph, [0.0]; color=:black, linestyle=:dash, linewidth=0.8)
vlines!(ax_ph, x_ticks; color=(:black, 0.3), linewidth=0.8)
Legend(fig_ph[1, 2],
       [LineElement(color=RGBAf(0.5, 0.5, 0.5, 0.8),      linewidth=2.0),
        LineElement(color=RGBAf(0.00, 0.620, 0.451, 0.8), linewidth=2.0),
        LineElement(color=RGBAf(0.00, 0.447, 0.698, 0.95), linewidth=2.5)],
       ["naive hypercube", "rejection hypercube", "mean model"])
save("$(result.dir)/results/hyp_vs_rejection_phonon_bands_$(N_cell)x$(N_cell)x$(N_cell).png", fig_ph)
display(fig_ph)

println("\nSaved to $(result.dir)/results/:")
println("  hyp_vs_rejection_born_margins.png, hyp_vs_rejection_summary.csv")
println("  naive/rejection_hypercube_committee_10.csv")
println("  naive/rejection_hypercube_phonon_committee_* (plots + freq CSVs)")
println("  hyp_vs_rejection_phonon_bands_$(N_cell)x$(N_cell)x$(N_cell).png, hyp_vs_rejection_phonon_members.csv")
