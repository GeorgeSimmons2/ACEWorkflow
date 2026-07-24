# rattled_energy_constraints_Al_20_4_6A_2_subset_50.jl
#
# "Crazy" energy-ordering prior for Al_20_4_6A_2 (subset_50_percent):
# constrain that RATTLED structures lie ABOVE the perfect crystal in energy
# at a range of lattice constants.  Because the ACE energy is linear in θ,
# each ordering is one inequality row:
#
#     (b_rattled(a,k) − b_perfect(a)) · θ  >  0
#
# with a ∈ collect(LinRange(0.9, 1.1, 10)) .* a_eq and ONE rattle per lattice
# constant — 10 ordering rows — plus the lattice-equilibrium constraints
#
#     b′(a_eq)·θ = 0     (a_eq stays stationary)
#     b″(a_eq)·θ > 0     (…and is a minimum of E(a))
#
# No elastic rows.  The point is to test whether energy ordering around the
# crystal plus the E(a) minimum condition is enough of a stability prior.
#
# Diagnostics after the fit: constraint margins, training RMSE, relaxed a_eq,
# elastic constants, E(a) volume curve, and a folded-mesh phonon check
# (one supercell Hessian eigendecomposition) for nominal vs constrained.
#
# The mean fit is followed by CONSTRAINED POPS: forest members whose
# closed-form solution violates any ordering are re-solved with their data pin
# plus the 30 rows; the committee is then rejection-sampled from the unclipped
# hypercube over the repaired forest against the same 30 rows and pushed
# through phonon bands.
#
# Outputs (to models/Al_20_4_6A_2_subset_50_percent/results/):
#   rattled_constrained_teta.csv
#   rattled_constraint_margins.csv    (a, rattle idx, ΔE nominal, ΔE constrained)
#   rattled_phonon_bands_comparison_<N>x<N>x<N>.png   (stacked panels)
#   rattled_phonon_freqs_{nominal,constrained}_THz.csv + x/tick CSVs
#   rattled_constrained_pops_forest.jls               (repaired forest + statuses)
#   rattled_pops_committee_10.csv                     (one member per row)
#   rattled_pops_phonon_committee_*                   (plots + freq CSVs)
#
# Run:  julia --project scripts/uq/rattled_energy_constraints_Al_20_4_6A_2_subset_50.jl

using LinearAlgebra, DelimitedFiles, Statistics, Printf, Random, Serialization
using SparseArrays, StaticArrays
using Unitful, ForwardDiff, ACEpotentials, ACEWorkflow
using AtomsBuilder                # bulk, rattle!
using AtomsCalculatorsUtilities.SitePotentials: hessian
using OSQP
using CairoMakie
import ACEWorkflow: phonon_committee

Random.seed!(20260723)   # reproducible rattles

element     = :Al
n_rattle    = 10        # rattled structures per lattice constant (10 a's × 1 = 10 rows)
rattle_r    = 0.05     # Å — rattle!(sys, r): uniform-in-ball displacement radius
N_cell      = 3        # supercell (N_cell³ conventional cells) for the structures
margin_eV   = 1e-3     # ΔE lower bound: "> 0" bounded away from zero so OSQP's
                       # 1e-6 tolerance cannot park a row at −ε (ΔE is O(0.1–1 eV))
N_committee = 10       # committee members pushed through phonon bands

# ── Load model ────────────────────────────────────────────────────────────────
result     = load_model(element, 20, 4, 6, 2; dataset_name="subset_50_percent")
model      = result.model
lin_params = result.lin_params
n_params   = length(lin_params)
P          = result.P
println("Model $(result.name): $n_params parameters, $(length(result.Y)) design rows")

Ap = Diagonal(result.W) * result.A / P
Yw = result.W .* result.Y

println("Relaxing mean model …")
a_eq = ACEWorkflow.relax_lattice_constant(model, element)
@printf("  a_eq = %.6f Å\n", a_eq)

a_grid = [1.0] .* a_eq

println("Computing b′ and b″ at a_eq …")
function lattice_basis(a_val)
    sys = ACEWorkflow.Elasticity.reference_system(element; a=a_val)
    ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(sys, model))
end
b_prime        = ForwardDiff.derivative(lattice_basis, a_eq)
b_double_prime = ForwardDiff.derivative(a -> ForwardDiff.derivative(lattice_basis, a), a_eq)

# ── Build the 30 energy-ordering rows ────────────────────────────────────────
supercell(a) = bulk(element; a=a*u"Å", cubic=true) * (N_cell, N_cell, N_cell)
basis_energy(sys) = ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(sys, model))

function build_rattle_rows(a_grid, n_rattle, rattle_r, lin_params)
    n_rows  = length(a_grid) * n_rattle
    rows    = zeros(n_rows, length(lin_params))
    meta    = zeros(n_rows, 3)                       # a, rattle idx, ΔE_nominal
    b_perf  = zeros(length(a_grid), length(lin_params))
    nat     = length(supercell(a_grid[1]))
    idx = 0
    for (ia, a) in enumerate(a_grid)
        b_p = basis_energy(supercell(a))
        b_perf[ia, :] = b_p
        for k in 1:n_rattle
            sys_r = supercell(a)
            rattle!(sys_r, rattle_r)
            idx += 1
            rows[idx, :] = basis_energy(sys_r) .- b_p
            meta[idx, :] = [a, k, dot(view(rows, idx, :), lin_params)]
            @printf("  a = %.4f Å  rattle %d:  ΔE_nominal = %+9.4f eV%s\n",
                    a, k, meta[idx, 3], meta[idx, 3] <= 0 ? "   ← violated" : "")
        end
    end
    return rows, meta, b_perf, nat
end

println("Building $(length(a_grid) * n_rattle) rattled-ordering rows " *
        "($(N_cell)×$(N_cell)×$(N_cell) cells, rattle radius $rattle_r Å) …")
rattle_rows, rattle_meta, b_perfect, Nat_super = build_rattle_rows(a_grid, n_rattle, rattle_r, lin_params)
@printf("Nominal model violates %d / %d orderings\n",
        count(rattle_meta[:, 3] .<= 0), size(rattle_rows, 1))

# ── Constrained ridge regression: ONLY the 30 ordering rows ──────────────────
function constrained_ridge_regression(X_train, Y_train, Gamma, constraint_matrix, constraint_bounds; lambda = 1.0 / size(X_train, 1))
    H = (X_train' * X_train .+ (lambda .* Gamma' * Gamma))
    b = - X_train' * Y_train
    osqp_model = OSQP.Model()
    OSQP.setup!(osqp_model; P=sparse(H), q=b, A=sparse(constraint_matrix / Gamma),
                l=constraint_bounds[1], u=constraint_bounds[2],
                max_iter=5_000_000_000, check_termination=1_000, verbose=true,
                eps_abs=1e-6, eps_rel=1e-6)
    results = OSQP.solve!(osqp_model)
    return Gamma \ results.x
end

# Full constraint set: lattice pin (equality) + curvature + the 10 orderings
all_constraints = vcat(b_prime', b_double_prime', rattle_rows)
lower_bounds    = vcat([0.0, 1e-9], fill(margin_eV, size(rattle_rows, 1)))
upper_bounds    = vcat([0.0, Inf],  fill(Inf,       size(rattle_rows, 1)))

constrained_teta = constrained_ridge_regression(Ap, Yw, P, all_constraints, (lower_bounds, upper_bounds))
writedlm("$(result.dir)/results/rattled_constrained_teta.csv", constrained_teta, ',')

# ── Constraint margins before/after ──────────────────────────────────────────
dE_con = rattle_rows * constrained_teta
open("$(result.dir)/results/rattled_constraint_margins.csv", "w") do io
    println(io, "a_Ang,rattle,dE_nominal_eV,dE_constrained_eV")
    writedlm(io, hcat(rattle_meta, dE_con), ',')
end
@printf("\nConstraint margins: nominal min = %+.4f eV  →  constrained min = %+.4f eV  (bound %.0e)\n",
        minimum(rattle_meta[:, 3]), minimum(dE_con), margin_eV)
@printf("Active rows (ΔE within 10%% of bound): %d / %d\n",
        count(dE_con .< 10 * margin_eV), length(dE_con))
@printf("Lattice pin:  b′·θ = %+.3e eV/Å (≈0)   b″·θ = %+.4f eV/Å² (>0)\n",
        dot(b_prime, constrained_teta), dot(b_double_prime, constrained_teta))

# ── Training-error cost of the constraints ───────────────────────────────────
rmse(x) = norm(Ap * x .- Yw) / sqrt(length(Yw))
@printf("\nWeighted training RMSE:  nominal %.6f  →  constrained %.6f  (+%.2f%%)\n",
        rmse(P * lin_params), rmse(P * constrained_teta),
        100 * (rmse(P * constrained_teta) / rmse(P * lin_params) - 1))

# ── Relaxed lattice constant + elastic constants ─────────────────────────────
function elastic_constants(θ, a)
    _, H_eq_a, _ = ACEWorkflow.Elasticity.strain_hessian_GPa(model, element; a=a)
    sysa = ACEWorkflow.Elasticity.reference_system(element; a=a)
    La   = SMatrix{3,3,Float64}(ustrip.(ACEWorkflow.Elasticity.lattice_matrix(sysa.cell.cell_vectors)))
    conv = 160.2176621 / abs(det(La))
    return dot(H_eq_a[1,1,:], θ) * conv, dot(H_eq_a[1,2,:], θ) * conv, dot(H_eq_a[4,4,:], θ) * conv
end

ACEpotentials.Models.set_linear_parameters!(model, constrained_teta)
a_eq_c = ACEWorkflow.relax_lattice_constant(model, element)
@printf("\nRelaxed lattice constant:  nominal %.6f Å  →  constrained %.6f Å\n", a_eq, a_eq_c)

C11_n, C12_n, C44_n = elastic_constants(lin_params, a_eq)
C11_c, C12_c, C44_c = elastic_constants(constrained_teta, a_eq_c)
println("\n── Elastic constants (GPa, each at its own a_eq) ───────────")
@printf("           %10s  %10s\n", "nominal", "constrained")
@printf("  C11  =   %8.3f    %8.3f\n", C11_n, C11_c)
@printf("  C12  =   %8.3f    %8.3f\n", C12_n, C12_c)
@printf("  C44  =   %8.3f    %8.3f\n", C44_n, C44_c)
@printf("  Born:  C11-C12 = %.3f   C11+2C12 = %.3f   C44 = %.3f  (all must be > 0)\n",
        C11_c - C12_c, C11_c + 2C12_c, C44_c)

# ── E(a) volume curve (per atom, relative to its minimum) ────────────────────
println("\n── E(a) volume curve, eV/atom relative to curve minimum ────")
E_nom = b_perfect * lin_params       ./ Nat_super
E_con = b_perfect * constrained_teta ./ Nat_super
@printf("  %-10s %12s %12s\n", "a (Å)", "nominal", "constrained")
for (ia, a) in enumerate(a_grid)
    @printf("  %-10.4f %12.5f %12.5f\n", a, E_nom[ia] - minimum(E_nom), E_con[ia] - minimum(E_con))
end

# ── Folded-mesh phonon check (one supercell Hessian each) ────────────────────
println("\n── Folded-mesh phonon check ($(N_cell)×$(N_cell)×$(N_cell) commensurate q-mesh) ─────")
function folded_mesh_check(label, a)
    sys = supercell(a)
    m   = ustrip(sys[1].mass)
    ω2  = eigvals(Symmetric(ustrip.(hessian(sys, model)) ./ m))
    f   = sign.(ω2) .* sqrt.(abs.(ω2)) .* FREQ_THz
    @printf("  %-12s a=%.5f Å:  ω ∈ [%+.4f, %.4f] THz  (acoustic zeros %.1e %.1e %.1e)%s\n",
            label, a, f[4], f[end], f[1], f[2], f[3],
            f[4] < 0 ? "   ← UNSTABLE" : "")
    return f
end

ACEpotentials.Models.set_linear_parameters!(model, lin_params)
folded_mesh_check("nominal", a_eq)
ACEpotentials.Models.set_linear_parameters!(model, constrained_teta)
folded_mesh_check("constrained", a_eq_c)

# ── Phonon band comparison: stacked panels, nominal above constrained ────────
println("\n── Phonon bands: nominal vs constrained ────────────────────")
println("  nominal …")
ACEpotentials.Models.set_linear_parameters!(model, lin_params)
x_n, freqs_n, ticks_n, labels_n = compute_phonon_bands(
    bulk(element; a=a_eq*u"Å"), supercell(a_eq), model; N_per_seg=30, n_modes=nothing)

println("  constrained …")
ACEpotentials.Models.set_linear_parameters!(model, constrained_teta)
x_c, freqs_c, ticks_c, labels_c = compute_phonon_bands(
    bulk(element; a=a_eq_c*u"Å"), supercell(a_eq_c), model; N_per_seg=30, n_modes=nothing)

fig_ph = Figure(size=(750, 700))
panels_ph = [("nominal",                       x_n, freqs_n, ticks_n, labels_n),
             ("rattled-ordering constrained",  x_c, freqs_c, ticks_c, labels_c)]
for (row, (ptitle, x, freqs, ticks, labs)) in enumerate(panels_ph)
    ax = Axis(fig_ph[row, 1];
              ylabel       = "Frequency (THz)",
              title        = ptitle,
              xticks       = (ticks, labs),
              xgridvisible = false)
    row == length(panels_ph) && (ax.xlabel = "Wave vector")
    for b in 1:size(freqs, 1)
        branch = freqs[b, :]
        # −0.05 THz tolerance so the Γ acoustic zeros don't flag a branch red
        color = minimum(branch) < -0.05 ? RGBAf(0.8, 0.1, 0.1, 0.9) :
                                          RGBAf(0.2, 0.4, 0.7, 0.9)
        lines!(ax, x, branch; color, linewidth=1.5)
    end
    hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=0.8)
    vlines!(ax, ticks; color=(:black, 0.3), linewidth=0.8)
end
save("$(result.dir)/results/rattled_phonon_bands_comparison_$(N_cell)x$(N_cell)x$(N_cell).png", fig_ph)
display(fig_ph)

writedlm("$(result.dir)/results/rattled_phonon_x_nominal.csv",              x_n,     ',')
writedlm("$(result.dir)/results/rattled_phonon_freqs_nominal_THz.csv",      freqs_n, ',')
writedlm("$(result.dir)/results/rattled_phonon_x_constrained.csv",          x_c,     ',')
writedlm("$(result.dir)/results/rattled_phonon_freqs_constrained_THz.csv",  freqs_c, ',')

# ═════════════════════════════════════════════════════════════════════════════
#  Constrained POPS: every forest member must satisfy the 30 ordering rows.
#  Members whose closed-form POPS solution already satisfies them pass through
#  untouched (inactive constraints → already exact); violators are re-solved
#  with their data pin  Ap[i,:]·θ̃ = Yw[i]  (prediction pinned — the data row
#  is NOT divided by Γ) plus the 30 ordering inequalities, OSQP eps 1e-6.
# ═════════════════════════════════════════════════════════════════════════════
println("\n── Constrained POPS with rattled-ordering rows ─────────────")
println("Computing POPS corrections (delta forest) …")
pops_corr = corrections(Ap, Yw, P; leverage_percentile=0.0)
println("  $(size(pops_corr, 1)) forest members")

forest_θ       = lin_params .+ pops_corr'                       # n_params × n_members
forest_margins = rattle_rows * forest_θ                         # n_rows × n_members
eq_resid       = vec(abs.(b_prime' * forest_θ))                 # |b′·θ_i|
curv           = vec(b_double_prime' * forest_θ)                # b″·θ_i
violates = vec(any(forest_margins .< margin_eV; dims=1)) .|
           (eq_resid .> 1e-6) .| (curv .<= 0)
@printf("  members violating orderings / lattice constraints: %d / %d\n",
        count(violates), length(violates))

function repair_forest_with_orderings(X_train, Y_train, Gamma, constraint_matrix, lb, ub,
                                      pops_corr, lin_params, violates)
    n_members = size(pops_corr, 1)
    con_pops  = Matrix{Float64}(undef, n_members, size(X_train, 2))
    statuses  = fill(:Feasible, n_members)     # members that never needed a QP
    H_s = sparse(X_train' * X_train .+ (1.0 / size(X_train, 1)) .* (Gamma' * Gamma))
    b   = - X_train' * Y_train
    C_phys = Matrix(constraint_matrix / Gamma)

    osqp_model = OSQP.Model()
    for i in 1:n_members
        if !violates[i]
            con_pops[i, :] = lin_params .+ vec(pops_corr[i, :])
        else
            A_full = vcat(X_train[i, :]', C_phys)
            l_full = vcat([Y_train[i]], lb)
            u_full = vcat([Y_train[i]], ub)
            OSQP.setup!(osqp_model; P=H_s, q=b, A=sparse(A_full), l=l_full, u=u_full,
                        max_iter=5_000_000, check_termination=25, verbose=false,
                        eps_abs=1e-6, eps_rel=1e-6)
            results = OSQP.solve!(osqp_model)
            statuses[i] = results.info.status
            con_pops[i, :] = Gamma \ results.x
        end
        i % 1000 == 0 && print("\r  $i / $n_members  (repaired so far: $(count(!=(:Feasible), view(statuses, 1:i)))) …")
    end
    println("\r  done.                                        ")
    return con_pops, statuses
end

con_pops, pops_statuses = repair_forest_with_orderings(Ap, Yw, P, all_constraints,
                                                       lower_bounds, upper_bounds,
                                                       pops_corr, lin_params, violates)
serialize("$(result.dir)/results/rattled_constrained_pops_forest.jls",
          (forest=con_pops, statuses=pops_statuses, rattle_meta=rattle_meta))

println("  repair solve statuses:")
for s in unique(pops_statuses)
    println("    ", s, ": ", count(==(s), pops_statuses))
end
good = [(pops_statuses[i] == :Feasible || pops_statuses[i] == :Solved) &&
        all(isfinite, view(con_pops, i, :)) for i in 1:size(con_pops, 1)]
@printf("  keeping %d / %d members for the proposal cloud\n", sum(good), length(good))
sum(good) >= 2 || error("too few solved members to build a proposal")
con_pops_good = con_pops[good, :]

post_margins = rattle_rows * con_pops_good'
@printf("  post-repair min ordering margin: %+.5f eV  (bound %.0e, OSQP tol 1e-6)\n",
        minimum(post_margins), margin_eV)
@printf("  post-repair max |b′·θ|: %.2e eV/Å,  min b″·θ: %+.4f eV/Å²\n",
        maximum(abs.(b_prime' * con_pops_good')), minimum(b_double_prime' * con_pops_good'))

# ── Committee: unclipped hypercube + rejection against the inequality rows ───
# (the b′ = 0 equality cannot survive a continuous proposal — it is enforced
#  in the pinned repair QPs; rejection tests curvature + the orderings)
ineq_rows_rej = vcat(b_double_prime', rattle_rows)
rej_bounds    = (vcat([1e-9], fill(margin_eV, size(rattle_rows, 1))),
                 vcat([Inf],  fill(Inf,       size(rattle_rows, 1))))

pops_eig, pops_bound = hypercube(con_pops_good; percentile_clipping=0.0)
committee_mat, _ = rejection_sample_hypercube(pops_eig, pops_bound, zeros(n_params),
                                              ineq_rows_rej, rej_bounds;
                                              number_of_committee_members=N_committee,
                                              max_attempts=5_000_000)
writedlm("$(result.dir)/results/rattled_pops_committee_$(N_committee).csv", committee_mat', ',')
committee = [committee_mat[:, i] for i in 1:size(committee_mat, 2)]

# ── Phonon bands of the committee (mean = rattled-constrained model) ─────────
ACEpotentials.Models.set_linear_parameters!(model, constrained_teta)
_, bands_pc, _, _ = phonon_committee(model, committee, result, element;
                                     N_cell=N_cell, file_prefix="rattled_pops_")
n_unstable = count(any(bands_pc[i + 1] .< -0.05) for i in 1:length(committee))
@printf("\nCommittee members with imaginary modes (< −0.05 THz): %d / %d\n",
        n_unstable, length(committee))

println("\nSaved:")
println("  $(result.dir)/results/rattled_constrained_teta.csv")
println("  $(result.dir)/results/rattled_constraint_margins.csv")
println("  $(result.dir)/results/rattled_phonon_bands_comparison_$(N_cell)x$(N_cell)x$(N_cell).png")
println("  $(result.dir)/results/rattled_phonon_{x,freqs}_{nominal,constrained} CSVs")
println("  $(result.dir)/results/rattled_constrained_pops_forest.jls")
println("  $(result.dir)/results/rattled_pops_committee_$(N_committee).csv")
println("  $(result.dir)/results/rattled_pops_phonon_committee_* (plots + freq CSVs)")
