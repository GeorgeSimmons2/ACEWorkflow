# phonon_eigenmode_constraints_Al_20_4_6A_2_subset_50.jl
#
# Physics-informed phonon-stability prior for the POPS committee of
# Al_20_4_6A_2 (subset_50_percent), built from the MEAN model's own phonon
# eigenmodes across the FCC band path — then compared head-to-head against the
# "old" method (Born + a_eq constraints only).
#
# Idea (all rows LINEAR in θ, so they drop straight into the QP + rejection):
#   1. ONE Hessian of the mean model on an N_super³ conventional supercell
#      → force constants → dynamical matrix D(q) at any q (src/Phonons).
#   2. At each commensurate band-path wavevector q, diagonalise D(q) to get the
#      phonon polarisation vectors e_ν(q).  The real-space displacement pattern
#      is  u_j = Re(e_ν · exp(i q·r_j)) / √m  (phase factors applied per atom).
#   3. Central finite differences of the basis energy along ±A·u give the
#      curvature  d²E/dA² = Σ_k θ_k (uᵀ H_k u), LINEAR in θ.  Divided by ‖u‖²
#      it is the Rayleigh quotient = ω²(q,ν) in eV/Å²/amu, so the row
#          row · θ ≥ (f_margin_THz / FREQ_THz)²
#      demands positive curvature (real frequency) along that phonon.
#      Self-check: row · lin_params must equal the eigenvalue ω²(q,ν).
#
#   COMMENSURABILITY: q must be commensurate with the displaced supercell, else
#   the pattern is discontinuous across the PBC boundary and the curvature is
#   not the true ω²(q).  We therefore take the commensurate q-points that lie
#   ON the Γ→X→U→L→Γ→K path (integer lattice points of the N_super supercell,
#   N_super % 4 == 0 so X,U,L,K are all commensurate).
#
# These phonon rows are combined with the a_eq minimum constraints
#   b′(a_eq)·θ = 0  (equality, mean fit)   and   b″(a_eq)·θ > 0  (row)
# and the Born rows (C44, C11−C12, C11+2C12 > 0), applied to the POPS forest
# and rejection-sampled.  OLD = Born+a_eq only; NEW = Born+a_eq+phonon.  Both
# committees are pushed through full phonon bands (phonon_committee) and the
# imaginary-mode counts compared.
#
# Outputs (models/Al_20_4_6A_2_subset_50_percent/results/):
#   phonon_eig_rows.jls                         (cached curvature rows + meta)
#   phonon_eig_committee_{old,new}_10.csv
#   phonon_eig_{old,new}_phonon_committee_*      (per-variant band plots + CSVs)
#   phonon_eig_bands_comparison_old_vs_new.png   (stacked panels)
#   phonon_eig_summary.csv
#
# Run:  julia --project [-t N] scripts/uq/phonon_eigenmode_constraints_Al_20_4_6A_2_subset_50.jl

using LinearAlgebra, DelimitedFiles, Statistics, Printf, Random, Serialization
using StaticArrays, Unitful, ForwardDiff, CairoMakie
using ACEpotentials, ACEWorkflow
using AtomsBuilder
using AtomsCalculatorsUtilities.SitePotentials: hessian
import ACEWorkflow: phonon_committee

Random.seed!(1234)

element        = :Al
N_super        = 4      # supercell for the curvature rows (N%4==0 → X,U,L,K commensurate)
N_cell_bands   = 3      # supercell for the full-band verification (phonon_committee)
f_margin_THz   = 0.2    # required frequency along each constrained phonon direction
ift_da_max     = 0.1    # Å — reject committee draws whose a_eq drifts beyond IFT validity
N_committee    = 10     # committee members per variant pushed through phonon bands
norm_cap_pct   = 100.0  # cap ‖θ−θ̄‖ at this percentile of the forest correction norms.
                        # The hypercube proposal draws ~80× the physical correction scale
                        # in high dim; this cap keeps committee members inside the forest's
                        # own norm envelope, and gives a coverage statement: no member is
                        # more extreme than norm_cap_pct% of the actual POPS corrections.
@assert N_super % 4 == 0 "N_super must be a multiple of 4 so X, U, L, K are commensurate"

# ── Model + POPS forest ──────────────────────────────────────────────────────
result     = load_model(element, 20, 4, 6, 2; dataset_name="subset_50_percent")
model      = result.model
lin_params = result.lin_params
n_params   = length(lin_params)
P          = result.P
println("Model $(result.name): $n_params parameters, $(length(result.Y)) design rows")

Ap = Diagonal(result.W) * result.A / P
Yw = result.W .* result.Y

println("Computing POPS corrections (delta forest) …")
pops_corr = corrections(Ap, Yw, P; leverage_percentile=0.0)
println("  $(size(pops_corr, 1)) forest members")

# ── Norm-cap: physical scale of the forest, and the committee cap ────────────
forest_norms = vec(sqrt.(sum(abs2, pops_corr; dims=2)))   # ‖θ_i − θ̄‖ per member
norm_cap     = quantile(forest_norms, norm_cap_pct / 100)
println("Forest correction-norm percentiles ‖δθ‖:")
for p in (50, 75, 90, 95, 99, 100)
    @printf("   %3d%%: %.4f\n", p, quantile(forest_norms, p / 100))
end
@printf("→ committee norm cap = %.4f  (%.0f%% coverage of forest correction norms)\n",
        norm_cap, norm_cap_pct)

# ── Reference geometry, Born rows, a_eq rows ─────────────────────────────────
println("Relaxing mean model …")
a_eq = ACEWorkflow.relax_lattice_constant(model, element)
@printf("  a_eq = %.6f Å\n", a_eq)

sys0 = ACEWorkflow.Elasticity.reference_system(element; a=a_eq)
L0   = ustrip.(ACEWorkflow.Elasticity.lattice_matrix(sys0.cell.cell_vectors))
eV_to_GPa = 160.2176621 / abs(det(L0))

H_basis = elastic_hessian_basis(model; element=element, a=a_eq)
c11_0 = reshape(H_basis, 36, n_params)[1, :]
c12_0 = reshape(H_basis, 36, n_params)[7, :]
c44_0 = reshape(H_basis, 36, n_params)[22, :]

function lattice_basis(a_val)
    sys = ACEWorkflow.Elasticity.reference_system(element; a=a_val)
    ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(sys, model))
end
b_prime        = ForwardDiff.derivative(lattice_basis, a_eq)
b_double_prime = ForwardDiff.derivative(a -> ForwardDiff.derivative(lattice_basis, a), a_eq)

born_rows  = vcat(c44_0',                 # C44        ≥ 0.1
                  (c11_0 .- c12_0)',      # C11 − C12  ≥ 1.0
                  (c11_0 .+ 2 .* c12_0)', # C11 + 2C12 ≥ 0.1
                  b_double_prime')        # b″·θ       ≥ 1e-9  (a_eq is a minimum)
born_lower = [0.1, 1.0, 0.1, 1e-9]

# ── Phonon eigenmode curvature rows ──────────────────────────────────────────
# integer lattice points of the N_super supercell that lie on segment v1→v2
function segment_int_points(v1, v2)
    d = v2 .- v1
    g = gcd(gcd(abs(d[1]), abs(d[2])), abs(d[3]))
    g = max(g, 1)
    step = d .÷ g
    return [v1 .+ j .* step for j in 0:g]
end

function build_phonon_rows(model, sys_super, sys_prim, a_eq, N, f_margin_THz, lin_params)
    println("  precomputing force constants (one supercell Hessian) …")
    fc  = precompute_force_constants(sys_prim, sys_super, model)
    pos = [SVector{3,Float64}(ustrip.(sys_super[i].position)) for i in 1:length(sys_super)]

    verts = Dict(:Γ=>[0,0,0], :X=>[0,N,0], :U=>[N÷4,N,N÷4],
                 :L=>[N÷2,N÷2,N÷2], :K=>[3N÷4,3N÷4,0])
    segs  = [(:Γ,:X),(:X,:U),(:U,:L),(:L,:Γ),(:Γ,:K)]
    intpts = Vector{Vector{Int}}()
    for (s1, s2) in segs, p in segment_int_points(verts[s1], verts[s2])
        push!(intpts, p)
    end
    unique!(intpts)
    filter!(p -> p != [0, 0, 0], intpts)     # drop Γ (acoustic translations)
    @printf("  %d commensurate band-path q-points (× up to 3 modes)\n", length(intpts))

    ω2_margin = (f_margin_THz / FREQ_THz)^2
    rows = Vector{Vector{Float64}}(); lows = Float64[]; meta = Any[]
    maxerr = 0.0
    for n in intpts
        q = (2π / (N * a_eq)) .* n
        F = eigen(Hermitian(dynamical_matrix_from_fc(fc, q)))
        evals, evecs = real.(F.values), F.vectors
        for ν in 1:3
            e = evecs[:, ν]
            p2 = 0.0
            for r in pos
                p2 += sum(abs2, real.(e .* exp(im * dot(q, r))))
            end
            mode = repeat(ComplexF64.(e), length(sys_super))
            E_design(A) = apply_mode_design(model, sys_super, mode, q, A)
            dE(A) = ForwardDiff.derivative(E_design, A)
            h = 1e-5
            row = (dE(h) .- dE(-h)) ./ (2h) ./ p2
            ω2_row = dot(row, lin_params)
            maxerr = max(maxerr, abs(ω2_row - evals[ν]) / max(abs(evals[ν]), 1e-6))
            f_mean = sign(evals[ν]) * sqrt(abs(evals[ν])) * FREQ_THz
            f_mean > f_margin_THz || continue    # skip modes the mean can't clear
            push!(rows, row); push!(lows, ω2_margin); push!(meta, (n=n, ν=ν, f_mean=f_mean))
        end
    end
    return reduce(vcat, transpose.(rows)), lows, meta, maxerr
end

sys_super = bulk(element; a=a_eq*u"Å", cubic=true) * (N_super, N_super, N_super)
sys_prim  = bulk(element; a=a_eq*u"Å")
@printf("Building phonon eigenmode rows on %d-atom supercell …\n", length(sys_super))

rows_file = "$(result.dir)/results/phonon_eig_rows.jls"
phonon_rows, phonon_lower, phonon_meta, row_maxerr =
    build_phonon_rows(model, sys_super, sys_prim, a_eq, N_super, f_margin_THz, lin_params)
@printf("  built %d curvature rows;  self-check max |row·θ − ω²|/ω² = %.2e\n",
        size(phonon_rows, 1), row_maxerr)
row_maxerr < 1e-3 || @warn "phonon-row self-check exceeded 1e-3 — commensurability/normalisation issue?"
serialize(rows_file, (rows=phonon_rows, lower=phonon_lower, meta=phonon_meta,
                      a_eq=a_eq, N_super=N_super, f_margin_THz=f_margin_THz))

ACEpotentials.Models.set_linear_parameters!(model, lin_params)   # restore mean

# ── Constraint sets: OLD (Born + a_eq) vs NEW (+ phonon eigenmode rows) ──────
ineq_old   = vcat(born_rows)
lower_old  = copy(born_lower)
ineq_new   = vcat(born_rows, phonon_rows)
lower_new  = vcat(born_lower, phonon_lower)

@printf("\nNominal model violates: Born %d/4, phonon %d/%d\n",
        count(born_rows * lin_params .< born_lower),
        count(phonon_rows * lin_params .< phonon_lower), size(phonon_rows, 1))

# ── Constrained ridge (mean) with b′=0 equality + inequalities ───────────────
using SparseArrays, OSQP
function constrained_mean(ineq_mat, ineq_low; lambda = 1.0 / size(Ap, 1))
    A_c = vcat(b_prime', ineq_mat)                    # b′ equality on top
    l_c = vcat([0.0], ineq_low)
    u_c = vcat([0.0], fill(Inf, length(ineq_low)))
    H = Ap' * Ap .+ lambda .* (P' * P)
    m = OSQP.Model()
    OSQP.setup!(m; P=sparse(H), q=-(Ap' * Yw), A=sparse(A_c / P), l=l_c, u=u_c,
                max_iter=2_000_000, check_termination=500, verbose=false,
                eps_abs=1e-6, eps_rel=1e-6)
    return P \ OSQP.solve!(m).x
end

# ── IFT lattice-shift trust gate (reject draws whose a_eq drifts too far) ────
K_ref = dot(lin_params, b_double_prime)
ift_da(θ) = -dot(b_prime, θ .- lin_params) / K_ref

# ── Committee for one variant: rejection-sample the box against the rows ─────
function run_variant(label, ineq_mat, ineq_low, prefix)
    println("\n══ $label ══════════════════════════════════════════════════")
    θ_mean = constrained_mean(ineq_mat, ineq_low)
    @printf("  constrained mean: C11=%.1f C12=%.1f C44=%.1f GPa  b′·θ=%.1e  b″·θ=%.3f\n",
            dot(c11_0, θ_mean)*eV_to_GPa, dot(c12_0, θ_mean)*eV_to_GPa,
            dot(c44_0, θ_mean)*eV_to_GPa, dot(b_prime, θ_mean), dot(b_double_prime, θ_mean))

    hyp_eig, hyp_bound = hypercube(pops_corr)          # unclipped, raw forest
    n_check = Ref(0); n_lin = Ref(0); n_da = Ref(0)
    predicate = θ -> begin
        n_check[] += 1
        all(ineq_low .<= ineq_mat * θ) || return false
        n_lin[] += 1
        abs(ift_da(θ)) <= ift_da_max || return false
        n_da[] += 1
        norm(θ .- lin_params) <= norm_cap          # physical-scale cap (coverage guarantee)
    end
    committee_mat, _ = rejection_sample_hypercube(hyp_eig, hyp_bound, θ_mean, predicate;
                                                  number_of_committee_members=N_committee,
                                                  max_attempts=20_000_000)
    @printf("  rejection funnel: %d drawn → %d rows → %d |δa| → %d norm-cap accepted\n",
            n_check[], n_lin[], n_da[], N_committee)
    writedlm("$(result.dir)/results/$(prefix)_committee_$(N_committee).csv", committee_mat', ',')
    committee = [committee_mat[:, i] for i in 1:size(committee_mat, 2)]

    ACEpotentials.Models.set_linear_parameters!(model, θ_mean)   # member 0 = this mean
    x_vals, all_freqs, x_ticks, labels =
        phonon_committee(model, committee, result, element; N_cell=N_cell_bands, file_prefix="$(prefix)_")
    ACEpotentials.Models.set_linear_parameters!(model, lin_params)

    δnorms = [norm(c .- lin_params) for c in committee]
    min_f = [minimum(all_freqs[i + 1]) for i in 1:N_committee]
    n_imag = count(<(-0.05), min_f)
    @printf("  committee ‖δθ‖ ∈ [%.3f, %.3f] (cap %.3f);  min band frequencies (THz):\n",
            minimum(δnorms), maximum(δnorms), norm_cap)
    for i in 1:N_committee
        @printf("    member %2d: ‖δθ‖=%.3f  min ω=%+8.4f %s\n",
                i, δnorms[i], min_f[i], min_f[i] < -0.05 ? "✗" : "✓")
    end
    @printf("  → %d / %d members phonon-UNSTABLE\n", n_imag, N_committee)
    return (; label, θ_mean, committee, all_freqs, x_vals, x_ticks, labels, min_f, δnorms, n_imag,
            funnel=(n_check[], n_lin[], n_da[]))
end

old = run_variant("OLD  (Born + a_eq only)",           ineq_old, lower_old, "phonon_eig_old")
new = run_variant("NEW  (Born + a_eq + phonon rows)",  ineq_new, lower_new, "phonon_eig_new")

# ── Stacked comparison figure: OLD (top) vs NEW (bottom) ─────────────────────
fig = Figure(size=(880, 780))
for (row, v) in enumerate((old, new))
    ax = Axis(fig[row, 1];
              ylabel="Frequency (THz)",
              title="$(v.label)   —   $(v.n_imag)/$N_committee members unstable",
              xticks=(v.x_ticks, v.labels), xgridvisible=false)
    row == 2 && (ax.xlabel = "Wave vector")
    # committee members: grey if stable, red if any imaginary mode
    for i in 1:N_committee
        col = v.min_f[i] < -0.05 ? RGBAf(0.80, 0.15, 0.15, 0.45) : RGBAf(0.5, 0.5, 0.5, 0.35)
        for b in 1:size(v.all_freqs[i+1], 1)
            lines!(ax, v.x_vals, v.all_freqs[i+1][b, :]; color=col, linewidth=1.0)
        end
    end
    # constrained mean on top
    for b in 1:size(v.all_freqs[1], 1)
        lines!(ax, v.x_vals, v.all_freqs[1][b, :]; color=RGBAf(0.0, 0.3, 0.7, 0.95), linewidth=2.0)
    end
    hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=0.8)
    vlines!(ax, v.x_ticks; color=(:black, 0.3), linewidth=0.8)
end
Label(fig[0, :], "POPS committee phonons — $(result.name): eigenmode-curvature constraints vs Born-only";
      fontsize=14)
save("$(result.dir)/results/phonon_eig_bands_comparison_old_vs_new.png", fig)
display(fig)

# ── Summary ──────────────────────────────────────────────────────────────────
open("$(result.dir)/results/phonon_eig_summary.csv", "w") do io
    println(io, "# $(result.name); f_margin=$f_margin_THz THz; N_super=$N_super; " *
                "$(size(phonon_rows,1)) phonon rows; norm_cap=$(round(norm_cap; digits=4)) " *
                "($(norm_cap_pct)% forest coverage)")
    println(io, "variant,n_members,n_unstable,min_freq_THz,max_dtheta,drawn,passed_rows,passed_da")
    for v in (old, new)
        @printf(io, "%s,%d,%d,%.4f,%.4f,%d,%d,%d\n",
                v.label, N_committee, v.n_imag, minimum(v.min_f), maximum(v.δnorms),
                v.funnel[1], v.funnel[2], v.funnel[3])
    end
end

println("\n══ RESULT ══════════════════════════════════════════════════")
@printf("  norm cap = %.4f  (%.0f%% coverage of forest correction norms)\n", norm_cap, norm_cap_pct)
@printf("  OLD (Born+a_eq):            %d / %d committee members phonon-unstable\n", old.n_imag, N_committee)
@printf("  NEW (+phonon eigenmodes):   %d / %d committee members phonon-unstable\n", new.n_imag, N_committee)
println("\nSaved to $(result.dir)/results/:")
println("  phonon_eig_bands_comparison_old_vs_new.png, phonon_eig_summary.csv")
println("  phonon_eig_committee_{old,new}_10.csv, phonon_eig_{old,new}_phonon_committee_*")
