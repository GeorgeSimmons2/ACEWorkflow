# forest_repair_committee_Al_20_4_6A_2_subset_50.jl
#
# The synthesis of the whole investigation.  The POPS box proposal is broken in
# high dim (box draws ~54× the forest scale, mostly phonon-unstable), but the
# forest MEMBERS themselves are honest, data-scaled models that are almost all
# phonon-stable — except a minority driven by high-residual observations.
#
# So: build the committee from real forest members, and REPAIR only the
# unstable minority with a constrained-POPS QP:
#
#   member_i = lin_params + δθ_i           (δθ_i = P⁻¹C⁻¹Apᵢᵀ·(residual_i/leverage_i))
#   if member_i satisfies  Born + b″>0 + eigenmode-curvature rows  → keep as-is
#   else  → re-solve pinning its data row  Ap[i]·θ̃ = Yw[i]  subject to
#            b′=0 (equality) and  Born + b″ + eigenmode  (inequalities), OSQP 1e-6.
#
# Detection & repair use the SAME linear rows (Born from the strain Hessian,
# a_eq from b′/b″, and the 30 mean-model phonon eigenmode curvature rows cached
# by phonon_eigenmode_constraints…jl).  Every member — kept or repaired — is
# then verified with FULL phonon bands (phonon_committee) at its own relaxed
# lattice constant.  Leverage/random members pass untouched; the residual-tail
# minority gets projected onto the stable side, at physical scale where the
# linear rows are valid.
#
# Outputs → models/Al_20_4_6A_2_subset_50_percent/results/forest_repair/ :
#   committee.csv / members.csv               (final committee + per-member table)
#   phonon_committee_* , bands.png            (full-band verification)
#   energy_parity.png / force_parity.png      (test-set parity with committee bounds)
#   energy_error_vs_uncertainty.png / force_… (calibration: error vs committee-bound,
#                                              with bias and error-violation/coverage %)
#
# Run:  julia --project [-t N] scripts/uq/forest_repair_committee_Al_20_4_6A_2_subset_50.jl

using LinearAlgebra, DelimitedFiles, Statistics, Printf, Random, Serialization
using SparseArrays, Unitful, ForwardDiff, CairoMakie, OSQP, StatsBase
using ACEpotentials, ACEWorkflow
using ACEpotentials: potential_energy
using ExtXYZ, AtomsCalculators
import AtomsCalculators: forces
import ACEWorkflow: phonon_committee

Random.seed!(1234)

test_stride = 10      # subsample of the test set for the parity / calibration study

element      = :Al
n_lev        = 5      # highest-leverage forest members in the draw
n_res        = 10     # highest-|residual| forest members (where instability concentrates)
n_rand       = 15     # random forest members (typical — should pass untouched)
f_margin_THz = 0.1    # stability margin for detection + repair (rows are ω² = curvature)
N_cell_bands = 3

# ── Model ────────────────────────────────────────────────────────────────────
result     = load_model(element, 20, 4, 6, 2; dataset_name="subset_50_percent")
model      = result.model
lin_params = result.lin_params
n_params   = length(lin_params)
P          = result.P
println("Model $(result.name): $n_params parameters")

Ap = Diagonal(result.W) * result.A / P
Yw = result.W .* result.Y
λ  = 1.0 / size(Ap, 1)

outdir = "$(result.dir)/results/forest_repair"
mkpath(outdir)
println("Outputs → $outdir")

# ── Leverage / residual / forest corrections ─────────────────────────────────
println("Leverage & residuals …")
C   = Symmetric(Ap' * Ap .+ λ .* (P' * P))
Cf  = cholesky(C)
AtX = Cf \ Matrix(Ap')
θ̃   = Cf \ (Ap' * Yw)
leverage = vec(sum(Ap' .* AtX; dims=1))
residual = Yw .- Ap * θ̃
forest_member(i) = lin_params .+ (P \ (AtX[:, i] .* (residual[i] / leverage[i])))

# ── Constraint rows: Born, a_eq (b′,b″), and cached eigenmode curvature rows ──
println("Relaxing mean & building Born / a_eq rows …")
a_eq = ACEWorkflow.relax_lattice_constant(model, element)
sys0 = ACEWorkflow.Elasticity.reference_system(element; a=a_eq)
L0   = ustrip.(ACEWorkflow.Elasticity.lattice_matrix(sys0.cell.cell_vectors))
eV_to_GPa = 160.2176621 / abs(det(L0))
H_basis = elastic_hessian_basis(model; element=element, a=a_eq)
c11_0 = reshape(H_basis, 36, n_params)[1, :]
c12_0 = reshape(H_basis, 36, n_params)[7, :]
c44_0 = reshape(H_basis, 36, n_params)[22, :]
born(θ) = (dot(c11_0, θ)*eV_to_GPa, dot(c12_0, θ)*eV_to_GPa, dot(c44_0, θ)*eV_to_GPa)

function lattice_basis(a_val)
    sys = ACEWorkflow.Elasticity.reference_system(element; a=a_val)
    ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(sys, model))
end
b_prime        = ForwardDiff.derivative(lattice_basis, a_eq)
b_double_prime = ForwardDiff.derivative(a -> ForwardDiff.derivative(lattice_basis, a), a_eq)

cache = deserialize("$(result.dir)/results/phonon_eig_rows.jls")
abs(cache.a_eq - a_eq) < 1e-4 || error("cached eigenmode rows built at a=$(cache.a_eq) ≠ $a_eq")
eig_rows  = cache.rows                     # n_eig × n_params, row·θ = ω²(q,ν)
n_eig     = size(eig_rows, 1)
ω2_margin = (f_margin_THz / FREQ_THz)^2
println("Loaded $n_eig cached eigenmode curvature rows (built at a=$(round(cache.a_eq;digits=5)))")

born_rows   = vcat(c44_0', (c11_0 .- c12_0)', (c11_0 .+ 2 .* c12_0)')
born_lower  = [0.1, 1.0, 0.1]
ineq_matrix = vcat(born_rows, b_double_prime', eig_rows)          # 3 + 1 + n_eig
ineq_lower  = vcat(born_lower, [1e-9], fill(ω2_margin, n_eig))
feasible(θ) = all(ineq_lower .<= ineq_matrix * θ)

# ── The committee draw ───────────────────────────────────────────────────────
lev_idx = sortperm(leverage; rev=true)[1:n_lev]
res_idx = Int[]
for i in sortperm(abs.(residual); rev=true)
    i in lev_idx && continue
    push!(res_idx, i); length(res_idx) == n_res && break
end
taken   = Set(vcat(lev_idx, res_idx))
rand_idx = Int[]
while length(rand_idx) < n_rand
    i = rand(1:length(Yw))
    (i in taken) && continue
    push!(rand_idx, i); push!(taken, i)
end
selected = vcat(lev_idx, res_idx, rand_idx)
groups   = vcat(fill("leverage", n_lev), fill("residual", n_res), fill("random", n_rand))
println("\nCommittee draw: $(n_lev) leverage + $(n_res) residual + $(n_rand) random = $(length(selected)) members")

# ── Constrained-POPS repair QP (pin data row + b′=0 + inequalities) ──────────
Hqp    = sparse(Ap' * Ap .+ λ .* (P' * P))
qqp    = -(Ap' * Yw)
phys_t = sparse(vcat(b_prime', ineq_matrix) / P)        # θ̃-space physics rows
osqp   = OSQP.Model()
function repair(oi)
    A_full = vcat(sparse(Ap[oi, :]'), phys_t)
    l_full = vcat([Yw[oi]], [0.0], ineq_lower)
    u_full = vcat([Yw[oi]], [0.0], fill(Inf, length(ineq_lower)))
    OSQP.setup!(osqp; P=Hqp, q=qqp, A=A_full, l=l_full, u=u_full,
                max_iter=2_000_000, check_termination=25, verbose=false, eps_abs=1e-6, eps_rel=1e-6)
    r = OSQP.solve!(osqp)
    return P \ r.x, r.info.status
end

# ── Build committee: keep feasible forest members, repair the rest ───────────
committee = Vector{Vector{Float64}}(undef, length(selected))
was_repaired = falses(length(selected))
statuses = fill(:kept, length(selected))
for (k, i) in enumerate(selected)
    θ = forest_member(i)
    if feasible(θ)
        committee[k] = θ
    else
        was_repaired[k] = true
        θr, st = repair(i)
        statuses[k] = st
        committee[k] = feasible(θr) ? θr : θ     # fall back to raw if QP failed
    end
end
@printf("Flagged for repair (violate rows): %d / %d  (leverage %d, residual %d, random %d)\n",
        count(was_repaired), length(selected),
        count(was_repaired[groups .== "leverage"]),
        count(was_repaired[groups .== "residual"]),
        count(was_repaired[groups .== "random"]))
bad_qp = count(k -> was_repaired[k] && statuses[k] != :Solved, 1:length(selected))
bad_qp > 0 && @warn "$bad_qp repair QPs did not solve cleanly (status ≠ :Solved)"
writedlm("$outdir/committee.csv", reduce(hcat, committee)', ',')

# ── Verify: full phonon bands (each member at its own relaxed a_eq) ──────────
println("\n── Full-band verification (phonon_committee) ───────────────")
ACEpotentials.Models.set_linear_parameters!(model, lin_params)
x_vals, all_freqs, x_ticks, labels =
    phonon_committee(model, committee, result, element; N_cell=N_cell_bands, file_prefix="forest_repair/")

min_f  = [minimum(all_freqs[i + 1]) for i in 1:length(committee)]
n_imag = count(<(-0.05), min_f)
println("\n  member results (‖δθ‖, min band ω):")
for (k, i) in enumerate(selected)
    C11, C12, C44 = born(committee[k])
    @printf("    %-9s obs %6d %s: ‖δθ‖=%.3f  C44=%6.1f  min ω=%+8.4f %s\n",
            groups[k], i, was_repaired[k] ? "[repaired]" : "[kept]    ",
            norm(committee[k] .- lin_params), C44, min_f[k], min_f[k] < -0.05 ? "✗" : "✓")
end
@printf("\n  → %d / %d committee members phonon-UNSTABLE after repair\n", n_imag, length(committee))

# ── Per-member CSV + figure ──────────────────────────────────────────────────
open("$outdir/members.csv", "w") do io
    println(io, "group,obs,repaired,dtheta_norm,C11_GPa,C12_GPa,C44_GPa,min_freq_THz")
    for (k, i) in enumerate(selected)
        C11, C12, C44 = born(committee[k])
        @printf(io, "%s,%d,%d,%.5f,%.3f,%.3f,%.3f,%.4f\n",
                groups[k], i, was_repaired[k], norm(committee[k] .- lin_params), C11, C12, C44, min_f[k])
    end
end

fig = Figure(size=(900, 540))
ax  = Axis(fig[1, 1]; xlabel="Wave vector", ylabel="Frequency (THz)",
           title="$(result.name) — forest committee with residual-tail repair",
           xticks=(x_ticks, labels), xgridvisible=false)
for k in 1:length(committee)
    col = min_f[k] < -0.05 ? RGBAf(0.80, 0.15, 0.15, 0.55) :
          was_repaired[k]  ? RGBAf(0.10, 0.55, 0.20, 0.55) : RGBAf(0.5, 0.5, 0.5, 0.35)
    for b in 1:size(all_freqs[k+1], 1)
        lines!(ax, x_vals, all_freqs[k+1][b, :]; color=col, linewidth=1.0)
    end
end
for b in 1:size(all_freqs[1], 1)
    lines!(ax, x_vals, all_freqs[1][b, :]; color=RGBAf(0.0, 0.3, 0.7, 0.95), linewidth=2.0)
end
hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=0.8)
vlines!(ax, x_ticks; color=(:black, 0.3), linewidth=0.8)
Legend(fig[1, 2],
       [LineElement(color=RGBAf(0.5,0.5,0.5,0.8), linewidth=2),
        LineElement(color=RGBAf(0.10,0.55,0.20,0.8), linewidth=2),
        LineElement(color=RGBAf(0.80,0.15,0.15,0.8), linewidth=2),
        LineElement(color=RGBAf(0.0,0.3,0.7,0.95), linewidth=2.5)],
       ["kept (stable)", "repaired", "unstable", "mean model"])
Label(fig[0, :], "$(count(was_repaired)) repaired, $n_imag/$(length(committee)) unstable after"; fontsize=13)
save("$outdir/bands.png", fig)
display(fig)

# ═════════════════════════════════════════════════════════════════════════════
#  Test-set parity + calibration/coverage study (committee UQ)
#  Mirrors the legacy elastic_constraints.jl studies: the repaired committee is
#  installed on the model; test energies/forces are predicted with per-point
#  committee min/max bounds; parity plots and an error-vs-committee-bound
#  histogram report bias and the error-violation (miscoverage) rate.
# ═════════════════════════════════════════════════════════════════════════════
println("\n── Test-set parity + coverage study ────────────────────────")
ACEpotentials.Models.set_committee!(model, committee)
ACEpotentials.Models.set_linear_parameters!(model, lin_params)

testing_configs = ExtXYZ.load("data/Al/manual_df_test_Al.xyz")[1:test_stride:end]
println("  $(length(testing_configs)) test configs (stride $test_stride)")

pred_E = Float64[]; true_E = Float64[]; loE = Float64[]; hiE = Float64[]
pred_F = Float64[]; true_F = Float64[]; loF = Float64[]; hiF = Float64[]
for config in testing_configs
    E, co_E = @committee potential_energy(config, model)
    push!(pred_E, ustrip(E)); push!(loE, minimum(ustrip.(co_E))); push!(hiE, maximum(ustrip.(co_E)))
    push!(true_E, ustrip(config[:dft_energy]))
    if haskey(config[1], :dft_forces)
        F, co_F = @committee forces(config, model)
        append!(true_F, reduce(vcat, ustrip.([at[:dft_forces] for at in config])))
        append!(pred_F, reduce(vcat, ustrip.(F)))
        for i in eachindex(co_F[1])
            fi = reduce(hcat, ustrip(co_F[k][i]) for k in eachindex(co_F))
            append!(loF, vec(minimum(fi; dims=2))); append!(hiF, vec(maximum(fi; dims=2)))
        end
    end
end

function parity_plot(t, p, lo, hi, xl, yl, path; col=:steelblue)
    rmse = sqrt(mean((p .- t).^2))
    f = Figure(size=(600, 600))
    ax = Axis(f[1, 1]; xlabel=xl, ylabel=yl, title="RMSE = $(round(rmse, sigdigits=3))")
    errorbars!(ax, t, p, p .- lo, hi .- p; whiskerwidth=6, color=(col, 0.5))
    scatter!(ax, t, p; color=col, markersize=6)
    lims = extrema([t; p]); lines!(ax, collect(lims), collect(lims); color=:black, linestyle=:dash)
    save(path, f); return rmse
end
eR = parity_plot(true_E, pred_E, loE, hiE, "DFT energy (eV)", "ACE energy (eV)", "$outdir/energy_parity.png")
@printf("  energy RMSE = %.4g eV\n", eR)
if !isempty(true_F)
    fR = parity_plot(true_F, pred_F, loF, hiF, "DFT force (eV/Å)", "ACE force (eV/Å)", "$outdir/force_parity.png"; col=:tomato)
    @printf("  force  RMSE = %.4g eV/Å\n", fR)
end

# error vs committee-bound histogram → bias + error-violation (miscoverage) rate
function error_vs_uncertainty(true_vals, pred_vals, lo_vals, hi_vals; label, path, nbins=60)
    true_vals, pred_vals, lo_vals, hi_vals = Float64.(true_vals), Float64.(pred_vals), Float64.(lo_vals), Float64.(hi_vals)
    errors = true_vals .- pred_vals
    mae = mean(abs.(errors))
    norm_err = errors ./ mae
    bound_err = vcat((true_vals .- lo_vals) ./ mae, (true_vals .- hi_vals) ./ mae)
    violated = (true_vals .< lo_vals) .| (true_vals .> hi_vals)
    ev = mean(violated); bias = mean(errors) / mae
    lim = maximum(abs.(vcat(norm_err, bound_err))); edges = range(-lim, lim; length=nbins + 1)
    d1 = (h = fit(Histogram, norm_err, edges).weights; max.(h ./ (sum(h) * step(edges)), 1e-3))
    d2 = (h = fit(Histogram, bound_err, edges).weights; max.(h ./ (sum(h) * step(edges)), 1e-3))
    f = Figure(size=(560, 520))
    ax = Axis(f[2, 1]; xlabel="Error / MAE", ylabel="density", yscale=log10,
              title="$label — coverage $(round((1-ev)*100, digits=1))%  (EV $(round(ev*100, digits=1))%, bias $(round(bias*100, digits=0))%)")
    l1 = stairs!(ax, edges[1:end-1], d1; step=:post, color=:black, linewidth=2)
    l2 = stairs!(ax, edges[1:end-1], d2; step=:post, color=:orange, linewidth=2)
    ylims!(ax, 1e-3, maximum(vcat(d1, d2)) * 3)
    Legend(f[1, 1], [l1, l2], ["test error", "committee bound"]; orientation=:horizontal, tellwidth=false)
    save(path, f); return ev, bias
end
evE, biasE = error_vs_uncertainty(true_E, pred_E, loE, hiE; label="Energy", path="$outdir/energy_error_vs_uncertainty.png")
@printf("  energy coverage = %.1f%%  (error-violation %.1f%%, bias %.0f%% MAE)\n", (1-evE)*100, evE*100, biasE*100)
if !isempty(true_F)
    evF, biasF = error_vs_uncertainty(true_F, pred_F, loF, hiF; label="Force", path="$outdir/force_error_vs_uncertainty.png")
    @printf("  force  coverage = %.1f%%  (error-violation %.1f%%, bias %.0f%% MAE)\n", (1-evF)*100, evF*100, biasF*100)
end
ACEpotentials.Models.set_linear_parameters!(model, lin_params)

println("\nAll outputs saved to $outdir/")
println("  committee.csv, members.csv, bands.png, phonon_committee_*")
println("  energy_parity.png, force_parity.png, energy/force_error_vs_uncertainty.png")
