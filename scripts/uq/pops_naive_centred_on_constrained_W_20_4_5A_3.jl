# pops_naive_centred_on_constrained_W_20_4_5A_3.jl
#
# QUESTION. Does re-centring a NAIVE POPS committee on the repulsive-core
# constrained parameters buy you core-correct EOS curves for free -- i.e. is the
# rejection step actually doing work, or does the constrained centre alone drag
# the whole cloud into the feasible set?
#
# The constrained fit satisfies b(a)·θ >= 0 on the core grid by construction, but
# the POPS cloud is a bounding box in the eigenbasis of the correction cloud, and
# nothing about that box respects the half-spaces b(a)·θ >= 0. Re-centring shifts
# the box; it does not shrink or tilt it. So the null hypothesis is that a good
# fraction of members still go negative in the core. This script measures that
# fraction rather than assuming it.
#
# THREE ARMS, all with the same cloud geometry, differing only in centre and in
# whether rejection is applied:
#
#   A  naive @ RLS          corrections(...)                 + sample_hypercube
#   B  naive @ constrained  corrections(...; coeffs = θ_con) + sample_hypercube
#   C  rejection @ constr.  same cloud as B                  + rejection_sample_hypercube
#
# Arm B is the thing being tested. Arm A is the control that shows what the
# unconstrained centre does, and arm C is the reference that is feasible by
# construction. Arms B and C share a cloud, so any difference between them is
# attributable to rejection alone.
#
# FEASIBILITY CRITERION -- the same one the constrained fit was solved under
# (scripts/repulsive_core/ZBL_core_ACE_correction.jl): the ACE energy of the
# 1-atom BCC cell must be non-negative across the compressed range
# a in [0.21, 2.18] Å. Violations are counted on two grids:
#   * the CONSTRAINT grid (50 points) the QP actually saw, and
#   * a 4x finer AUDIT grid, which catches members that thread between the nodes
#     the QP constrained -- a failure mode the QP itself cannot see.
# Monotonicity (dE/da < 0 through the core) is reported as a secondary check:
# non-negative but non-monotone is still not repulsive-core behaviour.
#
# MEMORY. POPSRegression.corrections computes leverage as diag(X*A), which
# materialises an M x M temporary: 146126^2 * 8 B = 171 GB for this model. The
# local pops_cloud below computes the same leverage row-by-row instead, which is
# arithmetically identical and costs nothing, keeping the peak near ~10 GB (the
# design matrix plus C\X'). That is the ONLY difference from the library call.
#
# OUTPUTS -> models/W_20_4_5A_3/results/pops_naive_centred_on_constrained/
#   eos_curves_<arm>.csv       n_grid x n_members energy curves
#   member_diagnostics.csv     per (arm, member): violations, min E, monotonicity
#   arm_summary.csv            violating-member fractions per arm
#   eos_bands_three_arms.png/pdf
#
# Run:  julia --project scripts/uq/pops_naive_centred_on_constrained_W_20_4_5A_3.jl [n_members] [lev_pct]

using LinearAlgebra, DelimitedFiles, Statistics, Printf, Random, SHA
using ACEpotentials, ACEWorkflow, AtomsBuilder, Unitful, CairoMakie
using ACEpotentials.Models: potential_energy_basis

@async while true; flush(stdout); sleep(5); end     # Julia block-buffers to files

element     = :W
n_members   = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 50
lev_pct     = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.5
sample_seed = 20260803

Random.seed!(sample_seed)

result = load_model(element, 20, 4, 5, 3)
model  = result.model
lin    = result.lin_params
θ_con  = vec(readdlm("$(result.dir)/positive_core_constrained_parameters.csv", ','))
@assert length(θ_con) == length(lin)

outdir = "$(result.dir)/results/pops_naive_centred_on_constrained"; mkpath(outdir)

Ap = Diagonal(result.W) * result.A / result.P
Yw = result.W .* result.Y
P  = result.P
M, K = size(Ap); λ = 1.0 / M
@printf("%s: %d rows × %d params\n", result.name, M, K)

# ── Core-region energy bases ─────────────────────────────────────────────────
# b(a) is the per-basis energy of the 1-atom BCC cell at lattice constant a, so
# E(a) = b(a)·θ is linear in the parameters and the whole study is a few matrix
# products once these are built.

bulk_basis(a) = ustrip.(potential_energy_basis(bulk(element, a=a*u"Å"), model))

a_constraint = collect(LinRange(0.21, 2.18, 50))    # grid the QP was solved on
a_audit      = collect(LinRange(0.21, 2.18, 200))   # finer grid for the audit
a_plot       = vcat(a_audit, collect(LinRange(2.18, 4.36, 50))[2:end])

@printf("building energy bases on %d lattice constants…\n", length(a_plot))
B_plot       = reduce(hcat, bulk_basis.(a_plot))'          # n_plot × K
idx_con      = [findmin(abs.(a_plot .- a))[2] for a in a_constraint]
idx_audit    = 1:length(a_audit)
B_constraint = B_plot[idx_con, :]

# ── POPS cloud (memory-lean leverage; see header) ────────────────────────────

function pops_cloud(X::AbstractMatrix{Float64}, Y::Vector{Float64}, Gamma::AbstractMatrix{Float64};
                    leverage_percentile::Float64, lambda::Float64, coeffs)
    C  = (Gamma' * Gamma .* lambda .+ X' * X)
    XA = C \ X'                                   # K × M
    leverage = [dot(view(X, i, :), view(XA, :, i)) for i in 1:size(X, 1)]
    errors   = Y .- (X * coeffs)
    mask     = leverage .>= quantile(leverage, leverage_percentile)
    pc = XA[:, mask]'
    pc = pc .* (errors[mask] ./ leverage[mask])
    pc = Gamma \ pc'
    return pc'
end

@printf("── POPS cloud @ RLS centre (leverage_percentile = %.2f) ──\n", lev_pct)
cloud_rls = pops_cloud(Ap, Yw, P; leverage_percentile=lev_pct, lambda=λ, coeffs=lin)
eig_rls, bnd_rls = hypercube(cloud_rls)

@printf("── POPS cloud @ CONSTRAINED centre ──\n")
cloud_con = pops_cloud(Ap, Yw, P; leverage_percentile=lev_pct, lambda=λ, coeffs=θ_con)
eig_con, bnd_con = hypercube(cloud_con)
@printf("  RLS cloud: %d directions, mean width %.4g\n", size(eig_rls,2), mean(bnd_rls[2,:] .- bnd_rls[1,:]))
@printf("  con cloud: %d directions, mean width %.4g\n", size(eig_con,2), mean(bnd_con[2,:] .- bnd_con[1,:]))

# ── Draw the three committees ────────────────────────────────────────────────

@printf("\n── arm A: naive @ RLS ──\n")
Θ_A, _ = sample_hypercube(eig_rls, bnd_rls, lin; number_of_committee_members=n_members)

@printf("── arm B: naive @ constrained ──\n")
Θ_B, _ = sample_hypercube(eig_con, bnd_con, θ_con; number_of_committee_members=n_members)

@printf("── arm C: rejection @ constrained (core feasibility enforced) ──\n")
is_core_feasible = θ -> all(B_constraint * θ .>= 0.0)
@assert is_core_feasible(θ_con) "the constrained centre itself violates the core constraint — check the CSV"
Θ_C, _ = rejection_sample_hypercube(eig_con, bnd_con, θ_con, is_core_feasible;
                                    number_of_committee_members = n_members,
                                    max_attempts = 2_000_000)

arms = [("A_naive_rls", Θ_A, lin), ("B_naive_constrained", Θ_B, θ_con), ("C_rejection_constrained", Θ_C, θ_con)]

# ── Score every member ───────────────────────────────────────────────────────

diag_rows = Vector{Any}[]
summary   = Vector{Any}[]

# CSV field formatting: @printf needs a literal format string and a fixed arg
# count, neither of which suits these heterogeneous rows.
csv_field(x::Integer)       = string(x)
csv_field(x::Bool)          = string(x)
csv_field(x::AbstractFloat) = @sprintf("%.8g", x)
csv_field(x)                = string(x)
csv_row(r) = join(csv_field.(r), ",")

for (name, Θ, centre) in arms
    E = B_plot * Θ                              # n_plot × n_members
    writedlm("$outdir/eos_curves_$(name).csv", hcat(a_plot, E), ',')

    n_viol_con   = Int[]
    n_viol_audit = Int[]
    min_E        = Float64[]
    monotone     = Bool[]

    for m in 1:n_members
        e_con   = E[idx_con,   m]
        e_audit = E[idx_audit, m]
        push!(n_viol_con,   count(<(0.0), e_con))
        push!(n_viol_audit, count(<(0.0), e_audit))
        push!(min_E,        minimum(e_audit))
        push!(monotone,     all(diff(e_audit) .< 0.0))
        push!(diag_rows, Any[name, m, n_viol_con[end], n_viol_audit[end],
                             min_E[end], monotone[end], norm(Θ[:, m] .- centre)])
    end

    frac_bad_con   = count(>(0), n_viol_con)   / n_members
    frac_bad_audit = count(>(0), n_viol_audit) / n_members
    frac_nonmono   = count(!, monotone)        / n_members
    push!(summary, Any[name, n_members, frac_bad_con, frac_bad_audit, frac_nonmono,
                       minimum(min_E), median(min_E)])

    @printf("\n  %-26s  violating (constraint grid) %5.1f%% | (audit grid) %5.1f%% | non-monotone %5.1f%% | worst E_min %.4g eV\n",
            name, 100*frac_bad_con, 100*frac_bad_audit, 100*frac_nonmono, minimum(min_E))
end

open("$outdir/member_diagnostics.csv", "w") do io
    println(io, "arm,member,n_violations_constraint_grid,n_violations_audit_grid,min_energy_eV,monotone_decreasing,norm_dtheta")
    for r in diag_rows
        println(io, csv_row(r))
    end
end

open("$outdir/arm_summary.csv", "w") do io
    println(io, "# model=$(result.name) seed=$(sample_seed) leverage_percentile=$(lev_pct)")
    println(io, "# constraint grid: $(length(a_constraint)) pts, audit grid: $(length(a_audit)) pts, a in [0.21, 2.18] Å")
    println(io, "arm,n_members,frac_violating_constraint_grid,frac_violating_audit_grid,frac_non_monotone,worst_min_energy_eV,median_min_energy_eV")
    for r in summary
        println(io, csv_row(r))
    end
end

# ── Figure: one panel per arm, shared axes ───────────────────────────────────

fig = Figure(size=(1000, 1100), fontsize=17)
titles = ["A — naive POPS @ RLS centre",
          "B — naive POPS @ CONSTRAINED centre",
          "C — rejection-sampled @ constrained centre"]

for (row, ((name, Θ, centre), title)) in enumerate(zip(arms, titles))
    E = B_plot * Θ
    ax = Axis(fig[row, 1]; title=title, xlabel="Lattice constant (Å)",
              ylabel="Energy per cell (eV)")
    band!(ax, a_plot, vec(minimum(E, dims=2)), vec(maximum(E, dims=2)); color=(:steelblue, 0.25))
    for m in 1:n_members
        lines!(ax, a_plot, E[:, m]; color=(:steelblue, 0.35), linewidth=0.8)
    end
    lines!(ax, a_plot, B_plot * centre; color=:black, linewidth=2.5, label="centre")
    hlines!(ax, [0.0]; color=(:red, 0.7), linewidth=1.5, linestyle=:dash, label="E = 0 (core floor)")
    vlines!(ax, [2.18]; color=(:black, 0.5), linewidth=2, linestyle=:dash, label="constraint range end")
    ylims!(ax, -14.0, 30.0)
    row == 1 && axislegend(ax; position=:rt, framevisible=false)
end
rowgap!(fig.layout, 20)
save("$outdir/eos_bands_three_arms.pdf", fig)
save("$outdir/eos_bands_three_arms.png", fig; px_per_unit=2)

@printf("\noutputs → %s/\n", outdir)
