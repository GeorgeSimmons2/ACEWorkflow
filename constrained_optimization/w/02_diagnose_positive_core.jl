# ─────────────────────────────────────────────────────────────────────────────
# 02_diagnose_positive_core.jl — is the W positive-core QP's solution well defined?
#
# 01_positive_core_qp.jl restored the solve and it did NOT reproduce the published θ:
# max|Δ| = 1.69, ‖Δ‖/‖θ‖ = 14.3%.  Before concluding anything about the published vector,
# find out whether the QP even HAS a well-determined solution at the tolerance it was run
# at.  The log says it does not:
#
#     status: Solved   iters: 19000   pri_res: 7.20e-01   dua_res: 6.48e+06
#     optimal objective: -1.982e+09      optimal rho estimate: 1.00e-06
#
# OSQP terminates on residuals measured relative to the problem's own scale, and this
# problem is scaled ~1e11 (the unconstrained bulk energy reaches -2.8e11 eV at a = 0.21 Å).
# At eps_abs = eps_rel = 1e-3 — the DEFAULT, which the original left in place — the
# effective absolute tolerance is enormous, so the solver stops far from the true optimum
# and WHERE it stops depends on the rho schedule.  That is a much better explanation for a
# 14% difference than "the published θ is wrong".
#
# This script settles it by evaluating three vectors against the SAME objective:
#
#   published   models/W_20_4_5A_3/positive_core_constrained_parameters.csv
#   loose       what 01 produced, at OSQP defaults
#   tight       a fresh solve at eps 1e-10 with polishing on
#
# For each: the QP objective, the worst constraint violation, and — the thing that
# actually matters — the largest difference in the plotted EOS curve.  The figure is a
# curve, so two θ that differ in norm but draw the same curve are the same answer.
#
# Run:  sbatch constrained_optimization/w/run_diagnose_positive_core.slurm
# Env:  EPS_TIGHT (default 1e-10)  MAXIT_TIGHT (default 2_000_000)
# ─────────────────────────────────────────────────────────────────────────────

using ACEpotentials, EmpiricalPotentials, ACEWorkflow, OSQP, SparseArrays
using ACEpotentials.Models: potential_energy_basis
using AtomsBase, StaticArrays, StatsBase, Unitful, ExtXYZ
using AtomsBuilder, LinearAlgebra, Statistics
using DelimitedFiles, Printf
import AtomsCalculators: potential_energy

element, totdeg, prior_param, rcut, v = :W, 20, 4, 5, 3
EPS_TIGHT   = parse(Float64, get(ENV, "EPS_TIGHT", "1e-10"))
MAXIT_TIGHT = parse(Int,     get(ENV, "MAXIT_TIGHT", "2000000"))

result = load_model(element, totdeg, prior_param, rcut, v)
model  = result.model

lattice_constants = collect(LinRange(0.21, 2.18, 50))          # the constrained grid
# the grid the FIGURE draws, from eos_with_pair_hist.jl
full_lattice = vcat(lattice_constants,
                    collect(LinRange(maximum(lattice_constants),
                                     2*maximum(lattice_constants), 50)[2:end]))

function bulk_energy_basis(element::Symbol, as::AbstractVector{Float64}, model)
    [ustrip.(potential_energy_basis(bulk(element, a=a*u"Å"), model)) for a in as]
end

bulk_bases = bulk_energy_basis(element, lattice_constants, model)
full_bases = bulk_energy_basis(element, full_lattice, model)
C_con  = Matrix(reduce(hcat, bulk_bases)')      # 50   × n, the constraint rows
C_full = Matrix(reduce(hcat, full_bases)')      # 99   × n, the plotted curve
println("bases built."); flush(stdout)

Gamma = result.P
Ap    = Diagonal(result.W) * result.A / Gamma
Yw    = result.W .* result.Y
H     = Ap'Ap .+ (1.0/size(Ap,1)) .* Gamma'Gamma
q     = -Ap'Yw
A_qp  = sparse(C_con / Gamma)
l, u  = zeros(size(C_con,1)), fill(Inf, size(C_con,1))
@printf("H %d×%d built.  ‖H‖₁ = %.3e, ‖q‖₂ = %.3e\n", size(H)..., norm(H,1), norm(q))
flush(stdout)

objective(θ) = (x = Gamma*θ; 0.5*dot(x, H*x) + dot(q, x))
violation(θ) = -min(0.0, minimum(C_con*θ))          # 0 if feasible
fitres(θ)    = norm(Ap*(Gamma*θ) .- Yw)

# ── the tight reference solve ────────────────────────────────────────────────
m = OSQP.Model()
OSQP.setup!(m; P=sparse(H), q=q, A=A_qp, l=l, u=u,
            max_iter=MAXIT_TIGHT, check_termination=1000, verbose=true,
            eps_abs=EPS_TIGHT, eps_rel=EPS_TIGHT, polish=true,
            adaptive_rho_interval=25)
t = @elapsed r = OSQP.solve!(m)
θ_tight = Gamma \ r.x
@printf("\ntight solve: status %s, %d iters, %.1f s, pri_res %.3e, dua_res %.3e\n",
        string(r.info.status), r.info.iter, t, r.info.pri_res, r.info.dua_res)
writedlm("$(result.dir)/repro_positive_core_tight.csv", θ_tight, ',')

# ── the three vectors, side by side ──────────────────────────────────────────
cands = Pair{String,Vector{Float64}}[]
push!(cands, "published" => vec(readdlm("$(result.dir)/positive_core_constrained_parameters.csv", ',')))
loose = "$(result.dir)/repro_positive_core_constrained_parameters.csv"
isfile(loose) && push!(cands, "loose (01)" => vec(readdlm(loose, ',')))
push!(cands, "tight"     => θ_tight)
push!(cands, "unconstrained" => result.lin_params)

println("\n══ the same objective, evaluated at each vector ═══════════════════")
@printf("%-15s %15s %13s %13s %13s\n", "vector", "objective", "max violation", "‖Ap x−Yw‖", "‖θ‖₂")
for (name, θ) in cands
    @printf("%-15s %15.6e %13.3e %13.6e %13.6e\n",
            name, objective(θ), violation(θ), fitres(θ), norm(θ))
end

println("\n══ what the FIGURE draws: bulk energy over 0.21–4.36 Å ════════════")
println("   (the curve is the claim, not the coefficients — two θ that draw the same")
println("    curve are the same answer as far as the figure is concerned)")
Es = Dict(name => C_full*θ for (name, θ) in cands)
ref = Es["published"]
@printf("%-15s %15s %15s\n", "vs published", "max |ΔE| (eV)", "max rel |ΔE|")
for (name, _) in cands
    name == "published" && continue
    d = abs.(Es[name] .- ref)
    @printf("%-15s %15.6e %15.3e\n", name, maximum(d), maximum(d ./ max.(abs.(ref), eps())))
end

println("\n══ and over the range the figure is actually READ at (a ≥ 1.0 Å) ══")
sel = full_lattice .>= 1.0
for (name, _) in cands
    name == "published" && continue
    d = abs.(Es[name][sel] .- ref[sel])
    @printf("%-15s max |ΔE| = %.6e eV   max rel = %.3e\n",
            name, maximum(d), maximum(d ./ max.(abs.(ref[sel]), eps())))
end
@printf("\nθ_tight → %s\n", "$(result.dir)/repro_positive_core_tight.csv")
