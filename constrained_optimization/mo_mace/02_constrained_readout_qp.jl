# ─────────────────────────────────────────────────────────────────────────────
# 02_constrained_readout_qp.jl — Eqs. (1)–(3) of Constrain_Mb_Elastics.pdf.
#
#   minimise   (1/2n) Σ_i ((E_DFT,i − E_MACE,i − D_i·δΘ)/N_i)²  +  (λ/2) ‖S δΘ‖²
#   subject to l ≤ A δΘ ≤ u          (C11, C12, C44 windows + zero stress at A0)
#
# δΘ is a linear corrector on the MACE-MPA-0 readout descriptors (lib_mace_readout.py; Perez et
# al. 2025 Eq. 10 for DESCRIPTOR=perez, the default): the corrected
# model is E_MACE + D·δΘ.  The QP is solved in θ̃ = S δΘ, S = diag(column std of D/N)
# (same idea as the ACE Γ-space QPs), so the ridge acts on per-atom energy scale.
# The last column (Σ_atoms 1) is the per-atom reference offset: mlearn's PBE and
# MPtrj's PBE differ by a constant, so it is left unpenalised.
#
# Three fits share λ, so the table isolates what the constraint costs:
#   offset   : only the constant column (MACE-MPA-0 as-is, reference shifted)
#   ridge    : all columns, no constraints
#   constr.  : all columns + constraints (OSQP)
#
# Inputs  : $OUTDIR from 01_build_design_and_constraints.py
# Outputs : $OUTDIR/delta_theta_constrained.csv, delta_theta_ridge.csv, delta_theta_offset.csv
#
# Run:  julia --project constrained_optimization/mo_mace/02_constrained_readout_qp.jl
# Env:  REPO OUTDIR DESCRIPTOR  LAMBDA (default 1e-4)  EPS (OSQP eps_abs = eps_rel, default 1e-9)
#       SCAN=1  prints a λ scan (train/test RMSE, ridge vs constrained) before the main fit
# ─────────────────────────────────────────────────────────────────────────────

using OSQP, SparseArrays, LinearAlgebra, DelimitedFiles, Printf, Statistics

REPO   = get(ENV, "REPO", "/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow")
DESCRIPTOR = get(ENV, "DESCRIPTOR", "perez")
OUTDIR = get(ENV, "OUTDIR", joinpath(REPO, "models", "Mo_MACE_MPA0_readout", DESCRIPTOR))
LAMBDA = parse(Float64, get(ENV, "LAMBDA", "1e-4"))
EPS    = parse(Float64, get(ENV, "EPS", "1e-9"))
GROUPS = ["Elastic", "AIMD-NVT", "Vacancy", "Surface"]

rd(f) = readdlm(joinpath(OUTDIR, f), ',', Float64)
Dtr, Etr = rd("D_train.csv"), rd("E_train.csv")
Dte, Ete = rd("D_test.csv"),  rd("E_test.csv")
A  = rd("A_con.csv");  l = vec(rd("l_con.csv"));  u = vec(rd("u_con.csv"))
conmeta = readlines(joinpath(OUTDIR, "con_meta.txt"))
CON_NAMES = ["C11", "C12", "C44", "dE/dε₁/V"]
base = [parse(Float64, split(first(filter(s -> startswith(s, n * " "), conmeta)))[3])
        for n in ("C11", "C12", "C44", "dE/de1/V")]

# per-atom rows
Xtr = Dtr ./ Etr[:, 3];  ytr = (Etr[:, 1] .- Etr[:, 2]) ./ Etr[:, 3]
Xte = Dte ./ Ete[:, 3];  yte = (Ete[:, 1] .- Ete[:, 2]) ./ Ete[:, 3]
n, p = size(Xtr)
@printf("train %d × %d, test %d, constraints %d, λ = %.1e\n", n, p, size(Xte, 1), size(A, 1), LAMBDA)

s = vec(std(Xtr; dims = 1)); s[s .< 1e-12] .= 1.0; s[end] = 1.0
S⁻¹ = Diagonal(1 ./ s)
Xs  = Xtr * S⁻¹
As  = A * S⁻¹
pen = ones(p); pen[end] = 0.0                     # offset column unpenalised

H(λ) = Symmetric(Xs' * Xs ./ n + λ * Diagonal(pen))
q    = -(Xs' * ytr) ./ n

ridge(λ) = S⁻¹ * (H(λ) \ (-q))

function constrained(λ; verbose = false)
    m = OSQP.Model()
    OSQP.setup!(m; P = sparse(triu(Matrix(H(λ)))), q = q, A = sparse(As), l = l, u = u,
                eps_abs = EPS, eps_rel = EPS, polish = true, max_iter = 1_000_000,
                verbose = verbose)
    r = OSQP.solve!(m)
    r.info.status in (:Solved, :Solved_inaccurate) ||
        error("OSQP status $(r.info.status) at λ = $λ")
    verbose && @printf("OSQP %s  iters %d  pri_res %.2e  dua_res %.2e  polish %s\n",
                       r.info.status, r.info.iter, r.info.pri_res, r.info.dua_res,
                       string(r.info.status_polish))
    return S⁻¹ * r.x
end

offset() = (δ = zeros(p); δ[end] = mean(ytr); δ)

rmse(X, y, δ)   = 1e3 * sqrt(mean((y .- X * δ) .^ 2))           # meV/atom
predC(δ)        = base .+ A * δ

if get(ENV, "SCAN", "0") == "1"
    println("\nλ scan (meV/atom)          ridge train/test      constr. train/test")
    for λ in 10.0 .^ (-8:1:-1)
        δr, δc = ridge(λ), constrained(λ)
        @printf("  λ = %.0e              %7.3f / %7.3f      %7.3f / %7.3f\n", λ,
                rmse(Xtr, ytr, δr), rmse(Xte, yte, δr), rmse(Xtr, ytr, δc), rmse(Xte, yte, δc))
    end
end

fits = [("offset", offset()), ("ridge", ridge(LAMBDA)), ("constr.", constrained(LAMBDA; verbose = true))]

println("\nRMSE (meV/atom)       train     test   " * join((@sprintf("%9s", g) for g in GROUPS), ""))
for (name, δ) in fits
    grp = [1e3 * sqrt(mean(((yte .- Xte * δ) .^ 2)[Ete[:, 4] .== k - 1])) for k in 1:4]
    @printf("  %-10s       %7.3f  %7.3f   %s\n", name, rmse(Xtr, ytr, δ), rmse(Xte, yte, δ),
            join((@sprintf("%9.3f", g) for g in grp), ""))
end

println("\nlinear prediction at A0 (GPa)   " * join((@sprintf("%11s", c) for c in CON_NAMES), ""))
@printf("  %-14s                 %s\n", "window lo", join((@sprintf("%11.2f", x) for x in base .+ l), ""))
@printf("  %-14s                 %s\n", "window hi", join((@sprintf("%11.2f", x) for x in base .+ u), ""))
for (name, δ) in fits
    @printf("  %-14s                 %s\n", name, join((@sprintf("%11.2f", x) for x in predC(δ)), ""))
end

for (name, δ) in fits
    f = joinpath(OUTDIR, "delta_theta_" * Dict("offset" => "offset", "ridge" => "ridge",
                                              "constr." => "constrained")[name] * ".csv")
    writedlm(f, δ, ',')
end
@printf("\n‖S δΘ‖  ridge %.3e   constrained %.3e\n", norm(s .* fits[2][2]), norm(s .* fits[3][2]))
println("→ $OUTDIR/delta_theta_{offset,ridge,constrained}.csv")
