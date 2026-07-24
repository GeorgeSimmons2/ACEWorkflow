# undotted_hessian_test.jl
#
# Extract the per-basis ("undotted") Hessian in ONE pass instead of computing
# the dotted Hessian n_params times.
#
# Where the dot happens (traced through ACEpotentials):
#   AtomsCalculatorsUtilities.SitePotentials.hessian(sys, V)
#     └ loops atoms, calls hessian_site(V, Rs, Zs, z0)          [assembly, ± pattern]
#         = ad_hessian_site = ForwardDiff.jacobian(eval_grad_site)
#           eval_grad_site(V, Rs, Zs, z0) = evaluate_ed(V.model, Rs, Zs, z0, ps, st)[2]
#                                          = Σ_k θ_k ∂B_k        ← THE DOT (in evaluate_ed)
#
# The undotted twin is `evaluate_basis(model, Rs, Zs, z0, ps, st)` → B (the
# per-basis site energies, exactly what potential_energy_basis sums).  So we
# nested-ForwardDiff `evaluate_basis` at the SITE level to get ∂²B_k/∂x² for
# ALL k at once, then assemble with the identical ± accumulation the native
# `hessian` uses.  Result: H_basis[:,:,k] with  Σ_k θ_k H_basis[:,:,k] == native
# hessian(sys, model_θ)  — verified below.
#
# Run:  julia --project scripts/uq/undotted_hessian_test.jl

using LinearAlgebra, StaticArrays, Printf, Statistics
using ACEpotentials, ACEWorkflow, ForwardDiff, Unitful
import ACEpotentials.Models: evaluate_basis
import AtomsCalculatorsUtilities.SitePotentials: PairList, get_neighbours, cutoff_radius
using AtomsCalculatorsUtilities.SitePotentials: hessian
using AtomsBuilder

# ── per-basis SITE Hessian: Hi[k, a, b] = ∂²B_k/∂x_a∂x_b  (3nR × 3nR per basis) ─
function site_basis_hessian(model, Rs, Zs, z0, ps, st)
    nR = length(Rs)
    x0 = collect(Float64, reinterpret(Float64, Rs))          # flat 3nR (Å, unitless)
    tovec(x) = [SVector{3,eltype(x)}(x[3i-2], x[3i-1], x[3i]) for i in 1:nR]
    Bfun(x)    = evaluate_basis(model, tovec(x), Zs, z0, ps, st)       # N_basis vector
    gradflat(x) = vec(ForwardDiff.jacobian(Bfun, x))                   # N_basis·3nR
    Hflat = ForwardDiff.jacobian(gradflat, x0)                         # (N_basis·3nR) × 3nR
    N_basis = length(Bfun(x0))
    return reshape(Hflat, N_basis, 3nR, 3nR)                           # [k, a, b]
end

# ── global per-basis Hessian, mirroring AtomsCalculatorsUtilities `hessian` ───
function hessian_basis_undotted(sys, V)
    nlist = PairList(sys, cutoff_radius(V))
    Nat = length(sys); D = 3; ps = V.ps; st = V.st; model = V.model
    Js, Rs, Zs, z0 = get_neighbours(sys, V, nlist, 1)
    N_basis = length(evaluate_basis(model, Rs, Zs, z0, ps, st))
    H = zeros(D*Nat, D*Nat, N_basis)
    for i in 1:Nat
        Js, Rs, Zs, z0 = get_neighbours(sys, V, nlist, i)
        Hi = site_basis_hessian(model, Rs, Zs, z0, ps, st)            # [k, 3nR, 3nR]
        Ji = (i-1)*D .+ (1:D)
        for (α1, j1) in enumerate(Js), (α2, j2) in enumerate(Js)
            A1 = (α1-1)*D .+ (1:D); A2 = (α2-1)*D .+ (1:D)
            J1 = (j1-1)*D .+ (1:D); J2 = (j2-1)*D .+ (1:D)
            @views for k in 1:N_basis
                blk = Hi[k, A1, A2]
                H[J1, J2, k] .+= blk
                H[J1, Ji, k] .-= blk
                H[Ji, J2, k] .-= blk
                H[Ji, Ji, k] .+= blk
            end
        end
    end
    return H                                                          # eV/Å², [i,j,k]
end

# ── Test on Al_12_4_6A_2 ──────────────────────────────────────────────────────
element = :Al
result  = load_model(element, 12, 4, 6, 2; dataset_name="")   # models/Al_12_4_6A_2_/
model   = result.model
lin_params = result.lin_params
n_params   = length(lin_params)
@printf("Model %s: %d basis functions\n", result.name, n_params)

a_eq = ACEWorkflow.relax_lattice_constant(model, element)
sys  = bulk(element; a=a_eq*u"Å", cubic=true) * (2, 2, 2)     # 32 atoms
@printf("Test system: %d atoms (%d×%d Hessian)\n", length(sys), 3length(sys), 3length(sys))

println("\nBuilding undotted per-basis Hessian (ONE pass) …")
t_undot = @elapsed H_basis = hessian_basis_undotted(sys, model)
@printf("  H_basis: %s  in %.2f s\n", size(H_basis), t_undot)

# reference: native dotted Hessian at θ = lin_params
ACEpotentials.Models.set_linear_parameters!(model, lin_params)
H_native = ustrip.(hessian(sys, model))

# reconstruct dotted from undotted:  Σ_k θ_k H_basis[:,:,k]
N3 = 3length(sys)
H_recon = reshape(reshape(H_basis, N3*N3, n_params) * lin_params, N3, N3)

err = maximum(abs.(H_recon .- H_native))
rel = err / maximum(abs.(H_native))
@printf("\n── Verification (θ = lin_params) ───────────────────────────\n")
@printf("  max|Σ θ_k H_basis[:,:,k] − native hessian| = %.3e  (rel %.3e)\n", err, rel)
println(rel < 1e-8 ? "  ✓ MATCH" : "  ✗ MISMATCH")

# also check a RANDOM θ (guards against accidental lin_params-specific agreement)
θr = randn(n_params)
ACEpotentials.Models.set_linear_parameters!(model, θr)
H_native_r = ustrip.(hessian(sys, model))
H_recon_r  = reshape(reshape(H_basis, N3*N3, n_params) * θr, N3, N3)
err_r = maximum(abs.(H_recon_r .- H_native_r)) / maximum(abs.(H_native_r))
@printf("  random θ:  rel error = %.3e  %s\n", err_r, err_r < 1e-8 ? "✓" : "✗")
ACEpotentials.Models.set_linear_parameters!(model, lin_params)

# ── Cost comparison vs the naive 314× dotted approach ────────────────────────
println("\n── Cost: undotted 1-pass vs dotted ×n_params ───────────────")
e_k = zeros(n_params); e_k[1] = 1.0
ACEpotentials.Models.set_linear_parameters!(model, e_k)
t_one = @elapsed ustrip.(hessian(sys, model))
ACEpotentials.Models.set_linear_parameters!(model, lin_params)
@printf("  one dotted hessian: %.2f s  →  ×%d ≈ %.1f s\n", t_one, n_params, t_one*n_params)
@printf("  undotted one-pass : %.2f s   (%.1f× faster)\n", t_undot, t_one*n_params/t_undot)
