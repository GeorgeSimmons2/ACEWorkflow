# include("create_model.jl")   # provides: model, Ap, Y, P, lin_params, H_6x6xd, V

using OSQP, SparseArrays, DelimitedFiles
using LinearAlgebra

# ── Target elastic constants (GPa) ────────────────────────────────────────────
# Al DFT reference values
C11_target = 116.3
C12_target =  64.8
C44_target =  30.9

CONV = 160.2176621   # conversion factor:  C_GPa = CONV / V * dot(H_6x6xd[i,j,:], θ)

# Convert GPa targets to raw (Hessian) units
C11_raw = C11_target * V / CONV
C12_raw = C12_target * V / CONV
C44_raw = C44_target * V / CONV

lower_c = [C11_raw, C12_raw, C44_raw]   # equality → lower = upper
upper_c = [C11_raw, C12_raw, C44_raw]

# Having lower = 0.0 just gets the solver to get arbitrarily close to 0
# which actually violates the constraint (computationally speaking)
# So we enforce that we should be > 1GPa
# 1. Define the base component rows from your descriptor matrix
# (These are still in the original parameter space)
row_c11 = H_6x6xd[1, 1, :]'
row_c12 = H_6x6xd[1, 2, :]'
row_c44 = H_6x6xd[4, 4, :]'

# 2. Construct the Born Stability Matrix
# We combine the rows to match the stability conditions
born_matrix = vcat(
    row_c11 .+ (2 .* row_c12),  # Bulk modulus (proportional to)
    row_c11 .- row_c12,      # Tetragonal shear modulus
    row_c44                 # Fundamental shear modulus
)

# 4. Set the "Safe" Lower Bounds
# Using a 2.0 GPa floor (scaled) to avoid the "mushy" Gamma point
floor_val = 5.0 * (V / CONV)
lower_c = [floor_val, floor_val, floor_val]
upper_c = [Inf, Inf, Inf]

# ── Helper: verify elastic constants from a parameter vector ──────────────────
function print_elastic(label, θ)
    c11 = round(dot(H_6x6xd[1,1,:], θ), digits=2)
    c12 = round(dot(H_6x6xd[1,2,:], θ), digits=2)
    c44 = round(dot(H_6x6xd[4,4,:], θ), digits=2)
    println("  $label →  C11=$c11  C12=$c12  C44=$c44 GPa")
end

# ── Constrained OLS ────────────────────────────────────────────────────────────
# minimise  ½ ||Ap x - Y||²  +  ½/n ||x||²
# subject to  (all_C_ij / P) x  =  [C11_raw, C12_raw, C44_raw]
# println("\n═══ Constrained OLS ═══════════════════════════════════════════")

# let
#     n     = size(Ap, 1)
#     H_qp  = sparse(Ap' * Ap .+ (1.0 / n) .* P' * P)
#     b_qp  = -(Ap' * Y)
#     A_con = sparse(all_C_ij / P)

#     m = OSQP.Model()
#     OSQP.setup!(m; P=H_qp, q=b_qp, A=A_con,
#                 l=lower_c, u=upper_c,
#                 verbose=true, max_iter=10_000,
#                 check_termination=200,
#                 eps_abs=1e-8, eps_rel=1e-8)
#     res = OSQP.solve!(m)

#     global constrained_OLS_params = P \ res.x
# end

# print_elastic("OLS result", constrained_OLS_params)
# println("  targets  →  C11=$C11_target  C12=$C12_target  C44=$C44_target GPa")

# writedlm("small_high_entropy_ace_model/constrained_OLS_params.csv",
#          constrained_OLS_params', ',')
# ACEpotentials.Models.set_linear_parameters!(model, constrained_OLS_params)
# ACEpotentials.save_model(model, "small_high_entropy_ace_model/constrained_OLS_model.json")
# println("Saved → constrained_OLS_model.json")

# ── Constrained POPS ───────────────────────────────────────────────────────────
# Each POPS member satisfies:
#   (1) the three elastic constant equalities  (shared across all members)
#   (2) interpolation of its own training point
#   (3) regularized objective  (like base_constrained_pops: adds (1/n)*P'*P to H)
#       — without this, the problem is massively underdetermined and members diverge
println("\n═══ Constrained POPS ══════════════════════════════════════════")

function constrained_pops_elastic(X_train, Y_train, Gamma,
                                   elastic_constraint, l_elastic, u_elastic;
                                   members_to_constrain=1:length(Y_train))
    n = size(X_train, 1)
    H_qp = sparse(X_train' * X_train .+ (1.0 / n) .* Gamma' * Gamma)
    b_qp = -(X_train' * Y_train)
    out  = zeros(length(members_to_constrain), size(X_train, 2))
    Threads.@threads for idx in 1:length(members_to_constrain)
        i      = members_to_constrain[idx]
        A_full = sparse(vcat(elastic_constraint, X_train[i,:]'))
        l_full = vcat(l_elastic, Y_train[i])
        u_full = vcat(u_elastic, Y_train[i])
        m = OSQP.Model()
        OSQP.setup!(m; P=H_qp, q=b_qp, A=A_full, l=l_full, u=u_full,
                    max_iter=500_000, check_termination=200,
                    eps_abs=1e-7, eps_rel=1e-7, verbose=true)
        res = OSQP.solve!(m)
        out[idx, :] = Gamma \ res.x
        println(idx)
    end
    return out
end

function constrained_pops_elastic(X_train, Y_train, Gamma,
                                  H_6x6xd;
                                  members_to_constrain=1:length(Y_train),
                                  buffer=3.0*V/CONV,
                                  naive_POPS=POPS_delta)
    n = size(X_train, 1)
    H_qp = sparse(X_train' * X_train .+ (1.0 / n) .* Gamma' * Gamma)
    b_qp = -(X_train' * Y_train)
    out  = zeros(length(members_to_constrain), size(X_train, 2))
    for idx in 1:length(members_to_constrain)
        i      = members_to_constrain[idx]
        C11_i, C12_i = dot(H_6x6xd[1,1,:], naive_POPS[i,:]), dot(H_6x6xd[1,2,:], naive_POPS[i,:])
        A_full = sparse(vcat(H_6x6xd[1,1,:]' / Gamma, H_6x6xd[1,2,:]' / Gamma, H_6x6xd[4,4,:]' / Gamma, X_train[i,:]'))
        l_full = vcat(C11_i, C12_i, buffer, Y_train[i])
        u_full = vcat(C11_i, C12_i, Inf, Y_train[i])
        m = OSQP.Model()
        OSQP.setup!(m; P=H_qp, q=b_qp, A=A_full, l=l_full, u=u_full,
                    max_iter=500_000, check_termination=200,
                    eps_abs=1e-7, eps_rel=1e-7, verbose=true)
        res = OSQP.solve!(m)
        out[idx, :] = Gamma \ res.x
        println(idx)
    end
    return out
end

# constrained_pops_params = constrained_pops_elastic(
#     Ap, Y, P, all_C_ij / P, lower_c, upper_c; members_to_constrain=negative_indices)
new_constrained_pops_params = constrained_pops_elastic(
    Ap, Y, P, H_6x6xd; members_to_constrain=negative_indices)

# println("Verifying first 5 members:")
# for i = 1:min(5, size(constrained_pops_params, 1))
#     print_elastic("[$i]", (CONV / V) .* constrained_pops_params[i, :])
# end
println("  targets  →  C11=$C11_target  C12=$C12_target  C44=$C44_target GPa")
# writedlm("high_entropy_POPS/buffer_constrained_POPS.csv", constrained_pops_params, ',')
# writedlm("high_entropy_POPS/positive_C44_POPS_coeffs.csv", constrained_pops_params, ',')
println("\nSaved → constrained_pops_elastic.csv")

# hypercube_eigs, hypercube_bounds = POPSRegression.hypercube(constrained_pops_params .- constrained_OLS_params')
# committee, dΘ = POPSRegression.sample_hypercube(hypercube_eigs, hypercube_bounds, constrained_OLS_params)

constrained_POPS_all = deepcopy(POPS_delta)
for i=1:size(new_constrained_pops_params,1)
    constrained_POPS_all[negative_indices[i],:] = new_constrained_pops_params[i,:]
end

hyp_eigs, hyp_bounds = POPSRegression.hypercube(constrained_POPS_all .- lin_params')
samples, _ = POPSRegression.rejection_sample_hypercube(hyp_eigs, hyp_bounds, lin_params, ([0.,0.,0.], [Inf,Inf,Inf]), vcat(H_6x6xd[1,1,:]', H_6x6xd[1,2,:]', H_6x6xd[4,4,:]'); number_of_committee_members=10)
writedlm("high_entropy_POPS/constrained_rejection_samples.csv", samples, ',')