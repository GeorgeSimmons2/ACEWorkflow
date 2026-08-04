# sdp_meanfit_diagnose_Al_12_4_6A_2.jl
#
# Follow-up to sdp_meanfit_prototype: the SDP reached a LOWER objective than the
# cutting-plane answer, which contradicts theory -- every cut
# e'D(q;theta)e >= omega_cut^2 is IMPLIED by the LMI, so the cut-QP is a
# RELAXATION and its optimum must be <= the SDP optimum.
#
# Three candidate explanations, distinguished here:
#   (A) OSQP terminated loosely (first-order ADMM, eps=1e-6, status never checked
#       for mean_fit) so theta_cp is simply not the optimum of its own QP;
#   (B) the phonon constraint is INACTIVE at the optimum, in which case the SDP
#       must coincide with the plain Born + b'=0 QP and the cuts bought nothing;
#   (C) the two scripts are not solving the same problem.
#
# Test: solve the SAME objective with Clarabel three ways -- no phonon rows at
# all, the full LMI, and the LMI plus a re-solve seeded from theta_cp -- and
# report objective, min omega, and which LMI blocks are active.

include(joinpath(@__DIR__, "..", "bandpath_phonon_uq", "lib.jl"))
using SparseArrays, LinearAlgebra
using JuMP
import Clarabel

element, dataset = :Al, ""
N_cell_fc, N_per_seg = 4, [20, 20, 20, 20, 60]
cut_margin_THz, qΓtol = 0.15, 5e-2

result = load_model(element, 12, 4, 6, 2; dataset_name=dataset)
model  = result.model; lin_params = result.lin_params; n_params = length(lin_params)
P = result.P; Ap = Diagonal(result.W)*result.A/P; Yw = result.W.*result.Y; λ = 1.0/size(Ap,1)

a_eq = ACEWorkflow.relax_lattice_constant(model, element)
sys0 = ACEWorkflow.Elasticity.reference_system(element; a=a_eq)
L0   = ustrip.(ACEWorkflow.Elasticity.lattice_matrix(sys0.cell.cell_vectors))
eV_to_GPa = 160.2176621/abs(det(L0))
H_el = elastic_hessian_basis(model; element=element, a=a_eq)
c11_0 = reshape(H_el,36,n_params)[1,:]; c12_0 = reshape(H_el,36,n_params)[7,:]; c44_0 = reshape(H_el,36,n_params)[22,:]
born(θ) = (dot(c11_0,θ)*eV_to_GPa, dot(c12_0,θ)*eV_to_GPa, dot(c44_0,θ)*eV_to_GPa)
lattice_basis(a) = ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(ACEWorkflow.Elasticity.reference_system(element; a=a), model))
b_prime        = ForwardDiff.derivative(lattice_basis, a_eq)
b_double_prime = ForwardDiff.derivative(a -> ForwardDiff.derivative(lattice_basis, a), a_eq)
born_rows  = vcat(c44_0', (c11_0.-c12_0)', (c11_0.+2 .*c12_0)', b_double_prime')
born_lower = [0.1, 1.0, 0.1, 1e-9]

bp = bandpath_Dk(result, model, element, a_eq, N_cell_fc; N_per_seg=N_per_seg)
ω2_cut = (cut_margin_THz / FREQ_THz)^2
nb = 3*bp.Np
keep = [iq for iq in eachindex(bp.Bq) if bp.qnorm[iq] >= qΓtol]
@printf("band path: %d q-points total, %d kept\n", length(bp.Bq), length(keep))
@printf("N_per_seg = %s  → sum = %d\n", string(N_per_seg), sum(N_per_seg)); flush(stdout)

Hqp = Matrix(Ap'*Ap .+ λ.*(P'*P)); Hqp = 0.5*(Hqp + Hqp'); qqp = -(Ap'*Yw)
const YWSQ = 0.5*dot(Yw, Yw)
osqp_obj(θ) = (x = P*θ; 0.5*dot(x, Hqp*x) + dot(qqp, x))     # exactly OSQP's objective

function solve_sdp(; with_lmi::Bool)
    m = Model(Clarabel.Optimizer); set_silent(m)
    @variable(m, x[1:n_params])
    @objective(m, Min, 0.5*x'*Hqp*x + qqp'*x)
    @constraint(m, (b_prime'/P)*x .== 0.0)
    @constraint(m, (born_rows/P)*x .>= born_lower)
    if with_lmi
        for iq in keep
            Bx = bp.Bq[iq] / P
            Xv = reshape(real.(Bx)*x, nb, nb); Yv = reshape(imag.(Bx)*x, nb, nb)
            Xs = Xv .- ω2_cut*Matrix(I, nb, nb)
            M  = [Xs -Yv; Yv Xs]
            @constraint(m, Symmetric(0.5*(M .+ transpose(M))) in PSDCone())
        end
    end
    optimize!(m)
    return P \ value.(x), termination_status(m)
end

println("\n── solving ────────────────────────────────────────────────────"); flush(stdout)
t1 = @elapsed ((θ_plain, st1) = solve_sdp(with_lmi=false))
@printf("  plain QP (Born + b′=0 only) : %s  [%.1f s]\n", st1, t1); flush(stdout)
t2 = @elapsed ((θ_lmi, st2) = solve_sdp(with_lmi=true))
@printf("  full LMI                    : %s  [%.1f s]\n", st2, t2); flush(stdout)

θ_cp = vec(readdlm("$(result.dir)/results/bandpath_undotted_ncell4_densek/theta_mean.csv", ','))

println("\n══ RESULTS ════════════════════════════════════════════════════")
@printf("%-24s %18s %12s %28s %12s\n", "solution", "OSQP objective", "min ω THz", "C11/C12/C44 GPa", "b′·θ")
for (nm, θ) in (("cutting plane (OSQP)", θ_cp), ("Clarabel plain QP", θ_plain), ("Clarabel full LMI", θ_lmi))
    c = born(θ)
    @printf("%-24s %18.10e %12.4f %10.1f/%.1f/%.1f %18.2e\n",
            nm, osqp_obj(θ), min_freq_stable(θ, bp), c[1], c[2], c[3], dot(b_prime, θ))
end

println("\n── is the phonon LMI active at the optimum? ───────────────────")
function block_margin(θ)
    worst = Inf; arg = 0
    for iq in keep
        D = Hermitian(reshape(bp.Bq[iq]*θ, nb, nb))
        v = minimum(eigvals(D)) - ω2_cut
        v < worst && (worst = v; arg = iq)
    end
    return worst, arg
end
for (nm, θ) in (("plain QP", θ_plain), ("full LMI", θ_lmi), ("cutting plane", θ_cp))
    w, iq = block_margin(θ)
    @printf("  %-14s worst (λ_min − ω²_cut) = %+.4e at iq=%d   (0 ⇒ constraint active)\n", nm, w, iq)
end

@printf("\n  ‖θ_lmi − θ_plain‖ = %.4e ; ‖θ_lmi − θ_cp‖ = %.4e ; ‖θ_cp‖ = %.4e\n",
        norm(θ_lmi .- θ_plain), norm(θ_lmi .- θ_cp), norm(θ_cp))
