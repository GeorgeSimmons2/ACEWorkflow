# sdp_meanfit_prototype_Al_12_4_6A_2.jl
#
# PROTOTYPE. Does a single semidefinite program reproduce the constrained mean
# that bandpath_committee_undotted_Al_12_4_6A_2_ncell4_densek.jl obtains by
# iterating cutting planes?
#
# The cutting-plane loop is Kelley constraint generation for the semi-infinite
# constraint "lambda_min(D(q;theta)) >= omega_cut^2 for every q on the path".
# Each cut e'D(q;theta)e >= omega_cut^2 is one supporting hyperplane of that
# constraint, valid because the Rayleigh quotient at FIXED polarisation is linear
# in theta (see cut_row in ../bandpath_phonon_uq/lib.jl).
#
# The constraint itself is a linear matrix inequality, since D is linear in theta:
#
#     D(q;theta) - omega_cut^2 * I  >=  0        (positive semidefinite)
#
# For FCC Al the primitive cell holds ONE atom, so each block is 3x3 -- 145
# q-points give 145 tiny LMIs over 91 variables. That is a small SDP, so the whole
# feasible set can be declared up front instead of discovered by iteration.
#
# Everything except the constraint representation is copied verbatim from the
# legacy script: same a_eq, same Born rows, same objective (OSQP's 1/2 x'Px + q'x
# in the preconditioned variable x = P*theta), same omega_cut, same near-Gamma
# exclusion (qΓtol = 5e-2, since the acoustic modes vanish at Gamma and the LMI
# would be infeasible there).
#
# Hermitian LMIs are posed through the standard real embedding
#     H = X + iY  psd  <=>  [X  -Y; Y  X]  psd        (X symmetric, Y antisymmetric)
# to avoid relying on complex-number support in the solver interface.
#
# Run (Clarabel/SCS are NOT in the project Project.toml; stack a scratch env):
#   SDPENV=/path/to/sdpenv
#   julia --project=. -e "using Pkg; Pkg.activate(ENV[\"SDPENV\"]); Pkg.add([\"JuMP\",\"Clarabel\"])"
#   JULIA_LOAD_PATH="@:$SDPENV:@stdlib" julia --project=. -t 8 \
#       scripts/uq/sdp_meanfit_prototype_Al_12_4_6A_2.jl

include(joinpath(@__DIR__, "..", "bandpath_phonon_uq", "lib.jl"))
using SparseArrays, LinearAlgebra
using JuMP
import Clarabel

element        = :Al
dataset        = ""
N_cell_fc      = 4
N_per_seg      = [20, 20, 20, 20, 60]
cut_margin_THz = 0.15
qΓtol          = 5e-2

result     = load_model(element, 12, 4, 6, 2; dataset_name=dataset)
model      = result.model; lin_params = result.lin_params; n_params = length(lin_params)
P = result.P; Ap = Diagonal(result.W)*result.A/P; Yw = result.W.*result.Y; λ = 1.0/size(Ap,1)
outdir = "$(result.dir)/results/sdp_prototype"; mkpath(outdir)
@printf("Model %s: %d params, %d threads\n", result.name, n_params, Threads.nthreads()); flush(stdout)

# ── reference geometry, Born + a_eq rows (verbatim from the legacy script) ────
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
@printf("a_eq = %.5f Å\n", a_eq); flush(stdout)

# ── band-path D_k(q) ─────────────────────────────────────────────────────────
bp = bandpath_Dk(result, model, element, a_eq, N_cell_fc; N_per_seg=N_per_seg)
ω2_cut = (cut_margin_THz / FREQ_THz)^2
nb = 3*bp.Np
keep = [iq for iq in eachindex(bp.Bq) if bp.qnorm[iq] >= qΓtol]
@printf("band path: %d q-points, %d kept (near-Γ excluded), blocks %dx%d\n",
        length(bp.Bq), length(keep), nb, nb); flush(stdout)

# ── the SDP ──────────────────────────────────────────────────────────────────
Hqp = Matrix(Ap'*Ap .+ λ.*(P'*P)); qqp = -(Ap'*Yw)
Hqp = 0.5*(Hqp + Hqp')                      # symmetrise against round-off

sdp = Model(Clarabel.Optimizer)
set_silent(sdp)
@variable(sdp, x[1:n_params])
@objective(sdp, Min, 0.5*x'*Hqp*x + qqp'*x)
@constraint(sdp, (b_prime'/P)*x .== 0.0)
@constraint(sdp, (born_rows/P)*x .>= born_lower)

t_build = @elapsed for iq in keep
    Bx = bp.Bq[iq] / P                       # (nb^2) × n_params, complex
    Xv = reshape(real.(Bx)*x, nb, nb)        # real part  (symmetric)
    Yv = reshape(imag.(Bx)*x, nb, nb)        # imag part  (antisymmetric)
    Xs = Xv .- ω2_cut*Matrix(I, nb, nb)
    M  = [Xs  -Yv;
          Yv   Xs]
    @constraint(sdp, Symmetric(0.5*(M .+ transpose(M))) in PSDCone())
end
@printf("built %d LMI blocks in %.1f s\n", length(keep), t_build); flush(stdout)

t_solve = @elapsed optimize!(sdp)
@printf("solve: %s in %.1f s\n", termination_status(sdp), t_solve); flush(stdout)
θ_sdp = P \ value.(x)

# ── compare against the cutting-plane answer ─────────────────────────────────
ref = "$(result.dir)/results/bandpath_undotted_ncell4_densek/theta_mean.csv"
θ_cp = vec(readdlm(ref, ','))
obj(θ) = (r = Ap*(P*θ) .- Yw; 0.5*dot(r,r) + 0.5*λ*dot(P*(P*θ), P*(P*θ)))

println("\n══ SDP vs cutting plane ═══════════════════════════════════════")
@printf("  ‖θ_sdp − θ_cp‖        = %.4e   (‖θ_cp‖ = %.4e)\n", norm(θ_sdp .- θ_cp), norm(θ_cp))
@printf("  max |Δθ|              = %.4e\n", maximum(abs.(θ_sdp .- θ_cp)))
@printf("  relative              = %.3e\n", norm(θ_sdp .- θ_cp)/norm(θ_cp))
for (nm, θ) in (("cutting plane", θ_cp), ("SDP", θ_sdp))
    @printf("  %-14s min ω = %+.4f THz | C11=%.1f C12=%.1f C44=%.1f GPa | b′·θ = %+.2e\n",
            nm, min_freq_stable(θ, bp), born(θ)..., dot(b_prime, θ))
end
@printf("  objective  cutting plane %.10e | SDP %.10e  (SDP lower by %.3e)\n",
        obj(θ_cp), obj(θ_sdp), obj(θ_cp)-obj(θ_sdp))

writedlm("$outdir/theta_mean_sdp.csv", θ_sdp, ',')
println("\nθ_sdp → $outdir/theta_mean_sdp.csv")
