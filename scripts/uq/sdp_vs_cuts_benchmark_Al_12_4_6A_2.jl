# sdp_vs_cuts_benchmark_Al_12_4_6A_2.jl
#
# Head-to-head, per POPS member: cutting-plane QP (the current pipeline) versus a
# single SDP. Answers two questions.
#
#  (1) Does the POPS interpolation equality Ap[i,:]·θ = Yw[i] survive exactly when
#      the phonon constraint becomes an LMI?  Conic solvers take linear equalities
#      natively alongside the PSD cone, so it should hold to solver tolerance --
#      measured here as |Ap[i,:]·θ − Yw[i]|.
#
#  (2) Which is faster per member?  The cutting plane pays k QP solves with a
#      constraint matrix that grows by one row per soft mode per iteration; the SDP
#      pays one solve with all 138 blocks declared up front.
#
# Both routes use the SAME objective, the SAME Born rows, the SAME b′·θ = 0, the
# SAME ω_cut and the SAME near-Γ exclusion. The only difference is how dynamical
# stability is represented. constrain_member/cutting-plane code is copied verbatim
# from bandpath_committee_undotted_Al_12_4_6A_2_ncell4_densek.jl.
#
# Run: JULIA_LOAD_PATH="@:$SDPENV:@stdlib" julia --project=. -t 8 \
#          scripts/uq/sdp_vs_cuts_benchmark_Al_12_4_6A_2.jl [n_members]

include(joinpath(@__DIR__, "..", "bandpath_phonon_uq", "lib.jl"))
using SparseArrays, LinearAlgebra, OSQP, Random
using JuMP
import Clarabel
Random.seed!(1234)

element, dataset = :Al, ""
N_cell_fc, N_per_seg = 4, [20, 20, 20, 20, 60]
cut_margin_THz, qΓtol = 0.15, 5e-2
max_cuts = 40
n_lev, n_res, n_rand = 5, 10, 15
n_bench = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 6

result = load_model(element, 12, 4, 6, 2; dataset_name=dataset)
model  = result.model; lin_params = result.lin_params; n_params = length(lin_params)
P = result.P; Ap = Diagonal(result.W)*result.A/P; Yw = result.W.*result.Y; λ = 1.0/size(Ap,1)
outdir = "$(result.dir)/results/sdp_prototype"; mkpath(outdir)

a_eq = ACEWorkflow.relax_lattice_constant(model, element)
sys0 = ACEWorkflow.Elasticity.reference_system(element; a=a_eq)
L0   = ustrip.(ACEWorkflow.Elasticity.lattice_matrix(sys0.cell.cell_vectors))
eV_to_GPa = 160.2176621/abs(det(L0))
H_el = elastic_hessian_basis(model; element=element, a=a_eq)
c11_0 = reshape(H_el,36,n_params)[1,:]; c12_0 = reshape(H_el,36,n_params)[7,:]; c44_0 = reshape(H_el,36,n_params)[22,:]
lattice_basis(a) = ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(ACEWorkflow.Elasticity.reference_system(element; a=a), model))
b_prime        = ForwardDiff.derivative(lattice_basis, a_eq)
b_double_prime = ForwardDiff.derivative(a -> ForwardDiff.derivative(lattice_basis, a), a_eq)
born_rows  = vcat(c44_0', (c11_0.-c12_0)', (c11_0.+2 .*c12_0)', b_double_prime')
born_lower = [0.1, 1.0, 0.1, 1e-9]

bp = bandpath_Dk(result, model, element, a_eq, N_cell_fc; N_per_seg=N_per_seg)
ω2_cut = (cut_margin_THz / FREQ_THz)^2
nb = 3*bp.Np
keep = [iq for iq in eachindex(bp.Bq) if bp.qnorm[iq] >= qΓtol]
BxR = Dict(iq => real.(bp.Bq[iq] / P) for iq in keep)      # precompute once, both routes
BxI = Dict(iq => imag.(bp.Bq[iq] / P) for iq in keep)
@printf("%d q-points, %d kept; %d LMI blocks of %dx%d\n",
        length(bp.Bq), length(keep), length(keep), 2nb, 2nb); flush(stdout)

# ── the same observation selection the legacy script uses ────────────────────
C = Symmetric(Ap'*Ap .+ λ.*(P'*P)); Cf = cholesky(C)
AtX = Cf\Matrix(Ap'); θ̃ = Cf\(Ap'*Yw)
leverage = vec(sum(Ap'.*AtX; dims=1)); residual = Yw .- Ap*θ̃
lev_idx = sortperm(leverage; rev=true)[1:n_lev]
res_idx = Int[]; for i in sortperm(abs.(residual); rev=true); i in lev_idx && continue; push!(res_idx,i); length(res_idx)==n_res && break; end
taken = Set(vcat(lev_idx,res_idx)); rand_idx = Int[]
while length(rand_idx) < n_rand; i = rand(1:length(Yw)); (i in taken) && continue; push!(rand_idx,i); push!(taken,i); end
selected = vcat(lev_idx, res_idx, rand_idx)[1:n_bench]
@printf("benchmarking %d members: %s\n", length(selected), string(selected)); flush(stdout)

# ── route A: cutting plane (verbatim) ────────────────────────────────────────
Hqp = sparse(Ap'*Ap .+ λ.*(P'*P)); qqp = -(Ap'*Yw); osqp = OSQP.Model()
function constrain_member(i, extra_rows, extra_lower)
    rows = vcat(born_rows, extra_rows); lowers = vcat(born_lower, extra_lower)
    A_full = vcat(sparse(Ap[i,:]'), sparse(b_prime'/P), sparse(rows/P))
    l = vcat([Yw[i]],[0.0],lowers); u = vcat([Yw[i]],[0.0],fill(Inf,length(lowers)))
    OSQP.setup!(osqp; P=Hqp, q=qqp, A=A_full, l=l, u=u, max_iter=4_000_000,
                check_termination=25, verbose=false, eps_abs=1e-6, eps_rel=1e-6)
    r = OSQP.solve!(osqp); return P \ r.x, r.info.status
end
function route_cuts(i)
    extra_rows = zeros(0, n_params); extra_lower = Float64[]; ncut = 0; iters = 0
    θ, st = constrain_member(i, extra_rows, extra_lower)
    for it in 0:max_cuts
        soft = soft_modes(θ, bp, ω2_cut); isempty(soft) && break
        iters += 1
        it == max_cuts && break
        for (iq, e) in soft; extra_rows = vcat(extra_rows, cut_row(iq, e, bp)'); push!(extra_lower, ω2_cut); end
        ncut += length(soft)
        θ, st = constrain_member(i, extra_rows, extra_lower)
    end
    return θ, ncut, iters, st
end

# ── route B: one SDP ─────────────────────────────────────────────────────────
function route_sdp(i)
    m = Model(Clarabel.Optimizer); set_silent(m)
    @variable(m, x[1:n_params])
    @objective(m, Min, 0.5*x'*Matrix(Hqp)*x + qqp'*x)
    @constraint(m, dot(Ap[i,:], x) == Yw[i])            # POPS interpolation, exact
    @constraint(m, (b_prime'/P)*x .== 0.0)
    @constraint(m, (born_rows/P)*x .>= born_lower)
    for iq in keep
        Xv = reshape(BxR[iq]*x, nb, nb); Yv = reshape(BxI[iq]*x, nb, nb)
        Xs = Xv .- ω2_cut*Matrix(I, nb, nb)
        M  = [Xs -Yv; Yv Xs]
        @constraint(m, Symmetric(0.5*(M .+ transpose(M))) in PSDCone())
    end
    optimize!(m)
    return P \ value.(x), termination_status(m)
end

interp_err(i, θ) = abs(dot(Ap[i,:], P*θ) - Yw[i])

println("\n", "="^108)
@printf("%-8s | %-34s | %-34s | %s\n", "obs", "CUTTING PLANE", "SDP", "agreement")
@printf("%-8s | %7s %5s %5s %9s | %7s %11s %9s | %10s\n",
        "", "s", "cuts", "iter", "min ω", "s", "status", "min ω", "‖Δθ‖")
println("="^108)
tot_c = 0.0; tot_s = 0.0
for i in selected
    tc = @elapsed ((θc, ncut, iters, stc) = route_cuts(i))
    ts = @elapsed ((θs, sts) = route_sdp(i))
    global tot_c += tc; global tot_s += ts
    @printf("%-8d | %7.1f %5d %5d %9.4f | %7.1f %11s %9.4f | %10.2e\n",
            i, tc, ncut, iters, min_freq_stable(θc, bp),
            ts, string(sts), min_freq_stable(θs, bp), norm(θs .- θc))
    @printf("         | interp err %.2e %19s| interp err %.2e %10s|\n",
            interp_err(i, θc), "", interp_err(i, θs), "")
    flush(stdout)
end
println("="^108)
@printf("TOTAL      cutting plane %.1f s | SDP %.1f s | speedup %.2fx\n",
        tot_c, tot_s, tot_c/tot_s)
