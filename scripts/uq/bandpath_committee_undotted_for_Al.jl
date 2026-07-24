# bandpath_committee_undotted_Al_20_4_6A_2_subset_50.jl
#
# Phonon-stable POPS committee on the UNDOTTED Hessian.  H_k = ∂²B_k/∂r² depends
# only on geometry, so at fixed a_eq it is built ONCE; every member's phonon
# check is D(q,θ) = Σ_k θ_k D_k(q) (matvec + tiny eigen) — cheap enough to be a
# rejection predicate.  Validity needs a fixed a_eq (b′·θ=0 pins each member's
# equilibrium onto the reference geometry).  Pipeline:
#   0. undotted H_basis (once, parallel) → D_k(q) on the FULL band path
#   1. constrain a_eq: mean fit + POPS proposal all pinned to b′=0, b″>0, Born
#   2. phonon sweep check (undotted) → flag naughty members
#   3. constrain naughty: cutting-plane OSQP, positive-curvature rows, a_eq held
#   4. rejection sample the committee, band-path phonon check as the predicate
#   5. verify every final sample; 6. compare vs naive POPS + coverage
#
# All phonon/undotted machinery + the (Γ-correct) plotting live in
# ../bandpath_phonon_uq/lib.jl so every model run shares one source of truth.
# To run for a different model: change `dataset` below.
#
# Run:  julia --project -t <N> scripts/uq/bandpath_committee_undotted_Al_20_4_6A_2_subset_50.jl

include(joinpath(@__DIR__, "..", "bandpath_phonon_uq", "lib.jl"))
using SparseArrays, OSQP, Random
Random.seed!(1234)

element        = :Al
dataset        = ""            # "" → full-dataset model (Al_20_4_6A_2_); e.g. "subset_50_percent"
a_experimental = nothing       # Float (Å) to pin a_eq experimentally; nothing → mean model
N_cell_fc      = 3
N_per_seg      = 20
cut_margin_THz = 0.15
n_lev, n_res, n_rand = 5, 10, 15
max_cuts       = 40
test_stride    = 20

result     = load_model(element, 20, 4, 6, 3; dataset_name=dataset)
model      = result.model; lin_params = result.lin_params; n_params = length(lin_params)
P = result.P; Ap = Diagonal(result.W)*result.A/P; Yw = result.W.*result.Y; λ = 1.0/size(Ap,1)
outdir = "$(result.dir)/results/bandpath_undotted"; mkpath(outdir)
@printf("Model %s: %d params, %d threads.  Outputs → %s\n", result.name, n_params, Threads.nthreads(), outdir)

# ── reference geometry, Born + a_eq rows ─────────────────────────────────────
a_mean = ACEWorkflow.relax_lattice_constant(model, element)
a_eq   = isnothing(a_experimental) ? a_mean : a_experimental
@printf("a_mean = %.5f Å;  reference a_eq = %.5f Å%s\n", a_mean, a_eq, isnothing(a_experimental) ? "" : "  (experimental)")
sys0 = ACEWorkflow.Elasticity.reference_system(element; a=a_eq)
L0   = ustrip.(ACEWorkflow.Elasticity.lattice_matrix(sys0.cell.cell_vectors)); eV_to_GPa = 160.2176621/abs(det(L0))
H_el = elastic_hessian_basis(model; element=element, a=a_eq)
c11_0 = reshape(H_el,36,n_params)[1,:]; c12_0 = reshape(H_el,36,n_params)[7,:]; c44_0 = reshape(H_el,36,n_params)[22,:]
born(θ) = (dot(c11_0,θ)*eV_to_GPa, dot(c12_0,θ)*eV_to_GPa, dot(c44_0,θ)*eV_to_GPa)
lattice_basis(a) = ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(ACEWorkflow.Elasticity.reference_system(element; a=a), model))
b_prime        = ForwardDiff.derivative(lattice_basis, a_eq)
b_double_prime = ForwardDiff.derivative(a -> ForwardDiff.derivative(lattice_basis, a), a_eq)
born_rows  = vcat(c44_0', (c11_0.-c12_0)', (c11_0.+2 .*c12_0)', b_double_prime')
born_lower = [0.1, 1.0, 0.1, 1e-9]

# ── Stage 0: undotted Hessian (once) → band-path D_k(q) over the FULL path ───
bp = bandpath_Dk(result, model, element, a_eq, N_cell_fc; N_per_seg=N_per_seg)
ω2_cut = (cut_margin_THz / FREQ_THz)^2
let fm = bands(lin_params, bp)
    @printf("Undotted band-path, mean model: ω ∈ [%.3f, %.3f] THz\n", minimum(fm), maximum(fm))
end

# ── Stage 1: constrain a_eq (mean fit) + leverage/residual forest ────────────
C = Symmetric(Ap'*Ap .+ λ.*(P'*P)); Cf = cholesky(C)
AtX = Cf\Matrix(Ap'); θ̃ = Cf\(Ap'*Yw)
leverage = vec(sum(Ap'.*AtX; dims=1)); residual = Yw .- Ap*θ̃
forest_member(i) = lin_params .+ (P \ (AtX[:, i] .* (residual[i]/leverage[i])))
Hqp = sparse(Ap'*Ap .+ λ.*(P'*P)); qqp = -(Ap'*Yw); osqp = OSQP.Model()
function constrain_member(i, extra_rows, extra_lower)
    rows = vcat(born_rows, extra_rows); lowers = vcat(born_lower, extra_lower)
    A_full = vcat(sparse(Ap[i,:]'), sparse(b_prime'/P), sparse(rows/P))
    l = vcat([Yw[i]],[0.0],lowers); u = vcat([Yw[i]],[0.0],fill(Inf,length(lowers)))
    OSQP.setup!(osqp; P=Hqp, q=qqp, A=A_full, l=l, u=u, max_iter=4_000_000, check_termination=25, verbose=false, eps_abs=1e-6, eps_rel=1e-6)
    r = OSQP.solve!(osqp); return P \ r.x, r.info.status
end
function mean_fit(extra_rows=zeros(0,n_params), extra_lower=Float64[])   # b′=0, Born + extra
    rows = vcat(born_rows, extra_rows); lowers = vcat(born_lower, extra_lower)
    A_full = vcat(sparse(b_prime'/P), sparse(rows/P))
    l = vcat([0.0],lowers); u = vcat([0.0],fill(Inf,length(lowers)))
    OSQP.setup!(osqp; P=Hqp, q=qqp, A=A_full, l=l, u=u, max_iter=4_000_000, check_termination=25, verbose=false, eps_abs=1e-6, eps_rel=1e-6)
    return P \ OSQP.solve!(osqp).x
end
# the MEAN must obey the same phonon constraints as the committee — sweep & cut it too
θ_mean = mean_fit(); mean_er = zeros(0,n_params); mean_el = Float64[]; mean_cuts = 0
for it in 0:max_cuts
    global θ_mean, mean_er, mean_el, mean_cuts
    soft = soft_modes(θ_mean, bp, ω2_cut); isempty(soft) && break
    for (iq,e) in soft; mean_er = vcat(mean_er, cut_row(iq,e,bp)'); push!(mean_el, ω2_cut); end
    mean_cuts += length(soft); θ_mean = mean_fit(mean_er, mean_el)
end
min_freq_stable(θ_mean, bp) >= cut_margin_THz - 1e-6 || @warn "mean model still soft after $mean_cuts cuts"
@printf("Constrained mean (%d phonon cuts): C11=%.1f C12=%.1f C44=%.1f GPa, b′·θ=%.1e, min ω=%.3f THz\n",
        mean_cuts, born(θ_mean)..., dot(b_prime, θ_mean), min_freq_stable(θ_mean, bp))

lev_idx = sortperm(leverage; rev=true)[1:n_lev]
res_idx = Int[]; for i in sortperm(abs.(residual); rev=true); i in lev_idx && continue; push!(res_idx,i); length(res_idx)==n_res && break; end
taken = Set(vcat(lev_idx,res_idx)); rand_idx = Int[]
while length(rand_idx) < n_rand; i = rand(1:length(Yw)); (i in taken) && continue; push!(rand_idx,i); push!(taken,i); end
selected = vcat(lev_idx, res_idx, rand_idx)

# ── Stages 2+3: a_eq-constrain each member, phonon-check, constrain naughty ──
println("\n── a_eq-constrain + phonon-repair the proposal cloud ───────")
committee = Vector{Vector{Float64}}(undef, length(selected)); n_cuts = zeros(Int, length(selected))
naive = [forest_member(i) for i in selected]
for (k, i) in enumerate(selected)
    extra_rows = zeros(0, n_params); extra_lower = Float64[]
    θ, _ = constrain_member(i, extra_rows, extra_lower)
    for it in 0:max_cuts
        soft = soft_modes(θ, bp, ω2_cut)
        isempty(soft) && break
        it == max_cuts && (@warn "obs $i hit max_cuts (min ω=$(round(min_freq_stable(θ,bp);digits=3)))"; break)
        for (iq, e) in soft; extra_rows = vcat(extra_rows, cut_row(iq, e, bp)'); push!(extra_lower, ω2_cut); end
        n_cuts[k] += length(soft)
        θ, _ = constrain_member(i, extra_rows, extra_lower)
    end
    committee[k] = θ
end
@printf("Repaired (≥1 cut): %d / %d\n", count(>(0), n_cuts), length(selected))

# ── Stage 4: rejection sample, band-path phonon check as the predicate ──────
println("── Rejection sample committee (phonon check = predicate) ───")
con_deltas = reduce(hcat, committee)' .- θ_mean'
hyp_eig, hyp_bound = hypercube(Matrix(con_deltas)); K_ref = dot(θ_mean, b_double_prime)
n_ck = Ref(0); n_ph = Ref(0)
predicate = θ -> begin
    n_ck[] += 1
    all(born_lower .<= born_rows*θ) || return false
    abs(dot(b_prime, θ .- θ_mean)/K_ref) <= 0.1 || return false
    n_ph[] += 1
    min_freq_stable(θ, bp) >= cut_margin_THz - 1e-6
end
rej_mat, _ = rejection_sample_hypercube(hyp_eig, hyp_bound, θ_mean, predicate;
                                        number_of_committee_members=length(selected), max_attempts=2_000_000)
@printf("  funnel: %d drawn → %d past Born/a_eq → %d accepted (%.1f%%)\n", n_ck[], n_ph[], length(selected), 100*length(selected)/n_ck[])
rej_committee = [rej_mat[:, i] for i in 1:size(rej_mat, 2)]
writedlm("$outdir/committee_rejection.csv", rej_mat', ',')
writedlm("$outdir/committee_repaired.csv", reduce(hcat, committee)', ',')
writedlm("$outdir/theta_mean.csv", θ_mean, ',')          # phonon-repaired constrained mean

# ── Stage 5: verify (non-acoustic band-path minimum) ────────────────────────
minf_rej   = [min_freq_stable(θ, bp) for θ in rej_committee]
minf_naive = [min_freq_stable(θ, bp) for θ in naive]
@printf("\n  rejection committee: min band ω ∈ [%.3f, %.3f] THz — %d/%d unstable\n",
        minimum(minf_rej), maximum(minf_rej), count(<(-0.05), minf_rej), length(minf_rej))
@printf("  naive POPS         : min band ω ∈ [%.3f, %.3f] THz — %d/%d unstable\n",
        minimum(minf_naive), maximum(minf_naive), count(<(-0.05), minf_naive), length(minf_naive))

# ── Stage 6: propagate to phonons (Γ-correct plots) + coverage ──────────────
plot_committee_bands(rej_committee, θ_mean,     bp, "$(result.name) — constrained committee (band-path, a_eq-fixed)", "$outdir/bands_constrained.png")
plot_committee_bands(naive,         lin_params, bp, "$(result.name) — naive POPS committee", "$outdir/bands_naive.png")

println("\n── Test-set coverage (constrained committee) ───────────────")
pr = committee_predictions(model, rej_committee, "data/Al/manual_df_test_Al.xyz"; stride=test_stride, point_params=θ_mean)
let ev = mean((pr.tE.<pr.loE).|(pr.tE.>pr.hiE))
    @printf("  energy RMSE=%.4g eV, coverage=%.1f%% (EV %.1f%%)\n", sqrt(mean((pr.pE.-pr.tE).^2)), (1-ev)*100, ev*100)
end
ACEpotentials.Models.set_linear_parameters!(model, lin_params)
println("\n══ RESULT ══════════════════════════════════════════════════")
@printf("  constrained committee: %d/%d phonon-unstable | naive: %d/%d\n",
        count(<(-0.05), minf_rej), length(minf_rej), count(<(-0.05), minf_naive), length(minf_naive))
println("All outputs → $outdir/")

@printf("  %d test configs\n", pr.n)
eR = parity_plot(pr.tE, pr.pE, pr.loE, pr.hiE, "DFT energy (eV)", "ACE energy (eV)", "$outdir/energy_parity.png")
cE = calibration_hist(pr.tE, pr.pE, pr.loE, pr.hiE; label="Energy", path="$outdir/energy_calibration.png")
@printf("  ENERGY  RMSE=%.4g eV   coverage=%.1f%%   bias=%.0f%% MAE\n", eR, cE.coverage, cE.bias)
if !isempty(pr.tF)
    fR = parity_plot(pr.tF, pr.pF, pr.loF, pr.hiF, "DFT force (eV/Å)", "ACE force (eV/Å)", "$outdir/force_parity.png"; col=:tomato)
    cF = calibration_hist(pr.tF, pr.pF, pr.loF, pr.hiF; label="Force", path="$outdir/force_calibration.png")
    @printf("  FORCE   RMSE=%.4g eV/Å coverage=%.1f%%   bias=%.0f%% MAE\n", fR, cF.coverage, cF.bias)
end
ACEpotentials.Models.set_linear_parameters!(model, lin_params)
println("\nSaved: bands_constrained.png, bands_naive.png, {energy,force}_parity.png, {energy,force}_calibration.png → $outdir/")
