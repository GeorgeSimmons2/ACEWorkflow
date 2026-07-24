# bandpath_committee_undotted_Al_20_4_6A_2_subset_50.jl
#
# Phonon-stable POPS committee built on the UNDOTTED Hessian.  Key efficiency:
# the per-basis supercell Hessian H_k = ∂²B_k/∂r² depends only on GEOMETRY, so
# at a FIXED lattice constant it is built ONCE; then every parameter vector's
# phonon check is  D(q,θ) = Σ_k θ_k D_k(q)  (a matvec) + a tiny eigen per q.
# Cheap enough to be a rejection predicate.
#
# This validity REQUIRES a fixed a_eq: b′·θ = 0 forces each member's
# equilibrium onto the reference geometry, so the one Hessian we built there is
# genuinely that member's Hessian.  Hence the pipeline:
#
#   0. undotted H_basis (once, parallel) → D_k(q) on a dense band path
#   1. constrain a_eq: mean fit + POPS proposal all pinned to b′=0
#   2. phonon sweep check (undotted D(q,θ)) → flag naughty members
#   3. constrain naughty: cutting-plane OSQP, positive-curvature rows from the
#      soft eigenvector-with-phase, a_eq held fixed
#   4. rejection sample the committee, band-path phonon check as the predicate
#   5. verify every final sample obeys the phonon constraints
#   6. propagate to phonons & compare vs naive POPS (bands overlay + coverage)
#
# The undotted-Hessian extraction is validated against the native dotted
# hessian in scripts/uq/undotted_hessian_test.jl (matches to ~1e-13).
#
# Run:  julia --project -t <N> scripts/uq/bandpath_committee_undotted_Al_20_4_6A_2_subset_50.jl

using LinearAlgebra, DelimitedFiles, Statistics, Printf, Random, Serialization
using SparseArrays, StaticArrays, Unitful, ForwardDiff, CairoMakie, OSQP, StatsBase
using ACEpotentials, ACEWorkflow
using ACEpotentials: potential_energy
import ACEpotentials.Models: evaluate_basis
using AtomsBuilder
using ExtXYZ, AtomsCalculators
import AtomsCalculators: forces
import AtomsCalculatorsUtilities.SitePotentials: PairList, get_neighbours, cutoff_radius
import ACEWorkflow: precompute_force_constants, dynamical_matrix_from_fc, fcc_band_path

Random.seed!(1234)

element        = :Al
a_experimental = nothing      # set to a Float (Å) to pin a_eq experimentally; nothing → mean model
N_cell_fc      = 3            # force-constant supercell (≥ 2×cutoff)
N_per_seg      = 20           # band-path q per segment
cut_margin_THz = 0.15         # required frequency along the band path
n_lev, n_res, n_rand = 5, 10, 15    # committee draw composition
max_cuts       = 40
test_stride    = 10

result     = load_model(element, 20, 4, 6, 4)
model      = result.model
lin_params = result.lin_params
n_params   = length(lin_params)
P          = result.P
Ap = Diagonal(result.W) * result.A / P
Yw = result.W .* result.Y
λ  = 1.0 / size(Ap, 1)
outdir = "$(result.dir)/results/bandpath_undotted"; mkpath(outdir)
@printf("Model %s: %d params, %d threads.  Outputs → %s\n", result.name, n_params, Threads.nthreads(), outdir)

# ═════════════════════════════════════════════════════════════════════════════
#  Undotted per-basis Hessian  (validated in undotted_hessian_test.jl)
# ═════════════════════════════════════════════════════════════════════════════
function site_basis_hessian(model, Rs, Zs, z0, ps, st)
    nR = length(Rs)
    x0 = collect(Float64, reinterpret(Float64, Rs))
    tovec(x) = [SVector{3,eltype(x)}(x[3i-2], x[3i-1], x[3i]) for i in 1:nR]
    Bfun(x)    = evaluate_basis(model, tovec(x), Zs, z0, ps, st)
    gradflat(x) = vec(ForwardDiff.jacobian(Bfun, x))
    Hflat = ForwardDiff.jacobian(gradflat, x0)
    N_basis = length(Bfun(x0))
    return reshape(Hflat, N_basis, 3nR, 3nR)
end

function hessian_basis_undotted(sys, V)              # → H[i,j,k] (eV/Å²), parallel over atoms
    nlist = PairList(sys, cutoff_radius(V)); Nat = length(sys); D = 3; ps = V.ps; st = V.st; m = V.model
    _, Rs0, Zs0, z00 = get_neighbours(sys, V, nlist, 1)
    N_basis = length(evaluate_basis(m, Rs0, Zs0, z00, ps, st))
    Ht = [zeros(D*Nat, D*Nat, N_basis) for _ in 1:Threads.nthreads()]
    Threads.@threads :static for i in 1:Nat
        H = Ht[Threads.threadid()]
        Js, Rs, Zs, z0 = get_neighbours(sys, V, nlist, i)
        Hi = site_basis_hessian(m, Rs, Zs, z0, ps, st)
        Ji = (i-1)*D .+ (1:D)
        for (α1, j1) in enumerate(Js), (α2, j2) in enumerate(Js)
            A1 = (α1-1)*D .+ (1:D); A2 = (α2-1)*D .+ (1:D)
            J1 = (j1-1)*D .+ (1:D); J2 = (j2-1)*D .+ (1:D)
            @views for k in 1:N_basis
                blk = Hi[k, A1, A2]
                H[J1,J2,k] .+= blk; H[J1,Ji,k] .-= blk; H[Ji,J2,k] .-= blk; H[Ji,Ji,k] .+= blk
            end
        end
    end
    return sum(Ht)
end

# ═════════════════════════════════════════════════════════════════════════════
#  Reference geometry, Born + a_eq rows
# ═════════════════════════════════════════════════════════════════════════════
a_mean = ACEWorkflow.relax_lattice_constant(model, element)
a_eq   = isnothing(a_experimental) ? a_mean : a_experimental
@printf("a_mean = %.5f Å;  reference a_eq = %.5f Å%s\n", a_mean, a_eq, isnothing(a_experimental) ? "" : "  (experimental)")

sys0 = ACEWorkflow.Elasticity.reference_system(element; a=a_eq)
L0   = ustrip.(ACEWorkflow.Elasticity.lattice_matrix(sys0.cell.cell_vectors))
eV_to_GPa = 160.2176621 / abs(det(L0))
H_el = elastic_hessian_basis(model; element=element, a=a_eq)
c11_0 = reshape(H_el, 36, n_params)[1, :]; c12_0 = reshape(H_el, 36, n_params)[7, :]; c44_0 = reshape(H_el, 36, n_params)[22, :]
born(θ) = (dot(c11_0,θ)*eV_to_GPa, dot(c12_0,θ)*eV_to_GPa, dot(c44_0,θ)*eV_to_GPa)
lattice_basis(a) = ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(ACEWorkflow.Elasticity.reference_system(element; a=a), model))
b_prime        = ForwardDiff.derivative(lattice_basis, a_eq)
b_double_prime = ForwardDiff.derivative(a -> ForwardDiff.derivative(lattice_basis, a), a_eq)
born_rows  = vcat(c44_0', (c11_0.-c12_0)', (c11_0.+2 .*c12_0)', b_double_prime')
born_lower = [0.1, 1.0, 0.1, 1e-9]

# ═════════════════════════════════════════════════════════════════════════════
#  Stage 0 — undotted Hessian (once) → band-path D_k(q)
# ═════════════════════════════════════════════════════════════════════════════
sys_prim  = bulk(element; a=a_eq*u"Å")
sys_super = bulk(element; a=a_eq*u"Å", cubic=true) * (N_cell_fc, N_cell_fc, N_cell_fc)
ACEpotentials.Models.set_linear_parameters!(model, lin_params)
fc0 = precompute_force_constants(sys_prim, sys_super, model); Np = fc0.Np

hb_file = "$(result.dir)/results/undotted_Hbasis_$(N_cell_fc)x$(N_cell_fc)x$(N_cell_fc).jls"
H_basis = if isfile(hb_file)
    c = deserialize(hb_file); abs(c.a_eq - a_eq) < 1e-4 || error("cached H_basis a=$(c.a_eq) ≠ $a_eq")
    println("Loaded undotted H_basis cache."); c.H_basis
else
    @printf("Building undotted per-basis Hessian on %d-atom supercell (parallel) …\n", length(sys_super))
    t = @elapsed Hb = hessian_basis_undotted(sys_super, model)
    @printf("  done: %s in %.1f min\n", size(Hb), t/60)
    serialize(hb_file, (H_basis=Hb, a_eq=a_eq, N_cell=N_cell_fc)); Hb
end
N3super = 3*length(sys_super)

# band path + per-basis D_k(q) (Bloch transform of each H_basis[:,:,k])
q_list, x_vals, x_ticks, labels, _ = fcc_band_path(fc0.L; N_per_seg=N_per_seg)
keep = [norm(q) > 5e-2 for q in q_list]; qpath = q_list[keep]
@printf("Band path: %d q (dropped %d near-Γ).  Precomputing D_k(q) …\n", length(qpath), count(!, keep))
Bq = Vector{Matrix{ComplexF64}}(undef, length(qpath))
for (iq, q) in enumerate(qpath)
    M = Matrix{ComplexF64}(undef, (3Np)^2, n_params)
    for k in 1:n_params
        M[:, k] = vec(dynamical_matrix_from_fc(merge(fc0, (H = reshape(H_basis[:, :, k], N3super, N3super),)), q))
    end
    Bq[iq] = M
end
ω2_cut = (cut_margin_THz / FREQ_THz)^2

# CHEAP undotted band-path phonon check  (this is what "any phonons we do" uses)
function bandpath_bands(θ)                 # frequencies (THz), 3Np × n_q
    F = Matrix{Float64}(undef, 3Np, length(qpath))
    for iq in eachindex(qpath)
        ev = eigvals(Hermitian(reshape(Bq[iq]*θ, 3Np, 3Np)))
        F[:, iq] = sign.(ev) .* sqrt.(abs.(ev)) .* FREQ_THz
    end
    return F
end
function bandpath_soft(θ)                  # (min freq THz, soft (iq,eigvec) list)
    fmin = Inf; soft = Tuple{Int,Vector{ComplexF64}}[]
    for iq in eachindex(qpath)
        Fe = eigen(Hermitian(reshape(Bq[iq]*θ, 3Np, 3Np)))
        for ν in 1:3Np
            f = sign(Fe.values[ν])*sqrt(abs(Fe.values[ν]))*FREQ_THz; f < fmin && (fmin = f)
            Fe.values[ν] < ω2_cut && push!(soft, (iq, Fe.vectors[:, ν]))
        end
    end
    return fmin, soft
end
# exact linear curvature row for mode e at path point iq (eigenvector-with-phase)
cut_row(iq, e) = (c = vec(conj(e)*transpose(e)); real(vec(transpose(c) * Bq[iq])))

# sanity: undotted mean spectrum should be sensible for Al
fmean = bandpath_bands(lin_params)
@printf("Undotted band-path check, mean model: ω ∈ [%.3f, %.3f] THz\n", minimum(fmean), maximum(fmean))

# ═════════════════════════════════════════════════════════════════════════════
#  Stage 1 — constrain a_eq (mean fit) + leverage/residual forest
# ═════════════════════════════════════════════════════════════════════════════
C = Symmetric(Ap'*Ap .+ λ.*(P'*P)); Cf = cholesky(C); Cmat = sparse(Matrix(C))
AtX = Cf \ Matrix(Ap'); θ̃ = Cf \ (Ap'*Yw)
leverage = vec(sum(Ap'.*AtX; dims=1)); residual = Yw .- Ap*θ̃
forest_member(i) = lin_params .+ (P \ (AtX[:, i] .* (residual[i]/leverage[i])))

Hqp = sparse(Ap'*Ap .+ λ.*(P'*P)); qqp = -(Ap'*Yw); osqp = OSQP.Model()
# constrained-POPS for a_eq (+ optional extra rows): pin obs i, b′=0, Born + extra
function constrain_member(i, extra_rows, extra_lower)
    rows = vcat(born_rows, extra_rows); lowers = vcat(born_lower, extra_lower)
    A_full = vcat(sparse(Ap[i, :]'), sparse(b_prime'/P), sparse(rows/P))
    l = vcat([Yw[i]], [0.0], lowers); u = vcat([Yw[i]], [0.0], fill(Inf, length(lowers)))
    OSQP.setup!(osqp; P=Hqp, q=qqp, A=A_full, l=l, u=u, max_iter=4_000_000,
                check_termination=25, verbose=false, eps_abs=1e-6, eps_rel=1e-6)
    r = OSQP.solve!(osqp); return P \ r.x, r.info.status
end
# mean fit at fixed a_eq (b′=0, Born) — no data pin
function mean_fit()
    A_full = vcat(sparse(b_prime'/P), sparse(born_rows/P))
    l = vcat([0.0], born_lower); u = vcat([0.0], fill(Inf, length(born_lower)))
    OSQP.setup!(osqp; P=Hqp, q=qqp, A=A_full, l=l, u=u, max_iter=4_000_000,
                check_termination=25, verbose=false, eps_abs=1e-6, eps_rel=1e-6)
    return P \ OSQP.solve!(osqp).x
end
θ_mean = mean_fit()
@printf("Constrained mean: C11=%.1f C12=%.1f C44=%.1f GPa, b′·θ=%.1e, bands min %.3f THz\n",
        born(θ_mean)..., dot(b_prime, θ_mean), minimum(bandpath_bands(θ_mean)))

# committee draw
lev_idx = sortperm(leverage; rev=true)[1:n_lev]
res_idx = Int[]; for i in sortperm(abs.(residual); rev=true); i in lev_idx && continue; push!(res_idx, i); length(res_idx)==n_res && break; end
taken = Set(vcat(lev_idx, res_idx)); rand_idx = Int[]
while length(rand_idx) < n_rand; i = rand(1:length(Yw)); (i in taken) && continue; push!(rand_idx, i); push!(taken, i); end
selected = vcat(lev_idx, res_idx, rand_idx)
groups = vcat(fill("leverage", n_lev), fill("residual", n_res), fill("random", n_rand))

# ═════════════════════════════════════════════════════════════════════════════
#  Stages 2+3 — a_eq-constrain each member, phonon-check, constrain the naughty
# ═════════════════════════════════════════════════════════════════════════════
println("\n── a_eq-constrain + phonon-repair the proposal cloud ───────")
committee = Vector{Vector{Float64}}(undef, length(selected)); n_cuts = zeros(Int, length(selected))
naive     = [forest_member(i) for i in selected]          # unconstrained, for comparison
for (k, i) in enumerate(selected)
    extra_rows = zeros(0, n_params); extra_lower = Float64[]
    θ, _ = constrain_member(i, extra_rows, extra_lower)   # constrained POPS for a_eq
    for it in 0:max_cuts
        fmin, soft = bandpath_soft(θ)
        isempty(soft) && break
        it == max_cuts && (@warn "obs $i hit max_cuts (min ω=$(round(fmin;digits=3)))"; break)
        for (iq, e) in soft; extra_rows = vcat(extra_rows, cut_row(iq, e)'); push!(extra_lower, ω2_cut); end
        n_cuts[k] += length(soft)
        θ, _ = constrain_member(i, extra_rows, extra_lower)
    end
    committee[k] = θ
end
@printf("Repaired (≥1 cut): %d / %d\n", count(>(0), n_cuts), length(selected))

# ═════════════════════════════════════════════════════════════════════════════
#  Stage 4 — rejection sample, band-path phonon check as the predicate
# ═════════════════════════════════════════════════════════════════════════════
println("── Rejection sample committee (phonon check = predicate) ───")
con_deltas = reduce(hcat, committee)' .- θ_mean'          # a_eq-fixed, stable cloud
hyp_eig, hyp_bound = hypercube(Matrix(con_deltas))
K_ref = dot(θ_mean, b_double_prime)
n_ck = Ref(0); n_ph = Ref(0)
predicate = θ -> begin
    n_ck[] += 1
    all(born_lower .<= born_rows*θ) || return false
    abs(dot(b_prime, θ .- θ_mean)/K_ref) <= 0.1 || return false     # a_eq band
    n_ph[] += 1
    first(bandpath_soft(θ)) >= cut_margin_THz - 1e-6                # phonon predicate (undotted)
end
rej_mat, _ = rejection_sample_hypercube(hyp_eig, hyp_bound, θ_mean, predicate;
                                        number_of_committee_members=length(selected), max_attempts=2_000_000)
@printf("  funnel: %d drawn → %d past Born/a_eq → %d accepted (%.1f%%)\n",
        n_ck[], n_ph[], length(selected), 100*length(selected)/n_ck[])
rej_committee = [rej_mat[:, i] for i in 1:size(rej_mat, 2)]
writedlm("$outdir/committee_rejection.csv", rej_mat', ',')
writedlm("$outdir/committee_repaired.csv", reduce(hcat, committee)', ',')

# ═════════════════════════════════════════════════════════════════════════════
#  Stage 5 — verify final samples obey the phonon constraints
# ═════════════════════════════════════════════════════════════════════════════
minf_rej   = [minimum(bandpath_bands(θ)) for θ in rej_committee]
minf_naive = [minimum(bandpath_bands(θ)) for θ in naive]
@printf("\n── Verification (undotted band path) ───────────────────────\n")
@printf("  rejection committee: min band ω ∈ [%.3f, %.3f] THz — %d/%d unstable\n",
        minimum(minf_rej), maximum(minf_rej), count(<(-0.05), minf_rej), length(minf_rej))
@printf("  naive POPS         : min band ω ∈ [%.3f, %.3f] THz — %d/%d unstable\n",
        minimum(minf_naive), maximum(minf_naive), count(<(-0.05), minf_naive), length(minf_naive))

# ═════════════════════════════════════════════════════════════════════════════
#  Stage 6 — propagate to phonons & compare vs naive (undotted bands + coverage)
# ═════════════════════════════════════════════════════════════════════════════
function committee_bands_plot(members, title, path)
    fig = Figure(size=(820, 500)); ax = Axis(fig[1,1]; xlabel="Wave vector", ylabel="Frequency (THz)",
        title=title, xticks=(x_ticks[2:end], labels[2:end]), xgridvisible=false)
    xg = cumsum([0.0; [norm(qpath[i]-qpath[i-1]) for i in 2:length(qpath)]])
    for θ in members
        F = bandpath_bands(θ); unstable = minimum(F) < -0.05
        for b in 1:3Np; lines!(ax, xg, F[b,:]; color = unstable ? RGBAf(0.8,0.15,0.15,0.4) : RGBAf(0.4,0.4,0.4,0.35), linewidth=1.0); end
    end
    Fm = bandpath_bands(θ_mean); for b in 1:3Np; lines!(ax, xg, Fm[b,:]; color=RGBAf(0,0.3,0.7,0.95), linewidth=2.0); end
    hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=0.8); save(path, fig)
end
committee_bands_plot(rej_committee, "$(result.name) — constrained committee (band-path, a_eq-fixed)", "$outdir/bands_constrained.png")
committee_bands_plot(naive,         "$(result.name) — naive POPS committee", "$outdir/bands_naive.png")

# test-set parity + coverage for the constrained committee
println("\n── Test-set coverage (constrained committee) ───────────────")
ACEpotentials.Models.set_committee!(model, rej_committee); ACEpotentials.Models.set_linear_parameters!(model, lin_params)
tc = ExtXYZ.load("data/Al/manual_df_test_Al.xyz")[1:test_stride:end]
pE=Float64[];tEn=Float64[];loE=Float64[];hiE=Float64[]
for cfg in tc
    E, co = @committee potential_energy(cfg, model)
    push!(pE, ustrip(E)); push!(loE, minimum(ustrip.(co))); push!(hiE, maximum(ustrip.(co))); push!(tEn, ustrip(cfg[:dft_energy]))
end
let err = tEn.-pE, mae = mean(abs.(tEn.-pE)); ev = mean((tEn.<loE).|(tEn.>hiE))
    @printf("  energy RMSE=%.4g eV, coverage=%.1f%% (EV %.1f%%)\n", sqrt(mean(err.^2)), (1-ev)*100, ev*100)
end
ACEpotentials.Models.set_linear_parameters!(model, lin_params)

println("\n══ RESULT ══════════════════════════════════════════════════")
@printf("  constrained committee: %d/%d phonon-unstable (band path)\n", count(<(-0.05), minf_rej), length(minf_rej))
@printf("  naive POPS committee : %d/%d phonon-unstable\n", count(<(-0.05), minf_naive), length(minf_naive))
println("All outputs → $outdir/")
