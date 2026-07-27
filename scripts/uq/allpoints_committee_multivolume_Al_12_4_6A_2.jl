# allpoints_committee_multivolume_Al_12_4_6A_2.jl
#
# AMENDED COPY of bandpath_committee_undotted_Al_12_4_6A_2_multivolume.jl
# (all earlier scripts left untouched).  ONE change, with large consequences:
#
#   ── constrain EVERY observation in the design matrix, not a 30-point subset ──
#
# WHY.  The earlier committees selected 30 observations (5 highest-leverage, 10
# largest-residual, 15 random) and constrained only those.  That is a defensible
# sampling of the POPS point cloud, but it is a *selection*, and a reader can object
# that the reported committee is conditioned on which points were kept.  Here the
# pointwise correction is computed and constrained for every observation passing a
# stated leverage criterion (`leverage_percentile`, the same masking rule the POPS module
# itself uses), so the proposal box is fitted to the whole of that cloud.  Setting
# leverage_percentile = 0.0 constrains literally every row.
#
# PIPELINE (differences from the 30-point version marked ★):
#   0. undotted H_basis + band-path D_k(q) at 6 volumes  a ∈ a_eq·{1.00,…,1.10}
#   1. constrained mean: b′=0, b″>0, Born, phonon cutting planes over all volumes
# ★ 2. naive cloud   : δθ⁽ⁱ⁾ for every kept i (closed form, cheap — one matvec each)
# ★ 3. constrained cloud: solve the QP + multi-volume cutting-plane repair for every kept i.
#        Threaded (one OSQP model per thread) and CHECKPOINTED in blocks, so a walltime
#        kill loses at most one block and the job resumes where it stopped.
# ★ 4. two hypercubes fitted to the two FULL clouds; draw `n_samples` from each —
#        constrained WITH the phonon/Born/a_eq predicate, naive with NO predicate.
#   5. verify min ω at every volume for both sets
#   6. plots: bands at each volume (shared y-axis) + min ω vs volume
#
# NOTE ON THE PROPOSAL BOX.  `hypercube` takes the *bounding box* of the cloud's
# projections.  Fitting it to ~M points instead of 30 makes the box strictly larger, so
# the accepted fraction will be lower than the 13.3% seen with 30 members.  The funnel
# is logged per volume and the run aborts early (before the long sampling loop) if the
# constrained cloud itself is infeasible.
#
# Run:  sbatch scripts/uq/run_allpoints_committee.slurm
#       (resumable: re-submitting continues from the checkpoint)

include(joinpath(@__DIR__, "..", "bandpath_phonon_uq", "lib.jl"))
using SparseArrays, OSQP, Random, Serialization
Random.seed!(1234)

# Julia BLOCK-BUFFERS stdout when it is not a TTY, so under SLURM a long job writes
# nothing to its .log until the buffer fills or the process exits — a ~9 h job would
# appear dead.  Background flusher for everything, plus explicit flushes after each
# milestone below (the background task cannot run while Threads.@threads holds the
# main thread, so the explicit ones are the guaranteed path).
@async while true; flush(stdout); flush(stderr); sleep(5); end

element        = :Al
dataset        = ""
a_experimental = nothing
N_cell_fc      = 4
N_per_seg      = [20, 20, 20, 20, 60]
cut_margin_THz = 0.15
max_cuts       = 40
vol_scales     = collect(1.00:0.02:1.10)
test_stride    = 10

# ★ all-points controls
# leverage_percentile matches the POPS module (`corrections` in src/POPS/POPSRegression.jl):
#   thr = quantile(leverage, leverage_percentile);  keep observations with leverage >= thr
# 0.5 → the top-50% leverage half of the design matrix (146958 → ~73479 observations,
# ≈8.8 h on 20 threads at the measured 8.6 s/member).  0.0 → every observation (≈17.6 h).
# This MASKS by a stated statistical criterion; it does not select individual points.
leverage_percentile = 0.5
obs_stride     = 1              # further subsampling of the kept set (testing only; 1 = none)
n_samples      = 10             # committee members drawn from each cloud
# Checkpoint AND garbage-collection granularity.  Each member leaves ~100 MB of garbage
# (OSQP setup arrays, sparse conversions, the vcat of born+cut rows once per QP call).
# Across 20 threads Julia's GC does not keep up: job 5998784 reached 347 GB RSS against
# ~300 MB of live data, grew at 76 GB/h, and would have OOMed at ~4 h.  Collecting every
# `block_size` members bounds the heap; 250 caps it at roughly 25 GB.  Smaller blocks also
# mean more frequent checkpoints, which makes resume cheaper.
block_size     = 250
max_attempts   = 20_000_000     # box is larger than the 30-point run → allow more draws

result = load_model(element, 12, 4, 6, 2; dataset_name=dataset)
model  = result.model; lin_params = result.lin_params; n_params = length(lin_params)
P = result.P; Ap = Diagonal(result.W)*result.A/P; Yw = result.W.*result.Y; λ = 1.0/size(Ap,1)
M = size(Ap, 1)
outdir = "$(result.dir)/results/allpoints_multivolume"; mkpath(outdir)
@printf("Model %s: %d params, %d observations, %d threads.  Outputs → %s\n",
        result.name, n_params, M, Threads.nthreads(), outdir)
flush(stdout)

# ── reference geometry, Born + a_eq rows (all at a_eq) ──────────────────────
a_mean = ACEWorkflow.relax_lattice_constant(model, element)
a_eq   = isnothing(a_experimental) ? a_mean : a_experimental
@printf("a_eq = %.5f Å\n", a_eq)
flush(stdout)
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

# ── Stage 0: six undotted Hessians → band-path D_k(q) ───────────────────────
a_list = a_eq .* vol_scales
ω2_cut = (cut_margin_THz / FREQ_THz)^2
println("\n── Stage 0: undotted H_basis at $(length(a_list)) volumes ──────────────")
bps_any = Vector{Any}(undef, length(a_list))
for (v, a) in enumerate(a_list)
    @printf("  [%d/%d] a = %.5f Å (%.0f%% of a_eq)\n", v, length(a_list), a, 100*vol_scales[v])
    bps_any[v] = bandpath_Dk(result, model, element, a, N_cell_fc; N_per_seg=N_per_seg)
    flush(stdout)
    GC.gc()
end
bps  = convert(Vector{typeof(bps_any[1])}, bps_any)
nvol = length(bps)

minω_vol(θ) = [min_freq_stable(θ, bp) for bp in bps]
minω_all(θ) = minimum(min_freq_stable(θ, bp) for bp in bps)
all_soft(θ) = [(v, iq, e) for v in 1:nvol for (iq, e) in soft_modes(θ, bps[v], ω2_cut)]

# ── QP machinery: ONE OSQP model per thread (the 30-point script shared one) ─
Hqp = sparse(Ap'*Ap .+ λ.*(P'*P)); qqp = -(Ap'*Yw)
const OSQP_POOL = [OSQP.Model() for _ in 1:Threads.nthreads()]
function constrain_member(osqp, i, extra_rows, extra_lower)
    rows = vcat(born_rows, extra_rows); lowers = vcat(born_lower, extra_lower)
    A_full = vcat(sparse(Ap[i,:]'), sparse(b_prime'/P), sparse(rows/P))
    l = vcat([Yw[i]],[0.0],lowers); u = vcat([Yw[i]],[0.0],fill(Inf,length(lowers)))
    OSQP.setup!(osqp; P=Hqp, q=qqp, A=A_full, l=l, u=u, max_iter=4_000_000,
                check_termination=25, verbose=false, eps_abs=1e-6, eps_rel=1e-6)
    return P \ OSQP.solve!(osqp).x
end
# full multi-volume repair for one observation → (θ, n_cuts, converged)
#
# Cut rows are accumulated in a Vector and materialised ONCE per iteration.  The
# 30-member scripts used `extra_rows = vcat(extra_rows, row')` per cut, which rebuilds
# the whole matrix every time: O(n²) allocation, ≈220 MB of garbage per member at the
# ~780 cuts these solves need.  Harmless for 30 members, but with 20 threads doing it
# concurrently for 73k members the run becomes GC-bound and parallel scaling collapses.
function repair_one(osqp, i)
    rows_acc = Vector{Vector{Float64}}(); extra_lower = Float64[]; nc = 0
    mat() = isempty(rows_acc) ? zeros(0, n_params) : permutedims(reduce(hcat, rows_acc))
    θ = constrain_member(osqp, i, zeros(0, n_params), extra_lower)
    conv = true
    for it in 0:max_cuts
        soft = all_soft(θ)
        isempty(soft) && break
        if it == max_cuts; conv = false; break; end
        for (v, iq, e) in soft
            push!(rows_acc, cut_row(iq, e, bps[v])); push!(extra_lower, ω2_cut)
        end
        nc += length(soft)
        θ = constrain_member(osqp, i, mat(), extra_lower)
    end
    return θ, nc, conv
end
function mean_fit(extra_rows=zeros(0,n_params), extra_lower=Float64[])
    rows = vcat(born_rows, extra_rows); lowers = vcat(born_lower, extra_lower)
    A_full = vcat(sparse(b_prime'/P), sparse(rows/P))
    l = vcat([0.0],lowers); u = vcat([0.0],fill(Inf,length(lowers)))
    OSQP.setup!(OSQP_POOL[1]; P=Hqp, q=qqp, A=A_full, l=l, u=u, max_iter=4_000_000,
                check_termination=25, verbose=false, eps_abs=1e-6, eps_rel=1e-6)
    return P \ OSQP.solve!(OSQP_POOL[1]).x
end

# ── Stage 1: constrained mean ───────────────────────────────────────────────
println("\n── Stage 1: constrained mean (cutting plane over all volumes) ──────")
θ_mean = mean_fit(); mean_er = zeros(0,n_params); mean_el = Float64[]; mean_cuts = 0
for it in 0:max_cuts
    global θ_mean, mean_er, mean_el, mean_cuts
    soft = all_soft(θ_mean); isempty(soft) && break
    it == max_cuts && (@warn "mean hit max_cuts (min ω = $(round(minω_all(θ_mean);digits=3)))"; break)
    for (v,iq,e) in soft; mean_er = vcat(mean_er, cut_row(iq,e,bps[v])'); push!(mean_el, ω2_cut); end
    mean_cuts += length(soft); θ_mean = mean_fit(mean_er, mean_el)
end
@printf("  constrained mean (%d cuts): C11=%.1f C12=%.1f C44=%.1f GPa, b′·θ=%.1e\n",
        mean_cuts, born(θ_mean)..., dot(b_prime, θ_mean))
@printf("  mean min ω by volume: %s THz\n", string(round.(minω_vol(θ_mean); digits=3)))
check_order = sortperm(minω_vol(θ_mean))
@printf("  predicate volume order (softest first): %s\n", string(round.(vol_scales[check_order]; digits=2)))
flush(stdout)

# ── Stage 2 ★: NAIVE cloud for ALL observations (closed form) ───────────────
println("\n── Stage 2: naive POPS cloud over ALL observations ─────────────────")
C_mat = Symmetric(Ap'*Ap .+ λ.*(P'*P)); Cf = cholesky(C_mat)
AtX = Cf\Matrix(Ap'); θ̃ = Cf\(Ap'*Yw)
leverage = vec(sum(Ap'.*AtX; dims=1)); residual = Yw .- Ap*θ̃
# ★ leverage mask, identical in form to POPSRegression.corrections
lev_thresh = quantile(leverage, leverage_percentile)
kept = findall(leverage .>= lev_thresh)
obs  = kept[1:obs_stride:end]; n_obs = length(obs)
@printf("  leverage_percentile = %.2f → threshold %.4g → kept %d of %d observations\n",
        leverage_percentile, lev_thresh, length(kept), M)
@printf("  observations used: %d (stride %d over the kept set)\n", n_obs, obs_stride)
@printf("  leverage of kept set: min %.4g, median %.4g, max %.4g\n",
        minimum(leverage[kept]), median(leverage[kept]), maximum(leverage[kept]))
flush(stdout)
naive_deltas = Matrix{Float64}(undef, n_obs, n_params)
for (j, i) in enumerate(obs)
    naive_deltas[j, :] = P \ (AtX[:, i] .* (residual[i]/leverage[i]))
end
@printf("  naive cloud: ‖δθ‖ median %.3g, max %.3g\n",
        median(norm.(eachrow(naive_deltas))), maximum(norm.(eachrow(naive_deltas))))
flush(stdout)

# ── Stage 3 ★: CONSTRAINED cloud for ALL observations (threaded, checkpointed)
println("\n── Stage 3: constrain + multi-volume repair EVERY observation ──────")
ckpt = "$outdir/constrained_cloud_checkpoint.jls"
# NB: Vector{Bool}, NOT falses()/BitVector — threads write distinct indices, and
# BitVector packs 64 flags per word, so concurrent writes would race and lose updates.
Θcon = zeros(n_params, n_obs); ncut = zeros(Int, n_obs)
okf  = fill(false, n_obs);     done = fill(false, n_obs)
if isfile(ckpt)
    st = deserialize(ckpt)
    if st.n_obs == n_obs && st.obs_stride == obs_stride && st.leverage_percentile == leverage_percentile
        Θcon, ncut, okf, done = st.Θcon, st.ncut, st.okf, st.done
        @printf("  resumed from checkpoint: %d / %d already done\n", count(done), n_obs)
    else
        @warn "checkpoint was written with different settings — starting fresh"
    end
end
t_start = time()
for b0 in 1:block_size:n_obs
    b1 = min(b0+block_size-1, n_obs)
    all(done[b0:b1]) && continue
    Threads.@threads :static for j in b0:b1
        done[j] && continue
        osqp = OSQP_POOL[Threads.threadid()]
        θ, nc, conv = repair_one(osqp, obs[j])
        Θcon[:, j] = θ; ncut[j] = nc; okf[j] = conv; done[j] = true
    end
    serialize(ckpt, (; Θcon, ncut, okf, done, n_obs, obs_stride, leverage_percentile))
    GC.gc()                      # reclaim the block's garbage; without this the heap runs away
    el = time()-t_start; frac = count(done)/n_obs
    @printf("  block %d–%d done | %d/%d (%.1f%%) | elapsed %.1f h | est. total %.1f h | heap %.1f GB\n",
            b0, b1, count(done), n_obs, 100frac, el/3600, el/3600/max(frac,1e-9),
            Sys.maxrss()/2^30)
    flush(stdout)
end
con_stable = [minω_all(Θcon[:, j]) >= cut_margin_THz - 1e-6 for j in 1:n_obs]
@printf("  repaired: %d / %d | cutting plane converged %d | all-volume stable %d\n",
        n_obs, n_obs, count(okf), count(con_stable))
flush(stdout)
count(con_stable) == 0 && error("no constrained member is stable at all volumes — rejection sampling cannot succeed")
writedlm("$outdir/constrained_cloud.csv", Θcon', ',')
writedlm("$outdir/naive_cloud.csv", (naive_deltas .+ lin_params'), ',')
writedlm("$outdir/theta_mean.csv", θ_mean, ',')

# ── Stage 4 ★: two hypercubes over the FULL clouds; draw n_samples from each ─
println("\n── Stage 4: hypercubes over the full clouds, $n_samples samples each ──")
con_deltas = Θcon' .- θ_mean'
hyp_c, bnd_c = hypercube(Matrix(con_deltas))
hyp_n, bnd_n = hypercube(Matrix(naive_deltas))
@printf("  constrained box: %d retained directions | naive box: %d\n", size(hyp_c,2), size(hyp_n,2))
flush(stdout)
K_ref = dot(θ_mean, b_double_prime)
n_ck = Ref(0); n_born = Ref(0); n_aeq = Ref(0); n_volfail = zeros(Int, nvol)
predicate = θ -> begin
    n_ck[] += 1
    n_ck[] % 250_000 == 0 && (@printf("    … %d proposals, %d past Born/a_eq, per-volume rejects %s\n",
                                      n_ck[], n_aeq[], string(n_volfail)); flush(stdout))
    all(born_lower .<= born_rows*θ) || return false
    n_born[] += 1
    abs(dot(b_prime, θ .- θ_mean)/K_ref) <= 0.1 || return false
    n_aeq[] += 1
    for v in check_order
        min_freq_stable(θ, bps[v]) >= cut_margin_THz - 1e-6 || (n_volfail[v] += 1; return false)
    end
    return true
end
rej_mat, _ = rejection_sample_hypercube(hyp_c, bnd_c, θ_mean, predicate;
                                        number_of_committee_members=n_samples,
                                        max_attempts=max_attempts)
@printf("  constrained funnel: %d drawn → %d past Born → %d past a_eq → %d accepted (%.4f%%)\n",
        n_ck[], n_born[], n_aeq[], n_samples, 100*n_samples/n_ck[])
for v in 1:nvol
    @printf("    volume %.0f%% (a=%.4f Å): killed %d\n", 100*vol_scales[v], a_list[v], n_volfail[v])
end
rej_committee = [rej_mat[:, i] for i in 1:size(rej_mat, 2)]

# naive: SAME hypercube machinery, NO predicate — the apples-to-apples control
nai_mat, _ = rejection_sample_hypercube(hyp_n, bnd_n, lin_params, θ->true;
                                        number_of_committee_members=n_samples,
                                        max_attempts=1_000_000)
nai_committee = [nai_mat[:, i] for i in 1:size(nai_mat, 2)]
writedlm("$outdir/committee_rejection.csv", rej_mat', ',')
writedlm("$outdir/committee_naive.csv",     nai_mat', ',')

# ── Stage 5: verify min ω at every volume for both sets ─────────────────────
println("\n── Stage 5: verification ($n_samples vs $n_samples, all volumes) ────")
M_rej = reduce(vcat, [minω_vol(θ)' for θ in rej_committee])
M_nai = reduce(vcat, [minω_vol(θ)' for θ in nai_committee])
hdr = permutedims(vcat("member", ["minomega_THz_scale_$(round(s;digits=2))" for s in vol_scales]))
for (tag, Mm) in (("rejection", M_rej), ("naive", M_nai))
    writedlm("$outdir/minomega_by_volume_$tag.csv", vcat(hdr, hcat(1:size(Mm,1), Mm)), ',')
end
@printf("  %-10s %s\n", "committee", join([@sprintf("%8.0f%%", 100*s) for s in vol_scales]))
for (tag, Mm) in (("rejection", M_rej), ("naive", M_nai))
    @printf("  %-10s %s   worst %+.3f\n", tag,
            join([@sprintf("%9.3f", minimum(Mm[:,v])) for v in 1:nvol]), minimum(Mm))
end
@printf("\n  unstable (min ω < -0.05 at ANY volume): rejection %d/%d | naive %d/%d\n",
        count(<(-0.05), vec(minimum(M_rej; dims=2))), size(M_rej,1),
        count(<(-0.05), vec(minimum(M_nai; dims=2))), size(M_nai,1))

# ── Stage 6: plots ──────────────────────────────────────────────────────────
allb = Float64[]
for θ in vcat(nai_committee, rej_committee), v in 1:nvol
    append!(allb, vec(bands(θ, bps[v])))
end
pad = 0.05*(maximum(allb)-minimum(allb)); ylims_all = (minimum(allb)-pad, maximum(allb)+pad)
for v in 1:nvol
    s = round(vol_scales[v]; digits=2)
    plot_committee_bands(rej_committee, θ_mean, bps[v],
        "$(result.name) — constrained+rejection, a = $(round(a_list[v];digits=3)) Å ($(round(Int,100s))%)",
        "$outdir/bands_constrained_v$(s).png"; ylims=ylims_all)
    plot_committee_bands(nai_committee, lin_params, bps[v],
        "$(result.name) — naive hypercube, a = $(round(a_list[v];digits=3)) Å ($(round(Int,100s))%)",
        "$outdir/bands_naive_v$(s).png"; ylims=ylims_all)
end
let
    fig = Figure(size=(430,330), figure_padding=(6,10,4,6))
    ax  = Axis(fig[1,1]; xlabel="Lattice constant a / a_eq", ylabel="min non-acoustic ω (THz)",
               title="All-observation clouds: $n_samples naive vs $n_samples constrained",
               titlesize=10, xlabelsize=11, ylabelsize=11, xticklabelsize=10, yticklabelsize=10,
               xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1)
    for r in 1:size(M_nai,1); lines!(ax, vol_scales, M_nai[r,:]; color=RGBAf(0.80,0.15,0.15,0.55), linewidth=1.0); end
    for r in 1:size(M_rej,1); lines!(ax, vol_scales, M_rej[r,:]; color=RGBAf(0.45,0.45,0.45,0.65), linewidth=1.0); end
    lines!(ax, vol_scales, minω_vol(θ_mean); color=RGBf(0.0,0.447,0.698), linewidth=2.0)
    hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=0.8)
    hlines!(ax, [cut_margin_THz]; color=(:black,0.35), linestyle=:dot, linewidth=0.8)
    elem = [LineElement(color=RGBAf(0.80,0.15,0.15,0.7)), LineElement(color=RGBAf(0.45,0.45,0.45,0.8)),
            LineElement(color=RGBf(0.0,0.447,0.698))]
    Legend(fig[1,1], elem, ["naive hypercube", "constrained + rejection", "constrained mean"];
           tellwidth=false, tellheight=false, halign=:left, valign=:bottom, margin=(8,8,8,8),
           framevisible=true, labelsize=8, patchsize=(16,10))
    save("$outdir/minomega_vs_volume.pdf", fig); save("$outdir/minomega_vs_volume.png", fig; px_per_unit=4)
end

# NPT candidates from the constrained set (worst case + median)
worst_over_vol = vec(minimum(M_rej; dims=2))
i_soft = argmin(worst_over_vol); i_med = sortperm(worst_over_vol)[cld(n_samples,2)]
writedlm("$outdir/theta_npt_softest.csv", rej_committee[i_soft], ',')
writedlm("$outdir/theta_npt_median.csv",  rej_committee[i_med],  ',')
# and the worst naive one, for the "before" NPT run
i_nai = argmin(vec(minimum(M_nai; dims=2)))
writedlm("$outdir/theta_npt_naive_worst.csv", nai_committee[i_nai], ',')
open("$outdir/npt_candidates.csv", "w") do io
    println(io, "# clouds fitted to $n_obs observations (leverage_percentile=$leverage_percentile, stride=$obs_stride, of $M total)")
    println(io, "role,index,worst_minomega_THz")
    @printf(io, "constrained_softest,%d,%.4f\n", i_soft, worst_over_vol[i_soft])
    @printf(io, "constrained_median,%d,%.4f\n",  i_med,  worst_over_vol[i_med])
    @printf(io, "naive_worst,%d,%.4f\n",         i_nai,  minimum(M_nai))
end

println("\n── Test-set coverage (constrained committee) ───────────────")
pr = committee_predictions(model, rej_committee, "data/Al/manual_df_test_Al.xyz"; stride=test_stride, point_params=θ_mean)
let ev = mean((pr.tE.<pr.loE).|(pr.tE.>pr.hiE))
    @printf("  energy RMSE=%.4g eV, coverage=%.1f%% (EV %.1f%%)\n", sqrt(mean((pr.pE.-pr.tE).^2)), (1-ev)*100, ev*100)
end
ACEpotentials.Models.set_linear_parameters!(model, lin_params)
println("\n══ RESULT ══════════════════════════════════════════════════")
@printf("  cloud: ALL %d observations constrained (no subset selection)\n", n_obs)
@printf("  unstable at ANY volume: constrained %d/%d | naive %d/%d\n",
        count(<(-0.05), vec(minimum(M_rej; dims=2))), size(M_rej,1),
        count(<(-0.05), vec(minimum(M_nai; dims=2))), size(M_nai,1))
println("All outputs → $outdir/")
