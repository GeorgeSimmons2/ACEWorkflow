# bandpath_committee_undotted_Al_12_4_6A_2_multivolume.jl
#
# AMENDED COPY of bandpath_committee_undotted_Al_12_4_6A_2_ncell4_densek.jl
# (both legacy scripts left untouched).  ONE change, everywhere it propagates:
#
#   ── enforce dynamical stability at SIX VOLUMES, not one ──
#     a ∈ a_eq · {1.00, 1.02, 1.04, 1.06, 1.08, 1.10}
#
# WHY.  The ncell4_densek committee is stable at a_eq (all 60 members min ω ∈
# [+0.278, +0.528] THz) yet the softest member left FCC within 7 ps of NPT at 300 K:
# coordination 12 → 9.5, NN 2.86 → 2.58 Å, ΔE ≈ −0.18 eV/atom, and its phonons at the
# thermally-expanded a(T) were −7.4 THz.  A curvature constraint at ONE geometry says
# nothing about the stretched lattice the barostat actually visits.  Constraining the
# whole a_eq → 1.1·a_eq path is the direct fix: every member must be dynamically stable
# over the entire volume range that thermal expansion can reach.
#
# WHAT CHANGES vs the single-volume script:
#   • Stage 0 builds SIX undotted H_basis / band paths (one per volume) instead of one.
#     H_k = ∂²B_k/∂r² is geometry-dependent, so each volume needs its own build
#     (~1.2 min each at 4×4×4/256 atoms; a_eq is already cached from the ncell4 run).
#   • Stage 1/3 cutting plane: each iteration collects soft modes from ALL volumes and
#     appends a cut row per (volume, q, branch).  The QP grows faster — expect ~6× the
#     rows of the single-volume run.
#   • Stage 4 rejection predicate: a draw is accepted only if it is stable at EVERY
#     volume.  Volumes are tested softest-first (measured on the constrained mean) so
#     the majority of rejects die on the first, cheapest test.
#
# WHAT DOES NOT CHANGE (deliberately):
#   • a_eq is still THE reference geometry: b′(a_eq)·θ = 0 still pins each member's
#     equilibrium there, and the Born rows are still evaluated at a_eq.  The extra
#     volumes only add "…and do not go soft when stretched" — they do not move a_eq.
#   • Sampler is plain unclipped-hypercube rejection (no Gaussian proposal).
#   • N_cell_fc = 4 and the dense Γ→K path are inherited from the ncell4_densek run.
#
# OUTPUTS → results/bandpath_undotted_multivolume/ (does not clobber earlier runs):
#   committee_{repaired,rejection}.csv, theta_mean.csv          — the committee itself
#   minomega_by_volume_{naive,repaired,rejection}.csv           — member × volume min ω
#   minomega_vs_volume.pdf/png                                  — THE money plot
#   bands_{naive,constrained}_v{scale}.pdf/png                  — shared y-axis across all
#   npt_candidates.csv + theta_npt_{softest,median}.csv         — feed the NPT follow-up
#
# Run:  sbatch scripts/uq/run_committee_multivolume.slurm       (hmem; 6 H_basis builds)

include(joinpath(@__DIR__, "..", "bandpath_phonon_uq", "lib.jl"))
using SparseArrays, OSQP, Random
Random.seed!(1234)

element        = :Al
dataset        = ""            # "" → full-dataset model (Al_12_4_6A_2_)
a_experimental = nothing       # Float (Å) to pin a_eq experimentally; nothing → mean model
N_cell_fc      = 4                         # converged for the 6 Å cutoff (3×3×3 was not)
N_per_seg      = [20, 20, 20, 20, 60]      # dense Γ→K (last segment)
cut_margin_THz = 0.15
n_lev, n_res, n_rand = 5, 10, 15
max_cuts       = 40
test_stride    = 10
vol_scales     = collect(1.00:0.02:1.10)   # a_eq → 1.1·a_eq in 2% steps → 6 Hessians
max_attempts   = 5_000_000                 # tighter feasible set than the 1-volume run

result     = load_model(element, 12, 4, 6, 2; dataset_name=dataset)
model      = result.model; lin_params = result.lin_params; n_params = length(lin_params)
P = result.P; Ap = Diagonal(result.W)*result.A/P; Yw = result.W.*result.Y; λ = 1.0/size(Ap,1)
outdir = "$(result.dir)/results/bandpath_undotted_multivolume"; mkpath(outdir)
@printf("Model %s: %d params, %d threads.  Outputs → %s\n", result.name, n_params, Threads.nthreads(), outdir)

# ── reference geometry, Born + a_eq rows (ALL at a_eq — unchanged) ───────────
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

# ── Stage 0: SIX undotted Hessians → band-path D_k(q) at each volume ─────────
a_list = a_eq .* vol_scales
ω2_cut = (cut_margin_THz / FREQ_THz)^2
println("\n── Stage 0: undotted H_basis at $(length(a_list)) volumes ──────────────")
@printf("  a_eq = %.5f Å → scales %s → a = %s\n", a_eq,
        string(round.(vol_scales; digits=2)), string(round.(a_list; digits=4)))
bps_any = Vector{Any}(undef, length(a_list))
for (v, a) in enumerate(a_list)
    @printf("  [%d/%d] a = %.5f Å (%.0f%% of a_eq)\n", v, length(a_list), a, 100*vol_scales[v])
    bps_any[v] = bandpath_Dk(result, model, element, a, N_cell_fc; N_per_seg=N_per_seg)
    GC.gc()                                   # drop the 429 MB H_basis before the next build
end
# narrow to a concrete element type: the predicate indexes this millions of times
bps  = convert(Vector{typeof(bps_any[1])}, bps_any)
nvol = length(bps)
for v in 1:nvol
    fm = bands(lin_params, bps[v])
    @printf("  mean model @ %.0f%%: ω ∈ [%+.3f, %+.3f] THz  (min non-acoustic %+.3f)\n",
            100*vol_scales[v], minimum(fm), maximum(fm), min_freq_stable(lin_params, bps[v]))
end

# stability across ALL volumes — the quantity this whole script is about
minω_vol(θ)  = [min_freq_stable(θ, bp) for bp in bps]
minω_all(θ)  = minimum(min_freq_stable(θ, bp) for bp in bps)
# soft modes over all volumes, tagged with which volume they came from
all_soft(θ)  = [(v, iq, e) for v in 1:nvol for (iq, e) in soft_modes(θ, bps[v], ω2_cut)]

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

# the MEAN must obey the same multi-volume phonon constraints as the committee
println("\n── Stage 1: constrained mean (cutting plane over all volumes) ──────")
θ_mean = mean_fit(); mean_er = zeros(0,n_params); mean_el = Float64[]; mean_cuts = 0
for it in 0:max_cuts
    global θ_mean, mean_er, mean_el, mean_cuts
    soft = all_soft(θ_mean); isempty(soft) && break
    it == max_cuts && (@warn "mean hit max_cuts (min ω over volumes = $(round(minω_all(θ_mean);digits=3)))"; break)
    for (v, iq, e) in soft; mean_er = vcat(mean_er, cut_row(iq, e, bps[v])'); push!(mean_el, ω2_cut); end
    mean_cuts += length(soft); θ_mean = mean_fit(mean_er, mean_el)
end
minω_all(θ_mean) >= cut_margin_THz - 1e-6 || @warn "mean model still soft after $mean_cuts cuts"
@printf("  constrained mean (%d cuts): C11=%.1f C12=%.1f C44=%.1f GPa, b′·θ=%.1e\n",
        mean_cuts, born(θ_mean)..., dot(b_prime, θ_mean))
@printf("  mean min ω by volume: %s THz\n", string(round.(minω_vol(θ_mean); digits=3)))

# test the volumes softest-first in the predicate → most rejects die on the first test
check_order = sortperm(minω_vol(θ_mean))
@printf("  predicate volume order (softest first): %s\n", string(round.(vol_scales[check_order]; digits=2)))

lev_idx = sortperm(leverage; rev=true)[1:n_lev]
res_idx = Int[]; for i in sortperm(abs.(residual); rev=true); i in lev_idx && continue; push!(res_idx,i); length(res_idx)==n_res && break; end
taken = Set(vcat(lev_idx,res_idx)); rand_idx = Int[]
while length(rand_idx) < n_rand; i = rand(1:length(Yw)); (i in taken) && continue; push!(rand_idx,i); push!(taken,i); end
selected = vcat(lev_idx, res_idx, rand_idx)

# ── Stages 2+3: a_eq-constrain each member, multi-volume phonon repair ───────
println("\n── Stages 2+3: constrain + repair the proposal cloud (all volumes) ──")
committee = Vector{Vector{Float64}}(undef, length(selected)); n_cuts = zeros(Int, length(selected))
naive = [forest_member(i) for i in selected]
t_rep = @elapsed for (k, i) in enumerate(selected)
    extra_rows = zeros(0, n_params); extra_lower = Float64[]
    θ, _ = constrain_member(i, extra_rows, extra_lower)
    for it in 0:max_cuts
        soft = all_soft(θ)
        isempty(soft) && break
        it == max_cuts && (@warn "obs $i hit max_cuts (min ω over volumes = $(round(minω_all(θ);digits=3)))"; break)
        for (v, iq, e) in soft; extra_rows = vcat(extra_rows, cut_row(iq, e, bps[v])'); push!(extra_lower, ω2_cut); end
        n_cuts[k] += length(soft)
        θ, _ = constrain_member(i, extra_rows, extra_lower)
    end
    committee[k] = θ
end
@printf("  repaired (≥1 cut): %d / %d   |  cut rows added: median %d, max %d  [%.1f min]\n",
        count(>(0), n_cuts), length(selected), round(Int, median(n_cuts)), maximum(n_cuts), t_rep/60)

# pre-flight: is the box centre region actually feasible?  If the repaired members
# themselves are not all-volume stable, rejection sampling is hopeless — say so now.
rep_ok = count(θ -> minω_all(θ) >= cut_margin_THz - 1e-6, committee)
@printf("  repaired members stable at ALL volumes: %d / %d\n", rep_ok, length(committee))
rep_ok == 0 && error("no repaired member is stable across all volumes — rejection sampling cannot succeed; raise max_cuts or widen cut_margin_THz")

# ── Stage 4: rejection sample; predicate = stable at EVERY volume ────────────
println("\n── Stage 4: rejection sample (predicate = stable at all $nvol volumes) ──")
con_deltas = reduce(hcat, committee)' .- θ_mean'
hyp_eig, hyp_bound = hypercube(Matrix(con_deltas)); K_ref = dot(θ_mean, b_double_prime)
n_ck = Ref(0); n_born = Ref(0); n_aeq = Ref(0); n_volfail = zeros(Int, nvol)
predicate = θ -> begin
    n_ck[] += 1
    n_ck[] % 250_000 == 0 && @printf("    … %d proposals, %d past Born/a_eq, per-volume rejects %s\n",
                                     n_ck[], n_aeq[], string(n_volfail))
    all(born_lower .<= born_rows*θ) || return false
    n_born[] += 1
    abs(dot(b_prime, θ .- θ_mean)/K_ref) <= 0.1 || return false
    n_aeq[] += 1
    for v in check_order                       # softest volume first → cheapest rejection
        min_freq_stable(θ, bps[v]) >= cut_margin_THz - 1e-6 || (n_volfail[v] += 1; return false)
    end
    return true
end
rej_mat, _ = rejection_sample_hypercube(hyp_eig, hyp_bound, θ_mean, predicate;
                                        number_of_committee_members=length(selected),
                                        max_attempts=max_attempts)
@printf("  funnel: %d drawn → %d past Born → %d past a_eq band → %d accepted (%.3f%%)\n",
        n_ck[], n_born[], n_aeq[], length(selected), 100*length(selected)/n_ck[])
for v in 1:nvol
    @printf("    volume %.0f%% (a=%.4f Å): killed %d proposals\n", 100*vol_scales[v], a_list[v], n_volfail[v])
end
rej_committee = [rej_mat[:, i] for i in 1:size(rej_mat, 2)]
writedlm("$outdir/committee_rejection.csv", rej_mat', ',')
writedlm("$outdir/committee_repaired.csv", reduce(hcat, committee)', ',')
writedlm("$outdir/theta_mean.csv", θ_mean, ',')

# ── Stage 5: verify — min ω for every member at every volume ────────────────
println("\n── Stage 5: verification (min ω at each volume) ────────────────────")
M_naive = reduce(vcat, [minω_vol(θ)' for θ in naive])
M_rep   = reduce(vcat, [minω_vol(θ)' for θ in committee])
M_rej   = reduce(vcat, [minω_vol(θ)' for θ in rej_committee])
hdr = permutedims(vcat("member", ["minomega_THz_scale_$(round(s;digits=2))" for s in vol_scales]))
for (tag, M) in (("naive", M_naive), ("repaired", M_rep), ("rejection", M_rej))
    writedlm("$outdir/minomega_by_volume_$tag.csv", vcat(hdr, hcat(1:size(M,1), M)), ',')
end
@printf("  %-10s %s\n", "committee", join([@sprintf("%8.0f%%", 100*s) for s in vol_scales]))
for (tag, M) in (("naive", M_naive), ("repaired", M_rep), ("rejection", M_rej))
    @printf("  %-10s %s   worst-over-volumes %+.3f\n", tag,
            join([@sprintf("%9.3f", minimum(M[:,v])) for v in 1:nvol]), minimum(M))
end
@printf("\n  unstable (min ω < -0.05 at ANY volume):  naive %d/%d | repaired %d/%d | rejection %d/%d\n",
        count(<(-0.05), vec(minimum(M_naive; dims=2))), size(M_naive,1),
        count(<(-0.05), vec(minimum(M_rep;   dims=2))), size(M_rep,1),
        count(<(-0.05), vec(minimum(M_rej;   dims=2))), size(M_rej,1))

# ── Stage 6: plots ──────────────────────────────────────────────────────────
# shared y-limits across EVERY band panel so all 12 line up
allb = Float64[]
for θ in vcat(naive, rej_committee), v in 1:nvol
    append!(allb, vec(bands(θ, bps[v])))
end
ylo_all = minimum(allb); yhi_all = maximum(allb)
pad = 0.05*(yhi_all - ylo_all); ylims_all = (ylo_all - pad, yhi_all + pad)
for v in 1:nvol
    s = round(vol_scales[v]; digits=2)
    plot_committee_bands(rej_committee, θ_mean, bps[v],
        "$(result.name) — constrained, a = $(round(a_list[v];digits=3)) Å ($(round(Int,100s))%)",
        "$outdir/bands_constrained_v$(s).png"; ylims=ylims_all)
    plot_committee_bands(naive, lin_params, bps[v],
        "$(result.name) — naive POPS, a = $(round(a_list[v];digits=3)) Å ($(round(Int,100s))%)",
        "$outdir/bands_naive_v$(s).png"; ylims=ylims_all)
end

# THE money plot: min ω vs volume, naive vs constrained
let
    fig = Figure(size=(430,330), figure_padding=(6,10,4,6))
    ax  = Axis(fig[1,1]; xlabel="Lattice constant a / a_eq", ylabel="min non-acoustic ω (THz)",
               title="Dynamical stability across the thermal-expansion range",
               titlesize=10, xlabelsize=11, ylabelsize=11, xticklabelsize=10, yticklabelsize=10,
               xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1)
    for r in 1:size(M_naive,1); lines!(ax, vol_scales, M_naive[r,:]; color=RGBAf(0.80,0.15,0.15,0.35), linewidth=0.8); end
    for r in 1:size(M_rej,1);   lines!(ax, vol_scales, M_rej[r,:];   color=RGBAf(0.45,0.45,0.45,0.40), linewidth=0.8); end
    lines!(ax, vol_scales, minω_vol(θ_mean); color=RGBf(0.0,0.447,0.698), linewidth=2.0)
    hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=0.8)
    hlines!(ax, [cut_margin_THz]; color=(:black,0.35), linestyle=:dot, linewidth=0.8)
    elem = [LineElement(color=RGBAf(0.80,0.15,0.15,0.6)), LineElement(color=RGBAf(0.45,0.45,0.45,0.7)),
            LineElement(color=RGBf(0.0,0.447,0.698))]
    Legend(fig[1,1], elem, ["naive POPS", "constrained (rejection)", "constrained mean"];
           tellwidth=false, tellheight=false, halign=:left, valign=:bottom, margin=(8,8,8,8),
           framevisible=true, labelsize=8, patchsize=(16,10))
    save("$outdir/minomega_vs_volume.pdf", fig); save("$outdir/minomega_vs_volume.png", fig; px_per_unit=4)
end

# ── NPT candidates: worst case + typical case, ranked by worst-over-volumes ──
worst_over_vol = vec(minimum(M_rej; dims=2))
i_soft = argmin(worst_over_vol)
i_med  = sortperm(worst_over_vol)[cld(length(worst_over_vol), 2)]
writedlm("$outdir/theta_npt_softest.csv", rej_committee[i_soft], ',')
writedlm("$outdir/theta_npt_median.csv",  rej_committee[i_med],  ',')
open("$outdir/npt_candidates.csv", "w") do io
    println(io, "# rejection-committee members ranked by worst min ω over volumes $(round.(vol_scales;digits=2))")
    println(io, "role,rejection_index,worst_minomega_THz,minomega_at_each_volume")
    @printf(io, "softest,%d,%.4f,%s\n", i_soft, worst_over_vol[i_soft], join(round.(M_rej[i_soft,:];digits=4), " "))
    @printf(io, "median,%d,%.4f,%s\n",  i_med,  worst_over_vol[i_med],  join(round.(M_rej[i_med,:];digits=4),  " "))
end
@printf("\n  NPT candidates → softest = rejection[%d] (worst %+.3f THz), median = rejection[%d] (worst %+.3f THz)\n",
        i_soft, worst_over_vol[i_soft], i_med, worst_over_vol[i_med])

# ── Test-set coverage (unchanged) ───────────────────────────────────────────
println("\n── Test-set coverage (constrained committee) ───────────────")
pr = committee_predictions(model, rej_committee, "data/Al/manual_df_test_Al.xyz"; stride=test_stride, point_params=θ_mean)
let ev = mean((pr.tE.<pr.loE).|(pr.tE.>pr.hiE))
    @printf("  energy RMSE=%.4g eV, coverage=%.1f%% (EV %.1f%%)\n", sqrt(mean((pr.pE.-pr.tE).^2)), (1-ev)*100, ev*100)
end
ACEpotentials.Models.set_linear_parameters!(model, lin_params)
println("\n══ RESULT ══════════════════════════════════════════════════")
@printf("  volumes constrained: %s × a_eq (%.4f–%.4f Å)\n", string(round.(vol_scales;digits=2)), first(a_list), last(a_list))
@printf("  unstable at ANY volume: constrained %d/%d | naive %d/%d\n",
        count(<(-0.05), vec(minimum(M_rej; dims=2))), size(M_rej,1),
        count(<(-0.05), vec(minimum(M_naive; dims=2))), size(M_naive,1))
@printf("  constrained worst-over-volumes min ω = %+.3f THz  (single-volume run: +0.278 THz at a_eq, but −7.4 THz at a(300 K))\n",
        minimum(M_rej))
println("All outputs → $outdir/")
