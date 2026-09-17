# make_paired_members.jl
#
# Builds the PAIRED before/after parameter vectors, so that the red and blue series
# differ ONLY by the constraint.
#
# ── THE PROBLEM THIS FIXES ──────────────────────────────────────────────────
# The published figure compares the worst member of the naive forest against the
# softest member of the rejection-sampled constrained committee.  Those are two
# unrelated vectors: ‖θ_blue − θ_red‖ = 9.04.  So the red/blue difference confounds
# "constraining helped" with "different member", and the figure cannot separate them.
#
# ── THE PAIRING EXISTS ALREADY ──────────────────────────────────────────────
# In the committee script, `committee[k]` is the cutting-plane repair of `naive[k]` —
# same observation index, same leverage/residual direction, the QP started from that
# member's own rows.  So committee_repaired.csv row k IS the constrained counterpart of
# naive forest member k, and naive[k] vs repaired[k] isolates the constraint exactly.
#
# ── WHICH k ────────────────────────────────────────────────────────────────
# The published red vector is forest member 15 (verified below: max |Δ| = 6.6e-9 against
# theta_naive_worst.csv, next closest member 0.145 away — unambiguous).  It is also the
# worst member under the multi-volume ranking, −11.4711 THz.  So the red series does not
# change; only blue does, from the rejection sample to naive[15]'s own repair.
#
#   naive[15]     worst over volumes  −11.4711 THz
#   repaired[15]  worst over volumes   +0.1500 THz     ‖Δθ‖ = 3.99
#
# ── WHAT YOU GIVE UP ───────────────────────────────────────────────────────
# Every repaired member terminates exactly at cut_margin_THz = 0.15, i.e. ON the
# constraint boundary — the cutting-plane loop stops as soon as it is satisfied.  So
# repaired[15] is the MINIMALLY constrained version of the naive member, not a typical
# draw from the constrained set.  The rejection-sampled member the figure currently uses
# is a sample from that set (+0.1503 to +0.2033 over the ensemble).
#
# Both comparisons are legitimate and answer different questions:
#   paired   (naive[15] → repaired[15])  does constraining fix THIS member?
#   unpaired (worst naive vs softest rejection)  is the constrained ENSEMBLE's worst case
#            better than the naive ensemble's worst case?
# Use paired when the claim is about the effect of the constraint.
#
# Reads the model matrices directly rather than through load_model — no ACE model is
# needed to rebuild the forest, and it drops the whole thing to ~10 s.
#
# Run:  julia --project npt_trajectories/make_paired_members.jl
#
using DelimitedFiles, LinearAlgebra, Random, Printf, Statistics

ROOT = normpath(joinpath(@__DIR__, ".."))
M   = get(ENV, "MODELDIR", joinpath(ROOT, "models", "Al_12_4_6A_2_"))
RES = "$M/results"
rd(p) = readdlm(p, ',')

@printf("loading matrices …\n"); flush(stdout)
t = @elapsed begin
    A = rd("$M/A.csv"); P = rd("$M/P.csv"); W = vec(rd("$M/W.csv")); Y = vec(rd("$M/Y.csv"))
    lin_params = vec(rd("$M/lin_params.csv"))
end
@printf("  A %s, P %s, %d obs, %d params  [%.1f s]\n", size(A), size(P), length(Y), length(lin_params), t)

Ap = Diagonal(W)*A/P; Yw = W.*Y; λ = 1.0/size(Ap,1)
n_params = length(lin_params)
C = Symmetric(Ap'*Ap .+ λ.*(P'*P)); Cf = cholesky(C)
AtX = Cf\Matrix(Ap'); θ̃ = Cf\(Ap'*Yw)
leverage = vec(sum(Ap'.*AtX; dims=1)); residual = Yw .- Ap*θ̃
forest_member(i) = lin_params .+ (P \ (AtX[:, i] .* (residual[i]/leverage[i])))

n_lev, n_res, n_rand = 5, 10, 15
Random.seed!(1234)
lev_idx = sortperm(leverage; rev=true)[1:n_lev]
res_idx = Int[]; for i in sortperm(abs.(residual); rev=true); i in lev_idx && continue; push!(res_idx,i); length(res_idx)==n_res && break; end
taken = Set(vcat(lev_idx,res_idx)); rand_idx = Int[]
while length(rand_idx) < n_rand; i = rand(1:length(Yw)); (i in taken) && continue; push!(rand_idx,i); push!(taken,i); end
selected = vcat(lev_idx, res_idx, rand_idx)
naive = [forest_member(i) for i in selected]
@printf("\nforest rebuilt: %d members, selected obs %s…\n", length(naive), selected[1:5])

pub = vec(rd("$RES/npt_thermal_expansion_naive_worst_member/theta_naive_worst.csv"))
d   = [maximum(abs.(θ .- pub)) for θ in naive]
k   = argmin(d)
@printf("\npublished θ_naive_worst matches forest member %d, max |Δ| = %.3e\n", k, d[k])
@printf("  (next closest member %d at %.3e — no ambiguity)\n", sortperm(d)[2], sort(d)[2])

mw = rd("$RES/bandpath_undotted_multivolume/minomega_by_volume_naive.csv")[2:end, 2:end]
worst = vec(minimum(mw; dims=2))
@printf("\nmulti-volume ranking: worst naive member is #%d (min ω = %+.4f THz)\n",
        argmin(worst), minimum(worst))
@printf("published member sits at forest index %d, worst-over-volumes %+.4f THz\n", k, worst[k])

rep = rd("$RES/bandpath_undotted_multivolume/committee_repaired.csv")
@printf("\ncommittee_repaired.csv: %s\n", size(rep))
θ_rep_k = collect(Float64, rep[k, :])
mr = rd("$RES/bandpath_undotted_multivolume/minomega_by_volume_repaired.csv")[2:end, 2:end]
@printf("repaired[%d] worst-over-volumes = %+.4f THz\n", k, minimum(mr[k, :]))
@printf("‖θ_repaired[%d] − θ_naive[%d]‖ = %.4f   (max |Δ| = %.4f)\n",
        k, k, norm(θ_rep_k .- naive[k]), maximum(abs.(θ_rep_k .- naive[k])))

pubcon = vec(rd("$RES/npt_multivolume_softest/theta_used.csv"))
@printf("\nfor contrast, the PUBLISHED blue member (rejection sample):\n")
@printf("  ‖θ_published_blue − θ_naive[%d]‖ = %.4f\n", k, norm(pubcon .- naive[k]))
@printf("  ‖θ_published_blue − θ_repaired[%d]‖ = %.4f\n", k, norm(pubcon .- θ_rep_k))

out = "$RES/paired_before_after"; mkpath(out)
writedlm("$out/theta_paired_naive.csv", naive[k], ',')
writedlm("$out/theta_paired_constrained.csv", θ_rep_k, ',')
@printf("\nwrote %s/theta_paired_{naive,constrained}.csv  (forest index %d)\n", out, k)
d[k] < 1e-6 || error("published θ_naive_worst does not match any forest member (best $(d[k])) — " *
                     "the forest has drifted and the pairing cannot be trusted")
println("\nNext:  bash npt_trajectories/run_pipeline.sh paired")
