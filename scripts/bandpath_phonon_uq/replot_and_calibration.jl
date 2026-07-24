# replot_and_calibration.jl
#
# (1) Regenerates the phonon band plots with the Γ-point bug FIXED — the bands
#     are drawn over the FULL band path so the acoustic branches reach 0 at Γ
#     and the high-symmetry ticks line up (the original run stripped the near-Γ
#     q-points, which are only meant to be skipped in the STABILITY check).
# (2) Test-set analysis of the constrained committee: energy & force parity
#     plots with committee error bars, and error-vs-committee-bound calibration
#     histograms (coverage / bias).
#
# Loads the saved committee + cached undotted Hessian — no rebuild.
#
# Run:  julia --project -t 4 scripts/bandpath_phonon_uq/replot_and_calibration.jl

include(joinpath(@__DIR__, "lib.jl"))
using Random

element      = :Al
dataset      = "subset_50_percent"      # model we analysed (has the saved committee)
N_cell       = 3
n_lev, n_res, n_rand = 5, 10, 15
test_xyz     = "data/Al/manual_df_test_Al.xyz"

result = load_model(element, 20, 4, 6, 2; dataset_name=dataset)
model  = result.model; lin_params = result.lin_params
cdir   = "$(result.dir)/results/bandpath_undotted"        # where the committee was saved
outdir = "$(result.dir)/results/bandpath_phonon_uq"; mkpath(outdir)
println("Committee source: $cdir\nOutputs → $outdir")

a_eq = ACEWorkflow.relax_lattice_constant(model, element)
bp   = bandpath_Dk(result, model, element, a_eq, N_cell; N_per_seg=20)
# phonon-repaired constrained mean (saved by the committee script); fall back to committee average
θ_mean_bands = isfile("$cdir/theta_mean.csv") ? vec(readdlm("$cdir/theta_mean.csv", ',')) :
                                                vec(mean(readdlm("$cdir/committee_repaired.csv", ','); dims=1))

# ── regenerate the same naive draw (raw forest members) for comparison ───────
P = result.P; Ap = Diagonal(result.W)*result.A/P; Yw = result.W.*result.Y; λ = 1.0/size(Ap,1)
C = Symmetric(Ap'*Ap .+ λ.*(P'*P)); Cf = cholesky(C); AtX = Cf\Matrix(Ap'); θ̃ = Cf\(Ap'*Yw)
leverage = vec(sum(Ap'.*AtX; dims=1)); residual = Yw .- Ap*θ̃
forest_member(i) = lin_params .+ (P \ (AtX[:, i] .* (residual[i]/leverage[i])))
Random.seed!(1234)
lev_idx = sortperm(leverage; rev=true)[1:n_lev]
res_idx = Int[]; for i in sortperm(abs.(residual); rev=true); i in lev_idx && continue; push!(res_idx,i); length(res_idx)==n_res && break; end
taken = Set(vcat(lev_idx,res_idx)); rand_idx = Int[]
while length(rand_idx) < n_rand; i = rand(1:length(Yw)); (i in taken) && continue; push!(rand_idx,i); push!(taken,i); end
naive = [forest_member(i) for i in vcat(lev_idx, res_idx, rand_idx)]

# ── (1) FIXED band plots ─────────────────────────────────────────────────────
rej = [readdlm("$cdir/committee_rejection.csv", ',')[i,:] for i in 1:30]
println("\nRegenerating band plots (Γ fixed) …")
plot_committee_bands(rej,  θ_mean_bands, bp, "$(result.name) — constrained committee (band-path, a_eq-fixed)", "$outdir/bands_constrained.png")
plot_committee_bands(naive, lin_params,  bp, "$(result.name) — naive POPS committee",                        "$outdir/bands_naive.png")
@printf("  constrained: min band ω = %+.3f THz (%d/%d unstable)\n",
        minimum(min_freq_stable(θ,bp) for θ in rej), count(θ->min_freq_stable(θ,bp)<-0.05, rej), length(rej))
@printf("  naive      : min band ω = %+.3f THz (%d/%d unstable)\n",
        minimum(min_freq_stable(θ,bp) for θ in naive), count(θ->min_freq_stable(θ,bp)<-0.05, naive), length(naive))

# ── (2) test-set parity + calibration for the constrained committee ──────────
println("\nTest-set predictions (constrained committee) …")
pr = committee_predictions(model, rej, test_xyz; stride=10, point_params=θ_mean_bands)
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
