# parity_calibration_pinned_Al_20_4_6A_3.jl
#
# Test-set parity and calibration for the two Al_20_4_6A_3 committees produced by
# scripts/uq/pinned_hypercube_rejection_Al_20_4_6A_3.jl — naive and rejection-sampled —
# at the same paper quality as the Al_12 figures.
#
# The plotting machinery is NOT duplicated here: it lives in lib_parity_calibration.jl,
# extracted from hypercube_full_cloud_bands_Al_12_4_6A_2.jl, which now includes the same
# file.  One copy, so the Al_12 and Al_20 figures cannot drift apart in styling.
#
# ── WHAT THE COMPARISON IS ──────────────────────────────────────────────────
# Both committees are drawn from the SAME hypercube with the same seed; the rejected one
# additionally required b″·θ > 0 and min ω ≥ tol of every member.  So any difference in
# parity or calibration is the price of the physics constraint, not sampling noise:
#
#   RMSE       should barely move — the point prediction is the same mean model in both.
#   COVERAGE   is the number to watch.  Rejection removes members, so the ensemble is
#              narrower and coverage can only fall.  How far it falls is how much
#              predictive spread the constraint costs.
#
# Coverage here is committee min/max envelope containment, not a calibrated interval, so
# read it as a relative measure between the two columns rather than an absolute.
#
# ── FAST MODEL LOAD ─────────────────────────────────────────────────────────
# load_model() reads the 5.2 GB A.csv; nothing here needs it — committee_predictions
# only touches the model object.  We build the model from the JSON plus lin_params.csv;
# set_linear_parameters! throws on a length mismatch, which is the check.  ARGS[2]="full"
# forces the real loader.
#
# Run:  julia --project -t 8 scripts/uq/parity_calibration_pinned_Al_20_4_6A_3.jl [stride] [full]
#   stride 20 = every 20th test configuration (default).  Use 1 for the whole test set.

include(joinpath(@__DIR__, "..", "bandpath_phonon_uq", "lib.jl"))
include(joinpath(@__DIR__, "lib_parity_calibration.jl"))

element  = :Al
stride   = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20
use_full = length(ARGS) >= 2 && ARGS[2] == "full"
FIGW     = parse(Float64, get(ENV, "FIGW", "540"))

MODELDIR = abspath(joinpath(@__DIR__, "..", "..", "models", "Al_20_4_6A_3_"))
SRC      = "$MODELDIR/results/pinned_hypercube_rejection"
outdir   = "$SRC/parity_calibration"; mkpath(outdir)
TEST     = abspath(joinpath(@__DIR__, "..", "..", "data", "Al", "manual_df_test_Al.xyz"))

if use_full
    result     = load_model(element, 20, 4, 6, 3; dataset_name="")
    model      = result.model
    lin_params = result.lin_params
else
    model, _   = ACEpotentials.load_model("$MODELDIR/Al_20_4_6A_3.json")
    lin_params = vec(readdlm("$MODELDIR/lin_params.csv", ','))
    ACEpotentials.Models.set_linear_parameters!(model, lin_params)
    @printf("fast load: %d parameters (skipped A.csv)\n", length(lin_params))
end
isfile(TEST) || error("test set not found: $TEST")
flush(stdout)

# ── the two committees ──────────────────────────────────────────────────────
# Rows are members (the producing script writes samples' with writedlm), so each row is
# a full coefficient vector θ = lin_params + δ, not a delta.
function read_committee(path, n_params)
    isfile(path) || error("""
        missing $path
        Run scripts/uq/pinned_hypercube_rejection_Al_20_4_6A_3.jl first — this script
        plots the committees it writes.""")
    M = readdlm(path, ',')
    size(M, 2) == n_params ||
        error("$path is $(size(M,2)) wide, model has $n_params parameters")
    return [collect(Float64, M[i, :]) for i in 1:size(M, 1)]
end

n_params = length(lin_params)
committees = [
    ("naive",    "naive hypercube — no rejection",                    read_committee("$SRC/samples_naive.csv",    n_params)),
    ("rejected", "rejection: b″·θ > 0 and min ω ≥ tol",               read_committee("$SRC/samples_rejected.csv", n_params)),
]
for (tag, _, mem) in committees
    @printf("%-9s committee: %d members × %d params\n", tag, length(mem), n_params)
end
flush(stdout)

# ── predictions ─────────────────────────────────────────────────────────────
# point_params = lin_params for BOTH: the committees are lin_params + δ, so the mean
# model is the common point prediction and the deviation histograms are comparable.
@printf("\n── test-set predictions (stride %d) ──\n", stride); flush(stdout)
preds = Pair{String,Any}[]
for (tag, ttl, mem) in committees
    t = @elapsed pr = committee_predictions(model, mem, TEST; stride=stride,
                                            point_params=lin_params, per_atom=true)
    @printf("  %-9s %d configs, %d force components  [%.1f s]\n",
            tag, pr.n, length(pr.tF), t)
    push!(preds, tag => (ttl=ttl, pr=pr))
    flush(stdout)
end

# ── figures ─────────────────────────────────────────────────────────────────
println()
for (tag, e) in preds
    parity_figure(e.pr, e.ttl, tag, outdir; FIGW=FIGW)
    calibration_figure(e.pr, e.ttl, tag, outdir; FIGW=FIGW)
end

# ── summary ─────────────────────────────────────────────────────────────────
open("$outdir/parity_calibration_summary.csv", "w") do io
    parity_calibration_header(io)
    for (tag, e) in preds; parity_calibration_row(io, tag, e.pr); end
end
println("\n══ SUMMARY ═══════════════════════════════════════════════════════")
@printf("%-10s %14s %14s %10s %10s\n", "committee", "E RMSE meV/at", "F RMSE eV/Å", "E cov %", "F cov %")
for (tag, e) in preds
    @printf("%-10s %14.4g %14.4g %10.1f %10.1f\n", tag,
            1000*pc_rmse(e.pr.pE, e.pr.tE), pc_rmse(e.pr.pF, e.pr.tF),
            pc_cover(e.pr.tE, e.pr.loE, e.pr.hiE), pc_cover(e.pr.tF, e.pr.loF, e.pr.hiF))
end
let n = preds[1].second.pr, r = preds[2].second.pr
    @printf("\ncoverage change from rejection: energy %+.1f pp, force %+.1f pp\n",
            pc_cover(r.tE, r.loE, r.hiE) - pc_cover(n.tE, n.loE, n.hiE),
            pc_cover(r.tF, r.loF, r.hiF) - pc_cover(n.tF, n.loF, n.hiF))
    println("  (negative is expected — rejection removes members, narrowing the envelope;")
    println("   the magnitude is what the physics constraint costs in predictive spread)")
end
serialize("$outdir/parity_calibration_predictions.jls",
          (; preds = Dict(tag => e.pr for (tag, e) in preds), stride, n_params))
println("\nsummary → $outdir/parity_calibration_summary.csv")
println("data    → $outdir/parity_calibration_predictions.jls")
