# parity_calibration_Al_16_4_6A_3.jl
#
# STAGE 3 of three.  Test-set parity and calibration for the two ensembles from
# scripts/uq/pinned_rejection_ensembles_Al_16_4_6A_3.jl — independent figures per
# ensemble, four in total.
#
# Plotting comes from scripts/uq/lib_parity_calibration.jl, shared with the Al_12 and
# Al_20 figures, so none of them can drift apart in styling.  MLST sizing throughout:
# built at final display width with 13/12/11 pt text, so Overleaf does no rescaling.
#
# ── ENERGIES ARE PER ATOM ───────────────────────────────────────────────────
# `per_atom=true`, so energies are eV/atom and the RMSE is quoted in meV/atom.  This is
# not cosmetic: the Al test set spans 2 to 25 atoms per configuration, with 2-atom
# frames the most common, so a TOTAL-energy RMSE is dominated by the largest cells and
# the parity spread is partly just cell-size spread.  Forces are already intensive and
# are never rescaled.  The July figures in results/bandpath_undotted/ were total-energy
# and are not comparable to these.
#
# ── WHAT TO READ ────────────────────────────────────────────────────────────
# Both ensembles are lin_params + δ from the SAME box with the same seed, differing only
# in the phonon predicate, so the point prediction is identical and the two are paired.
#   RMSE      should barely move — same mean model on both sides.
#   COVERAGE  is the number.  Rejection removes members, so the envelope narrows and
#             coverage can only fall; how far it falls is what the physics costs.
# Coverage is committee min/max envelope containment, NOT a calibrated interval — read
# it as a relative measure between the two columns.
#
# ── FAST MODEL LOAD ─────────────────────────────────────────────────────────
# committee_predictions only touches the model object, so the 1.9 GB A.csv is skipped.
# Pass "full" as the second argument to force the real loader.
#
# Run:  julia --project -t 8 scripts/uq/parity_calibration_Al_16_4_6A_3.jl [stride] [full]
#   stride 20 = every 20th test configuration (default).  1 for the whole set.
#   FIGW=540

include(joinpath(@__DIR__, "..", "bandpath_phonon_uq", "lib.jl"))
include(joinpath(@__DIR__, "lib_parity_calibration.jl"))

element  = :Al
stride   = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20
use_full = length(ARGS) >= 2 && ARGS[2] == "full"
FIGW     = parse(Float64, get(ENV, "FIGW", "540"))

MODELDIR = "models/Al_16_4_6A_3_"
SRC      = get(ENV, "SRC", "$MODELDIR/results/pinned_ensembles")
outdir   = get(ENV, "OUTDIR", "$SRC/parity_calibration"); mkpath(outdir)
TEST     = abspath(joinpath(@__DIR__, "..", "..", "data", "Al", "manual_df_test_Al.xyz"))

if use_full
    result = load_model(element, 16, 4, 6, 3; dataset_name="")
    model, lin_params = result.model, result.lin_params
else
    model, _ = ACEpotentials.load_model("$MODELDIR/Al_16_4_6A_3.json")
    lin_params = vec(readdlm("$MODELDIR/lin_params.csv", ','))
    ACEpotentials.Models.set_linear_parameters!(model, lin_params)
    @printf("fast load: %d parameters (A.csv skipped)\n", length(lin_params))
end
isfile(TEST) || error("test set not found: $TEST")
n_params = length(lin_params)

function read_ens(tag)
    f = "$SRC/ensemble_$(tag).csv"
    isfile(f) || error("""
        missing $f
        Run scripts/uq/pinned_rejection_ensembles_Al_16_4_6A_3.jl first.""")
    M = readdlm(f, ',')
    size(M, 2) == n_params || error("$f is $(size(M,2)) wide, model has $n_params")
    return [collect(Float64, M[i, :]) for i in 1:size(M, 1)]
end

ensembles = [("unconstrained", "unconstrained — pinned, no predicate",  read_ens("unconstrained")),
             ("constrained",   "constrained — phonon-positive",          read_ens("constrained"))]
for (tag, _, mem) in ensembles
    @printf("%-14s %d members × %d params\n", tag, length(mem), n_params)
end
flush(stdout)

# ── predictions ─────────────────────────────────────────────────────────────
# point_params = lin_params for BOTH: the ensembles are lin_params + δ, so the mean
# model is the common point prediction and the deviation histograms are comparable.
@printf("\n── test-set predictions (stride %d, per atom) ──\n", stride); flush(stdout)
preds = Pair{String,Any}[]
for (tag, ttl, mem) in ensembles
    t = @elapsed pr = committee_predictions(model, mem, TEST; stride=stride,
                                            point_params=lin_params, per_atom=true)
    @printf("  %-14s %d configs, %d force components  [%.1f s]\n",
            tag, pr.n, length(pr.tF), t)
    push!(preds, tag => (ttl=ttl, pr=pr)); flush(stdout)
end

# ── four independent figures ────────────────────────────────────────────────
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
println("\n══ SUMMARY (energies per atom) ═══════════════════════════════════")
@printf("%-14s %14s %14s %10s %10s\n",
        "ensemble", "E RMSE meV/at", "F RMSE eV/Å", "E cov %", "F cov %")
for (tag, e) in preds
    @printf("%-14s %14.4g %14.4g %10.1f %10.1f\n", tag,
            1000*pc_rmse(e.pr.pE, e.pr.tE), pc_rmse(e.pr.pF, e.pr.tF),
            pc_cover(e.pr.tE, e.pr.loE, e.pr.hiE), pc_cover(e.pr.tF, e.pr.loF, e.pr.hiF))
end
let u = preds[1].second.pr, c = preds[2].second.pr
    @printf("\ncoverage change from the phonon predicate: energy %+.1f pp, force %+.1f pp\n",
            pc_cover(c.tE, c.loE, c.hiE) - pc_cover(u.tE, u.loE, u.hiE),
            pc_cover(c.tF, c.loF, c.hiF) - pc_cover(u.tF, u.loF, u.hiF))
    @printf("RMSE change: energy %+.4g meV/atom, force %+.4g eV/Å\n",
            1000*(pc_rmse(c.pE, c.tE) - pc_rmse(u.pE, u.tE)),
            pc_rmse(c.pF, c.tF) - pc_rmse(u.pF, u.tF))
    println("  (negative coverage change is expected — rejection removes members and")
    println("   narrows the envelope; RMSE should barely move, same mean model both sides)")
end
serialize("$outdir/parity_calibration_predictions.jls",
          (; preds = Dict(tag => e.pr for (tag, e) in preds), stride, n_params,
             per_atom = true))
println("\nfigures → $outdir/{parity,calibration}_{unconstrained,constrained}.{pdf,png}")
println("summary → $outdir/parity_calibration_summary.csv")
