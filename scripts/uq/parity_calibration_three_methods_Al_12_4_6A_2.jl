# parity_calibration_three_methods_Al_12_4_6A_2.jl
#
# Test-set parity and calibration for the THREE committees in results/aeq_cheap_vs_expensive,
# in the same format as results/bandpath_phonon_uq (same lib functions, same styling).
#
#   A  committee_A_cheap_phononreject.csv   cheap a_eq-only QP over the masked cloud,
#                                           phonon positivity enforced by REJECTION
#   B  committee_B_naive.csv                naive POPS hypercube, no constraints
#   C  committee_C_expensive_phononreject.csv  top-30 leverage, FULL multi-volume
#                                           cutting-plane repair, then rejection
#
# The point of the comparison: A skips the ~800-cut-row OSQP per observation entirely.
# If its parity and calibration match B (which has no physics at all) while its members
# are phonon-stable, then the expensive per-datapoint repair is not buying accuracy.
#
# Each committee is centred on its own appropriate point model:
#   A and C → the constrained mean (bandpath_undotted_multivolume/theta_mean.csv)
#   B       → the REGULARISED (ridge) least-squares model, i.e. lin_params.  Verified:
#             ||lin_params − P⁻¹(Ap'Ap + λP'P)⁻¹Ap'Yw|| = 2.9e−08 against ||lin|| = 5.29,
#             with λ = 1/M — the same regularisation the POPS corrections are built from.
# so "spread about the point model" means the same thing in each panel.
#
# Outputs, per method, into results/aeq_cheap_vs_expensive/parity_calibration/<M>_*:
#   energy_parity, energy_calibration, force_parity, force_calibration  (pdf + png)
# plus parity_calibration_summary.csv with RMSE / coverage / bias for all three.
#
# RAW DATA IS SAVED so the figures can be redrawn without recomputing anything:
#   <M>_predictions.jls          the full committee_predictions NamedTuple; deserialize
#                                and pass straight back into parity_plot/calibration_hist
#   <M>_energy_predictions.csv   dft, ace, committee lo/hi per test configuration
#   <M>_force_predictions.csv    same per force component
#   <M>_energy_spread.csv        (member − point model) deviations, i.e. pr.dE
#   <M>_force_spread.csv         i.e. pr.dF — the calibration panels need these
#
# Run (all three, serial):
#   julia --project -t 8 scripts/uq/parity_calibration_three_methods_Al_12_4_6A_2.jl
# Run (ONE method, for array-parallel execution — the three are independent):
#   julia --project -t 8 scripts/uq/parity_calibration_three_methods_Al_12_4_6A_2.jl 0     # A
#   ...                                                                              1     # B
#   ...                                                                              2     # C
#   (or set SLURM_ARRAY_TASK_ID; see run_parity_calibration.slurm)
# Then merge the per-method rows into one summary:
#   julia --project scripts/uq/parity_calibration_three_methods_Al_12_4_6A_2.jl merge
#
# Optional second argument is the stride (default 10, matching the bandpath_phonon_uq run).

include(joinpath(@__DIR__, "..", "bandpath_phonon_uq", "lib.jl"))
using Serialization
@async while true; flush(stdout); sleep(5); end

element, dataset = :Al, ""
# ARGS[1]: nothing = all three | 0/1/2 = that method only | "merge" = combine rows
domerge = !isempty(ARGS) && lowercase(ARGS[1]) == "merge"
sel = if domerge
    nothing
elseif !isempty(ARGS) && tryparse(Int, ARGS[1]) !== nothing
    parse(Int, ARGS[1])
elseif haskey(ENV, "SLURM_ARRAY_TASK_ID")
    parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
else
    nothing
end
sel === nothing || (0 <= sel <= 2) || error("method index must be 0, 1 or 2 (got $sel)")
stride   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10
test_xyz = "data/Al/manual_df_test_Al.xyz"
isfile(test_xyz) || error("test set not found at $test_xyz (run from the repo root)")

HDR = "method,n_members,n_configs,E_rmse_eV,E_coverage_pct,E_bias_pctMAE,E_mean_width_eV,F_rmse_eVperA,F_coverage_pct,F_bias_pctMAE,F_mean_width_eVperA"
if domerge
    d = "models/Al_12_4_6A_2_/results/aeq_cheap_vs_expensive/parity_calibration"
    isdir(d) || error("no $d — run the per-method tasks first")
    rowfiles = sort(filter(f -> endswith(f, "_row.csv"), readdir(d)))
    isempty(rowfiles) && error("no *_row.csv in $d")
    open(joinpath(d, "parity_calibration_summary.csv"), "w") do io
        println(io, "# merged from: $(join(rowfiles, " "))")
        println(io, "# A and C centred on the constrained mean; B on the regularised (ridge) LSQ model = lin_params")
        println(io, HDR)
        for f in rowfiles; print(io, read(joinpath(d, f), String)); end
    end
    println("merged $(length(rowfiles)) rows → $d/parity_calibration_summary.csv")
    exit(0)
end

result = load_model(element, 12, 4, 6, 2; dataset_name=dataset)
model, lin = result.model, result.lin_params
SRC    = "$(result.dir)/results/aeq_cheap_vs_expensive"
outdir = "$SRC/parity_calibration"; mkpath(outdir)
θ_con  = vec(readdlm("$(result.dir)/results/bandpath_undotted_multivolume/theta_mean.csv", ','))

# (tag, csv, centre, colour) — B is centred on the regularised (ridge) LSQ model
methods = [
 ("A_cheap_reject",     "$SRC/committee_A_cheap_phononreject.csv",     θ_con, RGBf(0.0,0.447,0.698)),
 ("B_naive",            "$SRC/committee_B_naive.csv",                  lin,   RGBf(0.80,0.15,0.15)),
 ("C_expensive_repair", "$SRC/committee_C_expensive_phononreject.csv", θ_con, RGBf(0.0,0.62,0.451)),
]
@printf("test set %s (stride %d)\nout → %s\n\n", test_xyz, stride, outdir); flush(stdout)

sel === nothing || (methods = [methods[sel+1]])
@printf("running %d method(s): %s\n\n", length(methods), join(first.(methods), ", ")); flush(stdout)
rows = String[]
for (tag, csv, centre, col) in methods
    isfile(csv) || (@warn "missing $csv — skipping $tag"; continue)
    Θ = readdlm(csv, ',')
    size(Θ,2) == length(lin) || error("$csv has $(size(Θ,2)) columns, model has $(length(lin))")
    committee = [collect(Float64, Θ[k,:]) for k in 1:size(Θ,1)]
    @printf("── %s: %d members ──\n", tag, length(committee)); flush(stdout)

    pr = committee_predictions(model, committee, test_xyz; stride=stride, point_params=centre)
    eR = parity_plot(pr.tE, pr.pE, pr.loE, pr.hiE,
                     "DFT energy (eV)", "ACE energy (eV)",
                     "$outdir/$(tag)_energy_parity.png"; col=col)
    cE = calibration_hist(pr.tE, pr.pE, pr.loE, pr.hiE, pr.dE;
                          label="Energy — $tag", path="$outdir/$(tag)_energy_calibration.png")
    @printf("   energy : RMSE %.4g eV | coverage %.1f%% | bias %.0f%% MAE\n",
            eR, cE.coverage, cE.bias); flush(stdout)

    fR = NaN; cF = (coverage=NaN, bias=NaN, rmse=NaN)
    if !isempty(pr.tF)
        fR = parity_plot(pr.tF, pr.pF, pr.loF, pr.hiF,
                         "DFT force (eV/Å)", "ACE force (eV/Å)",
                         "$outdir/$(tag)_force_parity.png"; col=col)
        cF = calibration_hist(pr.tF, pr.pF, pr.loF, pr.hiF, pr.dF;
                              label="Force — $tag", path="$outdir/$(tag)_force_calibration.png")
        @printf("   force  : RMSE %.4g eV/Å | coverage %.1f%% | bias %.0f%% MAE\n",
                fR, cF.coverage, cF.bias); flush(stdout)
    end
    # ── save the raw predictions so the plots can be redrawn without recomputing ──
    serialize("$outdir/$(tag)_predictions.jls", pr)          # full NamedTuple: replot directly
    writedlm("$outdir/$(tag)_energy_predictions.csv",
             vcat(["dft_eV" "ace_eV" "committee_lo_eV" "committee_hi_eV"],
                  hcat(pr.tE, pr.pE, pr.loE, pr.hiE)), ',')
    writedlm("$outdir/$(tag)_energy_spread.csv",
             vcat(["member_minus_point_eV"], reshape(pr.dE, :, 1)), ',')
    if !isempty(pr.tF)
        writedlm("$outdir/$(tag)_force_predictions.csv",
                 vcat(["dft_eVperA" "ace_eVperA" "committee_lo" "committee_hi"],
                      hcat(pr.tF, pr.pF, pr.loF, pr.hiF)), ',')
        writedlm("$outdir/$(tag)_force_spread.csv",
                 vcat(["member_minus_point_eVperA"], reshape(pr.dF, :, 1)), ',')
    end
    @printf("   saved: %s_predictions.jls (+ energy/force predictions & spread CSVs)\n", tag); flush(stdout)

    # mean interval width — coverage is only meaningful alongside it
    wE = mean(pr.hiE .- pr.loE)
    wF = isempty(pr.tF) ? NaN : mean(pr.hiF .- pr.loF)
    @printf("   mean envelope width: energy %.4g eV, force %.4g eV/Å\n\n", wE, wF); flush(stdout)
    row = @sprintf("%s,%d,%d,%.6g,%.4f,%.4f,%.6g,%.6g,%.4f,%.4f,%.6g",
                   tag, length(committee), pr.n, eR, cE.coverage, cE.bias, wE,
                   fR, cF.coverage, cF.bias, wF)
    push!(rows, row)
    write("$outdir/$(tag)_row.csv", row * "\n")     # so a parallel run can be merged
end

if sel === nothing
    open("$outdir/parity_calibration_summary.csv", "w") do io
        println(io, "# test set $test_xyz, stride $stride")
        println(io, "# A and C centred on the constrained mean; B on the regularised (ridge) LSQ model = lin_params")
        println(io, HDR)
        foreach(r -> println(io, r), rows)
    end
    println("summary → $outdir/parity_calibration_summary.csv")
else
    println("wrote per-method row; merge with:  julia --project $(PROGRAM_FILE) merge")
end
ACEpotentials.Models.set_linear_parameters!(model, lin)
