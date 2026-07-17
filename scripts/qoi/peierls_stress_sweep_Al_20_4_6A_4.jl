# peierls_stress_sweep_Al_20_4_6A_4.jl
#
# Drives peierls_stress_gsf_pn.jl across all 10 Al_20_4_6A_4 models (5
# dataset sizes x {unconstrained, exact_constrained}), producing the headline
# figure: does exact-constraining the elastic constants let a model trained
# on LESS data match/beat an unconstrained model trained on MORE data, on a
# static 0K QoI (Peierls stress) that depends on those elastic constants?
#
# Requires Stage 1 (build_subsets_Al_20_4_6A_4.jl) and Stage 2
# (exact_constrain_Al_20_4_6A_4.jl) to have completed.
#
# Cheap relative to Stages 1-2 (the GSF curve is single-point, no
# relaxation), but still loads 10 copies of a 5476-basis-function model —
# fine interactively.
#
# Usage:
#   julia --project scripts/qoi/peierls_stress_sweep_Al_20_4_6A_4.jl

using ACEWorkflow, ACEpotentials, AtomsBuilder, Unitful
using DelimitedFiles, Printf, CairoMakie

include(joinpath(@__DIR__, "peierls_stress_gsf_pn.jl"))

const ELEMENT     = :Al
const TOTALDEGREE = 20
const SMOOTHNESS  = 4
const RCUT        = 6.0
const ORDER       = 4

const DATASET_NAMES = ["subset_5_percent", "subset_10_percent",
                        "subset_20_percent", "subset_50_percent", "full"]
const PERCENTS = Dict("subset_5_percent" => 5, "subset_10_percent" => 10,
                       "subset_20_percent" => 20, "subset_50_percent" => 50,
                       "full" => 100)

function evaluate_model(model, label, dataset_name)
    a_eq = ACEWorkflow.relax_lattice_constant(model, ELEMENT)
    C, _, _ = ACEWorkflow.strain_hessian_GPa(model, ELEMENT; a=a_eq)
    C11, C12, C44 = C[1,1], C[1,2], C[4,4]

    (; gsf, pn) = peierls_stress_static(model, ELEMENT, a_eq, C11, C12, C44)

    @printf("  [%-20s %-18s] a_eq=%.4f  C11=%.1f C12=%.1f C44=%.1f GPa  γ_us=%.4f eV/Å²  τ_P=%.4f GPa\n",
            dataset_name, label, a_eq, C11, C12, C44, gsf.gamma_us, pn.tau_P)

    return (; dataset_name, percent=PERCENTS[dataset_name], label,
              a_eq, C11, C12, C44, gamma_us=gsf.gamma_us, tau_P=pn.tau_P)
end

results = NamedTuple[]

println("═"^78)
println("  Peierls-stress sweep — Al_20_4_6A_4, unconstrained vs exact_constrained")
println("═"^78)

for dataset_name in DATASET_NAMES
    result = load_model(ELEMENT, TOTALDEGREE, SMOOTHNESS, RCUT, ORDER; dataset_name=dataset_name)

    push!(results, evaluate_model(result.model, "unconstrained", dataset_name))

    con_json = joinpath(result.dir, "exact_constrained_model.json")
    if isfile(con_json)
        model_con, _ = ACEpotentials.load_model(con_json)
        push!(results, evaluate_model(model_con, "exact_constrained", dataset_name))
    else
        @warn "Missing $con_json — run exact_constrain_Al_20_4_6A_4.jl first. Skipping." dataset_name
    end
end

# ── Save CSV ─────────────────────────────────────────────────────────────────
outdir = joinpath(@__DIR__, "..", "..", "models", "Al_20_4_6A_4", "results")
mkpath(outdir)

header = ["percent" "label" "a_eq" "C11" "C12" "C44" "gamma_us" "tau_P"]
rows   = [[r.percent r.label r.a_eq r.C11 r.C12 r.C44 r.gamma_us r.tau_P] for r in results]
writedlm(joinpath(outdir, "peierls_stress_sweep.csv"), vcat(header, rows...), ',')
println("\nSaved: $(joinpath(outdir, "peierls_stress_sweep.csv"))")

# ── Comparison plot ──────────────────────────────────────────────────────────
uncon = sort(filter(r -> r.label == "unconstrained",     results), by = r -> r.percent)
con   = sort(filter(r -> r.label == "exact_constrained", results), by = r -> r.percent)

fig = Figure(size=(600, 420))
ax  = Axis(fig[1, 1];
           xlabel = "Training data (%)",
           ylabel = "Peierls stress τ_P (GPa, PN estimate)",
           title  = "Al_20_4_6A_4 — Peierls stress vs. training-data fraction",
           xscale = log10,
           xticks = ([5, 10, 20, 50, 100], ["5", "10", "20", "50", "100"]))

scatterlines!(ax, [r.percent for r in uncon], [r.tau_P for r in uncon];
              color=RGBAf(0.85, 0.45, 0.05, 0.95), linewidth=2.5, markersize=10,
              label="unconstrained")
scatterlines!(ax, [r.percent for r in con], [r.tau_P for r in con];
              color=RGBAf(0.15, 0.4, 0.75, 0.95), linewidth=2.5, markersize=10,
              label="exact_constrained")

axislegend(ax; position=:rb, framevisible=false)

outpath = joinpath(outdir, "peierls_stress_vs_training_data.png")
save(outpath, fig)
display(fig)
println("Saved: $outpath")
