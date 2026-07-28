# gsf_peierls_exact_cij_Al_20_4_6A_4.jl
#
# Does pinning the elastic constants to experiment improve a dislocation observable?
#
# Compares two mean models of Al_20_4_6A_4 (5476 basis functions):
#   nominal      Al_20_4_6A_4.json          plain regularised least squares
#   constrained  exact_constrained_model.json   C11/C12/C44 pinned EXACTLY to the
#                                               literature targets below, with
#                                               b′(a_eq)·θ = 0 so the model keeps its
#                                               own relaxed equilibrium
#                (produced by scripts/elasticity/exact_constrain_Al_20_4_6A_4.jl)
#
#     C11 = 116.3 GPa,  C12 = 64.8 GPa,  C44 = 30.9 GPa
#
# and evaluates, for each:
#   • the rigid {111}⟨112⟩ generalized stacking-fault curve γ(u) and its peak γ_us
#   • the cubic elastic constants (verifying the constrained model hits its targets)
#   • the sinusoidal Peierls–Nabarro stress τ_P, which depends on BOTH γ_us and C_ij
#
# WHY THIS PAIR.  τ_P = f(γ_us, C11, C12, C44). Pinning C_ij removes the elastic half
# of that dependence exactly, so any remaining difference in τ_P is attributable to
# γ_us — and the Born inequalities used elsewhere in this work constrain elastic
# *stability* without constraining elastic *values*, which was shown to leave C11
# spanning 48–306 GPa across a constrained POPS ensemble. This is the controlled
# version of that observation.
#
# Each model uses its OWN relaxed a_eq (there is no shared-geometry linearity trick
# needed here, since we evaluate only two models rather than an ensemble).
#
# Reuses scripts/qoi/peierls_stress_gsf_pn.jl unmodified.
#
# Outputs → models/Al_20_4_6A_4/results/gsf_peierls_exact_cij/
#   gsf_curves_exact_cij.pdf/png   γ(u) for both models
#   gsf_peierls_exact_cij.csv      a_eq, C_ij, γ_us, G, ν, τ_P for both
#
# Run:  julia --project -t 8 scripts/uq/gsf_peierls_exact_cij_Al_20_4_6A_4.jl

include(joinpath(@__DIR__, "..", "bandpath_phonon_uq", "lib.jl"))
include(joinpath(@__DIR__, "..", "qoi", "peierls_stress_gsf_pn.jl"))
@async while true; flush(stdout); sleep(5); end

element   = :Al
layers    = 24
vacuum    = 15.0
n_steps   = 20
character = :edge
C11_TARGET, C12_TARGET, C44_TARGET = 116.3, 64.8, 30.9      # GPa, literature Al

result = load_model(element, 20, 4, 6, 4; dataset_name="")
dir    = result.dir
outdir = "$dir/results/gsf_peierls_exact_cij"; mkpath(outdir)
@printf("model dir: %s\n", dir); flush(stdout)

# the nominal json is named after the model; be tolerant of either convention
nominal_json = let cands = ["$dir/Al_20_4_6A_4.json", "$dir/model.json"]
    i = findfirst(isfile, cands)
    i === nothing && error("no nominal model json found; looked for:\n  " * join(cands, "\n  "))
    cands[i]
end
constr_json = let cands = ["$dir/exact_constrained_model.json", "$dir/constrained_model.json"]
    i = findfirst(isfile, cands)
    i === nothing && error("no constrained model json found; looked for:\n  " * join(cands, "\n  "))
    cands[i]
end
@printf("  nominal     : %s\n  constrained : %s\n\n", basename(nominal_json), basename(constr_json))
flush(stdout)

# ── evaluate one model end to end ───────────────────────────────────────────
function evaluate(tag, json)
    @printf("── %s (%s) ──\n", tag, basename(json)); flush(stdout)
    m, _ = ACEpotentials.load_model(json)
    a_eq = ACEWorkflow.relax_lattice_constant(m, element)
    # elastic_hessian_ENERGY: one 6x6 Hessian of the contracted energy. The _basis
    # variant builds a 6x6 Hessian per basis function -- 5476 of them for this model --
    # which is what you want for constraint ROWS but is wasteful when only the
    # contracted tensor is needed.
    sys0 = ACEWorkflow.Elasticity.reference_system(element; a=a_eq)
    Vcell = abs(det(ustrip.(ACEWorkflow.Elasticity.lattice_matrix(sys0.cell.cell_vectors))))
    Hε = elastic_hessian_energy(m; element=element, a=a_eq)          # eV
    Cmat = Hε .* (160.2176621 / Vcell)                                # GPa
    C11, C12, C44 = Cmat[1,1], Cmat[1,2], Cmat[4,4]
    @printf("   a_eq = %.5f Å\n", a_eq)
    @printf("   C11 = %7.2f  C12 = %7.2f  C44 = %7.2f GPa\n", C11, C12, C44)
    @printf("   vs target %7.2f %7.2f %7.2f  →  Δ = %+.2f %+.2f %+.2f\n",
            C11_TARGET, C12_TARGET, C44_TARGET,
            C11-C11_TARGET, C12-C12_TARGET, C44-C44_TARGET); flush(stdout)

    t0 = time()
    g  = gsf_curve(m, element, a_eq; layers=layers, vacuum=vacuum, n_steps=n_steps)
    @printf("   γ-surface: %d displacements in %.1f min\n", n_steps+1, (time()-t0)/60)
    @printf("   γ_us = %.6f eV/Å² = %.1f mJ/m²   (Al reference ≈ 160–180)\n",
            g.gamma_us, g.gamma_us*16021.77); flush(stdout)

    pn = peierls_nabarro_stress(C11, C12, C44, g.gamma_us, g.b_partial; character=character)
    stable = (pn.G > 0) && (-1 < pn.ν < 0.5) && isfinite(pn.tau_P)
    @printf("   G = %.2f GPa, ν = %.4f  →  %s\n", pn.G, pn.ν,
            stable ? "elastically stable" : "*** ELASTICALLY UNSTABLE — τ_P undefined ***")
    stable && @printf("   τ_P = %.5f GPa  (ζ = %.3f Å, τ_max = %.3f GPa)\n",
                      pn.tau_P, pn.zeta, pn.tau_max)
    println(); flush(stdout)
    return (; tag, a_eq, C11, C12, C44, gsf=g, pn, stable)
end

nom = evaluate("nominal (RLS)",          nominal_json)
con = evaluate("constrained (exact C_ij)", constr_json)

# ── figure ───────────────────────────────────────────────────────────────────
let fig = Figure(size=(520,340), figure_padding=(6,10,4,6))
    ax = Axis(fig[1,1]; xlabel="partial displacement  u / b", ylabel="γ (eV/Å²)",
              title="Rigid {111}⟨112⟩ γ-surface: RLS vs exact-C\$_{ij}\$ constrained",
              titlesize=10, xgridvisible=false, ygridvisible=false,
              xtickalign=1, ytickalign=1)
    lines!(ax, nom.gsf.u, nom.gsf.gamma; color=RGBf(0.80,0.15,0.15), linewidth=1.8)
    lines!(ax, con.gsf.u, con.gsf.gamma; color=RGBf(0.0,0.447,0.698), linewidth=1.8)
    hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=0.8)
    elem = [LineElement(color=RGBf(0.80,0.15,0.15)), LineElement(color=RGBf(0.0,0.447,0.698))]
    Legend(fig[1,1], elem,
           ["nominal (RLS), γ_us = $(round(nom.gsf.gamma_us*16021.77;digits=0)) mJ/m²",
            "exact C_ij, γ_us = $(round(con.gsf.gamma_us*16021.77;digits=0)) mJ/m²"];
           tellwidth=false, tellheight=false, halign=:left, valign=:top,
           margin=(8,8,8,8), framevisible=true, labelsize=8, patchsize=(16,10))
    save("$outdir/gsf_curves_exact_cij.pdf", fig)
    save("$outdir/gsf_curves_exact_cij.png", fig; px_per_unit=4)
end

open("$outdir/gsf_peierls_exact_cij.csv", "w") do io
    println(io, "# Al_20_4_6A_4: nominal RLS vs C11/C12/C44 pinned to $C11_TARGET/$C12_TARGET/$C44_TARGET GPa")
    println(io, "# rigid {111}<112> GSF, $layers layers, $vacuum A vacuum, PN character = $character")
    println(io, "model,a_eq_Ang,C11_GPa,C12_GPa,C44_GPa,gamma_us_eV_per_A2,gamma_us_mJ_per_m2,G_GPa,poisson,tau_P_GPa,elastically_stable")
    for r in (nom, con)
        @printf(io, "%s,%.6f,%.3f,%.3f,%.3f,%.6f,%.2f,%.3f,%.5f,%.6g,%s\n",
                r.tag, r.a_eq, r.C11, r.C12, r.C44, r.gsf.gamma_us,
                r.gsf.gamma_us*16021.77, r.pn.G, r.pn.ν, r.pn.tau_P, r.stable)
    end
end
writedlm("$outdir/gsf_curves_exact_cij.csv",
         vcat(["u" "gamma_nominal_eV_per_A2" "gamma_constrained_eV_per_A2"],
              hcat(nom.gsf.u, nom.gsf.gamma, con.gsf.gamma)), ',')

@printf("── comparison ──────────────────────────────────────────────\n")
@printf("  %-26s %10s %10s\n", "", "nominal", "exact C_ij")
@printf("  %-26s %10.4f %10.4f\n", "a_eq (Å)", nom.a_eq, con.a_eq)
@printf("  %-26s %10.2f %10.2f   (target %.1f)\n", "C11 (GPa)", nom.C11, con.C11, C11_TARGET)
@printf("  %-26s %10.2f %10.2f   (target %.1f)\n", "C12 (GPa)", nom.C12, con.C12, C12_TARGET)
@printf("  %-26s %10.2f %10.2f   (target %.1f)\n", "C44 (GPa)", nom.C44, con.C44, C44_TARGET)
@printf("  %-26s %10.1f %10.1f   (Al ≈ 160–180)\n", "γ_us (mJ/m²)",
        nom.gsf.gamma_us*16021.77, con.gsf.gamma_us*16021.77)
@printf("  %-26s %10.2f %10.2f\n", "G (GPa)", nom.pn.G, con.pn.G)
@printf("  %-26s %10.4f %10.4f\n", "ν", nom.pn.ν, con.pn.ν)
@printf("  %-26s %10.5g %10.5g\n", "τ_P (GPa)", nom.pn.tau_P, con.pn.tau_P)
println("\noutputs → $outdir/")
