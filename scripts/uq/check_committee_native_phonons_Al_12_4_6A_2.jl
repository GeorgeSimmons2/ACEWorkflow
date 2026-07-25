# check_committee_native_phonons_Al_12_4_6A_2.jl
#
# Independent, FROM-SCRATCH stability check of the saved constrained committee
# (rejection + repaired) for Al_12_4_6A_2_.  No undotted per-basis trick, NO cache:
# for every member we set its θ and call precompute_force_constants directly (native
# Hessian), then take the min non-acoustic band-path frequency.  Evaluated at the
# FIXED a_mean geometry — the same geometry the hypercube predicate used — so it is a
# clean control against the stale-cache undotted value (softest read -1.70 THz).
#
# Light model load (ACEpotentials.load_model on the JSON) so it runs on a CPU without
# touching the 258 MB forest matrices.
#
# Run:  julia --project -t <N> scripts/uq/check_committee_native_phonons_Al_12_4_6A_2.jl

include(joinpath(@__DIR__, "..", "bandpath_phonon_uq", "lib.jl"))

element = :Al; N_cell = 3; N_per_seg = 20; qΓtol = 5e-2
MODELDIR = "/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/models/Al_12_4_6A_2_"
cdir = "$MODELDIR/results/bandpath_undotted"

model, _  = ACEpotentials.load_model("$MODELDIR/Al_12_4_6A_2.json")
lin_params = vec(readdlm("$MODELDIR/lin_params.csv", ','))
ACEpotentials.Models.set_linear_parameters!(model, lin_params)
a_mean = ACEWorkflow.relax_lattice_constant(model, element)
@printf("Native from-scratch phonons at fixed a_mean = %.5f Å  (%d threads)\n", a_mean, Threads.nthreads())

sp, ss   = bulk_prim_super(element; a=a_mean, N_cell=N_cell)   # geometry fixed for all members
structure = AtomsBuilder.Chemistry.symmetry(element)
ACEpotentials.Models.set_linear_parameters!(model, lin_params)
fc_ref   = precompute_force_constants(sp, ss, model)
ql, xv, xt, lb, _ = _band_path(structure, fc_ref.L; N_per_seg=N_per_seg)
qn = norm.(ql); Np = fc_ref.Np
@printf("  path: %d q-points, %d branches\n", length(ql), 3Np)

# native min non-acoustic frequency for one θ (fresh precompute_force_constants)
function native_minomega(θ)
    ACEpotentials.Models.set_linear_parameters!(model, θ)
    fc = precompute_force_constants(sp, ss, model)
    m = Inf
    for (iq, q) in enumerate(ql)
        qn[iq] < qΓtol && continue
        ev = eigvals(Hermitian(dynamical_matrix_from_fc(fc, q)))
        m = min(m, minimum(sign.(ev) .* sqrt.(abs.(ev)) .* FREQ_THz))
    end
    return m
end

# sanity: the nominal (lin_params) model must be stable & sensible (~Al, max ω ≈ 9-10 THz)
mrefmin = native_minomega(lin_params)
@printf("  sanity — nominal lin_params model: min ω = %+.3f THz (should be > 0)\n", mrefmin)

rej    = readdlm("$cdir/committee_rejection.csv", ',')
rep    = readdlm("$cdir/committee_repaired.csv", ',')
θ_mean = vec(readdlm("$cdir/theta_mean.csv", ','))

println("\nComputing native phonons for 30 rejection + 30 repaired + mean …")
t = @elapsed begin
    mrej = [native_minomega(collect(Float64, r)) for r in eachrow(rej)]
    mrep = [native_minomega(collect(Float64, r)) for r in eachrow(rep)]
    mmean = native_minomega(θ_mean)
end
@printf("  done in %.1f min\n", t/60)

@printf("\n── Native from-scratch min ω (THz) at a_mean ────────────────\n")
@printf("  rejection : ∈ [%+.3f, %+.3f] — %d/%d unstable(<-0.05); softest = member %d\n",
        minimum(mrej), maximum(mrej), count(<(-0.05), mrej), length(mrej), argmin(mrej))
@printf("  repaired  : ∈ [%+.3f, %+.3f] — %d/%d unstable(<-0.05); softest = member %d\n",
        minimum(mrep), maximum(mrep), count(<(-0.05), mrep), length(mrep), argmin(mrep))
@printf("  mean model: %+.3f\n", mmean)
@printf("  repaired[15] (the NPT/stale-cache -1.70 culprit): %+.3f\n", mrep[15])

softest = min(minimum(mrej), minimum(mrep))
@printf("\nVERDICT: native softest = %+.3f THz  →  %s\n", softest,
        softest > -0.05 ? "committee STABLE ✓  (the undotted -1.70 was a STALE-CACHE artifact)" :
                          "GENUINELY soft — NOT a cache artifact; the rejection check was flawed")

open("$MODELDIR/results/native_committee_minomega.csv", "w") do io
    println(io, "set,index,min_omega_THz")
    for (i,v) in enumerate(mrej); @printf(io, "rejection,%d,%.4f\n", i, v); end
    for (i,v) in enumerate(mrep); @printf(io, "repaired,%d,%.4f\n", i, v); end
    @printf(io, "mean,0,%.4f\n", mmean)
end
println("Saved per-member native min ω → $MODELDIR/results/native_committee_minomega.csv")
