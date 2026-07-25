# phonon_pipeline_ncell_diagnosis_Al_12_4_6A_2.jl
#
# Why does the undotted band-path check (bandpath_Dk) disagree with the native
# precompute_force_constants phonons for some members (repaired[15]: -1.70 vs +0.15;
# nominal: +0.19 vs -0.50)?  Hypothesis: N_cell=3 with the 6 Å cutoff lets each atom
# see its own periodic image; the UNDOTTED path carries that raw-Hessian contamination
# into the phonons, while precompute_force_constants folds/ASR-corrects it.  If so, a
# bigger cell removes the self-images and the two pipelines converge.
#
# For θ ∈ {nominal, repaired[15], softest-rejection} and N_cell ∈ {3,4,5}, evaluate
# min non-acoustic band ω BOTH ways at fixed a_mean:
#   • native : dynamical_matrix_from_fc(fc, q)                       (fc.H, processed)
#   • raw    : dynamical_matrix_from_fc(merge(fc, H=hessian(ss)), q) (raw atomic H ≈ undotted)
#
# Run:  julia --project -t <N> scripts/uq/phonon_pipeline_ncell_diagnosis_Al_12_4_6A_2.jl

include(joinpath(@__DIR__, "..", "bandpath_phonon_uq", "lib.jl"))
using AtomsCalculatorsUtilities.SitePotentials: hessian

element = :Al; N_per_seg = 20; qΓtol = 5e-2; N_cells = [3, 4, 5]
MODELDIR = "/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/models/Al_12_4_6A_2_"
cdir = "$MODELDIR/results/bandpath_undotted"

model, _   = ACEpotentials.load_model("$MODELDIR/Al_12_4_6A_2.json")
lin_params = vec(readdlm("$MODELDIR/lin_params.csv", ','))
rep = readdlm("$cdir/committee_repaired.csv", ','); rej = readdlm("$cdir/committee_rejection.csv", ',')
θs = ["nominal"        => lin_params,
      "repaired[15]"   => collect(Float64, rep[15, :]),
      "rejection[6]"   => collect(Float64, rej[6, :])]      # member 6 = softest rejection (native check)

ACEpotentials.Models.set_linear_parameters!(model, lin_params)
a_mean = ACEWorkflow.relax_lattice_constant(model, element)
structure = AtomsBuilder.Chemistry.symmetry(element)
@printf("a_mean = %.5f Å;  model cutoff = 6 Å (the '6A')\n", a_mean)

function minω_both(θ, N_cell)
    ACEpotentials.Models.set_linear_parameters!(model, θ)
    sp, ss = bulk_prim_super(element; a=a_mean, N_cell=N_cell)
    fc = precompute_force_constants(sp, ss, model)
    H_atomic = ustrip.(hessian(ss, model))                    # raw atomic Hessian (≈ undotted sum)
    ql, _, _, _, _ = _band_path(structure, fc.L; N_per_seg=N_per_seg); qn = norm.(ql)
    mn(fcuse) = begin
        m = Inf
        for (iq, q) in enumerate(ql)
            qn[iq] < qΓtol && continue
            ev = eigvals(Hermitian(dynamical_matrix_from_fc(fcuse, q)))
            m = min(m, minimum(sign.(ev).*sqrt.(abs.(ev)).*FREQ_THz))
        end; m
    end
    box_half = ustrip(a_mean)*N_cell/2
    (native=mn(fc), raw=mn(merge(fc, (H=H_atomic,))), natoms=length(ss), box_half=box_half)
end

@printf("\n%-14s %6s %8s %5s %10s %10s %9s\n", "θ", "N_cell", "box/2(Å)", "atoms", "native", "raw(undot)", "Δ")
for N_cell in N_cells, (name, θ) in θs
    r = minω_both(θ, N_cell)
    flag = r.box_half < 6.0 ? "  ← cutoff>box/2 (self-images)" : ""
    @printf("%-14s %6d %8.2f %5d %+10.3f %+10.3f %9.3f%s\n",
            name, N_cell, r.box_half, r.natoms, r.native, r.raw, r.native - r.raw, flag)
end
println("\nIf native≈raw and both>0 at N_cell=4,5 → the -1.70 was N_cell=3 self-image contamination")
println("carried by the raw-Hessian (undotted) path. If they still differ → a construction (ASR) bug.")
