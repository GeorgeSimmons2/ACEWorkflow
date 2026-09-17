# surface_energy_Al_16_4_6A_3.jl
#
# Surface energy for Al_16_4_6A_3, across the two ensembles from
# scripts/uq/pinned_rejection_ensembles_Al_16_4_6A_3.jl.
#
# A thin driver, not a second implementation: it sets the configuration and includes
# scripts/qoi/surface_energy_vacuum.jl, so every model's version of this QoI stays identical in method.
#
# BOTH ensembles are centred on lin_params — they come from the same pinned hypercube
# with the same seed and differ only in the phonon-positivity predicate.  There is no
# theta_mean for this model, which is why ENS*_CENTRE has to be set explicitly.
#
# Because the pair is paired, any difference here is the predicate alone.  That is a
# cleaner attribution than the Al_12 QoIs, where the two committees differ in the
# constraint AND in which cloud the hypercube was fitted to.
#
# Run:  julia --project -t <ncores> scripts/qoi/surface_energy_Al_16_4_6A_3.jl
#   Every knob of the general script still applies; this only fixes the model and the
#   two ensembles.

const MD16  = "models/Al_16_4_6A_3_"
const SRC16 = "$MD16/results/pinned_ensembles"

for (k, v) in ("MODELDIR"    => MD16,
               "ENS1_CSV"    => "$SRC16/ensemble_unconstrained.csv",
               "ENS2_CSV"    => "$SRC16/ensemble_constrained.csv",
               "ENS1_TAG"    => "unconstrained",
               "ENS2_TAG"    => "constrained",
               "ENS1_LABEL"  => "unconstrained (no predicate)",
               "ENS2_LABEL"  => "constrained (phonon-positive)",
               "ENS1_CENTRE" => "lin_params",
               "ENS2_CENTRE" => "lin_params",
               "OUTDIR"      => "$SRC16/surface_energy")
    haskey(ENV, k) || (ENV[k] = v)      # never clobber what the caller set
end

isempty(ARGS) || error("""
    this driver fixes the two ensembles, so positional CSV arguments are ignored.
    Use scripts/qoi/surface_energy_vacuum.jl directly for different ones.""")

include(joinpath(@__DIR__, "surface_energy_vacuum.jl"))
