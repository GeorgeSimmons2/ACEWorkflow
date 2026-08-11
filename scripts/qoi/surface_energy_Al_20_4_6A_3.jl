# surface_energy_Al_20_4_6A_3.jl
#
# Surface energy for the Al_20_4_6A_3 model across its two committees — naive and
# rejection-sampled — from scripts/uq/pinned_hypercube_rejection_Al_20_4_6A_3.jl.
#
# A thin driver, not a second implementation: it sets the configuration and includes
# surface_energy_vacuum.jl, so Al_12 and Al_20 can never drift apart in method.
#
# ── HOW THIS DIFFERS FROM THE Al_12 RUN ─────────────────────────────────────
# BOTH committees are centred on lin_params.  They are drawn from the SAME hypercube
# with the same seed, the rejected one additionally requiring b″·θ > 0 and min ω ≥ tol
# of every member, so the pair is paired member-for-member and there is no theta_mean
# to point at.  The Al_12 pair is different: unconstrained around lin_params,
# constrained around the phonon-repaired theta_mean.
#
# The comparison is therefore narrower and cleaner than Al_12's.  There, the two
# committees differ in the constraint AND in which cloud the hypercube was fitted to.
# Here only the rejection predicate differs, so any change in γ is the predicate alone.
#
# ── COST ────────────────────────────────────────────────────────────────────
# 1829 parameters against Al_12's 91, and 50 members per committee against 30 — so
# roughly 100 members × 2 minimisations on a 256-atom cell, each energy evaluation
# about 20× the Al_12 cost.  Expect this to be substantially slower; set QOI_THREADS.
# The model is loaded from JSON + lin_params.csv, so the 5.2 GB A.csv is never read.
#
# Run:  julia --project -t <ncores> scripts/qoi/surface_energy_Al_20_4_6A_3.jl
#       SURFACE=111 julia --project -t <ncores> scripts/qoi/surface_energy_Al_20_4_6A_3.jl
#   Every knob of surface_energy_vacuum.jl still applies (SURFACE, N_SUPER, VACUUM,
#   NORMAL, QOI_THREADS, NBINS, FIGW); this only fixes the model and the two committees.

const MD20 = "models/Al_20_4_6A_3_"
const SRC20 = "$MD20/results/pinned_hypercube_rejection"

for (k, v) in ("MODELDIR"    => MD20,
               "ENS1_CSV"    => "$SRC20/samples_naive.csv",
               "ENS2_CSV"    => "$SRC20/samples_rejected.csv",
               "ENS1_TAG"    => "naive",
               "ENS2_TAG"    => "rejected",
               "ENS1_LABEL"  => "naive hypercube",
               "ENS2_LABEL"  => "rejection sampled",
               # both committees are lin_params + δ; there is no theta_mean here
               "ENS1_CENTRE" => "lin_params",
               "ENS2_CENTRE" => "lin_params",
               "OUTDIR"      => "$SRC20/surface_energy")
    # don't clobber anything the caller set deliberately
    haskey(ENV, k) || (ENV[k] = v)
end

isempty(ARGS) || error("""
    this driver fixes the two committees, so positional CSV arguments are ignored.
    Use scripts/qoi/surface_energy_vacuum.jl directly if you want different ones.""")

include(joinpath(@__DIR__, "surface_energy_vacuum.jl"))
