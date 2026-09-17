# diagnose_stale_cache_Al_12_4_6A_2.jl
#
# Confirm (or refute) that the STALE undotted-Hessian cache produced the spurious
# negative constrained-committee phonons for Al_12_4_6A_2_.
#
# Background: undotted_Hbasis_3x3x3_a4.04494.jls was built 03:04 on 24 Jul, BEFORE
# `bulk_prim_super` was added in c65b914 (15:36).  bandpath_Dk loaded that stale
# cache but paired it with a fresh bulk_prim_super FC structure → a possible
# atom-layout mismatch in the Bloch sum → spurious zone-boundary soft modes.  With
# the stale cache, the softest constrained member read -1.70 THz; the individual-
# Hessian plots (cache-free) show the whole constrained committee is STABLE.
#
# This script deletes-then-rebuilds nothing on its own — run it AFTER the stale
# a4.04494 cache has been removed, so bandpath_Dk rebuilds it fresh (current
# bulk_prim_super), then re-checks min_freq_stable for the saved rejection + repaired
# committees at a_mean.  If they come back stable → stale cache was the bug.
#
# Run:  julia --project -t <N> scripts/uq/diagnose_stale_cache_Al_12_4_6A_2.jl

include(joinpath(@__DIR__, "..", "bandpath_phonon_uq", "lib.jl"))

element = :Al; dataset = ""; N_cell = 3; N_per_seg = 20
result = load_model(element, 12, 4, 6, 2; dataset_name=dataset)
model  = result.model; lin_params = result.lin_params; n_params = length(lin_params)
cdir   = "$(result.dir)/results/bandpath_undotted"

a_mean = ACEWorkflow.relax_lattice_constant(model, element)
cachef = "$(result.dir)/results/undotted_Hbasis_$(N_cell)x$(N_cell)x$(N_cell)_a$(round(a_mean;digits=5)).jls"
@printf("a_mean = %.5f Å\n", a_mean)
@printf("cache at %s\n  → %s\n", basename(cachef),
        isfile(cachef) ? "PRESENT (will be loaded — delete it first to test a clean rebuild!)" :
                         "absent → bandpath_Dk will REBUILD it fresh (current bulk_prim_super)")

bp = bandpath_Dk(result, model, element, a_mean, N_cell; N_per_seg=N_per_seg)

rej    = readdlm("$cdir/committee_rejection.csv", ',')
rep    = readdlm("$cdir/committee_repaired.csv", ',')
θ_mean = vec(readdlm("$cdir/theta_mean.csv", ','))
minf_rej = [min_freq_stable(collect(Float64, r), bp) for r in eachrow(rej)]
minf_rep = [min_freq_stable(collect(Float64, r), bp) for r in eachrow(rep)]

@printf("\n── FRESH-cache undotted min ω at a_mean ─────────────────────\n")
@printf("  rejection : ∈ [%+.3f, %+.3f] THz — %d/%d unstable (<-0.05)\n",
        minimum(minf_rej), maximum(minf_rej), count(<(-0.05), minf_rej), length(minf_rej))
@printf("  repaired  : ∈ [%+.3f, %+.3f] THz — %d/%d unstable (<-0.05)\n",
        minimum(minf_rep), maximum(minf_rep), count(<(-0.05), minf_rep), length(minf_rep))
@printf("  mean model: min ω = %+.3f THz\n", min_freq_stable(θ_mean, bp))

softest = min(minimum(minf_rej), minimum(minf_rep))
@printf("\n  reference: stale cache gave softest = -1.70 THz;\n")
@printf("             individual-Hessian plots show the committee STABLE.\n")
@printf("\nVERDICT: fresh-cache softest = %+.3f THz → %s\n", softest,
        softest > -0.05 ? "committee STABLE ⇒ the STALE CACHE was the bug ✓" :
                          "STILL negative ⇒ not (only) the cache — deeper geometry/Bloch issue, investigate further")
