# paper_figures/manifest.jl
#
# The single source of truth for "which script makes which figure, and what does it
# need".  This file holds NO plotting code and NO copies of the analysis scripts —
# every entry points at the canonical script under scripts/, so there is exactly one
# version of each script in the repository and no chance of a collated copy drifting
# away from the one that actually produced a published figure.
#
# Adding a figure: append one entry to FIGURES.  Nothing else needs to change —
# check_inputs.jl and make_figures.jl both read this list.
#
# Field meanings
#   id       short slug; what you pass to make_figures.jl
#   title    what the figure shows, in the paper's terms
#   script   repo-relative path to the canonical script
#   cmd      exact command, run from the repository root
#   env      environment variables the script reads, with their defaults
#   outputs  every file the script writes that appears in (or backs) the paper
#   inputs   (path, producer, note) — `producer` is the script that makes it, or
#            :external for a data root that cannot be regenerated from this repo
#   notes    anything a reviewer needs to know to interpret a re-run

const REPO = normpath(joinpath(@__DIR__, ".."))

const FIGURES = [

(
 id      = "fullcloud_bands_parity_calibration",
 title   = "Constrained POPS scaled to the full leverage cloud: phonon bands of the "*
           "rejection-sampled committee, plus test-set parity and calibration for "*
           "both hypercube types (73k-delta cloud vs 30-member forest)",
 script  = "scripts/uq/hypercube_full_cloud_bands_Al_12_4_6A_2.jl",
 cmd     = "julia --project -t 8 scripts/uq/hypercube_full_cloud_bands_Al_12_4_6A_2.jl 30 20",
 env     = ["FIGW" => "540   # display width in pt; fonts stay 13/12/11, use 260 for a half-width slot"],
 outputs = ["models/Al_12_4_6A_2_/results/cutting_plane_full_cloud/bands_hypercube_full_cloud.pdf",
            "models/Al_12_4_6A_2_/results/cutting_plane_full_cloud/parity_fullcloud.pdf",
            "models/Al_12_4_6A_2_/results/cutting_plane_full_cloud/parity_forest30.pdf",
            "models/Al_12_4_6A_2_/results/cutting_plane_full_cloud/calibration_fullcloud.pdf",
            "models/Al_12_4_6A_2_/results/cutting_plane_full_cloud/calibration_forest30.pdf",
            "models/Al_12_4_6A_2_/results/cutting_plane_full_cloud/committee_rejection_full_cloud.csv",
            "models/Al_12_4_6A_2_/results/cutting_plane_full_cloud/hypercube_summary.csv",
            "models/Al_12_4_6A_2_/results/cutting_plane_full_cloud/parity_calibration_predictions.jls"],
 inputs  = [
   ("models/Al_12_4_6A_2_/Al_12_4_6A_2.json", "scripts/model_building/build_model.jl", "fitted ACE model"),
   ("models/Al_12_4_6A_2_/A.csv",             "scripts/model_building/build_model.jl", "design matrix, ~259 MB"),
   ("models/Al_12_4_6A_2_/Y.csv",             "scripts/model_building/build_model.jl", ""),
   ("models/Al_12_4_6A_2_/P.csv",             "scripts/model_building/build_model.jl", "preconditioner"),
   ("models/Al_12_4_6A_2_/W.csv",             "scripts/model_building/build_model.jl", "row weights"),
   ("models/Al_12_4_6A_2_/lin_params.csv",    "scripts/model_building/build_model.jl", "mean-fit coefficients"),
   ("models/Al_12_4_6A_2_/results/bandpath_undotted_ncell4_densek/theta_mean.csv",
    "scripts/uq/bandpath_committee_undotted_Al_12_4_6A_2_ncell4_densek.jl",
    "phonon-repaired constrained mean, ~2 KB"),
   ("models/Al_12_4_6A_2_/results/bandpath_undotted_ncell4_densek/committee_repaired.csv",
    "scripts/uq/bandpath_committee_undotted_Al_12_4_6A_2_ncell4_densek.jl",
    "the legacy 30-member forest, ~58 KB"),
   ("models/Al_12_4_6A_2_/results/cutting_plane_full_cloud/committee_stable.jls",
    "scripts/uq/cutting_plane_full_cloud_Al_12_4_6A_2.jl",
    "73,411 cutting-plane-constrained stable members, ~55 MB"),
   ("data/Al/manual_df_test_Al.xyz", :external, "held-out test set, ~18 MB"),
   ("scripts/bandpath_phonon_uq/lib.jl", :repo, "included helper library"),
 ],
 notes   = """
   Rejection sampling here uses the FULL constraint funnel: Born rows (c44, c11−c12,
   c11+2c12, b″) then |Δa_eq|/a ≤ 10% then min ω ≥ 0.15 THz.  Seeded (Random.seed!(1234)),
   but the acceptance rate depends on the RNG stream, so a re-run reproduces the figure
   and the statistics, not the individual member indices.
   Second positional argument is the test-set stride (20 = every 20th configuration).
   """,
),

(
 id      = "pinned_rejection_phonons",
 title   = "a_eq-pinned POPS committee, no-rejection vs rejection sampling: phonon "*
           "bands side by side on the undotted 4×4×4 band path, plus the stability-"*
           "margin histogram",
 script  = "scripts/uq/pinned_hypercube_rejection_Al_20_4_6A_3.jl",
 cmd     = "julia --project -t 40 scripts/uq/pinned_hypercube_rejection_Al_20_4_6A_3.jl 50",
 env     = ["ACCEPT_TOL"   => "-0.05  # THz; predicate threshold, defaults to unstable_tol",
            "MAX_ATTEMPTS" => "100000 # 2000 × n_members",
            "BUILD_THREADS" => "8     # only used if no band-path cache exists"],
 outputs = ["models/Al_20_4_6A_3_/results/pinned_hypercube_rejection/bands_pinned_rejection.pdf",
            "models/Al_20_4_6A_3_/results/pinned_hypercube_rejection/margin_pinned_rejection.pdf",
            "models/Al_20_4_6A_3_/results/pinned_hypercube_rejection/samples_naive.csv",
            "models/Al_20_4_6A_3_/results/pinned_hypercube_rejection/samples_rejected.csv",
            "models/Al_20_4_6A_3_/results/pinned_hypercube_rejection/min_freq_naive.csv",
            "models/Al_20_4_6A_3_/results/pinned_hypercube_rejection/min_freq_rejected.csv",
            "models/Al_20_4_6A_3_/results/pinned_hypercube_rejection/pinned_hypercube_rejection.jls"],
 inputs  = [
   ("models/Al_20_4_6A_3_/Al_20_4_6A_3.json", "scripts/model_building/build_model.jl", "fitted ACE model, 1829 params"),
   ("models/Al_20_4_6A_3_/A.csv",             "scripts/model_building/build_model.jl", "design matrix, ~5.2 GB — the peak-memory driver"),
   ("models/Al_20_4_6A_3_/Y.csv",             "scripts/model_building/build_model.jl", ""),
   ("models/Al_20_4_6A_3_/P.csv",             "scripts/model_building/build_model.jl", "preconditioner"),
   ("models/Al_20_4_6A_3_/W.csv",             "scripts/model_building/build_model.jl", "row weights"),
   ("models/Al_20_4_6A_3_/lin_params.csv",    "scripts/model_building/build_model.jl", "mean-fit coefficients"),
 ],
 notes   = """
   SELF-CONTAINED apart from the model: if no 4×4×4 undotted band-path cache is found it
   builds one (~15–20 min) and caches it, so a reviewer needs nothing else on disk.  It
   looks for a cache in, in order:
     models/Al_20_4_6A_3_/results/forest_phonon_stability/bandpath_4x4x4.jls
     models/Al_20_4_6A_3_/results/pinned_forest/bandpath_4x4x4_aref.jls
     models/Al_20_4_6A_3_/results/pinned_hypercube/bandpath_4x4x4_aref.jls
     models/Al_20_4_6A_3_/results/pinned_hypercube_rejection/bandpath_4x4x4_aref.jls

   Peak memory ~15–20 GB, dominated by reading A.csv and forming Ainv = C \\ Ap'.

   Both committees are drawn from the SAME box with the same seed, so the naive panel
   reproduces scripts/uq/pinned_hypercube_Al_20_4_6A_3.jl member-for-member and the
   comparison is paired.  No OSQP anywhere in this path, so unlike the constrained-QP
   committees it is exactly reproducible down to member identity.

   Integrity check printed on every run: the hypercube must keep 1828 of 1829
   directions.  1829 means the a_eq pin leaked and the figure is not what it claims.
   """,
),

]

figure_by_id(id) = (i = findfirst(f -> f.id == id, FIGURES);
                    i === nothing ? error("unknown figure id: $id\nknown: " *
                                          join([f.id for f in FIGURES], ", ")) : FIGURES[i])
