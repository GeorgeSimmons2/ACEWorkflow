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
 outputs = ["models/Al_12_4_6A_2_/results/cutting_plane_full_cloud/bands_fullcloud.pdf",
            "models/Al_12_4_6A_2_/results/cutting_plane_full_cloud/bands_forest30.pdf",
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
   ("scripts/uq/lib_parity_calibration.jl", :repo, "shared parity/calibration plotting"),
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

(
 id      = "naive_vs_constrained_bands",
 title   = "Unconstrained POPS vs constrained POPS: committee phonon bands side by "*
           "side on one shared frequency axis, each ensemble evaluated at the geometry "*
           "its members actually hold",
 script  = "scripts/uq/naive_vs_constrained_fullcloud_Al_12_4_6A_2.jl",
 cmd     = "julia --project -t 40 scripts/uq/naive_vs_constrained_fullcloud_Al_12_4_6A_2.jl 30",
 env     = ["FIGW"         => "540  # display width in pt; fonts stay 13/12/11",
            "HESS_THREADS" => "     # caps concurrent native builds; default all Julia threads"],
 outputs = ["models/Al_12_4_6A_2_/results/naive_vs_constrained/bands_naive_vs_constrained.pdf",
            "models/Al_12_4_6A_2_/results/naive_vs_constrained/bands_naive_shared_axis.pdf",
            "models/Al_12_4_6A_2_/results/naive_vs_constrained/bands_constrained_shared_axis.pdf",
            "models/Al_12_4_6A_2_/results/naive_vs_constrained/min_freq_naive.csv",
            "models/Al_12_4_6A_2_/results/naive_vs_constrained/min_freq_constrained.csv",
            "models/Al_12_4_6A_2_/results/naive_vs_constrained/samples_naive.csv",
            "models/Al_12_4_6A_2_/results/naive_vs_constrained/naive_vs_constrained.jls"],
 inputs  = [
   ("models/Al_12_4_6A_2_/Al_12_4_6A_2.json", "scripts/model_building/build_model.jl", "fitted ACE model"),
   ("models/Al_12_4_6A_2_/A.csv",             "scripts/model_building/build_model.jl", "design matrix, ~259 MB"),
   ("models/Al_12_4_6A_2_/Y.csv",             "scripts/model_building/build_model.jl", ""),
   ("models/Al_12_4_6A_2_/P.csv",             "scripts/model_building/build_model.jl", "preconditioner"),
   ("models/Al_12_4_6A_2_/W.csv",             "scripts/model_building/build_model.jl", "row weights"),
   ("models/Al_12_4_6A_2_/lin_params.csv",    "scripts/model_building/build_model.jl", "mean-fit coefficients"),
   ("models/Al_12_4_6A_2_/results/cutting_plane_full_cloud/committee_rejection_full_cloud.csv",
    "scripts/uq/hypercube_full_cloud_bands_Al_12_4_6A_2.jl",
    "the constrained committee — REUSED, not re-drawn, so panel (b) is the same ensemble as bands_fullcloud"),
   ("models/Al_12_4_6A_2_/results/bandpath_undotted_ncell4_densek/theta_mean.csv",
    "scripts/uq/bandpath_committee_undotted_Al_12_4_6A_2_ncell4_densek.jl",
    "constrained mean, the blue central line of panel (b)"),
   ("scripts/bandpath_phonon_uq/lib.jl", :repo, "included helper library"),
 ],
 notes   = """
   RUN THE fullcloud_bands_parity_calibration ENTRY FIRST — this one consumes its
   committee CSV so the constrained panel is that exact ensemble rather than a re-draw.

   Two evaluators, each used where it is exact, which is the point of the figure:
     constrained  prebuilt undotted per-basis Hessian at a_eq, 4×4×4.  b′·θ = 0 is
                  imposed on these members, so a_eq IS their equilibrium and one build
                  serves the whole ensemble.
     naive        native Hessian per member, each relaxed to its OWN lattice constant,
                  4×4×4.  Nothing pins these; evaluating them at a_eq would fold
                  residual stress into the phonons and make a soft mode unattributable
                  between bad parameters and wrong volume.
   Where both routes are valid they agree to ~1e-14 THz, so this is not a method
   confound.  The script prints a PIN CHECK (max |Δa| over the constrained members)
   that gates the first choice — if it is not ~0 the prebuilt operator is being used at
   the wrong geometry and the constrained panel is wrong.

   Do NOT substitute results/bands_individual_hessian/bands_naive_individual_hessian.pdf
   for panel (a): that figure is 10 members on a 3×3×3 cell (half-box 6.03 Å against a
   6 Å cutoff, so periodic images contaminate it) with a uniform q-path and its own
   y-limits.  Against bands_fullcloud it differs in four ways at once and the y-limits
   cannot be matched by cropping.

   ~31 native force-constant builds on a 256-atom cell, chunked over tasks with one
   deep-copied model each (native evaluation mutates the model, so a shared one is a
   data race).  Seeded; no OSQP in the naive path, so the naive committee is exactly
   reproducible.  The constrained committee is whatever the fullcloud run produced.
   """,
),

(
 id      = "fcc_compare_300K",
 title   = "FCC stability under NPT at 300 K: phonon dispersion of the constrained and "*
           "unconstrained members overlaid on one axis, with the RDF of each — "*
           "unconstrained above (leaves FCC), constrained below (survives)",
 script  = "scripts/uq/fcc_compare_figure_Al_12_4_6A_2.jl",
 cmd     = "julia --project -t 8 scripts/uq/fcc_compare_figure_Al_12_4_6A_2.jl 300",
 env     = ["FIGW" => "540  # display width in pt; fonts stay 11-13"],
 outputs = ["models/Al_12_4_6A_2_/results/fcc_compare_constrained_vs_naive_300K.pdf",
            "models/Al_12_4_6A_2_/results/fcc_compare_constrained_vs_naive_300K.png"],
 inputs  = [
   ("models/Al_12_4_6A_2_/Al_12_4_6A_2.json", "scripts/model_building/build_model.jl", "fitted ACE model"),
   ("models/Al_12_4_6A_2_/lin_params.csv",    "scripts/model_building/build_model.jl", "mean-fit coefficients"),
   ("models/Al_12_4_6A_2_/results/bandpath_undotted_multivolume/theta_npt_softest.csv",
    "scripts/uq/npt_multivolume_member_Al_12_4_6A_2.jl",
    "the MULTI-VOLUME constrained member (a_eq → 1.1·a_eq) — not the full-cloud committee"),
   ("models/Al_12_4_6A_2_/results/npt_thermal_expansion_naive_worst_member/theta_naive_worst.csv",
    "scripts/uq/fcc_instability_figure_Al_12_4_6A_2.jl", "the unconstrained worst member"),
   ("models/Al_12_4_6A_2_/results/npt_multivolume_softest/thermal_expansion_summary.csv",
    "scripts/uq/fcc_stability_figure_Al_12_4_6A_2.jl", "a(T) for the constrained member"),
   ("models/Al_12_4_6A_2_/results/npt_thermal_expansion_naive_worst_member/thermal_expansion_summary.csv",
    "scripts/uq/fcc_instability_figure_Al_12_4_6A_2.jl", "a(T) for the unconstrained member"),
   ("models/Al_12_4_6A_2_/results/npt_multivolume_softest/rdf_300K.csv",
    "scripts/uq/fcc_stability_figure_Al_12_4_6A_2.jl", "RDF, reused so the histogram is byte-identical to the published panel"),
   ("models/Al_12_4_6A_2_/results/npt_thermal_expansion_naive_worst_member/rdf_300K.csv",
    "scripts/uq/fcc_instability_figure_Al_12_4_6A_2.jl", "RDF, likewise"),
   ("scripts/bandpath_phonon_uq/lib.jl", :repo, "included helper library"),
 ],
 notes   = """
   THIS IS THE MULTI-VOLUME LINE, not the full-cloud one.  The constrained member here
   comes from the a_eq → 1.1·a_eq constrained study; the full-cloud committee behind
   fullcloud_bands_parity_calibration and naive_vs_constrained_bands is constrained at
   a_eq ONLY.  Do not conflate the two when writing the caption.

   Terminology in the figure is constrained / UNCONSTRAINED.  The directory and file
   names on disk still say `naive`, and the output filename is unchanged
   (fcc_compare_constrained_vs_naive_300K) so existing \\includegraphics keep resolving.

   IS THE UNDOTTED HESSIAN FAIR TO THE UNCONSTRAINED MEMBER?  Yes, and the answer is
   worth having to hand — it is the first thing a referee will ask.  Neither member is
   pinned to a_eq here.  Each gets its OWN band path built at its OWN NPT-measured
   a(300 K): a_con = 4.07786 Å, a_nai = 4.16663 Å, two separate bandpath_Dk calls (the
   two cache filenames differ, which is the tell).  H_k = ∂²B_k/∂r² is geometry-only, so
   built at a given `a` and contracted with θ it reproduces a native build at that
   geometry to ~1e-16 — "undotted" costs no accuracy, the only question is ever whether
   the geometry is right.  Here it is each member's own measured lattice constant, not
   an assumed one, which is MORE favourable to the unconstrained member than using its
   0 K a_eq would have been.

   Two caveats that are not unfairness but should be stated in the caption:
     • Quasi-harmonic — harmonic phonons at the thermally expanded lattice constant, not
       the true anharmonic 300 K spectrum.  Applies identically to both members.
     • The unconstrained member's 300 K structure is NO LONGER FCC (that is what its RDF
       panel shows).  Its FCC dispersion therefore describes the ideal FCC reference at
       that volume — the diagnostic for WHY it left FCC, not a description of what it
       actually holds.

   The RDFs are read from the CSVs written by the two original figure scripts, so the
   histograms are byte-identical to the published panels — the trajectories (~13M pair
   evaluations each) are not reprocessed.  The band structures ARE recomputed, but
   undotted_Hbasis() hits its cache at both lattice constants, so no Hessian is rebuilt.
   If those caches are absent the run is much longer.

   Fast model load by default (JSON + lin_params.csv, skipping the 259 MB A.csv, which
   nothing here needs).  Pass a second argument "full" to force the real loader.
   """,
),

(
 id      = "al20_parity_calibration",
 title   = "Test-set parity and calibration for the two Al_20_4_6A_3 pinned committees, "*
           "naive and rejection-sampled — what the physics constraint costs in "*
           "predictive spread",
 script  = "scripts/uq/parity_calibration_pinned_Al_20_4_6A_3.jl",
 cmd     = "julia --project -t 8 scripts/uq/parity_calibration_pinned_Al_20_4_6A_3.jl 20",
 env     = ["FIGW" => "540  # display width in pt; fonts stay 13/12/11"],
 outputs = ["models/Al_20_4_6A_3_/results/pinned_hypercube_rejection/parity_calibration/parity_naive.pdf",
            "models/Al_20_4_6A_3_/results/pinned_hypercube_rejection/parity_calibration/parity_rejected.pdf",
            "models/Al_20_4_6A_3_/results/pinned_hypercube_rejection/parity_calibration/calibration_naive.pdf",
            "models/Al_20_4_6A_3_/results/pinned_hypercube_rejection/parity_calibration/calibration_rejected.pdf",
            "models/Al_20_4_6A_3_/results/pinned_hypercube_rejection/parity_calibration/parity_calibration_summary.csv",
            "models/Al_20_4_6A_3_/results/pinned_hypercube_rejection/parity_calibration/parity_calibration_predictions.jls"],
 inputs  = [
   ("models/Al_20_4_6A_3_/Al_20_4_6A_3.json", "scripts/model_building/build_model.jl", "fitted ACE model, 1829 params"),
   ("models/Al_20_4_6A_3_/lin_params.csv",    "scripts/model_building/build_model.jl", "mean-fit coefficients"),
   ("models/Al_20_4_6A_3_/results/pinned_hypercube_rejection/samples_naive.csv",
    "scripts/uq/pinned_hypercube_rejection_Al_20_4_6A_3.jl", "50 members, no rejection"),
   ("models/Al_20_4_6A_3_/results/pinned_hypercube_rejection/samples_rejected.csv",
    "scripts/uq/pinned_hypercube_rejection_Al_20_4_6A_3.jl", "50 members, predicate applied"),
   ("data/Al/manual_df_test_Al.xyz", :external, "held-out test set, ~18 MB"),
   ("scripts/bandpath_phonon_uq/lib.jl", :repo, "included helper library"),
   ("scripts/uq/lib_parity_calibration.jl", :repo, "shared parity/calibration plotting"),
 ],
 notes   = """
   RUN THE pinned_rejection_phonons ENTRY FIRST — this plots the committees it writes.

   Same plotting code as the Al_12 figures: both scripts include
   scripts/uq/lib_parity_calibration.jl, so the two sets cannot drift apart in styling.

   Both committees use point_params = lin_params, so the point prediction is identical
   and the deviation histograms are directly comparable.  Read it as:
     RMSE      should barely move — same mean model on both sides.
     COVERAGE  is the number.  Rejection removes members, so the envelope narrows and
               coverage can only fall; how far it falls is what the constraint costs.
   Coverage is committee min/max envelope containment, NOT a calibrated interval — a
   relative measure between the two columns, not an absolute calibration claim.

   Fast model load: skips the 5.2 GB A.csv, which committee_predictions does not need.
   Second positional argument "full" forces the real loader; first is the test stride.
   """,
),

(
 id      = "thermal_expansion_aT",
 title   = "Lattice constant a(T) under NPT for the constrained (multi-volume) member "*
           "and the unconstrained worst member — scatter with a linear fit, α quoted",
 script  = "scripts/uq/replot_thermal_expansion_aT_Al_12_4_6A_2.jl",
 cmd     = "julia --project scripts/uq/replot_thermal_expansion_aT_Al_12_4_6A_2.jl",
 env     = ["FIGW" => "260  # half of \\linewidth — these are built for a side-by-side pair"],
 outputs = ["models/Al_12_4_6A_2_/results/npt_multivolume_softest/thermal_expansion_aT.pdf",
            "models/Al_12_4_6A_2_/results/npt_multivolume_softest/thermal_expansion_aT.png",
            "models/Al_12_4_6A_2_/results/npt_thermal_expansion_naive_worst_member/thermal_expansion_aT.pdf",
            "models/Al_12_4_6A_2_/results/npt_thermal_expansion_naive_worst_member/thermal_expansion_aT.png"],
 inputs  = [
   ("models/Al_12_4_6A_2_/results/npt_multivolume_softest/thermal_expansion_summary.csv",
    "scripts/uq/npt_multivolume_member_Al_12_4_6A_2.jl",
    "ARCHIVED NPT result — see notes, the driver's output directory has since changed"),
   ("models/Al_12_4_6A_2_/results/npt_thermal_expansion_naive_worst_member/thermal_expansion_summary.csv",
    "scripts/uq/npt_thermal_expansion_worst_member_Al_12_4_6A_2.jl",
    "ARCHIVED NPT result — see notes"),
 ],
 notes   = """
   REPLOT ONLY.  No molecular dynamics is repeated: the script reads the two
   thermal_expansion_summary.csv files the NPT runs already wrote and rewrites
   thermal_expansion_aT.{pdf,png} in place, so the curves are the published numbers.
   Filenames are unchanged, so existing \\includegraphics keep resolving.

   REPRODUCIBILITY GAP, stated rather than papered over.  Neither NPT driver in the
   current tree writes to the directory its summary lives in — npt_multivolume_member
   now writes to npt_multivolume_a_eq_con_then_rejection, and
   npt_thermal_expansion_worst_member to npt_thermal_expansion_worst_member.  The two
   summaries here are archived output of earlier configurations of those scripts.  The
   FIGURES are fully reproducible from the CSVs; regenerating the CSVs themselves needs
   the earlier driver settings, which are not recorded in the repository.  Ship the CSVs
   in the data archive — they are under 1 KB each.

   Scatter with error bars, ONE linear least-squares fit, alpha quoted top left.  Nothing
   else on the axes — the caveats below belong in the caption, not on the plot.

   WHICH POINTS ARE FITTED.  `fit_exclude` in the script's `runs` tuple lists
   temperatures that are PLOTTED but not FITTED.  The constrained run's 900 K point is
   excluded: still_fcc=false with mean_coord 9.12 against 12 for FCC, so the structure
   has transformed and fitting it would turn alpha into a number about a solid-solid
   transition.  The fit line is drawn only across the fitted range, so which points it
   covers is visible without an annotation.  alpha is computed from THAT fit
   (slope/intercept), so the quoted number always matches the line drawn; the value the
   producing run recorded goes to stdout for comparison, not onto the figure.

   FOR THE CAPTION, not the axes: the unconstrained worst member sits near -11 THz
   throughout and has left FCC by 300 K, so its alpha describes a transformed structure
   rather than thermal expansion of an FCC crystal.  Its summary carries no still_fcc
   column, so nothing in the data flags this — it has to be said in the text.

   SIZING.  Built at FIGW = 260 pt, half of \\linewidth, because the two are meant to sit
   side by side; the panel is near-square.  Place them at NATURAL SIZE in a
   0.5\\linewidth minipage or subfigure — passing width= to \\includegraphics reintroduces
   exactly the scaling this avoids.  FIGW=540 for a single full-width plot.

   Columns are parsed BY NAME from the header line.  The two files do not carry the same
   columns — only the constrained one has mean_coord / median_nn_Ang / still_fcc — so
   positional indexing would silently mis-plot one of them.
   """,
),

(
 id      = "surface_energy",
 title   = "Surface energy of Al(001) across the unconstrained and constrained POPS "*
           "committees, full relaxation per member",
 script  = "scripts/qoi/surface_energy_vacuum.jl",
 cmd     = "julia --project -t 40 scripts/qoi/surface_energy_vacuum.jl",
 env     = ["ELEMENT" => "Al", "N_SUPER" => "4", "VACUUM" => "12.0  # Å, must exceed the cutoff",
            "NORMAL" => "3    # cell vector stretched → (001)",
            "QOI_THREADS" => "", "NBINS" => "10", "FIGW" => "540"],
 outputs = ["models/Al_12_4_6A_2_/results/surface_energy/surface_energy.pdf",
            "models/Al_12_4_6A_2_/results/surface_energy/surface_energy.png",
            "models/Al_12_4_6A_2_/results/surface_energy/surface_energy_unconstrained.csv",
            "models/Al_12_4_6A_2_/results/surface_energy/surface_energy_constrained.csv",
            "models/Al_12_4_6A_2_/results/surface_energy/surface_energy.jls"],
 inputs  = [
   ("models/Al_12_4_6A_2_/Al_12_4_6A_2.json", "scripts/model_building/build_model.jl", "fitted ACE model"),
   ("models/Al_12_4_6A_2_/lin_params.csv",    "scripts/model_building/build_model.jl", "central model of the unconstrained committee"),
   ("models/Al_12_4_6A_2_/results/bandpath_undotted_ncell4_densek/theta_mean.csv",
    "scripts/uq/bandpath_committee_undotted_Al_12_4_6A_2_ncell4_densek.jl",
    "central model of the constrained committee"),
   ("models/Al_12_4_6A_2_/results/naive_vs_constrained/samples_naive.csv",
    "scripts/uq/naive_vs_constrained_fullcloud_Al_12_4_6A_2.jl", "unconstrained committee"),
   ("models/Al_12_4_6A_2_/results/cutting_plane_full_cloud/committee_rejection_full_cloud.csv",
    "scripts/uq/hypercube_full_cloud_bands_Al_12_4_6A_2.jl", "constrained committee"),
 ],
 notes   = """
   gamma = (E_slab - E_bulk) / 2A, obtained by stretching one cell vector of the relaxed
   bulk supercell by VACUUM so the periodic images separate, then relaxing positions at
   fixed cell.  Same atoms and same count on both sides, so the reference cancels
   exactly; the 2 is because periodicity gives two free surfaces.

   THIS IS THE WELL-CONDITIONED QoI, and it validates the model.  Central models give
   +0.861 J/m2 (lin_params) and +0.946 (theta_mean) against ~0.9-1.0 for Al(001) in DFT,
   with relaxation lowering gamma by ~0.02 J/m2 as it must.  All 60 committee members
   relaxed downhill.

   THE RESULT WORTH QUOTING is not the variance ratio (sigma_con/sigma_uncon = 0.60) but
   the count of unphysical members: 15/30 of the unconstrained committee predict a
   NEGATIVE surface energy -- a crystal that spontaneously cleaves -- against 3/30
   constrained.  The constrained mean is 1.004 J/m2, on the DFT value; the unconstrained
   mean is -0.215.  gamma is in no predicate, so this is the prior removing members that
   are unphysical rather than merely uncertain, in an observable it never touched.

   Geometry is checked, not assumed: VACUUM must exceed the model cutoff (else the two
   surfaces interact across the gap) and the slab must exceed 2x cutoff (else through the
   material).  At N_SUPER=4 the slab is 16.2 A against 12 A and both pass; the script
   warns rather than leaving it to the reader.
   """,
),

(
 id      = "vacancy_formation",
 title   = "Vacancy formation energy across the unconstrained and constrained POPS "*
           "committees, full minimisation per member",
 script  = "scripts/qoi/vacancy_formation.jl",
 cmd     = "julia --project -t 40 scripts/qoi/vacancy_formation.jl && " *
           "julia --project scripts/qoi/plot_vacancy_formation.jl",
 env     = ["ELEMENT" => "Al", "N_SUPER" => "4  # 4-atom cubic cell → 255-atom vacancy cell",
            "QOI_THREADS" => "", "NBINS" => "10", "XCLIP" => "1", "FIGW" => "540"],
 outputs = ["models/Al_12_4_6A_2_/results/vacancy_formation/vacancy_formation.pdf",
            "models/Al_12_4_6A_2_/results/vacancy_formation/vacancy_formation_hist.pdf",
            "models/Al_12_4_6A_2_/results/vacancy_formation/vacancy_formation_hist_unconstrained.pdf",
            "models/Al_12_4_6A_2_/results/vacancy_formation/vacancy_formation_hist_constrained.pdf",
            "models/Al_12_4_6A_2_/results/vacancy_formation/vacancy_formation_unconstrained.csv",
            "models/Al_12_4_6A_2_/results/vacancy_formation/vacancy_formation_constrained.csv",
            "models/Al_12_4_6A_2_/results/vacancy_formation/vacancy_formation.jls"],
 inputs  = [
   ("models/Al_12_4_6A_2_/Al_12_4_6A_2.json", "scripts/model_building/build_model.jl", "fitted ACE model"),
   ("models/Al_12_4_6A_2_/lin_params.csv",    "scripts/model_building/build_model.jl", "central model of the unconstrained committee"),
   ("models/Al_12_4_6A_2_/results/bandpath_undotted_ncell4_densek/theta_mean.csv",
    "scripts/uq/bandpath_committee_undotted_Al_12_4_6A_2_ncell4_densek.jl",
    "central model of the constrained committee"),
   ("models/Al_12_4_6A_2_/results/naive_vs_constrained/samples_naive.csv",
    "scripts/uq/naive_vs_constrained_fullcloud_Al_12_4_6A_2.jl", "unconstrained committee"),
   ("models/Al_12_4_6A_2_/results/cutting_plane_full_cloud/committee_rejection_full_cloud.csv",
    "scripts/uq/hypercube_full_cloud_bands_Al_12_4_6A_2.jl", "constrained committee"),
   ("scripts/qoi/plot_vacancy_formation.jl", :repo, "histogram replotter, second half of `cmd`"),
 ],
 notes   = """
   E_f = E_vac - (N-1) * E_coh, each member relaxing its OWN bulk cell (variable cell)
   and its OWN vacancy supercell (fixed cell).  N_SUPER=4 on the 4-atom conventional cell
   gives a 255-atom vacancy cell.

   DO NOT QUOTE THE ABSOLUTE VALUE.  The model gives E_f < 0 -- around -0.87 eV against
   ~+0.67 eV for Al -- and it is already negative BEFORE any relaxation, at ideal
   geometry, so the minimiser is not the cause.  Two other explanations were checked and
   rejected: the per-atom bulk energy is identical to machine precision across
   4/32/108/256-atom cells, and referencing the same supercell instead of the 4-atom cell
   changes nothing.

   The cause is CONDITIONING, not the model being broken -- see the surface_energy entry,
   where the same model lands on the DFT value.  E_f multiplies the per-atom cohesive
   energy by N-1 = 255, so a 4.7 meV/atom error is enough to flip its sign, which is an
   unremarkable error for any MLIP.  gamma has no such amplification: E_bulk enters once.
   Quote the surface energy as the validated QoI and treat E_f as a spread comparison
   only.

   HISTORY, so old numbers are not mistaken for current ones: an earlier run had
   `* (4,4,4)` applied to sys_bulk BEFORE the supercell step, giving a 256-atom "bulk"
   and a 16383-atom vacancy cell.  Any result quoting sigma_con/sigma_uncon = 0.023 came
   from that run and should not be used.
   """,
),

]

figure_by_id(id) = (i = findfirst(f -> f.id == id, FIGURES);
                    i === nothing ? error("unknown figure id: $id\nknown: " *
                                          join([f.id for f in FIGURES], ", ")) : FIGURES[i])
