# standalone_repo/manifest.jl
#
# What goes into the standalone reproduction repository, and in which tier.
#
# TIER A — committed.  Everything needed to rebuild every figure in the paper FROM SAVED
#          PARAMETERS.  Small enough to live in git (~25 MB): models without their design
#          matrices, every θ vector, the serialised band curves, the NPT summaries.
#          A reviewer who clones and runs `run_all.sh replot` needs nothing else.
#
# TIER B — archive.  Everything needed to RE-DERIVE those parameters: the design matrices
#          (7.4 GB of the 8 GB total), the 73k-member constrained cloud, the cached
#          Hessians, and the DFT datasets.  Fetched separately and checksummed.
#
# `rebuildable = true` marks a tier-B entry that is a cache, not a source: it can be
# regenerated from tier A + the rest of tier B, so a fetch may legitimately skip it.
#
# `producer` is the script that makes the file, or :external for a data root that cannot
# be regenerated from this repository at all.

# Code entries are whole directories, copied recursively, so anything living inside one
# travels with it whether or not it belongs to this paper.  EXCLUDE is pruned from the
# target after copying.  mo_mace/ is a separate MACE-MPA-0 / Mo elastic-constants project
# that shares the constrained_optimization/ directory and is nothing to do with the four
# figures; it was swept in by an earlier build.
const EXCLUDE = ["constrained_optimization/mo_mace"]

const MODELS = ["Al_12_4_6A_2_", "Al_16_4_6A_3_", "W_20_4_5A_3"]

entry(path, tier, producer, note; rebuildable=false) =
    (; path, tier, producer, note, rebuildable)

const FILES = [

# ══ code: the ACEWorkflow package and the pipeline ═══════════════════════════
entry("src", :code, :repo, "the ACEWorkflow package; load_model resolves models/ via @__DIR__, so it relocates"),
entry("Project.toml",  :code, :repo, "Julia 1.11.8"),
entry("Manifest.toml", :code, :repo, "pinned dependency versions — do not regenerate"),
entry("scripts/bandpath_phonon_uq/lib.jl",  :code, :repo, "band paths, undotted Hessians, committee plotting"),
entry("scripts/uq/lib_parity_calibration.jl", :code, :repo, "shared parity/calibration plotting"),
entry("scripts/model_building/build_model.jl", :code, :repo, "refits a model from a dataset — a refit, not a byte-identical restore"),
entry("constrained_optimization", :code, :repo, "the parameter-producing stages"),
entry("npt_trajectories",         :code, :repo, "constrain → MD → figure for the thermal expansion figure"),
entry("bands_four_panel",         :code, :repo, "the 2×2 phonon band figure"),
entry("eos_repulsive_core",       :code, :repo, "the W equation-of-state figure"),
entry("thermal_expansion_vs_experiment", :code, :repo, "a(T) against Wilson 1941"),
entry("standalone_repo", :code, :repo,
      "the builder that produces this repository: build_repo.jl, the hand-written tier " *
      "manifest, the archive script and the README/run_all templates.  Shipped so the " *
      "repository can be regenerated or updated without the original working tree"),

# ══ code: RLS vs constrained RLS at 200 K (fifth figure) ══════════════════════
# Individual files, not the directory: the working folder also holds exploration runs
# (±2%, POPS comparison, 400/600 K) that are not part of this figure.
[entry("scripts/uq/rls_constrained_pops_Al_12/$f", :code, :repo, n) for (f, n) in (
    ("README.md",                        "steps a)–e), plot data, caption notes"),
    ("run_200K.sh",                      "replot | all (SLURM-chained into repro_*)"),
    ("compare_rls_Al_12_4_6A_2.jl",      "a) constrained RLS: ω > 0 over a_eq·(1.00:0.02:1.10), 5×5×5"),
    ("run_constrain.slurm",              "a) on hmem"),
    ("run_npt.slurm",                    "b) 200 K NPT for MODEL = rls | constrained"),
    ("fig_200K_rls_vs_constrained.jl",   "d)+e) phonons at a(200 K), figure, plot data"))]...,
entry("npt_al16/npt_member_Al_16.jl", :code, :repo,
      "b) the NPT driver (MODELDIR selects the model; same protocol as the paper's NPT)"),

# ══ tier A: models without their design matrices ═════════════════════════════
[entry("models/$m/$f", :A,
       "scripts/model_building/build_model.jl",
       f == "lin_params.csv" ? "mean-fit coefficients" :
       f == "P.csv"          ? "preconditioner"        :
       f == "W.csv"          ? "row weights"           :
       f == "Y.csv"          ? "targets"               : "fitted ACE model")
 for m in MODELS
 for f in ("P.csv", "W.csv", "Y.csv", "lin_params.csv")]...,
entry("models/Al_12_4_6A_2_/Al_12_4_6A_2.json", :A, "scripts/model_building/build_model.jl", "fitted ACE model"),
entry("models/Al_16_4_6A_3_/Al_16_4_6A_3.json", :A, "scripts/model_building/build_model.jl", "fitted ACE model"),
entry("models/W_20_4_5A_3/W_20_4_5A_3.json",    :A, "scripts/model_building/build_model.jl", "fitted ACE model"),

# ══ tier A: the constrained parameters themselves ════════════════════════════
entry("models/W_20_4_5A_3/positive_core_constrained_parameters.csv", :A,
      "constrained_optimization/w/01_positive_core_qp.jl",
      "the W positive-core θ — the EOS figure's constrained curve.  Had NO producer until that script"),

[entry("models/Al_12_4_6A_2_/results/bandpath_undotted_ncell4_densek/$f", :A,
       "constrained_optimization/al12/01_committee_ncell4_densek.jl",
       "the a_eq-pinned committee stages 3 and 4 both read")
 for f in ("theta_mean.csv", "committee_repaired.csv", "committee_rejection.csv")]...,

entry("models/Al_12_4_6A_2_/results/cutting_plane_full_cloud/committee_rejection_full_cloud.csv", :A,
      "constrained_optimization/al12/03_hypercube_rejection.jl",
      "30 constrained members — the TOP-RIGHT panel of bands_four_panel"),
entry("models/Al_12_4_6A_2_/results/cutting_plane_full_cloud/hypercube_summary.csv", :A,
      "constrained_optimization/al12/03_hypercube_rejection.jl", "acceptance rates, retained directions"),

[entry("models/Al_12_4_6A_2_/results/naive_vs_constrained/$f", :A,
       "constrained_optimization/al12/04_naive_ensemble.jl",
       "the UNCONSTRAINED ensemble — the TOP-LEFT panel")
 for f in ("samples_naive.csv", "min_freq_naive.csv", "min_freq_constrained.csv",
           "naive_vs_constrained.jls")]...,
entry("models/Al_12_4_6A_2_/results/naive_vs_constrained/bands_four_panel_Al_12.jls", :A,
      "bands_four_panel/build_bands_cache_Al_12.jl",
      "the Al_12 band curves.  Rebuildable, but only from tier B — 31 native Hessians"),

# ── the two NPT committees (a separate pair of runs from al12/01) ────────────
[entry("models/Al_12_4_6A_2_/results/bandpath_undotted_multivolume/$f", :A,
       "npt_trajectories/committee_constrained_multivolume.jl",
       "multi-volume committee: phonon stability from a_eq out to 1.1·a_eq")
 for f in ("theta_mean.csv", "theta_npt_softest.csv", "theta_npt_median.csv",
           "committee_repaired.csv", "committee_rejection.csv", "npt_candidates.csv",
           "minomega_by_volume_naive.csv", "minomega_by_volume_repaired.csv",
           "minomega_by_volume_rejection.csv")]...,
[entry("models/Al_12_4_6A_2_/results/bandpath_undotted/$f", :A,
       "npt_trajectories/committee_aeq.jl",
       "a_eq-only committee — the NPT unconstrained arm selects its worst member from here")
 for f in ("theta_mean.csv", "committee_repaired.csv", "committee_rejection.csv")]...,

# ── the NPT results the thermal-expansion figure actually plots ──────────────
[entry("models/Al_12_4_6A_2_/results/npt_multivolume_softest/$f", :A,
       "npt_trajectories/npt_constrained_softest.jl", "the constrained (blue) NPT arm")
 for f in ("thermal_expansion_summary.csv", "theta_used.csv",
           "coordination_300K.csv", "rdf_300K.csv", "PROVENANCE.md")]...,
[entry("models/Al_12_4_6A_2_/results/npt_thermal_expansion_naive_worst_member/$f", :A,
       "npt_trajectories/npt_unconstrained_naive_worst.jl", "the unconstrained (red) NPT arm")
 for f in ("thermal_expansion_summary.csv", "theta_naive_worst.csv", "theta_con_soft.csv",
           "coordination_300K.csv", "rdf_300K.csv")]...,

# ── Al_16, the bottom row ───────────────────────────────────────────────────
entry("models/Al_16_4_6A_3_/results/pinned_ensembles/bands_two_ensembles.jls", :A,
      "constrained_optimization/al16/02_bands_two_ensembles.jl",
      "the Al_16 band curves the four-panel figure reads"),
[entry("models/Al_16_4_6A_3_/results/pinned_ensembles/bands_$t.csv", :A,
       "constrained_optimization/al16/02_bands_two_ensembles.jl",
       "per-member (relaxed a, min ω).  The ONLY surviving record of the published " *
       "20-member ensembles — their ensemble_*.csv were overwritten on 19 Aug")
 for t in ("constrained", "unconstrained")]...,

# The published figures themselves, so a rerun can be compared against what is in the
# paper without going back to the original tree.  bands and a(T) already travel with
# their own directories; these two live under models/ and were being left behind.
[entry("models/W_20_4_5A_3/results/eos_with_pair_hist.$ext", :A,
       "eos_repulsive_core/eos_with_pair_hist.jl", "the published W EOS figure")
 for ext in ("pdf", "png")]...,
[entry("models/Al_12_4_6A_2_/results/fcc_compare_constrained_vs_naive_300K.$ext", :A,
       "thermal_expansion_vs_experiment/fcc_compare_constrained_vs_naive.jl",
       "the published FCC-comparison figure")
 for ext in ("pdf", "png")]...,

entry("thermal_expansion_vs_experiment/wilson_1941_aluminium.csv", :A, :external,
      "Wilson 1941, Proc. Phys. Soc. 53 235 — the experimental a(T)"),

# ══ tier A: RLS vs constrained RLS at 200 K ══════════════════════════════════
entry("scripts/uq/rls_constrained_pops_Al_12/results_rls/theta_rls_constrained.csv", :A, "scripts/uq/rls_constrained_pops_Al_12/compare_rls_Al_12_4_6A_2.jl",
      "the constrained RLS θ — the blue model"),
entry("scripts/uq/rls_constrained_pops_Al_12/results_rls/bands_rls_vs_constrained.png", :A, "scripts/uq/rls_constrained_pops_Al_12/compare_rls_Al_12_4_6A_2.jl",
      "RLS vs constrained bands at each constrained volume"),
[entry("scripts/uq/rls_constrained_pops_Al_12/npt/$m/$f", :A, f in ("rdf_200K.csv", "coordination_200K.csv") ?
       "thermal_expansion_vs_experiment/make_rdf_coordination.jl" : "npt_al16/npt_member_Al_16.jl",
       "200 K NPT, $m")
 for m in ("rls", "constrained")
 for f in ("theta_used.csv", "a0.csv", "T200K/summary_row.csv", "rdf_200K.csv", "coordination_200K.csv")]...,
[entry("scripts/uq/rls_constrained_pops_Al_12/fig_200K/$f", :A, "scripts/uq/rls_constrained_pops_Al_12/fig_200K_rls_vs_constrained.jl",
       endswith(f, ".csv") ? "plot data" : "the figure")
 for f in ("fig_200K_rls_vs_constrained.pdf", "fig_200K_rls_vs_constrained.png", "metadata.csv",
           "bands_rls.csv", "bands_constrained.csv", "band_ticks.csv", "rdf_rls.csv", "rdf_constrained.csv")]...,

# ══ tier B: the 200 K trajectories — needed only to recompute the RDFs ═══════
[entry("scripts/uq/rls_constrained_pops_Al_12/npt/$m/T200K/md_trajectory.extxyz", :B, "npt_al16/npt_member_Al_16.jl",
       "200 K NPT trajectory, $m, 8.7 MB") for m in ("rls", "constrained")]...,

# ══ tier B: design matrices.  7.4 GB of the 8 GB total ═══════════════════════
entry("models/Al_12_4_6A_2_/A.csv", :B, "scripts/model_building/build_model.jl", "259 MB"),
entry("models/Al_16_4_6A_3_/A.csv", :B, "scripts/model_building/build_model.jl", "1.9 GB"),
entry("models/W_20_4_5A_3/A.csv",   :B, "scripts/model_building/build_model.jl", "5.2 GB"),

# ══ tier B: the constrained cloud ════════════════════════════════════════════
entry("models/Al_12_4_6A_2_/results/cutting_plane_full_cloud/committee_stable.jls", :B,
      "constrained_optimization/al12/02_cutting_plane_full_cloud.jl",
      "73,411 cutting-plane-constrained stable members, 55 MB"),
entry("models/Al_12_4_6A_2_/results/cutting_plane_full_cloud/member_diagnostics.csv", :B,
      "constrained_optimization/al12/02_cutting_plane_full_cloud.jl",
      "per-member cuts / iters / min ω / status — what `verify` compares"),

# ══ tier B: cached Hessians and band paths — regenerable ═════════════════════
entry("models/Al_12_4_6A_2_/results/undotted_Hbasis_4x4x4_a4.04494.jls", :B,
      "constrained_optimization/al12/01_committee_ncell4_densek.jl",
      "429 MB undotted per-basis Hessian at a_eq.  Filenames round a to 5 digits but are " *
      "BUILT at the unrounded a — read cache.a_eq, never the filename";
      rebuildable=true),
entry("models/Al_16_4_6A_3_/results/pinned_ensembles/bandpath_4x4x4_aref.jls", :B,
      "constrained_optimization/al16/01_pinned_rejection_ensembles.jl",
      "3-row undotted band path, 13.9 MB"; rebuildable=true),

# ══ tier B: the DFT datasets.  Cannot be regenerated from this repo ══════════
entry("data/Al/manual_df_train_Al.extxyz", :B, :external, "Al training set"),
entry("data/Al/manual_df_test_Al.xyz",     :B, :external, "Al held-out test set, used for parity/calibration"),
entry("data/W/df_W_train.extxyz",          :B, :external, "W training set — the EOS figure's pair-distance histogram"),
]

tier(t) = [f for f in FILES if f.tier === t]
