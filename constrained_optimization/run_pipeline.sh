#!/bin/bash
# run_pipeline.sh — rerun every constrained-optimisation stage behind the paper's figures,
# then check what came out against what is published.
#
#   bash constrained_optimization/run_pipeline.sh <stage>
#
# ┌ START HERE ────────────────────────────────────────────────────────────────┐
# │ w        The W positive-core QP.  Cheapest, and the one that closes a real  │
# │          dead end: until this script existed, the θ behind the EOS figure   │
# │          had NO producer anywhere in the repository.  Run it first.         │
# │                                                                            │
# │ verify   Local, seconds, safe at any point.  Compares whatever has been     │
# │          produced against the published vectors and writes REPRODUCTION.md. │
# │                                                                            │
# │ all      Everything, SLURM-chained: constrain → NPT → figures → verify.     │
# │          ~12–15 h, dominated by the two NPT sweeps.                         │
# └────────────────────────────────────────────────────────────────────────────┘
#
#   w           W  : the positive-core constrained ridge regression
#   al12        Al_12 : committee → cutting-plane cloud → hypercube+rejection → naive
#   al16        Al_16 : pinned ensembles (N_MEMBERS=20, ACCEPT_TOL=-0.05) → native bands
#   constrain   w + al12 + al16, in parallel (they share nothing)
#   npt         fresh NPT committees → MD on their members (repro_e2e_npt_*)
#   figures     rebuild all three figures FROM THE repro_ TREE (local, minutes)
#   verify      compare against published, write REPRODUCTION.md  (local, seconds)
#   all         constrain → npt → figures → verify, fully chained
#   status      what is queued, and what exists on disk
#
# Env:  REPO           repo root
#       RHO_INTERVAL   pinned OSQP rho schedule (default 25; 0 = original behaviour)
#       Z_TOL          verify: sampling-noise threshold (default 4.5)
#
# NOTHING HERE OVERWRITES A PUBLISHED ARTIFACT.  Every stage writes to a repro_ directory.
# That is not politeness: the published Al_16 ensembles were destroyed by a rerun that did
# not do this, which is why the bottom row of bands_four_panel had no producer.
set -euo pipefail

REPO=${REPO:-/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow}
HERE="$REPO/constrained_optimization"
A12="$REPO/models/Al_12_4_6A_2_/results"
A16="$REPO/models/Al_16_4_6A_3_/results"
STAGE=${1:-help}
export REPO
export RHO_INTERVAL=${RHO_INTERVAL:-25}

submit() { sbatch --parsable --export=ALL "$@"; }

# ── stage groups.  Each sets JOB_* so `all` can chain on them ────────────────
stage_w() {
  JOB_W=$(submit "$HERE/w/run_positive_core_qp.slurm")
  echo "    W positive-core QP        : job $JOB_W  → models/W_20_4_5A_3/repro_positive_core_constrained_parameters.csv"
}

stage_al12() {   # 01 → 02 → 03 → 04; 03 needs BOTH 01 and 02
  JOB_C1=$(submit "$HERE/al12/run_committee.slurm")
  JOB_C2=$(submit "$HERE/al12/run_cutting_plane.slurm")
  JOB_C3=$(submit --dependency=afterok:"$JOB_C1":"$JOB_C2" "$HERE/al12/run_hypercube_rejection.slurm")
  JOB_C4=$(submit --dependency=afterok:"$JOB_C3"            "$HERE/al12/run_naive_ensemble.slurm")
  echo "    Al_12 committee (stage 1) : job $JOB_C1  → repro_bandpath_undotted_ncell4_densek/"
  echo "    Al_12 cutting plane (2)   : job $JOB_C2  → repro_cutting_plane_full_cloud/"
  echo "    Al_12 hypercube+rej (3)   : job $JOB_C3  (after 1 and 2)"
  echo "    Al_12 naive ensemble (4)  : job $JOB_C4  (after 3)  → repro_naive_vs_constrained/"
}

stage_al16() {
  JOB_E1=$(submit "$HERE/al16/run_ensembles.slurm")
  JOB_E2=$(submit --dependency=afterok:"$JOB_E1" "$HERE/al16/run_bands.slurm")
  echo "    Al_16 ensembles (stage 1) : job $JOB_E1  → repro_pinned_ensembles/  (N_MEMBERS=20, ACCEPT_TOL=-0.05)"
  echo "    Al_16 native bands (2)    : job $JOB_E2  (after 1)"
}


# NPT, end to end.  Deliberately NOT `npt_trajectories/run_pipeline.sh all`, which (a) wrote
# into repro_npt_* where the 27 Aug reproduce-mode run already lives, and (b) never cleared
# THETA_FILE, so its MD reran the published θ instead of the new committee.
# Which MD route the figures read.  "e2e" = the rebuilt-committee run (repro_e2e_npt_*);
# "saved" = the run on the published θ vectors (repro_npt_*), which is what reproduces the
# published figures, since committee members are not reproducible run to run.
NPT_TAG=${NPT_TAG:-e2e}
# PFX selects the MD directory; SFX keeps the two routes' figures from overwriting each
# other, since only the a(T) and FCC figures depend on which MD route was used.
if [ "$NPT_TAG" = saved ]; then PFX=""; SFX="_saved"; else PFX="${NPT_TAG}_"; SFX="_${NPT_TAG}"; fi
stage_npt() {
  local NT="$REPO/npt_trajectories"
  JOB_NMV=$(COMMITTEE_OUT="$A12/repro_bandpath_undotted_multivolume" \
            submit "$NT/run_committee_constrained.slurm")
  JOB_NAEQ=$(COMMITTEE_OUT="$A12/repro_bandpath_undotted" \
             submit "$NT/run_committee_aeq.slurm")
  # THETA_FILE= (empty) makes both drivers select from COMMITTEE_DIR; THETA_REF=none
  # because fresh cutting-plane members are not expected to equal the published ones.
  # Both arms still write theta_used.csv, and `verify` compares those afterwards.
  JOB_MDC=$(THETA_FILE= THETA_REF=none \
            COMMITTEE_DIR="$A12/repro_bandpath_undotted_multivolume" \
            OUTDIR="$A12/repro_${PFX}npt_multivolume_softest" \
            submit --dependency=afterok:"$JOB_NMV" "$NT/run_npt_constrained_softest.slurm")
  JOB_MDU=$(THETA_FILE= THETA_REF=none \
            COMMITTEE_DIR="$A12/repro_bandpath_undotted" \
            OUTDIR="$A12/repro_${PFX}npt_thermal_expansion_naive_worst_member" \
            submit --dependency=afterok:"$JOB_NAEQ" "$NT/run_npt_unconstrained_naive_worst.slurm")
  echo "    multi-volume committee    : job $JOB_NMV  → repro_bandpath_undotted_multivolume/"
  echo "    a_eq committee            : job $JOB_NAEQ  → repro_bandpath_undotted/"
  echo "    MD constrained (softest)  : job $JOB_MDC  (after $JOB_NMV)  → repro_${PFX}npt_multivolume_softest/"
  echo "    MD unconstrained (worst)  : job $JOB_MDU  (after $JOB_NAEQ)  → repro_${PFX}npt_thermal_expansion_naive_worst_member/"
}

case "$STAGE" in

w)         echo "── W ─────────────────────────────────────────────────────────"; stage_w ;;
al12)      echo "── Al_12 ─────────────────────────────────────────────────────"; stage_al12 ;;
al16)      echo "── Al_16 ─────────────────────────────────────────────────────"; stage_al16 ;;

constrain)
  echo "── constrain: all three chains, in parallel ──────────────────────"
  stage_w; stage_al12; stage_al16
  echo
  echo "When they finish:  bash constrained_optimization/run_pipeline.sh verify"
  ;;

npt)
  echo "── NPT end to end: fresh committees → MD on THEIR members ─────────"
  stage_npt
  ;;

figures)
  echo "── figures, rebuilt FROM THE repro_ TREE (local) ─────────────────"
  cd "$REPO"

  echo "  [1/3] bands_four_panel — rebuilding the Al_12 band cache (slow: 31 native Hessians)"
  SRC_N="$A12/repro_naive_vs_constrained" \
  SRC_C="$A12/repro_cutting_plane_full_cloud" \
  COMMITTEE_DIR="$A12/repro_bandpath_undotted_ncell4_densek" \
  OUT="$A12/repro_naive_vs_constrained/bands_four_panel_Al_12.jls" \
    julia --project -t "${THREADS:-40}" bands_four_panel/build_bands_cache_Al_12.jl

  SRC12="$A12/repro_naive_vs_constrained/bands_four_panel_Al_12.jls" \
  SRC16="$A16/repro_pinned_ensembles/bands_two_ensembles.jls" \
  OUT="$REPO/bands_four_panel/repro_bands_four_panel" \
    julia --project bands_four_panel/plot_four_panel_bands.jl

  echo "  [2/3] eos_repulsive_core — from the freshly solved W θ"
  THETA_CON="$REPO/models/W_20_4_5A_3/repro_positive_core_constrained_parameters.csv" \
  OUT="$REPO/eos_repulsive_core/repro_eos_with_pair_hist" \
    julia --project -t "${THREADS:-8}" eos_repulsive_core/eos_with_pair_hist.jl

  echo "  [3/3] thermal_expansion_vs_experiment — from the repro_ NPT summaries"
  DIR_UNCON="$A12/repro_${PFX}npt_thermal_expansion_naive_worst_member" \
  DIR_CON="$A12/repro_${PFX}npt_multivolume_softest" \
  OUT="$REPO/thermal_expansion_vs_experiment/repro_thermal_expansion_aT_vs_experiment$SFX" \
    julia --project thermal_expansion_vs_experiment/plot_thermal_expansion_vs_experiment.jl

  echo "  [4/4] fcc_compare — was the lattice still FCC when a(T) was measured?"
  # rdf_<T>K.csv / coordination_<T>K.csv have no producer in the original tree: they were a
  # side effect of two legacy FIGURE scripts, so a pipeline rerun leaves the NPT output
  # without them and this figure cannot be drawn.  Generate them from each run's own
  # trajectory first (verified byte-identical to the published files).
  for d in "$A12/repro_${PFX}npt_multivolume_softest" \
           "$A12/repro_${PFX}npt_thermal_expansion_naive_worst_member"; do
    julia --project thermal_expansion_vs_experiment/make_rdf_coordination.jl "$d" 300
  done
  DIR_UNCON="$A12/repro_${PFX}npt_thermal_expansion_naive_worst_member" \
  DIR_CON="$A12/repro_${PFX}npt_multivolume_softest" \
  OUT="$REPO/thermal_expansion_vs_experiment/repro_fcc_compare_constrained_vs_naive_300K$SFX" \
    julia --project -t "${THREADS:-8}" thermal_expansion_vs_experiment/fcc_compare_constrained_vs_naive.jl

  echo
  echo "Figures written beside the published ones with a repro_ prefix — compare, do not replace."
  echo "NPT_TAG=$NPT_TAG selected the MD route: 'e2e' = rebuilt committees, '' = the saved published θ."
  ;;

verify)
  cd "$REPO"
  julia --project "$HERE/verify/compare_all.jl"
  ;;

all)
  echo "── stage 1: constrain ────────────────────────────────────────────"
  stage_w; stage_al12; stage_al16
  echo
  echo "── stage 2: NPT committees → MD, chained ─────────────────────────"
  stage_npt
  echo
  echo "── stage 3 and 4 are LOCAL.  Once everything above finishes: ─────"
  echo "     bash constrained_optimization/run_pipeline.sh verify    # the table that matters"
  echo "     bash constrained_optimization/run_pipeline.sh figures   # rebuild from repro_"
  ;;

status)
  echo "── queue ─────────────────────────────────────────────────────────"
  squeue -u "$USER" -o "%.10i %.22j %.9T %.10M %R" 2>/dev/null || echo "  (squeue unavailable)"
  echo
  echo "── on disk ───────────────────────────────────────────────────────"
  for f in \
    "$REPO/models/W_20_4_5A_3/repro_positive_core_constrained_parameters.csv" \
    "$A12/repro_bandpath_undotted_ncell4_densek/theta_mean.csv" \
    "$A12/repro_cutting_plane_full_cloud/committee_stable.jls" \
    "$A12/repro_cutting_plane_full_cloud/committee_rejection_full_cloud.csv" \
    "$A12/repro_naive_vs_constrained/samples_naive.csv" \
    "$A16/repro_pinned_ensembles/ensemble_constrained.csv" \
    "$A16/repro_pinned_ensembles/bands_two_ensembles.jls" \
    "$A12/repro_${PFX}npt_multivolume_softest/thermal_expansion_summary.csv" \
    "$A12/repro_${PFX}npt_thermal_expansion_naive_worst_member/thermal_expansion_summary.csv" ; do
    if [ -e "$f" ]; then printf "  %-8s %s\n" "$(du -h --apparent-size "$f" | cut -f1)" "${f#$REPO/}"
    else                 printf "  %-8s %s\n" "—" "${f#$REPO/}"; fi
  done
  ;;

*)
  sed -n '2,33p' "$0"
  ;;
esac
