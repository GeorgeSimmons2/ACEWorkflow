#!/bin/bash
# run_pipeline.sh — constrain → MD → figure, end to end.
#
#   bash npt_trajectories/run_pipeline.sh <stage>
#
# ┌ WHICH STAGE DO YOU WANT? ─────────────────────────────────────────────────┐
# │ published   Reproduce the figure AS PUBLISHED.  Skips stage 1 and reuses   │
# │             the existing committees, so the parameter vectors are the      │
# │             exact ones behind the figure.  Stage 2 aborts unless the       │
# │             members it selects are byte-identical to the published ones.   │
# │                                                                            │
# │ all         Regenerate EVERYTHING from the constraints down.  This will    │
# │             not reproduce the published vectors — see "Determinism" in     │
# │             README.md — so α will shift slightly.  Use this if the paper   │
# │             needs to claim the pipeline reproduces end to end.             │
# └────────────────────────────────────────────────────────────────────────────┘
#
#   constrain            stage 1 only  — build both committees
#   md                   stage 2 only  — NPT on freshly built committees
#   figure               stage 3 only  — replot from the PUBLISHED summaries (local)
#   figure-repro         stage 3 only  — replot from THIS run's output (use after `all`)
#   compare-published    how far did regenerating the committee move the answer?
#   all                  1 → 2 → 3, chained with SLURM dependencies
#   published            stage 2 → 3 against the published committees
#   verify-determinism   run stage 1 twice into separate dirs and compare
#
# Env:  REPO (repo root)   RHO_INTERVAL (pinned OSQP rho schedule, default 25)
set -euo pipefail

REPO=${REPO:-/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow}
HERE="$REPO/npt_trajectories"
RES="$REPO/models/Al_12_4_6A_2_/results"
STAGE=${1:-help}
export REPO
export RHO_INTERVAL=${RHO_INTERVAL:-25}

# every sbatch inherits the exported environment, which is how COMMITTEE_DIR /
# COMMITTEE_OUT / THETA_REF reach the Julia scripts
submit() { sbatch --parsable --export=ALL "$@"; }

stage1() {   # -> sets JOB_MV, JOB_AEQ
  echo "── stage 1: constrain  (RHO_INTERVAL=$RHO_INTERVAL) ───────────────"
  JOB_MV=$(submit "$HERE/run_committee_constrained.slurm")
  JOB_AEQ=$(submit "$HERE/run_committee_aeq.slurm")
  echo "    multi-volume committee : job $JOB_MV → results/repro_bandpath_undotted_multivolume/"
  echo "    a_eq committee         : job $JOB_AEQ → results/repro_bandpath_undotted/"
}

case "$STAGE" in

constrain)
  stage1
  ;;

all)
  stage1
  echo "── stage 2: MD, queued after stage 1 ──────────────────────────────"
  # THETA_REF=none because a regenerated committee does not reproduce the published
  # members, so checking against them would (correctly) abort every run.
  export THETA_REF=none
  JM1=$(COMMITTEE_DIR="$RES/repro_bandpath_undotted_multivolume" \
        submit --dependency=afterok:"$JOB_MV"  "$HERE/run_npt_constrained_softest.slurm")
  JM2=$(COMMITTEE_DIR="$RES/repro_bandpath_undotted" \
        submit --dependency=afterok:"$JOB_AEQ" "$HERE/run_npt_unconstrained_naive_worst.slurm")
  echo "    constrained MD   : job $JM1"
  echo "    unconstrained MD : job $JM2"
  echo
  echo "Stage 3 is local.  Once those finish:"
  echo "    bash npt_trajectories/run_pipeline.sh figure-repro       # plot THIS run"
  echo "    bash npt_trajectories/run_pipeline.sh compare-published  # how far did it move?"
  echo
  echo "NOTE: plain \`figure\` still plots the PUBLISHED summaries.  This run writes to"
  echo "      repro_* directories and does not touch them."
  ;;

md)
  echo "── stage 2 only: MD against the freshly built committees ───────────"
  export THETA_REF=none
  COMMITTEE_DIR="$RES/repro_bandpath_undotted_multivolume" \
    submit "$HERE/run_npt_constrained_softest.slurm"
  COMMITTEE_DIR="$RES/repro_bandpath_undotted" \
    submit "$HERE/run_npt_unconstrained_naive_worst.slurm"
  ;;

published)
  echo "── stage 2 → 3 against the PUBLISHED committees ────────────────────"
  echo "    aborts unless the selected members are byte-identical to the published ones"
  COMMITTEE_DIR="$RES/bandpath_undotted_multivolume" \
    submit "$HERE/run_npt_constrained_softest.slurm"
  COMMITTEE_DIR="$RES/bandpath_undotted" \
    submit "$HERE/run_npt_unconstrained_naive_worst.slurm"
  echo
  echo "Then:  bash npt_trajectories/run_pipeline.sh figure"
  ;;

figure)
  echo "── stage 3: figure, from the PUBLISHED summaries ───────────────────"
  RESDIR=${RESDIR:-$RES} julia --project="$REPO" \
    "$REPO/thermal_expansion_vs_experiment/plot_thermal_expansion_vs_experiment.jl"
  ;;

figure-repro)
  # `all` writes to repro_* leaf names, which differ from the published ones, so the
  # plotter has to be pointed at them explicitly -- RESDIR alone will not do it.
  echo "── stage 3: figure, from THIS pipeline run's output ────────────────"
  for d in "$RES/repro_npt_thermal_expansion_naive_worst_member" "$RES/repro_npt_multivolume_softest"; do
    [ -f "$d/thermal_expansion_summary.csv" ] || {
      echo "    missing $d/thermal_expansion_summary.csv"
      echo "    stage 2 has not finished -- check squeue"; exit 1; }
  done
  DIR_UNCON="$RES/repro_npt_thermal_expansion_naive_worst_member" \
  DIR_CON="$RES/repro_npt_multivolume_softest" \
  OUT=${OUT:-"$REPO/thermal_expansion_vs_experiment/thermal_expansion_aT_vs_experiment_repro"} \
    julia --project="$REPO" \
    "$REPO/thermal_expansion_vs_experiment/plot_thermal_expansion_vs_experiment.jl"
  ;;

compare-published)
  # How far did regenerating the committee move the answer?  Members are not
  # reproducible, so this is the number that says whether that matters.
  echo "── fresh committee vs published ────────────────────────────────────"
  julia --project="$REPO" -e '
    using DelimitedFiles, Printf
    R = ARGS[1]
    pairs = [("committee (softest member)", "bandpath_undotted_multivolume/theta_npt_softest.csv",
                                            "repro_bandpath_undotted_multivolume/theta_npt_softest.csv"),
             ("committee (mean model)",     "bandpath_undotted_multivolume/theta_mean.csv",
                                            "repro_bandpath_undotted_multivolume/theta_mean.csv")]
    for (lab, a, b) in pairs
        pa = joinpath(R, a); pb = joinpath(R, b)
        (isfile(pa) && isfile(pb)) || (@printf("%-28s not available yet\n", lab); continue)
        x = readdlm(pa, Char(0x2c)); y = readdlm(pb, Char(0x2c))
        @printf("%-28s max |d| = %.4e\n", lab, maximum(abs.(x .- y)))
    end
    for (lab, a, b) in [("alpha / a(T) summary", "npt_multivolume_softest", "repro_npt_multivolume_softest")]
        pa = joinpath(R, a, "thermal_expansion_summary.csv")
        pb = joinpath(R, b, "thermal_expansion_summary.csv")
        (isfile(pa) && isfile(pb)) || (@printf("%-28s not available yet\n", lab); continue)
        println("\n-- published --");   print(read(pa, String))
        println("-- regenerated --"); print(read(pb, String))
    end' "$RES"
  ;;

verify-determinism)
  # The claim under test: pinning OSQP's rho schedule makes the constrained committee
  # reproducible.  For scale, the UNPINNED original differs between identical runs by
  # max |Δθ| = 4.31 on the softest member.  Run B is queued after A rather than
  # alongside it — two jobs sharing a node would perturb each other's timing, which is
  # the very thing under test.
  echo "── determinism check: stage 1 twice, RHO_INTERVAL=$RHO_INTERVAL ────"
  A=$(COMMITTEE_OUT="$RES/determinism_A" submit "$HERE/run_committee_constrained.slurm")
  B=$(COMMITTEE_OUT="$RES/determinism_B" submit --dependency=afterok:"$A" \
      "$HERE/run_committee_constrained.slurm")
  echo "    run A: job $A → results/determinism_A/"
  echo "    run B: job $B → results/determinism_B/  (queued after A)"
  echo
  echo "When both finish, compare:"
  echo "    bash npt_trajectories/run_pipeline.sh compare-determinism"
  ;;

compare-determinism)
  julia --project="$REPO" -e '
    using DelimitedFiles, Printf
    R = ARGS[1]
    for f in ("theta_mean.csv", "theta_npt_softest.csv", "committee_rejection.csv")
        pa = joinpath(R, "determinism_A", f); pb = joinpath(R, "determinism_B", f)
        if !isfile(pa) || !isfile(pb)
            @printf("%-26s missing (run verify-determinism first)\n", f); continue
        end
        a = readdlm(pa, Char(0x2c)); b = readdlm(pb, Char(0x2c))
        if size(a) != size(b)
            @printf("%-26s SHAPE DIFFERS %s vs %s\n", f, size(a), size(b))
        else
            @printf("%-26s max |Δ| = %.3e%s\n", f, maximum(abs.(a .- b)),
                    maximum(abs.(a .- b)) == 0 ? "   ← bit-exact" : "")
        end
    end
    println("\nFor scale: the unpinned original differs by 4.31 on theta_npt_softest.")' "$RES"
  ;;

*)
  sed -n '2,30p' "$0"
  exit 1
  ;;
esac
