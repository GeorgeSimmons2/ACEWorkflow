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
#   figure               stage 3 only  — replot from the summary CSVs (local, seconds)
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
  echo "Stage 3 is local.  Once those finish, point it at the fresh outputs:"
  echo "    bash npt_trajectories/run_pipeline.sh figure"
  echo "  (set RESDIR if you want the figure built from the repro_* directories)"
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
  echo "── stage 3: figure ─────────────────────────────────────────────────"
  RESDIR=${RESDIR:-$RES} julia --project="$REPO" \
    "$REPO/thermal_expansion_vs_experiment/plot_thermal_expansion_vs_experiment.jl"
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
