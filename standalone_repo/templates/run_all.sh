#!/bin/bash
# run_all.sh — every way to run this repository.
#
#   bash run_all.sh replot        rebuild all five figures from the SAVED parameters (tier A only)
#   bash run_all.sh constrain     re-derive the parameters: W QP, Al_12 chain, Al_16 chain
#   bash run_all.sh npt-saved     NPT on the PUBLISHED θ — the route that reproduces the figures
#   bash run_all.sh npt-rebuilt   NPT on freshly rebuilt committees — a different member, so
#                                 a(T) and the FCC figure will NOT match the paper
#   bash run_all.sh all           constrain + both NPT routes, SLURM-chained
#   bash run_all.sh figures       rebuild all four FROM the rerun (NPT_TAG=saved|e2e)
#   bash run_all.sh verify        compare what exists against the published θ    (seconds)
#
# The fifth, RLS vs constrained RLS at 200 K, has its own runner and rerun mode:
#   bash scripts/uq/rls_constrained_pops_Al_12/run_200K.sh all
#
# `replot` is the one that works on a fresh clone. Three of the four paper figures need only
# committed tier-A data; fcc_compare also evaluates phonons, so it needs the Al_12 design
# matrix from tier B. Start with replot.
#
# Everything else needs tier B:  bash scripts/fetch_data.sh
#
# WHICH NPT ROUTE.  Committee members are not reproducible run to run (a cutting-plane
# cascade amplifies any solver difference), so rebuilding the committee selects a DIFFERENT
# softest member: measured alpha 2.7 -> 3.6 x10^-5/K. `npt-saved` reruns the published theta
# vectors and reproduces the figures; `npt-rebuilt` is the honest end-to-end run and will
# not match the paper. Both are kept, and they write to separate directories.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO
MODE=${1:-help}

need_tierB() {
  if [ ! -f "$REPO/models/Al_12_4_6A_2_/A.csv" ]; then
    echo "Tier B is not present — '$MODE' needs the design matrices."
    echo "Run:  bash scripts/fetch_data.sh"
    exit 1
  fi
}

case "$MODE" in

replot)
  echo "── rebuilding every figure from the saved parameters ──────────────"
  cd "$REPO"
  echo "  [1/3] bands_four_panel"
  julia --project bands_four_panel/plot_four_panel_bands.jl
  echo "  [2/3] eos_repulsive_core   (needs data/W/df_W_train.extxyz from tier B for the"
  echo "        lower histogram panel; the upper EOS panel needs only tier A)"
  julia --project -t "${THREADS:-8}" eos_repulsive_core/eos_with_pair_hist.jl
  echo "  [3/4] thermal_expansion_vs_experiment — a(T) against Wilson 1941"
  julia --project thermal_expansion_vs_experiment/plot_thermal_expansion_vs_experiment.jl
  echo "  [4/4] fcc_compare — still FCC at 300 K?  (needs tier B: the model design matrix)"
  julia --project -t "${THREADS:-8}" thermal_expansion_vs_experiment/fcc_compare_constrained_vs_naive.jl
  echo "  [5/5] RLS vs constrained RLS at 200 K  (tier A; builds two 4×4×4 Hessians if not cached)"
  bash scripts/uq/rls_constrained_pops_Al_12/run_200K.sh replot
  ;;

constrain|reproduce)
  need_tierB
  bash "$REPO/constrained_optimization/run_pipeline.sh" constrain
  echo
  echo "When those finish:  bash run_all.sh verify"
  ;;

npt-saved)
  need_tierB
  bash "$REPO/npt_trajectories/run_pipeline.sh" reproduce
  ;;

npt-rebuilt)
  need_tierB
  bash "$REPO/constrained_optimization/run_pipeline.sh" npt
  ;;

all)
  need_tierB
  bash "$REPO/constrained_optimization/run_pipeline.sh" constrain
  bash "$REPO/npt_trajectories/run_pipeline.sh" reproduce
  bash "$REPO/constrained_optimization/run_pipeline.sh" npt
  ;;

figures)
  bash "$REPO/constrained_optimization/run_pipeline.sh" figures
  ;;

verify)
  julia --project "$REPO/constrained_optimization/verify/compare_all.jl"
  ;;

*)
  sed -n '2,18p' "$0"
  ;;
esac
