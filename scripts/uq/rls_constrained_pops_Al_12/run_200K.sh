#!/bin/bash
# run_200K.sh — the 200 K RLS vs constrained RLS figure (fig_200K/), every step.
#
#   bash scripts/uq/rls_constrained_pops_Al_12/run_200K.sh replot   RDF + phonons + figure from the SAVED runs (minutes)
#   bash scripts/uq/rls_constrained_pops_Al_12/run_200K.sh all      re-derive everything into repro_* (SLURM-chained, ~2 h)
#
# Run from the repository root.  Steps (README.md has the details):
#   a) constrained RLS   compare_rls_Al_12_4_6A_2.jl            (RLS itself = models/Al_12_4_6A_2_/lin_params.csv)
#   b) 200 K NPT         run_npt.slurm → npt_al16/npt_member_Al_16.jl, both models, same seed
#   c) RDF               thermal_expansion_vs_experiment/make_rdf_coordination.jl  (a(200 K) comes from b)
#   d+e) phonons, figure, plot data   fig_200K_rls_vs_constrained.jl
set -euo pipefail
REPO=${REPO:-$PWD}; cd "$REPO"
D=scripts/uq/rls_constrained_pops_Al_12
[ -f "$D/run_200K.sh" ] || { echo "run from the repository root"; exit 1; }
mkdir -p "$D/logs"

case "${1:-help}" in
replot)
  for m in rls constrained; do julia --project thermal_expansion_vs_experiment/make_rdf_coordination.jl "$D/npt/$m" 200; done
  julia --project -t "${THREADS:-8}" "$D/fig_200K_rls_vs_constrained.jl"
  ;;
all)
  [ -f models/Al_12_4_6A_2_/A.csv ] || { echo "needs tier B (the design matrix): bash scripts/fetch_data.sh"; exit 1; }
  R=$REPO/$D
  J0=$(OUTDIR=$R/repro_results_rls sbatch --parsable --export=ALL "$D/run_constrain.slurm")
  J1=$(sbatch --parsable --array=1 --job-name=npt_rls_Al12 \
        --export=ALL,MODEL=rls,NPTDIR=$R/repro_npt "$D/run_npt.slurm")
  J2=$(sbatch --parsable --array=1 --job-name=npt_con_Al12 --dependency=afterok:$J0 \
        --export=ALL,MODEL=constrained,NPTDIR=$R/repro_npt,CON_THETA=$R/repro_results_rls/theta_rls_constrained.csv "$D/run_npt.slurm")
  J3=$(sbatch --parsable --job-name=fig200K_Al12 --dependency=afterok:$J1:$J2 --partition=compute --nodes=1 --ntasks=8 \
        --mem-per-cpu=3988 --time=01:00:00 --output="$D/logs/%x_%j.log" \
        --wrap="cd $REPO && for m in rls constrained; do julia --project thermal_expansion_vs_experiment/make_rdf_coordination.jl $R/repro_npt/\$m 200; done && NPT=$R/repro_npt OUTDIR=$R/repro_fig_200K julia --project -t 8 $D/fig_200K_rls_vs_constrained.jl")
  echo "constrain $J0 → NPT rls $J1 / constrained $J2 → RDF + figure $J3   (outputs under $D/repro_*)"
  ;;
*) sed -n 2,13p "$0" ;;
esac
