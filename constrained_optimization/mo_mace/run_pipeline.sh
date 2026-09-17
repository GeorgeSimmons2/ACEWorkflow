#!/bin/bash
# Constrain_Mb_Elastics.pdf — MACE-MPA-0 readout QP with Mo elastic-constant constraints.
# Steps and env vars: see README.md in this directory.
set -euo pipefail
REPO=${REPO:-/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${PY:-$REPO/python/mace_venv/bin/python}
export REPO

"$PY" "$HERE/00_fetch_mlearn_mo.py"
"$PY" "$HERE/01_build_design_and_constraints.py"
julia --project="$REPO" "$HERE/02_constrained_readout_qp.jl"
"$PY" "$HERE/03_verify_corrected_model.py"
