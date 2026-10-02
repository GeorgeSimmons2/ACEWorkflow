#!/bin/bash
# build_pdf.sh — Markdown to PDF for the notes in this directory.
#
# pandoc is not installed system-wide on the cluster; the pypandoc_binary wheel ships its own
# binary, so it lives in the project venv:
#
#     python/mace_venv/bin/pip install pypandoc_binary
#
# xelatex (/usr/bin/xelatex, or `module load texlive/2026`) is the engine, because the notes are
# full of Unicode (Å, δΘ, ᵀ, ⁻¹, ‖) that pdflatex cannot set.  DejaVu covers all of it.
#
# ```latex fences are rewritten to display math first, so the formulas typeset instead of
# appearing as code.  Everything else is left alone.
#
# Usage:  bash constrained_optimization/mo_mace/build_pdf.sh [file.md ...]     (default: EXPLAINER.md)
# Env:    PY (python with pypandoc_binary)  ENGINE (default xelatex)

set -euo pipefail
REPO=${REPO:-/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${PY:-$REPO/python/mace_venv/bin/python}
ENGINE=${ENGINE:-xelatex}
PANDOC=$("$PY" -c 'import pypandoc; print(pypandoc.get_pandoc_path())')

for md in "${@:-$HERE/EXPLAINER.md}"; do
  pdf="${md%.md}.pdf"
  tmp="$(mktemp -t mdpdf.XXXXXX.md)"
  # ```latex ... ``` -> $$ ... $$  (display math)
  awk '
    /^```latex$/ { print "$$"; inmath = 1; next }
    inmath && /^```$/ { print "$$"; inmath = 0; next }
    { print }
  ' "$md" > "$tmp"

  "$PANDOC" "$tmp" -o "$pdf" \
    --pdf-engine="$ENGINE" \
    --from=markdown+pipe_tables+tex_math_dollars \
    --toc --toc-depth=2 \
    -V geometry:margin=2.2cm \
    -V mainfont='DejaVu Serif' \
    -V sansfont='DejaVu Sans' \
    -V monofont='DejaVu Sans Mono' \
    -V fontsize=10pt \
    -V colorlinks=true \
    -V linkcolor=black -V urlcolor=blue \
    -M title="$(sed -n 's/^# //p' "$md" | head -1)"
  rm -f "$tmp"
  echo "$pdf"
done
