# Band-path phonon-UQ study

Reusable study for the phonon-stable POPS committee of a linear ACE potential,
built on the **undotted per-basis Hessian**. Kept self-contained here for the
paper.

## Idea
The ACE energy is linear in the parameters `θ`, so the position Hessian is
`H(θ) = Σ_k θ_k H_k`. The per-basis `H_k` depend only on geometry, so at a fixed
lattice constant they are built **once**; every committee member's phonon check
is then `D(q,θ) = Σ_k θ_k D_k(q)` — a matvec plus a tiny eigen per `q`. That
makes a full-Brillouin-zone phonon check cheap enough to use as a constraint /
rejection predicate across a whole committee.

Validity needs a fixed equilibrium: members are constrained to `b′·θ=0`
(stationary) and `b″·θ>0` (minimum) so the single Hessian is genuinely each
member's Hessian.

## Files
- `lib.jl` — shared machinery: undotted Hessian (parallel over atoms), band-path
  `D_k(q)`, `bands`/`min_freq_stable`/`soft_modes`, **fixed** band plotting
  (full path → acoustic branches reach 0 at Γ), parity + calibration helpers.
- `replot_and_calibration.jl` — regenerates the band plots (Γ-fixed) and the
  test-set energy/force parity + error-vs-committee-bound calibration histograms
  from the saved committee + cached Hessian. Fast (no rebuild).
- `gruneisen.jl` — mode Grüneisen along the band path and thermodynamic `γ(T)`,
  each with committee mean ± 1σ, from undotted Hessians at three volumes.
- `summary.tex` / `summary.pdf` — methods + results write-up for the paper.

The committee itself is produced by
`../uq/bandpath_committee_undotted_Al_20_4_6A_2_subset_50.jl`; the undotted
Hessian is validated against the native dotted `hessian` in
`../uq/undotted_hessian_test.jl` (matches to ~1e-13).

## Run
```bash
# committee (once; heavy build, cached ~260 MB):
julia --project -t 40 scripts/uq/bandpath_committee_undotted_Al_20_4_6A_2_subset_50.jl
# analysis (fast; loads cache + saved committee):
julia --project -t 4  scripts/bandpath_phonon_uq/replot_and_calibration.jl
julia --project -t 40 scripts/bandpath_phonon_uq/gruneisen.jl      # builds ±volume Hessians
# PDF:
cd scripts/bandpath_phonon_uq && pdflatex summary.tex
```
Outputs land in `models/Al_20_4_6A_2_subset_50_percent/results/bandpath_phonon_uq/`.
Change `dataset` at the top of each script to point at a different model.
