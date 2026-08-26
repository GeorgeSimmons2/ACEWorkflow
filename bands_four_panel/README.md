# `bands_four_panel/` — unconstrained vs constrained POPS phonons, both models, one figure

Collapses three existing figures into a single 2×2 panel:

|                          | unconstrained (left)                         | constrained (right)                               |
| ------------------------ | -------------------------------------------- | ------------------------------------------------- |
| **top** `Al_12_4_6A_2_`  | `naive_vs_constrained/bands_naive_shared_axis.pdf` | `naive_vs_constrained/bands_constrained_shared_axis.pdf` |
| **bottom** `Al_16_4_6A_3_` | `pinned_ensembles/bands_two_ensembles_clean.pdf` (left panel) | same file, right panel                 |

Mean model in blue, **every** ensemble member in grey — stable or not. The unstable
members are no longer coloured crimson: the figure's claim is that you can see the
right column has nothing below `ω = 0` without being told which lines to look at.
Soft-mode counts are printed at run time and belong in the caption.

## Running it

```bash
# once — the Al_12 band curves have to be rebuilt (see below).  Slow: 31 native
# 4×4×4 Hessians.  Give it cores.
julia --project -t 40 bands_four_panel/build_bands_cache_Al_12.jl

# every time after that — seconds
julia --project bands_four_panel/plot_four_panel_bands.jl
```

Output: `bands_four_panel/bands_four_panel.{pdf,png}`.

## Why there is a build step

The Al_16 run serialised its band curves, so its two panels replot instantly. The
Al_12 run did not — `naive_vs_constrained.jls` holds only summary statistics (min ω,
each member's relaxed `a`, the shared y-limits). The curves exist only inside the
published PDFs, and a PDF cannot be restyled into a new panel.

`build_bands_cache_Al_12.jl` therefore **recomputes** them — but does not **re-sample**
them. Both ensembles are read back verbatim from the CSVs the original run wrote, and
each unconstrained member is rebuilt at the lattice constant that run recorded for it,
so nothing depends on an RNG seed. It then checks every member's recomputed min ω
against the saved value and refuses quietly-wrong output:

- hard error if today's `a_eq` differs from the original run's by > 1e-6 Å (the model
  on disk would not be the one behind the published figure);
- hard error if the supercell size disagrees;
- warning if any member's min ω moves by > 1e-6 THz.

Inputs it reads:

| file                                                          | what                                     |
| ------------------------------------------------------------- | ---------------------------------------- |
| `naive_vs_constrained/samples_naive.csv`                      | 30 unconstrained members (30 × 91)       |
| `naive_vs_constrained/min_freq_naive.csv` col 2               | each one's own relaxed `a`               |
| `cutting_plane_full_cloud/committee_rejection_full_cloud.csv` | 30 constrained members                   |
| `bandpath_undotted_ncell4_densek/theta_mean.csv`              | the constrained centre `θ_mean`          |
| `results/undotted_Hbasis_4x4x4_a4.04494.jls`                  | prebuilt undotted Hessian (auto-located) |

## Two evaluators, and why they are not interchangeable

- **Constrained** — one prebuilt undotted per-basis Hessian at `a_eq`. These members
  have `b′·θ = 0` imposed, so `a_eq` *is* their equilibrium and `Σ_k θ_k D_k(q)` is the
  exact operator for all 30. The pin residual is re-checked at run time
  (`|Δa| / a_eq < 1e-3`), not assumed.
- **Unconstrained** — a native Hessian per member at its *own* lattice constant
  (`a` spans 3.98–4.30 Å). Evaluating these at `a_eq` would fold residual stress into
  the phonons, and a soft mode read off that would be unattributable between bad
  parameters and wrong volume — which is precisely the claim the figure makes.

## Axis choices

**Frequency is shared within a row, not between rows.** Within a row it has to be:
unconstrained vs constrained is the comparison, and independent axes would autoscale
away the width difference being claimed. Between rows it must not be: Al_12's
unconstrained members reach ≈ −23 THz against Al_16's ≈ −7, so one global axis squeezes
the entire Al_16 row into the middle third of its panels. `SHARE_Y=1` forces a global
axis if you want it.

**The path coordinate is remapped.** The two models relax to different lattice
constants, so their Γ→X, X→U, … lengths differ by ~0.3%; drawn raw, the high-symmetry
ticks would not line up between rows, which in a column-aligned figure reads as an
error. Each row is mapped piecewise-linearly onto the mean of the two tick vectors —
exact at every high-symmetry point, monotone in between. The pre-remap discrepancy is
printed so it can be audited.

## Environment variables

| var             | default                        | effect                                        |
| --------------- | ------------------------------ | --------------------------------------------- |
| `FIGW`          | `540`                          | width in **points** = the width the figure is displayed at in the paper |
| `ASPECT`        | `0.78`                         | height / width                                |
| `SHARE_Y`       | `0`                            | `1` = one frequency axis over all four panels |
| `LETTERS`       | `0`                            | `1` = (a)–(d) panel letters                   |
| `ROW1`, `ROW2`  | model name + parameter count   | left-hand row labels                          |
| `SRC12`,`SRC16` | see script                     | input `.jls`                                  |
| `OUT`           | `bands_four_panel/bands_four_panel` | output stem                              |
| `RELAX`         | `0` (build script)             | `1` = re-relax each member instead of reusing the saved `a` |
| `HESS_THREADS`  | all Julia threads (build)      | cap on concurrent native Hessian builds       |
