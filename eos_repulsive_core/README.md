# `eos_repulsive_core/` — W equation of state with the training pair-distance histogram

Regenerates `models/W_20_4_5A_3/results/eos_with_pair_hist.pdf`.

```bash
julia --project -t 8 eos_repulsive_core/eos_with_pair_hist.jl
```

Two stacked panels on a shared x-axis:

- **top** — bulk energy vs lattice constant for `ACE constrained`, `ACE unconstrained + ZBL`
  and `ZBL` alone, with a dashed line at 2.18 Å, the largest lattice constant the
  positive-core constraint was imposed at;
- **bottom** — the pair-distance distribution of the whole W training set
  (`data/W/df_W_train.extxyz`), so you can see the constrained region sits below where
  the data actually lives.

Inputs: the `W_20_4_5A_3` model, `models/W_20_4_5A_3/positive_core_constrained_parameters.csv`,
and the training set. Runtime a few minutes, dominated by loading the model and the
~10⁷ pair distances.

## Three defects fixed in this copy

The original, `scripts/repulsive_core/ZBL_core_ACE_correction.jl`, **cannot be run top to
bottom** — it only ever worked in a REPL that already had the right packages loaded and
where the broken block was skipped by hand. All three are marked `# [REPRO]`:

1. **`@testset` with no `using Test`.** The script has an embedded test for the periodic
   pair-distance routine but never imports `Test`, and none of the packages it does
   import re-export it (checked). Hoisted to the top — the test passes, 3/3.
2. **`readdlm` used ~30 lines before `using DelimitedFiles`.** Same fix.
3. **A dead POPS block that also crashes.** `pops_eig`, `pops_bound` and
   `con_pops_samples` are assigned and never referenced again; the figure does not
   depend on them. `hypercube()` throws `ArgumentError: matrix contains Infs or NaNs`
   there, because `constrained_pointwise_corrections` divides by `leverage`, and
   **6 leverage entries are exactly zero**. The script now prints the leverage range and
   that count, and guards the block off. Set `POPS_BLOCK=1` to re-enable and reproduce
   the failure.

Nothing else is changed — no physics, no fit, no plotting.

Two smaller robustness fixes, also marked:

- the training set is addressed from the repo root (`TRAINSET` to override), so the
  script no longer depends on the process working directory;
- `OUT` overrides the output stem. It **defaults to the published path**, so a plain
  rerun overwrites `models/W_20_4_5A_3/results/eos_with_pair_hist.{pdf,png}` in place —
  which keeps existing `\includegraphics` resolving, but set `OUT` if you want the
  published file left alone.

## Worth knowing about those 6 zero-leverage rows

They are a property of the fit, not of this figure, and the figure is unaffected. But
any other POPS calculation on this model divides by the same vector, so anything that
computes pointwise corrections for `W_20_4_5A_3` will produce `Inf` unless those rows are
dropped or the leverage is floored. Worth checking before reusing this model for UQ.
