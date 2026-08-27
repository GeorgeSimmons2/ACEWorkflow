# `npt_trajectories/` — the NPT runs behind the a(T) figure

Reproduces the molecular dynamics that `thermal_expansion_vs_experiment/` plots. The
plotting script only reads `thermal_expansion_summary.csv`; these two drivers are what
produce it.

| script                              | series in the figure | member                                             |
| ----------------------------------- | -------------------- | -------------------------------------------------- |
| `npt_unconstrained_naive_worst.jl`  | **red**              | worst unconstrained POPS member of the 30-draw      |
| `npt_constrained_softest.jl`        | **blue**             | softest member of the multi-volume constrained committee (row 18) |

```bash
sbatch npt_trajectories/run_npt_unconstrained_naive_worst.slurm   # ~12 h wall limit
sbatch npt_trajectories/run_npt_constrained_softest.slurm         # ~8 h  (published run: 4 h 27 min)
```

Both write to a **fresh** `results/repro_*` directory so a rerun cannot overwrite the
published trajectories the figure is built from. Set `OUTDIR` to land somewhere else.

## Read this before running: the working copies have drifted

These are **pinned copies**. The originals in `scripts/uq/` have since been repurposed
for other studies, so running them today reproduces a *different member into a different
directory*. Every difference is marked `# [REPRO]` in the pinned files:

**`npt_constrained_softest.jl`** vs `scripts/uq/npt_multivolume_member_Al_12_4_6A_2.jl`

| line             | working copy today                     | pinned back to                        |
| ---------------- | -------------------------------------- | ------------------------------------- |
| `npt_member`     | `:cheap_rej`                           | `:softest`                            |
| `committee_dir`  | `results/aeq_cheap_vs_expensive`       | `results/bandpath_undotted_multivolume` |
| `outdir`         | `results/npt_multivolume_a_eq_con_then_rejection` | `results/repro_npt_multivolume_softest` |

**`npt_unconstrained_naive_worst.jl`** vs `scripts/uq/npt_thermal_expansion_worst_member_Al_12_4_6A_2.jl`

| line          | working copy today       | pinned back to                                        |
| ------------- | ------------------------ | ----------------------------------------------------- |
| `npt_vector`  | `:softest_constrained`   | `:worst_naive`                                        |
| `outdir`      | `results/npt_thermal_expansion_worst_member` | `results/repro_npt_thermal_expansion_naive_worst_member` |

Plus, in both, the `include` path for `lib.jl` (this directory is one level from the
repo root, not two) and a reproduction check described below. **No physics, no MD
parameter and no analysis code was changed** — diff them against the originals from
line 18 onwards to confirm.

## The reproduction check

Neither script hard-codes a parameter vector; each selects its member from an upstream
committee, exactly as the published run did. That means a regenerated committee would
silently change which member gets simulated. Both therefore compare the vector they
selected against the one the published run saved next to its own outputs, and **abort**
if they differ:

- constrained → `results/npt_multivolume_softest/theta_used.csv`
- unconstrained → `results/npt_thermal_expansion_naive_worst_member/theta_naive_worst.csv`

The constrained side is already verified: `bandpath_undotted_multivolume/theta_npt_softest.csv`
and `npt_multivolume_softest/theta_used.csv` agree to **max |Δ| = 0.0** over all 91
coefficients. The unconstrained member is re-derived from the POPS forest each run
(deterministic, `Random.seed!(1234)`), so its check runs at execution time.

## Upstream dependencies

Neither driver is standalone — each reads a committee produced earlier in the pipeline.
If `models/Al_12_4_6A_2_/results/` already contains these, nothing extra is needed.

| needed by     | directory                          | produced by                                                        |
| ------------- | ---------------------------------- | ------------------------------------------------------------------ |
| constrained   | `bandpath_undotted_multivolume/`   | `scripts/uq/bandpath_committee_undotted_Al_12_4_6A_2_multivolume.jl` (`run_committee_multivolume.slurm`) |
| unconstrained | `bandpath_undotted/`               | `scripts/uq/bandpath_committee_undotted_Al_12_4_6A_2.jl`            |

Both also need the model itself — `models/Al_12_4_6A_2_/{A,P,W,Y,lin_params}.csv` — which
is why the SLURM scripts default `REPO` to the main checkout rather than a worktree.

## What the runs actually do

Identical MD settings on both sides, so the red/blue comparison is not confounded:

- 4×4×4 FCC supercell, 256 atoms, started from that member's **own** 0 K relaxed a₀
- NPT at 0 Pa — pure thermal expansion — Langevin thermostat + `MonteCarloBarostat`
- `dt` = 1 fs, friction 0.01 fs⁻¹
- 10,000 equilibration steps (discarded) + 20,000 production steps, logged every 50
  → 601 frames per temperature, 401 of them production
- T = 300, 500, 700, 900 K

The box is cubic but **changes every frame** under the barostat, so any pair analysis
must use each frame's own cell — both drivers already do this in their RDF/MSD.

### How `a_Ang` and `a_std_Ang` are computed

Per production frame, the cell volume is converted to a lattice constant first, and the
statistics are taken over those:

```julia
a_prod = cbrt.(vol_hist[prod]) ./ N_super     # N_super = 4
a_T    = mean(a_prod);  a_T_std = std(a_prod)
```

So `a_std_Ang` is the **sample standard deviation of the per-frame lattice constant** —
the width of the equilibrium NPT volume-fluctuation distribution, not a standard error
on the mean. It matches ⟨δV²⟩ = k_BT·V·κ_T to about 3% at 300 K (0.00483 Å predicted
against 0.00496 Å recorded, using the experimental bulk modulus), and it grows with
temperature for that reason. It carries no parameter uncertainty: each is one committee
member's own MD.

### The FCC-survival diagnostic

The constrained driver records mean coordination and median nearest-neighbour distance
per temperature and flags whether the lattice was still FCC when a(T) was measured. This
exists because an earlier single-volume run reported "6.3 % thermal expansion" for a cell
that had actually transformed (coordination 12 → 9.5). It is why the constrained 900 K
point is plotted but never fitted in the figure — coordination 9.12, `still_fcc=false`.

The unconstrained driver predates that diagnostic and does not record it, which is why
its summary CSV has fewer columns. The plotting script parses columns by name for
exactly this reason.

## Provenance of the published runs

`models/Al_12_4_6A_2_/results/npt_multivolume_softest/PROVENANCE.md` records the
constrained run in full: member 18 of the multi-volume rejection committee, identified
three independent ways, SLURM jobs 5997446 (committee) and 5997448 (MD).
