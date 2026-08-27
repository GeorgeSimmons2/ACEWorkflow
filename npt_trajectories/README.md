# `npt_trajectories/` — constrain → MD → figure

The full pipeline behind `thermal_expansion_vs_experiment/`. Stage 3 (the plot) only
reads summary CSVs; everything upstream of them lives here.

```
stage 1  constrain   committee_constrained_multivolume.jl   → the blue member
                     committee_aeq.jl                       → the red member's dependency
stage 2  MD          npt_constrained_softest.jl             → blue a(T)
                     npt_unconstrained_naive_worst.jl       → red a(T)
stage 3  figure      ../thermal_expansion_vs_experiment/plot_…jl
```

```bash
bash npt_trajectories/run_pipeline.sh published            # reproduce the figure as published
bash npt_trajectories/run_pipeline.sh all                  # regenerate everything from the constraints down
bash npt_trajectories/run_pipeline.sh verify-determinism   # does stage 1 reproduce itself?
bash npt_trajectories/run_pipeline.sh figure               # local, seconds
```

**`published` and `all` are not the same thing, and the difference is the whole point of
the next section.** `published` reuses the existing committees, so the parameter vectors
are the exact ones behind the figure. `all` rebuilds them, and will not land on the same
vectors.

---

## Determinism: why rerunning stage 1 does not give you the same parameters

**Measured, not assumed.** The multi-volume committee was rerun under an identical seed
(job 6000261) into `results/bandpath_undotted_multivolume_rerun/`. Against the original:

| quantity                            | max &#124;Δ&#124; |
| ----------------------------------- | ----------------- |
| `theta_mean.csv` (the mean model)   | 9.12 × 10⁻³       |
| `theta_npt_softest.csv` (blue member) | **4.31**        |
| `committee_rejection.csv` (all rows) | **7.28**         |

and the softest member's *index* moved from #18 to #22.

Two things follow. The **mean model is effectively reproducible** — it is a single
well-posed QP and converges to the same point every time. Individual **committee members
are not**: each is the end of a cutting-plane cascade (median ~864 added rows, max 1605),
which amplifies any tiny solver difference into a completely different accepted vector.

### It is the solver, not the RNG

The naive arm uses the same sampler, the same `Random.seed!(1234)` and **no QP**, and
reproduces bit-exactly (worst-over-volumes −11.4711 both runs, same member #15). That is
a controlled attribution, not a guess: the only difference between the arms is the QP.

### The mechanism

`OSQP.setup!` in the committee scripts never sets `adaptive_rho_interval`, so it takes
the default **0** — which tells OSQP to derive the number of iterations between rho
updates from **measured setup and iteration time**. Confirmed by reading the settings
back out of a live workspace: the effective value is filled in automatically from 0 to
50. Different wall-clock timing → different rho schedule → different iterate path →
different accepted member.

So the red series is already reproducible; only the blue one is not.

### The fix, and its honest status

Both pinned committee scripts now pass `adaptive_rho_interval = RHO_INTERVAL`
(default 25), removing the only timing-dependent input in the QP path.

**This is not yet proven sufficient.** On a small probe QP the timing-derived interval
happened to be stable across processes, so the mechanism is demonstrated but the cure is
not. The real problem is where it matters — repeated `setup!` calls on a matrix growing
to ~1600 rows, under cluster load. Verify before relying on it:

```bash
bash npt_trajectories/run_pipeline.sh verify-determinism   # runs stage 1 twice
bash npt_trajectories/run_pipeline.sh compare-determinism  # reports max |Δθ|
```

Run B is queued *after* A rather than alongside it — two jobs sharing a node would
perturb each other's timing, which is the very thing under test. For scale, the unpinned
original differs by 4.31; a working pin should read 0.

Note that pinning **changes** the rho schedule, so it cannot recover the previously
published members. It makes future runs reproducible; it does not reproduce the past.

### So which mode should the paper use?

| you want                                        | use                                   |
| ----------------------------------------------- | ------------------------------------- |
| the figure exactly as it stands                  | `published` — reuses the saved vectors |
| "our pipeline reproduces end to end"             | `verify-determinism`, then `all`, then requote α |

If `verify-determinism` comes back non-zero, the constrained arm cannot be made
reproducible by this route and the only honest options are to ship the saved θ as data
(what `published` does) or to change solver.

**Do not quote a member index anywhere.** `rejection[18]` does not name the same vector
twice. Quote ensemble statistics — worst-over-volumes, median, acceptance rate, coverage
— which are stable to under 1%.

---

## These are pinned copies; the working copies have drifted

The originals in `scripts/uq/` have been repurposed since these runs, so running them
today reproduces a *different member into a different directory*. Every difference is
marked `# [REPRO]` in the pinned files.

**`npt_constrained_softest.jl`** vs `scripts/uq/npt_multivolume_member_Al_12_4_6A_2.jl`

| line            | working copy today                                | pinned back to                          |
| --------------- | ------------------------------------------------- | --------------------------------------- |
| `npt_member`    | `:cheap_rej`                                       | `:softest`                              |
| `committee_dir` | `results/aeq_cheap_vs_expensive`                   | `results/bandpath_undotted_multivolume` (via `COMMITTEE_DIR`) |
| `outdir`        | `results/npt_multivolume_a_eq_con_then_rejection`  | `results/repro_npt_multivolume_softest` |

**`npt_unconstrained_naive_worst.jl`** vs `scripts/uq/npt_thermal_expansion_worst_member_Al_12_4_6A_2.jl`

| line         | working copy today                            | pinned back to                                          |
| ------------ | --------------------------------------------- | ------------------------------------------------------- |
| `npt_vector` | `:softest_constrained`                        | `:worst_naive`                                          |
| `outdir`     | `results/npt_thermal_expansion_worst_member`  | `results/repro_npt_thermal_expansion_naive_worst_member` |

**`committee_*.jl`** vs their originals: the pinned OSQP rho schedule above, and an
output directory defaulting somewhere fresh.

Plus, in all four, the `include` path for `lib.jl` (this directory is one level from the
repo root, not two). **No constraint, sampler, MD parameter or analysis code is
changed** — diff them against the originals to confirm.

## The reproduction check

Neither MD driver hard-codes a parameter vector; each selects its member from a
committee. Both therefore compare what they selected against the vector the published
run saved, and **abort** if it differs:

- constrained → `results/npt_multivolume_softest/theta_used.csv`
- unconstrained → `results/npt_thermal_expansion_naive_worst_member/theta_naive_worst.csv`

Already verified for the constrained arm: `bandpath_undotted_multivolume/theta_npt_softest.csv`
and `npt_multivolume_softest/theta_used.csv` agree to **max |Δ| = 0.0** over all 91
coefficients — so `published` mode will pass. Set `THETA_REF=none` to skip the check
when deliberately running freshly generated members (`run_pipeline.sh all` does this for
you). Both drivers now also write their own `theta_used.csv`, so every run leaves a
reference behind.

## What the MD actually does

Identical settings on both sides, so the red/blue comparison is not confounded:

- 4×4×4 FCC supercell, 256 atoms, started from that member's **own** 0 K relaxed a₀
- NPT at 0 Pa — pure thermal expansion — Langevin thermostat + `MonteCarloBarostat`
- `dt` = 1 fs, friction 0.01 fs⁻¹
- 10,000 equilibration steps (discarded) + 20,000 production, logged every 50
  → 601 frames per temperature, 401 of them production
- T = 300, 500, 700, 900 K

The box is cubic but **changes every frame** under the barostat, so any pair analysis
must use each frame's own cell — both drivers already do.

### How `a_Ang` and `a_std_Ang` are computed

Per production frame, volume is converted to a lattice constant first, then the
statistics are taken over those:

```julia
a_prod = cbrt.(vol_hist[prod]) ./ N_super     # N_super = 4
a_T    = mean(a_prod);  a_T_std = std(a_prod)
```

So `a_std_Ang` is the **sample standard deviation of the per-frame lattice constant** —
the width of the equilibrium NPT volume-fluctuation distribution, not a standard error on
the mean. It matches ⟨δV²⟩ = k_BT·V·κ_T to about 3% at 300 K (0.00483 Å predicted against
0.00496 Å recorded, using the experimental bulk modulus), which is why it grows with
temperature. It carries no parameter uncertainty: each is one member's own MD.

### The FCC-survival diagnostic

The constrained driver records mean coordination and median nearest-neighbour distance
per temperature and flags whether the lattice was still FCC when a(T) was measured — an
earlier run reported "6.3 % thermal expansion" for a cell that had transformed
(coordination 12 → 9.5). It is why the 900 K point is plotted but never fitted.

Caveat carried over from the original: coordination inside a fixed 3.3 Å cutoff is
confounded by expansion, since the cutoff does not scale with `a`. For a real FCC
verdict use `scripts/uq/fcc_order_ensemble_Al_12_4_6A_2.py` (q̄₆), not the coordination
number.

The unconstrained driver predates this diagnostic and has fewer summary columns, which is
why the plotting script parses columns by name.

## Runtimes and dependencies

| stage                    | job                                | published runtime |
| ------------------------ | ---------------------------------- | ----------------- |
| constrain (multi-volume) | `run_committee_constrained.slurm`  | 21 min (job 5997446) |
| constrain (a_eq)         | `run_committee_aeq.slurm`          | similar           |
| MD constrained           | `run_npt_constrained_softest.slurm`| 4 h 27 min (job 5997448) |
| MD unconstrained         | `run_npt_unconstrained_naive_worst.slurm` | ~4 h       |
| figure                   | local                              | seconds           |

Everything needs the model itself — `models/Al_12_4_6A_2_/{A,P,W,Y,lin_params}.csv` —
which is why `REPO` defaults to the main checkout rather than a worktree.

`models/Al_12_4_6A_2_/results/npt_multivolume_softest/PROVENANCE.md` records the
published constrained run in full.
