# `constrained_optimization/` — where the parameters come from

`bands_four_panel/`, `eos_repulsive_core/` and `npt_trajectories/` all **replot from saved
θ**. This folder is the layer below them: the constrained optimisation runs that produced
those θ in the first place, pinned so they can be rerun, and a verifier that says how
closely a rerun matches.

```bash
bash constrained_optimization/run_pipeline.sh w        # start here — closes a real dead end
bash constrained_optimization/run_pipeline.sh verify   # local, seconds, safe any time
bash constrained_optimization/run_pipeline.sh all      # everything, SLURM-chained, ~12–15 h
```

Every stage writes to a `repro_*` directory. Nothing here can overwrite a published
artifact — see "How the Al_16 ensembles were lost" below for why that is not politeness.

---

## The chain

```
data/{Al,W}/*.extxyz ─► scripts/model_building/build_model.jl ─► models/<name>/{json,A,Y,P,W,lin_params}

  Al_12_4_6A_2_                                    → bands_four_panel TOP ROW
    al12/01_committee_ncell4_densek.jl             → theta_mean.csv, committee_repaired.csv
    al12/02_cutting_plane_full_cloud.jl            → committee_stable.jls   (73,479 members)
    al12/03_hypercube_rejection.jl      (needs 1,2)→ committee_rejection_full_cloud.csv  [right panel]
    al12/04_naive_ensemble.jl           (needs 1,3)→ samples_naive.csv, min_freq_naive.csv [left panel]

  Al_16_4_6A_3_                                    → bands_four_panel BOTTOM ROW
    al16/01_pinned_rejection_ensembles.jl          → ensemble_{un,}constrained.csv
    al16/02_bands_two_ensembles.jl                 → bands_two_ensembles.jls

  W_20_4_5A_3                                      → eos_repulsive_core
    w/01_positive_core_qp.jl                       → positive_core_constrained_parameters.csv

  Al_12, again, for the NPT figure                 → thermal_expansion_vs_experiment
    npt_trajectories/committee_{constrained_multivolume,aeq}.jl   (already pinned there)
```

Stage 1 of the Al_12 chain runs **first** even though stage 2 is the expensive one: stages 3
and 4 both read its `theta_mean.csv` and `committee_repaired.csv`.

The NPT committees are a **separate pair of runs** from `al12/01` — a multi-volume one
(phonon stability imposed from a_eq out to 1.1·a_eq) and an a_eq-only one. They already have
pinned copies under `npt_trajectories/`, so `run_pipeline.sh npt` hands off to that driver
rather than duplicating it.

---

## Two dead ends this folder exists to close

### The W θ had no producer at all

`models/W_20_4_5A_3/positive_core_constrained_parameters.csv` backs the "ACE constrained"
curve in `eos_repulsive_core/`. Four scripts read it —
`eos_repulsive_core/eos_with_pair_hist.jl`, `scripts/repulsive_core/ZBL_core_ACE_correction.jl`,
its `_uncertainty` sibling, and `scripts/uq/pops_naive_centred_on_constrained_W_20_4_5A_3.jl`
— and in **every one** the line that would generate it is commented out:

```julia
# ace_positive_core_constrained_parameters = constrained_ridge_regression(Ap, Yw, Gamma, constraint_matrix, bounds)
ace_positive_core_constrained_parameters = vec(readdlm("$(result.dir)/positive_core_constrained_parameters.csv", ','))
```

The function was defined, the constraint matrix and bounds were still built — and then
thrown away. `w/01_positive_core_qp.jl` restores the call. Everything feeding it is lifted
verbatim from `ZBL_core_ACE_correction.jl:1-62`; the constraint is unchanged (bulk energy
≥ 0 at 50 lattice constants on `LinRange(0.21, 2.18)` Å).

### How the Al_16 ensembles were lost

The bottom row of `bands_four_panel` comes from `bands_two_ensembles.jls` (11 Aug), built
from a **20-member** ensemble — `bands_{un,}constrained.csv` still have 20 rows. The
`ensemble_*.csv` and `pinned_ensembles.jls` on disk today are a **10,000-member** run from
19 Aug at `ACCEPT_TOL = 0.005`, written into the same directory. The published ensembles are
gone.

Measured, not assumed: the 19 Aug members are different draws, not a superset. The published
unconstrained ensemble is soft at members {2, 4, 6, 18}; the 19 Aug run's first 20 are soft
at {6, 14, 19}.

Two defaults had drifted, and **both** matter:

| | header comment says | code default became |
|---|---|---|
| `N_MEMBERS` | 20 | 10000 |
| `ACCEPT_TOL` | −0.05 | 0.005 |

`ACCEPT_TOL` changes the predicate, so the accepted stream diverges at the first proposal
that passes −0.05 but fails 0.005. `N_MEMBERS` changes the **unconstrained** arm too, which
is the non-obvious one:

> `sample_hypercube` draws `U = rand(Float64, (N, d))` and takes member *i* from **row** *i*.
> Julia fills column-major, so member *i*'s random numbers sit at stream positions
> {*i*, *i*+N, *i*+2N, …}. Change N and every member changes, at a fixed seed.
>
> `rejection_sample_hypercube` draws one d-vector per proposal in a loop, so it *does* have
> the prefix property — the first 20 accepted of an N=10000 run equal an N=20 run's,
> provided the predicate is identical.

`al16/01` pins both defaults back. Whether that recovers the published ensembles is checked
at the end of `al16/02`, against the per-member `(relaxed a, min ω)` table that survived.

---

## What "reproduced" means, per stage

Three mechanisms live in this pipeline and they have genuinely different reproducibility.
A single pass/fail would be a lie in one direction or the other, so `verify` reports a tier:

| tier | applies to | why | threshold |
|---|---|---|---|
| `exact` | the naive Al_12 ensemble, the Al_16 pinned draws | no QP in the member path — a seeded draw, or a closed-form rank-one correction | bit-exact. Measured: the naive arm reproduces exactly |
| `converged` | `theta_mean`, the W positive-core θ | one well-posed QP; the iterate path may differ, the optimum does not | max&#124;Δθ&#124; ≤ 1e-2. Measured: 9.12e-3 on a full `theta_mean` rerun |
| `statistical` | every cutting-plane committee | a cascade of up to ~1600 added rows amplifies any solver difference into a different accepted vector *and a different member order* | distribution only, at 4.5σ of sampling noise |

The `statistical` threshold is in units of **sampling noise**, not of θ. Two *correct*
30-member draws from the same distribution have means differing by about σ·√(2/n) = 0.26σ,
so a fixed fractional tolerance would reject them every time. `verify` reports

```
z_mean = |μ_A − μ_B| / (σ̄ · sqrt(2/n))        z_std = |σ_A − σ_B| / σ̄ · sqrt(2(n−1))
```

for the worst of the 91 (or 684) coefficients. Calibration on published data: identical
ensembles score 0.00; related-but-different ones (repaired vs rejection) score 4.8/6.2;
unrelated ones score 5.0/13.4. `z_std` is much the sharper discriminator — constraining
narrows the cloud more than it moves its centre.

**Do not quote a member index anywhere.** `rejection[18]` does not name the same vector
twice.

### Why committee members are not reproducible, and what was done about it

`OSQP.setup!` left `adaptive_rho_interval` at its default of 0, which tells OSQP to derive
the iterations between rho updates from **measured setup and iteration time**. Different
wall-clock timing → different rho schedule → different iterate path → after a cutting-plane
cascade, a completely different accepted vector. Measured on a controlled rerun: the mean
model moved by 9.12e-3, individual members by **4.31**, and the softest member's index moved
18 → 22. The naive arm — same sampler, same seed, *no QP* — reproduced bit-exactly, which
attributes the difference to the solver rather than the RNG.

Every pinned copy here passes `adaptive_rho_interval = RHO_INTERVAL` (default 25).
**This is not yet proven sufficient** — the mechanism is demonstrated but the cure is not, on
a matrix growing to ~1600 rows under cluster load. Set `RHO_INTERVAL=0` to restore the
original behaviour. Pinning changes the rho schedule, so it cannot recover previously
published members; it makes future runs reproducible.

---

## These are pinned copies

`scripts/` is never edited. Every stage is a copy with each difference marked `# [REPRO]`:

| file | pinned copy of | [REPRO] differences |
|---|---|---|
| `al12/01_committee_ncell4_densek.jl` | `scripts/uq/bandpath_committee_undotted_Al_12_4_6A_2_ncell4_densek.jl` | include path; `RHO_INTERVAL`; `outdir` → `repro_…`, overridable |
| `al12/02_cutting_plane_full_cloud.jl` | `scripts/uq/cutting_plane_full_cloud_Al_12_4_6A_2.jl` | same three |
| `al12/03_hypercube_rejection.jl` | `scripts/uq/hypercube_full_cloud_bands_Al_12_4_6A_2.jl` | include paths; `COMMITTEE_DIR`/`CLOUD_DIR`/`OUTDIR` to read stages 1 and 2 from `repro_` |
| `al12/04_naive_ensemble.jl` | `scripts/uq/naive_vs_constrained_fullcloud_Al_12_4_6A_2.jl` | include path; `SRC`/`COMMITTEE_DIR`/`OUTDIR` |
| `al16/01_pinned_rejection_ensembles.jl` | `scripts/uq/pinned_rejection_ensembles_Al_16_4_6A_3.jl` | `N_MEMBERS` 10000→20; `ACCEPT_TOL` 0.005→−0.05; repo-root paths; `outdir` → `repro_…`; band-path cache reused read-only |
| `al16/02_bands_two_ensembles.jl` | `scripts/qoi/bands_two_ensembles_Al_16_4_6A_3.jl` | include path; repo-root paths; `SRC` → `repro_…`; a comparison against the published per-member table appended |
| `w/01_positive_core_qp.jl` | — new; the setup is verbatim from `scripts/repulsive_core/ZBL_core_ACE_correction.jl:1-62` | the QP is actually called; `using Test, DelimitedFiles` hoisted; the dead POPS block omitted |

**No constraint, sampler, predicate, MD parameter or physics setting is changed anywhere.**
Confirm it yourself — the diffs are small and all mechanical:

```bash
diff <(grep -v '^#' scripts/uq/cutting_plane_full_cloud_Al_12_4_6A_2.jl) \
     <(grep -v '^#' constrained_optimization/al12/02_cutting_plane_full_cloud.jl)
```

Two figure scripts also gained env hooks so they can be rebuilt from the `repro_` tree.
Both **default to the published paths**, so a plain rerun of either is unchanged:
`bands_four_panel/build_bands_cache_Al_12.jl` (`SRC_N`, `SRC_C`, `COMMITTEE_DIR`) and
`eos_repulsive_core/eos_with_pair_hist.jl` (`THETA_CON`).

### The dead POPS block in the W script

`ZBL_core_ACE_correction.jl` computes `pops_eig, pops_bound = hypercube(...)` and never uses
the result; `hypercube()` throws `ArgumentError: matrix contains Infs or NaNs` there, because
`constrained_pointwise_corrections` divides by `leverage` and **6 leverage entries are exactly
zero**. The QP does not touch any of it, so `w/01` omits it. That is a property of the W fit,
not of this figure — anything else computing pointwise POPS corrections for `W_20_4_5A_3`
will produce `Inf` unless those rows are dropped or the leverage floored.

---

## Runtimes and resources

| stage | partition | published runtime | why |
|---|---|---|---|
| `w/01` | hmem | minutes–1 h | `A.csv` is 5.2 GB of text (146,126 × 1,829); the QP itself is 1,829² and trivial |
| `al12/01` | hmem, 20 | ~20 min | cold undotted `H_basis` build dominates |
| `al12/02` | compute, 42 | ~11 min at 16 threads | 73,479 QPs at ~74 ms each; checkpoints every 5,000 |
| `al12/03` | hmem, 8 | ~30 min | hypercube over a 73k-member cloud + test-set parity |
| `al12/04` | hmem, 40 | hours | 30 **native** 4×4×4 Hessians, one per naive member at its own lattice constant |
| `al16/01` | hmem, 40 | ~1 h | `A.csv` 1.9 GB; 3-row undotted band path |
| `al16/02` | compute, 40 | hours | 40 native Hessians |
| NPT ×2 | — | ~4.5 h each | see `npt_trajectories/README.md` |

Everything needs the models under `models/<name>/`, which the repository does not track.

## Files

```
run_pipeline.sh      the driver — read its header, it is the usage
lib_verify.jl        comparison primitives + the tier definitions
verify/compare_all.jl  every reproduced θ vs its published counterpart → REPRODUCTION.md
REPRODUCTION.md      generated; the table that answers "did this reproduce?"
al12/  al16/  w/     the pinned stages, one .slurm each
```

---

## Finding: the W positive-core QP is under-determined at its own tolerance

`w/01` ran and the constraint is real — the unconstrained fit has **46 of 50** bulk
energies negative on the constrained grid (minimum −2.8 × 10¹¹ eV at a = 0.21 Å); the
solved one has **0 of 50**. But the reproduced θ does not match the published one:

```
max |Δ|  = 1.688       ‖Δ‖₂/‖θ‖₂ = 14.3%
```

That is far outside the `converged` tier, and the reason is in OSQP's own output:

```
status: Solved   iters: 19000   pri_res: 7.20e-01   dua_res: 6.48e+06
optimal objective: -1.982e+09    optimal rho estimate: 1.00e-06
```

OSQP terminates on residuals measured **relative to the problem's own scale**, and this
problem is scaled ~10¹¹. At `eps_abs = eps_rel = 1e-3` — the defaults, which the original
left in place — the effective absolute tolerance is enormous, so the solver stops a long
way from the true optimum and *where* it stops depends on the rho schedule. A 14%
difference is what that mechanism predicts; it is not evidence that the published vector
is wrong.

`w/02_diagnose_positive_core.jl` settles it by evaluating the published θ, the loose θ and
a fresh `eps = 1e-10` polished solve against the **same** objective, and — the thing that
actually matters — comparing the **EOS curve each one draws**. The figure is a curve, so
two θ that differ in norm but draw the same curve are the same answer. Read
`REPRODUCTION.md` and that job's log together before concluding anything about the figure.

**If you reuse this QP for anything, tighten `eps_abs`/`eps_rel`.** The defaults are not
appropriate to a problem carrying 10¹¹ eV energies, and nothing in the original flagged it.
