# paper_figures — reproducing the figures

Everything needed to rebuild the figures in *Physics Informed Priors for MLIPs with UQ*,
collated in one place for reviewers.

This directory contains **no plotting code and no copies of the analysis scripts**. Each
figure points at its canonical script under `scripts/`, so there is exactly one version
of every script in the repository. A collated copy would eventually drift from the one
that actually produced a published figure, and then nobody could tell which was which.

## Usage

From the repository root:

```bash
julia --project paper_figures/check_inputs.jl     # is the data actually here?
julia --project paper_figures/make_figures.jl     # build everything
julia --project paper_figures/make_figures.jl pinned_rejection_phonons   # or just one
```

`check_inputs.jl` exits non-zero if anything is missing, so it works as a gate in a
batch script. `DRY_RUN=1` makes `make_figures.jl` print commands without running them.

## Read this before cloning and expecting it to work

**The repository tracks no data.** `.gitignore` excludes `models/`, `results/`, `*.csv`,
`*.xyz` and `*.json`, which is every fitted model, every design matrix, every dataset and
every cached intermediate. A fresh clone gives you the code and nothing to run it on.

Two kinds of input:

- **External roots** — cannot be regenerated from this repository. These are the DFT
  datasets under `data/Al/` (training `manual_df_train_Al.extxyz` and its subsets, test
  `manual_df_test_Al.xyz`) and the fitted models under `models/<name>/`
  (`<name>.json`, `A.csv`, `Y.csv`, `P.csv`, `W.csv`, `lin_params.csv`). The models are
  reproducible from the datasets via `scripts/model_building/build_model.jl`, but that is
  a refit, not a byte-identical restore. They must come from the data archive that
  accompanies the paper.
- **Intermediates** — regenerable. Every one is listed in `manifest.jl` with the script
  that produces it, so `check_inputs.jl` tells you exactly what to run and in what order.

Sizes worth knowing before you start (apparent size, i.e. what you actually transfer —
`du` on a compressed filesystem will report roughly half): `models/Al_20_4_6A_3_/A.csv`
is 5.2 GB and drives a ~15–20 GB memory peak; `models/Al_12_4_6A_2_/A.csv` is 259 MB;
`committee_stable.jls` is 55 MB; `manual_df_test_Al.xyz` is 18 MB.

`check_inputs.jl` prints the real size of everything it finds, so run it first.

## Figures

| id | script | what it shows |
|---|---|---|
| `fullcloud_bands_parity_calibration` | `scripts/uq/hypercube_full_cloud_bands_Al_12_4_6A_2.jl` | Constrained POPS scaled to the full leverage cloud — committee phonon bands, plus test-set parity and calibration for both hypercube types |
| `pinned_rejection_phonons` | `scripts/uq/pinned_hypercube_rejection_Al_20_4_6A_3.jl` | a_eq-pinned committee, no-rejection vs rejection sampling, side by side on the undotted 4×4×4 band path |

### Dependency chain

```
data/Al/*.extxyz ──► scripts/model_building/build_model.jl ──► models/<name>/{json,A,Y,P,W,lin_params}
                                                                   │
   ┌───────────────────────────────────────────────────────────────┴──────────────┐
   │  Al_12_4_6A_2                                                 Al_20_4_6A_3   │
   ▼                                                                              ▼
bandpath_committee_undotted_Al_12_4_6A_2_ncell4_densek.jl          pinned_hypercube_rejection_*.jl
   └─► theta_mean.csv, committee_repaired.csv                      (self-contained: builds and
   │                                                                caches its own 4×4×4 band
   ▼                                                                path if none is found)
cutting_plane_full_cloud_Al_12_4_6A_2.jl
   └─► committee_stable.jls  (73,411 constrained members)
   │
   ▼
hypercube_full_cloud_bands_Al_12_4_6A_2.jl  + data/Al/manual_df_test_Al.xyz
   └─► bands, parity ×2, calibration ×2
```

## Reproducibility caveats

Stated per figure in `manifest.jl`, but the general rule:

- Committees drawn through **OSQP** (the constrained-QP route) do not reproduce member
  identities run to run — the solver's path depends on floating-point details. Aggregate
  statistics are stable to well under 1%. Never read anything into a member index.
- Committees drawn through **`sample_hypercube` / `rejection_sample_hypercube`** with a
  seed and no QP are exactly reproducible, member for member. `pinned_rejection_phonons`
  is one of these.
- Rejection *rates* depend on the RNG stream even when seeded, because the number of
  proposals varies.

## Adding a figure

Append one entry to `FIGURES` in `manifest.jl` — id, title, script, command, env vars,
outputs, inputs with their producers, and any caveat a reviewer needs. `check_inputs.jl`
and `make_figures.jl` both read that list and need no changes.
