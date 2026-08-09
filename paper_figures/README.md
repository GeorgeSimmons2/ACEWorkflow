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
| `naive_vs_constrained_bands` | `scripts/uq/naive_vs_constrained_fullcloud_Al_12_4_6A_2.jl` | Unconstrained vs constrained POPS committee bands on one shared frequency axis — **run `fullcloud_bands_parity_calibration` first**, this reuses its committee |
| `fcc_compare_300K` | `scripts/uq/fcc_compare_figure_Al_12_4_6A_2.jl` | FCC stability under NPT at 300 K — dispersions overlaid, unconstrained RDF above the constrained one. **Multi-volume line**, not the full-cloud one |
| `al20_parity_calibration` | `scripts/uq/parity_calibration_pinned_Al_20_4_6A_3.jl` | Parity and calibration for the two Al_20 pinned committees — **run `pinned_rejection_phonons` first** |
| `thermal_expansion_aT` | `scripts/uq/replot_thermal_expansion_aT_Al_12_4_6A_2.jl` | a(T) under NPT for the constrained and unconstrained members — scatter, linear fit, α quoted. Built at 260 pt for a side-by-side pair. Replot only, runs no MD |
| `surface_energy` | `scripts/qoi/surface_energy_vacuum.jl` | Al(001) surface energy across both committees, full relaxation per member. **The validated QoI** |
| `vacancy_formation` | `scripts/qoi/vacancy_formation.jl` | Vacancy formation energy across both committees. Spread comparison only — see the caveat below |

### The two QoIs, and which one to quote

Both propagate the committees into an observable that is **in no predicate** — not the
Born rows, not the a_eq pin, not the phonon cut — so any tightening is the prior reaching
something it never constrained.

- **`surface_energy` is the one to quote.** The central models give +0.861 and
  +0.946 J/m² against ~0.9–1.0 for Al(001) in DFT, and relaxation lowers γ by ~0.02 J/m²
  as it must. The headline is not the variance ratio (0.60) but that **15/30 unconstrained
  members predict a negative surface energy** — a crystal that spontaneously cleaves —
  against 3/30 constrained, with the constrained mean landing on the DFT value.
- **`vacancy_formation` is a spread comparison only.** The model gives E_f < 0, already
  negative before any relaxation. This is conditioning, not a broken model: E_f multiplies
  the per-atom cohesive energy by N−1 = 255, so a 4.7 meV/atom error flips its sign.
  γ has no such amplification, which is why the same model succeeds there.

Two constraint studies run in parallel through this repository and should not be
conflated in captions:

- **a_eq only** — the full-cloud line. `fullcloud_bands_parity_calibration` and
  `naive_vs_constrained_bands`. Constraints (Born rows, `b′·θ = 0` as a hard equality in
  the cutting-plane QP, phonon stability) are all imposed at the single relaxed lattice
  constant.
- **Multi-volume** — `fcc_compare_300K`. Phonon stability imposed from a_eq out to
  1.1·a_eq, which is what survives NPT. An a_eq-only member can satisfy every constraint
  and still leave FCC once the cell is allowed to expand.

Shared plotting code lives in `scripts/uq/lib_parity_calibration.jl` (parity and
calibration, used by both the Al_12 and Al_20 entries) and
`scripts/bandpath_phonon_uq/lib.jl` (band paths, committee predictions).

### One known reproducibility gap

`thermal_expansion_aT` reads two archived `thermal_expansion_summary.csv` files. Neither
NPT driver in the current tree writes to the directory its summary lives in — the output
paths were changed after those runs. The **figures** are fully reproducible from the
CSVs; regenerating the **CSVs** would need the earlier driver settings, which are not
recorded here. Both files are under 1 KB, so ship them in the data archive. Everything
else in this directory regenerates from the external roots via the listed producers.

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
   └─► bands ×2, parity ×2, calibration ×2
   │   └─► committee_rejection_full_cloud.csv
   ▼
naive_vs_constrained_fullcloud_Al_12_4_6A_2.jl
   └─► naive vs constrained bands, shared frequency axis
```

Ordering matters in one place only: `naive_vs_constrained_bands` consumes
`committee_rejection_full_cloud.csv`, so `fullcloud_bands_parity_calibration` has to run
first. `make_figures.jl` runs entries in manifest order, which respects this; it also
skips rather than half-fails if the CSV is absent.

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
