# Constrain Mo elastics in MACE-MPA-0 (`Constrain_Mb_Elastics.pdf`)

A linear corrector on the MACE-MPA-0 readout descriptors, as in Perez et al., npj Comput.
Mater. 11, 263 (2025), §III.G, Eq. 10. The corrector is fitted to Mo DFT energies subject to
linear constraints that put C11, C12 and C44 in experimental windows.

## Maths

The corrected model is `E(δΘ) = E_MACE-MPA-0 + D·δΘ`, and D is summed over atoms.

- **`DESCRIPTOR=perez` (default):** `D = Σ[h⁰(0e), h¹, 1]`, 257 columns. This is the scalar
  input to the readout layer, the 128 + 128 = 256-dimensional D of Perez Eq. 10, plus a per-atom
  offset. It is realised by patching `readouts[0].linear` and wrapping `readouts[1]` with an
  added linear term.
- **`DESCRIPTOR=readout`:** `D = Σ[h⁰(0e), σ(W h¹), 1]`, 145 columns. This is the input to the
  last linear map of each readout, so δΘ just changes the existing readout weights. It is less
  expressive (16 layer-1 directions, not 128).

The loss is `(1/2n) Σ ((E_DFT − E_MACE − D·δΘ)/N)² + (λ/2)‖S δΘ‖²`. That is Eq. (2) with per-atom
rows, plus a ridge term towards the foundation model (δΘ = 0). It is solved in θ̃ = S δΘ, where
S is the column std. The offset column is left unpenalised: it absorbs the difference between
the DFT set's PBE reference and MPtrj's.

The constraints are Eq. (3), written for the corrected model per unit volume at a BCC lattice
constant A0:

```
C_exp(1−tol) ≤ C_MACE + (160.2/V) ∂²D/∂ε_i∂ε_j · δΘ ≤ C_exp(1+tol)    (C11, C12, C44)
|∂E_MACE/∂ε₁ + ∂D/∂ε₁ · δΘ| / V ≤ P_TOL                                (zero stress)
```

- `C_ij = V⁻¹ ∂²E/∂ε_i∂ε_j` with `F = I + ε` equals the stress–strain elastic constants only at
  zero stress, so the stress row is required. It also stops the correction moving the
  equilibrium lattice away from A0.
- Voigt ε₄ = 2ε_yz. BCC is a Bravais lattice, so there is no internal relaxation.
- MPA-0's ZBL pair term is active for Mo: its cutoff is 2·r_cov = 3.08 Å and the nearest
  neighbour is at 2.73 Å. It is frozen, so it lives in E_MACE and C_MACE, never in D.

Exactness checks, all asserted:

- **Step 01:** a random δΘ written into the model reproduces `E_MACE + D·δΘ` on real configs.
  It also reproduces the strain derivatives, `∂E_patched = ∂E_MACE + ∂D·δΘ`.
- **Step 03:** the fitted δΘ is checked the same way, then C_ij, relaxed a, and energy and force
  RMSE are recomputed from the patched model itself. Forces are not in the fit.

## Choices the spec leaves open (all env vars)

- **DFT data:** the mlearn Mo set (Zuo et al., JPCA 124, 731 (2020); PBE; 194 train / 23 test;
  Elastic / AIMD / Vacancy / Surface).
- **Experimental C_ij:** the spec says to take them from Smirnova et al. 2020, but I couldn't
  access it. The defaults are Dickinson & Armstrong 1967 at 273 K, **463.7/157.8/109.2 GPa**, as
  tabulated in Dal Corso, arXiv:2406.16634, Table II. The 0 K extrapolation there is
  480.0/155.8/112.4 GPa. MACE is a static 0 K model, and the 273 K→0 K shift (~3% on C11) is
  larger than `REL_TOL=0.01`. **Check against Smirnova's table** and set `C11= C12= C44=`.
- **A0:** `A0=mace` (default) is MPA-0's relaxed a. `A0=exp` is 3.147 Å (room temperature).
- **Other defaults:** `REL_TOL=0.01`, `P_TOL=0.1` GPa, `STRAIN_H=0.005`, `LAMBDA=1e-4`.
  `SCAN=1` prints a λ scan.
- **Known tension:** PBE itself gets C44 ≈ 9% below experiment (Dal Corso). A fit to PBE energies
  therefore competes with the experimental constraint, which shows up in the ridge-vs-constrained
  RMSE.

## Run

```bash
cd /storage/astro2/phupfb/PhD/acestuff/ACEWorkflow
sbatch constrained_optimization/mo_mace/run_pipeline.slurm            # DESCRIPTOR=perez
sbatch --export=ALL,DESCRIPTOR=readout constrained_optimization/mo_mace/run_pipeline.slurm
```

The environment (`python/mace_venv`, gitignored) was built with:

```bash
python3 -m venv python/mace_venv
python/mace_venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
python/mace_venv/bin/pip install mace-torch ase
```

Outputs go to `models/Mo_MACE_MPA0_readout/<DESCRIPTOR>/`: design and constraint CSVs,
`delta_theta_{offset,ridge,constrained}.csv` and `mace_mpa0_mo_{…}.model`. Loading a `perez`
model needs this directory on `PYTHONPATH`.
