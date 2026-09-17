# Constrain Mo elastics in MACE-MPA-0 (`Constrain_Mb_Elastics.pdf`)

Readout-only correction of MACE-MPA-0 for Mo. The message-passing layers stay frozen, and a
correction δΘ to the readout is fitted to Mo DFT energies by least squares, subject to linear
constraints that put C11, C12 and C44 in experimental windows.

## Plan → implementation

| Spec | Where | How |
|---|---|---|
| NN part is the basis, readout parameters are learned (Perez et al. 2025) | `lib_mace_readout.py` | ScaleShiftMACE: `E = ΣE0 + Σ_atoms[scale·(pair + w₀·x₀ + w₁·x₁) + shift]`. Hooks read `x₀` (128 scalar node features into `readouts[0].linear`) and `x₁` (16 post-activation hidden units into `readouts[1].linear_2`). `D = Σ_atoms[x₀, x₁, 1]` has 145 columns and `E_MACE = E_frozen + D·Θ₀` **exactly**. This is checked on real configs in step 01. |
| Eq. (2): `E_DFT − E_MACE − Θ·D` | `01_…py`, `02_…jl` | Per-atom rows. `δΘ` is fitted to the MACE residual, so δΘ = 0 is MACE-MPA-0. |
| Eq. (3): `C_exp < ∂²(D·Θ)/∂ε_i∂ε_j < C_exp` | `01_…py` | `∂²D/∂ε²` by central finite differences on the 2-atom BCC cell (no internal relaxation is needed in BCC). Rows are `(160.2/V)∂²D`, in GPa. The window is `C_exp(1 ± REL_TOL)`. There is also a stress row `|∂E/∂ε₁|/V ≤ P_TOL`, so that A0 stays the equilibrium lattice and the constrained C_ij are equilibrium C_ij. |
| Eq. (1) with OSQP.jl | `02_…jl` | QP in θ̃ = S δΘ (S is the column std) with ridge λ. The offset column is left unpenalised. `eps_abs = eps_rel = 1e-9` with polish (the loose defaults made the W QP irreproducible). Offset-only, ridge and constrained fits are reported side by side. |
| Does it actually work | `03_…py` | Writes δΘ into the readout weights and recomputes everything from the patched model: energy identity, C_ij at A0 and at the patched model's own relaxed a, and energy **and force** RMSE (forces are not in the fit). Saves `.model` files. |

## Choices the spec leaves open (change via env)

- **DFT data:** the mlearn Mo set (Zuo et al., JPCA 124, 731 (2020); PBE; 194 train / 23 test;
  Elastic / AIMD / Vacancy / Surface). Its PBE reference differs from MPtrj's by a
  constant per atom, which the constant column absorbs.
- **Experimental targets:** C11/C12/C44 = 464.7/161.5/108.9 GPa, a₀ = 3.147 Å. These are the
  experimental values commonly quoted alongside the Mo potentials compared in Smirnova et al. 2020.
  **Check them against that paper's table** and override with `C11= C12= C44=`.
- **Where the constraint sits:** `A0=mace` (default) uses MACE-MPA-0's own relaxed a;
  `A0=exp` uses 3.147 Å, and the stress row then pulls the lattice constant there too.
- `REL_TOL=0.01`, `P_TOL=0.1` GPa, `STRAIN_H=0.005`, `LAMBDA=1e-4` (run step 02 with `SCAN=1`
  to see the λ scan).
- **Model:** `medium-mpa-0` (the model named in Eq. 2).

## Run

```bash
cd /storage/astro2/phupfb/PhD/acestuff/ACEWorkflow
# one-off environment (CPU torch + mace-torch 0.3.16; python/ is gitignored)
python3 -m venv python/mace_venv
python/mace_venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
python/mace_venv/bin/pip install mace-torch ase

bash constrained_optimization/mo_mace/run_pipeline.sh
```

Outputs are in `models/Mo_MACE_MPA0_readout/`: design and constraint CSVs,
`delta_theta_{offset,ridge,constrained}.csv` and `mace_mpa0_mo_{…}.model`.
