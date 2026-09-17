"""
01_build_design_and_constraints.py — design matrix and elastic constraints for the
MACE-MPA-0 readout QP (Constrain_Mb_Elastics.pdf, Eqs. 1–3).

Per structure i:   D_i = Σ_atoms descriptor (DESCRIPTOR=perez: 257 cols; readout: 145 —
                   see lib_mace_readout.py),   r_i = E_DFT,i − E_MACE,i
Corrected model:   E(δΘ) = E_MACE + D·δΘ        (δΘ = 0 is MACE-MPA-0)

Constraints on the corrected model at the BCC lattice constant A0 (2-atom cubic cell, V):

   C_lo ≤ C_MACE + (160.2/V) ∂²D/∂ε_i∂ε_j · δΘ ≤ C_hi      for C11, C12, C44
   −P_TOL ≤ ∂E_MACE/∂ε₁/V + (160.2/V) ∂D/∂ε₁ · δΘ ≤ P_TOL   (zero stress at A0)

Spec Eq. (3) writes C_exp < ∂²(D·Θ) < C_exp.  Read literally the correction alone would have
to carry the whole C_ij and there is no 1/V; the physically meaningful statement is the one
above (C_ij of the corrected model, per volume), with the window C_exp·(1 ± REL_TOL).
The stress row matters: C_ij = V⁻¹∂²E/∂ε² equals the stress–strain coefficients only at
zero stress, and without it the correction can move the equilibrium lattice away from A0.

Experimental targets — spec says "as in Smirnova et al., PRMaterials 4, 013605 (2020)", which
could not be accessed.  Defaults are Dickinson & Armstrong, J. Appl. Phys. 38, 602 (1967),
273 K: C11/C12/C44 = 463.7/157.8/109.2 GPa (as tabulated in Dal Corso, arXiv:2406.16634,
Table II; 0 K-extrapolated: 480.0/155.8/112.4).  MACE is a static-lattice (0 K, no ZPE)
model, and the 273 K→0 K shift (~3% on C11) exceeds REL_TOL=1%: choose deliberately.
CHECK AGAINST SMIRNOVA TABLE and override with C11= C12= C44=.

Checks (exact, must pass): for a random δΘ, the patched MACE model reproduces
E_MACE + D·δΘ on real configs, and ∂E/∂ε of the patched model = ∂E_MACE/∂ε + ∂D/∂ε·δΘ.

Outputs ($OUTDIR, default $REPO/models/Mo_MACE_MPA0_readout/$DESCRIPTOR/):
   D_train.csv, D_test.csv          descriptor rows (no header)
   E_train.csv, E_test.csv          columns: E_DFT, E_MACE, n_atoms, group_id
   A_con.csv, l_con.csv, u_con.csv  constraint rows (GPa per unit δΘ) and bounds on δΘ
   con_meta.txt                     human-readable summary (read by 02 and 03)

Run:  python/mace_venv/bin/python constrained_optimization/mo_mace/01_build_design_and_constraints.py
Env:  REPO OUTDIR DESCRIPTOR(perez|readout)  A0 (Å, "exp" = 3.147 (298 K), default "mace")
      REL_TOL (0.01)  P_TOL (GPa, 0.1)  STRAIN_H (0.005)  C11 C12 C44 (GPa)
"""

import os
import sys
import time

import numpy as np
from ase.io import read

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_mace_readout as L  # noqa: E402

REPO = os.environ.get("REPO", "/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow")
MODE = L.descriptor_mode()
OUTDIR = os.environ.get("OUTDIR", os.path.join(REPO, "models", "Mo_MACE_MPA0_readout", MODE))
os.makedirs(OUTDIR, exist_ok=True)

A_EXP = 3.147  # Å, room temperature
TARGETS = {  # GPa — Dickinson & Armstrong 1967, 273 K (verify against Smirnova 2020)
    "C11": float(os.environ.get("C11", 463.7)),
    "C12": float(os.environ.get("C12", 157.8)),
    "C44": float(os.environ.get("C44", 109.2)),
}
REL_TOL = float(os.environ.get("REL_TOL", 0.01))
P_TOL = float(os.environ.get("P_TOL", 0.1))
H = float(os.environ.get("STRAIN_H", 0.005))
A0_SPEC = os.environ.get("A0", "mace")

calc = L.load_calc()
model = calc.models[0]
hooks = L.ReadoutHooks(model, MODE)
ncol = L.n_columns(model, MODE)
print(f"DESCRIPTOR={MODE}: {ncol} columns → {OUTDIR}")

# ── design matrices ──────────────────────────────────────────────────────────
groups = ["Elastic", "AIMD-NVT", "Vacancy", "Surface"]
frames = {}
for split in ("train", "test"):
    frames[split] = read(os.path.join(REPO, "data", "Mo", f"mlearn_Mo_{split}.extxyz"), ":")
    t = time.time()
    Ds, Es = [], []
    for at in frames[split]:
        E_dft = at.get_potential_energy()
        E_mace, D = L.energy_and_descriptor(calc, at, hooks)
        Ds.append(D)
        Es.append([E_dft, E_mace, len(at), groups.index(at.info["config_type"])])
    Ds, Es = np.array(Ds), np.array(Es)
    np.savetxt(os.path.join(OUTDIR, f"D_{split}.csv"), Ds, delimiter=",")
    np.savetxt(os.path.join(OUTDIR, f"E_{split}.csv"), Es, delimiter=",")
    r = (Es[:, 0] - Es[:, 1]) / Es[:, 2]
    print(f"{split}: {len(Ds)} configs in {time.time()-t:.0f}s; "
          f"(E_DFT−E_MACE)/N mean {r.mean():+.4f} eV, std {1e3*r.std():.2f} meV/atom", flush=True)
    if split == "train":
        D_train = Ds

# random correction for the exactness checks, scaled so each column moves E by ~meV/atom
rng = np.random.default_rng(0)
col_std = (D_train / D_train[:, -1:]).std(0)
col_std[col_std < 1e-12] = 1.0
dtheta_test = 1e-3 * rng.standard_normal(ncol) / col_std
calc_test = L.load_calc()
calc_test.models[0] = L.apply_correction(model, dtheta_test, MODE)
for at in frames["train"][:3]:
    E_mace, D = L.energy_and_descriptor(calc, at, hooks)
    x = at.copy(); x.calc = calc_test
    err = x.get_potential_energy() - (E_mace + D @ dtheta_test)
    print(f"  identity E_patched − (E_MACE + D·δΘ) = {err:+.2e} eV  (D·δΘ = {D @ dtheta_test:+.3e}, {len(at)} atoms)")
    assert abs(err) < 1e-6, "descriptor is not an exact linearisation of the correction"

# ── lattice constant ─────────────────────────────────────────────────────────
a_mace = L.relaxed_bcc_a(calc)
A0 = {"exp": A_EXP, "mace": a_mace}.get(A0_SPEC)
A0 = float(A0_SPEC) if A0 is None else A0
print(f"MACE relaxed a = {a_mace:.4f} Å;  constraining at A0 = {A0:.4f} Å ({A0_SPEC})")
cell = L.bcc_cubic(A0)
V = cell.get_volume()


def energy_with(c):
    def f(at):
        at.calc = c
        return at.get_potential_energy()
    return f


def descriptor(at):
    return L.energy_and_descriptor(calc, at, hooks)[1]


dE = L.strain_derivatives(energy_with(calc), cell, H)
dD = L.strain_derivatives(descriptor, cell, H)
dE_2h = L.strain_derivatives(energy_with(calc), cell, 2 * H)
dE_test = L.strain_derivatives(energy_with(calc_test), cell, H)
k = L.EV_A3_TO_GPA / V

C_mace = {"C11": k * dE["d11"], "C12": k * dE["d12"], "C44": k * dE["d44"]}
sigma_mace = k * dE["d1"]
print(f"MACE-MPA-0 at A0: C11 {C_mace['C11']:.1f}  C12 {C_mace['C12']:.1f}  "
      f"C44 {C_mace['C44']:.1f} GPa   ∂E/∂ε₁/V {sigma_mace:+.3f} GPa")
print(f"  (FD step 2h: C11 {k*dE_2h['d11']:.1f}  C12 {k*dE_2h['d12']:.1f}  C44 {k*dE_2h['d44']:.1f})")
for key in ("d1", "d11", "d12", "d44"):
    lhs, rhs = dE_test[key] - dE[key], dD[key] @ dtheta_test
    print(f"  {key}: ∂E_patched − ∂E_MACE {lhs:+.6e}   ∂D·δΘ {rhs:+.6e} eV")
    assert abs(lhs - rhs) < 1e-6 * max(1.0, abs(dE[key])), key

rows, lo, hi, names = [], [], [], []
for c, key in (("C11", "d11"), ("C12", "d12"), ("C44", "d44")):
    rows.append(k * dD[key])
    lo.append(TARGETS[c] * (1 - REL_TOL) - C_mace[c])
    hi.append(TARGETS[c] * (1 + REL_TOL) - C_mace[c])
    names.append(c)
rows.append(k * dD["d1"])
lo.append(-P_TOL - sigma_mace)
hi.append(P_TOL - sigma_mace)
names.append("dE/de1/V")

np.savetxt(os.path.join(OUTDIR, "A_con.csv"), np.array(rows), delimiter=",")
np.savetxt(os.path.join(OUTDIR, "l_con.csv"), np.array(lo), delimiter=",")
np.savetxt(os.path.join(OUTDIR, "u_con.csv"), np.array(hi), delimiter=",")
with open(os.path.join(OUTDIR, "con_meta.txt"), "w") as f:
    f.write(f"descriptor {MODE}\nA0 {A0:.6f}\nV {V:.6f}\na_mace {a_mace:.6f}\nstrain_h {H}\n"
            f"rel_tol {REL_TOL}\np_tol {P_TOL}\n")
    for n, lval, hval in zip(names, lo, hi):
        base = C_mace.get(n, sigma_mace)
        f.write(f"{n} mace {base:.4f} target [{base+lval:.4f}, {base+hval:.4f}]\n")
print(open(os.path.join(OUTDIR, "con_meta.txt")).read())
