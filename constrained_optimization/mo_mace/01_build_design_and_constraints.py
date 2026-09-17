"""
01_build_design_and_constraints.py — design matrix and elastic constraints for the
MACE-MPA-0 readout QP (Constrain_Mb_Elastics.pdf, Eqs. 1–3).

Per structure i:   D_i = Σ_atoms [x₀, x₁, 1]   (145,  see lib_mace_readout.py)
                   r_i = E_DFT,i − E_MACE,i
Constraints at the BCC lattice constant A0 (cubic, 2-atom cell, volume V):

   C_lo ≤ C_MACE + (160.2/V) ∂²D/∂ε_i∂ε_j · δΘ ≤ C_hi      for C11, C12, C44
   −P_TOL ≤ σ_MACE + (160.2/V) ∂D/∂ε₁ · δΘ ≤ P_TOL          (keeps A0 at zero stress,
                                                            so the C_ij are equilibrium ones)

The spec writes the window as C_exp < · < C_exp; it is realised as C_exp·(1 ± REL_TOL).

Experimental targets: Smirnova et al., PRMaterials 4, 013605 (2020).  The numbers below
are the experimental Mo values commonly quoted with those potentials (a₀ = 3.147 Å,
C11/C12/C44 = 464.7/161.5/108.9 GPa) — CHECK THEM against the paper's table; they are
the only place the experiment enters.

Outputs ($OUTDIR, default $REPO/models/Mo_MACE_MPA0_readout/):
   D_train.csv, D_test.csv          raw descriptor rows (no header), 145 columns
   E_train.csv, E_test.csv          columns: E_DFT, E_MACE, n_atoms, group_id
   theta0.csv                       current readout Θ₀ (so E_MACE = E_frozen + D·Θ₀)
   A_con.csv, l_con.csv, u_con.csv  constraint rows (GPa per unit δΘ) and bounds on δΘ
   con_meta.txt                     human-readable summary

Run:  python/mace_venv/bin/python constrained_optimization/mo_mace/01_build_design_and_constraints.py
Env:  REPO OUTDIR  A0 (Å, "exp" = 3.147, or default "mace" = MACE-MPA-0 relaxed a)
      REL_TOL (default 0.01)  P_TOL (GPa, default 0.1)  STRAIN_H (default 0.005)
      C11 C12 C44 (GPa, override targets)
"""

import os
import sys
import time

import numpy as np
from ase.io import read

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_mace_readout as L  # noqa: E402

REPO = os.environ.get("REPO", "/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow")
OUTDIR = os.environ.get("OUTDIR", os.path.join(REPO, "models", "Mo_MACE_MPA0_readout"))
os.makedirs(OUTDIR, exist_ok=True)

A_EXP = 3.147
TARGETS = {  # GPa — Smirnova et al. 2020 experimental column (verify)
    "C11": float(os.environ.get("C11", 464.7)),
    "C12": float(os.environ.get("C12", 161.5)),
    "C44": float(os.environ.get("C44", 108.9)),
}
REL_TOL = float(os.environ.get("REL_TOL", 0.01))
P_TOL = float(os.environ.get("P_TOL", 0.1))
H = float(os.environ.get("STRAIN_H", 0.005))
A0_SPEC = os.environ.get("A0", "mace")

calc = L.load_calc()
model = calc.models[0]
hooks = L.ReadoutHooks(model)
th0 = L.theta0(model)
print(f"readout basis: {hooks.n0} + {hooks.n1} + 1 = {len(th0)} columns")

# ── design matrices ──────────────────────────────────────────────────────────
groups = ["Elastic", "AIMD-NVT", "Vacancy", "Surface"]
for split in ("train", "test"):
    frames = read(os.path.join(REPO, "data", "Mo", f"mlearn_Mo_{split}.extxyz"), ":")
    t = time.time()
    Ds, Es = [], []
    for at in frames:
        E_dft = at.get_potential_energy()
        E_mace, D = L.energy_and_descriptor(calc, at, hooks)
        Ds.append(D)
        Es.append([E_dft, E_mace, len(at), groups.index(at.info["config_type"])])
    Ds, Es = np.array(Ds), np.array(Es)
    np.savetxt(os.path.join(OUTDIR, f"D_{split}.csv"), Ds, delimiter=",")
    np.savetxt(os.path.join(OUTDIR, f"E_{split}.csv"), Es, delimiter=",")
    r = (Es[:, 0] - Es[:, 1]) / Es[:, 2]
    print(f"{split}: {len(frames)} configs in {time.time()-t:.0f}s; "
          f"(E_DFT−E_MACE)/N mean {r.mean():+.4f} eV, std {1e3*r.std():.2f} meV/atom")

# the linear identity, on a few real configs (must be ~1e-10 eV)
for at in frames[:3]:
    at = at.copy(); at.calc = None
    err = L.check_linear_identity(calc, at, hooks, th0)
    print(f"  identity check E_MACE − D·Θ₀ − E_frozen = {err:+.2e} eV  ({len(at)} atoms)")
    assert abs(err) < 1e-6, "readout basis is not an exact linearisation of the model"
np.savetxt(os.path.join(OUTDIR, "theta0.csv"), th0, delimiter=",")

# ── lattice constant ─────────────────────────────────────────────────────────
a_mace = L.relaxed_bcc_a(calc)
A0 = {"exp": A_EXP, "mace": a_mace}.get(A0_SPEC)
A0 = float(A0_SPEC) if A0 is None else A0
print(f"MACE relaxed a = {a_mace:.4f} Å;  constraining at A0 = {A0:.4f} Å ({A0_SPEC})")
cell = L.bcc_cubic(A0)
V = cell.get_volume()


def energy(at):
    at.calc = calc
    return at.get_potential_energy()


def descriptor(at):
    return L.energy_and_descriptor(calc, at, hooks)[1]


dE = L.strain_derivatives(energy, cell, H)
dD = L.strain_derivatives(descriptor, cell, H)
dE_2h = L.strain_derivatives(energy, cell, 2 * H)
k = L.EV_A3_TO_GPA / V

C_mace = {"C11": k * dE["d11"], "C12": k * dE["d12"], "C44": k * dE["d44"]}
sigma_mace = k * dE["d1"]
print(f"MACE-MPA-0 at A0: C11 {C_mace['C11']:.1f}  C12 {C_mace['C12']:.1f}  "
      f"C44 {C_mace['C44']:.1f} GPa   ∂E/∂ε₁/V {sigma_mace:+.2f} GPa")
print(f"  (step 2h: C11 {k*dE_2h['d11']:.1f}  C12 {k*dE_2h['d12']:.1f}  C44 {k*dE_2h['d44']:.1f})")
# consistency: the D derivatives must reproduce the MACE derivatives through Θ₀
for key in ("d1", "d11", "d12", "d44"):
    assert abs(dD[key] @ th0 - dE[key]) < 1e-5 * max(1.0, abs(dE[key])), key

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
    f.write(f"A0 {A0:.6f}\nV {V:.6f}\na_mace {a_mace:.6f}\nstrain_h {H}\n"
            f"rel_tol {REL_TOL}\np_tol {P_TOL}\n")
    for n, lval, hval in zip(names, lo, hi):
        base = C_mace.get(n, sigma_mace)
        f.write(f"{n} mace {base:.4f} target [{base+lval:.4f}, {base+hval:.4f}]\n")
print(open(os.path.join(OUTDIR, "con_meta.txt")).read())
print(f"→ {OUTDIR}")
