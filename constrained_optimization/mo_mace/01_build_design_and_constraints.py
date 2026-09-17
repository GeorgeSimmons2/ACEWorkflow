"""
01_build_design_and_constraints.py — design matrix, force normal equations and elastic
constraints for the MACE-MPA-0 readout QP (Constrain_Mb_Elastics.pdf, Eqs. 1–3).

Per structure i:   D_i = Σ_atoms descriptor (DESCRIPTOR=perez: 257 cols; readout: 145 —
                   see lib_mace_readout.py),   r_i = E_DFT,i − E_MACE,i
Corrected model:   E(δΘ) = E_MACE + D·δΘ        (δΘ = 0 is MACE-MPA-0)
                   F(δΘ) = F_MACE + G·δΘ,  G = −∂D/∂r  (3N × ncol, exact reverse-mode autograd)

Forces (FORCES=1, default): G is never written; per split and per config group the script
stores the force normal equations, from which the QP and every force RMSE follow exactly:
   GtG = Σ GᵀG,  Gtf = Σ GᵀΔF,  ftf = Σ ΔFᵀΔF,  nF = Σ 3N        with ΔF = F_DFT − F_MACE
   ‖ΔF − Gδ‖² = ftf − 2δ·Gtf + δᵀ GtG δ
Cost: one batched reverse pass over all descriptor columns per config (~10× a looped
backward).  Configs are spread over N_WORKERS processes (MACE on CPU saturates ~10 threads).

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

Checks (exact, run FIRST, must pass): for a random δΘ the patched MACE model reproduces
E_MACE + D·δΘ, F_MACE + G·δΘ (on an AIMD config, non-zero forces) and
∂E/∂ε = ∂E_MACE/∂ε + ∂D/∂ε·δΘ.

Outputs ($OUTDIR, default $REPO/models/Mo_MACE_MPA0_readout/$DESCRIPTOR/):
   D_train.csv, D_test.csv          descriptor rows (no header)
   E_train.csv, E_test.csv          columns: E_DFT, E_MACE, n_atoms, group_id
   GtG_<split>_g<k>.csv, Gtf_<split>_g<k>.csv, fstats_<split>.csv (rows k: ftf, nF)   [FORCES=1]
   A_con.csv, l_con.csv, u_con.csv  constraint rows (GPa per unit δΘ) and bounds on δΘ
   con_meta.txt                     human-readable summary (read by 02 and 03)

Run:  python/mace_venv/bin/python constrained_optimization/mo_mace/01_build_design_and_constraints.py
Env:  REPO OUTDIR DESCRIPTOR(perez|readout)  A0 (Å, "exp" = 3.147 (298 K), default "mace")
      REL_TOL (0.01)  P_TOL (GPa, 0.1)  STRAIN_H (0.005)  C11 C12 C44 (GPa)
      FORCES (1)  N_WORKERS (4; threads each = OMP_NUM_THREADS / N_WORKERS)
"""

import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch
from ase.io import read

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_mace_readout as L  # noqa: E402

REPO = os.environ.get("REPO", "/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow")
MODE = L.descriptor_mode()
OUTDIR = os.environ.get("OUTDIR", os.path.join(REPO, "models", "Mo_MACE_MPA0_readout", MODE))

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
FORCES = os.environ.get("FORCES", "1") == "1"
N_WORKERS = int(os.environ.get("N_WORKERS", 4))
THREADS = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count()))
GROUPS = ["Elastic", "AIMD-NVT", "Vacancy", "Surface"]


def exactness_checks(calc, model, hooks, frames, ncol):
    """Random δΘ: patched model vs linear prediction for energies and forces."""
    rng = np.random.default_rng(0)
    D_sample = np.array([L.energy_and_descriptor(calc, a, hooks)[1] for a in frames[:10]])
    col_std = (D_sample / D_sample[:, -1:]).std(0)
    col_std[col_std < 1e-12] = 1.0
    dtheta = 1e-3 * rng.standard_normal(ncol) / col_std
    calc_test = L.load_calc()
    calc_test.models[0] = L.apply_correction(model, dtheta, MODE)
    for at in frames[:3]:
        E_mace, D = L.energy_and_descriptor(calc, at, hooks)
        x = at.copy(); x.calc = calc_test
        err = x.get_potential_energy() - (E_mace + D @ dtheta)
        print(f"  identity E_patched − (E_MACE + D·δΘ) = {err:+.2e} eV  (D·δΘ = {D @ dtheta:+.3e}, {len(at)} atoms)")
        assert abs(err) < 1e-6, "descriptor is not an exact linearisation of the correction"
    if FORCES:
        at = next(a for a in frames if a.info["config_type"] == "AIMD-NVT")  # non-zero forces
        x0 = at.copy(); x0.calc = calc
        x1 = at.copy(); x1.calc = calc_test
        dF_model = (x1.get_forces() - x0.get_forces()).ravel()
        Gd = L.descriptor_force_jacobian(calc, at, hooks) @ dtheta
        err = np.abs(dF_model - Gd).max()
        print(f"  force identity max|F_patched − F_MACE − G·δΘ| = {err:.2e} eV/Å  (max|G·δΘ| = {np.abs(Gd).max():.2e})")
        assert np.abs(Gd).max() > 1e-8 and err < 1e-6 * max(1.0, np.abs(Gd).max()), "force Jacobian is wrong"
    return calc_test, dtheta


def build_split(split, frames, ncol):
    t = time.time()
    jobs = [(a.copy(), a.get_potential_energy(), a.get_forces().copy() if FORCES else None) for a in frames]
    for a, _, _ in jobs:
        a.calc = None
    Ds, Es = [], []
    GtG = np.zeros((len(GROUPS), ncol, ncol))
    Gtf = np.zeros((len(GROUPS), ncol))
    fst = np.zeros((len(GROUPS), 2))
    ctx = mp.get_context("spawn")
    with ctx.Pool(N_WORKERS, initializer=L.init_worker, initargs=(MODE, THREADS // N_WORKERS)) as pool:
        for ic, (at, out) in enumerate(zip(frames, pool.imap(L.config_blocks, jobs))):
            g = GROUPS.index(at.info["config_type"])
            Ds.append(out["D"])
            Es.append([out["E_dft"], out["E_mace"], out["n"], g])
            if FORCES:
                GtG[g] += out["GtG"]
                Gtf[g] += out["Gtf"]
                fst[g] += [out["ftf"], out["nF"]]
            if ic % 10 == 0 or ic == len(frames) - 1:
                print(f"  {split} {ic+1}/{len(frames)}  {time.time()-t:.0f}s", flush=True)
    Ds, Es = np.array(Ds), np.array(Es)
    np.savetxt(os.path.join(OUTDIR, f"D_{split}.csv"), Ds, delimiter=",")
    np.savetxt(os.path.join(OUTDIR, f"E_{split}.csv"), Es, delimiter=",")
    r = (Es[:, 0] - Es[:, 1]) / Es[:, 2]
    print(f"{split}: {len(Ds)} configs in {time.time()-t:.0f}s; "
          f"(E_DFT−E_MACE)/N mean {r.mean():+.4f} eV, std {1e3*r.std():.2f} meV/atom", flush=True)
    if FORCES:
        for k in range(len(GROUPS)):
            np.savetxt(os.path.join(OUTDIR, f"GtG_{split}_g{k}.csv"), GtG[k], delimiter=",")
            np.savetxt(os.path.join(OUTDIR, f"Gtf_{split}_g{k}.csv"), Gtf[k], delimiter=",")
        np.savetxt(os.path.join(OUTDIR, f"fstats_{split}.csv"), fst, delimiter=",")
        print(f"  {split} MACE-MPA-0 force RMSE {1e3*np.sqrt(fst[:,0].sum()/fst[:,1].sum()):.1f} meV/Å", flush=True)


def constraints(calc, calc_test, hooks, dtheta_test):
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


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    torch.set_num_threads(THREADS)
    calc = L.load_calc()
    model = calc.models[0]
    hooks = L.ReadoutHooks(model, MODE)
    ncol = L.n_columns(model, MODE)
    print(f"DESCRIPTOR={MODE}: {ncol} columns, FORCES={int(FORCES)}, "
          f"{N_WORKERS} workers × {THREADS // N_WORKERS} threads → {OUTDIR}", flush=True)
    frames = {s: read(os.path.join(REPO, "data", "Mo", f"mlearn_Mo_{s}.extxyz"), ":") for s in ("train", "test")}

    calc_test, dtheta_test = exactness_checks(calc, model, hooks, frames["train"], ncol)
    for split in ("train", "test"):
        build_split(split, frames[split], ncol)
    constraints(calc, calc_test, hooks, dtheta_test)


if __name__ == "__main__":
    main()
