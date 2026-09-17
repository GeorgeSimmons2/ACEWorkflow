"""
03_verify_corrected_model.py — build the corrected MACE-MPA-0 and measure it directly.

The QP only guarantees the *linearised* C_ij.  Here δΘ is written into the readout
weights (lib_mace_readout.apply_correction) and everything is recomputed from the
patched model itself:

  1. identity: E_patched − (E_MACE + D·δΘ) on test configs            (must be ~1e-8 eV)
  2. C11, C12, C44 and ∂E/∂ε₁/V at A0 from finite differences of the patched model
  3. the patched model's own relaxed a and its C_ij there
  4. energy and force RMSE vs DFT (train/test) — forces were NOT in the fit, so this is
     the honest check that the readout change has not damaged them

Saves the patched models as $OUTDIR/mace_mpa0_mo_<fit>.model (loadable with
mace.calculators.MACECalculator(model_paths=..., default_dtype="float64")).

Run:  python/mace_venv/bin/python constrained_optimization/mo_mace/03_verify_corrected_model.py
Env:  REPO OUTDIR  FITS (comma list from offset,ridge,constrained; default all)  STRAIN_H
"""

import os
import sys

import numpy as np
import torch
from ase.io import read

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_mace_readout as L  # noqa: E402

REPO = os.environ.get("REPO", "/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow")
OUTDIR = os.environ.get("OUTDIR", os.path.join(REPO, "models", "Mo_MACE_MPA0_readout"))
FITS = os.environ.get("FITS", "offset,ridge,constrained").split(",")
H = float(os.environ.get("STRAIN_H", 0.005))

meta = {}
for line in open(os.path.join(OUTDIR, "con_meta.txt")):
    parts = line.split()
    meta[parts[0]] = parts
A0 = float(meta["A0"][1])

calc = L.load_calc()
base_model = calc.models[0]
hooks = L.ReadoutHooks(base_model)
data = {s: read(os.path.join(REPO, "data", "Mo", f"mlearn_Mo_{s}.extxyz"), ":") for s in ("train", "test")}
D_test = np.loadtxt(os.path.join(OUTDIR, "D_test.csv"), delimiter=",")
E_test = np.loadtxt(os.path.join(OUTDIR, "E_test.csv"), delimiter=",")


def elastic(calc, a):
    cell = L.bcc_cubic(a)
    k = L.EV_A3_TO_GPA / cell.get_volume()

    def energy(at):
        at.calc = calc
        return at.get_potential_energy()

    d = L.strain_derivatives(energy, cell, H)
    return k * np.array([d["d11"], d["d12"], d["d44"], d["d1"]])


def errors(calc, frames, offset):
    dE, dF = [], []
    for at in frames:
        E_ref, F_ref = at.get_potential_energy(), at.get_forces()
        x = at.copy()
        x.calc = calc
        dE.append((x.get_potential_energy() - E_ref) / len(x) + offset)
        dF.append((x.get_forces() - F_ref).ravel())
    return 1e3 * np.sqrt(np.mean(np.square(dE))), 1e3 * np.sqrt(np.mean(np.square(np.concatenate(dF))))


fmt = lambda v: "  ".join(f"{x:8.2f}" for x in v)  # noqa: E731
print(f"targets  (C11 C12 C44 σ):  lo {fmt([float(meta[c][4].strip('[,')) for c in ('C11','C12','C44','dE/de1/V')])}")
print(f"                           hi {fmt([float(meta[c][5].strip(']')) for c in ('C11','C12','C44','dE/de1/V')])}")
C_base = elastic(calc, A0)
print(f"MACE-MPA-0 at A0={A0:.4f}:     {fmt(C_base)}")

rows = [("MACE-MPA-0", base_model, np.zeros(D_test.shape[1]))]
for fit in FITS:
    rows.append((fit, None, np.loadtxt(os.path.join(OUTDIR, f"delta_theta_{fit}.csv"), delimiter=",")))

print("\nfit            identity   C11      C12      C44    σ@A0  |  a_relax  C11      C12      C44"
      "  |  E rmse tr/te (meV/at)  F rmse tr/te (meV/Å)")
for name, model, dtheta in rows:
    model = model if model is not None else L.apply_correction(base_model, dtheta)
    calc.models[0] = model
    try:
        ident = 0.0
        if name != "MACE-MPA-0":
            for j, at in enumerate(data["test"][:5]):
                x = at.copy(); x.calc = calc
                ident = max(ident, abs(x.get_potential_energy() - (E_test[j, 1] + D_test[j] @ dtheta)))
        C_A0 = elastic(calc, A0)
        a_rel = L.relaxed_bcc_a(calc, a_guess=A0)
        C_rel = elastic(calc, a_rel)
        # reference offset: MACE-MPA-0 and the offset-free fits are compared after removing
        # the mean per-atom shift, so the RMSE measures shape, not the PBE reference
        E_tr = np.loadtxt(os.path.join(OUTDIR, "E_train.csv"), delimiter=",")
        off = 0.0 if abs(dtheta[-1]) > 0 else np.mean((E_tr[:, 0] - E_tr[:, 1]) / E_tr[:, 2])
        e_tr, f_tr = errors(calc, data["train"], off)
        e_te, f_te = errors(calc, data["test"], off)
        print(f"{name:12s}  {ident:8.1e}  {fmt(C_A0)}  |  {a_rel:.4f}  {fmt(C_rel[:3])}"
              f"  |  {e_tr:7.2f} / {e_te:7.2f}       {f_tr:7.1f} / {f_te:7.1f}", flush=True)
        if name != "MACE-MPA-0":
            path = os.path.join(OUTDIR, f"mace_mpa0_mo_{name}.model")
            torch.save(model, path)
    finally:
        calc.models[0] = base_model
print(f"\npatched models → {OUTDIR}/mace_mpa0_mo_<fit>.model")
