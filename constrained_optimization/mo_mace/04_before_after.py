"""
04_before_after.py — test-set energy and force errors and C11, C12, C44: MACE-MPA-0 vs constrained.

Loads the SAVED model files written by 03 (not an in-memory patch), so this also checks the
deployable artefact.

  before : MACE-MPA-0 as downloaded.  Energy errors are quoted raw AND after removing the
           per-atom PBE-reference offset (the offset-only fit, delta_theta_offset.csv);
           the second is the fair comparison — the offset is a reference shift, not physics.
  after  : mace_mpa0_mo_constrained.model

Energies: per-atom RMSE and MAE on data/Mo/mlearn_Mo_test.extxyz, overall and per group.
Forces: component RMSE and MAE (meV/Å), overall and per group (the offset does not touch forces).
C_ij: finite differences at each model's OWN relaxed BCC lattice constant, and at A0 from 01.

Writes $OUTDIR/before_after.csv  (rows: model; columns: metrics).

Run:  python/mace_venv/bin/python constrained_optimization/mo_mace/04_before_after.py
Env:  REPO OUTDIR DESCRIPTOR  AFTER (model file name, default mace_mpa0_mo_constrained.model)
      STRAIN_H  C11 C12 C44 (reference values printed alongside; default Dickinson & Armstrong 273 K)
"""

import csv
import os
import sys

import numpy as np
from ase.io import read

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # the perez pickle references lib_mace_readout.LinearCorrectedReadout
import lib_mace_readout as L  # noqa: E402
from mace.calculators import MACECalculator  # noqa: E402

REPO = os.environ.get("REPO", "/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow")
MODE = L.descriptor_mode()
OUTDIR = os.environ.get("OUTDIR", os.path.join(REPO, "models", "Mo_MACE_MPA0_readout", MODE))
AFTER = os.environ.get("AFTER", "mace_mpa0_mo_constrained.model")
H = float(os.environ.get("STRAIN_H", 0.005))
EXP = [float(os.environ.get(k, v)) for k, v in (("C11", 463.7), ("C12", 157.8), ("C44", 109.2))]
GROUPS = ["Elastic", "AIMD-NVT", "Vacancy", "Surface"]

A0 = float(next(l.split()[1] for l in open(os.path.join(OUTDIR, "con_meta.txt")) if l.startswith("A0 ")))
offset = float(np.loadtxt(os.path.join(OUTDIR, "delta_theta_offset.csv"), delimiter=",")[-1])
test = read(os.path.join(REPO, "data", "Mo", "mlearn_Mo_test.extxyz"), ":")

before = L.load_calc()
after = MACECalculator(model_paths=os.path.join(OUTDIR, AFTER), device="cpu", default_dtype="float64")


def elastic(calc, a):
    cell = L.bcc_cubic(a)
    k = L.EV_A3_TO_GPA / cell.get_volume()

    def energy(at):
        at.calc = calc
        return at.get_potential_energy()

    d = L.strain_derivatives(energy, cell, H)
    return k * np.array([d["d11"], d["d12"], d["d44"]])


def per_atom_errors(calc, shift=0.0):
    err, grp, ferr, fgrp = [], [], [], []
    for at in test:
        x = at.copy()
        x.calc = calc
        err.append((x.get_potential_energy() - at.get_potential_energy()) / len(at) + shift)
        grp.append(at.info["config_type"])
        df = (x.get_forces() - at.get_forces()).ravel()
        ferr.append(df)
        fgrp += [at.info["config_type"]] * df.size
    return 1e3 * np.array(err), np.array(grp), 1e3 * np.concatenate(ferr), np.array(fgrp)  # meV/atom, meV/Å


def summarise(name, calc, shift, elastics=True):
    e, g, f, fg = per_atom_errors(calc, shift)
    row = {"model": name, "E_rmse_test": np.sqrt(np.mean(e**2)), "E_mae_test": np.mean(np.abs(e)),
           "F_rmse_test": np.sqrt(np.mean(f**2)), "F_mae_test": np.mean(np.abs(f))}
    for grp in GROUPS:
        m = g == grp
        row[f"E_rmse_{grp}"] = np.sqrt(np.mean(e[m] ** 2)) if m.any() else np.nan
        m = fg == grp
        row[f"F_rmse_{grp}"] = np.sqrt(np.mean(f[m] ** 2)) if m.any() else np.nan
    if not elastics:
        return row
    a = L.relaxed_bcc_a(calc, a_guess=A0)
    row["a_relaxed"] = a
    for key, v in zip(("C11", "C12", "C44"), elastic(calc, a)):
        row[key] = v
    for key, v in zip(("C11_A0", "C12_A0", "C44_A0"), elastic(calc, A0)):
        row[key] = v
    return row


rows = [
    summarise("MACE-MPA-0 (raw)", before, 0.0, elastics=False),
    summarise("MACE-MPA-0 (+offset)", before, offset),
    summarise(f"constrained ({AFTER})", after, 0.0),
]
# (+offset) row: E_MACE + offset·N, offset = delta_theta_offset[-1] = mean_train (E_DFT − E_MACE)/N,
# so the per-atom error is (E_MACE − E_DFT)/N + offset.  The offset does not change C_ij.

print(f"\nMo test set: {len(test)} configs   A0 = {A0:.4f} Å   descriptor = {MODE}")
print(f"\n{'':32s}{'E RMSE':>9s}{'E MAE':>9s}" + "".join(f"{g:>10s}" for g in GROUPS) + "   (meV/atom)")
for r in rows:
    print(f"{r['model']:32s}{r['E_rmse_test']:9.2f}{r['E_mae_test']:9.2f}"
          + "".join(f"{r[f'E_rmse_{g}']:10.2f}" for g in GROUPS))
print(f"\n{'':32s}{'F RMSE':>9s}{'F MAE':>9s}" + "".join(f"{g:>10s}" for g in GROUPS) + "   (meV/Å)")
for r in rows[1:]:
    print(f"{r['model']:32s}{r['F_rmse_test']:9.1f}{r['F_mae_test']:9.1f}"
          + "".join(f"{r[f'F_rmse_{g}']:10.1f}" for g in GROUPS))

print(f"\n{'':32s}{'a (Å)':>8s}{'C11':>9s}{'C12':>9s}{'C44':>9s}   at own a   |{'C11':>9s}{'C12':>9s}{'C44':>9s}   at A0 (GPa)")
print(f"{'experiment':32s}{'':8s}" + "".join(f"{v:9.1f}" for v in EXP))
for r in rows[1:]:
    print(f"{r['model']:32s}{r['a_relaxed']:8.4f}" + "".join(f"{r[k]:9.1f}" for k in ("C11", "C12", "C44"))
          + "              |" + "".join(f"{r[k]:9.1f}" for k in ("C11_A0", "C12_A0", "C44_A0")))
print("\nerror vs experiment at own a, before → after:  "
      + "   ".join(f"{k} {100*(rows[1][k]-x)/x:+.1f}% → {100*(rows[2][k]-x)/x:+.1f}%"
                   for k, x in zip(("C11", "C12", "C44"), EXP)))

path = os.path.join(OUTDIR, "before_after.csv")
with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[-1].keys()), restval="")
    w.writeheader()
    w.writerows(rows)
print(f"\n→ {path}")
