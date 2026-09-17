#!/usr/bin/env python3
"""
Convert the carbon pandas-pickle datasets (data/C/*.pckl.gzip) into extxyz files
that train through the same path as the Al data.

Each pickle row carries a full ASE `Atoms` object (`ase_atoms`) plus `energy`
and `forces`.  We write them with the keys the ACEWorkflow trainer expects
(`build_model` uses energy_key=:dft_energy, force_key=:dft_forces):
  - info["dft_energy"]  ← total energy (eV)
  - arrays["dft_forces"]← per-atom forces (eV/Å)
Lattice and pbc come straight off the Atoms object.

ENERGY CHOICE: defaults to the raw `energy` column, the direct analogue of the
Al `dft_energy` (ACE fits its own one-body/reference term).  Pass
--energy energy_corrected to use the shifted/corrected energies instead.

Usage:
  python3 scripts/model_building/pckl_to_extxyz.py                 # train + both test tables
  python3 scripts/model_building/pckl_to_extxyz.py --energy energy_corrected
  python3 scripts/model_building/pckl_to_extxyz.py data/C/df_C_set1.pckl.gzip
"""
import argparse, os
import numpy as np
import pandas as pd
from ase.io import write

DEFAULT_INPUTS = [
    "data/C/df_C_train.pckl.gzip",
    "data/C/df_C_test_Table-4.pckl.gzip",
    "data/C/df_C_test_Table-7.pckl.gzip",
]


def convert(pckl_path, energy_col="energy", force_col="forces"):
    # pandas infers gzip only from `.gz`; our files end in `.gzip`, so be explicit
    comp = "gzip" if pckl_path.endswith(".gzip") else "infer"
    df = pd.read_pickle(pckl_path, compression=comp)
    frames = []
    for _, row in df.iterrows():
        at = row["ase_atoms"].copy()
        at.info["dft_energy"] = float(row[energy_col])
        f = np.asarray(row[force_col], dtype=float)
        assert f.shape == (len(at), 3), f"force shape {f.shape} != ({len(at)},3)"
        at.new_array("dft_forces", f)
        frames.append(at)
    out = pckl_path.replace(".pckl.gzip", ".extxyz").replace(".pckl", ".extxyz")
    write(out, frames, format="extxyz")
    nat = sum(len(a) for a in frames)
    print(f"  {os.path.basename(pckl_path):32s} -> {os.path.basename(out):28s} "
          f"{len(frames):5d} configs, {nat:7d} atoms   (energy='{energy_col}')")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*", default=DEFAULT_INPUTS)
    ap.add_argument("--energy", default="energy",
                    help="energy column: 'energy' (raw, default) or 'energy_corrected'")
    args = ap.parse_args()
    print(f"pckl -> extxyz  (dft_energy from '{args.energy}', dft_forces from 'forces')")
    for p in args.inputs:
        convert(p, energy_col=args.energy)
