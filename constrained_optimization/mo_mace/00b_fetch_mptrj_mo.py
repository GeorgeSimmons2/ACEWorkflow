"""
00b_fetch_mptrj_mo.py — the Mo frames of MPtrj, the data MACE-MP-0/MPA-0 were trained on.

This is the alternative to 00_fetch_mlearn_mo.py.  mlearn (Zuo et al. 2020) is an independent
benchmarking set, so its VASP reference differs from MPtrj's by a constant per atom; MPtrj IS the
foundation training set, so the fit residual is what the foundation model actually left behind and
the offset column should come out near zero (worth checking as a sanity test).

ENERGY FIELD.  MACE-MP models were trained on the RAW VASP energies, `uncorrected_total_energy`,
not `corrected_total_energy` (which carries MP2020 GGA/GGA+U and anion corrections).  Picking the
wrong one shows up as a large constant offset.  Override with ENERGY_KEY if that changes.

The source is MPtrj_2022.9_full.json, 12.2 GB (figshare 23713842, file 41619375):
  mp_id -> frame_id -> {structure (pymatgen dict), uncorrected_total_energy, force, stress, ...}
It is streamed with ijson (yajl2_c backend), so only one material is in memory at a time.
NOTE the 1.18 GB zip on that record (chgnet_mptraj.json) is a DIFFERENT, JARVIS-style
repackaging: a flat list with per-atom `total_energy` and no uncorrected/corrected split.  It is
not what MACE was trained on, so it is not used here.

Writes  $REPO/data/Mo/<tag>_Mo_{train,test}.extxyz  (tag = mptrj for PURE_MO=1, mptrjall for
PURE_MO=0, so the two extractions cannot overwrite each other)  (energy eV, forces eV/A, config_type = mp_id).
The split is by MATERIAL (mp_id), not by frame, so frames of one relaxation trajectory cannot
straddle it — frames within a trajectory are near-duplicates and would leak.

Run:  python/mace_venv/bin/python constrained_optimization/mo_mace/00b_fetch_mptrj_mo.py
Env:  REPO  SRC (json path)  PURE_MO (1 = only frames of pure Mo; 0 = every Mo-containing frame)
      MAX_FRAMES (cap, 0 = no cap; sampled across materials AFTER the scan, not truncated)
      TEST_FRAC (0.1)  SEED (0)  ENERGY_KEY (uncorrected_total_energy)
"""

import os
import random

import ijson
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

REPO = os.environ.get("REPO", "/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow")
OUT = os.path.join(REPO, "data", "Mo")
SRC = os.environ.get("SRC", os.path.join(OUT, "mptrj", "MPtrj_2022.9_full.json"))
PURE_MO = os.environ.get("PURE_MO", "1") == "1"
TAG = "mptrj" if PURE_MO else "mptrjall"
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", 0))
TEST_FRAC = float(os.environ.get("TEST_FRAC", 0.1))
SEED = int(os.environ.get("SEED", 0))
ENERGY_KEY = os.environ.get("ENERGY_KEY", "uncorrected_total_energy")


def to_atoms(frame, name):
    s = frame["structure"]
    cell = np.array(s["lattice"]["matrix"], dtype=float)
    sym = [site["species"][0]["element"] for site in s["sites"]]
    pos = np.array([site["xyz"] for site in s["sites"]], dtype=float)
    E = frame.get(ENERGY_KEY)
    F = frame.get("force")
    if E is None or F is None:
        return None
    at = Atoms(sym, positions=pos, cell=cell, pbc=True)
    at.calc = SinglePointCalculator(at, energy=float(E), forces=np.array(F, dtype=float))
    at.info["config_type"] = name
    return at


def main():
    print(f"streaming {SRC} ({os.path.getsize(SRC) / 1e9:.1f} GB, ijson {ijson.backend})", flush=True)

    by_material, n_seen, n_kept = {}, 0, 0
    with open(SRC, "rb") as fh:
        for mp_id, frames in ijson.kvitems(fh, "", use_float=True):
            kept = []
            for frame_id, frame in frames.items():
                n_seen += 1
                sites = frame.get("structure", {}).get("sites", [])
                els = {site["species"][0]["element"] for site in sites}
                if "Mo" not in els or (PURE_MO and els != {"Mo"}):
                    continue
                at = to_atoms(frame, mp_id)
                if at is not None:
                    kept.append(at)
                    n_kept += 1
            if kept:
                by_material[mp_id] = kept
            if n_seen % 200000 < len(frames):
                print(f"  {n_seen} frames scanned, {n_kept} kept, "
                      f"{len(by_material)} materials", flush=True)

    mats = sorted(by_material)
    rng = random.Random(SEED)
    rng.shuffle(mats)
    if MAX_FRAMES and n_kept > MAX_FRAMES:
        # thin every material by the same factor, so the cap keeps the spread over materials
        # (truncating the scan would keep only whatever came first in the file)
        keep = {}
        order = [(m, i) for m in mats for i in range(len(by_material[m]))]
        rng.shuffle(order)
        for m, i in order[:MAX_FRAMES]:
            keep.setdefault(m, []).append(by_material[m][i])
        by_material = {m: v for m, v in keep.items() if v}
        mats = [m for m in mats if m in by_material]
        n_kept = sum(len(v) for v in by_material.values())
        print(f"  capped to {n_kept} frames over {len(mats)} materials (MAX_FRAMES={MAX_FRAMES})")
    n_test = max(1, round(TEST_FRAC * len(mats)))
    split = {"test": mats[:n_test], "train": mats[n_test:]}
    print(f"\n{n_kept} frames from {len(mats)} materials "
          f"({'pure Mo' if PURE_MO else 'Mo-containing'}); {n_seen} frames scanned")

    for name, ids in split.items():
        frames = [at for m in ids for at in by_material[m]]
        path = os.path.join(OUT, f"{TAG}_Mo_{name}.extxyz")
        write(path, frames, format="extxyz")
        sizes = [len(a) for a in frames]
        print(f"{name}: {len(frames)} frames from {len(ids)} materials, "
              f"{sum(sizes)} atoms (max cell {max(sizes) if sizes else 0}) -> {path}")


if __name__ == "__main__":
    main()
