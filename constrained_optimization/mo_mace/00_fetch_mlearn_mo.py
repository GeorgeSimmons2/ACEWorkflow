"""
00_fetch_mlearn_mo.py — the Mo DFT data for Eq. (2) of Constrain_Mb_Elastics.pdf.

The spec does not name a DFT set.  This uses the public Mo set of Zuo et al.,
J. Phys. Chem. A 124, 731 (2020) (github.com/materialsvirtuallab/mlearn, data/Mo):
PBE / VASP, 194 train + 23 test configurations in four groups
(Elastic, AIMD-NVT, Vacancy, Surface).  Its PBE reference differs from MPtrj's by a
constant per atom, which the constant column of D absorbs.

Writes  $REPO/data/Mo/mlearn_Mo_{train,test}.extxyz  (energy in eV, forces in eV/Å,
config_type = mlearn group).

Run:  python/mace_venv/bin/python constrained_optimization/mo_mace/00_fetch_mlearn_mo.py
"""

import json
import os
import urllib.request

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

REPO = os.environ.get("REPO", "/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow")
OUT = os.path.join(REPO, "data", "Mo")
URL = "https://raw.githubusercontent.com/materialsvirtuallab/mlearn/master/data/Mo/{}.json"


def to_atoms(rec):
    s = rec["structure"]
    cell = np.array(s["lattice"]["matrix"])
    sym = [site["species"][0]["element"] for site in s["sites"]]
    pos = np.array([site["xyz"] for site in s["sites"]])
    at = Atoms(sym, positions=pos, cell=cell, pbc=True)
    E = float(rec["outputs"]["energy"])
    F = np.array(rec["outputs"]["forces"])
    at.calc = SinglePointCalculator(at, energy=E, forces=F)
    at.info["config_type"] = rec["group"]
    at.info["description"] = rec["description"].replace('"', "'")
    assert len(at) == rec["num_atoms"]
    return at


os.makedirs(OUT, exist_ok=True)
for src, dst in (("training", "train"), ("test", "test")):
    raw = os.path.join(OUT, f"Mo_{src}.json")
    if not os.path.isfile(raw):
        urllib.request.urlretrieve(URL.format(src), raw)
    frames = [to_atoms(r) for r in json.load(open(raw))]
    path = os.path.join(OUT, f"mlearn_Mo_{dst}.extxyz")
    write(path, frames, format="extxyz")
    print(f"{len(frames):4d} configs → {path}")
