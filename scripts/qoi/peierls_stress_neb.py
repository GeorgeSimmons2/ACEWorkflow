#!/usr/bin/env python3
"""
peierls_stress_neb.py

Publishable-quality Peierls-stress estimate for an FCC {111}<112> Shockley
partial in Al, via strain-controlled NEB (epsilon-NEB), following the
methodology of:

    Si, Zhang, Chen, Wormald, Anglin, McDowell & Zhu, "Atomistic
    determination of Peierls barriers of dislocation glide in nickel,"
    J. Mech. Phys. Solids 178 (2023) 105359.
    https://doi.org/10.1016/j.jmps.2023.105359

  1. Build a Shockley-partial glide cylinder and its two glide-endpoint
     configurations with matscipy.dislocation (FCCEdgeShockleyPartial /
     FCCScrewShockleyPartial), following the pattern in matscipy's own
     tutorial notebook (docs/applications/disloc_mobility.ipynb).
  2. For a range of applied engineering shear strains gamma, apply a
     uniform simple shear to ALL atoms (interior + the FixAtoms boundary
     shell), then re-relax the interior with the boundary shell fixed at
     its new, sheared positions -- the cylindrical analogue of Si et al.'s
     slab-based epsilon-NEB protocol (Section 3.1 of the paper above).
  3. NEB-relax the band between the two strained endpoints (ase.mep.NEB,
     climbing image) and extract the barrier via ase.mep.NEBTools.
  4. Fit barrier vs. resolved shear stress to the Zhu et al. (2008)
     power law, Q(tau) = A * (1 - tau/tau_P)**alpha (Si et al. Eq. 2).
     The fitted tau_P is the Peierls stress: the resolved shear stress at
     which the barrier vanishes.

Uses the ACE potential via pyjulip (https://github.com/casv2/pyjulip),
which loads an ACEpotentials.jl model .json directly as an ASE calculator
-- point MODEL_JSON at either the nominal or exact_constrained_model.json
produced by scripts/model_building/ and scripts/elasticity/ in this repo.

── Caveats (read before trusting the numbers) ──────────────────────────────
* Axis convention: this assumes matscipy's cylinder builders use x = glide
  direction (in-plane, perpendicular to the dislocation line), y = glide-
  plane normal, z = dislocation line -- inferred from matscipy's own
  make_barrier_configurations (`center = [cent_x, 0.0, 0.0]`) and the
  documented z-periodicity of dislocation cylinders. Sanity-check this
  against `disloc.view_cyl(glide_ini)` for your installed matscipy version
  before trusting results.
* resolved_shear_stress() converts strain -> stress via the cubic C44
  shear modulus. This is a Voigt-style estimate valid near the <100> axes,
  not an exact anisotropic (Stroh) resolved stress on the actual {111}
  glide plane -- fine for a relative comparison across models, not for a
  literature-grade absolute Peierls stress.
* This models the LEADING Shockley partial only (nucleating a/6<112> from
  the perfect lattice), not the full dissociated <110> glide event. Use
  matscipy's CubicCrystalDissociatedDislocation / build_glide_quadrupoles
  if you need the trailing-partial barrier too.
* Compute cost: N_STRAINS full NEB relaxations, each with N_IMAGES ACE
  evaluations per optimizer step. Start with a small cylinder radius and
  few images/strains to sanity-check before scaling up.

Usage:
    python scripts/qoi/peierls_stress_neb.py \\
        --model-json models/Al_20_4_6A_4/exact_constrained_model.json \\
        --character edge --cylinder-radius 40 --n-images 7 \\
        --gamma-max 0.02 --n-strains 5
"""

import argparse

import numpy as np
from scipy.optimize import curve_fit

from ase.mep import NEB, NEBTools
from ase.optimize import FIRE

from matscipy.dislocation import (
    get_elastic_constants,
    FCCEdgeShockleyPartial,
    FCCScrewShockleyPartial,
)

import pyjulip


# ── Calculator ───────────────────────────────────────────────────────────

def load_calculator(model_json):
    """ASE calculator for an ACEpotentials.jl model .json, via pyjulip."""
    return pyjulip.ACE1(model_json)


# ── Dislocation setup ────────────────────────────────────────────────────

def build_dislocation(calc, element, character):
    alat, C11, C12, C44 = get_elastic_constants(calculator=calc, symbol=element, verbose=False)
    cls = FCCEdgeShockleyPartial if character == "edge" else FCCScrewShockleyPartial
    disloc = cls(alat, C11, C12, C44, symbol=element)
    return disloc, alat, C11, C12, C44


# ── Strain-controlled (epsilon-NEB) boundary condition ──────────────────

def apply_resolved_shear(atoms, gamma, character):
    """
    Uniform simple shear that resolves a Peach-Koehler glide force on the
    dislocation, in matscipy's (x=glide direction, y=glide-plane normal,
    z=line) convention -- see module docstring.

    character="edge"  -> Burgers vector along x -> conjugate shear is
                          u_x(y) = gamma*y   (resolves sigma_xy)
    character="screw" -> Burgers vector along z -> conjugate shear is
                          u_z(y) = gamma*y   (resolves sigma_yz)

    Displaces every atom, including the FixAtoms boundary shell -- the
    shell is then re-fixed at its new, sheared position by relaxing with
    `apply_constraint=False` already applied here (mirrors
    matscipy.dislocation.make_barrier_configurations, which uses the same
    `set_positions(..., apply_constraint=False)` pattern to move fixed
    atoms to a new reference position).
    """
    pos = atoms.get_positions().copy()
    y = pos[:, 1] - pos[:, 1].mean()
    if character == "edge":
        pos[:, 0] += gamma * y
    else:
        pos[:, 2] += gamma * y
    atoms.set_positions(pos, apply_constraint=False)
    return atoms


def resolved_shear_stress(gamma, C44_GPa):
    """Engineering shear strain -> resolved shear stress (GPa), via C44.
    See module docstring: a Voigt-style estimate, not an exact anisotropic
    resolution onto the {111} glide plane."""
    return C44_GPa * gamma


# ── Relaxation / NEB ──────────────────────────────────────────────────────

def relax(atoms, calc, fmax, steps=500):
    atoms.calc = calc
    FIRE(atoms, logfile=None).run(fmax=fmax, steps=steps)
    return atoms


def barrier_at_strain(disloc, calc, model_json, character, gamma,
                       cylinder_r, n_images, fmax, neb_steps):
    """Epsilon-NEB barrier (eV) for one applied resolved shear strain gamma."""
    _, glide_ini, glide_fin = disloc.build_glide_configurations(radius=cylinder_r)

    for atoms in (glide_ini, glide_fin):
        apply_resolved_shear(atoms, gamma, character)
        relax(atoms, calc, fmax=fmax)

    # One calculator instance per image (safe default for NEB -- avoids any
    # cross-image state issues in the underlying Julia calculator). Share a
    # single `calc` across images instead if this is too slow to reload.
    images = [glide_ini.copy() for _ in range(n_images - 1)] + [glide_fin.copy()]
    for image in images:
        image.calc = load_calculator(model_json)

    neb = NEB(images, climb=True)
    neb.interpolate(method="idpp", apply_constraint=True)
    FIRE(neb, logfile=None).run(fmax=fmax, steps=neb_steps)

    Ef, dE = NEBTools(images).get_barrier()
    return Ef, dE, images


# ── Peierls stress extraction ────────────────────────────────────────────

def peierls_power_law(tau, A, tau_P, alpha):
    """Zhu et al. (2008) barrier-vs-stress fit (Si et al. 2023, Eq. 2)."""
    return A * np.clip(1.0 - tau / tau_P, 0.0, None) ** alpha


def fit_peierls_stress(taus, barriers):
    taus = np.asarray(taus, dtype=float)
    barriers = np.asarray(barriers, dtype=float)
    p0 = [barriers[np.argmin(taus)], 1.5 * taus.max(), 1.6]
    (A, tau_P, alpha), _ = curve_fit(peierls_power_law, taus, barriers, p0=p0, maxfev=20000)
    return tau_P, A, alpha


# ── Driver ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-json", required=True,
                   help="ACEpotentials.jl model .json (nominal or exact_constrained_model.json)")
    p.add_argument("--element", default="Al")
    p.add_argument("--character", choices=["edge", "screw"], default="edge")
    p.add_argument("--cylinder-radius", type=float, default=40.0, help="Angstrom")
    p.add_argument("--n-images", type=int, default=7)
    p.add_argument("--gamma-max", type=float, default=0.02, help="max engineering shear strain")
    p.add_argument("--n-strains", type=int, default=5)
    p.add_argument("--fmax", type=float, default=1e-2, help="eV/Angstrom")
    p.add_argument("--neb-steps", type=int, default=1000)
    p.add_argument("--out-csv", default="peierls_stress_neb_results.csv")
    args = p.parse_args()

    calc = load_calculator(args.model_json)
    disloc, alat, C11, C12, C44 = build_dislocation(calc, args.element, args.character)
    print(f"a={alat:.4f} A  C11={C11:.1f}  C12={C12:.1f}  C44={C44:.1f} GPa  "
          f"character={args.character}")

    gammas = np.linspace(0.0, args.gamma_max, args.n_strains)
    taus, barriers = [], []
    for gamma in gammas:
        Ef, dE, _ = barrier_at_strain(disloc, calc, args.model_json, args.character, gamma,
                                       args.cylinder_radius, args.n_images, args.fmax,
                                       args.neb_steps)
        tau = resolved_shear_stress(gamma, C44)
        print(f"  gamma={gamma:.4f}  tau={tau:.4f} GPa  barrier={Ef:.5f} eV  dE={dE:.5f} eV")
        taus.append(tau)
        barriers.append(Ef)

    tau_P, A, alpha = fit_peierls_stress(taus, barriers)
    print(f"\nFitted Peierls stress: tau_P = {tau_P:.4f} GPa  "
          f"(A={A:.5f} eV, alpha={alpha:.3f})")

    np.savetxt(args.out_csv, np.column_stack([gammas, taus, barriers]),
               header="gamma,tau_GPa,barrier_eV", delimiter=",", comments="")
    print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    main()
