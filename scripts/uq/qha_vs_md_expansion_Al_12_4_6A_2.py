#!/usr/bin/env python3
"""
qha_vs_md_expansion_Al_12_4_6A_2.py

Reconciles two thermal-expansion numbers for the SAME constrained models, which
appear to contradict each other, and shows the contradiction is the result.

THE APPARENT CONTRADICTION
--------------------------
Constraining makes the models stiff:

    Gruneisen  gamma  1.02  (committee mean; experiment 2.1)   -> LOW
    bulk mod   B      92.6 GPa (committee mean; experiment 76) -> HIGH
    gamma_us          360 mJ/m^2 (experiment 160-180)          -> HIGH

Quasiharmonically, alpha = gamma*Cv/(3*B*V): a low gamma and a high B both push
alpha DOWN. The models should under-expand. Yet the NPT MD measures alpha = 50.2
+- 6.1 x10^-6/K for the constrained mean over 300-700 K, roughly twice Al's 23.

Both cannot be quasiharmonic behaviour.

THE RESOLUTION
--------------
The quasiharmonic prediction from the models' OWN gamma and B is 9.2, while MD
gives 50.2 -- a factor of 5.5. The MD expansion is therefore not phonon
anharmonicity at all. It is the approach to the competing low-coordination phase
that sits ~0.18 eV/atom below FCC (see the multivolume-phonon-constraints note):
the lattice inflates as a precursor to a transformation, not as a thermal
population of anharmonic phonons.

Supporting evidence from the same ensemble, all consistent with that reading:

  * member_3   falls into the competing phase outright, disordered by 300 K
                (q6bar 0.248), despite a healthy +0.170 THz harmonic margin
  * constrained mean  stays crystalline but its q6bar decays monotonically
                0.549 -> 0.511 -> 0.478 over 300-700 K: progressive distortion
                toward the transformation while still crystalline
  * member_13  the stiffest member, far from the instability, has alpha
                -0.9 +- 6.8, i.e. consistent with zero -- the quasiharmonic
                answer for an over-stiff model, as expected when nothing
                anharmonic intervenes

So the ensemble spans both regimes: away from the instability the models behave
quasiharmonically and under-expand (member_13); near it they over-expand
(constrained mean); past it they transform (member_3). The over-expansion and the
over-stiffening are not in conflict -- they are different distances from the same
competing minimum.

WHY IT MATTERS FOR THE METHOD
-----------------------------
Harmonic positivity constraints, at any number of volumes, cannot address this by
construction: member_3 satisfies phonon positivity at all six constrained volumes
and still transforms at 300 K. The competing-phase row (B_transformed - B_FCC).c > 0
is the constraint that targets it, and is one linear QP row because ACE energy is
linear in c.

Inputs (already on disk, nothing recomputed):
  results/gruneisen_constrained_vs_naive/gruneisen_summary.csv   gamma
  results/gsf_peierls_committee/gsf_peierls_summary.csv          C11, C12 -> B
  results/npt_ensemble/fcc_order_verdict.csv                     MD alpha

Usage:  python3 scripts/uq/qha_vs_md_expansion_Al_12_4_6A_2.py [results_dir]
Output: <results_dir>/qha_vs_md_expansion.csv
"""
import sys, os, csv, math
import numpy as np

KB   = 8.617333e-5      # eV/K
GPA  = 0.00624151       # eV/A^3 per GPa
A_EQ = 4.044935892057517


def alpha_qha(gamma, B_GPa, a=A_EQ):
    """Linear thermal expansion coefficient, 1e-6/K, Dulong-Petit Cv per atom."""
    V = a ** 3 / 4                      # FCC conventional cell holds 4 atoms
    return gamma * (3 * KB) / (B_GPa * GPA * V) / 3 * 1e6


def load(path, group, *cols):
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(l for l in fh if not l.startswith('#')):
            out.setdefault(r[group], []).append([float(r[c]) for c in cols])
    return {k: np.array(v) for k, v in out.items()}


def main(root):
    gam = load(os.path.join(root, "gruneisen_constrained_vs_naive",
                            "gruneisen_summary.csv"), "committee", "gamma_300K")
    ela = load(os.path.join(root, "gsf_peierls_committee",
                            "gsf_peierls_summary.csv"), "committee", "C11_GPa", "C12_GPa")

    # sanity check: the formula must reproduce Al from independent literature inputs
    ref = alpha_qha(2.10, 76.0, 4.050)
    assert abs(ref - 23.0) < 0.5, f"constants wrong: Al check gave {ref:.1f}, expected 23.0"
    print(f"  sanity check: gamma=2.1, B=76 GPa -> alpha = {ref:.1f} (Al expt 23.0)  OK\n")

    print(f"  {'committee':14s} {'gamma':>16s} {'B (GPa)':>16s} {'alpha_QHA':>10s}")
    print("  " + "-" * 60)
    rows = []
    for comm in sorted(gam):
        g = gam[comm][:, 0]
        C11, C12 = ela[comm][:, 0], ela[comm][:, 1]
        B = (C11 + 2 * C12) / 3
        a_mean = alpha_qha(g.mean(), B.mean())
        a_med = alpha_qha(np.median(g), np.median(B))
        print(f"  {comm:14s} {g.mean():7.2f} +-{g.std():5.2f} "
              f"{B.mean():8.1f} +-{B.std():5.1f} {a_mean:10.1f}")
        rows.append((comm, g.mean(), g.std(), B.mean(), B.std(), a_mean, a_med))

    md = os.path.join(root, "npt_ensemble", "fcc_order_verdict.csv")
    md_alpha = {}
    if os.path.exists(md):
        with open(md) as fh:
            for r in csv.DictReader(l for l in fh if not l.startswith('#')):
                if r.get("alpha_1e6_per_K"):
                    md_alpha[r["run"]] = (float(r["alpha_1e6_per_K"]),
                                          float(r["alpha_se_1e6_per_K"] or "nan"))

    q = alpha_qha(gam["constrained"][:, 0].mean(),
                  ((ela["constrained"][:, 0] + 2 * ela["constrained"][:, 1]) / 3).mean())
    print(f"\n  quasiharmonic prediction, constrained models : {q:.1f}")
    for run, (a, se) in sorted(md_alpha.items()):
        tag = f"{a:.1f} +- {se:.1f}"
        note = f"  -> MD/QHA = {a/q:.1f}x" if a > 2 * se else "  -> consistent with zero (quasiharmonic regime)"
        print(f"  MD, {run:22s}: {tag:>16s}{note}")

    dest = os.path.join(root, "qha_vs_md_expansion.csv")
    with open(dest, "w", newline="") as fh:
        fh.write("# alpha = gamma*Cv/(3*B*V), Dulong-Petit Cv, a_eq = %.6f\n" % A_EQ)
        fh.write("# MD alpha from npt_ensemble/fcc_order_verdict.csv (crystalline temperatures only)\n")
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["committee", "gamma_mean", "gamma_sd", "B_mean_GPa", "B_sd_GPa",
                    "alpha_qha_mean", "alpha_qha_median"])
        for r in rows:
            w.writerow([r[0]] + [f"{x:.4f}" for x in r[1:]])
        w.writerow([])
        w.writerow(["run", "alpha_md", "alpha_md_se", "alpha_qha_constrained", "ratio"])
        for run, (a, se) in sorted(md_alpha.items()):
            w.writerow([run, f"{a:.2f}", f"{se:.2f}", f"{q:.2f}", f"{a/q:.2f}"])
    print(f"\n  wrote {dest}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "models/Al_12_4_6A_2_/results")
