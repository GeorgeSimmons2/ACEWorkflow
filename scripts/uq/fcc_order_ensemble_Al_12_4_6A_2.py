#!/usr/bin/env python3
"""
fcc_order_ensemble_Al_12_4_6A_2.py

Post-hoc FCC-retention verdict for the NPT ensembles, replacing the `still_fcc`
column written by scripts/uq/npt_ensemble_member_Al_12_4_6A_2.jl.

WHY THIS EXISTS
---------------
The MD script decides FCC survival from a mean coordination number counted
inside a HARD-CODED nn_cutoff = 3.3 A (line 73), with the criterion <Z> >= 11.
That cutoff does not scale with the lattice constant, and these trajectories
expand by up to 6%:

    a = 4.0449 (0 K)     first shell a/sqrt2 = 2.860   cutoff 3.3 sits 15% above
    a = 4.2887 (900 K)   first shell         = 3.033   cutoff 3.3 sits  8.8% above

At the expanded lattice constants ordinary thermal spread pushes genuine
first-shell neighbours outside 3.3 A, so they go uncounted and <Z> falls without
any structural change. The criterion therefore partly measures the lattice
constant rather than the structure, and it penalises exactly those models that
expand most. Using it, the multi-volume constrained mean was scored as having
left FCC at 700 K; it has not.

The radial alternative r1/a (peak of g(r) over the lattice constant, ideal
0.70711) is no better: the argmax of g(r) shifts BELOW the mean neighbour
distance at high T through ordinary anharmonicity, and shifts further the more
the model expands -- the same bias in a different coordinate.

WHAT IS USED INSTEAD
--------------------
The Lechner-Dellago averaged Steinhardt bond-orientational order parameter q6bar,
computed with a cutoff that scales as 0.8536*a (midway between the FCC first
shell a/sqrt2 and second shell a, so it tracks the cell).

q6bar is angular, so it is insensitive both to the cutoff placement and to
radial anharmonicity. Reference values: ideal FCC 0.575, thermally excited FCC
0.45-0.57, disordered/liquid < 0.20.

Verdict: crystalline if q6bar > 0.35 for >= 80% of atoms.

The ensemble supplies its own controls at matched temperature -- member_13 is
perfect FCC (q6bar 0.547 at 700 K) and member_3 is disordered (0.273 at 700 K) --
so the threshold is calibrated in-sample rather than taken from literature.

Thermal expansion alpha is refitted over the crystalline temperatures only.

The lattice constant used for that fit is taken from thermal_expansion_summary.csv
(the full production-window average), NOT from the handful of frames used for
q6bar. For the stiffest members the entire thermal drift (~0.002 A over 400 K) is
smaller than the instantaneous box fluctuation (sigma ~ 0.005-0.009 A), so a
few-frame average does not even recover the sign of alpha. alpha is reported with
a standard error propagated from the per-temperature box fluctuation, which for
the over-rigid members correctly comes out consistent with zero.

Usage
-----
  python3 scripts/uq/fcc_order_ensemble_Al_12_4_6A_2.py \
      models/Al_12_4_6A_2_/results/npt_ensemble

Outputs -> <ensemble_dir>/fcc_order_verdict.csv
"""
import sys, os, re, csv, math
import numpy as np
from scipy.special import sph_harm

NREP      = 4        # supercell replication used by the NPT driver
Q6_SOLID  = 0.35     # per-atom crystalline threshold
FRAC_MIN  = 0.80     # fraction of atoms that must exceed it
N_FRAMES  = 4        # equilibrated frames averaged over (end of trajectory)
TEMPS     = (300, 500, 700, 900)


def read_extxyz_tail(path, n_frames):
    """Return the last n_frames as (cell 3x3, positions Nx3)."""
    frames = []
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            n = int(line.strip())
            comment = f.readline()
            m = re.search(r'Lattice="([^"]+)"', comment)
            cell = np.array([float(x) for x in m.group(1).split()]).reshape(3, 3)
            pos = np.empty((n, 3))
            for i in range(n):
                p = f.readline().split()
                pos[i] = [float(p[1]), float(p[2]), float(p[3])]
            frames.append((cell, pos))
            if len(frames) > n_frames:
                frames.pop(0)
    return frames


def q6bar(cell, pos, nrep=NREP):
    """Lechner-Dellago averaged q6, one value per atom."""
    L = np.diag(cell).copy()
    n = len(pos)
    a = (L / nrep).mean()
    rc = 0.8536 * a                       # midway between FCC shells 1 and 2

    d = pos[:, None, :] - pos[None, :, :]
    d -= L * np.round(d / L)              # minimum image, orthorhombic cell
    r = np.sqrt((d ** 2).sum(-1))
    np.fill_diagonal(r, 1e9)
    nb = r < rc

    q = np.zeros((n, 13), dtype=complex)
    for i in range(n):
        j = np.where(nb[i])[0]
        if len(j) == 0:
            continue
        v, rr = d[i, j], r[i, j]
        theta = np.arccos(np.clip(v[:, 2] / rr, -1, 1))
        phi = np.arctan2(v[:, 1], v[:, 0])
        for k, m in enumerate(range(-6, 7)):
            q[i, k] = sph_harm(m, 6, phi, theta).mean()

    qb = np.zeros_like(q)
    for i in range(n):
        j = np.append(np.where(nb[i])[0], i)   # atom i together with its shell
        qb[i] = q[j].mean(0)
    return np.sqrt(4 * np.pi / 13 * (np.abs(qb) ** 2).sum(1)), a


def read_summary(path):
    """{T_K: (a, a_std)} from the MD driver's production-window averages."""
    d = {}
    if not os.path.exists(path):
        return d
    with open(path) as fh:
        for row in csv.DictReader(l for l in fh if not l.startswith('#')):
            d[int(float(row['T_K']))] = (float(row['a_Ang']), float(row['a_std_Ang']))
    return d


def fit_alpha(T, A, S, a0):
    """Weighted linear fit a(T); returns alpha and its standard error, both 1e-6/K."""
    w = 1.0 / np.maximum(S, 1e-12) ** 2
    Sw, ST, SA = w.sum(), (w * T).sum(), (w * A).sum()
    STT, STA = (w * T * T).sum(), (w * T * A).sum()
    den = Sw * STT - ST * ST
    slope = (Sw * STA - ST * SA) / den
    slope_se = math.sqrt(Sw / den)
    return slope / a0 * 1e6, slope_se / a0 * 1e6


def main(root):
    runs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    out = []
    print(f"{'run':17s} {'T':>5s} {'a':>8s} {'q6bar':>15s} {'ordered':>8s}  verdict")
    print("-" * 74)
    for run in runs:
        summ = read_summary(os.path.join(root, run, "thermal_expansion_summary.csv"))
        rows = []
        for T in TEMPS:
            traj = os.path.join(root, run, f"T{T}K", "md_trajectory.extxyz")
            if not os.path.exists(traj):
                continue
            frames = read_extxyz_tail(traj, N_FRAMES)
            vals = np.concatenate([q6bar(c, x)[0] for c, x in frames])
            # production-window average, not the few frames used for q6bar
            a, a_std = summ.get(T, (float('nan'), float('nan')))
            frac = float(np.mean(vals > Q6_SOLID))
            cry = frac >= FRAC_MIN
            rows.append((T, a, a_std, vals.mean(), vals.std(), frac, cry))
            print(f"{run:17s} {T:5d} {a:8.4f} {vals.mean():7.3f} +- {vals.std():5.3f} "
                  f"{100*frac:7.1f}%  {'crystalline' if cry else 'DISORDERED'}")
        if not rows:
            continue
        # the crystalline count is a property of the trajectories alone and must not
        # depend on the summary csv, which the MD driver writes only when a run ends
        cr = [r for r in rows if r[6]]
        fit = [r for r in cr if np.isfinite(r[1])]
        a0 = summ.get(0, (float('nan'), 0.0))[0]
        if len(fit) >= 2 and np.isfinite(a0):
            alpha, se = fit_alpha(np.array([r[0] for r in fit], float),
                                  np.array([r[1] for r in fit], float),
                                  np.array([r[2] for r in fit], float), a0)
            sig = "consistent with zero" if abs(alpha) < 2 * se else f"{abs(alpha)/se:.1f} sigma"
            print(f"{'':17s} -> crystalline {len(cr)}/{len(rows)}   "
                  f"alpha = {alpha:.1f} +- {se:.1f} x10^-6 /K  (n={len(fit)}, {sig})")
        else:
            alpha = se = float('nan')
            why = "run still in progress" if not summ else "fewer than 2 crystalline points"
            print(f"{'':17s} -> crystalline {len(cr)}/{len(rows)}   "
                  f"alpha not fittable ({why})")
        for (T, a, a_std, m, s, f, cry) in rows:
            out.append((run, T, a, m, s, f, cry, alpha, se))
        print()

    dest = os.path.join(root, "fcc_order_verdict.csv")
    with open(dest, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["# supersedes still_fcc in thermal_expansion_summary.csv, which used a "
                    "fixed 3.3 A neighbour cutoff that does not scale with the lattice constant"])
        w.writerow([f"# q6bar cutoff 0.8536*a, solid if q6bar>{Q6_SOLID} for >={100*FRAC_MIN:.0f}% of atoms, "
                    f"averaged over last {N_FRAMES} frames"])
        w.writerow(["# a_Ang is the production-window average from "
                    "thermal_expansion_summary.csv; alpha is a weighted fit over "
                    "crystalline temperatures only"])
        w.writerow(["run", "T_K", "a_Ang", "q6bar_mean", "q6bar_std",
                    "frac_ordered", "crystalline", "alpha_1e6_per_K", "alpha_se_1e6_per_K"])
        for r in out:
            w.writerow([r[0], r[1], f"{r[2]:.6f}", f"{r[3]:.4f}", f"{r[4]:.4f}",
                        f"{r[5]:.4f}", str(r[6]).lower(),
                        "" if math.isnan(r[7]) else f"{r[7]:.2f}",
                        "" if math.isnan(r[8]) else f"{r[8]:.2f}"])
    print(f"wrote {dest}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "models/Al_12_4_6A_2_/results/npt_ensemble")
