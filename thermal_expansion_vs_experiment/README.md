# `thermal_expansion_vs_experiment/` — a(T) for both NPT members, against Wilson 1941

One set of axes replacing two separate figures, with an experimental reference:

| colour    | series                                                                    |
| --------- | ------------------------------------------------------------------------- |
| **red**   | unconstrained POPS, worst member — `results/npt_thermal_expansion_naive_worst_member/` |
| **blue**  | constrained POPS, softest member (multi-volume) — `results/npt_multivolume_softest/`   |
| **black** | experiment — A. J. C. Wilson, *Proc. Phys. Soc.* **53** (1941) 235         |

```bash
julia --project thermal_expansion_vs_experiment/plot_thermal_expansion_vs_experiment.jl
```

Output: `thermal_expansion_vs_experiment/thermal_expansion_aT_vs_experiment.{pdf,png}`.
No molecular dynamics is repeated — both model series are read from the
`thermal_expansion_summary.csv` files the NPT runs already wrote, so the points are the
published numbers.

To regenerate those trajectories from scratch, see **`../npt_trajectories/`**, which
holds pinned copies of the two NPT drivers. Pinned because the working copies in
`scripts/uq/` have since been repurposed and no longer reproduce these runs.

## The structural companion figure

`fcc_compare_constrained_vs_naive.jl` (copied from `scripts/uq/fcc_compare_figure_Al_12_4_6A_2.jl`)
reads the **same two NPT runs** and answers what a(T) alone cannot: was the lattice still
FCC when a(T) was measured? Phonon bands of both members at their own a(T) on the left,
their RDFs stacked on the right.

```bash
julia --project -t 8 thermal_expansion_vs_experiment/fcc_compare_constrained_vs_naive.jl [T]
```

At 300 K: constrained min ω **+0.324 THz** at a = 4.07786 Å, unconstrained **−11.368 THz**
at a = 4.16664 Å. It honours the same `DIR_UNCON` / `DIR_CON` variables as the a(T)
plotter, so one pair of settings drives both figures, and it takes θ for the constrained
arm from that run's own `theta_used.csv` (byte-identical to the committee's copy, max
|Δ| = 0.0, but with no committee dependency).

**It writes to the published stem by default**, i.e. a plain rerun overwrites
`models/Al_12_4_6A_2_/results/fcc_compare_constrained_vs_naive_300K.{pdf,png}` in place.
That is deliberate — it keeps existing `\includegraphics` resolving — but set `OUT` if
you want the published file untouched.

## The experimental data needs a unit conversion, and it matters

`wilson_1941_aluminium.csv` is Wilson's table from p. 240 of the paper (ten points,
0–650 °C), not the rounded four-figure table in the abstract. Every row was checked
against his own least-squares parabola and difference column; all ten agree to the last
published digit.

**The values are in kX (Siegbahn x-ray) units, although the paper writes "A."**
throughout — the rename to kX came in 1947, precisely because of this confusion. Two
independent confirmations:

- Wilson quotes agreement with Jette & Foote (1935) 4.04133 and Ievinš & Straumanis
  (1936) 4.04143, both classic kX-scale values for Al;
- his 25 °C spacing of 4.04134 is 0.207 % below the modern accepted 4.0495 Å, and
  0.207 % *is* the kX → Å conversion.

Plotted raw, the experimental curve would sit ~0.008 Å low and the models would look
systematically over-expanded by an amount that is pure unit error. The script converts
with `KX_TO_ANG` (default 1.00202) and prints the check:

```
Wilson's 25 °C spacing 4.04134 kX → 4.04950 Å   (accepted 4.0495 Å, Δ = +0.00000 Å)
```

The factor is not perfectly settled — values from 1.00202 to ~1.00208 are in use — but
the spread is 2×10⁻⁴ Å, an order of magnitude below the smallest feature here and
comparable to Wilson's own quoted error.

## α depends on the temperature window, so all three are printed

Aluminium's expansion coefficient is **not** constant: Wilson measures it rising from
22 to 37 ×10⁻⁶ K⁻¹ over his range. A single linear α is therefore a range average, and
two αs are comparable only over the same window. Every run prints three:

| window    | range                         | unconstrained | constrained | experiment |
| --------- | ----------------------------- | ------------- | ----------- | ---------- |
| `native`  | each series' own convention   | 5.6           | 2.7         | 2.9        |
| `common`  | 0–700 K (**default**)         | 6.1           | 2.7         | 2.7        |
| `overlap` | 273–700 K                     | 3.7           | 2.6         | 2.7        |

(×10⁻⁵ K⁻¹.) `common` is the default: it uses every physically valid model point, so
each fit line spans nearly all of its own data instead of a short central segment. The
caveat, stated rather than hidden, is that the models' T = 0 point is a *static* lattice
constant with no thermal motion and the experiment has no counterpart to it, so the low
end is not strictly like-for-like — `overlap` is the window that is. `native` reproduces
what the two figures this replaces quote, so those numbers stay checkable.

**The choice does not change the conclusion.** The constrained member lands on the
measurement (2.6–2.7 against 2.7) and the unconstrained one does not. In fact no
straight line describes the unconstrained series at all: its a(T) is not even monotonic
(4.167 Å at 300 K, 4.139 Å at 500 K), which the scatter of red points around the red
line shows directly.

## Points plotted but not fitted

The constrained run's 900 K point is plotted and never fitted, under every window:
`still_fcc=false` and mean coordination 9.12 against 12 for FCC, so the structure has
transformed and fitting it would turn α into a number about a solid–solid transition.
It is the blue point sitting well above the blue line. Each fit line is drawn only
across its own fitted range, so which points it covers is visible without annotation.

## Environment variables

| var          | default                                   | effect                                     |
| ------------ | ----------------------------------------- | ------------------------------------------ |
| `FIGW`       | `380`                                     | width in **points** = the width the figure is displayed at |
| `ASPECT`     | `0.82`                                    | height / width                             |
| `WIN`        | `common`                                  | `common` \| `overlap` \| `native`          |
| `FIT_MIN`    | `0`                                       | low edge of the `common` window            |
| `FIT_MAX`    | `700`                                     | high edge of the `common` window           |
| `KX_TO_ANG`  | `1.00202`                                 | kX → Å conversion for the experimental data |
| `RESDIR`     | `models/Al_12_4_6A_2_/results`            | where the NPT summaries live               |
| `WILSON`     | `wilson_1941_aluminium.csv`               | the transcribed table                      |
| `OUT`        | `thermal_expansion_vs_experiment/thermal_expansion_aT_vs_experiment` | output stem |

`FIGW` is in points and must equal the width the figure is **displayed** at in the
paper. Building at 540 and letting LaTeX shrink it is what made the text tiny before;
use `\includegraphics[]{}` at natural size, with no `width=` factor.
