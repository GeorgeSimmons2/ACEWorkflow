# Scaling the constrained POPS pipeline — work report

**Period covered:** 29 July – 4 August 2026
**Model:** `Al_12_4_6A_2_` (91 params) unless stated; `Al_20_4_6A_4` (5476 params) for the capacity test.

---

## 1. Al_20 naive ensemble — the result that changes the paper's framing

Five **completely naive** POPS hypercube samples of `Al_20_4_6A_4`, NPT at 300/500/700/900 K,
4×4×4, judged by q̄₆ (`fcc_order_ensemble_Al_12_4_6A_2.py`, unmodified).
Jobs `6001832` (sampling, 25 min) and `6001833_[0-4]` (~10 h each).

| arm | a_eq spread | FCC | α definable | α (10⁻⁶/K) |
|---|---|---|---|---|
| Al_12 naive (91 par) | 6.41% | 6/20 | 1/5 | 9.3 single |
| Al_12 constrained + rejection | 0.000% | 12/20 | 3/5 | 4.8 ± 8.1 |
| **Al_20 naive (5476 par)** | **0.130%** | **20/20** | **5/5** | **26.1 ± 2.0** |
| experiment | | | | 23–26 |

Every member crystalline at every temperature; q̄₆ falls smoothly 0.552 → 0.48; max
D ≈ 5×10⁻¹² m²/s.

**The unconstrained large model beats the constrained small model on every axis.** So the
instability the constraint scheme fixes is substantially a **capacity artefact of the
91-parameter model**, not a generic property of naive POPS. This corroborates the γ_us result
already noted at `paper_main.tex:645`.

**Action required (not taken — needs George's decision):** the Results section currently implies
naive POPS is generically unphysical. The defensible claim is that it is unphysical *at low
capacity*, i.e. constraints buy robustness exactly when the model is under-parameterised —
which is when POPS error bars are widest and most needed. Whether constraints still add
anything at 5476 params is **untested**; the 4×4×4 undotted Hessian is ~26 GB there.

---

## 2. Paper and documentation

* **Ensemble Results section filled** (`docs/paper_main.tex`, commit `50f33d6`). All five `\NUM{}`
  placeholders resolved, three `\TODO`s closed. Headline framed as α **definability** (3/5 vs 1/5),
  not accuracy. FCC-retention margin reported as **not significant** — pairs cluster within five
  models, pooled Fisher p = 0.11 is anticonservative, model-level 2/5 vs 1/5 gives p = 1.00.
* **Melting excluded quantitatively:** max D anywhere in either Al_12 ensemble is
  1.3×10⁻¹⁰ m²/s = 2.5% of liquid Al. All failures are solid–solid.
* **Sign correction:** the transformed phase is 0.18 eV/atom **below** FCC, not above. The
  unconstrained model *prefers* that structure; FCC is metastable under those parameters.
* **`docs/phonon_folding_note.tex`** — 5-page standalone note separating the Bloch transform
  (real space → D(q); premise is translational symmetry) from zone folding (large BZ → small BZ);
  covers p2s/s2p indexing, the PBC image sum, and q-commensurability.
* **`docs/report_constrained_pops_scaleup.md`** — this file.

---

## 3. Figures

* **`fcc_compare_figure_Al_12_4_6A_2.jl`** (commit `06abef4`) — merges the constrained and naive
  FCC panels: both dispersions on one shared axis (left), both RDFs stacked (right), coordination
  dropped. Reuses `rdf_<T>K.csv` so histograms are byte-identical to the published panels; hits
  the cached Hessians at both lattice constants. Verified: min ω reproduces at +0.324 / −11.368 THz.
* **Figure text sizing** — Makie's `size` is in **points**. A figure built at 900 pt and placed at
  `width=\linewidth` (~510 pt) has its text scaled by 0.57, so 11 pt labels land at ~6 pt. All new
  figures are built at their final display width (540 pt) with 12–13 pt fonts. **Do not add a
  width scale factor in `\includegraphics`.**

---

## 4. Can the cutting-plane iteration be replaced by an SDP?

`D(q;θ) − ω²_cut I ⪰ 0` is a **linear matrix inequality** (D is affine in θ), and for FCC Al the
blocks are 3×3, so the whole thing is a small SDP. Prototyped with Clarabel in a scratch env
stacked on the project (`Project.toml` untouched).

**Verdict: correct, more accurate, but slower — not adopted.**

| | cutting plane (OSQP) | SDP (Clarabel) |
|---|---|---|
| 6-member benchmark | **2.5 s** | 19.0 s (0.13× speedup) |
| POPS interpolation error | 1e-5 … 3.6e-4 | **7e-13 … 8.5e-12** |
| min ω | agrees to ~1e-3 THz | agrees |

Five of six members needed **zero cuts**, so the cutting plane is usually one ~10 ms QP while
Clarabel with 138 PSD blocks costs ~0.9 s. But OSQP satisfies the POPS interpolation equality —
the property that *defines* a POPS member — only to ~1e-4.

Also established: the phonon LMI is **inactive at the mean-fit optimum** (Born + `b′·θ=0` alone
give min ω = 0.1715 THz > 0.15), so the cutting plane never fired for the mean. `‖Δθ‖ = 4.14`
against `‖θ‖ = 2.18` between OSQP and Clarabel on the *same* QP, for a 3.7e-5 relative objective
difference — a very flat direction, and OSQP stops early along it.

Scripts: `sdp_meanfit_prototype_`, `sdp_meanfit_diagnose_`, `sdp_vs_cuts_benchmark_`.

---

## 5. Cutting-plane constraint of the FULL top-50%-leverage cloud

`cutting_plane_full_cloud_Al_12_4_6A_2.jl` + `run_cutting_plane_full_cloud.slurm`.
**73,479 of 146,958 observations**, one constrained POPS delta each.

```
OSQP :Solved            73479 / 73479 (100%)
dynamically stable      73479 / 73479 (100.000%)   min ω > 0
met margin (≥0.15 THz)  73411 / 73479 ( 99.907%)
needed ≥1 cut           17471 ( 23.8% )   mean 2.02, max 367
hit max_cuts (40)         132
min ω range             [0.1495, 0.4088] THz
CPU total               3.96 CPU-hours   (mean 194 ms/member)
```

**Nothing is dynamically unstable.** The 68 that miss the margin do so by parts per million after
hitting `max_cuts` — a tolerance miss, not an instability. Both criteria are reported separately
and `committee_stable.jls` uses the stricter one.

**Cost in context:** the training set is 7,893 configs / 46,355 atoms (mean 5.9 atoms), confirming
the design matrix as 7,893 energy + 3×46,355 force rows = 146,958. At a plausible ~5 CPU-min per
small-cell DFT, 3.96 CPU-hours ≈ **50 DFT calculations, ~0.6% of the DFT that trained the model**.
That is the number worth quoting: constraints on 73k ensemble members for well under 1% of the
training DFT.

Two engineering notes: `born_rows/P` and `b_prime'/P` were being recomputed inside every
`constrain_member` call (hoisted out); and the legacy module-scope `OSQP.Model()` is mutated by
`setup!` so is **not thread-safe** — one model per thread now.

---

## 6. Hypercube, rejection sampling, coverage

`hypercube_full_cloud_bands_Al_12_4_6A_2.jl`. Both arms are
**constrained deltas → hypercube → rejection sample (30 drawn)**; they differ only in which set of
deltas the hypercube was fitted to.

| hypercube built from | directions | width mean/max | acceptance | E cov | F cov |
|---|---|---|---|---|---|
| 73,404 deltas | **63** | 0.55 / 16.5 | 10.49% | **90.1%** | **97.5%** |
| 30 forest deltas | 30 | 0.40 / 7.8 | 28.57% | 52.4% | 71.7% |

RMSE identical in both (0.9551 eV, 0.4058 eV/Å) — the point model `θ_mean` is unchanged, only the
error bars move.

**The forest ensemble is badly under-dispersed** (52% energy coverage against a nominal ~95%).
30 points span at most 30 directions in 91-dimensional parameter space, so that hypercube is
rank-deficient and *cannot* generate spread in the other 61.

**But the taller bands are an artefact, not extra physics:**

| | ω_max (THz) | median |
|---|---|---|
| actual full-cloud members (n=400) | 8.91 – 14.96 | 12.13 |
| hypercube draws from that cloud | 9.17 – 24.18 | 16.79 |
| actual forest members (n=30) | 9.23 – 14.99 | 12.07 |

The real deltas are **equally dispersed in both clouds**. The 1.38× inflation comes from
`hypercube()` running at `percentile_clipping = 0` (bounds are extrema, which grow with sample
size) and from a box in 63 dimensions being almost entirely corner. So the coverage gain is part
legitimate (rank) and part over-widening (box looseness).

`forest ∩ full cloud`: 19 of 30 forest members are *literally* full-cloud members (distance
1e-9…1e-6); the rest come from the bottom half of the leverage distribution, which the full cloud
excludes. Neither set contains the other.

---

## 7. Structural results (no computation needed)

1. **Equalities are free.** Every constrained delta satisfies `b′·θ = 0`, so `b′·δᵢ = 0`; the
   hypercube's eigenvectors span the row space of δ ⊂ `null(b′)`; therefore **every** box point
   satisfies the equality exactly. Confirmed empirically — the a_eq band rejected *nothing*
   (`passed Born` = `passed a_eq` in both funnels). Any future linear equality (pin C₁₁, γ_us,
   a_eq at several volumes) is likewise free at sampling time.

2. **Inequalities and the PSD constraint are preserved by the convex hull, not the box.**
   `D(q;θ)` is affine in θ and the PSD cone is convex, so any convex combination of stable members
   is stable at every q, exactly. The hypercube strictly contains the hull, which is precisely why
   Born rejects 33% and the phonon check another 84%. **Acceptance rate measures how much of the
   box lies outside the hull — nothing more physical.**

3. **Sampling the hull cheaply** — uniform-on-simplex fails: `Cov ≈ S/(N+1)`, so Dirichlet(1) over
   73k members is 271× too narrow. Viable: random k-subset simplices (k tunes dispersion, `S/(k+1)`);
   **subsampling the 73,404 certified members directly** (zero cost, 100% acceptance, true cloud
   dispersion); or boundary sampling (random direction, average top-m by projection). Hit-and-run
   is not viable — an LP per step at 73k variables, O*(d³) mixing at d = 63.

4. **The cloud's numerical rank is 63, not 90.** One missing direction is `b′`; the other 27 are
   directions POPS corrections never explore even with 73,479 members. A hard ceiling on what any
   hypercube from this cloud can represent.

---

## 8. Open items

| item | status |
|---|---|
| **Band path is 141 q-points, not 145** | Correction **not yet applied**. `N_per_seg = [20,20,20,20,60]` sums to 140 + closing point = 141 (script prints `141 q-points total, 138 kept`). The **16/145 = 11%** commensurability figure has the wrong denominator *and* a numerator counted against the wrong path. Appears in `docs/phonon_folding_note.tex`, `paper_main.tex`, the figure caption, and memory `bandpath-md-commensurability`. Needs recomputing. |
| Paper reframing after the Al_20 result | Needs George's decision (§1) |
| Panel relabel in `bands_hypercube_full_cloud` | Fix is in the script; not yet rerun |
| `verify_claims_Al_12_4_6A_2.jl` | Still broken, untracked |
| Do constraints help at 5476 params? | Untested; needs ~26 GB Hessian |
| `max_cuts` 40 → ~100 | Would convert the 132 tolerance misses into clean passes |

---

## Artefacts

**Scripts** (`scripts/uq/`): `fcc_compare_figure_`, `sdp_meanfit_prototype_`,
`sdp_meanfit_diagnose_`, `sdp_vs_cuts_benchmark_`, `cutting_plane_full_cloud_`,
`run_cutting_plane_full_cloud.slurm`, `hypercube_full_cloud_bands_` (all `…_Al_12_4_6A_2`).

**Data** (`models/Al_12_4_6A_2_/results/cutting_plane_full_cloud/`): `committee_full_cloud.jls`
(57 MB), `committee_stable.jls` (54.6 MB), `member_diagnostics.csv`,
`committee_rejection_full_cloud.csv`, `hypercube_summary.csv`,
`bands_hypercube_full_cloud.*`, `parity_full_cloud.*`, `calibration_full_cloud.*`.

**Al_20** (`models/Al_20_4_6A_4/results/`): `pops_naive_committee/`, `npt_naive/` incl.
`fcc_order_verdict.csv`.
