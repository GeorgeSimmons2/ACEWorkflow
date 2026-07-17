# phonon_bands_C44_gamma_gradient.jl
#
# Compares the ACE phonon dispersion near Γ against continuum elasticity theory.
#
# Along Γ→X (the cubic [100] direction), the doubly-degenerate transverse-
# acoustic (TA) branch is linear near Γ with slope equal to the shear sound
# velocity:
#     v_T = sqrt(C44 / ρ)      (Christoffel equation, cubic crystal, [100])
#
# C44 here is the *macroscopic* elastic constant (from the strain Hessian at
# zero q), while the TA branch slope is measured from the *dynamical matrix*
# at small but finite q — two independent calculations that should agree if
# a potential is internally consistent.
#
# This script draws that continuum-elasticity line (a straight line through
# the origin, gradient = v_T) for both:
#   - the unconstrained (nominal least-squares) model, and
#   - the exact_constrained_model (C11, C12, C44 pinned to literature values
#     via exact_elastic_constraints.jl),
# overlaid on each model's actual TA phonon branch near Γ. The constrained
# model's TA branch should collapse onto the target-C44 gradient line; the
# unconstrained model's need not.
#
# Usage:
#   julia --project scripts/phonons/phonon_bands_C44_gamma_gradient.jl

using ACEWorkflow, ACEpotentials
using AtomsBuilder, AtomsBase, Unitful, StaticArrays, LinearAlgebra
using CairoMakie, Printf

const _eV_J   = 1.602176634e-19     # J eV⁻¹
const _Å_m    = 1.0e-10             # m Å⁻¹
const _amu_kg = 1.66053906660e-27   # kg amu⁻¹

const C44_TARGET_GPa = 30.9   # literature target used by exact_elastic_constraints.jl

# ─────────────────────────────────────────────────────────────────────────────
"""
    ta_gradient_THz_per_Ainv(C44_GPa, a_eq)

Continuum-elasticity slope dω/dq (THz per Å⁻¹) of the transverse-acoustic
phonon branch along Γ→X ([100]), from the shear sound velocity
v_T = sqrt(C44 / ρ) at lattice constant `a_eq`.
"""
function ta_gradient_THz_per_Ainv(C44_GPa, a_eq)
    sys0 = ACEWorkflow.Elasticity.reference_system(:Al; a=a_eq)   # 1-atom primitive cell
    L0   = SMatrix{3,3,Float64}(
               ustrip.(ACEWorkflow.Elasticity.lattice_matrix(sys0.cell.cell_vectors)))
    V0   = abs(det(L0))                 # Å³ per atom
    m    = ustrip(sys0[1].mass)         # amu

    ρ = (m * _amu_kg) / (V0 * _Å_m^3)   # kg/m³
    v = sqrt(C44_GPa * 1e9 / ρ)         # m/s

    # ω[rad/s] = v · q[m⁻¹] = v · q[Å⁻¹] · 1e10  →  f[THz] = ω / (2π·1e12)
    return v * 1e-2 / (2π)              # THz / Å⁻¹
end

"""
    ta_branch(x_vals, freqs, x_ticks)

Restrict to the Γ→X segment (x_vals ≤ x_ticks[2]) and, at each q-point, take
the mean of the two lowest (near-degenerate) frequencies as the TA branch.
"""
function ta_branch(x_vals, freqs, x_ticks)
    iX = findlast(x -> x <= x_ticks[2] + 1e-9, x_vals)
    xs = x_vals[1:iX]
    ta = [sum(sort(freqs[:, iq])[1:2]) / 2 for iq in 1:iX]
    return xs, ta
end

# ─────────────────────────────────────────────────────────────────────────────
"""
    bands_and_C44(model, label; N=5)

Relax the lattice constant, compute the phonon bands (N×N×N supercell for the
force constants) and the macroscopic C44 (strain-Hessian basis) for `model`.
"""
function bands_and_C44(model, label; N=5)
    println("\n--- $label ---")
    a_eq = ACEWorkflow.relax_lattice_constant(model, :Al)
    @printf("  a_eq = %.6f Å\n", a_eq)

    sys_prim  = bulk(:Al; a=a_eq * u"Å")
    sys_super = bulk(:Al; a=a_eq * u"Å", cubic=true) * (N, N, N)
    x_vals, freqs, x_ticks, labels = compute_phonon_bands(
        sys_prim, sys_super, model; N_per_seg=30, n_modes=nothing)

    C, _, _ = strain_hessian_GPa(model, :Al; a=a_eq)
    C44 = C[4, 4]
    @printf("  C44 = %.3f GPa\n", C44)

    return (; a_eq, x_vals, freqs, x_ticks, labels, C44)
end

# ─────────────────────────────────────────────────────────────────────────────
#  Script
# ─────────────────────────────────────────────────────────────────────────────

println("Loading unconstrained (mean) model ...")
result = load_model(:Al, 14, 4, 6, 2; dataset_name="subset_50_percent")
model_uncon = result.model

println("Loading exact_constrained_model ...")
model_con, _ = ACEpotentials.load_model("$(result.dir)/exact_constrained_model.json")

uncon = bands_and_C44(model_uncon, "unconstrained (mean) model")
con   = bands_and_C44(model_con,   "exact_constrained model")

xs_u, ta_u = ta_branch(uncon.x_vals, uncon.freqs, uncon.x_ticks)
xs_c, ta_c = ta_branch(con.x_vals,   con.freqs,   con.x_ticks)

slope_uncon  = ta_gradient_THz_per_Ainv(uncon.C44,     uncon.a_eq)
slope_con    = ta_gradient_THz_per_Ainv(con.C44,       con.a_eq)
slope_target = ta_gradient_THz_per_Ainv(C44_TARGET_GPa, con.a_eq)

println()
println(repeat('─', 66))
@printf("  %-22s  %10s  %16s\n", "Model", "C44 (GPa)", "TA slope (THz/Å⁻¹)")
println("  ", repeat('-', 62))
@printf("  %-22s  %10.3f  %16.4f\n", "unconstrained",       uncon.C44,       slope_uncon)
@printf("  %-22s  %10.3f  %16.4f\n", "exact_constrained",   con.C44,         slope_con)
@printf("  %-22s  %10.3f  %16.4f\n", "target (literature)", C44_TARGET_GPa,  slope_target)
println(repeat('─', 66))

# ── Plot ─────────────────────────────────────────────────────────────────────
fig = Figure(size=(600, 420))
ax  = Axis(fig[1, 1];
           xlabel = "Wave vector  Γ → X  (Å⁻¹)",
           ylabel = "Frequency (THz)",
           title  = "TA phonon branch near Γ vs. C44 continuum-elasticity gradient")

xmax = max(maximum(xs_u), maximum(xs_c))

lines!(ax, xs_u, ta_u;
       color=RGBAf(0.85, 0.45, 0.05, 0.95), linewidth=2.5,
       label="unconstrained — TA branch")
lines!(ax, [0.0, xmax], [0.0, slope_uncon * xmax];
       color=RGBAf(0.85, 0.45, 0.05, 0.7), linestyle=:dash, linewidth=1.5,
       label="unconstrained — C44 gradient")

lines!(ax, xs_c, ta_c;
       color=RGBAf(0.15, 0.4, 0.75, 0.95), linewidth=2.5,
       label="exact_constrained — TA branch")
lines!(ax, [0.0, xmax], [0.0, slope_con * xmax];
       color=RGBAf(0.15, 0.4, 0.75, 0.7), linestyle=:dash, linewidth=1.5,
       label="exact_constrained — C44 gradient")

lines!(ax, [0.0, xmax], [0.0, slope_target * xmax];
       color=(:black, 0.6), linestyle=:dot, linewidth=1.5,
       label="target C44 = $(C44_TARGET_GPa) GPa")

axislegend(ax; position=:lt, framevisible=false, labelsize=10)

outpath = "$(result.dir)/results/phonon_C44_gamma_gradient.png"
save(outpath, fig)
display(fig)
println("\nSaved: $outpath")
