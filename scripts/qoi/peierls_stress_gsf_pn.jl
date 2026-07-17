# peierls_stress_gsf_pn.jl
#
# Static-0K Peierls-stress estimate for FCC Al, built from two pieces that
# are each cheap to compute from an ACE model:
#
#   1. The {111}<112> generalized stacking-fault (GSF) energy curve γ(u) —
#      rigid (unrelaxed) single-point energies of an {111} slab with the
#      atoms above the mid-plane shifted in-plane by u. The peak, γ_us
#      (unstable stacking-fault energy), controls how hard it is to nucleate
#      the leading Shockley partial.
#   2. The classic sinusoidal Peierls-Nabarro (PN) closed form, which turns
#      γ_us plus the model's own elastic constants (C11, C12, C44) into a
#      dislocation width and a Peierls stress.
#
# Both stages are static 0K (no MD, no relaxation) and depend directly on
# the model's elastic constants, so they're a good discriminator between an
# unconstrained fit and one that's been pinned to target C_ij values.
#
# ── Geometry ──────────────────────────────────────────────────────────────
# The {111} in-plane cell is spanned by two <110>-type primitive vectors
# a1, a2 (60° apart, |a1|=|a2|=a/√2) with an ABC stacking sequence along the
# normal. A fractional in-plane shift of (1/3, 1/3) in the (a1,a2) basis is
# exactly the a/6<112> Shockley-partial displacement (it takes A-stacking
# registry directly to B-stacking registry) — so sweeping t ∈ [0,1] with
# shift t·(a1+a2)/3 traces out the leading-partial glide path.
#
# A vacuum gap isolates the slab so the two exposed {111} surfaces don't
# interact with the internal fault plane; because the shift only moves atoms
# rigidly *within* the periodic plane, the (unchanged) surface energy
# contribution cancels exactly in E(u) − E(0), leaving only the internal
# stacking-fault excess energy.
#
# ── Peierls–Nabarro closed form (sinusoidal misfit potential) ──────────────
# Ansatz γ(u) = (γ_us/2)·(1 − cos(2πu/b))  ⇒  max slope τ_max = π·γ_us/b.
# Minimizing the PN energy functional (elastic self-energy of a continuous
# disregistry distribution + misfit energy) gives the dislocation half-width
# and the classic Peierls stress:
#
#     ζ   = K·b / (4π·τ_max)   = K·b² / (4π²·γ_us)
#     τ_P = 2K · exp(−4π·ζ/b)
#
# with K = G (screw) or G/(1−ν) (edge) the isotropic dislocation-energy
# prefactor. Different textbooks use half-width vs. full-width conventions
# that shift this by an O(1) factor in the exponent — this implementation is
# internally consistent, so *relative* comparisons across models (which is
# the point here) are robust to that convention even though the absolute
# τ_P is not publication-grade.

using AtomsBase, Unitful, LinearAlgebra, StaticArrays, Printf
using AtomsCalculators: potential_energy
using ACEWorkflow

# ── {111} bicrystal slab with a rigid in-plane fault shift ────────────────

"""
    fcc111_gsf_slab(element, a, layers, vacuum, t)

Build an FCC `{111}` slab (in-plane hex cell, ABC stacking, vacuum along the
normal) with atoms in the upper half rigidly shifted in-plane by
`t · (a1+a2)/3` — the leading `a/6<112>` Shockley-partial displacement at
fraction `t ∈ [0,1]`. Returns `(sys, area)`.
"""
function fcc111_gsf_slab(element, a, layers, vacuum, t)
    a2d = a / sqrt(2)
    a1  = a2d * [1.0, 0.0, 0.0]
    a2  = a2d * [0.5, sqrt(3)/2, 0.0]
    d   = a / sqrt(3)                       # interlayer spacing
    a3  = [0.0, 0.0, layers * d + vacuum]

    shift_vecs = ([0.0, 0.0], [1/3, 1/3], [2/3, 2/3])   # ABC registry
    fault_shift = t .* (a1 .+ a2) ./ 3                  # partial-glide displacement
    i_cut = layers ÷ 2

    atoms = Vector{Atom}(undef, layers)
    for i in 0:layers-1
        s  = shift_vecs[mod(i, 3) + 1]
        z  = i * d
        pos = s[1] * a1 + s[2] * a2 + [0.0, 0.0, z]
        i >= i_cut && (pos = pos .+ fault_shift)
        atoms[i + 1] = Atom(element, pos .* u"Å")
    end

    sys  = periodic_system(atoms, [a1, a2, a3] .* u"Å")
    area = norm(cross(a1, a2))
    return sys, area
end

"""
    gsf_curve(model, element, a_eq; layers=24, vacuum=15.0, n_steps=20)

Rigid (unrelaxed, static 0K) γ-surface along the leading Shockley-partial
glide path. Returns `(; u, gamma, gamma_us, b_partial)` where `u` is the
fractional partial displacement (0 to 1), `gamma` the excess energy per unit
area (eV/Å²) at each `u`, `gamma_us = maximum(gamma)`, and `b_partial` the
partial Burgers-vector magnitude (Å) implied by the same geometry.
"""
function gsf_curve(model, element, a_eq; layers=24, vacuum=15.0, n_steps=20)
    sys0, area = fcc111_gsf_slab(element, a_eq, layers, vacuum, 0.0)
    n_atoms = length(sys0)
    E0 = ustrip(potential_energy(sys0, model))

    ts    = range(0.0, 1.0, length=n_steps + 1)
    gamma = Vector{Float64}(undef, length(ts))
    for (k, t) in enumerate(ts)
        sys_t = t == 0.0 ? sys0 : fcc111_gsf_slab(element, a_eq, layers, vacuum, t)[1]
        E_t   = ustrip(potential_energy(sys_t, model))
        gamma[k] = (E_t - E0) / area
    end

    a2d = a_eq / sqrt(2)
    a1  = a2d * [1.0, 0.0, 0.0]
    a2  = a2d * [0.5, sqrt(3)/2, 0.0]
    b_partial = norm((a1 .+ a2) ./ 3)

    return (; u=collect(ts), gamma, gamma_us=maximum(gamma), b_partial)
end

# ── Isotropic elastic averages (Voigt) ─────────────────────────────────────

"""
    voigt_shear_and_poisson(C11, C12, C44)

Voigt-averaged isotropic shear modulus `G` (GPa) and Poisson ratio `ν` from
the cubic elastic constants (GPa).
"""
function voigt_shear_and_poisson(C11, C12, C44)
    B = (C11 + 2C12) / 3
    G = (C11 - C12 + 3C44) / 5
    ν = (3B - 2G) / (2 * (3B + G))
    return (; G, ν)
end

# ── Peierls–Nabarro closed form ─────────────────────────────────────────────

const EV_PER_ANG3_TO_GPA = 160.2176621

"""
    peierls_nabarro_stress(C11, C12, C44, gamma_us_eV_Ang2, b_partial_Ang; character=:edge)

Sinusoidal Peierls–Nabarro estimate of the Peierls stress (GPa) for the
leading Shockley partial, from the model's own elastic constants (GPa) and
the GSF unstable-stacking energy `gamma_us_eV_Ang2` (eV/Å²) at Burgers-vector
magnitude `b_partial_Ang` (Å). `character` selects the isotropic
dislocation-energy prefactor: `:edge` → K=G/(1−ν), `:screw` → K=G. Shockley
partials are of genuinely mixed character; `:edge` is used as a single
defensible default (see module docstring) — this is a simplification, not a
literature-grade Peierls stress.

Returns `(; G, ν, K, tau_max, zeta, tau_P)`, all in GPa except `zeta` (Å).
"""
function peierls_nabarro_stress(C11, C12, C44, gamma_us_eV_Ang2, b_partial_Ang;
                                 character::Symbol=:edge)
    (; G, ν) = voigt_shear_and_poisson(C11, C12, C44)
    K = character == :screw ? G : G / (1 - ν)

    gamma_us_GPa_Ang = gamma_us_eV_Ang2 * EV_PER_ANG3_TO_GPA   # eV/Å² -> GPa·Å
    tau_max = π * gamma_us_GPa_Ang / b_partial_Ang             # GPa
    zeta    = K * b_partial_Ang / (4π * tau_max)                # Å
    tau_P   = 2K * exp(-4π * zeta / b_partial_Ang)              # GPa

    return (; G, ν, K, tau_max, zeta, tau_P)
end

"""
    peierls_stress_static(model, element, a_eq, C11, C12, C44;
                           layers=24, vacuum=15.0, n_steps=20, character=:edge)

Full pipeline: GSF curve → γ_us → Peierls–Nabarro τ_P. Returns
`(; gsf, pn)` (the outputs of `gsf_curve` and `peierls_nabarro_stress`).
"""
function peierls_stress_static(model, element, a_eq, C11, C12, C44;
                                layers=24, vacuum=15.0, n_steps=20, character::Symbol=:edge)
    gsf = gsf_curve(model, element, a_eq; layers, vacuum, n_steps)
    pn  = peierls_nabarro_stress(C11, C12, C44, gsf.gamma_us, gsf.b_partial; character)
    return (; gsf, pn)
end
