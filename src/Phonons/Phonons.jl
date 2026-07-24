module Phonons

using LinearAlgebra, StaticArrays
using Unitful
using Printf
using DelimitedFiles
using AtomsCalculatorsUtilities.SitePotentials: hessian
import AtomsBuilder
using AtomsBuilder: bulk
using CairoMakie
using ACEpotentials
using ..Elasticity: relax_lattice_constant

include("phonon_displacements.jl")
include("phonon_bands.jl")
include("phonon_committee.jl")

# ── Low-level dynamical matrix (phonopy conventions) ─────────────────────────
export get_dynamical_matrix_at_q!
export get_dynamical_matrices_at_qpoints
export get_q_cart
export phonon_mode_energy

# ── Phonon band structure ─────────────────────────────────────────────────────
export FREQ_THz, THz_to_meV
export precompute_force_constants
export dynamical_matrix_from_fc
export dynamical_matrix_ace
export eigenvalues_to_freq_THz
export dq_eigensystem
export fcc_band_path
export bcc_band_path
export hcp_band_path
export bulk_prim_super
export compute_phonon_bands
export plot_phonon_bands
export plot_phonon_energy
export plot_phonon_comparison
export phonon_committee

end
