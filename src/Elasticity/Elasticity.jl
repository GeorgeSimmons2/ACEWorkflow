module Elasticity

using LinearAlgebra, StaticArrays, ForwardDiff
using AtomsBase
using AtomsCalculators
using AtomsBuilder
using ACEpotentials
using Unitful
using Optim
import AtomsCalculatorsUtilities.SitePotentials: PairList, cutoff_radius, get_neighbours, energy_unit

include("born_elastic_constants.jl")
include("cubic_elastic.jl")
include("get_hessian.jl")

# ── Born / AutoDiff elastic tensor ────────────────────────────────────────────
export voigt_to_eps, strained_system
export born_C_voigt, born_operator_voigt, elastic_from_operator

# ── Cubic elastic constants via finite-difference stress ─────────────────────
export strain_C11, strain_C12, strain_C44, compute_stress
export compute_cubic_elastic_constants_local

# ── Strain Hessian / dynamical matrix w.r.t. ACE basis ───────────────────────
export elastic_hessian_energy, elastic_hessian_basis
export hessian_basis
export dynamical_matrix, dynamical_matrix_basis_at_k, dynamical_matrix_at_k
export unstable_modes
export apply_mode, apply_mode_design, apply_mode_energy
export second_deriv_of_model, second_deriv_of_model_dotted
export strained_cell_design_prescaled
export strain_hessian_lattice_constant_derivative, strain_hessian_lattice_constant_derivative_ad

# ── Cell relaxation + convenience Hessian-to-GPa helper ──────────────────────
export relax_lattice_constant, strain_hessian_GPa

end
