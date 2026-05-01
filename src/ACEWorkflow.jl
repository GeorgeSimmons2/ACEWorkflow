module ACEWorkflow

include("POPS/POPSRegression.jl")
include("Elasticity/Elasticity.jl")
include("Phonons/Phonons.jl")
include("Models/Models.jl")

using .POPSRegression
using .Elasticity
using .Phonons
using .Models

# Re-export POPS
export corrections, hypercube, sample_hypercube, rejection_sample_hypercube

# Re-export Elasticity
export voigt_to_eps, strained_system
export born_C_voigt, born_operator_voigt, elastic_from_operator
export strain_C11, strain_C12, strain_C44, compute_stress
export compute_cubic_elastic_constants_local
export elastic_hessian_energy, elastic_hessian_basis
export hessian_basis
export dynamical_matrix, dynamical_matrix_basis_at_k, dynamical_matrix_at_k
export unstable_modes
export apply_mode, apply_mode_design, apply_mode_energy
export second_deriv_of_model, second_deriv_of_model_dotted
export relax_lattice_constant, strain_hessian_GPa

# Re-export Phonons
export get_dynamical_matrix_at_q!
export get_dynamical_matrices_at_qpoints
export get_q_cart
export phonon_mode_energy

# Re-export Models
export load_model, build_model, model_name, model_dir

end
