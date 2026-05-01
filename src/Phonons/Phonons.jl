module Phonons

using LinearAlgebra, StaticArrays

include("phonon_displacements.jl")

# ── Low-level dynamical matrix (phonopy conventions) ─────────────────────────
export get_dynamical_matrix_at_q!
export get_dynamical_matrices_at_qpoints
export get_q_cart
export phonon_mode_energy

end
