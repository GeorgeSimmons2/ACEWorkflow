using ACEWorkflow, ACEpotentials, AtomsBuilder, Unitful, CairoMakie, ExtXYZ, AtomsBase
using AtomsCalculators: potential_energy
using Test
using Statistics, LinearAlgebra, DelimitedFiles

element = :W
model, _ = ACEpotentials.load_model("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/models/W_20_4_5A_3/W_20_4_5A_3.json")
result   = (; dir="/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/models/W_20_4_5A_3")

function apply_deformation(delta, deformation, relaxed_cell, a, element)
    cell_vectors = relaxed_cell.cell.cell_vectors
    box         = deformation(cell_vectors, delta)
    frac_basis  = [SVector(0.0, 0.0, 0.0)]
    cartesian_positions = [box[1] * f[1] + box[2] * f[2] + box[3] * f[3] for f in frac_basis]
    atoms = [AtomsBase.Atom(element, pos) for pos in cartesian_positions]
    return periodic_system(atoms, box)
end

function apply_xy_shear(cell_vectors, δ)
    # Strain tensor for xy shear
    ε = [
        1.0    δ/2    0.0
        δ/2    1.0    0.0
        0.0    0.0   1.0
    ]

    # Apply strain to each lattice vector
    strained_vectors = Tuple([SVector{3}(ε * vec) for vec in cell_vectors])
    return strained_vectors
end

@test apply_deformation(0.0, apply_xy_shear, bulk(:W), 3.15, :W).cell == bulk(:W).cell
@test potential_energy(apply_deformation(0.0, apply_xy_shear, bulk(:W), 3.15, :W), model) == potential_energy(bulk(:W), model)


a_eq = ACEWorkflow.relax_lattice_constant(model, element)

deformation_deltas = collect(LinRange(-1.9, 1.9, 10))
deformed_cells = [apply_deformation(delta, apply_xy_shear, bulk(:W, a=a_eq*u"Å"), a_eq*u"Å", :W) for delta in deformation_deltas]
potential_energies = [potential_energy(deformed_cell, model) for deformed_cell in deformed_cells]