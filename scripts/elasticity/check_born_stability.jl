function make_Al_fcc_unit_cell(lattice)
    a = 4.05u"Å"  # Al lattice constant (you can also rely on input lattice)

    # FCC basis positions (fractional coords)
    frac_basis = [
        SVector(0.0, 0.0, 0.0),
        SVector(0.5, 0.5, 0.0),
        SVector(0.5, 0.0, 0.5),
        SVector(0.0, 0.5, 0.5),
    ]

    # Convert to cartesian coordinates
    cartesian_positions = [
        lattice[1] * f[1] + lattice[2] * f[2] + lattice[3] * f[3] for f in frac_basis
    ]

    atoms = [Atom(:Al, pos) for pos in cartesian_positions]

    # Construct the periodic system
    unit_cell = periodic_system(atoms, lattice; pbc=NTuple{3,Bool}((true, true, true)))
    return unit_cell
end
include("CubicElastic.jl")

a = 4.05
cell_vectors = (
    SVector(a,0,0)*u"Å",
    SVector(0,a,0)*u"Å",
    SVector(0,0,a)*u"Å"
)

results = compute_cubic_elastic_constants_local(
    model,
    make_Al_fcc_unit_cell,
    cell_vectors
)

println(results)