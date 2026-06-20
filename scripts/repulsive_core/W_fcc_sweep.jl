using AtomsBase
using StaticArrays
using Unitful
using CairoMakie, ACEWorkflow

element = :W
result = load_model(element, 20, 4, 5, 3)
model = result.model

lattice_constants = LinRange(0.2, 4.5, 100)

function fcc_cell(el, a)
    a = ustrip(u"Å", a) * 1u"Å"   # normalize type stability

    particles = [
        Atom(el, SVector(0.0u"Å", 0.0u"Å", 0.0u"Å")),
        Atom(el, SVector(0.0u"Å", 0.5a,   0.5a)),
        Atom(el, SVector(0.5a,   0.0u"Å", 0.5a)),
        Atom(el, SVector(0.5a,   0.5a,   0.0u"Å")),
    ]

    cell_vectors = (
        SVector(a, 0.0u"Å", 0.0u"Å"),
        SVector(0.0u"Å", a, 0.0u"Å"),
        SVector(0.0u"Å", 0.0u"Å", a),
    )

    return FlexibleSystem(
        particles,
        cell_vectors,
        (true, true, true)
    )
end


function bulk_energy_curve_basis(element::Symbol, lattice_constants::AbstractVector{Float64}, model)

    energies    = []

    for a in lattice_constants
        ats = fcc_cell(element, a * u"Å")
        E = ACEpotentials.Models.potential_energy_basis(ats, model)
        push!(energies, ustrip(E))
    end

    return energies
end

bulk_energy_bases = bulk_energy_curve_basis(element, lattice_constants, model)

non_pair_energies = [dot(model.ps[1], basis[1:length(model.ps[1])]) for basis in bulk_energy_bases]
pair_energies = [dot(model.ps[2], basis[1+length(model.ps[1]):end]) for basis in bulk_energy_bases]
full_energies = non_pair_energies .+ pair_energies

fig = Figure(size=(1200, 800))
ax  = Axis(fig[1,1], title="Bulk Energy Decomposed vs Lattice Constant", xlabel="Lattice constant (Å)", ylabel="Energy per atom (eV)")
lines!(ax, lattice_constants, non_pair_energies; linestyle=:dash, color=:green, label="3- and 4-body")
lines!(ax, lattice_constants, pair_energies; linestyle=:dash, color=:blue, label="2-body")
lines!(ax, lattice_constants, full_energies; linestyle=:dash, color=:red, label="All body orders")
ylims!(ax, -300, 300)
Legend(fig[1,2], ax)
save("$(result.dir)/results/unconstrained_decomposed_bulk_fcc_energy_curve.png", fig)

using DelimitedFiles
constrained_params = vec(readdlm("$(result.dir)/repulsive_constrained_params.csv", ','))
ACEpotentials.Models.set_linear_parameters!(model, constrained_params)

non_pair_energies = [dot(model.ps[1], basis[1:length(model.ps[1])]) for basis in bulk_energy_bases]
pair_energies = [dot(model.ps[2], basis[1+length(model.ps[1]):end]) for basis in bulk_energy_bases]
full_energies = non_pair_energies .+ pair_energies

fig = Figure(size=(1200, 800))
ax  = Axis(fig[1,1], title="Constrained Bulk Energy Decomposed vs Lattice Constant", xlabel="Lattice constant (Å)", ylabel="Energy per atom (eV)")
lines!(ax, lattice_constants, non_pair_energies; linestyle=:dash, color=:green, label="3- and 4-body")
lines!(ax, lattice_constants, pair_energies; linestyle=:dash, color=:blue, label="2-body")
lines!(ax, lattice_constants, full_energies; linestyle=:dash, color=:red, label="All body orders")
ylims!(ax, -300, 300)
Legend(fig[1,2], ax)
save("$(result.dir)/results/constrained_decomposed_bulk_fcc_energy_curve.png", fig)
