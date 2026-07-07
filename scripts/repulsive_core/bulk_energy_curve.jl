using ACEWorkflow, ACEpotentials, AtomsBuilder, Unitful, CairoMakie, Test
using AtomsCalculators: potential_energy
using ACEpotentials: @committee

element = :W
result = load_model(element, 20, 4, 6, 3)
model = result.model

lattice_constants = LinRange(0.2, 4.5, 100)

function bulk_energy_curve(element::Symbol, lattice_constants::AbstractVector{Float64}, model)

    energies    = []

    for a in lattice_constants
        ats = bulk(element, a=a*u"Å")
        E = potential_energy(ats, model)
        push!(energies, ustrip(E))
    end

    return energies
end

function bulk_energy_curve_basis(element::Symbol, lattice_constants::AbstractVector{Float64}, model)

    energies    = []

    for a in lattice_constants
        ats = bulk(element, a=a*u"Å")
        E = ACEpotentials.Models.potential_energy_basis(ats, model)
        push!(energies, ustrip(E))
    end

    return energies
end

bulk_energies = bulk_energy_curve(element, lattice_constants, model)
bulk_energy_bases = bulk_energy_curve_basis(element, lattice_constants, model)
non_pair_energies = [dot(model.ps[1], basis[1:length(model.ps[1])]) for basis in bulk_energy_bases]
pair_energies = [dot(model.ps[2], basis[1+length(model.ps[1]):end]) for basis in bulk_energy_bases]

@test bulk_energies == non_pair_energies .+ pair_energies

# Energy per atom
n_atoms = length(bulk(:Al, a=3.0u"Å"))  # let AtomsBuilder tell us
e_mean = Float64.(bulk_energies) ./ n_atoms

fig = Figure()
ax  = Axis(fig[1,1], title="Bulk Energy vs Lattice Constant", xlabel="Lattice constant (Å)", ylabel="Energy per atom (eV)")
lines!(ax, lattice_constants, e_mean; color=:steelblue)
save("$(result.dir)/results/bulk_energy_curve.png", fig)

fig = Figure(size=(1200, 800))
ax  = Axis(fig[1,1], title="Bulk Energy Decomposed vs Lattice Constant", xlabel="Lattice constant (Å)", ylabel="Energy per atom (eV)")
lines!(ax, lattice_constants, non_pair_energies; linestyle=:dash, color=:green, label="3- and 4-body")
lines!(ax, lattice_constants, pair_energies; linestyle=:dash, color=:blue, label="2-body")
ylims!(ax, -250, 200)
Legend(fig[1,2], ax)
save("$(result.dir)/results/decomposed_bulk_energy_curve.png", fig)
