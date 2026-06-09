using ACEWorkflow, ACEpotentials, AtomsBuilder, Unitful, CairoMakie
using AtomsCalculators: potential_energy
using ACEpotentials: @committee

element = :Al
result = load_model(element, 20, 4, 6, 3)
model = result.model

lattice_constants = LinRange(2.5, 4.2, 100)

function bulk_energy_curve(element::Symbol, lattice_constants::AbstractVector{Float64}, model)

    energies    = []

    for a in lattice_constants
        ats = bulk(element, a=a*u"Å")
        E = potential_energy(ats, model)
        push!(energies, ustrip(E))
    end

    return energies
end

bulk_energies = bulk_energy_curve(element, lattice_constants, model)

# Energy per atom
n_atoms = length(bulk(:Al, a=3.0u"Å"))  # let AtomsBuilder tell us
e_mean = Float64.(bulk_energies) ./ n_atoms

fig = Figure()
ax  = Axis(fig[1,1], title="Bulk Energy vs Lattice Constant", xlabel="Lattice constant (Å)", ylabel="Energy per atom (eV)")
lines!(ax, lattice_constants, e_mean; color=:steelblue)
save("$(result.dir)/results/bulk_energy_curve.png", fig)
