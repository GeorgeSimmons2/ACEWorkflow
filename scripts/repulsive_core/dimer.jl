using ACEWorkflow, ACEpotentials, AtomsBuilder, Unitful, CairoMakie 
using AtomsCalculators: potential_energy

result = load_model(:Al, 12, 4, 6, 3)
model = result.model
element = :Al
distances = LinRange(0.2, 4.5, 100)

function dimer_curve(element::Symbol, distances::AbstractVector{Float64}, model)
    
    energies = []

    for (i, distance) in enumerate(distances)
        ats = AtomsBuilder._flexible_system([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]] .* u"Å", [element, element], (1.01 * maximum(distances)) .* [1. 0. 0.; 0. 1. 0.; 0. 0. 1.] .*u"Å", (false, false, false))
        push!(energies, potential_energy(ats, model))
    end

    return energies
end

dimer_energies = dimer_curve(element, distances, model)

fig = Figure()
ax  = Axis(fig[1,1], title="Dimer Curve", xlabel="Distance (Å)", ylabel="Energy (eV)")
lines!(ax, distances, ustrip.(dimer_energies))
Legend(fig[1,1], ax, position=:rt)
save("$(result.dir)/results/dimer_curve.png", fig)
