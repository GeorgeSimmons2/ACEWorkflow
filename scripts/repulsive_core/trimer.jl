using ACEWorkflow, ACEpotentials, AtomsBuilder, Unitful, CairoMakie 
using AtomsCalculators: potential_energy

result = load_model(:W, 20, 4, 5, 3)
model = result.model

distances = LinRange(1.0, 4.5, 100)
element   = :Al

function dimer_energy(element::Symbol, r::Float64, model)
    box = 1.01 * r * 2
    ats = AtomsBuilder._flexible_system(
        [[0.0, 0.0, 0.0], [r, 0.0, 0.0]] .* u"Å",
        [element, element],
        box .* [1. 0. 0.; 0. 1. 0.; 0. 0. 1.] .* u"Å",
        (false, false, false))
    return potential_energy(ats, model)
end

function trimer_3body_curve(element::Symbol, distances::AbstractVector{Float64}, model)

    energies_3body = []

    for distance in distances
        # Linear trimer: atoms at 0, d, 2d  →  pairs: (1,2)=d, (2,3)=d, (1,3)=2d
        box_x = 1.01 * 2 * maximum(distances)
        ats = AtomsBuilder._flexible_system(
            [[0.0, 0.0, 0.0], [distance, 0.0, 0.0], [0.5 * distance, distance * (sqrt(3) / 2), 0.0]] .* u"Å",
            [element, element, element],
            box_x .* [1. 0. 0.; 0. 1. 0.; 0. 0. 1.] .* u"Å",
            (false, false, false))
        E_trimer = potential_energy(ats, model)

        # Subtract the three pairwise contributions
        E_12 = dimer_energy(element, distance,        model)   # atoms 1-2, r = d

        push!(energies_3body, E_trimer - (3 * E_12))
    end

    return energies_3body
end

trimer_energies = trimer_3body_curve(element, distances, model)

fig = Figure()
ax  = Axis(fig[1,1], title="Trimer 3-body Interaction", xlabel="Nearest-neighbour distance (Å)", ylabel="3-body energy (eV)")
lines!(ax, distances, ustrip.(trimer_energies))
hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=0.8)
save("$(result.dir)/results/trimer_3body_curve.png", fig)
